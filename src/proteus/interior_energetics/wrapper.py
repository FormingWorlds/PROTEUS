# Generic interior wrapper
from __future__ import annotations

import gc
import logging
import os
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.optimize as optimise

from proteus.interior_energetics.common import Interior_t
from proteus.outgas.wrapper import calc_target_elemental_inventories
from proteus.utils.constants import M_earth, R_earth, const_G, element_list
from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

# Counter for consecutive Zalmoxis convergence failures during time evolution.
# Reset on each successful structure update. Crash after max_consecutive.
#
# The consecutive-failure counters live on the per-run Interior_t state
# object (interior_energetics/common.py), so concurrent Proteus
# instances in one Python process cannot share failure streaks. The
# abort thresholds below are immutable and stay module-level.
# Allow up to 8 consecutive Zalmoxis failures before aborting. Consecutive
# `pressure=False, density=False, mass=False` streaks can occur transiently;
# 8 retries give recoverable streaks room to recover without hiding a genuine
# deadlock.
_ZALMOXIS_MAX_CONSECUTIVE_FAILS = 8

# Post-acceptance mass-anchor tolerance enforced on every Zalmoxis-success
# boundary in update_structure_from_interior. Zalmoxis' internal
# solver_tol_outer (default 3e-3) is a numerical convergence target,
# not a coupling contract: it leaves room for ~0.3 % drift between
# hf_row['M_int'] and the dry mass target. This wrapper-level guard
# sets the contract to 3e-3 (0.3 %), matching Zalmoxis' own
# tolerance_outer noise floor on hot mantle profiles. A tighter 1e-3
# target collides with Zalmoxis' fixed-point band, so it is not used here.
# The <0.1 % mass conservation contract is delivered by the Newton + brentq
# bracketing in Zalmoxis' outer loop, not by this wrapper guard.
# This guard remains a safety net against off-attractor results
# (9-15 % off target) and gross corruption.
_ZALMOXIS_MASS_ANCHOR_TOL = 3e-3

# Abort threshold for consecutive SPIDER CVode failures during time
# evolution; the counter resets on each successful SPIDER call.
_SPIDER_MAX_CONSECUTIVE_FAILS = 3

# Abort threshold for consecutive Aragog retry-ladder exhaustions; the
# counter resets on each successful Aragog call.
_ARAGOG_MAX_CONSECUTIVE_FAILS = 3

log = logging.getLogger('fwl.' + __name__)


def get_core_density(config: Config, hf_row: dict) -> float:
    """Resolve core density: numerical value from config, or from hf_row if 'self'."""
    val = config.interior_struct.core_density
    if val == 'self':
        return float(hf_row.get('core_density', 10738.0))
    return float(val)


def get_core_heatcap(config: Config, hf_row: dict) -> float:
    """Resolve core heat capacity: numerical value from config, or from hf_row if 'self'."""
    val = config.interior_struct.core_heatcap
    if val == 'self':
        return float(hf_row.get('core_heatcap', 450.0))
    return float(val)


def update_gravity(hf_row: dict):
    """
    Update surface gravity.
    """
    hf_row['gravity'] = const_G * hf_row['M_int'] / (hf_row['R_int'] * hf_row['R_int'])


def _prevent_warming_clamp_active(config: Config) -> bool:
    """Return True iff the early T_magma = min(new, prev) ratchet should fire.

    The clamp is enabled by ``planet.prevent_warming``. The runaway-T fallback
    elsewhere in run_interior remains active regardless. The user-facing
    advisory at config-load time and at run start (see config/_config.py and
    utils/terminate.py) flags the energy-conservation caveats associated with
    this clamp.
    """
    return bool(config.planet.prevent_warming)


def calculate_core_mass(hf_row: dict, config: Config):
    """
    Calculate the core mass of the planet.

    ``core_frac`` is cubed as a fraction of the planet radius, so this is
    only valid when ``core_frac_mode == 'radius'``. The only structure
    path that reaches this function is ``interior_struct.module ==
    'spider'``, for which the config validator rejects
    ``core_frac_mode = 'mass'``. The guard below makes that dependency
    explicit so a relaxed validator cannot silently cube a mass fraction
    as a radius fraction.
    """
    if config.interior_struct.core_frac_mode != 'radius':
        raise RuntimeError(
            'calculate_core_mass cubes core_frac as a radius fraction, but '
            'core_frac_mode=%r; this path requires radius mode.'
            % config.interior_struct.core_frac_mode
        )
    rho_core = get_core_density(config, hf_row)
    hf_row['M_core'] = (
        rho_core
        * 4.0
        / 3.0
        * np.pi
        * (hf_row['R_int'] * config.interior_struct.core_frac) ** 3.0
    )


def update_planet_mass(hf_row: dict):
    """
    Calculate total planet mass, as sum of dry+wet parts.

    Whole-planet oxygen accounting (issue #677): M_ele sums over ALL
    elements in ``element_list``, including O. The atmospheric and
    dissolved O mass produced by CALLIOPE (under the fO2 buffer) is
    therefore counted in M_planet = M_int + M_ele, keeping the
    bookkeeping symmetric so M_atm cannot exceed M_planet at
    high H budgets.

    Mantle FeO-bound oxygen remains implicit in the PALEOS density
    tables that drive ``M_int``; we don't double-count it here.

    This site pairs with the dry-mass target in
    ``load_zalmoxis_configuration``: both currently treat the mantle EOS
    as bare silicate (``dry_mantle = true``, the only configuration the
    config gate admits). If wet-mantle support is enabled there, this
    sum must stop counting the dissolved inventory in ``M_ele``, because
    a wet-mantle ``M_int`` already contains that mass.
    """

    # Update total element mass. O is included alongside H/C/N/S
    # (issue #677). .get() default of 0.0
    # makes the sum safe for pre-IC hf_row states where some element
    # columns may not have been initialised yet.
    hf_row['M_ele'] = 0.0
    for e in element_list:
        hf_row['M_ele'] += float(hf_row.get(e + '_kg_total', 0.0))

    # Add to total planet mass
    hf_row['M_planet'] = hf_row['M_int'] + hf_row['M_ele']


def get_nlevb(config: Config):
    """
    Get number of interior basic-nodes (level edges) from config.
    """
    match config.interior_energetics.module:
        case 'spider':
            return int(config.interior_energetics.num_levels)
        case 'aragog':
            return int(config.interior_energetics.num_levels)
        case 'boundary':
            return 2
        case 'dummy':
            return 2
    raise ValueError(f"Invalid interior module selected '{config.interior_energetics.module}'")


# The 10 phase-property files Aragog's EntropyEOS and SPIDER's lookup
# loader both expect, in SPIDER's canonical P-S header format.
_SPIDER_EOS_PHASE_FILES = (
    'temperature_melt.dat',
    'temperature_solid.dat',
    'density_melt.dat',
    'density_solid.dat',
    'heat_capacity_melt.dat',
    'heat_capacity_solid.dat',
    'adiabat_temp_grad_melt.dat',
    'adiabat_temp_grad_solid.dat',
    'thermal_exp_melt.dat',
    'thermal_exp_solid.dat',
)

# P-S melting curves. Aragog's `_load_spider_phase_boundary` hardcodes
# these filenames. SPIDER's bundled lookup_data ships them under the
# `{solidus,liquidus}_A11_H13.dat` names; we rename on copy so a
# single canonical layout satisfies both solvers.
_SPIDER_EOS_MELTING_CURVES = ('solidus_P-S.dat', 'liquidus_P-S.dat')


def _rectangularize_spider_ps_file(src: str, dst: str) -> None:
    """Read a SPIDER P-S phase-property table and write a strictly
    rectangular version to dst.

    SPIDER's bundled P-S tables (``SPIDER/lookup_data/1TPa-dK09-elec-free/``
    and the corresponding Zenodo 19473625 uploads) are written with a
    ``# 5 NX NY`` header where NX is the number of pressure points per
    entropy slice and NY is the number of entropy slices. The data
    layout iterates P fastest (inner loop) and S slowest (outer loop),
    which matches what Aragog's EntropyEOS expects. HOWEVER, SPIDER's
    table generator produces a tiny P drift across S slices (relative
    drift ~1e-8), because each slice is computed independently from
    the underlying equation of state. SPIDER's own C loader
    (``interp.c::Interp2dCreateAndSet``) works around this by taking
    the first NX rows as the canonical P grid (line 183-188) and
    ignoring the drift. Aragog's Python loader uses
    ``np.unique(P_all)`` which returns ~NX*NY values for the drifted
    grid, and then scipy's ``RegularGridInterpolator`` refuses the
    mismatch with ``ValueError: There are X points and Y values in
    dimension 0``.

    This helper normalises the grid: it snaps every P value to the
    canonical value from the first S slice and every S value to the
    first value in each slice, then writes a clean rectangular file
    that Aragog can load without modification. Bit-level floating-
    point artefacts are eliminated at the level of the 1e-8 drift,
    which is six orders of magnitude below the physical resolution of
    the tables (~1 K in T, ~1 kg/m^3 in rho).
    """
    with open(src) as f:
        header_lines = []
        for _ in range(5):
            header_lines.append(f.readline())

    h = header_lines[0].strip().lstrip('#').split()
    n_head = int(h[0])
    NX = int(h[1])
    NY = int(h[2])

    data = np.genfromtxt(src, skip_header=n_head)
    if data.shape[0] != NX * NY:
        raise ValueError(f'{src}: header says NX*NY = {NX * NY} rows, file has {data.shape[0]}')

    # SPIDER convention: P varies fastest (inner), S varies slowest
    # (outer). First NX rows give the canonical P grid, first row of
    # each block gives the canonical S grid.
    P_canonical = data[:NX, 0]
    S_canonical = data[::NX, 1][:NY]
    Q_matrix = data[:, 2].reshape(NY, NX)

    # Sanity: the drift should be tiny.
    P_all = data[:, 0].reshape(NY, NX)
    max_p_drift = float(np.max(np.abs(P_all - P_canonical)))
    if max_p_drift / max(abs(P_canonical).max(), 1e-30) > 1e-4:
        # Drift larger than 1e-4 relative is not a SPIDER-style rounding
        # artefact; it means the file is genuinely non-rectangular and
        # we cannot rectangularise it safely.
        raise ValueError(
            f'{src}: P grid drift across S slices is {max_p_drift:.3e} '
            f'absolute, > 1e-4 relative. File is not quasi-rectangular '
            f'and cannot be rectangularised.'
        )

    with open(dst, 'w') as out:
        for line in header_lines:
            out.write(line)
        for yi in range(NY):
            S_val = S_canonical[yi]
            for xi in range(NX):
                out.write(f'{P_canonical[xi]:.18e}\t{S_val:.18e}\t{Q_matrix[yi, xi]:.18e}\n')


def _load_spider_ps_phase_table(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a rectangularized SPIDER P-S phase-property table.

    Reads a 3-column file in SPIDER's canonical P-S format:
      Line 1: ``# n_header NX NY``  (n_header = lines to skip;
              NX = number of P points per S slice;
              NY = number of S slices)
      Line ``n_header``: ``# P_scale S_scale value_scale``
              (multiply column to recover SI units).
    Data layout iterates P fastest (inner), S slowest (outer).

    Returns
    -------
    P_grid : (NX,) float ndarray, P axis in Pa
    S_grid : (NY,) float ndarray, S axis in J/kg/K
    val    : (NX, NY) float ndarray, value(P_i, S_j) in SI units
    """
    with open(path) as f:
        header_lines = [f.readline() for _ in range(5)]

    h = header_lines[0].strip().lstrip('#').split()
    n_head = int(h[0])
    NX = int(h[1])
    NY = int(h[2])
    scales = header_lines[n_head - 1].strip().lstrip('#').split()
    P_scale = float(scales[0])
    S_scale = float(scales[1])
    val_scale = float(scales[2])

    data = np.loadtxt(path, skiprows=n_head)
    if data.shape[0] != NX * NY:
        raise ValueError(f'{path}: header says NX*NY={NX * NY} rows, file has {data.shape[0]}')

    # Reshape as (NY, NX) since S is outer, P is inner; first row of
    # the reshape is one S slice across all P. Transpose to (NX, NY)
    # to match (P_grid, S_grid) ordering.
    P_grid = data[:, 0].reshape(NY, NX)[0, :] * P_scale
    S_grid = data[:, 1].reshape(NY, NX)[:, 0] * S_scale
    val = (data[:, 2].reshape(NY, NX) * val_scale).T  # (NX, NY)
    return P_grid, S_grid, val


def _derive_ps_melting_curve(
    pt_path: str,
    phase_table_path: str,
    target_path: str,
    *,
    label: str = 'curve',
) -> dict:
    """Derive a SPIDER P-S melting curve by inverting a P-T curve through
    an EoS temperature lookup table.

    For each (P_i, T_target_i) in ``pt_path``, find S_i such that
    ``temperature_grid(P_i, S_i) = T_target_i``. Writes the resulting
    (P, S) pairs into ``target_path`` in SPIDER's canonical 2-column
    format with 5 header lines.

    Why this exists
    ---------------
    The WB17 pipeline's runtime entropy-solver melting curve must be
    derived from the user-configured ``interior_struct.melting_dir``
    P-T file, not byte-copied from the upstream WB+2018 distribution.
    This helper keeps the bookkeeping consistent: both the
    lever-rule mushy-zone path (in aragog.py) and the entropy-form
    integrator (via Aragog's EntropyEOS) read from the same canonical
    P-T specification.

    Parameters
    ----------
    pt_path : str
        Path to the configured P-T melting file (column format: P [Pa]
        and T [K], one pair per line, optional ``#``-prefixed comments).
    phase_table_path : str
        Path to the rectangularized SPIDER P-S temperature lookup table
        (``temperature_solid.dat`` for the solidus inversion,
        ``temperature_melt.dat`` for the liquidus).
    target_path : str
        Output path for the derived P-S melting curve.
    label : str, optional
        Short label used in log messages and the warning summary.

    Returns
    -------
    dict
        Diagnostic summary with keys:
          - ``n_points``: number of (P, S) pairs written
          - ``n_clipped_below``: count of T_target points below the
            lookup grid's T floor at the corresponding P (S clipped
            to S_min)
          - ``n_clipped_above``: count of T_target points above the
            lookup grid's T ceiling (S clipped to S_max)
          - ``max_inversion_residual_K``: maximum |T_recovered -
            T_target| over all clean inversions (excludes clipped
            points). A round-trip sanity metric.

    Notes
    -----
    Inversion is 1D linear interpolation in S at fixed P, evaluated
    on the EoS's S grid. Temperature is monotonically increasing in
    S at fixed P for both phase tables (entropy increases with T at
    fixed P, by the second law), so a linear interp is well-posed
    without bracketing acrobatics. Resolution is set by the EoS S
    grid (~125 points for the standard 1TPa-dK09 lookup), giving
    ~10 K precision in T at 100 GPa.
    """
    pt = np.loadtxt(pt_path, comments='#')
    if pt.ndim != 2 or pt.shape[1] != 2:
        raise ValueError(f'{pt_path}: expected 2-column (P, T) format, got shape {pt.shape}')
    P_target = pt[:, 0]
    T_target = pt[:, 1]

    P_grid, S_grid, T_grid = _load_spider_ps_phase_table(phase_table_path)
    # Sort P_target into the grid range; clip outside.
    P_clipped = np.clip(P_target, P_grid[0], P_grid[-1])
    n_p_clipped = int(np.sum(P_target != P_clipped))
    if n_p_clipped:
        log.warning(
            'derive_ps_melting %s: %d/%d P points outside EoS P grid [%.3e, %.3e] Pa, '
            'clipped to grid edge',
            label,
            n_p_clipped,
            len(P_target),
            P_grid[0],
            P_grid[-1],
        )

    # For each P_target, build T(S) at that P by interpolating the
    # T_grid in P, then linearly interp T_target -> S.
    n_clip_below = 0
    n_clip_above = 0
    residuals = []
    S_out = np.empty_like(P_clipped)
    # Pre-find P bracket indices for efficiency
    p_idx = np.searchsorted(P_grid, P_clipped, side='right') - 1
    p_idx = np.clip(p_idx, 0, len(P_grid) - 2)
    P_lo = P_grid[p_idx]
    P_hi = P_grid[p_idx + 1]
    w = (P_clipped - P_lo) / (P_hi - P_lo)

    for i in range(len(P_clipped)):
        # T column at P_clipped[i] from linear interp in P
        T_col = (1 - w[i]) * T_grid[p_idx[i], :] + w[i] * T_grid[p_idx[i] + 1, :]
        # Monotone in S? Verify and warn if not.
        if np.any(np.diff(T_col) <= 0):
            # Locally non-monotonic: usually a cold-end EoS artefact.
            # Sort by T for the interp; this picks the principal branch.
            order = np.argsort(T_col)
            T_col_sorted = T_col[order]
            S_sorted = S_grid[order]
        else:
            T_col_sorted = T_col
            S_sorted = S_grid

        if T_target[i] < T_col_sorted[0]:
            S_out[i] = S_sorted[0]
            n_clip_below += 1
        elif T_target[i] > T_col_sorted[-1]:
            S_out[i] = S_sorted[-1]
            n_clip_above += 1
        else:
            S_out[i] = float(np.interp(T_target[i], T_col_sorted, S_sorted))
            # Round-trip residual: re-interpolate T at the recovered S.
            T_back = float(np.interp(S_out[i], S_grid, T_col))
            residuals.append(abs(T_back - T_target[i]))

    if n_clip_below or n_clip_above:
        log.warning(
            'derive_ps_melting %s: clipped %d points below T grid floor and '
            '%d points above T ceiling',
            label,
            n_clip_below,
            n_clip_above,
        )

    # Write SPIDER canonical 2-column P-S file. Use the same scaling
    # factors as the WB+2018 distribution (P_scale=1e9, S_scale=K_B*N_A
    # for MgSiO3 = 4824266.85) so existing loaders accept the file
    # unchanged.
    P_scale = 1.0e9
    S_scale = 4824266.84604467  # = R_universal * 1000 / M_MgSiO3, hard-coded in WB17 dist
    P_nondim = P_clipped / P_scale
    S_nondim = S_out / S_scale

    with open(target_path, 'w') as out:
        out.write(f'# 5 {len(P_clipped)}\n')
        out.write('# Pressure [nondim], Entropy [nondim]\n')
        out.write('# column * scaling factor = SI units: Pressure [Pa], Entropy [J/kg/K]\n')
        out.write('# scaling factors (constant) for each column given on line below\n')
        out.write(f'# {P_scale} {S_scale}\n')
        for p, s in zip(P_nondim, S_nondim):
            out.write(f'{p:.18e} {s:.18e}\n')

    summary = {
        'n_points': len(P_clipped),
        'n_p_clipped': n_p_clipped,
        'n_clipped_below': n_clip_below,
        'n_clipped_above': n_clip_above,
        'max_inversion_residual_K': float(max(residuals)) if residuals else 0.0,
    }
    log.info(
        'derive_ps_melting %s: wrote %d points to %s; max inversion residual %.3f K',
        label,
        summary['n_points'],
        target_path,
        summary['max_inversion_residual_K'],
    )
    return summary


def _override_melting_curves_from_pt(
    eos_dir: str,
    solidus_pt_path: str,
    liquidus_pt_path: str,
    *,
    label_prefix: str = '',
) -> None:
    """Replace existing P-S melting curves in ``eos_dir`` with derivations
    from the configured P-T file.

    This is the single-source-of-truth glue: regardless of whether
    the P-S melting curves were byte-copied from the WB+2018 distribution
    (Case 2), inherited from the SPIDER submodule fallback (Case 3), or
    auto-generated by Zalmoxis (Case 1, PALEOS path), we overwrite them
    with derivations from the user-configured ``melting_dir`` P-T pair.

    Sanity prints the in-grid round-trip residual, the number of clipped
    points, and the change in T_sol(135 GPa) before/after the override
    (when applicable). All warnings flow through the standard logger.
    """
    sol_target = os.path.join(eos_dir, 'solidus_P-S.dat')
    liq_target = os.path.join(eos_dir, 'liquidus_P-S.dat')
    sol_phase = os.path.join(eos_dir, 'temperature_solid.dat')
    liq_phase = os.path.join(eos_dir, 'temperature_melt.dat')
    if not (os.path.isfile(sol_phase) and os.path.isfile(liq_phase)):
        log.warning(
            'override_melting %s: missing temperature_{solid,melt}.dat in %s; '
            'cannot derive melting curves, leaving in place',
            label_prefix,
            eos_dir,
        )
        return

    log.info(
        'override_melting %s: deriving P-S melting tables from %s / %s via %s + %s',
        label_prefix,
        solidus_pt_path,
        liquidus_pt_path,
        sol_phase,
        liq_phase,
    )

    summary_sol = _derive_ps_melting_curve(
        solidus_pt_path,
        sol_phase,
        sol_target,
        label=f'{label_prefix}_solidus',
    )
    summary_liq = _derive_ps_melting_curve(
        liquidus_pt_path,
        liq_phase,
        liq_target,
        label=f'{label_prefix}_liquidus',
    )
    if (
        summary_sol['n_clipped_below']
        + summary_sol['n_clipped_above']
        + summary_liq['n_clipped_below']
        + summary_liq['n_clipped_above']
    ) > 0:
        log.warning(
            'override_melting %s: total %d points clipped to EoS T grid edges '
            '(solidus: %d below, %d above; liquidus: %d below, %d above). '
            'These points fall outside the EoS lookup coverage and are '
            'pinned to the nearest valid S; the resulting curve is a '
            'straight extrapolation in T-S there.',
            label_prefix,
            summary_sol['n_clipped_below']
            + summary_sol['n_clipped_above']
            + summary_liq['n_clipped_below']
            + summary_liq['n_clipped_above'],
            summary_sol['n_clipped_below'],
            summary_sol['n_clipped_above'],
            summary_liq['n_clipped_below'],
            summary_liq['n_clipped_above'],
        )


def _is_spider_ps_format(path: str) -> bool:
    """Cheap first-line sniff to distinguish P-S tables from P-T tables.

    SPIDER's canonical P-S format starts with ``# 5 <n_S> <n_P>`` (5 is
    the number of header lines the loader expects). The P-T
    format (shipped in the Zenodo 17417017 record) starts with
    ``#pressure temperature density``. Both formats use the same
    filenames, so a content sniff is the only way to distinguish them
    when populating spider_eos_dir from FWL_DATA.
    """
    try:
        with open(path) as f:
            first = f.readline().strip()
    except OSError:
        return False
    # P-S header is exactly "# 5 <int> <int>".
    parts = first.split()
    return len(parts) >= 2 and parts[0] == '#' and parts[1] == '5'


def _provide_spider_eos_tables(config: Config, outdir: str, dirs: dict) -> None:
    """Ensure that Aragog and SPIDER can find a complete P-S lookup set.

    Populates ``output/<case>/data/spider_eos/`` with the 12 files both
    solvers need at runtime (10 phase-property files + 2 P-S melting
    curves). This is the PROTEUS-side data-resolution layer that lets
    ``interior_energetics.module = "aragog"`` work with
    ``interior_struct.module = "spider"``. In that combination no other
    code path produces the P-S tables Aragog needs at runtime, so this
    helper is their sole producer; without it the combination raises
    ``FileNotFoundError`` at solver setup.

    Resolution order (first available wins):

    1. **Already populated**: if ``dirs['spider_eos_dir']`` is set and
       the target directory already contains the 12 expected files, do
       nothing. This keeps Zalmoxis's ``generate_spider_tables()``
       output path and cache semantics untouched.

    2. **FWL_DATA (Zenodo 19473625)**: if the canonical Zenodo download
       target exists and is complete, copy the 12 files into the output
       directory. This is the self-sufficient path: once the user runs
       ``proteus get all`` (or any non-offline start), the Zenodo record
       populates FWL_DATA and subsequent runs find the complete set
       here.

    3. **SPIDER submodule fallback**: if FWL_DATA is incomplete but the
       SPIDER git submodule is cloned at ``dirs['spider']/lookup_data/``,
       copy the 10 phase files verbatim and rename the
       ``{solidus,liquidus}_A11_H13.dat`` melting curves to
       ``{solidus,liquidus}_P-S.dat``. This supports the
       workflow for users who have the submodule but haven't
       refreshed their FWL_DATA tree.

    4. **Hard failure**: if neither source yields a complete set, raise
       ``FileNotFoundError`` with a clear message pointing the user at
       ``proteus get all`` or the Zenodo record.

    Side effects: sets ``dirs['spider_eos_dir']``,
    ``dirs['spider_solidus_ps']``, ``dirs['spider_liquidus_ps']``.
    """
    target_dir = os.path.join(outdir, 'data', 'spider_eos')

    # Single source of truth: when `interior_struct.melting_dir` is
    # set in config, the runtime entropy-solver melting curves
    # (data/spider_eos/{solidus,liquidus}_P-S.dat) are *derived* from
    # that P-T file via the EoS's own T(P,S) lookup, rather than
    # byte-copied from the upstream WB+2018 distribution or auto-
    # generated by Zalmoxis. This keeps WB17 runs configured with
    # `melting_dir = "PALEOS-Fei2021"` using the configured P-S curves
    # at runtime instead of the WB+2018 curves.
    melting_dir = getattr(config.interior_struct, 'melting_dir', None)
    derive_melting = melting_dir is not None
    if derive_melting:
        from proteus.utils.data import GetFWLData as _GetFWL

        melting_pt_dir = _GetFWL() / 'interior_lookup_tables' / 'Melting_curves' / melting_dir
        sol_pt_path = melting_pt_dir / 'solidus_P-T.dat'
        liq_pt_path = melting_pt_dir / 'liquidus_P-T.dat'
        if not (sol_pt_path.is_file() and liq_pt_path.is_file()):
            log.warning(
                'melting_dir=%s configured but P-T files missing at %s; '
                'falling back to byte-copy from upstream EoS distribution',
                melting_dir,
                melting_pt_dir,
            )
            derive_melting = False

    # Case 1: already populated (e.g. by an earlier call this session or
    # by Zalmoxis's generate_spider_tables in a prior structure solve).
    existing = dirs.get('spider_eos_dir')
    if existing and os.path.isdir(existing):
        missing = [
            f
            for f in (_SPIDER_EOS_PHASE_FILES + _SPIDER_EOS_MELTING_CURVES)
            if not os.path.isfile(os.path.join(existing, f))
        ]
        if not missing:
            log.debug('spider_eos_dir already populated at %s, reusing', existing)
            if derive_melting:
                _override_melting_curves_from_pt(
                    existing,
                    str(sol_pt_path),
                    str(liq_pt_path),
                    label_prefix=f'reuse[{melting_dir}]',
                )
            dirs['spider_solidus_ps'] = os.path.join(existing, 'solidus_P-S.dat')
            dirs['spider_liquidus_ps'] = os.path.join(existing, 'liquidus_P-S.dat')
            return
        log.debug(
            'spider_eos_dir=%s exists but is missing %d file(s); repopulating',
            existing,
            len(missing),
        )

    os.makedirs(target_dir, exist_ok=True)

    # Import lazily so the helper is usable outside of a full PROTEUS
    # install (e.g. unit tests that stub out FWL_DATA).
    from proteus.utils.data import GetFWLData

    fwl_data = GetFWLData()
    zenodo_root = (
        fwl_data
        / 'interior_lookup_tables'
        / '1TPa-dK09-elec-free'
        / 'MgSiO3_Wolf_Bower_2018_1TPa'
    )

    # Case 2: FWL_DATA (Zenodo 19473625) complete set.
    # We check BOTH that all 12 files exist AND that density_melt.dat
    # is in the canonical SPIDER P-S format (not the P-T format
    # from the Zenodo 17417017 record, which ships under the
    # same filenames). Without this content check a FWL_DATA tree
    # holding the Zenodo 17417017 (P-T format) record would be silently
    # accepted and Aragog's EntropyEOS would then crash trying to parse
    # the P-T file as a P-S table.
    zenodo_files = list(_SPIDER_EOS_PHASE_FILES) + list(_SPIDER_EOS_MELTING_CURVES)
    zenodo_missing = [f for f in zenodo_files if not (zenodo_root / f).is_file()]
    zenodo_format_ok = not zenodo_missing and _is_spider_ps_format(
        str(zenodo_root / 'density_melt.dat')
    )
    if not zenodo_format_ok and not zenodo_missing:
        log.warning(
            'FWL_DATA EOS tables at %s exist but density_melt.dat is not '
            'in SPIDER P-S format. This usually means the directory was '
            'populated by the Zenodo 17417017 record (P-T format). '
            'Falling through to the SPIDER submodule. Refresh FWL_DATA '
            'with `proteus get all` to fetch Zenodo 19473625.',
            zenodo_root,
        )
    if zenodo_format_ok:
        log.info('Providing P-S EOS tables to spider_eos_dir from FWL_DATA (Zenodo 19473625)')
        for f in _SPIDER_EOS_PHASE_FILES:
            src = str(zenodo_root / f)
            dst = os.path.join(target_dir, f)
            if not os.path.isfile(dst):
                _rectangularize_spider_ps_file(src, dst)
        for f in _SPIDER_EOS_MELTING_CURVES:
            src = zenodo_root / f
            dst = os.path.join(target_dir, f)
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
        if derive_melting:
            _override_melting_curves_from_pt(
                target_dir,
                str(sol_pt_path),
                str(liq_pt_path),
                label_prefix=f'fwl[{melting_dir}]',
            )
        dirs['spider_eos_dir'] = target_dir
        dirs['spider_solidus_ps'] = os.path.join(target_dir, 'solidus_P-S.dat')
        dirs['spider_liquidus_ps'] = os.path.join(target_dir, 'liquidus_P-S.dat')
        return

    # Case 3: SPIDER submodule fallback. The 10 phase files have
    # canonical names; the 2 melting curves need the _A11_H13 -> _P-S
    # rename on copy.
    spider_bundle = None
    spider_root = dirs.get('spider')
    if spider_root:
        candidate = os.path.join(spider_root, 'lookup_data', '1TPa-dK09-elec-free')
        if os.path.isdir(candidate):
            spider_bundle = candidate

    if spider_bundle is not None:
        melt_map = {
            'solidus_P-S.dat': 'solidus_A11_H13.dat',
            'liquidus_P-S.dat': 'liquidus_A11_H13.dat',
        }
        phase_missing = [
            f
            for f in _SPIDER_EOS_PHASE_FILES
            if not os.path.isfile(os.path.join(spider_bundle, f))
        ]
        melt_missing = [
            canonical
            for canonical, legacy in melt_map.items()
            if not os.path.isfile(os.path.join(spider_bundle, legacy))
        ]
        if not phase_missing and not melt_missing:
            log.info(
                'Providing P-S EOS tables to spider_eos_dir from SPIDER submodule '
                '(bundled lookup_data). FWL_DATA was incomplete: missing %d file(s).',
                len(zenodo_missing),
            )
            for f in _SPIDER_EOS_PHASE_FILES:
                src = os.path.join(spider_bundle, f)
                dst = os.path.join(target_dir, f)
                if not os.path.isfile(dst):
                    _rectangularize_spider_ps_file(src, dst)
            for canonical, legacy in melt_map.items():
                src = os.path.join(spider_bundle, legacy)
                dst = os.path.join(target_dir, canonical)
                if not os.path.isfile(dst):
                    shutil.copy2(src, dst)
            if derive_melting:
                _override_melting_curves_from_pt(
                    target_dir,
                    str(sol_pt_path),
                    str(liq_pt_path),
                    label_prefix=f'spider_submodule[{melting_dir}]',
                )
            dirs['spider_eos_dir'] = target_dir
            dirs['spider_solidus_ps'] = os.path.join(target_dir, 'solidus_P-S.dat')
            dirs['spider_liquidus_ps'] = os.path.join(target_dir, 'liquidus_P-S.dat')
            return
        log.warning(
            'SPIDER submodule lookup_data present at %s but incomplete '
            '(phase missing=%d, melting curves missing=%d). Falling through.',
            spider_bundle,
            len(phase_missing),
            len(melt_missing),
        )

    # Case 4: neither source yielded a complete set.
    raise FileNotFoundError(
        'Could not provide SPIDER/Aragog P-S EOS tables at '
        f'{target_dir}. FWL_DATA source '
        f'{zenodo_root} is missing {len(zenodo_missing)} of '
        f'{len(zenodo_files)} required files '
        f'({zenodo_missing[:3]}...), and the SPIDER submodule fallback '
        f'was unavailable at {dirs.get("spider", "<no spider dir set>")}'
        '/lookup_data/1TPa-dK09-elec-free/. Run `proteus get all` to '
        'fetch Zenodo record 19473625, or ensure the SPIDER submodule '
        'is cloned.'
    )


def determine_interior_radius(
    dirs: dict, config: Config, hf_all: pd.DataFrame, hf_row: dict, outdir: str
):
    """
    Determine the interior radius (R_int) of the planet.

    This uses the interior model's hydrostatic integration to estimate the planet's
    interior mass from a given radius. The radius is then adjusted until the interior mass
    achieves the target mass provided by the user in the config file.
    """

    log.info('Using %s interior module to solve structure' % config.interior_energetics.module)

    # Provide P-S lookup tables for Aragog's entropy solver (and SPIDER
    # when it runs under this structure path). Mirrors the
    # generate_spider_tables() call at the top of the zalmoxis and dummy
    # structure paths. The helper resolves from FWL_DATA/Zenodo first,
    # then the SPIDER submodule as a fallback.
    if config.interior_energetics.module in ('spider', 'aragog'):
        _provide_spider_eos_tables(config, outdir, dirs)

    # R_int override: bypass the root finder and use a fixed radius.
    # This is needed for SPIDER/Aragog parity runs where the two
    # energetics modules have different Adams-Williamson density
    # implementations and the root finder converges to different R_int
    # for the same target mass.
    R_int_override = getattr(config.planet, 'R_int_override', None)
    if R_int_override is not None and R_int_override > 0:
        log.info(
            'R_int override active: using R_int = %.1f m = %.3f R_earth '
            '(bypassing root finder)',
            R_int_override,
            R_int_override / R_earth,
        )
        hf_row['R_int'] = float(R_int_override)
        calculate_core_mass(hf_row, config)

        if config.interior_energetics.module == 'spider':
            spider_dir = dirs['spider']
        else:
            spider_dir = None
        int_o = Interior_t(
            get_nlevb(config),
            spider_dir=spider_dir,
            eos_dir=config.interior_struct.eos_dir,
        )
        int_o.ic = 1
        hf_row['gravity'] = 9.81
        run_interior(dirs, config, hf_all, hf_row, int_o)
        update_gravity(hf_row)
        # Refresh the volatile inventory and total planet mass so the override
        # path leaves hf_row in the same state as the root-finder and the dummy
        # and Zalmoxis structure paths, which all populate M_ele and M_planet.
        calc_target_elemental_inventories(dirs, config, hf_row)
        update_planet_mass(hf_row)
        log.info('R_int: %.1e m  = %.3f R_earth', hf_row['R_int'], hf_row['R_int'] / R_earth)
        return

    # Initial guess for interior radius and gravity
    if config.interior_energetics.module == 'spider':
        spider_dir = dirs['spider']
    else:
        spider_dir = None
    int_o = Interior_t(
        get_nlevb(config), spider_dir=spider_dir, eos_dir=config.interior_struct.eos_dir
    )
    int_o.ic = 1
    hf_row['R_int'] = R_earth
    calculate_core_mass(hf_row, config)
    hf_row['gravity'] = 9.81

    # Target mass
    M_target = config.planet.mass_tot * M_earth

    # We need to solve for the state hf_row[M_planet] = config.planet.mass_tot
    # This function takes R_int as the input value, and returns the mass residual
    def _resid(x):
        hf_row['R_int'] = x

        log.debug('Try R = %.2e m = %.3f R_earth' % (x, x / R_earth))

        # Use interior model to get dry mass from radius
        calculate_core_mass(hf_row, config)
        run_interior(dirs, config, hf_all, hf_row, int_o, verbose=False)
        update_gravity(hf_row)

        # Get wet mass
        calc_target_elemental_inventories(dirs, config, hf_row)

        # Get total planet mass
        update_planet_mass(hf_row)

        # Calculate residual
        res = hf_row['M_planet'] - M_target
        log.debug('    yields M = %.5e kg , resid = %.3e kg' % (hf_row['M_planet'], res))

        return res

    # Set tolerance
    match config.interior_energetics.module:
        case 'aragog':
            rtol = config.interior_energetics.rtol
            atol = config.interior_energetics.aragog.tolerance_struct
        case 'spider':
            rtol = config.interior_energetics.rtol
            atol = config.interior_energetics.spider.tolerance_struct
        case _:
            rtol = 1e-7
            atol = 1e2

    # Find the radius
    r = optimise.root_scalar(
        _resid,
        method='secant',
        xtol=atol,
        rtol=rtol,
        maxiter=20,
        x0=hf_row['R_int'],
        x1=hf_row['R_int'] * 1.5,
    )
    # A non-converged secant (flat residual, EOS clamp, or NaN inside _resid)
    # would otherwise propagate a garbage radius into calculate_core_mass and
    # the rest of the trajectory. Hard-fail instead, as the Zalmoxis path does
    # on a mass-anchor violation.
    if not r.converged or not np.isfinite(r.root) or r.root <= 0.0:
        raise RuntimeError(
            'Interior radius secant solve failed: converged=%s, root=%r, flag=%r'
            % (r.converged, r.root, getattr(r, 'flag', None))
        )
    hf_row['R_int'] = float(r.root)
    calculate_core_mass(hf_row, config)
    run_interior(dirs, config, hf_all, hf_row, int_o)
    update_gravity(hf_row)

    # Result
    log.info('Found solution for interior structure')
    log.info(
        'M_planet: %.1e kg = %.3f M_earth' % (hf_row['M_planet'], hf_row['M_planet'] / M_earth)
    )
    log.info('R_int: %.1e m  = %.3f R_earth' % (hf_row['R_int'], hf_row['R_int'] / R_earth))
    log.info(' ')


def determine_interior_radius_with_dummy(
    dirs: dict, config: Config, hf_all: pd.DataFrame, hf_row: dict, outdir: str
):
    """Determine interior structure using Noack & Lasbleis (2020) scaling laws.

    Ultra-fast analytical parameterization replacing Zalmoxis. Fills all
    hf_row keys and writes output files needed by SPIDER/Aragog.
    """
    from proteus.interior_struct.dummy import solve_dummy_structure

    nlev_b = get_nlevb(config)
    num_spider_nodes = nlev_b if config.interior_energetics.module == 'spider' else 0

    spider_mesh_file = solve_dummy_structure(
        config,
        hf_row,
        outdir,
        num_spider_nodes=num_spider_nodes,
    )

    if spider_mesh_file:
        dirs['spider_mesh'] = spider_mesh_file
        dirs['spider_mesh_prev'] = spider_mesh_file + '.prev'

    # Generate P-S EOS tables for SPIDER/Aragog (if PALEOS)
    if config.interior_energetics.module in ('spider', 'aragog'):
        from proteus.interior_struct.zalmoxis import generate_spider_tables

        spider_tables = generate_spider_tables(config, outdir)
        if spider_tables is not None:
            dirs['spider_eos_dir'] = spider_tables['eos_dir']
            dirs['spider_solidus_ps'] = spider_tables['solidus_path']
            dirs['spider_liquidus_ps'] = spider_tables['liquidus_path']

    # Derived quantities
    hf_row['M_mantle'] = hf_row['M_int'] - hf_row['M_core']

    # Run first interior step
    int_o = Interior_t(
        nlev_b, spider_dir=dirs.get('spider'), eos_dir=config.interior_struct.eos_dir
    )
    int_o.ic = 1
    run_interior(dirs, config, hf_all, hf_row, int_o, verbose=False)
    update_gravity(hf_row)

    calc_target_elemental_inventories(dirs, config, hf_row)
    update_planet_mass(hf_row)

    log.info('Dummy structure solve complete')
    log.info(
        'R_int: %.1e m = %.3f R_earth, M_int: %.1e kg = %.3f M_earth',
        hf_row['R_int'],
        hf_row['R_int'] / R_earth,
        hf_row['M_int'],
        hf_row['M_int'] / M_earth,
    )


def determine_interior_radius_with_zalmoxis(
    dirs: dict, config: Config, hf_all: pd.DataFrame, hf_row: dict, outdir: str
):
    """
    Determine the interior radius (R_int) of the planet using Zalmoxis.

    When the interior module is SPIDER, also writes a SPIDER-format mesh
    file from the Zalmoxis structure solution and stores the path in
    ``dirs['spider_mesh']`` for subsequent calls.
    """

    log.info('Using Zalmoxis to solve for interior structure')
    from proteus.interior_struct.zalmoxis import zalmoxis_solver

    nlev_b = get_nlevb(config)
    spider_dir = dirs.get('spider') if config.interior_energetics.module == 'spider' else None
    int_o = Interior_t(nlev_b, spider_dir=spider_dir, eos_dir=config.interior_struct.eos_dir)
    int_o.ic = 1

    # Set Zalmoxis to 'adiabatic' mode for T-dependent mantle EOS.
    # NOTE: In practice, Zalmoxis converges the structure using a linear T
    # guess and breaks on mass convergence BEFORE the adiabat gate activates.
    # The adiabat flag is still set so that (a) the correct EOS code paths
    # are selected inside Zalmoxis, and (b) standalone Zalmoxis can use the
    # adiabat if the gate is ever fixed.  SPIDER provides its own T(r)
    # through entropy evolution, so the linear T initial guess is fine.
    _TDEP_PREFIXES = ('WolfBower2018', 'RTPress100TPa')
    _temp_mode_override: str | None = None
    if (
        config.interior_energetics.module == 'spider'
        and config.planet.temperature_mode == 'isothermal'
        and config.interior_struct.zalmoxis.mantle_eos.startswith(_TDEP_PREFIXES)
    ):
        log.info(
            'Overriding Zalmoxis temperature_mode from isothermal to adiabatic '
            'for SPIDER coupling with T-dependent mantle EOS (local override, '
            'does not mutate config)',
        )
        _temp_mode_override = 'adiabatic'

    # Request SPIDER mesh file if interior module is SPIDER.
    # Pass the override as a parameter instead of mutating config.planet.
    num_spider_nodes = nlev_b if config.interior_energetics.module == 'spider' else 0
    _cmb_radius, spider_mesh_file = zalmoxis_solver(
        config,
        outdir,
        hf_row,
        num_spider_nodes=num_spider_nodes,
        temperature_mode_override=_temp_mode_override,
    )

    # Store mesh file path for subsequent SPIDER calls
    if spider_mesh_file:
        dirs['spider_mesh'] = spider_mesh_file
        # Save initial mesh as baseline for blending comparisons
        prev_path = spider_mesh_file + '.prev'
        shutil.copy2(spider_mesh_file, prev_path)
        dirs['spider_mesh_prev'] = prev_path

    # Mesh convergence starts inactive (no blending needed at init)
    dirs['mesh_shift_active'] = False
    dirs['mesh_convergence_steps'] = 0

    # Generate P-S EOS tables from PALEOS if applicable.
    # Zalmoxis converts its PALEOS P-T EOS data into the P-S format that
    # both SPIDER and Aragog (entropy IC verify) need. Both PALEOS unified
    # and PALEOS-2phase layouts are supported. For non-PALEOS mantle EOS
    # (WolfBower2018, RTPress100TPa) Zalmoxis returns None and we fall
    # back to the static FWL_DATA Zenodo P-S tables. Aragog's entropy IC
    # verify just needs *some* P-S surface to map S_init -> T; the time
    # evolution uses the PALEOS-2phase P-T tables generated in
    # aragog.py:setup_solver.
    if config.interior_energetics.module in ('spider', 'aragog'):
        from proteus.interior_struct.zalmoxis import generate_spider_tables

        spider_tables = generate_spider_tables(config, outdir)
        if spider_tables is not None:
            dirs['spider_eos_dir'] = spider_tables['eos_dir']
            dirs['spider_solidus_ps'] = spider_tables['solidus_path']
            dirs['spider_liquidus_ps'] = spider_tables['liquidus_path']
        else:
            _provide_spider_eos_tables(config, outdir, dirs)

    # NOTE: run_interior runs with the *original* temperature_mode (restored
    # by the finally block above), not the overridden 'adiabatic'.  This is
    # correct: the Zalmoxis solver already used the adiabatic mode to compute
    # the structure, and run_interior (SPIDER/ARAGOG) manages its own T(r).
    run_interior(dirs, config, hf_all, hf_row, int_o)


def equilibrate_initial_state(dirs: dict, config: Config, hf_row: dict, outdir: str):
    """Iterate CALLIOPE + Zalmoxis until structure and composition converge.

    Runs volatile partitioning (CALLIOPE with optional binodal H2) followed
    by structure re-computation (Zalmoxis) in a loop. No SPIDER call.
    Convergence is checked on relative changes in R_int and P_surf.

    Called from proteus.py after the initial structure solve and volatile
    inventory setup. The converged state provides a self-consistent starting
    point for SPIDER's first time step.

    Parameters
    ----------
    dirs : dict
        Directory paths (including spider_mesh, spider_eos_dir).
    config : Config
        PROTEUS configuration.
    hf_row : dict
        Current helpfile row with volatile masses, R_int, P_surf, T_magma, etc.
    outdir : str
        Output directory for Zalmoxis files.
    """
    from proteus.interior_struct.zalmoxis import generate_spider_tables, zalmoxis_solver
    from proteus.outgas.wrapper import calc_target_elemental_inventories, run_outgassing

    max_iter = config.interior_struct.zalmoxis.equilibrate_max_iter
    tol = config.interior_struct.zalmoxis.equilibrate_tol
    nlev_b = get_nlevb(config)
    num_spider_nodes = nlev_b if config.interior_energetics.module == 'spider' else 0

    log.info(
        'Starting init equilibration loop (max %d iter, tol %.1f%%)',
        max_iter,
        tol * 100,
    )

    # Initialize convergence metrics (used in the warning if max_iter reached)
    delta_R = 1.0
    delta_P = 1.0

    for i in range(max_iter):
        R_old = float(hf_row.get('R_int', 0.0))
        P_old = float(hf_row.get('P_surf', 0.0))

        # 1. Volatile partitioning: recompute elemental targets and
        #    run CALLIOPE to get atmosphere/melt distribution
        calc_target_elemental_inventories(dirs, config, hf_row)
        run_outgassing(dirs, config, hf_row)

        # 2. Re-compute structure with updated composition
        #    (volatile_profile is built inside zalmoxis_solver from hf_row)
        _cmb_radius, spider_mesh_file = zalmoxis_solver(
            config, outdir, hf_row, num_spider_nodes=num_spider_nodes
        )

        # Update M_mantle from Zalmoxis results (M_int and M_core are set
        # by zalmoxis_solver, but M_mantle is not). run_outgassing needs
        # an up-to-date M_mantle for dissolved fraction calculations.
        hf_row['M_mantle'] = float(hf_row.get('M_int', 0.0)) - float(hf_row.get('M_core', 0.0))

        # Update mesh path if written
        if spider_mesh_file:
            dirs['spider_mesh'] = spider_mesh_file
            prev_path = spider_mesh_file + '.prev'
            shutil.copy2(spider_mesh_file, prev_path)
            dirs['spider_mesh_prev'] = prev_path

        # 3. Check convergence
        R_new = float(hf_row.get('R_int', 0.0))
        P_new = float(hf_row.get('P_surf', 0.0))

        delta_R = abs(R_new - R_old) / R_old if R_old > 0 else 1.0
        delta_P = abs(P_new - P_old) / P_old if P_old > 0 else 1.0

        log.info(
            'Equilibration iter %d/%d: dR/R=%.4f, dP/P=%.4f (R=%.3e m, P=%.2f bar)',
            i + 1,
            max_iter,
            delta_R,
            delta_P,
            R_new,
            P_new,
        )

        if delta_R < tol and delta_P < tol:
            log.info('Equilibration converged after %d iterations', i + 1)
            break
    else:
        log.warning(
            'Equilibration did not converge after %d iterations (dR/R=%.4f, dP/P=%.4f)',
            max_iter,
            delta_R,
            delta_P,
        )

    # 4. Regenerate SPIDER EOS tables with final composition
    if config.interior_energetics.module in ('spider', 'aragog'):
        spider_tables = generate_spider_tables(config, outdir)
        if spider_tables is not None:
            dirs['spider_eos_dir'] = spider_tables['eos_dir']
            dirs['spider_solidus_ps'] = spider_tables['solidus_path']
            dirs['spider_liquidus_ps'] = spider_tables['liquidus_path']


def solve_structure(
    dirs: dict, config: Config, hf_all: pd.DataFrame, hf_row: dict, outdir: str
):
    """
    Solve for the planet structure based on the method set in the configuration file.

    If the structure is set by the radius, then this is trivial because the radius is used
    as an input to the interior modules anyway. If the structure is set by mass, then it is
    solved as an inverse problem for now.
    """

    # Set by total mass (mantle + core + volatiles)
    if config.planet.mass_tot is not None:
        # Choose the method to determine the interior radius
        match config.interior_struct.module:
            case 'dummy':
                return determine_interior_radius_with_dummy(
                    dirs, config, hf_all, hf_row, outdir
                )
            case 'spider':
                return determine_interior_radius(dirs, config, hf_all, hf_row, outdir)
            case 'zalmoxis':
                # Zalmoxis computes its own radius; temporarily disable orbital
                # feedback during structure solve (restored after return).
                _orig_orbit_module = config.orbit.module
                config.orbit.module = 'dummy'
                try:
                    if config.params.stop.solid.phi_crit < 0.01:
                        log.warning(
                            'phi_crit=%.4f is below 0.01. Zalmoxis cases may plateau '
                            'at ~0.9%% melt fraction, so phi_crit < 0.01 can prevent '
                            'the simulation from terminating. Consider phi_crit >= 0.01.',
                            config.params.stop.solid.phi_crit,
                        )
                    return determine_interior_radius_with_zalmoxis(
                        dirs, config, hf_all, hf_row, outdir
                    )
                finally:
                    config.orbit.module = _orig_orbit_module
        raise ValueError(
            f"Invalid structure interior module selected '{config.interior_struct.module}'"
        )

    else:
        raise ValueError('planet.mass_tot must be set to solve for the interior structure')


def run_interior(
    dirs: dict,
    config: Config,
    hf_all: pd.DataFrame,
    hf_row: dict,
    interior_o: Interior_t,
    atmos_o=None,
    verbose: bool = True,
    write_data: bool = True,
):
    """Run interior mantle evolution model.

    Parameters
    ----------
        dirs : dict
            Dictionary of directories.
        config : Config
            Model configuration
        hf_all : pd.DataFrame
            Dataframe of historical runtime variables
        hf_row : dict
            Dictionary of current runtime variables
        interior_o : Interior_t
            Interior struct.
        atmos_o : Atmos_t or None
            Atmosphere struct. Required only for the boundary backend; the
            other backends ignore it.
        verbose : bool
            Verbose printing enabled.
        write_data : bool
            Write per-timestep data files (NetCDF/JSON) to disk. When False,
            the solver still runs but skips the data file write. Used by the
            dt_write time guard to prevent excessive output during rapid
            early evolution. SPIDER JSON writes are unaffected (managed by
            the C binary).
    """

    # Use the appropriate interior model
    if verbose:
        log.info('Evolve interior...')
    log.debug('Using %s module to evolve interior' % config.interior_energetics.module)

    # Write tidal heating file
    if config.interior_energetics.heat_tidal:
        interior_o.write_tides(dirs['output'])

    if config.interior_energetics.module == 'spider':
        # Import
        from proteus.interior_energetics.spider import ReadSPIDER, RunSPIDER

        # Run SPIDER (pass external mesh file if available from Zalmoxis).
        # Note: write_data is not forwarded here. SPIDER JSON output is
        # controlled by the C binary; Python cannot suppress it per-timestep.
        mesh_file = dirs.get('spider_mesh')
        try:
            RunSPIDER(dirs, config, hf_all, hf_row, interior_o, mesh_file=mesh_file)
            interior_o.spider_fail_count = 0
        except RuntimeError as e:
            interior_o.spider_fail_count += 1
            log.warning(
                'SPIDER CVode failure #%d/%d. '
                'Keeping previous interior state for this step. Error: %s',
                interior_o.spider_fail_count,
                _SPIDER_MAX_CONSECUTIVE_FAILS,
                str(e)[:200],
            )
            if interior_o.spider_fail_count >= _SPIDER_MAX_CONSECUTIVE_FAILS:
                log.error(
                    'SPIDER failed %d consecutive times. Aborting.',
                    interior_o.spider_fail_count,
                )
                raise
            # Skip ReadSPIDER; keep hf_row values from previous step.
            # The interior state is stale by one coupling step, but the
            # atmosphere + outgassing still advance. The stale interior
            # pushes the planet past the stiff rheological transition.
            #
            # IMPORTANT: still advance interior_o.dt and _spider_cumulative_time
            # so that main-loop bookkeeping stays consistent. Without this,
            # the main loop at proteus.py:496 increments hf_row['Time'] by the
            # previous successful dt (stale) while _spider_cumulative_time is
            # frozen, leaving the two counters desynchronised by one dtswitch
            # per fallback event. That in turn confuses the retry ladder on
            # the next attempt.
            from proteus.interior_energetics.timestep import next_step

            dtswitch = next_step(
                config,
                dirs,
                hf_row,
                hf_all,
                1.0,
                interior_o=interior_o,
            )
            interior_o._spider_cumulative_time += dtswitch
            interior_o.dt = dtswitch
            return
        sim_time, output = ReadSPIDER(dirs, config, hf_row['R_int'], interior_o)

    elif config.interior_energetics.module == 'aragog':
        from proteus.interior_energetics.aragog import AragogRunner

        runner = AragogRunner(config, dirs, hf_row, hf_all, interior_o)
        try:
            sim_time, output = runner.run_solver(
                hf_row,
                interior_o,
                dirs,
                write_data=write_data,
            )
            interior_o.aragog_fail_count = 0
        except RuntimeError as e:
            interior_o.aragog_fail_count += 1
            log.warning(
                'Aragog retry-ladder exhaustion #%d/%d. '
                'Keeping previous interior state for this step. Error: %s',
                interior_o.aragog_fail_count,
                _ARAGOG_MAX_CONSECUTIVE_FAILS,
                str(e)[:200],
            )
            if interior_o.aragog_fail_count >= _ARAGOG_MAX_CONSECUTIVE_FAILS:
                log.error(
                    'Aragog failed %d consecutive times. Aborting.',
                    interior_o.aragog_fail_count,
                )
                raise
            # Skip output update; keep hf_row values from previous step.
            # Atmosphere + outgassing still advance, pushing the planet
            # past the stiff regime. Same pattern as SPIDER fallback above.
            from proteus.interior_energetics.timestep import next_step

            dtswitch = next_step(
                config,
                dirs,
                hf_row,
                hf_all,
                1.0,
                interior_o=interior_o,
            )
            interior_o.dt = dtswitch
            return

    elif config.interior_energetics.module == 'boundary':
        from proteus.interior_energetics.boundary import BoundaryRunner

        BoundaryRunnerInstance = BoundaryRunner(
            config, dirs, hf_row, hf_all, interior_o, atmos_o
        )
        sim_time, output = BoundaryRunnerInstance.run_solver(hf_row, interior_o, dirs)

    elif config.interior_energetics.module == 'dummy':
        # Import
        from proteus.interior_energetics.dummy import run_dummy_int

        # Run dummy interior
        sim_time, output = run_dummy_int(config, dirs, hf_row, hf_all, interior_o)

    # Read output
    for k in output.keys():
        if k in hf_row.keys():
            val = output[k]
            # Convert numpy arrays and scalars to Python scalars for NumPy 2.0 compatibility
            if isinstance(val, np.generic):
                hf_row[k] = val.item()
            elif np.isscalar(val):
                hf_row[k] = val
            else:
                try:
                    arr = np.asarray(val)
                    if arr.size == 1 and hasattr(arr, 'item'):
                        hf_row[k] = arr.item()
                    else:
                        hf_row[k] = val
                except Exception as exc:
                    log.error(
                        'Failed to convert output value for key %r (%r) to a NumPy array/scalar (%s: %s)',
                        k,
                        val,
                        type(exc).__name__,
                        exc,
                    )
                    raise

    # Update rheological parameters
    #    Only calculate viscosity here if using dummy module
    calc_visc = bool(config.interior_energetics.module == 'dummy')
    interior_o.update_rheology(visc=calc_visc)

    # Ensure values are >= 0
    for k in ('M_mantle', 'M_mantle_liquid', 'M_mantle_solid', 'M_core', 'Phi_global'):
        hf_row[k] = max(hf_row[k], 0.0)

    # Check that the new temperature is remotely reasonable
    if not (0 < hf_row['T_magma'] < 1e6):
        UpdateStatusfile(dirs, 21)
        raise ValueError('T_magma is out of range: %g K' % float(hf_row['T_magma']))

    # Update dry interior mass
    hf_row['M_int'] = hf_row['M_mantle'] + hf_row['M_core']

    # Update planet mass
    update_planet_mass(hf_row)

    # Apply step limiters
    if hf_row['Time'] > 0:
        # Prevent increasing surface temperature, if enabled. Gated by
        # _prevent_warming_clamp_active(); the runaway-T fallback below
        # remains active regardless.
        T_magma_prev = float(hf_all.iloc[-1]['T_magma'])
        T_surf_prev = float(hf_all.iloc[-1]['T_surf'])
        Phi_global_prev = float(hf_all.iloc[-1]['Phi_global'])
        F_int_prev = float(hf_all.iloc[-1]['F_int'])
        if _prevent_warming_clamp_active(config) and (interior_o.ic == 2):
            hf_row['Phi_global'] = min(hf_row['Phi_global'], Phi_global_prev)
            hf_row['T_magma'] = min(hf_row['T_magma'], T_magma_prev)
            hf_row['T_surf'] = min(hf_row['T_surf'], T_surf_prev)
            hf_row['F_int'] = min(hf_row['F_int'], F_int_prev)

        # F_int positivity floor under prevent_warming, applied for all
        # ic values (not just ic == 2). SPIDER's JSON output can produce
        # a slightly-negative F_int on the first post-restart step (ic
        # = 1) because the thermal state is read from the previous
        # solver epoch; the floor is what stopped a negative flux from
        # propagating to the helpfile + atmosphere BC before this floor
        # was relocated out of ReadSPIDER in the 7g commit.
        if _prevent_warming_clamp_active(config):
            hf_row['F_int'] = max(1.0e-8, hf_row['F_int'])

        # Do not allow massive increases to T_magma or T_surf.
        #
        # T_magma uses the SPIDER/Aragog/dummy tolerance formula for
        # every backend. T_surf uses Tsurf_event_change ONLY in the
        # boundary backend (Calder's BL model has a terminal ODE event
        # at |T_surf - T_surf_0| = Tsurf_event_change, so reusing that
        # threshold here keeps the outer-loop cap consistent with the
        # inner-loop event). For all other backends T_surf shares the
        # T_magma budget.
        dT_delta_magma = config.interior_energetics.tmagma_atol
        dT_delta_magma += config.interior_energetics.tmagma_rtol * T_magma_prev

        if config.interior_energetics.module == 'boundary':
            dT_delta_surf = float(config.interior_energetics.boundary.Tsurf_event_change)
        else:
            dT_delta_surf = dT_delta_magma

        if hf_row['T_magma'] > T_magma_prev + dT_delta_magma:
            log.warning('Prevented large increase to T_magma!')
            log.warning('   Clipped from %.2f K' % hf_row['T_magma'])
            hf_row['T_magma'] = T_magma_prev + dT_delta_magma
            hf_row['Phi_global'] = Phi_global_prev

        if hf_row['T_surf'] > T_surf_prev + dT_delta_surf:
            log.warning('Prevented large increase to T_surf!')
            log.warning('   Clipped from %.2f K' % hf_row['T_surf'])
            hf_row['T_surf'] = T_surf_prev + dT_delta_surf

    # Print result of interior module
    if verbose:
        log.info('    T_magma    = %.3f K' % float(hf_row['T_magma']))
        log.info('    Phi_global = %.3f  ' % float(hf_row['Phi_global']))
        log.info('    RF_depth   = %.3f  ' % float(hf_row['RF_depth']))
        log.info('    F_int      = %.2e W m-2' % float(hf_row['F_int']))
        if config.interior_energetics.heat_tidal:
            log.info('    F_tidal    = %.2e W m-2' % float(hf_row['F_tidal']))
        if config.interior_energetics.heat_radiogenic:
            log.info('    F_radio    = %.2e W m-2' % float(hf_row['F_radio']))

    # Actual time step size.
    # Use SPIDER's actual sim_time (read from 'time_years' inside the JSON,
    # not from the llround'd filename) to compute the true dt. This fixes
    # the desync where PROTEUS advanced by dtswitch while SPIDER only
    # evolved to a tsurf_poststep_change truncation point. The previous
    # approach (always using dtswitch) caused PROTEUS's clock to race
    # ahead of SPIDER's internal state by hundreds to thousands of years
    # when the poststep change limit was hit every step.
    interior_o.dt = float(sim_time) - hf_row['Time']
    if interior_o.dt <= 0:
        # sim_time <= hf_row['Time'] happens for every backend at the first
        # step (ic == 1 returns sim_time == Time), and for a genuine solver
        # rollback mid-run (rare; e.g. SPIDER reloading a stale JSON state).
        # Fall back to dtswitch so the main loop neither stalls nor goes
        # backwards. This is expected at initialisation and only noteworthy
        # afterwards, so the message is demoted to debug on the first step.
        from proteus.interior_energetics.timestep import next_step

        dtswitch = next_step(
            config,
            dirs,
            hf_row,
            hf_all,
            1.0,
            interior_o=interior_o,
        )
        module = config.interior_energetics.module
        if interior_o.ic == 1:
            log.debug(
                '%s reported sim_time (%.2f yr) == hf_row[Time] (%.2f yr) at '
                'initialisation; using dtswitch=%.2f yr',
                module,
                float(sim_time),
                hf_row['Time'],
                dtswitch,
            )
        else:
            log.warning(
                '%s sim_time (%.2f yr) <= hf_row[Time] (%.2f yr); '
                'falling back to dtswitch=%.2f yr',
                module,
                float(sim_time),
                hf_row['Time'],
                dtswitch,
            )
        interior_o.dt = dtswitch

    # Note: Aragog mesh refresh after a Zalmoxis re-solve runs through the
    # normal coupling path. update_structure_from_interior (below) writes a
    # new zalmoxis_output.dat with current R_int / R_core / gravity. The
    # next interior step reaches AragogRunner.setup_or_update_solver
    # (aragog.py:116), which calls update_structure to copy the updated
    # scalars into solver.parameters.mesh, then solver.reset(), which
    # re-reads the external EOS file (entropy_solver.py:441 with
    # eos_method=2). The init-time equilibrate_initial_state loop runs
    # CALLIOPE + Zalmoxis before the Aragog solver exists, so no refresh
    # is needed there: the first Aragog setup_solver call after
    # equilibration sees the final zalmoxis_output.dat.


def update_structure_from_interior(
    dirs: dict,
    config: Config,
    hf_row: dict,
    interior_o: Interior_t,
    last_struct_time: float,
    last_Tmagma: float,
    last_Phi: float,
) -> tuple[float, float, float]:
    """Re-run Zalmoxis with SPIDER's current T(r) to update structure.

    Uses a hybrid trigger: fires when either the relative change in
    T_magma or the absolute change in Phi_global exceeds configured
    thresholds, subject to a minimum interval (floor) and maximum
    interval (ceiling).  When ``update_interval == 0``, no dynamic
    updates are performed (structure is computed only at init).

    Writes SPIDER's temperature profile to a file, runs Zalmoxis in
    prescribed-temperature mode, and writes an updated mesh file for the
    next SPIDER call.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories.
    config : Config
        Model configuration.
    hf_row : dict
        Current runtime variables.
    interior_o : Interior_t
        Interior state with current T(r) on staggered nodes.
    last_struct_time : float
        Simulation time [yr] of the last structure update.
    last_Tmagma : float
        T_magma [K] at the last structure update.
    last_Phi : float
        Phi_global at the last structure update.

    Returns
    -------
    tuple[float, float, float]
        (last_struct_time, last_Tmagma, last_Phi), updated to current
        values if an update occurred, otherwise returned unchanged.
    """
    no_update = (last_struct_time, last_Tmagma, last_Phi)

    # Dynamic updates disabled
    if config.interior_struct.zalmoxis.update_interval <= 0:
        return no_update

    current_time = hf_row['Time']
    elapsed = current_time - last_struct_time

    # Evaluate triggers
    triggered = False
    reason = ''

    # Mesh convergence trigger: bypasses normal floor when mesh is still
    # converging toward the true Zalmoxis solution after blending
    mesh_converging = dirs.get('mesh_shift_active', False)
    if mesh_converging and elapsed >= config.interior_struct.zalmoxis.mesh_convergence_interval:
        triggered = True
        reason = (
            f'mesh convergence (elapsed {elapsed:.1f} yr '
            f'>= {config.interior_struct.zalmoxis.mesh_convergence_interval:.1f} yr)'
        )

    # Floor: don't update too frequently (only for non-convergence triggers)
    if not triggered and elapsed < config.interior_struct.zalmoxis.update_min_interval:
        return no_update

    # Ceiling: guaranteed update after max interval
    if not triggered and elapsed >= config.interior_struct.zalmoxis.update_interval:
        triggered = True
        reason = f'ceiling ({elapsed:.1f} yr >= {config.interior_struct.zalmoxis.update_interval:.1f} yr)'

    # Stale-aware ceiling: trigger if elapsed time since the LAST
    # SUCCESSFUL structure refresh (not the last call) exceeds
    # update_stale_ceiling. Without this, a sequence of failed
    # re-solves resets `last_struct_time` to the failure time and
    # the next ceiling waits a full update_interval, meaning Aragog
    # can integrate through an entire window with a frozen mesh
    # (tens of kyr). Bypasses the
    # update_min_interval floor for the same reason as the
    # mesh-converging trigger: this is a recovery path, not a
    # routine refresh.
    if not triggered:
        stale_ceiling = config.interior_struct.zalmoxis.update_stale_ceiling
        if stale_ceiling > 0:
            last_success = float(
                getattr(interior_o, 'last_successful_struct_time', float('-inf'))
            )
            if last_success > float('-inf'):
                stale_elapsed = current_time - last_success
                if stale_elapsed >= stale_ceiling:
                    triggered = True
                    reason = (
                        f'stale-aware ceiling ({stale_elapsed:.1f} yr since '
                        f'last successful re-solve >= {stale_ceiling:.1f} yr)'
                    )

    # Phi_global absolute change (primary trigger: directly reflects rheological state)
    if not triggered:
        dPhi = abs(hf_row['Phi_global'] - last_Phi)
        if dPhi >= config.interior_struct.zalmoxis.update_dphi_abs:
            triggered = True
            reason = f'dPhi={dPhi:.3f} >= {config.interior_struct.zalmoxis.update_dphi_abs}'

    # T_magma relative change (secondary: catches cases where Phi is constant but T changes)
    if not triggered and last_Tmagma > 0:
        dT_frac = abs(hf_row['T_magma'] - last_Tmagma) / last_Tmagma
        if dT_frac >= config.interior_struct.zalmoxis.update_dtmagma_frac:
            triggered = True
            reason = (
                f'dT/T={dT_frac:.3f} >= {config.interior_struct.zalmoxis.update_dtmagma_frac}'
            )

    # Composition change: check dissolved volatile fractions.
    # When the binodal or CALLIOPE changes how much H2/H2O is dissolved,
    # the mantle density profile shifts, requiring a structure update.
    comp_changed = False
    if not triggered:
        M_mantle = float(hf_row.get('M_mantle', 0.0))
        if M_mantle > 0:
            for species in ('H2O', 'H2'):
                w_new = float(hf_row.get(f'{species}_kg_liquid', 0.0)) / M_mantle
                w_old = dirs.get(f'_last_w_{species}_liquid', w_new)
                if w_old > 1e-6:
                    dw = abs(w_new - w_old) / w_old
                    dw_threshold = config.interior_struct.zalmoxis.update_dw_comp_abs
                    if dw >= dw_threshold:
                        triggered = True
                        comp_changed = True
                        reason = f'd_w_{species}={dw:.3f} >= {dw_threshold}'
                        break

    if not triggered:
        return no_update

    log.info('Updating structure from interior T(r) via Zalmoxis (trigger: %s)', reason)

    outdir = dirs['output']

    # Build SPIDER's mantle T(r) in ascending radius (CMB to surface)
    # interior_o.radius is basic nodes (surface to CMB), temp is staggered nodes
    r_stag = 0.5 * (interior_o.radius[:-1] + interior_o.radius[1:])
    r_ascending = r_stag[::-1]
    T_ascending = interior_o.temp[::-1]

    # Build T(r,P) interpolator from SPIDER/Aragog output in memory.
    # SPIDER only covers the mantle (CMB to surface), so hold T constant
    # at the CMB value for the core region.
    R_cmb = float(np.squeeze(r_ascending[0]))
    T_cmb = float(np.squeeze(T_ascending[0]))

    # Capture arrays in closure for the temperature function
    _r_asc = np.asarray(r_ascending, dtype=float)
    _T_asc = np.asarray(T_ascending, dtype=float)
    _R_cmb = R_cmb
    _T_cmb = T_cmb

    def temperature_function(r, P):
        if r <= _R_cmb:
            return _T_cmb
        return float(np.interp(r, _r_asc, _T_asc))

    from proteus.interior_struct.zalmoxis import zalmoxis_solver

    # Save current mesh as baseline for blending
    prev_path = dirs.get('spider_mesh_prev')
    current_mesh = dirs.get('spider_mesh')
    if current_mesh and os.path.isfile(current_mesh):
        if not prev_path:
            prev_path = current_mesh + '.prev'
            dirs['spider_mesh_prev'] = prev_path
        # Skip the copy when src and dst already point to the same file. On
        # the Zalmoxis-failure fall-back the previous structure is restored by
        # pointing spider_mesh at its own .prev, so this snapshot would become
        # a self copy and shutil.copy2 raises SameFileError. The .prev already
        # holds the last-good baseline in that case, so skipping is correct.
        if os.path.abspath(current_mesh) != os.path.abspath(prev_path):
            shutil.copy2(current_mesh, prev_path)

    # Atomically capture zalmoxis_output.dat -> .prev BEFORE the call.
    # zalmoxis_solver also writes .prev (zalmoxis.py:1888) but only if
    # the call advances past the pre-write checkpoint. When Zalmoxis fails
    # earlier (e.g. Picard plateau), its in-solver .prev save is skipped and
    # the on-disk .prev still reflects the call from one iter further back.
    # The fallback at line ~1768 then restores from this stale .prev,
    # producing an EOS-vs-mesh mismatch with hf_row's _saved_structure
    # (which is consistent with the most-recent successful state).
    # Capturing here unconditionally guarantees .prev == "state at start
    # of this update_structure_from_interior call" == _saved_structure.
    _output_zalmoxis_path = os.path.join(outdir, 'data', 'zalmoxis_output.dat')
    if os.path.isfile(_output_zalmoxis_path):
        try:
            shutil.copy2(_output_zalmoxis_path, _output_zalmoxis_path + '.prev')
        except OSError as _exc:
            log.warning(
                'Could not capture zalmoxis_output.dat -> .prev pre-call: %s',
                _exc,
            )

    nlev_b = get_nlevb(config)
    num_spider_nodes = nlev_b if config.interior_energetics.module == 'spider' else 0

    # Save current structure values for fallback on convergence failure
    _saved_structure = {
        k: hf_row[k]
        for k in (
            'R_int',
            'M_int',
            'M_core',
            'M_mantle',
            'P_surf',
            'R_core',
            'P_center',
            'rho_avg',
        )
        if k in hf_row
    }

    # Also hand the (r, T) arrays to Zalmoxis explicitly. The JAX path
    # needs them in r-indexed form because the default P-indexed
    # tabulation in jax_eos/wrapper.py collapses for this closure
    # (T_asc varies with r and ignores P).
    # The numpy path ignores temperature_arrays and uses the callable.
    # Force strict ascending sort by r: jnp.interp requires monotonic
    # increasing xp, and ``r_ascending = r_stag[::-1]`` above can end up
    # descending depending on Aragog's per-call radius ordering. An
    # explicit argsort is cheap (~150 elements) and removes the
    # convention-dependence. A descending array arriving as
    # [surface, ..., CMB] would otherwise produce ``Final M=0``
    # failures in JAX.
    _r_for_arrays = np.asarray(_r_asc, dtype=float)
    _T_for_arrays = np.asarray(_T_asc, dtype=float)
    _order = np.argsort(_r_for_arrays)
    _r_for_arrays = np.ascontiguousarray(_r_for_arrays[_order])
    _T_for_arrays = np.ascontiguousarray(_T_for_arrays[_order])

    try:
        import time as _zalmoxis_time

        _zalmoxis_wall_t0 = _zalmoxis_time.monotonic()
        _cmb_radius, spider_mesh_file = zalmoxis_solver(
            config,
            outdir,
            hf_row,
            num_spider_nodes=num_spider_nodes,
            temperature_function=temperature_function,
            temperature_arrays=(_r_for_arrays, _T_for_arrays),
        )
        _zalmoxis_wall = _zalmoxis_time.monotonic() - _zalmoxis_wall_t0
        # Mass-anchor check: enforce |M_int / M_int_target - 1| <
        # _ZALMOXIS_MASS_ANCHOR_TOL after every successful Zalmoxis call.
        # Raise RuntimeError on violation so the except-block
        # fall-back path runs (restore _saved_structure, set
        # _structure_stale=True, increment interior_o.zalmoxis_fail_count). This
        # treats a too-loose-converged Zalmoxis result the same as a
        # non-converged one.
        _M_target = float(hf_row.get('M_int_target', 0.0) or 0.0)
        _M_int = float(hf_row.get('M_int', 0.0) or 0.0)
        if _M_target > 0.0:
            _mass_rel_err = abs(_M_int / _M_target - 1.0)
            if _mass_rel_err > _ZALMOXIS_MASS_ANCHOR_TOL:
                raise RuntimeError(
                    'Zalmoxis mass-anchor violation: '
                    '|M_int / M_int_target - 1| = %.3e > tol=%.3e '
                    '(M_int=%.4e kg, M_target=%.4e kg). '
                    'Treating as non-converged.'
                    % (
                        _mass_rel_err,
                        _ZALMOXIS_MASS_ANCHOR_TOL,
                        _M_int,
                        _M_target,
                    )
                )
        if interior_o.zalmoxis_fail_count > 0:
            # Quantify how often the relaxed budget actually saved a run: log
            # the streak length before zeroing so post-hoc analysis can grep
            # "consecutive-failure streak reset" from proteus_00.log.
            log.info(
                'Zalmoxis consecutive-failure streak reset on success '
                '(streak length = %d / %d, trigger: %s)',
                interior_o.zalmoxis_fail_count,
                _ZALMOXIS_MAX_CONSECUTIVE_FAILS,
                reason,
            )
        interior_o.zalmoxis_fail_count = 0  # Reset on success
        # Clear the stale-structure flag so downstream consumers
        # (Aragog setup_or_update_solver) can rely on it. The flag is
        # set to True on fall-back and cleared here on success, so
        # Aragog can tell whether it is running on a fresh or stale mesh.
        hf_row['_structure_stale'] = False
        # Anchor the stale-aware ceiling on the last SUCCESSFUL
        # re-solve (vs `last_struct_time` which is reset on every
        # call regardless of success).
        try:
            interior_o.last_successful_struct_time = float(current_time)
        except AttributeError:
            # Mock callers without Interior_t: best-effort no-op.
            pass
        # Per-re-solve wall-time trace logged at INFO so the
        # convergence cadence cost can be read from proteus_00.log.
        log.info(
            'Zalmoxis re-solve wall: %.2f s (trigger: %s)',
            _zalmoxis_wall,
            reason,
        )
    except RuntimeError as e:
        interior_o.zalmoxis_fail_count += 1
        log.warning(
            'Zalmoxis convergence failure #%d/%d during time evolution. '
            'Falling back to previous structure. Error: %s',
            interior_o.zalmoxis_fail_count,
            _ZALMOXIS_MAX_CONSECUTIVE_FAILS,
            str(e)[:200],
        )
        if interior_o.zalmoxis_fail_count >= _ZALMOXIS_MAX_CONSECUTIVE_FAILS:
            log.error(
                'Zalmoxis failed %d consecutive times. Aborting.',
                interior_o.zalmoxis_fail_count,
            )
            raise
        # Restore previous structure values
        hf_row.update(_saved_structure)
        hf_row['_structure_stale'] = True
        # Prefer the .prev snapshot written before the failed Zalmoxis
        # call (line ~1232) rather than dirs['spider_mesh'], which may
        # point to the partial file Zalmoxis was writing when it
        # raised. Falls back to dirs['spider_mesh'] only if no .prev
        # exists (first Zalmoxis failure of the run before any
        # snapshot was saved).
        spider_mesh_file = prev_path or dirs.get('spider_mesh')
        _cmb_radius = float(hf_row.get('R_core', 0.0))
        # Also restore zalmoxis_output.dat from
        # its .prev backup when the wrapper-level mass-anchor check (or
        # any other post-zalmoxis_solver wrapper RuntimeError) raises.
        # zalmoxis_solver creates the .prev backup atomically, but if
        # the raise happens AFTER zalmoxis_solver returns successfully
        # (e.g. the mass-anchor check), the new file is on disk and
        # Aragog will crash on the next iter with EOS-vs-mesh
        # inconsistency unless we roll back here.
        try:
            _output_zalmoxis = os.path.join(outdir, 'data', 'zalmoxis_output.dat')
            _output_prev = _output_zalmoxis + '.prev'
            if os.path.isfile(_output_prev):
                shutil.copy2(_output_prev, _output_zalmoxis)
                log.info(
                    'Fall-back: restored %s from %s',
                    _output_zalmoxis,
                    _output_prev,
                )
        except Exception as _exc:
            log.warning(
                'Could not restore zalmoxis_output.dat from .prev on fall-back: %s',
                _exc,
            )

    if spider_mesh_file:
        dirs['spider_mesh'] = spider_mesh_file

        # Blend mesh to limit per-update radius shift
        from proteus.interior_energetics.spider import blend_mesh_files

        actual_shift = blend_mesh_files(
            prev_path or '',
            spider_mesh_file,
            max_shift=config.interior_struct.zalmoxis.mesh_max_shift,
        )
        still_converging = actual_shift > config.interior_struct.zalmoxis.mesh_max_shift

        # Track convergence steps; give up after 20 consecutive blends
        # to avoid infinite rapid-update loops when Zalmoxis and SPIDER
        # persistently disagree (e.g. extreme mass / low CMF)
        max_convergence_steps = 20
        n_conv = dirs.get('mesh_convergence_steps', 0)
        if still_converging:
            n_conv += 1
            if n_conv > max_convergence_steps:
                log.warning(
                    'Mesh convergence did not complete after %d steps '
                    '(shift still %.1f%%), reverting to normal triggers',
                    max_convergence_steps,
                    actual_shift * 100,
                )
                still_converging = False
                n_conv = 0
            else:
                log.info(
                    'Mesh convergence step %d/%d: shift %.1f%% clamped to %.1f%%',
                    n_conv,
                    max_convergence_steps,
                    actual_shift * 100,
                    config.interior_struct.zalmoxis.mesh_max_shift * 100,
                )
        else:
            n_conv = 0

        dirs['mesh_shift_active'] = still_converging
        dirs['mesh_convergence_steps'] = n_conv

        # Update .prev for next iteration. Skip the copy when src and dst
        # already point to the same file, which happens on the
        # Zalmoxis-failure fallback path: the fallback sets
        # `spider_mesh_file = prev_path`, so the usual "save current to
        # .prev" becomes a self copy and shutil.copy2 raises
        # SameFileError. The abspath guard below skips the copy in that
        # case.
        if not prev_path:
            prev_path = spider_mesh_file + '.prev'
            dirs['spider_mesh_prev'] = prev_path
        if os.path.abspath(spider_mesh_file) != os.path.abspath(prev_path):
            shutil.copy2(spider_mesh_file, prev_path)

        # Remap entropy in the latest SPIDER JSON to match the new mesh.
        # Without this, the old dS/dxi applied on the new xi grid produces
        # incorrect absolute entropy, causing CVode failures at high mass.
        if config.interior_energetics.module == 'spider':
            from proteus.interior_energetics.spider import (
                get_all_output_times,
                remap_entropy_for_new_mesh,
            )

            try:
                sim_times = get_all_output_times(dirs['output'])
            except Exception as exc:
                log.warning(
                    'Could not retrieve SPIDER output times from %s; '
                    'skipping entropy remap: %s',
                    dirs['output'],
                    exc,
                )
                sim_times = []
            if len(sim_times) > 0:
                latest_json = os.path.join(dirs['output'], 'data', '%.0f.json' % sim_times[-1])
                # When global_miscibility is enabled, SPIDER's domain
                # extends to R_solvus, not R_int. Use the appropriate
                # radius for entropy remapping.
                if config.interior_struct.zalmoxis.global_miscibility and 'R_solvus' in hf_row:
                    remap_radius = hf_row['R_solvus']
                else:
                    remap_radius = hf_row['R_int']
                remap_entropy_for_new_mesh(
                    json_path=latest_json,
                    new_mesh_file=spider_mesh_file,
                    radius_phys=remap_radius,
                )
    else:
        # No mesh file produced: reset convergence state
        dirs['mesh_shift_active'] = False
        dirs['mesh_convergence_steps'] = 0

    # Clean up temporary arrays
    del r_stag, r_ascending, T_ascending
    gc.collect()

    # Regenerate SPIDER-format P-S EOS tables when composition changed
    # substantially. For dry 1 M_Earth CHILI this never fires: pure
    # MgSiO3 is a planet-state-invariant material EOS, so the pre-built
    # tables are stable for the entire evolution. The comp_changed path
    # is reached in wet runs where binodal redistribution or degassing
    # shifts mantle volatile fractions by > 5% (SPIDER reads the fresh
    # file on next call; Aragog's in-memory EntropyEOS, built once during
    # AragogRunner.setup_solver, is NOT invalidated here, so Aragog would
    # silently use the stale in-memory tables).
    #
    # KNOWN GAP: for Aragog + wet runs we would need to (i) reload
    # EntropyEOS from the regenerated files, (ii) re-install the JAX
    # CVODE factory so its captured eos_jax pytree matches the new
    # tables, (iii) bounds-check the cached _last_entropy against the
    # new [S_min, S_max] range. Dry runs do not need this; it is a
    # precondition for quantitative wet-run work.
    if comp_changed and config.interior_energetics.module in ('spider', 'aragog'):
        from proteus.interior_struct.zalmoxis import generate_spider_tables

        spider_tables = generate_spider_tables(config, outdir)
        if spider_tables is not None:
            dirs['spider_eos_dir'] = spider_tables['eos_dir']
            dirs['spider_solidus_ps'] = spider_tables['solidus_path']
            dirs['spider_liquidus_ps'] = spider_tables['liquidus_path']
            log.info('Regenerated SPIDER EOS tables (composition change)')
            if config.interior_energetics.module == 'aragog':
                log.warning(
                    'Aragog: regenerated P-S tables on composition change, '
                    'but Aragog in-memory EntropyEOS is not refreshed. '
                    'Known gap for wet runs. Dry runs are not affected.'
                )

    # Update composition sentinels for next trigger check
    M_mantle = float(hf_row.get('M_mantle', 0.0))
    if M_mantle > 0:
        for species in ('H2O', 'H2'):
            dirs[f'_last_w_{species}_liquid'] = (
                float(hf_row.get(f'{species}_kg_liquid', 0.0)) / M_mantle
            )

    log.info(
        'Structure updated: R_int=%.3e m, gravity=%.3f m/s^2',
        hf_row['R_int'],
        hf_row['gravity'],
    )
    return (current_time, float(hf_row['T_magma']), float(hf_row['Phi_global']))


def get_all_output_times(output_dir: str, model: str):
    if model == 'spider':
        from proteus.interior_energetics.spider import get_all_output_times as _get_output_times
    elif model == 'aragog':
        from proteus.interior_energetics.aragog import get_all_output_times as _get_output_times
    else:
        return []

    return _get_output_times(output_dir)


def read_interior_data(output_dir: str, model: str, times: list):
    if len(times) == 0:
        return []

    if model == 'spider':
        from proteus.interior_energetics.spider import read_jsons

        return read_jsons(output_dir, times)

    elif model == 'aragog':
        from proteus.interior_energetics.aragog import read_ncdfs

        return read_ncdfs(output_dir, times)

    else:
        return []
