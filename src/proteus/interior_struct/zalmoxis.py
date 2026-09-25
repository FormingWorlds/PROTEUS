# Zalmoxis interior module
from __future__ import annotations

import functools
import hashlib
import logging
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from zalmoxis.mixing import _PALEOS_UNIFIED_NAMES
from zalmoxis.solver import main

from proteus.config import Config
from proteus.data import (
    EOS_CHABRIER_2021,
    EOS_PALEOS_H2O,
    EOS_PALEOS_IRON,
    EOS_PALEOS_MGSIO3,
    EOS_PALEOS_MGSIO3_UNIFIED,
    EOS_RTPRESS_100TPA,
    EOS_SEAGER_2007,
    EOS_WOLF_BOWER_2018,
    dataset_dir,
)
from proteus.interior_struct.common import solvus_radius
from proteus.utils.constants import (
    FEI2021_LIQUIDUS_P_CALIB_PA,
    PALEOS_EOS_PREFIXES,
    PALEOS_REGISTRY_KEYS,
    TDEP_EOS_PREFIXES,
    VOLATILE_EOS_MAP,
    M_earth,
    R_earth,
    element_list,
)
from proteus.utils.data import RELOCATE_HINT, GetFWLData, get_zalmoxis_melting_curves
from proteus.utils.helper import (
    _strip_fraction_tokens,
    energetics_eos_key,
    eos_components,
    generates_paleos_tables,
    is_mgsio3,
    paleos_companion_keys,
    twophase_registry_key,
)

# Set up logging
log = logging.getLogger('fwl.' + __name__)

# --- liquidus_super super-liquidus adiabat solver tunables ----------------
# The solver searches surface temperature for the coolest adiabat that clears
# the configured liquidus by delta_T_super at its most-constraining depth. The
# search window is anchored to the configured liquidus (not a fixed Kelvin
# band), so it adapts to whatever melting curve is in use.
_SUPERLIQ_N_POINTS = 200  # adiabat sampling for the binding-depth search
_SUPERLIQ_SCAN_SPAN_K = 4000.0  # coarse-scan span above T_liq(P_surf)
_SUPERLIQ_SCAN_STEPS = 20  # coarse-scan point count over that span
_SUPERLIQ_MAX_EXTENSIONS = 3  # doubling-step extensions past a still-valid scan top
_SUPERLIQ_REFINE_LEVELS = 2  # spacing halvings when no coarse-scan point is valid
_SUPERLIQ_N_BISECT = 12  # delta-crossing bisection iterations (sub-Kelvin final)
_SUPERLIQ_N_CEILING_BISECT = 12  # last-valid/first-invalid bisection iterations
_SUPERLIQ_MAX_S_DRIFT = 1.0e-3  # max fractional entropy drift for an in-table adiabat
_SUPERLIQ_DEFAULT_MUSHY = 0.8  # fallback solidus = factor * liquidus

# Entries kept in each super-liquidus memo before the least recently used goes.
_SUPERLIQ_CACHE_MAXSIZE = 64


class _LRUDict(OrderedDict):
    """Mapping that keeps at most ``maxsize`` entries.

    Reads and writes mark an entry as most recently used; an insert beyond
    ``maxsize`` drops the least recently used entry, so a long-lived process
    (grid driver, notebook) holds a bounded number of solves.

    Parameters
    ----------
    maxsize : int
        Maximum number of entries, at least 1.
    """

    def __init__(self, maxsize: int = _SUPERLIQ_CACHE_MAXSIZE):
        if maxsize < 1:
            raise ValueError(f'maxsize must be >= 1, got {maxsize}')
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.maxsize:
            self.popitem(last=False)

    def copy(self):
        """Return a shallow copy with the same entries, order and ``maxsize``."""
        new = type(self)(maxsize=self.maxsize)
        for key, value in self.items():
            OrderedDict.__setitem__(new, key, value)
        return new


# Per-process memo so the three IC call sites (structure solve, energetics
# entropy IC, Aragog cross-check) share one solve for a given input instead of
# repeating the ~30-probe search. Keyed on the physical inputs only, so it is
# deterministic; tests clear it between cases (see _clear_superliquidus_cache).
_SUPERLIQ_CACHE: _LRUDict = _LRUDict()
# Traceback-free copies of anchor failures that depend only on the key
# (InitialConditionError and wrapped numerical errors), keyed like _SUPERLIQ_CACHE.
_SUPERLIQ_FAILED: _LRUDict = _LRUDict()

# CMB temperature [K] of the most recently solved super-liquidus adiabat. A
# structure solve driven by an external temperature source discards this anchor
# (the external T(r) is the temperature source), so those calls reuse the value
# the internal-dispatch initial-condition solve already produced instead of
# repeating the scan-and-bisection at a drifted P_cmb. None until the first
# solve completes.
_SUPERLIQ_LAST_ANCHOR: float | None = None
# (delta_T_super, mantle_eos) of the solve that set _SUPERLIQ_LAST_ANCHOR.
_SUPERLIQ_LAST_ANCHOR_FOR: tuple | None = None

# Set once a run has reported that the Zalmoxis JAX structure path is not
# viable for the configured EOS, so the numpy-fallback provenance is logged a
# single time instead of on every re-solve.
_JAX_NONVIABLE_LOGGED: bool = False


def _clear_superliquidus_cache() -> None:
    """Drop the cached super-liquidus solves (used by tests to avoid leakage).

    Also clears ``common._ANCHOR_CAP_WARNED``, the anchor-cap warnings that
    share the lifetime of these solves.
    """
    global _SUPERLIQ_LAST_ANCHOR, _SUPERLIQ_LAST_ANCHOR_FOR, _JAX_NONVIABLE_LOGGED
    from proteus.interior_energetics.common import _ANCHOR_CAP_WARNED

    _SUPERLIQ_CACHE.clear()
    _SUPERLIQ_FAILED.clear()
    _ANCHOR_CAP_WARNED.clear()
    _SUPERLIQ_LAST_ANCHOR = None
    _SUPERLIQ_LAST_ANCHOR_FOR = None
    _JAX_NONVIABLE_LOGGED = False


def _log_jax_nonviable_once(core_eos: str, mantle_eos: str) -> bool:
    """Report the JAX to numpy structure fallback once per run.

    The Zalmoxis logger is not attached to the run's file handlers, so a
    JAX-path fallback inside Zalmoxis reaches stderr but never the run log.
    This surfaces the fallback from the PROTEUS side, where the fwl logger is
    wired, so the run log records why structure solves used the numpy ODE path.
    The message is emitted only on the first non-viable solve; later solves in
    the same run stay silent to keep the log readable.

    Parameters
    ----------
    core_eos : str
        Configured Zalmoxis core equation of state.
    mantle_eos : str
        Resolved Zalmoxis mantle equation of state.

    Returns
    -------
    bool
        True if this call emitted the message, False if it was already logged.
    """
    global _JAX_NONVIABLE_LOGGED
    if _JAX_NONVIABLE_LOGGED:
        return False
    log.info(
        'Zalmoxis JAX structure path is not viable for the configured EOS '
        '(core=%s, mantle=%s); structure solves use the numpy ODE path.',
        core_eos,
        mantle_eos,
    )
    _JAX_NONVIABLE_LOGGED = True
    return True


# Module-level cache for density seeding between Zalmoxis calls.
# Stores the last successful density profile so the next call can
# use it as a starting point for the Picard iteration. The cache only
# seeds the iterative solver; it never changes the converged result.
# `key` records which planet the seed belongs to so an in-process
# multi-planet driver does not seed one planet from another.
_density_cache = {'density': None, 'radii': None, 'key': None}


def _structure_cache_key(config):
    """Signature identifying the planet that owns a cached density seed.

    Parameters
    ----------
    config : Config
        Active configuration object.

    Returns
    -------
    tuple
        Structural determinants of the interior solve. Built from the
        total planet mass and core/mantle fractions, all of which are
        fixed for a given planet across a trajectory, so the seed is
        reused within one planet's evolution but never shared between
        two different planets solved in the same process.
    """
    return (
        config.planet.mass_tot,
        config.interior_struct.core_frac,
        config.interior_struct.zalmoxis.mantle_mass_fraction,
    )


def get_zalmoxis_output_filepath(outdir: str):
    """Returns the output file path for Zalmoxis data.
    Args:
        outdir (str): Output directory.
    Returns:
        str: Path to the output file.
    """
    return os.path.join(outdir, 'data', 'zalmoxis_output.dat')


def validate_zalmoxis_output_schema(
    output_path: str,
    hf_row: dict,
    rtol_radius: float = 1e-6,
    rtol_mass: float = 5e-2,
    mantle_mass_ref: float | None = None,
) -> None:
    """Verify zalmoxis_output.dat is consistent with hf_row scalars.

    The file is the contract Aragog reads inside ``solver.reset()``
    (eos_method=2). This check confirms the file's last r matches
    ``hf_row['R_int']``, and that the mantle mass matches
    ``hf_row['M_int'] - hf_row['M_core']``.
    Catches file I/O corruption, column-order mistakes, truncation,
    encoding drift, and Aragog/Zalmoxis schema desync at the file
    handover boundary.

    The mantle mass is taken from ``mantle_mass_ref`` when the caller supplies
    it: the structure's accumulator total ``mass_enclosed[-1]`` minus the exact
    core-mass target ``cmb_mass``. Re-integrating the coarse output nodes with a
    grid trapezoid instead diverges from the sub-grid-substepped RK45 integral
    across the steep interior density profile, reaching ~10% at high planet mass,
    which would false-reject a structure that actually conserves mass. With the
    reference supplied, part (b) reduces to a check that the core-mantle split
    point is well resolved: the residual it measures is the CMB-node snap
    overshoot ``mass_enclosed[cmb_index] - cmb_mass`` relative to the mantle
    mass, a fraction of one boundary shell. File-density-column corruption is
    caught downstream by Aragog's EOS-vs-mesh consistency at ``solver.reset()``;
    the radius check (part (a)) catches truncation and column swaps directly.
    When no reference is supplied the check falls back to the grid-trapezoidal
    shell-sum.

    Parameters
    ----------
    output_path : str
        Path to the just-written ``zalmoxis_output.dat``.
    hf_row : dict
        PROTEUS hf_row holding the scalar truth (R_int, M_int, M_core).
    rtol_radius : float
        Relative tolerance for the top-of-mantle vs. R_int check.
        Default 1e-6: the file's last r and ``hf_row['R_int']`` come
        from the same variable in zalmoxis_solver, so equality is
        exact modulo float-string round-trip noise. Tight tolerance
        catches truncation, last-line corruption, and column-swap
        bugs at the bit level.
    rtol_mass : float
        Relative tolerance for the integrated mantle mass vs.
        ``M_int - M_core``. Default 5e-2 (5%) reflects two stacked
        sources of legitimate mismatch:
        (a) integrator-method difference: Zalmoxis' ``mass_enclosed``
        is the ODE state from RK45 with sub-grid substepping, while
        the schema check re-integrates via a grid-trapezoidal
        shell-sum. On stiff CHILI density profiles this shows
        about 0.8 to 2.0 %.
        (b) ``blend_mesh_files`` post-write modifies the file in
        place but does not update hf_row scalars. When blending fires
        with alpha < 1 (capping large R-shifts), the file's integrated
        mass can drift up to about 5% from the unblended
        hf_row['M_int'], so this check has to tolerate it.
        5e-2 keeps a >2x margin over the worst legitimate noise
        while still catching gross corruption (column swap,
        truncation, byte-flip) at >>5 %. **The tight mass-conservation
        contract (<0.1 %) lives in the wrapper-level mass-anchor
        check on hf_row['M_int'] / hf_row['M_int_target']**, not here.

    Raises
    ------
    RuntimeError
        On any violation (file unreadable, wrong shape, radius
        mismatch, mass mismatch). Caller (the wrapper's except-block)
        catches this and routes through the fall-back path.
    """
    try:
        data = np.loadtxt(output_path)
    except Exception as exc:
        raise RuntimeError(
            'zalmoxis_output.dat schema violation: could not reload '
            f'the just-written file ({output_path}): {exc}'
        )
    if data.size == 0 or data.ndim != 2 or data.shape[1] != 5:
        raise RuntimeError(
            'zalmoxis_output.dat schema violation: unexpected shape '
            f'{data.shape if data.size else "empty"} '
            f'(expected (N, 5)) for {output_path}'
        )

    r_file = data[:, 0]
    rho_file = data[:, 2]

    # (a) top-of-mantle radius == hf_row['R_int']
    r_top = float(r_file[-1])
    r_int_hf = float(hf_row.get('R_int', 0.0))
    if r_int_hf > 0:
        r_rel = abs(r_top / r_int_hf - 1.0)
        if r_rel > rtol_radius:
            raise RuntimeError(
                'zalmoxis_output.dat schema violation: top-of-mantle '
                f'r={r_top:.6e} from file differs from '
                f'hf_row[R_int]={r_int_hf:.6e} '
                f'(rel={r_rel:.3e} > {rtol_radius:.1e})'
            )

    # (b) mantle integrated mass == hf_row['M_int'] - hf_row['M_core'].
    # Prefer the structure's own RK45 accumulator mantle mass when the
    # caller supplies it (mantle_mass_ref). That value is the sub-grid
    # substepped ODE integral, exact to the solver tolerance. Re-integrating
    # the coarse output nodes with a grid trapezoid instead diverges from it
    # across the steep core-mantle density jump (the CMB-node snap attributes
    # a whole boundary shell to one side), reaching ~10% at high planet mass,
    # which would false-reject a structure that actually conserves mass. Only
    # when no reference is available do we fall back to the grid-trapezoidal
    # shell-sum (4/3 pi (r2^3 - r1^3) * rho_avg).
    if mantle_mass_ref is not None:
        mantle_mass = float(mantle_mass_ref)
        mass_source = 'accumulator'
    else:
        shells = (
            (4.0 / 3.0)
            * np.pi
            * (r_file[1:] ** 3 - r_file[:-1] ** 3)
            * 0.5
            * (rho_file[1:] + rho_file[:-1])
        )
        mantle_mass = float(np.sum(shells))
        mass_source = 'trapezoid'
    M_int_hf = float(hf_row.get('M_int', 0.0))
    M_core_hf = float(hf_row.get('M_core', 0.0))
    expected_mantle = M_int_hf - M_core_hf
    if expected_mantle > 0:
        m_rel = abs(mantle_mass / expected_mantle - 1.0)
        log.debug(
            'zalmoxis mass-closure: mantle split m_rel=%.3e (%s, rtol_mass=%.1e)',
            m_rel,
            mass_source,
            rtol_mass,
        )
        if m_rel > rtol_mass:
            raise RuntimeError(
                'zalmoxis_output.dat schema violation: '
                f'mantle mass ({mass_source})={mantle_mass:.6e} kg '
                'differs from hf_row[M_int - M_core]='
                f'{expected_mantle:.6e} kg '
                f'(rel={m_rel:.3e} > {rtol_mass:.1e})'
            )


def build_volatile_profile(hf_row: dict, mantle_eos: str):
    """Build a VolatileProfile from helpfile volatile masses.

    Computes per-phase (liquid/solid) mass fractions for dissolved volatiles
    that have Zalmoxis EOS tables. Returns None if no volatiles are dissolved
    or if the mantle liquid/solid masses are unavailable.

    Ownership of the liquid/solid split: the coupled PROTEUS path takes
    the per-phase masses from the outgassing chemistry (CALLIOPE/Aragog
    equilibrium via the helpfile), which already solved solubility.
    Zalmoxis's own ``partition_rule`` hook is a structure-side
    idealization for standalone Zalmoxis runs with no chemistry coupled;
    it is deliberately not used here.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row with volatile mass keys (e.g. ``H2O_kg_liquid``,
        ``H2O_kg_solid``) and mantle mass keys: ``M_mantle`` with
        ``Phi_global`` (preferred, structural), or ``M_mantle_liquid`` and
        ``M_mantle_solid`` as a fallback.
    mantle_eos : str
        Primary mantle EOS identifier (e.g. ``'PALEOS:MgSiO3'``).

    Returns
    -------
    VolatileProfile or None
        Profile with per-phase fractions, or None if not applicable.
    """
    from zalmoxis.mixing import VolatileProfile

    # Denominators: the melt concentration must be taken against the same
    # reservoir the chemistry used and the structure carries. CALLIOPE
    # dissolves into M_mantle * Phi_global, and the structure integrates
    # the (structural) mantle mass, so split the structural M_mantle by
    # Phi_global. Aragog's M_mantle_liquid/M_mantle_solid use its PALEOS
    # EOS cell masses instead, ~7% denser than the wet structural mantle,
    # which diluted w_liquid and lost the same fraction of dissolved
    # water from the structure. Fall back to the per-phase masses when
    # the structural fields are absent (e.g. minimal test rows).
    M_mantle = float(hf_row.get('M_mantle', 0.0))
    Phi_global = float(hf_row.get('Phi_global', float('nan')))
    if M_mantle > 0 and np.isfinite(Phi_global):
        Phi_global = min(max(Phi_global, 0.0), 1.0)
        M_liq = Phi_global * M_mantle
        M_sol = (1.0 - Phi_global) * M_mantle
    else:
        M_liq = float(hf_row.get('M_mantle_liquid', 0.0))
        M_sol = float(hf_row.get('M_mantle_solid', 0.0))

    # Need mantle mass data to compute fractions
    if M_liq + M_sol <= 0:
        return None

    w_liquid = {}
    w_solid = {}
    has_nonzero = False

    for species, eos_name in VOLATILE_EOS_MAP.items():
        kg_liq = float(hf_row.get(f'{species}_kg_liquid', 0.0))
        kg_sol = float(hf_row.get(f'{species}_kg_solid', 0.0))

        # Only include species with meaningful dissolved mass
        if kg_liq + kg_sol <= 0:
            continue

        # Mass fraction in liquid phase
        w_l = kg_liq / M_liq if M_liq > 0 else 0.0
        # Mass fraction in solid phase
        w_s = kg_sol / M_sol if M_sol > 0 else 0.0

        w_liquid[eos_name] = w_l
        w_solid[eos_name] = w_s
        has_nonzero = True

    if not has_nonzero:
        return None

    # Normalize: total volatile fraction in each phase must not exceed 0.95
    # (at least 5% silicate). Clamp proportionally if sum exceeds the limit.
    for w_dict in (w_liquid, w_solid):
        total = sum(w_dict.values())
        max_volatile_frac = 0.95
        if total > max_volatile_frac:
            scale = max_volatile_frac / total
            for k in w_dict:
                w_dict[k] *= scale

    log.info(
        'Built VolatileProfile: liquid=%s, solid=%s',
        {k: f'{v:.4f}' for k, v in w_liquid.items()},
        {k: f'{v:.4f}' for k, v in w_solid.items()},
    )

    return VolatileProfile(
        w_liquid=w_liquid,
        w_solid=w_solid,
        primary_component=mantle_eos,
    )


def extend_mantle_eos_with_volatiles(mantle_eos: str, volatile_profile) -> str:
    """Extend a single-component mantle EOS string with volatile components.

    If a VolatileProfile is provided and the mantle EOS is a single component,
    this adds the volatile EOS components with small placeholder fractions.
    The actual fractions are overridden at each ODE step by the VolatileProfile.

    Parameters
    ----------
    mantle_eos : str
        Base mantle EOS string (e.g. ``'PALEOS:MgSiO3'``).
    volatile_profile : VolatileProfile or None
        Profile containing volatile EOS component names.

    Returns
    -------
    str
        Extended EOS string (e.g.
        ``'PALEOS:MgSiO3:0.98+PALEOS:H2O:0.01+Chabrier:H:0.01'``),
        or the original string if no extension needed.
    """
    if volatile_profile is None:
        return mantle_eos

    # Don't modify if already multi-component
    if '+' in mantle_eos:
        return mantle_eos

    # Collect all volatile EOS components from the profile
    all_vol_components = set()
    for d in (volatile_profile.w_liquid, volatile_profile.w_solid):
        all_vol_components.update(d.keys())

    if not all_vol_components:
        return mantle_eos

    # Build extended string with small placeholder fractions
    # (actual fractions set by VolatileProfile at each radius)
    n_vol = len(all_vol_components)
    placeholder = 0.01  # 1% each
    primary_frac = max(0.5, 1.0 - n_vol * placeholder)
    parts = [f'{mantle_eos}:{primary_frac:.4f}']
    for comp in sorted(all_vol_components):
        parts.append(f'{comp}:{placeholder:.4f}')

    extended = '+'.join(parts)
    log.info('Extended mantle EOS: %s -> %s', mantle_eos, extended)
    return extended


def _get_target_surface_pressure(config: Config, hf_row: dict) -> float:
    """Determine the surface pressure boundary condition for Zalmoxis.

    Parameters
    ----------
    config : Config
        PROTEUS configuration object.
    hf_row : dict
        Current helpfile row.

    Returns
    -------
    float
        Target surface pressure in Pa.
    """
    # After outgassing has run, use the atmospheric surface pressure
    p_surf_bar = hf_row.get('P_surf', 0)
    if np.isfinite(p_surf_bar) and p_surf_bar > 0:
        return p_surf_bar * 1e5  # bar -> Pa

    # First call, before outgassing. Estimate from initial volatile
    # partial pressures specified in the config.
    _SPECIES = ('H2O', 'CO2', 'N2', 'S2', 'SO2', 'H2S', 'NH3', 'H2', 'CH4', 'CO')
    try:
        gas_prs = config.planet.gas_prs
        p_init_bar = sum(float(getattr(gas_prs, s, 0)) for s in _SPECIES)
        p_init_pa = p_init_bar * 1e5  # bar -> Pa
    except (TypeError, ValueError, AttributeError):
        p_init_pa = 0.0

    # Floor at 1 atm (bare rock), ceiling at 1 GPa
    return max(101325.0, min(p_init_pa, 1e9))


def _resolve_zalmoxis_temperature_mode(mode: str) -> str:
    """Map a PROTEUS temperature_mode to the Zalmoxis structure-solve mode.

    PROTEUS supports more IC modes than Zalmoxis needs to know about for
    its (M-R via hydrostatic + EOS) structure solve. The mapping here
    decouples PROTEUS-side IC bookkeeping from the Zalmoxis-side T(r)
    integration:

    - 'accretion', 'isentropic'      -> 'adiabatic'   (surface-anchored)
    - 'liquidus_super'               -> 'adiabatic_from_cmb'
    - all other modes pass through unchanged.

    The 'liquidus_super' mapping pairs with
    :func:`_resolve_zalmoxis_cmb_temperature`, which supplies the CMB
    temperature of the solved super-liquidus adiabat as the anchor for
    Zalmoxis's upward integration.
    """
    if mode in ('accretion', 'isentropic'):
        return 'adiabatic'
    if mode == 'liquidus_super':
        return 'adiabatic_from_cmb'
    return mode


def _resolve_zalmoxis_cmb_temperature(
    config: Config,
    hf_row: dict,
    mode: str,
    external_temperature_source: bool = False,
) -> float:
    """Resolve cmb_temperature for the Zalmoxis structure call.

    For 'liquidus_super', returns the CMB temperature of the solved
    super-liquidus adiabat (see :func:`solve_superliquidus_adiabat`), using
    hf_row['P_cmb'] when populated or a Noack & Lasbleis (2020) mass-aware
    P_cmb estimate on the very first call; the next structure iteration
    recomputes it against the converged Zalmoxis P_cmb. The energetics IC is
    solved separately on the P-S tables (compute_initial_entropy).

    When ``external_temperature_source`` is set the structure solve is driven
    by an evolved T(r) profile (or the super-liquidus adiabat callable during
    the IC re-solve) rather than the internal temperature-mode dispatch, so the
    super-liquidus CMB anchor is not the temperature source for this call and
    the solved value is discarded. Those calls reuse the anchor the
    internal-dispatch IC solve already produced and skip the scan-and-bisection;
    when no anchor was solved for the same ``delta_T_super`` and mantle EOS they
    fall back to ``config.planet.tcmb_init``. This
    avoids re-solving (and possibly raising the unreachable-superheat error) on
    every evolution re-solve over a value nothing consumes.

    With a PALEOS mantle and spider or aragog energetics, if the anchor
    fails at this P_cmb (``InitialConditionError``, which includes numerical
    integration failures), the last solved anchor for the same
    ``delta_T_super`` and mantle EOS, or else ``config.planet.tcmb_init``,
    is used with a warning. The initial entropy then re-solves the anchor at
    the P_cmb of the structure it gets, which after this fallback is the
    fallback structure, and decides whether a molten state exists; it is
    solved before the equilibration loop and again after it. Otherwise the
    error propagates, since no later step re-solves the anchor. A resumed
    run that restores the entropy snapshot re-solves neither the structure
    anchor nor the initial entropy; an Aragog resume that falls back to a
    fresh initial entropy solves it again at the current P_cmb.

    For all other modes, returns config.planet.tcmb_init verbatim.
    """
    if mode != 'liquidus_super':
        return config.planet.tcmb_init

    if external_temperature_source:
        anchor = _SUPERLIQ_LAST_ANCHOR
        # Reuse the anchor only for the superheat and mantle EOS it was solved for.
        if anchor is not None and _SUPERLIQ_LAST_ANCHOR_FOR == _superliq_anchor_for(config):
            log.debug(
                'liquidus_super: structure solve uses an external temperature '
                'source; reusing the last solved CMB anchor T_cmb=%.0f K and '
                'skipping the super-liquidus re-solve.',
                anchor,
            )
            return float(anchor)
        log.debug(
            'liquidus_super: structure solve uses an external temperature '
            'source with no super-liquidus solve for this superheat and mantle '
            'EOS; using tcmb_init=%.0f K as the unconsumed CMB anchor.',
            float(config.planet.tcmb_init),
        )
        return float(config.planet.tcmb_init)

    # Anchor the Zalmoxis structure-solve adiabat at the CMB temperature of the
    # P-T super-liquidus adiabat. The energetics IC is solved on the P-S
    # tables, so its adiabat differs from this anchor by the P-T vs P-S
    # liquidus offset (tens of K at 1 M_Earth).
    from proteus.interior_energetics.common import InitialConditionError

    try:
        res = solve_superliquidus_adiabat(config, hf_row)
    except InitialConditionError as exc:
        if not _anchor_failure_deferred(config):
            raise
        from proteus.utils.structure_estimate import resolve_P_cmb

        P_cmb, estimated = resolve_P_cmb(hf_row, config)
        # The last anchor stands in only for the same superheat and mantle EOS.
        fallback = _SUPERLIQ_LAST_ANCHOR
        source = 'the last solved anchor'
        if fallback is None or _SUPERLIQ_LAST_ANCHOR_FOR != _superliq_anchor_for(config):
            fallback, source = float(config.planet.tcmb_init), 'tcmb_init'
        log.warning(
            'liquidus_super CMB anchor for Zalmoxis: no P-T anchor at %sP_cmb=%.0f GPa '
            '(%s); using %s, T_cmb=%.0f K, for this structure solve.',
            'the estimated ' if estimated else '',
            P_cmb / 1e9,
            exc,
            source,
            float(fallback),
        )
        return float(fallback)
    log.info(
        'liquidus_super CMB anchor for Zalmoxis: T_cmb=%.0f K (fully molten, '
        '%.0f K above the liquidus; surface T=%.0f K, P_cmb=%.0f GPa).',
        res['cmb_T'],
        res['achieved_superheat'],
        res['surface_T'],
        res['P_cmb'] / 1e9,
    )
    return float(res['cmb_T'])


def _anchor_failure_deferred(config: Config) -> bool:
    """Whether a failed P-T anchor in a structure solve can fall back.

    ``compute_initial_entropy`` re-solves the anchor at the converged P_cmb only
    for a generated PALEOS table set under spider or aragog energetics; everywhere else no
    later step does, so the failure must propagate.

    Parameters
    ----------
    config : Config
        PROTEUS configuration.

    Returns
    -------
    bool
        True when the initial entropy re-solves the anchor.
    """
    from proteus.utils.helper import generates_paleos_tables

    return generates_paleos_tables(config.interior_struct) and (
        config.interior_energetics.module in ('spider', 'aragog')
    )


def _superliq_anchor_for(config: Config) -> tuple:
    """The (delta_T_super, mantle_eos) an anchor was solved for."""
    return (
        round(float(config.planet.delta_T_super), 3),
        str(getattr(config.interior_struct.zalmoxis, 'mantle_eos', None)),
    )


def _superliq_cache_key(config: Config, P_cmb: float) -> tuple:
    """Memo key of the super-liquidus anchor solve at ``P_cmb`` [Pa]."""
    return (round(P_cmb / 1e6), *_superliq_anchor_for(config))


def solve_superliquidus_adiabat(config: Config, hf_row: dict | None) -> dict:
    """Memoised super-liquidus anchor solve, see ``_solve_superliquidus_adiabat``.

    A failure of the integration with ``ValueError``, ``KeyError``,
    ``IndexError``, ``ArithmeticError`` or ``RuntimeError``
    (``common.ANCHOR_NUMERICAL_ERRORS``) is raised as a chained
    ``InitialConditionError`` that names the type and P_cmb.
    ``NotImplementedError``, ``RecursionError`` and ``UnicodeError``
    (``common.ANCHOR_PASSTHROUGH_ERRORS``), and every other type such as
    ``TypeError``, ``AttributeError``, ``OSError`` or ``MemoryError``,
    propagate unchanged and are not memoised. An ``InitialConditionError``
    from the solve is raised as is. Both are memoised under the same key as
    a success, as a copy without traceback, so repeated structure solves at
    one P_cmb do not repeat a failing scan.

    Parameters
    ----------
    config : Config
        PROTEUS configuration.
    hf_row : dict or None
        Helpfile row; ``hf_row['P_cmb']`` is used when populated.

    Returns
    -------
    dict
        See ``_solve_superliquidus_adiabat``.

    Raises
    ------
    InitialConditionError
        See ``_solve_superliquidus_adiabat``; also for a numerical failure of
        the integration (chained). A memoised failure is raised again with
        the same message, chained to the stored copy.
    FileNotFoundError
        If an EOS or melting-curve file is missing.
    """
    from proteus.interior_energetics.common import (
        ANCHOR_NUMERICAL_ERRORS,
        ANCHOR_PASSTHROUGH_ERRORS,
        InitialConditionError,
    )
    from proteus.utils.structure_estimate import resolve_P_cmb

    P_cmb = resolve_P_cmb(hf_row, config)[0]
    key = _superliq_cache_key(config, P_cmb)
    if key in _SUPERLIQ_FAILED:
        first = _SUPERLIQ_FAILED[key]
        raise InitialConditionError(str(first)) from first
    try:
        return _solve_superliquidus_adiabat(config, hf_row)
    except InitialConditionError as exc:
        _SUPERLIQ_FAILED[key] = InitialConditionError(str(exc))
        raise
    except (ZalmoxisMissingEOSFilesError, *ANCHOR_PASSTHROUGH_ERRORS):
        raise
    except ANCHOR_NUMERICAL_ERRORS as exc:
        msg = (
            f'liquidus_super: the PALEOS P-T anchor failed at P_cmb={P_cmb / 1e9:.0f} GPa '
            f'({type(exc).__name__}: {exc}).'
        )
        _SUPERLIQ_FAILED[key] = InitialConditionError(msg)
        raise InitialConditionError(msg) from exc


def _solve_superliquidus_adiabat(config: Config, hf_row: dict | None) -> dict:
    """Solve for the coolest fully molten adiabat with a controlled superheat.

    The ``liquidus_super`` initial condition starts the mantle on a single
    adiabat (uniform specific entropy) that lies a minimum of
    ``config.planet.delta_T_super`` Kelvin above the configured liquidus,
    evaluated at the most-constraining mantle depth. Solving for the surface
    temperature that achieves this margin fixes the whole isentropic profile.

    The superheat is checked against whatever solidus/liquidus parameterisation
    is configured, and the surface-temperature search window is anchored to the
    surface liquidus, so the initial condition adapts to the melting curve in
    use rather than relying on a fixed surface temperature or entropy value.

    When the binding (minimum-superheat) depth is shallow, as it is for the
    PALEOS MgSiO3 liquidus, the solved entropy is essentially independent of
    planet mass, which keeps a mass grid on a common initial adiabat. That is a
    property of the adiabat-vs-liquidus slope ordering, not a guarantee: for a
    steeper liquidus the binding can migrate toward the core-mantle boundary, in
    which case the solve logs a warning (the surface anchor is then weakly
    constrained and approaches a CMB-liquidus anchor in the melting curve's
    extrapolated regime).

    Parameters
    ----------
    config : Config
        PROTEUS configuration. Uses ``planet.delta_T_super`` (required superheat
        in K), ``planet.mass_tot``, ``interior_struct.core_frac`` and
        ``interior_struct.zalmoxis.mantle_eos``.
    hf_row : dict or None
        Helpfile row. ``hf_row['P_cmb']`` is used when populated; otherwise a
        Noack & Lasbleis (2020) mass-aware estimate is used on the first call.

    Returns
    -------
    dict
        ``surface_T`` [K], ``S_target`` [J/(kg K)], ``cmb_T`` [K],
        ``achieved_superheat`` [K], ``binding_P`` [Pa], ``P_cmb`` [Pa],
        ``clamped`` (True when the requested superheat was unreachable) and
        ``window_limited`` (True when the clamp comes from the search window,
        not from the EOS table).

    Raises
    ------
    InitialConditionError
        If no valid molten adiabat exists in the surface-temperature scan, at
        the midpoints of its intervals or in the extensions above it, if the
        scan is valid, then invalid, then
        valid again after its first valid point (one leading invalid band at
        the cold end is tolerated; a second transition is not, since the
        bisection below needs a single validity edge), if the solved adiabat
        leaves the EOS table, or if no fully-molten adiabat is reachable
        within the table (the hottest valid adiabat still sits below the
        liquidus). A reachable but insufficient ``delta_T_super`` does not
        raise: the solve clamps to the largest achievable superheat, reports
        it in ``achieved_superheat`` and emits a warning.
    """
    from proteus.interior_energetics.common import InitialConditionError

    try:
        from zalmoxis.eos_export import compute_entropy_adiabat
        from zalmoxis.melting_curves import paleos_liquidus
    except (ImportError, ModuleNotFoundError) as e:
        raise InitialConditionError(
            'liquidus_super mode requires Zalmoxis '
            '(zalmoxis.eos_export.compute_entropy_adiabat and '
            'zalmoxis.melting_curves.paleos_liquidus); import failed: '
            f'{e}'
        )

    delta = float(config.planet.delta_T_super)

    from proteus.utils.structure_estimate import resolve_P_cmb

    P_cmb, estimated = resolve_P_cmb(hf_row, config)
    if estimated:
        log.warning(
            'liquidus_super: hf_row["P_cmb"] not yet populated; using '
            'Noack & Lasbleis (2020) mass-aware fallback P_cmb=%.1f GPa '
            '(mass_tot=%.2f M_Earth). The structure anchor is re-derived '
            'against the converged Zalmoxis P_cmb on the next iteration.',
            P_cmb / 1e9,
            float(config.planet.mass_tot),
        )

    mantle_eos = config.interior_struct.zalmoxis.mantle_eos
    P_surface = 1e5  # 1 bar surface anchor for the adiabat
    global _SUPERLIQ_LAST_ANCHOR, _SUPERLIQ_LAST_ANCHOR_FOR
    _cache_key = _superliq_cache_key(config, P_cmb)
    if _cache_key in _SUPERLIQ_CACHE:
        cached = dict(_SUPERLIQ_CACHE[_cache_key])
        _SUPERLIQ_LAST_ANCHOR = float(cached['cmb_T'])
        _SUPERLIQ_LAST_ANCHOR_FOR = _superliq_anchor_for(config)
        return cached

    mat_dicts = load_zalmoxis_material_dictionaries()
    solid_eos, liquid_eos = resolve_2phase_mgsio3_paths(mantle_eos, mat_dicts, required=True)
    eos_file = energetics_entry(mantle_eos, mat_dicts)[1].get('eos_file', '')
    eos_file = eos_file or solid_eos or ''
    melt_funcs = load_zalmoxis_solidus_liquidus_functions(mantle_eos, config)
    if melt_funcs is not None:
        sol_func, liq_func = melt_funcs
    else:
        liq_func = paleos_liquidus

        def sol_func(P):
            return _SUPERLIQ_DEFAULT_MUSHY * np.asarray(paleos_liquidus(P))

    def _probe(T_surf: float) -> dict:
        """Adiabat from this surface T: minimum superheat over depth + validity.

        The adiabat is isentropic by construction, so ``valid`` requires a
        uniform ``S_profile``: a bracket failure (the deep adiabat exhausting
        the EOS table) plateaus the profile and breaks isentropy, which a
        non-uniform ``S_profile``, a NaN, or a cooling-with-depth segment all
        flag. This catches the plateau that a monotonicity-only test misses.
        """
        result = compute_entropy_adiabat(
            eos_file=eos_file,
            T_surface=float(T_surf),
            P_surface=P_surface,
            P_cmb=P_cmb,
            n_points=_SUPERLIQ_N_POINTS,
            solidus_func=sol_func,
            liquidus_func=liq_func,
            solid_eos_file=solid_eos,
            liquid_eos_file=liquid_eos,
        )
        P = np.asarray(result['P'], dtype=float)
        T = np.asarray(result['T'], dtype=float)
        S_target = result['S_target']
        S_prof = np.asarray(result['S_profile'], dtype=float)
        order = np.argsort(P)
        P, T = P[order], T[order]
        finite = bool(
            np.isfinite(T).all() and np.isfinite(S_target) and np.isfinite(S_prof).all()
        )
        s_drift = (
            (float(np.max(S_prof)) - float(np.min(S_prof))) / max(abs(float(S_target)), 1.0)
            if finite
            else np.inf
        )
        liq = np.asarray(liq_func(P), dtype=float)
        i = int(np.argmin(T - liq))
        # A liquidus that is undefined (NaN) at some adiabat pressure cannot
        # show the adiabat is molten there, so the adiabat is invalid.
        valid = bool(
            finite
            and s_drift < _SUPERLIQ_MAX_S_DRIFT
            and np.all(np.diff(T) > -1.0)  # no gross cooling-with-depth
            and np.isfinite(liq).all()
        )
        return {
            'superheat': float(T[i] - liq[i]),
            'binding_P': float(P[i]),
            'S_target': float(S_target),
            'cmb_T': float(T[-1]),
            'valid': valid,
        }

    def _bisect_delta(T_lo: float, T_hi: float) -> float:
        """Coolest surface T in [T_lo, T_hi] whose adiabat is valid and at least
        ``delta`` above the liquidus; an invalid midpoint counts as not satisfied.
        ``T_hi`` must already satisfy both conditions: it is returned as is if no
        midpoint does."""
        for _ in range(_SUPERLIQ_N_BISECT):
            mid = 0.5 * (T_lo + T_hi)
            dm = _probe(mid)
            if dm['valid'] and dm['superheat'] >= delta:
                T_hi = mid
            else:
                T_lo = mid
        return T_hi

    # Coarse-scan surface temperature over a fixed window above the SURFACE
    # LIQUIDUS, so the search adapts to the configured melting curve. The scan
    # stops at the first valid point that reaches delta: the crossing is then
    # bracketed, and hotter adiabats cannot change the answer. A cold-end band
    # just above the surface liquidus can be invalid (the PALEOS adiabat there
    # has a cooling-with-depth kink of a few K) while hotter adiabats are
    # valid, so one leading invalid band is tolerated.
    T_liq_surf = float(np.asarray(liq_func(P_surface)).reshape(-1)[0])
    scan_T = np.linspace(T_liq_surf, T_liq_surf + _SUPERLIQ_SCAN_SPAN_K, _SUPERLIQ_SCAN_STEPS)
    scan_step = float(scan_T[1] - scan_T[0])
    scan: list[tuple[float, dict]] = []
    for T_scan in scan_T:
        d = _probe(float(T_scan))
        scan.append((float(T_scan), d))
        if d['valid'] and d['superheat'] >= delta:
            break

    if not any(d['valid'] for _, d in scan):
        # A valid band narrower than the scan spacing, or one that starts
        # above the scan window, is missed by the coarse points: halve the
        # spacing up to _SUPERLIQ_REFINE_LEVELS times, then extend above the
        # scan top with doubling steps, stopping at the first valid adiabat.
        # A valid band narrower than the finest spacing is still missed.
        found = False
        for _ in range(_SUPERLIQ_REFINE_LEVELS):
            refined: list[tuple[float, dict]] = [scan[0]]
            for (T_a, _d_a), (T_b, d_b) in zip(scan, scan[1:]):
                T_mid = 0.5 * (T_a + T_b)
                if not found:
                    d_mid = _probe(T_mid)
                    refined.append((T_mid, d_mid))
                    found = d_mid['valid']
                refined.append((T_b, d_b))
            scan = refined
            if found:
                break
        if not found:
            T_ext, step = scan[-1][0], scan_step
            for _ in range(_SUPERLIQ_MAX_EXTENSIONS):
                T_ext += step
                d_ext = _probe(T_ext)
                scan.append((T_ext, d_ext))
                if d_ext['valid']:
                    break
                step *= 2.0

    # A second invalid->valid transition after the first valid point breaks
    # the single validity edge the bisections below rely on.
    first_valid_idx: int | None = None
    last_valid_idx: int | None = None
    seen_invalid_after_first = False
    for idx, (_, d) in enumerate(scan):
        if d['valid']:
            if seen_invalid_after_first:
                raise InitialConditionError(
                    'liquidus_super: the coarse surface-temperature scan is valid, '
                    'then invalid, then valid again at P_cmb=%.0f GPa; the '
                    'single-validity-edge assumption behind this solve does not hold '
                    'for this EOS/melting-curve combination.' % (P_cmb / 1e9)
                )
            if first_valid_idx is None:
                first_valid_idx = idx
            last_valid_idx = idx
        elif first_valid_idx is not None:
            seen_invalid_after_first = True

    points: list[tuple[float, dict]] = [(T, d) for T, d in scan if d['valid']]
    if not points:
        raise InitialConditionError(
            'liquidus_super: no valid molten adiabat found while solving the '
            f'initial condition (P_cmb={P_cmb / 1e9:.0f} GPa, surface '
            f'liquidus={T_liq_surf:.0f} K), in the scan, at the midpoints of its '
            f'intervals, or above it up to T={scan[-1][0]:.0f} K. The EOS table '
            'may not support a molten mantle at this pressure.'
        )
    for (T_a, d_a), (T_b, d_b) in zip(points, points[1:]):
        if d_b['superheat'] < d_a['superheat'] - 1.0:
            raise InitialConditionError(
                'liquidus_super: superheat decreases with increasing surface '
                f'temperature between T={T_a:.0f} K ({d_a["superheat"]:.0f} K) and '
                f'T={T_b:.0f} K ({d_b["superheat"]:.0f} K) at P_cmb={P_cmb / 1e9:.0f} '
                'GPa; the assumed monotone superheat-vs-surface-temperature relation '
                'does not hold for this EOS/melting-curve combination.'
            )

    # points is the contiguous valid run in scan (the check above rules out a
    # second one), so the point right after its last entry, if any, is the
    # first invalid point past the ceiling.
    first_invalid_T = scan[last_valid_idx + 1][0] if last_valid_idx + 1 < len(scan) else None
    window_limited = False
    if first_invalid_T is None and points[-1][1]['superheat'] < delta:
        # The valid run reaches the end of the scan: extend upward with
        # doubling steps (at most 3) until a point is invalid or reaches
        # delta, so a table ceiling just past the scan window is not mistaken
        # for the search-window limit.
        step = scan_step
        T_last, d_last = points[-1]
        for _ in range(_SUPERLIQ_MAX_EXTENSIONS):
            T_next = T_last + step
            d_next = _probe(T_next)
            if not d_next['valid']:
                first_invalid_T = T_next
                break
            if d_next['superheat'] < d_last['superheat'] - 1.0:
                raise InitialConditionError(
                    'liquidus_super: superheat decreases with increasing surface '
                    f'temperature during the search-window extension, between '
                    f'T={T_last:.0f} K ({d_last["superheat"]:.0f} K) and '
                    f'T={T_next:.0f} K ({d_next["superheat"]:.0f} K) at '
                    f'P_cmb={P_cmb / 1e9:.0f} GPa; the assumed monotone '
                    'superheat-vs-surface-temperature relation does not hold for '
                    'this EOS/melting-curve combination.'
                )
            points.append((T_next, d_next))
            T_last, d_last = T_next, d_next
            if d_next['superheat'] >= delta:
                break
            step *= 2.0
        else:
            window_limited = True

    T_best, d_best = points[-1]
    T_sub_delta = T_best  # last point known below delta, for the delta-crossing bracket below
    ceiling_bisected = False
    if d_best['superheat'] < delta and first_invalid_T is not None:
        # Bisect the validity edge between the last-valid and first-invalid
        # points to the table's true ceiling; the raw coarse/extension-grid
        # point can sit well short of it. Superheat is monotonic up to the
        # edge, so the refined edge is also the refined maximum.
        ceiling_bisected = True
        T_lo, T_hi = T_best, float(first_invalid_T)
        d_lo = d_best
        for _ in range(_SUPERLIQ_N_CEILING_BISECT):
            mid = 0.5 * (T_lo + T_hi)
            dm = _probe(mid)
            if dm['valid']:
                T_lo, d_lo = mid, dm
            else:
                T_hi = mid
        T_best, d_best = T_lo, d_lo

    reached_delta = d_best['superheat'] >= delta
    if d_best['superheat'] < 0 and not reached_delta:
        limit = 'the surface-temperature search window' if window_limited else 'the EOS table'
        raise InitialConditionError(
            'liquidus_super: no fully-molten initial condition is reachable within '
            f'{limit}; even the hottest valid adiabat (surface T={T_best:.0f} K) '
            f'is {-d_best["superheat"]:.0f} K below the liquidus at '
            f'P={d_best["binding_P"] / 1e9:.3g} GPa.'
        )

    clamped = not reached_delta
    if clamped:
        T_solved = T_best
        if window_limited:
            # The scan plus every extension stayed valid without reaching
            # delta: the search window, not the EOS table, is the limit here.
            log.warning(
                'liquidus_super: the requested superheat of %.0f K was not reached at '
                'P_cmb=%.0f GPa within the surface-temperature search window (the EOS '
                'table was not exhausted); clamped to the largest superheat sampled, '
                '%.0f K, at the top of the window (surface T=%.0f K). Widen the '
                'search window if a larger superheat is physically expected.',
                delta,
                P_cmb / 1e9,
                d_best['superheat'],
                T_solved,
            )
        else:
            log.warning(
                'liquidus_super: the requested superheat of %.0f K is not reachable '
                'at P_cmb=%.0f GPa within the EOS table; clamped to the largest '
                'achievable superheat of %.0f K (surface T=%.0f K). Lower '
                'delta_T_super if a full superheat margin is wanted.',
                delta,
                P_cmb / 1e9,
                d_best['superheat'],
                T_solved,
            )
    elif points[0][1]['superheat'] >= delta:
        if first_valid_idx == 0:
            # The coolest scanned adiabat already meets the margin (e.g.
            # delta=0): it is the coolest fully molten adiabat available.
            T_solved = points[0][0]
        else:
            # A leading invalid band ends somewhere between the last invalid
            # scan point and the first valid one; the coolest adiabat that is
            # both valid and at least delta above the liquidus lies in that
            # bracket, so refine it instead of returning the coarse point.
            T_solved = _bisect_delta(scan[first_valid_idx - 1][0], points[0][0])
    elif ceiling_bisected:
        # The crossing was reached only by refining the validity edge past
        # every coarse/extension-grid point, so it lies between T_sub_delta
        # (last point known below delta) and the refined T_best.
        T_solved = _bisect_delta(T_sub_delta, T_best)
    else:
        # points is monotonic in superheat (checked above), so the first
        # point at or above delta brackets the crossing with its predecessor.
        T_lo, T_hi = points[0][0], points[0][0]
        for T_pt, d_pt in points:
            if d_pt['superheat'] < delta:
                T_lo = T_pt
            else:
                T_hi = T_pt
                break
        T_solved = _bisect_delta(T_lo, T_hi)

    final = _probe(T_solved)
    if not final['valid']:
        raise InitialConditionError(
            f'liquidus_super: solved surface T={T_solved:.0f} K yielded an '
            f'out-of-table adiabat (P_cmb={P_cmb / 1e9:.0f} GPa); the EOS table '
            'is exhausted at this mass.'
        )

    binding_frac = final['binding_P'] / P_cmb
    if final['binding_P'] > FEI2021_LIQUIDUS_P_CALIB_PA:
        # The tightest superheat is set where the liquidus is an extrapolation
        # (beyond its calibration). The margin there is uncertain and the
        # construction approaches a CMB-liquidus anchor, which gives a
        # cold-surface IC; warn so a melting-curve swap that moves the binding
        # into the extrapolated regime does not silently land there. (A binding
        # that is fractionally deep but below the calibration, as for a
        # low-mass planet whose whole mantle is shallow, is fine.)
        log.warning(
            'liquidus_super: the minimum-superheat depth (binding P=%.3g GPa) '
            'is beyond the liquidus calibration (~%.0f GPa), so the superheat '
            'margin there is set against an extrapolated liquidus. Verify the '
            'liquidus parameterisation is appropriate for this EOS and mass.',
            final['binding_P'] / 1e9,
            FEI2021_LIQUIDUS_P_CALIB_PA / 1e9,
        )

    if not clamped:
        log.info(
            'liquidus_super: surface T=%.0f K gives a fully molten adiabat at least '
            '%.0f K above the liquidus (achieved %.0f K at P=%.3g GPa = %.0f%% of '
            'P_cmb); T_cmb=%.0f K, S=%.1f J/(kg K), P_cmb=%.0f GPa.',
            T_solved,
            delta,
            final['superheat'],
            final['binding_P'] / 1e9,
            binding_frac * 100.0,
            final['cmb_T'],
            final['S_target'],
            P_cmb / 1e9,
        )
    out = {
        'surface_T': float(T_solved),
        'S_target': float(final['S_target']),
        'cmb_T': float(final['cmb_T']),
        'achieved_superheat': float(final['superheat']),
        'binding_P': float(final['binding_P']),
        'P_cmb': P_cmb,
        'clamped': clamped,
        'window_limited': bool(clamped and window_limited),
    }
    _SUPERLIQ_CACHE[_cache_key] = dict(out)
    _SUPERLIQ_LAST_ANCHOR = float(out['cmb_T'])
    _SUPERLIQ_LAST_ANCHOR_FOR = _superliq_anchor_for(config)
    return out


def load_zalmoxis_configuration(
    config: Config,
    hf_row: dict,
    temperature_mode_override: str | None = None,
    external_temperature_source: bool = False,
):
    """Loads the model configuration for Zalmoxis and calculates the dry mass of the planet based on the total mass and the mass of volatiles.
    Args:
        config (Config): The configuration object containing the Zalmoxis parameters.
        hf_row (dict): A dictionary containing the mass of volatiles and other parameters.
        temperature_mode_override: Optional local override for
            ``config.planet.temperature_mode``. Lets callers force a different
            structure-solve mode (e.g. 'adiabatic' for SPIDER coupling with
            T-dependent mantle EOS) without mutating the shared Config object.
            When None, falls back to ``config.planet.temperature_mode``.
        external_temperature_source: True when the caller passes an external
            temperature function or array so the structure solve follows an
            evolved T(r) instead of the internal temperature-mode dispatch. In
            'liquidus_super' mode this lets ``cmb_temperature`` reuse the last
            solved super-liquidus anchor rather than re-solving it, since the
            anchor is not consumed when an external source drives the solve.
    Returns:
        dict: A dictionary containing the Zalmoxis configuration parameters.
    """

    # The Zalmoxis solver consumes core_frac as a mass fraction and does
    # not read core_frac_mode. Warn if a radius fraction was requested, so
    # the user is not surprised by the mass-fraction interpretation.
    if config.interior_struct.core_frac_mode == 'radius':
        log.warning(
            'interior_struct.core_frac_mode = "radius" has no effect with the '
            'zalmoxis module: core_frac (%.3f) is interpreted as a mass '
            'fraction. Use core_frac_mode = "mass", or switch to the dummy or '
            'spider module if a radius fraction is intended.',
            config.interior_struct.core_frac,
        )

    # Setup target planet mass (input parameter) as the total mass of the planet (dry mass + volatiles) [kg]
    total_planet_mass = config.planet.mass_tot * M_earth

    log.debug(
        'Total target planet mass (dry mass + volatiles): %s kg '
        'with EOS: core=%s, mantle=%s, ice=%s',
        total_planet_mass,
        config.interior_struct.zalmoxis.core_eos,
        config.interior_struct.zalmoxis.mantle_eos,
        config.interior_struct.zalmoxis.ice_layer_eos or 'none',
    )

    # Volatile mass (O included) excluded from the structure target: the full inventory
    # with dry_mantle, else only the atmosphere, since the wet-mantle EOS holds the dissolved
    # mass. Element columns absent before the first inventory count as zero (.get below).
    dry_mantle = config.interior_struct.zalmoxis.dry_mantle
    M_volatiles = 0.0
    for e in element_list:
        if dry_mantle:
            M_volatiles += float(hf_row.get(e + '_kg_total', 0.0))
        else:
            M_volatiles += float(hf_row.get(e + '_kg_atm', 0.0))

    log.debug(f'Volatile mass: {M_volatiles} kg')
    log.debug(
        'Mass budget: total=%.6e kg (%.4f M_earth), volatiles=%.6e kg (%.2f%%)',
        total_planet_mass,
        config.planet.mass_tot,
        M_volatiles,
        100.0 * M_volatiles / total_planet_mass if total_planet_mass > 0 else 0,
    )

    # Calculate the target planet mass (dry mass) by subtracting the mass of volatiles from the total planet mass
    planet_mass = total_planet_mass - M_volatiles

    log.debug(f'Target planet mass (dry mass): {planet_mass} kg ')

    # Build per-layer EOS config dict from PROTEUS config fields
    layer_eos_config = {
        'core': config.interior_struct.zalmoxis.core_eos,
        'mantle': config.interior_struct.zalmoxis.mantle_eos,
    }
    if config.interior_struct.zalmoxis.ice_layer_eos is not None:
        layer_eos_config['ice_layer'] = config.interior_struct.zalmoxis.ice_layer_eos

    # This dict only sees the dry layer_eos_config here; zalmoxis_solver
    # rebuilds it after extend_mantle_eos_with_volatiles adds a
    # dissolved-volatile component, so that component gets the real mzf too.
    mzf = config.interior_struct.zalmoxis.mushy_zone_factor
    mushy_zone_factors = _build_mushy_zone_factors(layer_eos_config, mzf)

    zc = config.interior_struct.zalmoxis
    log.debug(
        'Zalmoxis config: num_layers=%d, outer_solver=%s, tol_outer=%.1e, '
        'tol_inner=%.1e, use_jax=%s, use_anderson=%s',
        zc.num_levels,
        zc.outer_solver,
        zc.solver_tol_outer,
        zc.solver_tol_inner,
        zc.use_jax,
        zc.use_anderson,
    )

    # Core fraction. The Zalmoxis solver consumes `core_mass_fraction`
    # literally as a mass fraction and does not read `core_frac_mode`, so
    # `core_frac` is always interpreted as a mass fraction here regardless
    # of the mode. `core_frac_mode = "radius"` is only honoured by the
    # dummy and spider structure modules; a warning is emitted above when
    # it is combined with zalmoxis.

    # 'liquidus_super' can read the mantle EOS table here before
    # zalmoxis_solver's own pre-flight check runs, raising a bare
    # FileNotFoundError. That check never inspects the melting-curve
    # directory a WolfBower2018/RTPress100TPa mantle also needs, so a
    # missing one re-raises unconverted here and in zalmoxis_solver itself.
    try:
        cmb_temperature = _resolve_zalmoxis_cmb_temperature(
            config,
            hf_row,
            temperature_mode_override or config.planet.temperature_mode,
            external_temperature_source=external_temperature_source,
        )
    except FileNotFoundError:
        check_zalmoxis_eos_files(layer_eos_config, load_zalmoxis_material_dictionaries())
        raise

    return {
        'planet_mass': planet_mass,
        'core_mass_fraction': config.interior_struct.core_frac,
        'core_frac_mode': config.interior_struct.core_frac_mode,
        'mantle_mass_fraction': config.interior_struct.zalmoxis.mantle_mass_fraction,
        # For the structure solve, 'accretion' and 'isentropic' both reduce
        # to 'adiabatic' inside Zalmoxis. Zalmoxis only solves the structure
        # (M-R via hydrostatic + EOS); the entropy IC for Aragog/SPIDER is
        # set independently from config.planet.ini_entropy. 'accretion'
        # delays White+Li T-profile until after structure converges;
        # 'isentropic' (CHILI protocol) means the energetics solver
        # consumes ini_entropy, not the Zalmoxis T-profile, so the choice
        # of structure-solve T is decoupled from the IC. 'adiabatic_from_cmb'
        # is passed through to Zalmoxis with the CMB-anchor temperature so
        # the structure-side T(r) integrates upward from T_cmb at R_cmb,
        # matching the entropy that the energetics solver receives via
        # compute_initial_entropy. 'liquidus_super' maps to
        # 'adiabatic_from_cmb' here, with cmb_temperature derived from the
        # Fei+2021 liquidus at the converged P_cmb (or a Noack & Lasbleis
        # (2020) mass-aware P_cmb estimate on the very first call before
        # Zalmoxis has populated P_cmb) plus delta_T_super. The anchor is
        # recomputed against the converged P_cmb on the next structure
        # iteration. temperature_mode_override lets SPIDER coupling force
        # adiabatic without mutating the shared Config object (see proteus
        # rules §"Config mutability").
        'temperature_mode': _resolve_zalmoxis_temperature_mode(
            temperature_mode_override or config.planet.temperature_mode
        ),
        'surface_temperature': config.planet.tsurf_init,
        'cmb_temperature': cmb_temperature,
        'center_temperature': config.planet.tcenter_init,
        'temp_profile_file': None,
        'layer_eos_config': layer_eos_config,
        'mushy_zone_factor': mzf,
        'mushy_zone_factors': mushy_zone_factors,
        'num_layers': config.interior_struct.zalmoxis.num_levels,
        'target_surface_pressure': _get_target_surface_pressure(config, hf_row),
        # Solver tolerances and iteration limits
        'tolerance_outer': config.interior_struct.zalmoxis.solver_tol_outer,
        'tolerance_inner': config.interior_struct.zalmoxis.solver_tol_inner,
        'max_iterations_outer': config.interior_struct.zalmoxis.solver_max_iter_outer,
        'max_iterations_inner': config.interior_struct.zalmoxis.solver_max_iter_inner,
        # JAX+diffrax structure path and Anderson Picard acceleration,
        # both opt-in and defaulting off. See `Zalmoxis.use_jax` /
        # `Zalmoxis.use_anderson` in proteus.config._struct.
        'use_jax': config.interior_struct.zalmoxis.use_jax,
        'use_anderson': config.interior_struct.zalmoxis.use_anderson,
        # outer mass-radius solver dispatch ('newton' default |
        # 'picard'). When 'newton', Zalmoxis uses Newton + brentq
        # bracketing on f(R) = M(R) - M_target instead of the
        # damped-Picard fixed-point loop. Newton requires tight
        # integrator tolerances; we auto-apply newton_relative_tolerance
        # / newton_absolute_tolerance when the Newton path is selected.
        'outer_solver': config.interior_struct.zalmoxis.outer_solver,
        # Newton-specific knobs (newton_max_iter, newton_tol) AND
        # tightened integrator tolerances are passed ONLY when the
        # Newton path is selected. Picard runs see the dict without
        # any Newton keys and without tightened tolerances, so a
        # future Zalmoxis guard against unknown keys would not break
        # Picard callers.
        **(
            {
                'newton_max_iter': (config.interior_struct.zalmoxis.newton_max_iter),
                'newton_tol': config.interior_struct.zalmoxis.newton_tol,
                'relative_tolerance': (
                    config.interior_struct.zalmoxis.newton_relative_tolerance
                ),
                'absolute_tolerance': (
                    config.interior_struct.zalmoxis.newton_absolute_tolerance
                ),
            }
            if config.interior_struct.zalmoxis.outer_solver == 'newton'
            else {}
        ),
    }


def _zalmoxis_jax_structure_viable(mat_dicts: dict, core_eos: str, mantle_eos: str) -> bool:
    """Report whether the Zalmoxis JAX structure path can run for an EOS pair.

    Mirrors, on static registry properties, the preconditions that
    ``zalmoxis.jax_eos.wrapper.solve_structure_via_jax`` enforces before
    integrating: the mantle registry entry must be one of the two
    supported representations, the unified single-table layout (a flat
    entry with format ``paleos_unified`` carrying an ``eos_file``, the
    production default) or the 2-phase PALEOS layout (both
    ``solid_mantle`` and ``melted_mantle`` dict sub-tables), and the core
    entry must resolve to the ``paleos_unified`` format. A ``paleos_api``
    entry qualifies on either layer because
    ``zalmoxis.eos.paleos_api_cache.resolve_registry_entry`` materialises
    it to ``paleos_unified`` (with an on-disk ``eos_file``) in place
    before the JAX dispatch checks it.

    When a precondition fails, the Zalmoxis dispatch raises inside
    the JAX wrapper and falls back to the numpy ODE, which consumes
    ``temperature_function`` rather than ``temperature_arrays``; the
    caller must therefore keep the temperature callable in play.

    Parameters
    ----------
    mat_dicts : dict
        EOS registry from :func:`load_zalmoxis_material_dictionaries`.
    core_eos : str
        Core EOS identifier (registry key).
    mantle_eos : str
        Mantle EOS identifier (registry key). Trailing mass-fraction
        tokens from volatile-extended strings are stripped before the
        lookup.

    Returns
    -------
    bool
        True when the JAX dispatch preconditions hold for both layers.
        False otherwise, including unknown registry keys: the
        conservative answer routes the solve to the numpy path with the
        callable attached.
    """
    mantle_entry = mat_dicts.get(_strip_fraction_tokens(str(mantle_eos)))
    if not isinstance(mantle_entry, dict):
        return False
    mantle_format = mantle_entry.get('format')
    if mantle_format == 'paleos_unified':
        mantle_viable = bool(mantle_entry.get('eos_file'))
    elif mantle_format == 'paleos_api':
        mantle_viable = True
    else:
        mantle_viable = isinstance(mantle_entry.get('solid_mantle'), dict) and isinstance(
            mantle_entry.get('melted_mantle'), dict
        )
    if not mantle_viable:
        return False
    core_entry = mat_dicts.get(_strip_fraction_tokens(str(core_eos)))
    if not isinstance(core_entry, dict):
        return False
    return core_entry.get('format') in ('paleos_unified', 'paleos_api')


def _volatile_profile_jax_viable(
    volatile_profile, mat_dicts: dict, mantle_eos_extended
) -> bool:
    """Report whether a ``VolatileProfile`` fits the Zalmoxis JAX wet envelope.

    Mirrors, on static profile and registry properties, the checks that
    ``zalmoxis.jax_eos.wrapper._validate_wet_mantle`` and the volatile
    cache load in ``zalmoxis.jax_eos.wrapper.solve_structure_via_jax``
    enforce before integrating a wet mantle: no ``global_miscibility``,
    no ``x_interior`` entries, zero (or absent) weight on the primary
    silicate component, the primary present in the mantle mixture,
    exactly one active volatile in the mixture, no mixture components
    outside the profile, the volatile not ``Chabrier:H`` (its binodal
    suppression factor is not ported to the JAX RHS), and a volatile
    registry entry that resolves to the ``paleos_unified`` format with
    an ``eos_file`` (``paleos_api`` entries materialise to that in
    place).

    When any check fails, the Zalmoxis dispatch raises inside the JAX
    wrapper and falls back to the numpy ODE, which consumes
    ``temperature_function`` rather than ``temperature_arrays``; the
    caller must therefore keep the temperature callable in play.

    Parameters
    ----------
    volatile_profile : VolatileProfile or None
        Profile built by :func:`build_volatile_profile`. None reads as
        not viable; callers gate the dry case separately.
    mat_dicts : dict
        EOS registry from :func:`load_zalmoxis_material_dictionaries`.
    mantle_eos_extended : str
        Mantle EOS string after
        :func:`extend_mantle_eos_with_volatiles`, whose components (with
        fraction tokens stripped) are the mantle mixture Zalmoxis
        parses.

    Returns
    -------
    bool
        True when the profile matches the JAX wet envelope. False
        otherwise, including malformed or unrecognised input: the
        conservative answer routes the solve to the numpy path with the
        callable attached.
    """
    try:
        if volatile_profile is None:
            return False
        if getattr(volatile_profile, 'global_miscibility', False):
            return False
        if getattr(volatile_profile, 'x_interior', None):
            return False
        primary = str(volatile_profile.primary_component)
        w_liquid = dict(volatile_profile.w_liquid)
        w_solid = dict(volatile_profile.w_solid)
        # A nonzero weight on the primary silicate is degenerate input
        # the wrapper rejects (numpy's blend() double-counts it).
        if float(w_liquid.get(primary, 0.0)) > 0.0 or float(w_solid.get(primary, 0.0)) > 0.0:
            return False
        components = [
            _strip_fraction_tokens(token) for token in str(mantle_eos_extended).split('+')
        ]
        if primary not in components:
            return False
        managed = (set(w_liquid) | set(w_solid)) - {primary}
        active = [
            key
            for key in managed
            if (float(w_liquid.get(key, 0.0)) > 0.0 or float(w_solid.get(key, 0.0)) > 0.0)
            and key in components
        ]
        if len(active) != 1:
            return False
        # Unmanaged extra mixture components keep their fraction only on
        # the numpy path; the wrapper rejects them.
        if any(c != primary and c not in managed for c in components):
            return False
        (vol_eos,) = active
        if vol_eos == 'Chabrier:H':
            return False
        vol_entry = mat_dicts.get(vol_eos)
        if not isinstance(vol_entry, dict):
            return False
        vol_format = vol_entry.get('format')
        if vol_format == 'paleos_api':
            return True
        return vol_format == 'paleos_unified' and bool(vol_entry.get('eos_file'))
    except (AttributeError, TypeError, ValueError):
        return False


def load_zalmoxis_material_dictionaries():
    """Build an EOS registry dict with file paths pointing to FWL_DATA.

    Returns the same dict format as Zalmoxis ``EOS_REGISTRY``, but with
    every ``eos_file`` path resolved to its fwl-io dataset directory in
    ``FWL_DATA`` instead of ``ZALMOXIS_ROOT/data/``. The data root is the one
    ``download_zalmoxis_eos()`` fetches into, so Zalmoxis, when called from
    PROTEUS, reads the files that were fetched.

    Returns
    -------
    dict
        Flat dict keyed by EOS identifier string (e.g.
        ``"Seager2007:iron"``, ``"PALEOS:MgSiO3"``, ``"Chabrier:H"``).
    """
    root = GetFWLData()
    seager_dir = dataset_dir(EOS_SEAGER_2007, data_root=root)

    _seager_iron = {'eos_file': str(seager_dir / 'eos_seager07_iron.txt')}
    _seager_silicate = {'eos_file': str(seager_dir / 'eos_seager07_silicate.txt')}
    _seager_water = {'eos_file': str(seager_dir / 'eos_seager07_water.txt')}

    # Wolf & Bower 2018
    wb_dir = dataset_dir(EOS_WOLF_BOWER_2018, data_root=root)
    _wb_melted = {
        'eos_file': str(wb_dir / 'density_melt.dat'),
        'adiabat_grad_file': str(wb_dir / 'adiabat_temp_grad_melt.dat'),
    }
    _wb_solid = {'eos_file': str(wb_dir / 'density_solid.dat')}

    # RTPress 100 TPa
    rt_dir = dataset_dir(EOS_RTPRESS_100TPA, data_root=root)
    _rt_melted = {
        'eos_file': str(rt_dir / 'density_melt.dat'),
        'adiabat_grad_file': str(rt_dir / 'adiabat_temp_grad_melt.dat'),
    }

    # PALEOS 2-phase MgSiO3 (separate solid/liquid, Zenodo 19680050): the
    # 150 pts/decade (default) and 600 pts/decade (highres) pairs share one dataset.
    paleos2ph_dir = dataset_dir(EOS_PALEOS_MGSIO3, data_root=root)
    paleos2ph_hr_dir = paleos2ph_dir
    _paleos2ph_melted = {
        'eos_file': str(paleos2ph_dir / 'paleos_mgsio3_tables_pt_proteus_liquid.dat'),
        'format': 'paleos',
    }
    _paleos2ph_solid = {
        'eos_file': str(paleos2ph_dir / 'paleos_mgsio3_tables_pt_proteus_solid.dat'),
        'format': 'paleos',
    }
    _paleos2ph_melted_highres = {
        'eos_file': str(
            paleos2ph_hr_dir / 'paleos_mgsio3_tables_pt_proteus_liquid_highres.dat'
        ),
        'format': 'paleos',
    }
    _paleos2ph_solid_highres = {
        'eos_file': str(paleos2ph_hr_dir / 'paleos_mgsio3_tables_pt_proteus_solid_highres.dat'),
        'format': 'paleos',
    }

    # PALEOS unified tables, one dataset per material
    _paleos_iron = {
        'eos_file': str(
            dataset_dir(EOS_PALEOS_IRON, data_root=root) / 'paleos_iron_eos_table_pt.dat'
        ),
        'format': 'paleos_unified',
    }
    _paleos_mgsio3 = {
        'eos_file': str(
            dataset_dir(EOS_PALEOS_MGSIO3_UNIFIED, data_root=root)
            / 'paleos_mgsio3_eos_table_pt.dat'
        ),
        'format': 'paleos_unified',
    }
    _paleos_h2o = {
        'eos_file': str(
            dataset_dir(EOS_PALEOS_H2O, data_root=root) / 'paleos_water_eos_table_pt.dat'
        ),
        'format': 'paleos_unified',
    }

    # Chabrier H/He
    chabrier_dir = dataset_dir(EOS_CHABRIER_2021, data_root=root)
    _chabrier_h = {
        'eos_file': str(chabrier_dir / 'EOS_Chabrier2021_HHe' / 'chabrier2021_H.dat'),
        'format': 'paleos_unified',
    }

    # PALEOS-API live tabulation entries. These carry only a GridSpec at
    # build time; the dispatch layer (``zalmoxis.eos.paleos_api_cache``)
    # mutates them in place on first density query to populate ``eos_file``
    # and rewrite ``format`` to the downstream value (``paleos_unified`` or
    # ``paleos``). Cache keys are SHA+grid-hash under
    # ``$ZALMOXIS_ROOT/data/EOS_PALEOS_API/``; cold-cache cost is a one-time
    # generator run (see ``zalmoxis.eos.paleos_api``).
    from zalmoxis.eos.paleos_api import (
        make_default_grid_h2o,
        make_default_grid_iron,
        make_default_grid_mgsio3,
    )

    _paleos_api_iron = {
        'format': 'paleos_api',
        'material': 'iron',
        'grid_spec': make_default_grid_iron(),
    }
    _paleos_api_mgsio3 = {
        'format': 'paleos_api',
        'material': 'mgsio3',
        'grid_spec': make_default_grid_mgsio3(),
    }
    _paleos_api_h2o = {
        'format': 'paleos_api',
        'material': 'h2o',
        'grid_spec': make_default_grid_h2o(),
        'h2o_table_path': None,
    }
    _paleos_api_2ph_mgsio3_melted = {
        'format': 'paleos_api_2phase',
        'material': 'mgsio3',
        'side': 'liquid',
        'grid_spec': make_default_grid_mgsio3(),
    }
    _paleos_api_2ph_mgsio3_solid = {
        'format': 'paleos_api_2phase',
        'material': 'mgsio3',
        'side': 'solid',
        'grid_spec': make_default_grid_mgsio3(),
    }

    return {
        # Seager2007 static
        'Seager2007:iron': {'core': _seager_iron},
        'Seager2007:MgSiO3': {'mantle': _seager_silicate},
        'Seager2007:H2O': {'ice_layer': _seager_water},
        # Wolf & Bower 2018 T-dependent
        'WolfBower2018:MgSiO3': {
            'core': _seager_iron,
            'melted_mantle': _wb_melted,
            'solid_mantle': _wb_solid,
        },
        # RTPress 100 TPa extended melt + WB2018 solid
        'RTPress100TPa:MgSiO3': {
            'core': _seager_iron,
            'melted_mantle': _rt_melted,
            'solid_mantle': _wb_solid,
        },
        # PALEOS 2-phase MgSiO3 (Zenodo 19680050; 150 pts/decade default,
        # 600 pts/decade as -highres variant for sensitivity tests).
        'PALEOS-2phase:MgSiO3': {
            'core': _seager_iron,
            'melted_mantle': _paleos2ph_melted,
            'solid_mantle': _paleos2ph_solid,
        },
        'PALEOS-2phase:MgSiO3-highres': {
            'core': _seager_iron,
            'melted_mantle': _paleos2ph_melted_highres,
            'solid_mantle': _paleos2ph_solid_highres,
        },
        # PALEOS unified
        'PALEOS:iron': _paleos_iron,
        'PALEOS:MgSiO3': _paleos_mgsio3,
        'PALEOS:H2O': _paleos_h2o,
        # PALEOS-API live-tabulated (dispatch populates eos_file on demand)
        'PALEOS-API:iron': _paleos_api_iron,
        'PALEOS-API:MgSiO3': _paleos_api_mgsio3,
        'PALEOS-API:H2O': _paleos_api_h2o,
        'PALEOS-API-2phase:MgSiO3': {
            'core': _seager_iron,
            'melted_mantle': _paleos_api_2ph_mgsio3_melted,
            'solid_mantle': _paleos_api_2ph_mgsio3_solid,
        },
        # Chabrier H/He
        'Chabrier:H': _chabrier_h,
    }


#: EOS names whose density blend applies a configured mushy_zone_factor.
#: Imported from Zalmoxis so both sides read the same list.
_UNIFIED_PALEOS_MATERIALS = _PALEOS_UNIFIED_NAMES


def _build_mushy_zone_factors(layer_eos_config: dict, mzf: float) -> dict:
    """Build the per-material mushy zone factor dict for the unified PALEOS tables.

    Parameters
    ----------
    layer_eos_config : dict
        Per-layer EOS identifier strings, keyed by layer name. A value may
        join several components with ``'+'`` and carry mass-fraction tokens
        appended by :func:`extend_mantle_eos_with_volatiles`.
    mzf : float
        The configured ``mushy_zone_factor``.

    Returns
    -------
    dict
        Maps each name in :data:`_UNIFIED_PALEOS_MATERIALS` to ``mzf`` if a
        layer actually uses it, else to 1.0.

    Notes
    -----
    On the PALEOS-API coupled path at ``mzf = 1.0``, ``F_atm`` in the three
    Time=0 rows differs by up to a few percent between otherwise identical
    runs, with the largest difference at the second row. The difference
    falls to run-to-run noise by t = 12 yr. These figures come from a single
    pair of runs and are an order-of-magnitude statement, not a bound.
    """
    configured_eos = {
        _strip_fraction_tokens(token.strip())
        for v in layer_eos_config.values()
        if v
        for token in str(v).split('+')
    }
    return {
        name: (mzf if name in configured_eos else 1.0) for name in _UNIFIED_PALEOS_MATERIALS
    }


class ZalmoxisMissingEOSFilesError(RuntimeError):
    """A layer's configured EOS identifier names a table file absent on disk."""


def check_zalmoxis_eos_files(
    layer_eos_config: dict, mat_dicts: dict, paleos_companions: bool = False
) -> None:
    """Fail fast when a selected EOS table file is missing on disk.

    Walks the registry entries selected by ``layer_eos_config`` and
    collects every referenced table path that does not exist, then
    raises one actionable error. Without this check the solver emits
    one read error per shell and ends in a non-convergence failure that
    hides the real cause. PALEOS-API entries (live tabulation) are
    materialised here, so a failed build stops the run before any solve.

    Parameters
    ----------
    layer_eos_config : dict
        Per-layer EOS identifier strings (``'core'``, ``'mantle'``, ...).
    mat_dicts : dict
        EOS registry from :func:`load_zalmoxis_material_dictionaries`.
    paleos_companions : bool
        Also require the :func:`paleos_companion_keys` of the mantle.

    Raises
    ------
    ZalmoxisMissingEOSFilesError
        If any selected EOS table file is missing, naming every missing
        path and the command that downloads them.
    """
    from zalmoxis.eos.dispatch import _is_paleos_api

    selected = [
        (role, key)
        for role, identifier in layer_eos_config.items()
        for key in eos_components(identifier)
    ]
    if paleos_companions:
        companions = paleos_companion_keys(layer_eos_config.get('mantle', ''))
        selected += [('mantle', key) for key in companions]
    missing: set[str] = set()
    for layer_role, key in selected:
        entry = mat_dicts.get(key)
        if entry is None:
            # Unknown identifiers fail later with a registry error.
            continue
        if _is_paleos_api(entry):
            try:
                from zalmoxis.eos.paleos_api_cache import resolve_registry_entry

                resolve_registry_entry(entry)
            except Exception as exc:
                missing.add(f'{key} (PALEOS-API tables not built: {type(exc).__name__}: {exc})')
                continue
        # Nested entries map layer roles to flat entries; their 'core' sub-entry
        # counts only for the core role or when it is the only sub-entry.
        if 'eos_file' in entry:
            subentries = [entry]
        else:
            has_other_roles = any(role != 'core' for role in entry)
            subentries = [
                sub
                for role, sub in entry.items()
                if role != 'core' or layer_role == 'core' or not has_other_roles
            ]
        for sub in subentries:
            if not isinstance(sub, dict):
                continue
            for field in ('eos_file', 'adiabat_grad_file'):
                path = sub.get(field)
                if path and not os.path.isfile(path):
                    missing.add(path)
    if missing:
        listing = '\n  '.join(sorted(missing))
        raise ZalmoxisMissingEOSFilesError(
            f'Interior EOS table file(s) not found:\n  {listing}\n'
            'Download them with '
            '`proteus get interiordata --config-path <config.toml>`, '
            f'or run `proteus start` once without --offline. {RELOCATE_HINT}'
        )


def energetics_entry(mantle_eos: str, mat_dicts: dict) -> tuple[str | None, dict]:
    """Energetics key of a mantle EOS and its registry entry (empty when absent)."""
    key = energetics_eos_key(mantle_eos)
    return key, mat_dicts.get(key or '', {})


def require_paleos_tables(config: Config, outdir: str) -> None:
    """Stop the run before any solve when a table of its EOS set is missing.

    Requires every table file of the core, mantle and ice-layer EOS and,
    for a mantle with a PALEOS component, the MgSiO3 2-phase pair; a resumed
    SPIDER run that keeps its P-S tables is held only to its layer tables. The
    liquidus_super initial adiabat requires the pair for any mantle. With
    ``dry_mantle = false`` the dissolved-volatile tables are also required. A
    mantle with a non-MgSiO3 component gets one WARNING that names the tables
    its energetics and its liquidus_super initial adiabat use.

    Parameters
    ----------
    config : Config
        Configuration object with struct.zalmoxis settings.
    outdir : str
        The run output directory.

    Raises
    ------
    ZalmoxisMissingEOSFilesError
        Naming every missing file and the command that downloads it.
    """
    zc = config.interior_struct.zalmoxis
    layers = {'core': zc.core_eos, 'mantle': zc.mantle_eos}
    if zc.ice_layer_eos is not None:
        layers['ice_layer'] = zc.ice_layer_eos
    if not zc.dry_mantle:
        # Dissolved volatiles join the mantle EOS during the run.
        layers['volatiles'] = '+'.join(VOLATILE_EOS_MAP.values())
    mat_dicts = load_zalmoxis_material_dictionaries()
    kept = config.params.resume and _resumed_ps_tables(
        outdir,
        lambda: _ps_resume_key(config, *energetics_entry(zc.mantle_eos, mat_dicts), mat_dicts),
    )
    liquidus_super = config.planet.temperature_mode == 'liquidus_super'
    if liquidus_super:
        layers['anchor'] = twophase_registry_key(zc.mantle_eos)
    # Aragog re-solves the initial condition on the pair when the mesh changes.
    check_zalmoxis_eos_files(
        layers,
        mat_dicts,
        paleos_companions=not kept or config.interior_energetics.module == 'aragog',
    )
    components = eos_components(zc.mantle_eos)
    mixture = len(set(components)) > 1
    if all(is_mgsio3(c) for c in components):
        return
    uses = []
    if config.interior_energetics.module in ('spider', 'aragog'):
        if not generates_paleos_tables(config.interior_struct):
            uses.append(
                'the energetics use the eos_dir P-S tables (by default Wolf and Bower 2018) '
                'and the melting_dir curves'
            )
        else:
            uses.append(
                'the energetics and melting curves are PALEOS MgSiO3 '
                f'({energetics_eos_key(zc.mantle_eos)}, solidus = '
                f'{zc.mushy_zone_factor:.2f} x PALEOS liquidus)'
            )
    if liquidus_super:
        uses.append('the liquidus_super initial adiabat is solved on the MgSiO3 2-phase tables')
    if uses:
        head = (
            'the non-MgSiO3 components enter only the structure density; '
            if mixture
            else 'the structure uses its density, while '
        )
        log.warning('mantle_eos=%s: %s%s', zc.mantle_eos, head, '; '.join(uses))


def resolve_2phase_mgsio3_paths(mantle_eos: str, mat_dicts: dict, required: bool = False):
    """Return the 2-phase MgSiO3 table paths, stopping on a missing one when required.

    Parameters
    ----------
    mantle_eos : str
        Configured mantle EOS.
    mat_dicts : dict
        Zalmoxis material dictionaries.
    required : bool
        Raise instead of returning None for a missing table; callers set it for a
        run with a generated PALEOS table set, which requires the pair.

    Returns
    -------
    tuple[str | None, str | None]
        See :func:`_twophase_mgsio3_paths`.

    Raises
    ------
    ZalmoxisMissingEOSFilesError
        If ``required`` and a table of the pair is not available.
    """
    paths = _twophase_mgsio3_paths(mantle_eos, mat_dicts)
    if required and None in paths:
        missing = [p for p, ok in zip(('solid', 'liquid'), paths) if ok is None]
        raise ZalmoxisMissingEOSFilesError(
            f'PALEOS 2-phase MgSiO3 tables {twophase_registry_key(mantle_eos)} not '
            f'available: {", ".join(missing)}. Download them with '
            f'`proteus get interiordata --config-path <config.toml>`. {RELOCATE_HINT}'
        )
    return paths


def _twophase_mgsio3_paths(mantle_eos: str, mat_dicts: dict):
    """Return (solid_eos_path, liquid_eos_path) for the 2-phase MgSiO3 tables.

    Selects the API key (``PALEOS-API-2phase:MgSiO3``) when ``mantle_eos``
    is from the PALEOS-API family, otherwise the shipped key
    (``PALEOS-2phase:MgSiO3``). For API entries, calls
    :func:`zalmoxis.eos.paleos_api_cache.resolve_registry_entry` to
    materialise cached ``.dat`` paths in place; this is required because
    :func:`load_zalmoxis_material_dictionaries` rebuilds a fresh registry
    each call, so any earlier in-place mutation is lost.

    Returns
    -------
    tuple[str | None, str | None]
        Absolute filesystem path for each table that exists on disk,
        ``None`` for the other; both are ``(None, None)`` when the
        registry has no entry for the resolved key or the PALEOS-API
        resolver is unavailable. The two entries are resolved and
        checked independently, so one can be a path while the other is
        ``None``. Caller is responsible for treating each ``None``
        individually as "this table is not available".
    """
    twophase_key = twophase_registry_key(mantle_eos)
    use_api = twophase_key.startswith('PALEOS-API')
    twophase = mat_dicts.get(twophase_key, {})
    if not twophase:
        log.warning(
            'resolve_2phase_mgsio3_paths: registry has no entry for %s '
            '(mantle_eos=%s); 2-phase fallback disabled. This is a '
            'silent-wrong landmine if the caller continues with empty paths.',
            twophase_key,
            mantle_eos,
        )
        return None, None
    if use_api:
        try:
            from zalmoxis.eos.paleos_api_cache import resolve_registry_entry

            resolve_registry_entry(twophase)
        except (ImportError, ModuleNotFoundError) as e:
            log.warning(
                'resolve_2phase_mgsio3_paths: PALEOS-API resolver unavailable '
                '(%s); cannot materialise %s tables.',
                e,
                twophase_key,
            )
            return None, None
    solid_eos = twophase.get('solid_mantle', {}).get('eos_file', '')
    liquid_eos = twophase.get('melted_mantle', {}).get('eos_file', '')
    solid_eos = solid_eos if solid_eos and os.path.isfile(solid_eos) else None
    liquid_eos = liquid_eos if liquid_eos and os.path.isfile(liquid_eos) else None
    return solid_eos, liquid_eos


def load_zalmoxis_solidus_liquidus_functions(mantle_eos: str, config: Config):
    """Loads the solidus and liquidus functions for Zalmoxis based on the mantle EOS.

    Melting curves are needed for two purposes:
    1. Temperature-dependent density in the mushy zone (WolfBower2018, RTPress).
    2. phi(r) blending in VolatileProfile (any EOS with dissolved volatiles).

    The curves follow the energetics key of the mantle EOS
    (:func:`proteus.utils.helper.energetics_eos_key`), the same key as the
    energetics. For WolfBower2018/RTPress100TPa, loads SPIDER-format P-T files from FWL_DATA.
    For PALEOS unified and PALEOS-2phase, the liquidus comes from the analytic
    Belonoshko+2005 / Fei+2021 curve (Zalmoxis ``'PALEOS-liquidus'``) which is
    the basis Zalmoxis uses for MgSiO3 phase separation, and the solidus is
    derived as T_sol = T_liq * mushy_zone_factor. This keeps the curves used
    for phi-blending and 2-phase nabla_ad consistent with the unified PALEOS
    density-interpolation phase boundaries.

    Parameters
    ----------
    mantle_eos : str
        Mantle EOS string (e.g. ``"WolfBower2018:MgSiO3"``, ``"PALEOS:MgSiO3"``,
        ``"PALEOS-2phase:MgSiO3"``).
    config : Config
        PROTEUS configuration object.

    Returns
    -------
    tuple or None
        (solidus_func, liquidus_func) callable P [Pa] -> T [K], or None.
    """
    key = energetics_eos_key(mantle_eos) or ''
    if key.startswith(TDEP_EOS_PREFIXES):
        return get_zalmoxis_melting_curves(config)

    # PALEOS unified and 2-phase share the analytic PALEOS-liquidus (Belonoshko+2005 /
    # Fei+2021), with solidus = mushy_zone_factor x liquidus. The 2-phase nabla_ad needs
    # both curves, and without them the unified path falls back to phi = 0.5.
    if key.startswith(PALEOS_EOS_PREFIXES):
        try:
            from zalmoxis.melting_curves import (
                derive_solidus_from_liquidus,
                get_solidus_liquidus_functions,
            )

            _, liquidus_func = get_solidus_liquidus_functions(
                solidus_id='Stixrude14-solidus',  # required by API but unused; solidus is built below as mushy_zone_factor * liquidus (constant depression of the PALEOS liquidus)
                liquidus_id='PALEOS-liquidus',
            )
            mzf = config.interior_struct.zalmoxis.mushy_zone_factor
            solidus_func = derive_solidus_from_liquidus(liquidus_func, mzf)
            log.info(
                'PALEOS melting curves (%s): liquidus from PALEOS, '
                'solidus = liquidus * %.2f (mushy_zone_factor)',
                mantle_eos,
                mzf,
            )
            return solidus_func, liquidus_func
        except Exception as e:
            log.warning('Could not load PALEOS melting curves: %s', e)
            return None

    return None


def scale_temperature_profile_for_aragog(
    config: Config, mantle_radii: np.ndarray, mantle_temperature_profile: np.ndarray
):
    """Scales the temperature profile obtained from Zalmoxis to match the number of levels required by Aragog.
    Args:
        config (Config): The configuration object containing the configuration parameters.
        mantle_radii (np.ndarray): The radial positions of the mantle layers from Zalmoxis.
        mantle_temperature_profile (np.ndarray): The temperature profile of the mantle layers from Zalmoxis.
    Returns:
        np.ndarray: The scaled temperature profile matching the number of levels in Aragog.
    """

    # Number of levels in Aragog mesh
    mesh_grid_size = config.interior_energetics.num_levels - 1

    # Create new evenly spaced radial positions for Aragog
    radii_to_interpolate = np.linspace(mantle_radii[0], mantle_radii[-1], mesh_grid_size)

    # Cubic interpolation onto the Aragog radial mesh
    cubic_interp = interp1d(mantle_radii, mantle_temperature_profile, kind='cubic')
    return cubic_interp(radii_to_interpolate)


def write_spider_mesh_file(
    outdir: str,
    mantle_radii: np.ndarray,
    mantle_pressure: np.ndarray,
    mantle_density: np.ndarray,
    mantle_gravity: np.ndarray,
    num_basic: int,
) -> str:
    """Write an external mesh file for SPIDER from Zalmoxis mantle profiles.

    Interpolates the Zalmoxis mantle arrays onto uniformly-spaced SPIDER
    basic and staggered nodes, then writes the mesh file in the format
    expected by SPIDER's ``SetMeshFromExternalFile()``.

    Parameters
    ----------
    outdir : str
        PROTEUS output directory (file is written to ``outdir/data/``).
    mantle_radii : np.ndarray
        Radial positions from CMB to surface, ascending [m].
    mantle_pressure : np.ndarray
        Pressure at each radius [Pa].
    mantle_density : np.ndarray
        Density at each radius [kg/m^3].
    mantle_gravity : np.ndarray
        Gravity magnitude at each radius [m/s^2] (positive).
    num_basic : int
        Number of SPIDER basic nodes (shell boundaries).

    Returns
    -------
    str
        Path to the written mesh file.
    """
    num_staggered = num_basic - 1
    R_surf = float(mantle_radii[-1])
    R_cmb = float(mantle_radii[0])

    # Basic nodes: uniform spacing from surface to CMB (descending r)
    r_b = np.linspace(R_surf, R_cmb, num_basic)
    # Staggered nodes: midpoints between consecutive basic nodes
    r_s = 0.5 * (r_b[:-1] + r_b[1:])

    # Interpolate Zalmoxis profiles onto node positions
    # mantle_radii is ascending, np.interp requires ascending xp
    P_b = np.interp(r_b, mantle_radii, mantle_pressure)
    rho_b = np.interp(r_b, mantle_radii, mantle_density)
    g_b = np.interp(r_b, mantle_radii, mantle_gravity)

    P_s = np.interp(r_s, mantle_radii, mantle_pressure)
    rho_s = np.interp(r_s, mantle_radii, mantle_density)
    g_s = np.interp(r_s, mantle_radii, mantle_gravity)

    # Negate gravity for SPIDER convention (inward-pointing, negative)
    g_b = -np.abs(g_b)
    g_s = -np.abs(g_s)

    # Write mesh file
    mesh_path = os.path.join(outdir, 'data', 'spider_mesh.dat')
    with open(mesh_path, 'w') as f:
        f.write(f'# {num_basic} {num_staggered}\n')
        for i in range(num_basic):
            f.write(f'{r_b[i]:.15e} {P_b[i]:.15e} {rho_b[i]:.15e} {g_b[i]:.15e}\n')
        for i in range(num_staggered):
            f.write(f'{r_s[i]:.15e} {P_s[i]:.15e} {rho_s[i]:.15e} {g_s[i]:.15e}\n')

    log.info(
        'Wrote SPIDER mesh file: %s (%d basic + %d staggered nodes)',
        mesh_path,
        num_basic,
        num_staggered,
    )

    return mesh_path


# Name of the pointer file that records a shared PROTEUS_PS_CACHE_DIR table
# location inside a run's output/<run>/data directory. A resumed run rebuilds
# dirs['spider_eos_dir'] from the per-run output/<run>/data/spider_eos path,
# which a shared-cache run never populates; the pointer lets resume follow the
# tables to the shared cache without re-deriving the cache key.
PS_CACHE_POINTER_NAME = 'spider_eos_cache.txt'


def _write_ps_cache_pointer(outdir: str, cache_dir: str) -> None:
    """Record a shared PS-cache table location for resume.

    Writes the absolute `cache_dir` into ``<outdir>/data`` under
    :data:`PS_CACHE_POINTER_NAME`. Silent on I/O error: the pointer is a
    resume convenience, not a correctness requirement for the current run.

    Parameters
    ----------
    outdir : str
        The run output directory.
    cache_dir : str
        The shared PROTEUS_PS_CACHE_DIR table directory to record.
    """

    pointer = os.path.join(outdir, 'data', PS_CACHE_POINTER_NAME)
    try:
        os.makedirs(os.path.dirname(pointer), exist_ok=True)
        with open(pointer, 'w') as f:
            f.write(os.path.abspath(cache_dir))
    except OSError:
        pass


def read_ps_cache_pointer(outdir: str) -> str | None:
    """Return the shared PS-cache table directory recorded for `outdir`.

    Reads the pointer written by :func:`_write_ps_cache_pointer`. Used by
    the resume path to locate PROTEUS_PS_CACHE_DIR tables that live outside
    the run directory.

    Parameters
    ----------
    outdir : str
        The run output directory.

    Returns
    -------
    str or None
        The recorded table directory, or None when the pointer is absent,
        unreadable, or empty.
    """

    pointer = os.path.join(outdir, 'data', PS_CACHE_POINTER_NAME)
    if not os.path.isfile(pointer):
        return None
    try:
        with open(pointer) as f:
            cache_dir = f.read().strip()
    except OSError:
        return None
    return cache_dir or None


# Kept-table directories already reported in this process.
_PS_RESUME_REPORTED: set[str] = set()


def _resumed_ps_tables(outdir: str, current_key) -> dict | None:
    """Return the P-S tables a resumed run already uses, whatever their key.

    Looks in the per-run ``data/spider_eos`` directory, then in the shared
    cache directory recorded by :func:`_write_ps_cache_pointer`, for a marker
    with both phase-boundary files. A resumed run continues on these tables
    even when the current key differs, so it does not switch tables part way
    through its evolution. The first time a directory is kept in a process,
    ``current_key()`` is called and a differing key is logged at WARNING with
    both keys, naming the generator identity when only that differs or the
    marker predates it. When ``current_key()`` raises, the tables are still
    kept and a WARNING gives the reason the key was not checked.

    Parameters
    ----------
    outdir : str
        The run output directory.
    current_key : callable
        No-argument callable returning the key the current code would build
        (from :func:`_ps_cache_key`); it raises when that key cannot be built.

    Returns
    -------
    dict or None
        The same keys as :func:`generate_spider_tables`, or None when neither
        location holds a marker with both phase-boundary files.
    """

    candidates = [os.path.join(outdir, 'data', 'spider_eos')]
    pointed = read_ps_cache_pointer(outdir)
    if pointed:
        candidates.append(pointed)
    for eos_dir in candidates:
        marker = os.path.join(eos_dir, '.cache_info.txt')
        solidus_path = os.path.join(eos_dir, 'solidus_P-S.dat')
        liquidus_path = os.path.join(eos_dir, 'liquidus_P-S.dat')
        if not all(os.path.isfile(f) for f in (marker, solidus_path, liquidus_path)):
            continue
        try:
            with open(marker) as f:
                stored = f.read().strip()
        except OSError:
            continue
        if eos_dir not in _PS_RESUME_REPORTED:
            _PS_RESUME_REPORTED.add(eos_dir)
            _report_kept_ps_tables(eos_dir, stored, current_key)
        return {
            'eos_dir': eos_dir,
            'solidus_path': solidus_path,
            'liquidus_path': liquidus_path,
        }
    return None


def _report_kept_ps_tables(eos_dir: str, stored: str, current_key) -> None:
    """Log at WARNING how kept P-S tables differ from the current key, if they do."""
    try:
        cache_key = current_key()
    except Exception as exc:
        log.warning(
            'Resumed run keeps its original energetics P-S entropy tables in %s '
            '(stored key %s); the current key is not checked: %s',
            eos_dir,
            stored,
            exc,
        )
        return
    if stored == cache_key:
        return
    want_base, _, want_gen = cache_key.partition('_gen=')
    base, has_gen, gen = stored.partition('_gen=')
    if base == want_base:
        change = 'the new table generator (generator %s, current %s)' % (
            gen if has_gen else 'unknown',
            want_gen,
        )
    else:
        change = 'the changed settings'
    log.warning(
        'Resumed run keeps its original energetics P-S entropy tables in %s and '
        'ignores %s: stored key %s, current key %s. The structure solve uses the '
        'current melting curves.',
        eos_dir,
        change,
        stored,
        cache_key,
    )


def _publish_ps_tables(src_dir: str, dest_dir: str) -> None:
    """Move every file from a staging dir into the shared cache dir atomically.

    Each entry is relocated with :func:`os.replace`, an atomic rename within
    one filesystem, so a concurrent reader of `dest_dir` sees each table
    either absent or complete, never half-written. Entries are published in
    sorted order for deterministic behaviour; the caller writes the cache
    marker only after this returns so the marker's presence implies every
    table is in place.

    Parameters
    ----------
    src_dir : str
        Staging directory holding freshly generated tables.
    dest_dir : str
        Shared cache directory to publish into; must exist.
    """

    for name in sorted(os.listdir(src_dir)):
        os.replace(os.path.join(src_dir, name), os.path.join(dest_dir, name))


def _atomic_write_text(path: str, text: str) -> None:
    """Write `text` to `path` so readers never observe a partial file.

    The content is written to a uniquely named temporary file in the same
    directory and then moved into place with :func:`os.replace`, an atomic
    rename within one filesystem. Concurrent runs sharing the destination
    therefore either see the previous file or a complete new one, never a
    truncated write.

    Parameters
    ----------
    path : str
        Destination file path.
    text : str
        Content to write.
    """

    dest_dir = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix='.tmp-', dir=dest_dir)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _ps_cache_key(
    *,
    P_max: float,
    nP: int,
    nS: int,
    mzf: float,
    layout: str,
    mantle_eos: str,
    eos_file: str | None,
    solid_eos: str | None,
    liquid_eos: str | None,
    generator: str | None = None,
) -> str:
    """Build the identity string for a generated P-S EOS table set.

    The key is stored in the table directory's ``.cache_info.txt`` marker and,
    under ``PROTEUS_PS_CACHE_DIR``, sanitised into the cache subdirectory name.
    It must encode every input that changes the table contents so that a shared
    cache never serves tables built from a different mantle EOS. The resolved
    EOS file paths are folded in through a short digest; two configs that share
    ``P_max``/``nP``/``nS``/``mzf``/``layout`` but resolve to different EOS
    files therefore land on distinct keys and cannot cross-reuse tables.

    Parameters
    ----------
    P_max : float
        Upper pressure bound of the lookup grid, in Pa.
    nP, nS : int
        Pressure and entropy grid resolutions.
    mzf : float
        Mushy-zone factor relating the synthetic solidus to the liquidus.
    layout : str
        ``'unified'`` or ``'2phase'`` PALEOS table layout.
    mantle_eos : str
        Registry name of the mantle EOS, kept as a readable label in the key.
    eos_file, solid_eos, liquid_eos : str or None
        Resolved paths of the tables that seed the generation. These
        distinguish EOS that share a registry name but resolve to different
        files (e.g. distinct PALEOS-API table versions).
    generator : str or None
        Identity of the Zalmoxis table generator; defaults to
        :func:`_ps_generator_identity`, so tables built by a different
        Zalmoxis release or source land on a distinct key.

    Returns
    -------
    str
        Deterministic cache-identity string.
    """
    eos_identity = '|'.join(str(p) for p in (mantle_eos, eos_file, solid_eos, liquid_eos))
    eos_digest = hashlib.sha1(eos_identity.encode()).hexdigest()[:12]
    eos_name = re.sub(r'[^A-Za-z0-9]+', '-', str(mantle_eos)).strip('-')
    if generator is None:
        generator = _ps_generator_identity()
    gen_name = re.sub(r'[^A-Za-z0-9]+', '-', str(generator)).strip('-')
    return (
        f'P_max={P_max:.6e}_nP={nP}_nS={nS}_mzf={mzf}'
        f'_layout={layout}_eos={eos_name}-{eos_digest}_gen={gen_name}'
    )


@functools.lru_cache(maxsize=1)
def _ps_generator_identity() -> str:
    """Identity of the Zalmoxis code that generates the P-S tables.

    The installed Zalmoxis version plus a digest of the table generator
    (``zalmoxis.eos_export``) and the melting curves it reads
    (``zalmoxis.melting_curves``). The digest covers editable installs, whose
    version metadata is fixed at install time while the source moves on.

    Returns
    -------
    str
        ``'<version>-<digest>'``, or ``'<version>'`` when a module source file
        cannot be read.
    """
    import zalmoxis
    import zalmoxis.eos_export
    import zalmoxis.melting_curves

    version = str(getattr(zalmoxis, '__version__', 'unknown'))
    h = hashlib.sha1()
    try:
        for mod in (zalmoxis.eos_export, zalmoxis.melting_curves):
            h.update(Path(mod.__file__).read_bytes())
    except (OSError, TypeError):
        return version
    return f'{version}-{h.hexdigest()[:12]}'


class _NoPSTables(ValueError):
    """The mantle EOS gives no PALEOS P-S tables; ``level`` is the log level of the reason."""

    def __init__(self, reason: str, level: int = logging.WARNING):
        super().__init__(reason)
        self.level = level


def _ps_table_inputs(config: Config, mantle_eos: str, eos_entry: dict, mat_dicts: dict):
    """Resolve the PALEOS files and the cache key of the P-S tables.

    Materialises PALEOS-API entries, which can build their tables on a cold
    cache. Logs nothing about a missing or non-PALEOS EOS; it raises instead.

    Parameters
    ----------
    config : Config
        Configuration object with struct.zalmoxis settings.
    mantle_eos : str
        Energetics key of the mantle EOS (:func:`energetics_eos_key`).
    eos_entry : dict
        Registry entry of that key.
    mat_dicts : dict
        Zalmoxis material dictionaries.

    Returns
    -------
    tuple
        ``(eos_file, solid_eos, liquid_eos, is_twophase, P_max, cache_key)``.

    Raises
    ------
    _NoPSTables
        When the mantle EOS is not PALEOS or its files are missing.
    """
    # PALEOS-API live tabulation: materialise cached .dat paths in place so the
    # downstream format / eos_file lookups see concrete paths. No-op for
    # non-PALEOS-API entries. First call on a cold cache triggers generation.
    from zalmoxis.eos.dispatch import _is_paleos_api

    if _is_paleos_api(eos_entry):
        from zalmoxis.eos.paleos_api_cache import resolve_registry_entry

        log.info(
            'PALEOS-API live tabulation: resolving cached tables for %s '
            '(cold-cache build may take up to ~1 h at 600 pts/decade)',
            mantle_eos,
        )
        resolve_registry_entry(eos_entry)

    # Detect format: paleos_unified vs PALEOS-2phase (nested dict).
    is_unified = eos_entry.get('format') == 'paleos_unified'
    is_twophase = (
        'melted_mantle' in eos_entry
        and 'solid_mantle' in eos_entry
        and isinstance(eos_entry.get('melted_mantle'), dict)
        and isinstance(eos_entry.get('solid_mantle'), dict)
    )

    if not (is_unified or is_twophase):
        raise _NoPSTables(
            f'mantle EOS {mantle_eos} is neither PALEOS unified nor PALEOS-2phase',
            logging.INFO,
        )

    # Resolve unified file (if present) and 2-phase files (if present).
    eos_file = eos_entry.get('eos_file', '')
    eos_file = eos_file if eos_file and os.path.isfile(eos_file) else None

    if is_twophase:
        solid_eos = eos_entry['solid_mantle'].get('eos_file', '')
        liquid_eos = eos_entry['melted_mantle'].get('eos_file', '')
    else:
        # Unified mantle: the property surfaces come from the 2-phase pair of the
        # same family (API or shipped), which generate_spider_tables requires.
        solid_eos, liquid_eos = resolve_2phase_mgsio3_paths(mantle_eos, mat_dicts)

    solid_eos = solid_eos if solid_eos and os.path.isfile(solid_eos) else None
    liquid_eos = liquid_eos if liquid_eos and os.path.isfile(liquid_eos) else None

    # For the 2-phase path, the downstream Zalmoxis functions still require
    # an `eos_file` positional (used only to seed default property
    # interpolators that are immediately overridden by the 2-phase ones).
    # Pass the solid table as a sentinel: any valid PALEOS table works.
    if eos_file is None:
        if solid_eos is not None:
            eos_file = solid_eos
        else:
            raise _NoPSTables(
                f'no PALEOS EOS file is available for {mantle_eos} '
                '(unified missing and 2-phase incomplete)'
            )

    if is_twophase and not (solid_eos and liquid_eos):
        raise _NoPSTables(
            f'PALEOS-2phase entry {mantle_eos} is missing its solid or liquid file'
        )

    # Determine pressure range from planet mass (higher mass needs wider range)
    mass_tot = config.planet.mass_tot or 1.0
    # P_max for the SPIDER P-S lookup grid. Must cover the actual P_cmb
    # of the planet; the 10 TPa cap covers very massive rocky planets
    # (mass_tot well above 2) without hitting the table edge. See
    # interior_energetics/aragog.py for the matching cap and the
    # comment on EOS / melting-curve calibration ranges.
    P_max = min(1.0e13, 150e9 * mass_tot + 200e9)

    # Table resolution from config
    nP = config.interior_struct.zalmoxis.lookup_nP
    nS = config.interior_struct.zalmoxis.lookup_nS

    mzf = config.interior_struct.zalmoxis.mushy_zone_factor
    layout = '2phase' if is_twophase else 'unified'
    cache_key = _ps_cache_key(
        P_max=P_max,
        nP=nP,
        nS=nS,
        mzf=mzf,
        layout=layout,
        mantle_eos=mantle_eos,
        eos_file=eos_file,
        solid_eos=solid_eos,
        liquid_eos=liquid_eos,
    )

    return eos_file, solid_eos, liquid_eos, is_twophase, P_max, cache_key


def _ps_resume_key(
    config: Config, key: str | None, eos_entry: dict | None, mat_dicts: dict
) -> str:
    """Current P-S cache key for the resume warning, without building PALEOS-API tables."""
    from zalmoxis.eos.dispatch import _is_paleos_api

    if not eos_entry:
        mantle_eos = config.interior_struct.zalmoxis.mantle_eos
        raise _NoPSTables(f'mantle EOS {mantle_eos} is not in the material dictionary')
    if _is_paleos_api(eos_entry):
        raise ValueError('a PALEOS-API mantle EOS is not resolved on resume')
    return _ps_table_inputs(config, key, eos_entry, mat_dicts)[-1]


def generate_spider_tables(config: Config, outdir: str):
    """Generate P-S EOS tables and phase boundaries from PALEOS data.

    Produces P-S lookup tables for density, temperature, heat capacity,
    thermal expansion, and adiabatic gradient, plus solidus/liquidus phase
    boundaries in S(P) format. These are consumed by the entropy-IC verify
    in Aragog (and by SPIDER if the structure module is SPIDER).

    Supports two PALEOS layouts:

    1. ``paleos_unified`` (e.g. ``PALEOS:MgSiO3``): the structural backbone is
       the single unified P-T table covering both phases plus mushy zone, while
       the per-phase property surfaces are built from the sibling two-phase
       solid + liquid tables, so the densities stay resolved across the melting-curve
       discontinuity. The solidus is derived from ``mushy_zone_factor *
       liquidus`` (default 0.8, the constant Stixrude 2014 solidus/liquidus
       ratio applied to the PALEOS liquidus); the liquidus is the analytic
       PALEOS Belonoshko+2005 / Fei+2021 curve.
    2. ``PALEOS-2phase:<solid>`` (e.g. ``PALEOS-2phase:MgSiO3``): separate
       solid + liquid PALEOS tables. The solidus is derived exactly as in the
       unified layout, ``mushy_zone_factor * liquidus`` (default 0.8); the
       two-phase tables additionally supply the latent-heat entropy gap
       between the liquid-table entropy at the liquidus and the solid-table
       entropy at the derived solidus.

    For non-PALEOS EOS types (WolfBower2018, RTPress100TPa), returns None
    and the caller is expected to fall back on pre-existing SPIDER tables.
    A PALEOS mantle with a missing table raises instead.

    Parameters
    ----------
    config : Config
        Configuration object with struct.zalmoxis settings.
    outdir : str
        Output directory. Tables are written to ``outdir/data/spider_eos/``, or,
        when the ``PROTEUS_PS_CACHE_DIR`` environment variable is set, to a
        subdirectory of it named after the sanitised :func:`_ps_cache_key` string, which
        independent runs with the same key share.

    Returns
    -------
    dict or None
        Keys ``'eos_dir'``, ``'solidus_path'``, ``'liquidus_path'`` with
        absolute paths. Returns None if the mantle EOS is not PALEOS.

    Raises
    ------
    ZalmoxisMissingEOSFilesError
        When a table of a PALEOS mantle or its MgSiO3 2-phase pair is missing.
    """
    from zalmoxis.eos_export import generate_spider_eos_tables, generate_spider_phase_boundaries
    from zalmoxis.melting_curves import (
        derive_solidus_from_liquidus,
        get_solidus_liquidus_functions,
    )

    mantle_eos = config.interior_struct.zalmoxis.mantle_eos

    # Use FWL_DATA paths (not ZALMOXIS_ROOT) for EOS file lookup
    mat_dicts = load_zalmoxis_material_dictionaries()
    key, eos_entry = energetics_entry(mantle_eos, mat_dicts)

    # A resumed run stays on the tables it started with; the key only feeds the warning.
    if config.params.resume:
        resumed = _resumed_ps_tables(
            outdir, lambda: _ps_resume_key(config, key, eos_entry, mat_dicts)
        )
        if resumed is not None:
            return resumed

    if not eos_entry or key not in PALEOS_REGISTRY_KEYS:
        log.info(
            'Mantle EOS %s has no PALEOS table set; using pre-existing SPIDER tables.',
            mantle_eos,
        )
        return None
    if paleos_companion_keys(mantle_eos):
        check_zalmoxis_eos_files({'mantle': mantle_eos}, mat_dicts, paleos_companions=True)
    try:
        inputs = _ps_table_inputs(config, key, eos_entry, mat_dicts)
    except _NoPSTables as exc:
        log.log(exc.level, 'No PALEOS P-S tables: %s; using pre-existing SPIDER tables.', exc)
        return None
    eos_file, solid_eos, liquid_eos, is_twophase, P_max, cache_key = inputs
    if solid_eos and liquid_eos:
        log.info('Using PALEOS-2phase tables for entropy-IC table generation')
    nP = config.interior_struct.zalmoxis.lookup_nP
    nS = config.interior_struct.zalmoxis.lookup_nS
    if config.params.resume:
        log.warning(
            'Resumed run has no kept P-S entropy tables in %s or at its shared-cache '
            'pointer; it continues on the tables of the current key %s, built now if absent',
            os.path.join(outdir, 'data', 'spider_eos'),
            cache_key,
        )

    mzf = config.interior_struct.zalmoxis.mushy_zone_factor
    # Phase boundaries: PALEOS-liquidus is the analytic Belonoshko+2005 /
    # Fei+2021 Simon-Glatzel curve. The derived solidus is
    # T_solidus = T_liquidus * mushy_zone_factor for both layouts. A two-phase
    # table adds the latent-heat gap between liquid_table.s(P, T_liq) and
    # solid_table.s(P, T_sol) with T_sol = mushy_zone_factor * T_liq.
    _, liquidus_func = get_solidus_liquidus_functions(
        solidus_id='Stixrude14-solidus',  # unused, but API requires it
        liquidus_id='PALEOS-liquidus',
    )
    solidus_func = derive_solidus_from_liquidus(liquidus_func, mzf)
    if is_twophase:
        # This solidus_func also reaches Zalmoxis's own 2-phase structure
        # solve via load_zalmoxis_solidus_liquidus_functions, so mzf moves
        # nabla_ad there too, separately from the density blend below.
        log.info(
            'PALEOS-2phase phase boundaries: solidus = liquidus * %.2f '
            '(mushy_zone_factor); latent heat from 2-phase tables',
            mzf,
        )
    else:
        log.info(
            'PALEOS unified phase boundaries: solidus = liquidus * %.2f (mushy_zone_factor)',
            mzf,
        )

    # Table location: output/<run>/data/spider_eos, or with PROTEUS_PS_CACHE_DIR a
    # shared directory keyed by cache_key (fields in _ps_cache_key), so runs with
    # the same key reuse one slow full-resolution PALEOS P-S table.
    _ps_cache_root = os.environ.get('PROTEUS_PS_CACHE_DIR')
    if _ps_cache_root:
        _safe_key = cache_key.replace('.', 'p').replace('=', '-').replace('+', '')
        spider_eos_dir = os.path.join(_ps_cache_root, _safe_key)
        os.makedirs(spider_eos_dir, exist_ok=True)
        # The shared-cache tables live outside the run directory, so a resumed
        # run cannot find them via the per-run output/<run>/data/spider_eos
        # path. Leave a pointer so resume can follow the tables to the cache.
        _write_ps_cache_pointer(outdir, spider_eos_dir)
    else:
        spider_eos_dir = os.path.join(outdir, 'data', 'spider_eos')

    # Cache check: skip regeneration if tables exist and pressure range unchanged.
    # The pressure range depends on planet mass, which doesn't change during evolution.
    cache_marker = os.path.join(spider_eos_dir, '.cache_info.txt')
    if os.path.isfile(cache_marker):
        with open(cache_marker) as f:
            existing_key = f.read().strip()
        if existing_key == cache_key:
            # Tables are up to date. File names must match the writer in
            # zalmoxis.eos_export.generate_spider_phase_boundaries, which
            # emits solidus_P-S.dat / liquidus_P-S.dat.
            solidus_path = os.path.join(spider_eos_dir, 'solidus_P-S.dat')
            liquidus_path = os.path.join(spider_eos_dir, 'liquidus_P-S.dat')
            if os.path.isfile(solidus_path) and os.path.isfile(liquidus_path):
                log.info(
                    'Reusing cached PALEOS-derived P-S entropy tables (P_max=%.2e, %dx%d)',
                    P_max,
                    nP,
                    nS,
                )
                return {
                    'eos_dir': spider_eos_dir,
                    'solidus_path': solidus_path,
                    'liquidus_path': liquidus_path,
                }
        else:
            log.info(
                'Regenerating P-S entropy tables in %s: cache key %s does not match %s',
                spider_eos_dir,
                existing_key,
                cache_key,
            )

    # Choose where to generate. For a shared PROTEUS_PS_CACHE_DIR the tables are
    # written into a private staging directory on the same filesystem and then
    # published into spider_eos_dir with per-file atomic renames, so two cluster
    # runs that miss the cache marker at the same moment cannot interleave
    # partial writes into the shared table directory. For a per-run directory
    # there is no sharing, so tables are generated in place as before.
    if _ps_cache_root:
        gen_dir = tempfile.mkdtemp(prefix='.gen-', dir=os.path.dirname(spider_eos_dir))
    else:
        gen_dir = spider_eos_dir

    try:
        # Generate phase boundaries
        log.info(
            'Generating PALEOS-derived P-S phase boundaries (%d P points)...',
            nP,
        )
        generate_spider_phase_boundaries(
            solidus_func=solidus_func,
            liquidus_func=liquidus_func,
            eos_file=eos_file,
            P_range=(1e5, P_max),
            n_P=nP,
            output_dir=gen_dir,
            solid_eos_file=solid_eos,
            liquid_eos_file=liquid_eos,
        )

        # Generate full EOS tables
        log.info(
            'Generating PALEOS-derived P-S EOS tables (%d x %d)...',
            nP,
            nS,
        )
        generate_spider_eos_tables(
            eos_file=eos_file,
            solidus_func=solidus_func,
            liquidus_func=liquidus_func,
            P_range=(1e5, P_max),
            n_P=nP,
            n_S=nS,
            output_dir=gen_dir,
            solid_eos_file=solid_eos,
            liquid_eos_file=liquid_eos,
        )

        # Publish staged tables into the shared cache with atomic renames.
        if gen_dir != spider_eos_dir:
            _publish_ps_tables(gen_dir, spider_eos_dir)
    finally:
        if gen_dir != spider_eos_dir:
            shutil.rmtree(gen_dir, ignore_errors=True)

    # Write the cache marker last, atomically, so a concurrent reader only
    # trusts the directory once every table file is already in place.
    try:
        _atomic_write_text(cache_marker, cache_key)
    except OSError:
        pass

    # File names match the writer in zalmoxis.eos_export, which emits
    # solidus_P-S.dat / liquidus_P-S.dat; use the published locations.
    return {
        'eos_dir': spider_eos_dir,
        'solidus_path': os.path.join(spider_eos_dir, 'solidus_P-S.dat'),
        'liquidus_path': os.path.join(spider_eos_dir, 'liquidus_P-S.dat'),
    }


def compute_structure_mass_desync(radii, density, mass_enclosed) -> float:
    """Relative divergence between the density-profile mass integral and the
    structure ODE accumulator total.

    The Zalmoxis structure ODE accumulates the enclosed mass with an RK45
    integrator (``mass_enclosed[-1]``). A direct trapezoid of the shell mass
    ``4 pi r^2 rho`` over the converged ``(radii, density)`` profile is an
    independent estimate of the same total. The two agree only when the
    density Picard iteration and the structure ODE are fully co-converged;
    their relative divergence is the mass self-consistency diagnostic tracked
    for issue #68.

    Parameters
    ----------
    radii : array_like
        Radial node positions from the converged structure solve [m].
    density : array_like
        Density at each radial node [kg m-3].
    mass_enclosed : array_like
        Cumulative enclosed mass from the structure ODE; the last element is
        the accumulator total [kg].

    Returns
    -------
    float
        ``|trapezoid - accumulator| / accumulator``, or 0.0 when the
        accumulator total is non-finite or non-positive (degenerate or empty
        profile), which keeps the metric defined for a failed structure solve.
    """
    accumulator_total = float(mass_enclosed[-1])
    if not np.isfinite(accumulator_total) or accumulator_total <= 0.0:
        return 0.0
    r = np.asarray(radii, dtype=float)
    rho = np.asarray(density, dtype=float)
    shell_mass_trapezoid = float(np.trapezoid(4.0 * np.pi * r**2 * rho, r))
    return abs(shell_mass_trapezoid - accumulator_total) / accumulator_total


def zalmoxis_solver(
    config: Config,
    outdir: str,
    hf_row: dict,
    num_spider_nodes: int = 0,
    temperature_function=None,
    temperature_mode_override: str | None = None,
    temperature_arrays=None,
):
    """Run the Zalmoxis solver to compute the interior structure of a planet.

    Parameters
    ----------
    config : Config
        Configuration object.
    outdir : str
        Output directory where results will be saved.
    hf_row : dict
        Dictionary containing volatile masses and other parameters.
    num_spider_nodes : int
        Number of SPIDER basic nodes. If > 0, writes a SPIDER mesh file
        and returns its path as the second element of the return tuple.
    temperature_function : callable or None, optional
        External temperature function ``f(r, P) -> T`` in (m, Pa, K).
        When provided, bypasses Zalmoxis's internal temperature mode
        dispatch. Used to pass SPIDER/Aragog T(r) profiles in memory.
    temperature_arrays : tuple[ndarray, ndarray] or None, optional
        Explicit r-indexed ``(r_arr, T_arr)`` for the Zalmoxis JAX path.
        Consumed only when ``use_jax=True``, the configured EOS pair
        can take the Zalmoxis JAX dispatch (a unified or 2-phase PALEOS
        mantle plus a unified PALEOS core; see
        :func:`_zalmoxis_jax_structure_viable`), and the solve is either
        dry (no ``VolatileProfile`` in play) or wet with a profile
        inside the Zalmoxis JAX wet envelope (see
        :func:`_volatile_profile_jax_viable`). In that case the
        external callable is withheld so the inner Picard converges on
        Zalmoxis' internal linear-T profile while the JAX RHS integrates
        against the arrays. For any other configuration the solve
        runs on the numpy path, which consumes ``temperature_function``,
        and the callable is therefore passed through. See Zalmoxis'
        ``solve_structure_via_jax`` docstring for why both kwargs can be
        passed together.
    temperature_mode_override : str or None, optional
        Local override for ``config.planet.temperature_mode``. Lets callers
        force a different structure-solve mode without mutating the shared
        Config object (see proteus rules §"Config mutability"). When None,
        the Config value is used. Currently used by
        ``determine_interior_radius_with_zalmoxis`` to force ``adiabatic``
        for SPIDER coupling with a T-dependent mantle EOS.

    Returns
    -------
    cmb_radius : float
        Core-mantle boundary radius [m].
    spider_mesh_file : str or None
        Path to the SPIDER mesh file, or None if ``num_spider_nodes == 0``.
    """

    # Load the Zalmoxis configuration parameters. Flag an external temperature
    # source so 'liquidus_super' skips re-solving the super-liquidus CMB anchor
    # on this call: an evolved T(r) (or the IC adiabat callable) drives the
    # structure here, so the anchor is discarded and reusing the last solved
    # value avoids the scan-and-bisection on every re-solve.
    external_temperature_source = (
        temperature_function is not None or temperature_arrays is not None
    )
    config_params = load_zalmoxis_configuration(
        config,
        hf_row,
        temperature_mode_override=temperature_mode_override,
        external_temperature_source=external_temperature_source,
    )

    # Build volatile profile from dissolved volatile masses (if available).
    # This enables phi(r)-weighted volatile blending inside the Zalmoxis ODE.
    # Skipped when dry_mantle=True: the structure solver then uses only
    # the canonical mantle EOS tables.
    mantle_eos = config.interior_struct.zalmoxis.mantle_eos
    if config.interior_struct.zalmoxis.dry_mantle:
        volatile_profile = None
        log.debug(
            'Structure solver: dry_mantle=True, skipping VolatileProfile '
            '(mantle EOS uses %s tables only).',
            mantle_eos,
        )
    else:
        volatile_profile = build_volatile_profile(hf_row, mantle_eos)

    # Configure global miscibility if enabled. The config gate rejects
    # global_miscibility at load time until the Zalmoxis pin supports
    # per-shell volatile profiles, so this branch is exercised only via
    # test doubles; it stays in place for when the gate lifts.
    if config.interior_struct.zalmoxis.global_miscibility and volatile_profile is not None:
        volatile_profile.global_miscibility = True
        # Initialize x_interior from current dissolved masses
        M_mantle = float(hf_row.get('M_mantle', 0.0))
        if M_mantle > 0:
            H2_kg_liquid = float(hf_row.get('H2_kg_liquid', 0.0))
            if H2_kg_liquid > 0:
                volatile_profile.x_interior['Chabrier:H'] = H2_kg_liquid / (
                    M_mantle + H2_kg_liquid
                )
            H2O_kg_liquid = float(hf_row.get('H2O_kg_liquid', 0.0))
            if H2O_kg_liquid > 0:
                volatile_profile.x_interior['PALEOS:H2O'] = H2O_kg_liquid / (
                    M_mantle + H2O_kg_liquid
                )

    # Extend mantle EOS string with volatile components so the LayerMixture
    # includes them (VolatileProfile overrides fractions at each radius).
    # Rebuild mushy_zone_factors afterward so a dissolved component (e.g.
    # PALEOS:H2O, Chabrier:H) gets the real mzf instead of the 1.0 default
    # from the dry EOS string load_zalmoxis_configuration saw.
    if volatile_profile is not None:
        config_params['layer_eos_config']['mantle'] = extend_mantle_eos_with_volatiles(
            config_params['layer_eos_config']['mantle'], volatile_profile
        )
        config_params['mushy_zone_factors'] = _build_mushy_zone_factors(
            config_params['layer_eos_config'],
            config.interior_struct.zalmoxis.mushy_zone_factor,
        )

    # Get the output location for Zalmoxis output and create the file if it does not exist
    output_zalmoxis = get_zalmoxis_output_filepath(outdir)
    open(output_zalmoxis, 'a').close()

    # JAX-path wall_timeout: Zalmoxis' default is 300 s, which is a
    # sanity cap. The bench (``bench_performance.py``) and the JAX
    # parity fixture override it to 3600 s because the first JAX call
    # on a cold JIT can incur compilation time on top of the solve.
    # Mirror that here when the caller opted into JAX, so a cold first
    # call does not fall into the best-solution branch and trip the
    # downstream array-write path.
    if config_params.get('use_jax') and 'wall_timeout' not in config_params:
        config_params['wall_timeout'] = 3600.0

    # JAX structure path gate: the JAX wrapper's P-indexed adiabat
    # tabulation collapses for P-ignoring callables (see
    # tools/benchmarks/bench_coupled_tempfunc.py).
    # The fix is to pass ``temperature_arrays=(r_arr, T_arr)`` instead,
    # which routes the RHS to the r-indexed branch. We have arrays from
    # ``update_structure_from_interior`` but NOT from PROTEUS init or
    # equilibration (Zalmoxis constructs its own internal linear/adiabat
    # guess for those, and that guess also ignores P). For calls with
    # neither arrays nor a caller-provided callable, keep the defensive
    # downgrade to the numpy path; the one-time init/equilibration cost
    # (~70 s each, ~2-4 calls) is negligible against a 3-4 h full run.
    if temperature_function is None and temperature_arrays is None:
        if config_params.get('use_jax') or config_params.get('use_anderson'):
            log.debug(
                'Zalmoxis call has no temperature_function or '
                'temperature_arrays: disabling use_jax and use_anderson '
                'for this call (the internal T-dispatch path collapses '
                'for P-ignoring callables).'
            )
            config_params['use_jax'] = False
            config_params['use_anderson'] = False

    # Run structure solve: use miscibility wrapper when enabled
    mat_dicts = load_zalmoxis_material_dictionaries()
    check_zalmoxis_eos_files(config_params['layer_eos_config'], mat_dicts)
    melt_funcs = load_zalmoxis_solidus_liquidus_functions(mantle_eos, config)
    input_data_dir = os.path.join(outdir, 'data')

    # Temperature-source dispatch for this call. temperature_arrays can be
    # consumed only by the Zalmoxis JAX inner path, which requires a
    # JAX-capable EOS layout (a unified or 2-phase PALEOS mantle plus a
    # unified PALEOS core) and, on wet solves, a VolatileProfile inside
    # the Zalmoxis JAX wet envelope (exactly one paleos_unified volatile
    # blended into the mantle; see _volatile_profile_jax_viable). Only in
    # that case is the external callable withheld: the JAX RHS integrates
    # against the arrays while the numpy Picard helper converges quickly
    # on Zalmoxis' internal linear-T profile (passing the callable there
    # lands Picard near PALEOS phase-boundary clamps and costs roughly
    # two orders of magnitude more wall time at the same JAX arrays). For
    # every other EOS configuration, and for wet solves whose profile
    # falls outside the envelope, the JAX dispatch declines and the numpy
    # ODE runs; that path consumes only the callable, so it must pass
    # through for the solve to follow the evolved T(r) instead of
    # rebuilding the internal temperature_mode profile from the hot
    # initial anchor.
    _use_jax_active = bool(config_params.get('use_jax'))
    _jax_viable = _zalmoxis_jax_structure_viable(
        mat_dicts, config.interior_struct.zalmoxis.core_eos, mantle_eos
    )
    # Surface a JAX->numpy fallback once from the PROTEUS side (see helper).
    if _use_jax_active and not _jax_viable:
        _log_jax_nonviable_once(config.interior_struct.zalmoxis.core_eos, mantle_eos)
    _wet_jax_viable = volatile_profile is None or _volatile_profile_jax_viable(
        volatile_profile, mat_dicts, config_params['layer_eos_config']['mantle']
    )
    if _use_jax_active and _jax_viable and volatile_profile is not None and not _wet_jax_viable:
        log.debug(
            'VolatileProfile falls outside the Zalmoxis JAX wet envelope; '
            'the structure solve stays on the numpy path with the callable.'
        )
    _drop_callable = (
        _use_jax_active and temperature_arrays is not None and _jax_viable and _wet_jax_viable
    )
    _tf_effective = None if _drop_callable else temperature_function
    if _drop_callable:
        log.debug(
            'Structure-solve T source: temperature_arrays via the Zalmoxis '
            'JAX path; external callable withheld for this call.'
        )
    elif temperature_function is not None:
        log.debug(
            'Structure-solve T source: external temperature_function '
            '(consumed by the numpy path).'
        )
    else:
        log.debug(
            'Structure-solve T source: internal %r mode dispatch.',
            config_params.get('temperature_mode'),
        )

    if config.interior_struct.zalmoxis.global_miscibility:
        from zalmoxis.solver import solve_miscible_interior

        # Build H2 mass targets from current volatile inventories
        h2_mass_targets = {}
        H2_kg_total = float(hf_row.get('H2_kg_total', 0.0))
        H2_kg_atm = float(hf_row.get('H2_kg_atm', 0.0))
        H2_kg_dissolved = H2_kg_total - H2_kg_atm
        if H2_kg_dissolved > 0:
            h2_mass_targets['Chabrier:H'] = H2_kg_dissolved

        H2O_kg_liquid = float(hf_row.get('H2O_kg_liquid', 0.0))
        if H2O_kg_liquid > 0:
            h2_mass_targets['PALEOS:H2O'] = H2O_kg_liquid

        model_results = solve_miscible_interior(
            config_params,
            material_dictionaries=mat_dicts,
            melting_curves_functions=melt_funcs,
            input_dir=input_data_dir,
            volatile_profile=volatile_profile,
            temperature_function=_tf_effective,
            temperature_arrays=temperature_arrays,
            h2_mass_targets=h2_mass_targets,
            max_iterations=config.interior_struct.zalmoxis.miscibility_max_iter,
            mass_tolerance=config.interior_struct.zalmoxis.miscibility_tol,
        )

        # Write solvus info to hf_row
        if model_results.get('solvus_radius') is not None:
            hf_row['R_solvus'] = model_results['solvus_radius']
            hf_row['T_solvus'] = model_results['solvus_temperature']
            hf_row['P_solvus'] = model_results['solvus_pressure']
        hf_row['X_H2_int'] = model_results.get('x_interior_converged', {}).get(
            'Chabrier:H', 0.0
        )

        log.info(
            'Global miscibility: solvus R=%.2e m, T=%.0f K, P=%.2e Pa, '
            'X_H2_int=%.4f, converged=%s (%d iters)',
            hf_row.get('R_solvus', 0.0),
            hf_row.get('T_solvus', 0.0),
            hf_row.get('P_solvus', 0.0),
            hf_row.get('X_H2_int', 0.0),
            model_results.get('miscibility_converged', False),
            model_results.get('miscibility_iterations', 0),
        )
    else:
        # _tf_effective carries the temperature-source decision hoisted
        # above: the callable is withheld only when the JAX inner path
        # will actually consume temperature_arrays (JAX-viable EOS
        # layout and, on wet solves, an in-envelope VolatileProfile);
        # on any other configuration the numpy path runs and the
        # callable passes through. Warm-starts
        # stay disabled on the JAX path: they drive Anderson into
        # oscillation and do not help otherwise, because the inner
        # Picard plateau at diff=0.1 is set by the lever-rule EOS kink,
        # not by initial density quality, so warm-start cannot collapse
        # the bail count.
        # Reuse the cached density profile as a Picard seed only when it
        # belongs to this same planet; otherwise start cold. Seeding never
        # changes the converged result, only the iteration count. The seed
        # and the central-pressure hint follow the temperature-source
        # dispatch: they are withheld only on the JAX-arrays path (where
        # warm-starts drive Anderson into oscillation) and applied on the
        # numpy path, where they cut the Newton iteration count for
        # repeated re-solves of the same planet.
        _seed_match = not _drop_callable and _density_cache.get('key') == _structure_cache_key(
            config
        )
        _seed_density = _density_cache['density'] if _seed_match else None
        _seed_radii = _density_cache['radii'] if _seed_match else None
        model_results = main(
            config_params,
            material_dictionaries=mat_dicts,
            melting_curves_functions=melt_funcs,
            input_dir=input_data_dir,
            volatile_profile=volatile_profile,
            temperature_function=_tf_effective,
            temperature_arrays=temperature_arrays,
            p_center_hint=None if _drop_callable else hf_row.get('P_center'),
            initial_density=_seed_density,
            initial_radii=_seed_radii,
        )

    # Extract results from the model
    radii = model_results['radii']
    density = model_results['density']
    gravity = model_results['gravity']
    pressure = model_results['pressure']
    temperature = model_results['temperature']
    mass_enclosed = model_results['mass_enclosed']
    cmb_mass = model_results['cmb_mass']
    core_mantle_mass = model_results['core_mantle_mass']
    converged = model_results['converged']
    converged_pressure = model_results['converged_pressure']
    converged_density = model_results['converged_density']
    converged_mass = model_results['converged_mass']

    # Adaptive retry: if the primary call did not converge, retry once with
    # relaxed tolerances. Retry fires on any non-converged result, including
    # the case where pressure, density, and mass flags are all False (e.g. a
    # timeout while the outer mass loop is still drifting), giving a clean
    # second attempt from fresh initial conditions; the wall_timeout cap
    # still bounds worst-case wall.
    #
    # If the primary's best_mass_error already exceeds 5 % the retry is
    # skipped entirely: at the structural-Picard plateau (7-10 %) looser
    # tolerance does not rescue a structurally-unsupported configuration.
    # Retry remains useful for transient flickers near the tolerance edge.
    primary_best_mass_error = model_results.get('best_mass_error')
    skip_retry_high_error = (
        primary_best_mass_error is not None and primary_best_mass_error > 0.05
    )
    if not converged and skip_retry_high_error:
        log.warning(
            'Zalmoxis primary call did not converge '
            '(pressure=%s, density=%s, mass=%s) AND best_mass_error=%.2e '
            '> 0.05; skipping retry (looser tolerance cannot rescue a '
            'structural-Picard plateau).',
            converged_pressure,
            converged_density,
            converged_mass,
            primary_best_mass_error,
        )
    if not converged and not skip_retry_high_error:
        retry_tol = config_params.get('tolerance_outer', 3e-3) * 3
        retry_iter = int(config_params.get('max_iterations_outer', 100) * 2)
        log.warning(
            'Zalmoxis primary call did not converge '
            '(pressure=%s, density=%s, mass=%s); retrying with relaxed '
            'tolerance (tol_outer=%.1e, max_iter=%d)',
            converged_pressure,
            converged_density,
            converged_mass,
            retry_tol,
            retry_iter,
        )
        config_params_retry = dict(config_params)
        config_params_retry['tolerance_outer'] = retry_tol
        config_params_retry['max_iterations_outer'] = retry_iter
        # Cap retry wall_timeout at 600 s. The JAX-path primary uses
        # 3600 s, which is appropriate for a successful first solve, but
        # the retry is a second-chance attempt under relaxed tolerances;
        # if it has not converged within 10 min of fresh outer iters it
        # is unlikely to succeed at any wall.
        config_params_retry['wall_timeout'] = 600.0

        # The retry copies use_jax unchanged, so the hoisted
        # temperature-source gate (_tf_effective) applies as-is.
        model_results = main(
            config_params_retry,
            material_dictionaries=mat_dicts,
            melting_curves_functions=melt_funcs,
            input_dir=input_data_dir,
            volatile_profile=volatile_profile,
            temperature_function=_tf_effective,
            temperature_arrays=temperature_arrays,
        )

        radii = model_results['radii']
        density = model_results['density']
        gravity = model_results['gravity']
        pressure = model_results['pressure']
        temperature = model_results['temperature']
        mass_enclosed = model_results['mass_enclosed']
        cmb_mass = model_results['cmb_mass']
        core_mantle_mass = model_results['core_mantle_mass']
        converged = model_results['converged']
        converged_pressure = model_results['converged_pressure']
        converged_density = model_results['converged_density']
        converged_mass = model_results['converged_mass']

        if converged:
            log.info('Zalmoxis converged on retry with relaxed tolerances')

    # Check convergence before proceeding. Non-converged solutions
    # (e.g. when EOS table range is exceeded) produce garbage values
    # that would corrupt the simulation state.
    if not converged:
        diag = (
            f'Zalmoxis did not converge: '
            f'pressure={converged_pressure}, density={converged_density}, '
            f'mass={converged_mass}. '
            f'Final M={mass_enclosed[-1]:.2e} kg, R={radii[-1]:.2e} m. '
            f'EOS: core={config.interior_struct.zalmoxis.core_eos}, '
            f'mantle={config.interior_struct.zalmoxis.mantle_eos}.'
        )
        log.error(diag)
        # Dump the exact arguments and the final model_results for offline
        # standalone replay. The pickle lands in <outdir>/data/, gitignored
        # via PROTEUS' default .gitignore. Numbered so multiple failures
        # within one run don't overwrite each other.
        try:
            import pickle
            import time as _ftime

            dump_dir = os.path.join(outdir, 'data')
            os.makedirs(dump_dir, exist_ok=True)
            stamp = int(_ftime.time())
            dump_path = os.path.join(dump_dir, f'zalmoxis_failure_{stamp}.pkl')
            r_arr_d, T_arr_d = (None, None)
            if temperature_arrays is not None:
                r_arr_d = np.asarray(temperature_arrays[0]).copy()
                T_arr_d = np.asarray(temperature_arrays[1]).copy()
            with open(dump_path, 'wb') as _fh:
                pickle.dump(
                    {
                        'config_params': dict(config_params),
                        'temperature_arrays': (r_arr_d, T_arr_d),
                        'hf_row_subset': {
                            k: hf_row[k]
                            for k in (
                                'P_center',
                                'M_int',
                                'R_int',
                                'T_magma',
                                'Phi_global',
                                'T_surf',
                            )
                            if k in hf_row
                        },
                        'model_results_keys': sorted(model_results.keys()),
                        'final_M': float(mass_enclosed[-1]),
                        'final_R': float(radii[-1]),
                        'best_mass_error': model_results.get('best_mass_error'),
                        'flags': {
                            'pressure': bool(converged_pressure),
                            'density': bool(converged_density),
                            'mass': bool(converged_mass),
                        },
                    },
                    _fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            log.warning('Failure args dumped to %s', dump_path)
        except Exception as _dump_exc:
            log.warning('Failed to dump Zalmoxis failure args: %s', _dump_exc)
        raise RuntimeError(diag)

    # Extract the index of the core-mantle boundary mass in the mass array
    cmb_index = np.argmax(mass_enclosed >= cmb_mass)

    # Extract the planet radius and core-mantle boundary radius
    planet_radius = radii[-1]
    cmb_radius = radii[cmb_index]

    # Recompute density and temperature against the accurate T(r) when the
    # solve took the JAX+temperature_arrays path (_drop_callable). On that
    # path `temperature_function` is withheld so the numpy Picard converges
    # quickly using Zalmoxis' internal linear-T fallback. The JAX integrator
    # still uses `temperature_arrays` (Aragog's true T(r)) and produces
    # correct P(r), M(r), g(r), but `model_results` returns `density =
    # EOS(P, T_linear_fallback)` and `temperature = T_linear_fallback`
    # because those come from the Picard helper. Aragog later reads
    # `zalmoxis_output.dat` for mesh construction (`eos_method=2`), so a
    # stale density column would feed the wrong cell masses into its energy
    # evolution (~10% T-driven density error at the CMB in the
    # PALEOS-2phase melt regime). Recompute both columns here from
    # (P, T_aragog) using numpy EOS before any downstream consumer reads
    # them. When the callable was honored on the numpy path instead, the
    # solver's own columns already reflect the evolved T(r) and no rebuild
    # runs. Wet solves rebuild the mantle through the same per-shell blend
    # the numpy solver's density update evaluates (calculate_mixed_density
    # on the extended mantle mixture with the VolatileProfile at the local
    # melt fraction), so the written column keeps the dissolved-volatile
    # contribution instead of collapsing to the bare dry mantle EOS.
    if _drop_callable:
        from zalmoxis.eos.dispatch import calculate_density as _calc_rho

        # Wet path: the extended mantle mixture (primary silicate plus
        # profile-managed volatiles), parsed from the same EOS string the
        # solver parses, blended per shell by the profile.
        _mantle_mixture = None
        if volatile_profile is not None:
            from zalmoxis.mixing import calculate_mixed_density as _calc_rho_mixed
            from zalmoxis.mixing import parse_layer_components as _parse_mixture

            _mantle_mixture = _parse_mixture(config_params['layer_eos_config']['mantle'])

        _r_ref, _T_ref = temperature_arrays
        _T_ref_cmb = float(_T_ref[0])
        _core_eos = config.interior_struct.zalmoxis.core_eos
        _mantle_eos = config.interior_struct.zalmoxis.mantle_eos
        _mzf = config.interior_struct.zalmoxis.mushy_zone_factor
        _sol_f, _liq_f = (
            melt_funcs
            if isinstance(melt_funcs, tuple) and len(melt_funcs) == 2
            else (None, None)
        )
        _interp_cache: dict = {}
        _rho_fixed = np.zeros(len(radii))
        _T_fixed = np.zeros(len(radii))
        for _i in range(len(radii)):
            _r = float(radii[_i])
            _P = float(pressure[_i])
            # r-indexed T: below r_ref[0] (= first Aragog staggered node
            # near CMB), clamp to T_cmb; else interp along the Aragog grid.
            if _r <= float(_r_ref[0]):
                _T = _T_ref_cmb
            else:
                _T = float(np.interp(_r, _r_ref, _T_ref))
            _T_fixed[_i] = _T
            if _mantle_mixture is not None and _i >= cmb_index:
                # Wet mantle node: same evaluation as the numpy solver's
                # density update. calculate_mixed_density computes the
                # local melt fraction from the melting curves at this
                # node's (P, T) (lever rule in compute_melt_fraction) and
                # takes the per-component fractions from
                # volatile_profile.apply_to_mixture. The condensed and
                # binodal sigmoid parameters stay at the zalmoxis.mixing
                # defaults, the same values the solver resolves when
                # config_params carries no overrides (PROTEUS sets none).
                _rho = _calc_rho_mixed(
                    _P,
                    _T,
                    _mantle_mixture,
                    mat_dicts,
                    _sol_f,
                    _liq_f,
                    _interp_cache,
                    mushy_zone_factors=config_params.get('mushy_zone_factors'),
                    volatile_profile=volatile_profile,
                )
                if _rho is not None and not np.isfinite(_rho):
                    _rho = None
            else:
                _eos_here = _core_eos if _i < cmb_index else _mantle_eos
                _rho = _calc_rho(
                    _P,
                    mat_dicts,
                    _eos_here,
                    _T,
                    _sol_f,
                    _liq_f,
                    interpolation_functions=_interp_cache,
                    mushy_zone_factor=_mzf,
                )
            _rho_fixed[_i] = float(_rho) if _rho is not None else float(density[_i])
        density = _rho_fixed
        temperature = _T_fixed
        log.debug(
            'Rebuilt density/temperature against T_aragog (JAX path). '
            'density: CMB=%.1f kg/m^3, surface=%.1f kg/m^3. '
            'T: CMB=%.1f K, surface=%.1f K.',
            float(density[cmb_index]),
            float(density[-1]),
            float(temperature[cmb_index]),
            float(temperature[-1]),
        )

    # Calculate the average density of the planet using the calculated mass and radius
    average_density = mass_enclosed[-1] / (4 / 3 * np.pi * radii[-1] ** 3)

    # Mass self-consistency diagnostic (issue #68): record how far the density
    # profile's trapezoid mass integral diverges from the structure ODE
    # accumulator total, so the helpfile carries the desync as a quantified
    # per-run metric rather than only the pass/fail output-schema guard.
    hf_row['struct_mass_desync_frac'] = compute_structure_mass_desync(
        radii, density, mass_enclosed
    )
    log.debug(
        'Structure mass self-consistency desync: %.2e',
        hf_row['struct_mass_desync_frac'],
    )

    # Cache density for next call's Picard seeding. Used by both numpy
    # and JAX paths when use_anderson=False (Anderson + warm-start
    # oscillates, see the warm-start gate above).
    _density_cache['density'] = density.copy()
    _density_cache['radii'] = np.asarray(radii).copy()
    _density_cache['key'] = _structure_cache_key(config)

    # Final results of the Zalmoxis interior model. One summary line at INFO so
    # a long coupled run stays readable across hundreds of re-solves; the
    # per-field breakdown drops to debug for when a single solve is inspected.
    log.info(
        'Zalmoxis interior solved: R=%.3f R_earth, dry M=%.3f M_earth, '
        'core R_frac=%.4f, converged=%s (P=%s, rho=%s, M=%s)',
        planet_radius / R_earth,
        mass_enclosed[-1] / M_earth,
        cmb_radius / planet_radius,
        converged,
        converged_pressure,
        converged_density,
        converged_mass,
    )
    log.debug(
        f'Interior (dry calculated mass) mass: {mass_enclosed[-1]} kg or approximately {mass_enclosed[-1] / M_earth:.2f} M_earth'
    )
    log.debug(
        f'Interior radius: {planet_radius:.2e} m or {planet_radius / R_earth:.2f} R_earth'
    )
    log.debug(f'Core radius: {cmb_radius:.2e} or {cmb_radius / R_earth:.2f} R_earth')
    log.debug(f'Core-mantle boundary mass: {mass_enclosed[cmb_index]:.2e} kg')
    log.debug(f'Mantle density at the core-mantle boundary: {density[cmb_index]:.2e} kg/m^3')
    log.debug(f'Core density at the core-mantle boundary: {density[cmb_index - 1]:.2e} kg/m^3')
    log.debug(f'Pressure at the core-mantle boundary: {pressure[cmb_index]:.2e} Pa')
    log.debug(f'Pressure at the center: {pressure[0]:.2e} Pa')
    log.debug(f'Average density: {average_density:.2e} kg/m^3')
    log.debug(
        f'Core-mantle boundary mass fraction: {mass_enclosed[cmb_index] / mass_enclosed[-1]:.3f}'
    )
    log.debug(f'Core radius fraction: {cmb_radius / planet_radius:.4f}')
    log.debug(
        f'Inner mantle radius fraction: {radii[np.argmax(mass_enclosed >= core_mantle_mass)] / planet_radius:.4f}'
    )
    log.debug(
        f'Overall Convergence Status: {converged} with Pressure: {converged_pressure}, Density: {converged_density}, Mass: {converged_mass}'
    )

    # Self-consistent initial thermal state (White+Li 2025, Boujibar+2020).
    # Honor temperature_mode_override here as well: the accretion thermal
    # state only runs when the user actually wants accretion mode, not
    # when SPIDER has forced a local adiabatic override.
    _effective_temp_mode = temperature_mode_override or config.planet.temperature_mode
    if _effective_temp_mode == 'accretion':
        from zalmoxis.energetics import initial_thermal_state

        cmf = config.interior_struct.core_frac
        mantle_eos = config.interior_struct.zalmoxis.mantle_eos

        # Build PALEOS-derived nabla_ad and C_p when PALEOS EOS is configured.
        # This uses the actual EOS tables for the adiabatic gradient and heat
        # capacities instead of the constant defaults (Gruneisen adiabat,
        # Dulong-Petit C_Fe=450, C_sil=1250 J/kg/K from White+Li 2025).
        nabla_ad_func = None
        cp_iron_func = None
        cp_silicate_func = None
        C_iron = 450.0
        C_silicate = 1250.0

        if 'PALEOS' in mantle_eos:
            try:
                import math

                from scipy.interpolate import LinearNDInterpolator
                from zalmoxis.eos.interpolation import load_paleos_unified_table

                # Get EOS file paths from the material dictionaries
                mat_dicts = load_zalmoxis_material_dictionaries()
                mantle_mat = mat_dicts.get(mantle_eos, {})
                core_mat = mat_dicts.get(config.interior_struct.zalmoxis.core_eos, {})

                # Build nabla_ad(P, T) from PALEOS MgSiO3 unified table
                mantle_file = mantle_mat.get('eos_file', '')
                if mantle_file and os.path.isfile(mantle_file):
                    _cache = load_paleos_unified_table(mantle_file)

                    def _paleos_nabla_ad(P_Pa, T_K, _c=_cache):
                        if P_Pa <= 0 or T_K <= 0:
                            return 0.3
                        lp = max(_c['logp_min'], min(math.log10(P_Pa), _c['logp_max']))
                        lt = max(_c['logt_min'], min(math.log10(T_K), _c['logt_max']))
                        try:
                            v = float(_c['nabla_ad_interp']([[lp, lt]])[0])
                            if np.isfinite(v) and v > 0:
                                return v
                        except Exception:
                            pass
                        return 0.3

                    nabla_ad_func = _paleos_nabla_ad
                    log.info('Using PALEOS nabla_ad(P,T) for initial thermal state adiabat')

                # Build C_p(P, T) interpolators from PALEOS tables for
                # mass-weighted integration over the radial structure
                def _build_cp_func(eos_file, fallback_cp):
                    """Build a C_p(P, T) interpolator from a PALEOS table."""
                    if not eos_file or not os.path.isfile(eos_file):
                        return None
                    _data = np.genfromtxt(eos_file, usecols=range(9), comments='#')
                    _P, _T, _cp = _data[:, 0], _data[:, 1], _data[:, 5]
                    _valid = (_P > 0) & np.isfinite(_cp) & (_cp > 0) & (_cp < 5000)
                    if np.sum(_valid) < 10:
                        return None
                    _lp = np.log10(_P[_valid])
                    _lt = np.log10(_T[_valid])
                    _interp = LinearNDInterpolator(list(zip(_lp, _lt)), _cp[_valid])

                    def _cp_func(P_Pa, T_K, _i=_interp, _fb=fallback_cp):
                        if P_Pa <= 0 or T_K <= 0:
                            return _fb
                        v = float(_i(math.log10(P_Pa), math.log10(T_K)))
                        if np.isfinite(v) and 0 < v < 5000:
                            return v
                        return _fb

                    return _cp_func

                core_file = core_mat.get('eos_file', '')
                cp_iron_func = _build_cp_func(core_file, C_iron)
                cp_silicate_func = _build_cp_func(mantle_file, C_silicate)

                if cp_iron_func is not None:
                    log.info('Using PALEOS C_p(P,T) for iron (mass-weighted integration)')
                if cp_silicate_func is not None:
                    log.info('Using PALEOS C_p(P,T) for silicate (mass-weighted integration)')

            except Exception as e:
                log.warning(
                    'Could not build PALEOS thermal properties: %s. Using constants.', e
                )

        thermal = initial_thermal_state(
            model_results,
            core_mass_fraction=cmf,
            T_radiative_eq=hf_row.get('T_eqm', 255.0),
            f_accretion=config.planet.f_accretion,
            f_differentiation=config.planet.f_differentiation,
            C_iron=C_iron,
            C_silicate=C_silicate,
            nabla_ad_func=nabla_ad_func,
            cp_iron_func=cp_iron_func,
            cp_silicate_func=cp_silicate_func,
        )
        hf_row['T_surf_accr'] = thermal['T_surf_accr']
        # Key consumed by Aragog setup_solver and _set_entropy_ic
        hf_row['T_surface_initial'] = thermal['T_surf_accr']
        hf_row['U_grav_diff'] = thermal['U_differentiated']
        hf_row['U_grav_undiff'] = thermal['U_undifferentiated']
        hf_row['DeltaT_accretion'] = thermal['Delta_T_accretion']
        hf_row['DeltaT_differentiation'] = thermal['Delta_T_differentiation']
        hf_row['DeltaT_adiabat'] = thermal['Delta_T_adiabat']
        hf_row['core_state_initial'] = thermal['core_state']

        # Store the adiabatic T(r) profile for interior solver initialization.
        # SPIDER/Aragog use this to set the initial temperature/entropy profile.
        hf_row['_initial_T_profile'] = thermal['T_profile']
        hf_row['_initial_T_radii'] = thermal['radii']
        hf_row['_initial_T_pressure'] = thermal['pressure']

        log.info(
            'Initial thermal state (White+Li 2025): T_CMB=%.0f K, '
            'T_surf_accr=%.0f K, DeltaT_G=%.0f K, DeltaT_D=%.0f K, '
            'DeltaT_ad=%.0f K, core=%s',
            thermal['T_cmb'],
            thermal['T_surf_accr'],
            thermal['Delta_T_accretion'],
            thermal['Delta_T_differentiation'],
            thermal['Delta_T_adiabat'],
            thermal['core_state'],
        )

    # Update the surface radius, interior radius, and mass in the hf_row
    hf_row['R_int'] = planet_radius
    hf_row['R_core'] = cmb_radius
    hf_row['M_int'] = mass_enclosed[-1]
    hf_row['M_core'] = mass_enclosed[cmb_index]
    hf_row['gravity'] = gravity[-1]

    if config.interior_energetics.module == 'boundary':
        # Boundary backend reads its initial potential and surface
        # temperatures off the freshly solved structure rather than off
        # the [interior_energetics.boundary] config block. Take the
        # uppermost-mantle node for T_magma and the radial surface
        # node for T_surf.
        hf_row['T_magma'] = temperature[-2]
        hf_row['T_surf'] = temperature[-1]
    hf_row['P_center'] = model_results.get('p_center')
    hf_row['P_cmb'] = float(pressure[cmb_index])
    # Expose the dry mass target Zalmoxis converged toward, so the
    # wrapper can enforce a mass-anchor contract
    # |M_int / M_int_target - 1| < _MASS_ANCHOR_TOL post-acceptance.
    # Zalmoxis' internal solver_tol_outer (default 3e-3) is a numerical
    # tolerance, not a coupling contract: it leaves room for ~0.3 % drift
    # between hf_row['M_int'] and the conserved planet mass. The wrapper
    # check tightens this to 1e-3 to satisfy the <0.1 % conservation
    # target for the 1-10 M_Earth coupling.
    hf_row['M_int_target'] = float(config_params.get('planet_mass', 0.0))

    # Self-consistent core density from Zalmoxis structure
    if cmb_radius > 0:
        hf_row['core_density'] = mass_enclosed[cmb_index] / (4.0 / 3.0 * np.pi * cmb_radius**3)
    else:
        hf_row['core_density'] = 0.0

    # Core heat capacity: when 'self', use Dulong-Petit for iron (~450 J/kg/K).
    # When numeric, use the config value directly.
    cfg_heatcap = config.interior_struct.core_heatcap
    hf_row['core_heatcap'] = 450.0 if cfg_heatcap == 'self' else float(cfg_heatcap)

    log.debug(f'Saving Zalmoxis output to {output_zalmoxis}')

    # Select mantle arrays (to match the mesh needed for Aragog)
    mantle_radii = radii[cmb_index:]
    mantle_pressure = pressure[cmb_index:]
    mantle_density = density[cmb_index:]
    mantle_gravity = gravity[cmb_index:]
    mantle_temperature = temperature[cmb_index:]

    # Scale mantle temperature to match Aragog temperature profile format
    mantle_temperature_scaled = scale_temperature_profile_for_aragog(
        config, mantle_radii, mantle_temperature
    )

    # Write temperature profile to a separate file for Aragog to read
    np.savetxt(
        os.path.join(outdir, 'data', 'zalmoxis_output_temp.txt'), mantle_temperature_scaled
    )

    # Scalar-g control knob: when
    # ``interior_energetics.aragog.scalar_gravity_override`` is True,
    # collapse the radial gravity array into a uniform scalar (the
    # surface value from hf_row['gravity']) for the files Aragog and
    # SPIDER both read. Aragog's per-node path at solver.reset() then
    # interpolates to that constant everywhere, giving a constant-gravity
    # interior structure.
    scalar_g_override = config.interior_energetics.aragog.scalar_gravity_override
    if scalar_g_override:
        g_scalar = float(hf_row.get('gravity', 9.81))
        mantle_gravity_out = np.full_like(mantle_gravity, g_scalar)
        log.info(
            'scalar_gravity_override=True: collapsing zalmoxis_output.dat + '
            'spider_mesh.dat gravity column to uniform %.4f m/s^2',
            g_scalar,
        )
    else:
        mantle_gravity_out = mantle_gravity

    # Backup the existing zalmoxis_output.dat before overwriting, so a
    # schema-violation raise can restore the last-good file. Without this,
    # the wrapper's fall-back path reverts hf_row but leaves the
    # just-written (failing-schema) file on disk, and the next Aragog
    # setup_or_update_solver crashes on EOS-vs-mesh inconsistency. The
    # .prev copy lives alongside the primary file.
    import shutil as _shutil

    _output_prev = output_zalmoxis + '.prev'
    if os.path.isfile(output_zalmoxis):
        try:
            _shutil.copy2(output_zalmoxis, _output_prev)
        except Exception as _exc:
            log.warning(
                'Could not backup %s before new write: %s',
                output_zalmoxis,
                _exc,
            )

    # Save final grids to the output file for the mantle for Aragog
    with open(output_zalmoxis, 'w') as f:
        for i in range(len(mantle_radii)):
            f.write(
                f'{mantle_radii[i]:.17e} {mantle_pressure[i]:.17e} {mantle_density[i]:.17e} {mantle_gravity_out[i]:.17e} {mantle_temperature[i]:.17e}\n'
            )

    # Schema check at the Zalmoxis -> Aragog file-handover boundary. On
    # violation: restore the .prev backup (so Aragog reads consistent
    # state on the next iteration via the wrapper's fall-back) and raise
    # RuntimeError.
    # Pass the structure's accurate mantle mass (accumulator total minus the
    # exact core-mass target) so the schema check does not re-integrate the
    # coarse output nodes with a grid trapezoid, which diverges from the RK45
    # accumulator by ~10% at high planet mass and false-rejects valid runs.
    mantle_mass_accumulator = float(mass_enclosed[-1]) - float(cmb_mass)
    try:
        validate_zalmoxis_output_schema(
            output_zalmoxis, hf_row, mantle_mass_ref=mantle_mass_accumulator
        )
    except RuntimeError:
        if os.path.isfile(_output_prev):
            try:
                _shutil.copy2(_output_prev, output_zalmoxis)
                log.warning(
                    'Schema violation: restored %s from %s before re-raise',
                    output_zalmoxis,
                    _output_prev,
                )
            except Exception as _restore_exc:
                log.warning(
                    'Schema violation: could not restore %s from %s: %s',
                    output_zalmoxis,
                    _output_prev,
                    _restore_exc,
                )
        raise

    # Determine SPIDER domain: [R_cmb, R_solvus] when global_miscibility is
    # enabled, otherwise [R_cmb, R_surface] (standard).
    spider_radii = mantle_radii
    spider_pressure = mantle_pressure
    spider_density = mantle_density
    spider_gravity = mantle_gravity

    R_solvus = solvus_radius(config, hf_row.get('R_solvus'), planet_radius, R_inner=cmb_radius)
    if R_solvus is not None:
        # Truncate arrays at the solvus: SPIDER only evolves the
        # miscible interior below the binodal surface
        solvus_mask = mantle_radii <= R_solvus * 1.001  # small tolerance
        if np.any(solvus_mask):
            spider_radii = mantle_radii[solvus_mask]
            spider_pressure = mantle_pressure[solvus_mask]
            spider_density = mantle_density[solvus_mask]
            spider_gravity = mantle_gravity[solvus_mask]
            log.info(
                'SPIDER domain truncated at solvus: R_solvus=%.3e m '
                '(%.2f R_earth), %d of %d shells',
                R_solvus,
                R_solvus / R_earth,
                len(spider_radii),
                len(mantle_radii),
            )

    # Write SPIDER mesh file if requested. Re-uses the possibly-collapsed
    # gravity array so the SPIDER path gets the same scalar-g override
    # behaviour when the flag is on.
    spider_mesh_file = None
    if num_spider_nodes > 0:
        if scalar_g_override:
            spider_gravity_out = np.full_like(
                spider_gravity, float(hf_row.get('gravity', 9.81))
            )
        else:
            spider_gravity_out = spider_gravity
        spider_mesh_file = write_spider_mesh_file(
            outdir,
            spider_radii,
            spider_pressure,
            spider_density,
            spider_gravity_out,
            num_spider_nodes,
        )

    return cmb_radius, spider_mesh_file
