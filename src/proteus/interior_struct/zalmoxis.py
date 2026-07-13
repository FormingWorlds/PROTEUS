# Zalmoxis interior module
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import platformdirs
from scipy.interpolate import interp1d
from zalmoxis.solver import main

from proteus.config import Config
from proteus.utils.constants import (
    FEI2021_LIQUIDUS_P_CALIB_PA,
    M_earth,
    R_earth,
    element_list,
)
from proteus.utils.data import get_zalmoxis_eos_dir, get_zalmoxis_melting_curves

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))

# Set up logging
log = logging.getLogger('fwl.' + __name__)

# --- liquidus_super super-liquidus adiabat solver tunables ----------------
# The solver searches surface temperature for the coolest adiabat that clears
# the configured liquidus by delta_T_super at its most-constraining depth. The
# search window is anchored to the configured liquidus (not a fixed Kelvin
# band), so it adapts to whatever melting curve is in use.
_SUPERLIQ_N_POINTS = 200  # adiabat sampling for the binding-depth search
_SUPERLIQ_SCAN_SPAN_K = 4000.0  # scan T_surf over [T_liq(P_surf), +span]
_SUPERLIQ_SCAN_STEPS = 20  # coarse scan resolution over that span
_SUPERLIQ_N_BISECT = 12  # surface-T bisection iterations (sub-Kelvin final)
_SUPERLIQ_MAX_S_DRIFT = 1.0e-3  # max fractional entropy drift for an in-table adiabat
_SUPERLIQ_DEFAULT_MUSHY = 0.8  # fallback solidus = factor * liquidus

# Per-process memo so the three IC call sites (structure solve, energetics
# entropy IC, Aragog cross-check) share one solve for a given input instead of
# repeating the ~30-probe search. Keyed on the physical inputs only, so it is
# deterministic; tests clear it between cases (see _clear_superliquidus_cache).
_SUPERLIQ_CACHE: dict = {}

# CMB temperature [K] of the most recently solved super-liquidus adiabat. A
# structure solve driven by an external temperature source discards this anchor
# (the external T(r) is the temperature source), so those calls reuse the value
# the internal-dispatch initial-condition solve already produced instead of
# repeating the scan-and-bisection at a drifted P_cmb. None until the first
# solve completes.
_SUPERLIQ_LAST_ANCHOR: float | None = None

# Set once a run has reported that the Zalmoxis JAX structure path is not
# viable for the configured EOS, so the numpy-fallback provenance is logged a
# single time instead of on every re-solve.
_JAX_NONVIABLE_LOGGED: bool = False


def _clear_superliquidus_cache() -> None:
    """Drop the cached super-liquidus solves (used by tests to avoid leakage)."""
    global _SUPERLIQ_LAST_ANCHOR, _JAX_NONVIABLE_LOGGED
    _SUPERLIQ_CACHE.clear()
    _SUPERLIQ_LAST_ANCHOR = None
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


# Mapping from PROTEUS volatile species to Zalmoxis EOS component names.
# Only species with Zalmoxis EOS tables are included.
_VOLATILE_EOS_MAP = {
    'H2O': 'PALEOS:H2O',
    'H2': 'Chabrier:H',
}


def _make_derived_solidus(liquidus_func, mushy_zone_factor: float):
    """Create a solidus function as T_sol(P) = T_liq(P) * mushy_zone_factor.

    The PALEOS unified path has no tabulated solidus, so the solidus is
    derived as a constant fraction of the liquidus. The default factor of
    0.8 is the solidus-to-liquidus ratio of the Stixrude (2014) MgSiO3
    melting parametrisation (doi:10.1098/rsta.2013.0076), so the derived
    solidus tracks that experimental melting relation rather than being an
    arbitrary depression.

    Parameters
    ----------
    liquidus_func : callable
        P [Pa] -> T_liquidus [K].
    mushy_zone_factor : float
        Cryoscopic depression factor in [0.7, 1.0]. Default 0.8 follows
        Stixrude (2014).

    Returns
    -------
    callable
        P [Pa] -> T_solidus [K].
    """

    def solidus_func(P):
        return liquidus_func(P) * mushy_zone_factor

    return solidus_func


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
        ``H2O_kg_solid``) and mantle mass keys (``M_mantle_liquid``,
        ``M_mantle_solid``).
    mantle_eos : str
        Primary mantle EOS identifier (e.g. ``'PALEOS:MgSiO3'``).

    Returns
    -------
    VolatileProfile or None
        Profile with per-phase fractions, or None if not applicable.
    """
    from zalmoxis.mixing import VolatileProfile

    M_liq = float(hf_row.get('M_mantle_liquid', 0.0))
    M_sol = float(hf_row.get('M_mantle_solid', 0.0))

    # Need mantle mass data to compute fractions
    if M_liq + M_sol <= 0:
        return None

    w_liquid = {}
    w_solid = {}
    has_nonzero = False

    for species, eos_name in _VOLATILE_EOS_MAP.items():
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
    P_cmb estimate on the very first call. The energetics IC step
    (compute_initial_entropy) solves the same adiabat against the converged
    Zalmoxis P_cmb, so any first-call P_cmb mismatch is self-correcting after
    one round-trip.

    When ``external_temperature_source`` is set the structure solve is driven
    by an evolved T(r) profile (or the super-liquidus adiabat callable during
    the IC re-solve) rather than the internal temperature-mode dispatch, so the
    super-liquidus CMB anchor is not the temperature source for this call and
    the solved value is discarded. Those calls reuse the anchor the
    internal-dispatch IC solve already produced and skip the scan-and-bisection;
    before any solve has run they fall back to ``config.planet.tcmb_init``. This
    avoids re-solving (and possibly raising the unreachable-superheat error) on
    every evolution re-solve over a value nothing consumes.

    For all other modes, returns config.planet.tcmb_init verbatim.
    """
    if mode != 'liquidus_super':
        return config.planet.tcmb_init

    if external_temperature_source:
        anchor = _SUPERLIQ_LAST_ANCHOR
        if anchor is not None:
            log.debug(
                'liquidus_super: structure solve uses an external temperature '
                'source; reusing the last solved CMB anchor T_cmb=%.0f K and '
                'skipping the super-liquidus re-solve.',
                anchor,
            )
            return float(anchor)
        log.debug(
            'liquidus_super: structure solve uses an external temperature '
            'source before any super-liquidus solve; using tcmb_init=%.0f K as '
            'the unconsumed CMB anchor.',
            float(config.planet.tcmb_init),
        )
        return float(config.planet.tcmb_init)

    # Anchor the Zalmoxis structure-solve adiabat at the CMB temperature of the
    # solved super-liquidus adiabat the energetics IC uses, so both share the
    # CMB anchor. Note the two profiles are integrated by different methods
    # (Zalmoxis forward-integrates nabla_ad on the structure mesh; the
    # energetics IC inverts the P-S table), so they coincide at the anchor and
    # may differ in the interior by the adiabat-integration error.
    res = solve_superliquidus_adiabat(config, hf_row)
    log.info(
        'liquidus_super CMB anchor for Zalmoxis: T_cmb=%.0f K (fully molten, '
        '%.0f K above the liquidus; surface T=%.0f K, P_cmb=%.0f GPa).',
        res['cmb_T'],
        res['achieved_superheat'],
        res['surface_T'],
        res['P_cmb'] / 1e9,
    )
    return float(res['cmb_T'])


def solve_superliquidus_adiabat(config: Config, hf_row: dict | None) -> dict:
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
        ``achieved_superheat`` [K], ``binding_P`` [Pa] and ``P_cmb`` [Pa].

    Raises
    ------
    RuntimeError
        If the requested superheat cannot be reached before the deep adiabat
        exhausts the EOS table (the mantle is too deep to be molten with that
        much superheat at this mass). The message reports the largest
        achievable superheat so the user can lower ``delta_T_super``.
    """
    try:
        from zalmoxis.eos_export import compute_entropy_adiabat
        from zalmoxis.melting_curves import paleos_liquidus
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            'liquidus_super mode requires Zalmoxis '
            '(zalmoxis.eos_export.compute_entropy_adiabat and '
            'zalmoxis.melting_curves.paleos_liquidus); import failed: '
            f'{e}'
        )

    delta = float(config.planet.delta_T_super)

    P_cmb = hf_row.get('P_cmb') if isinstance(hf_row, dict) else None
    if not P_cmb or P_cmb <= 0:
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        P_cmb = estimate_P_cmb_NL20(
            float(config.planet.mass_tot),
            float(config.interior_struct.core_frac),
            str(config.interior_struct.core_frac_mode),
        )
        log.warning(
            'liquidus_super: hf_row["P_cmb"] not yet populated; using '
            'Noack & Lasbleis (2020) mass-aware fallback P_cmb=%.1f GPa '
            '(mass_tot=%.2f M_Earth). The energetics initial condition is '
            're-derived against the converged Zalmoxis P_cmb on the next '
            'iteration.',
            P_cmb / 1e9,
            float(config.planet.mass_tot),
        )
    P_cmb = float(P_cmb)

    mantle_eos = config.interior_struct.zalmoxis.mantle_eos
    P_surface = 1e5  # 1 bar surface anchor for the adiabat
    global _SUPERLIQ_LAST_ANCHOR
    _cache_key = (round(P_cmb / 1e6), round(delta, 3), str(mantle_eos))
    if _cache_key in _SUPERLIQ_CACHE:
        cached = dict(_SUPERLIQ_CACHE[_cache_key])
        _SUPERLIQ_LAST_ANCHOR = float(cached['cmb_T'])
        return cached

    mat_dicts = load_zalmoxis_material_dictionaries()
    solid_eos, liquid_eos = resolve_2phase_mgsio3_paths(mantle_eos, mat_dicts)
    eos_file = mat_dicts.get(mantle_eos, {}).get('eos_file', '') or solid_eos or ''
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
        valid = bool(
            finite
            and s_drift < _SUPERLIQ_MAX_S_DRIFT
            and np.all(np.diff(T) > -1.0)  # no gross cooling-with-depth
        )
        liq = np.asarray(liq_func(P), dtype=float)
        i = int(np.argmin(T - liq))
        return {
            'superheat': float(T[i] - liq[i]),
            'binding_P': float(P[i]),
            'S_target': float(S_target),
            'cmb_T': float(T[-1]),
            'valid': valid,
        }

    # Scan surface temperature over a window anchored to the SURFACE LIQUIDUS,
    # so the search adapts to the configured melting curve instead of a fixed
    # Kelvin band. The coolest valid molten adiabat brackets the delta crossing
    # from below, so delta_T_super is honoured as a true minimum (no hot floor).
    T_liq_surf = float(np.asarray(liq_func(P_surface)).reshape(-1)[0])
    scan = []
    for T_surf in np.linspace(
        T_liq_surf, T_liq_surf + _SUPERLIQ_SCAN_SPAN_K, _SUPERLIQ_SCAN_STEPS
    ):
        d = _probe(float(T_surf))
        if d['valid'] and np.isfinite(d['superheat']):
            scan.append((float(T_surf), d))
            # Stop once the increasing branch has bracketed the delta crossing;
            # scanning the full span is only needed to locate the table ceiling
            # in the unreachable-superheat (error) case.
            if d['superheat'] >= delta:
                break
    if not scan:
        raise RuntimeError(
            'liquidus_super: no valid molten adiabat found while solving the '
            f'initial condition (P_cmb={P_cmb / 1e9:.0f} GPa, surface '
            f'liquidus={T_liq_surf:.0f} K). The EOS table may not support a '
            'molten mantle at this pressure.'
        )

    # Superheat rises with surface temperature until the deep adiabat hits the
    # EOS-table ceiling; restrict the solve to that increasing branch.
    superheats = [d['superheat'] for _, d in scan]
    branch = scan[: int(np.argmax(superheats)) + 1]
    if branch[-1][1]['superheat'] < delta:
        ceil_T, ceil_d = branch[-1]
        raise RuntimeError(
            'liquidus_super: cannot initialise a fully molten mantle with '
            f'delta_T_super={delta:.0f} K at P_cmb={P_cmb / 1e9:.0f} GPa. The '
            f'largest achievable superheat is {ceil_d["superheat"]:.0f} K (at '
            f'surface T={ceil_T:.0f} K) before the deep adiabat exhausts the '
            'EOS table. Lower delta_T_super or the planet mass, or extend the '
            'EOS table to higher temperature.'
        )

    # Bracket the delta crossing on the increasing branch, then bisection-refine.
    if branch[0][1]['superheat'] >= delta:
        # The coolest in-table adiabat already meets the margin (e.g. delta=0,
        # or the cool end of the valid band is itself table-limited): it is the
        # coolest fully molten adiabat available, so return it.
        T_solved = branch[0][0]
    else:
        T_lo, T_hi = branch[0][0], branch[-1][0]
        for k in range(1, len(branch)):
            if branch[k][1]['superheat'] >= delta:
                T_lo, T_hi = branch[k - 1][0], branch[k][0]
                break
        for _ in range(_SUPERLIQ_N_BISECT):
            mid = 0.5 * (T_lo + T_hi)
            dm = _probe(mid)
            if dm['valid'] and dm['superheat'] >= delta:
                T_hi = mid
            else:
                T_lo = mid
        T_solved = T_hi

    final = _probe(T_solved)
    if not final['valid']:
        raise RuntimeError(
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
            'liquidus_super: the minimum-superheat depth (binding P=%.0f GPa) '
            'is beyond the liquidus calibration (~%.0f GPa), so the superheat '
            'margin there is set against an extrapolated liquidus. Verify the '
            'liquidus parameterisation is appropriate for this EOS and mass.',
            final['binding_P'] / 1e9,
            FEI2021_LIQUIDUS_P_CALIB_PA / 1e9,
        )

    log.info(
        'liquidus_super: surface T=%.0f K gives a fully molten adiabat at least '
        '%.0f K above the liquidus (achieved %.0f K at P=%.0f GPa = %.0f%% of '
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
    }
    _SUPERLIQ_CACHE[_cache_key] = dict(out)
    _SUPERLIQ_LAST_ANCHOR = float(out['cmb_T'])
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

    # Calculate the volatile mass excluded from the structure target.
    # Whole-planet oxygen accounting (issue #677): atmospheric+dissolved O
    # is summed alongside H/C/N/S so the dry-mass target passed to
    # Zalmoxis correctly reserves space for the O that CALLIOPE places in
    # atmospheric H2O, CO2, SO2, etc. Mantle FeO-bound O remains in M_int
    # implicitly via the PALEOS density tables; we do not double-count it.
    # With dry_mantle the full inventory is excluded (the mantle EOS
    # represents bare silicate). When the mantle EOS carries dissolved
    # volatiles (dry_mantle = false), only the atmospheric inventory may
    # be excluded: the dissolved mass is already part of the wet-mantle
    # EOS, and subtracting it again would remove it twice. Escaped mass is already debited from the
    # *_kg_* inventories and needs no separate term.
    # Defensive .get(): some pre-IC paths invoke Zalmoxis before
    # calc_target_elemental_inventories has populated all element columns.
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

    # Mushy zone factor: controls width of partially molten region in PALEOS
    # unified EOS. Applied as T_solidus = T_liquidus * mushy_zone_factor.
    mzf = config.interior_struct.zalmoxis.mushy_zone_factor
    mushy_zone_factors = {
        'PALEOS:iron': mzf,
        'PALEOS:MgSiO3': mzf,
        'PALEOS:H2O': mzf,
    }

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
        # Zalmoxis has populated P_cmb) plus delta_T_super. The energetics
        # IC step recomputes this
        # exact same anchor against the converged P_cmb, so the structure
        # solve and the entropy IC stay in agreement after the first
        # round-trip. temperature_mode_override lets SPIDER coupling force
        # adiabatic without mutating the shared Config object (see proteus
        # rules §"Config mutability").
        'temperature_mode': _resolve_zalmoxis_temperature_mode(
            temperature_mode_override or config.planet.temperature_mode
        ),
        'surface_temperature': config.planet.tsurf_init,
        'cmb_temperature': _resolve_zalmoxis_cmb_temperature(
            config,
            hf_row,
            temperature_mode_override or config.planet.temperature_mode,
            external_temperature_source=external_temperature_source,
        ),
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
    integrating: the mantle registry entry must carry both ``solid_mantle``
    and ``melted_mantle`` dict sub-tables (the 2-phase PALEOS layout), and
    the core entry must resolve to the ``paleos_unified`` format. A
    ``paleos_api`` core qualifies because
    ``zalmoxis.eos.paleos_api_cache.resolve_registry_entry`` materialises
    it to ``paleos_unified`` in place before the JAX dispatch checks it.

    When either precondition fails, the Zalmoxis dispatch raises inside
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
    if not isinstance(mantle_entry.get('solid_mantle'), dict):
        return False
    if not isinstance(mantle_entry.get('melted_mantle'), dict):
        return False
    core_entry = mat_dicts.get(_strip_fraction_tokens(str(core_eos)))
    if not isinstance(core_entry, dict):
        return False
    return core_entry.get('format') in ('paleos_unified', 'paleos_api')


def load_zalmoxis_material_dictionaries():
    """Build an EOS registry dict with file paths pointing to FWL_DATA.

    Returns the same dict format as Zalmoxis ``EOS_REGISTRY``, but with
    every ``eos_file`` path resolved under ``FWL_DATA/zalmoxis_eos/``
    instead of ``ZALMOXIS_ROOT/data/``.  This ensures that Zalmoxis,
    when called from PROTEUS, reads EOS data from the central FWL_DATA
    location managed by ``download_zalmoxis_eos()``.

    Returns
    -------
    dict
        Flat dict keyed by EOS identifier string (e.g.
        ``"Seager2007:iron"``, ``"PALEOS:MgSiO3"``, ``"Chabrier:H"``).
    """
    eos_base = get_zalmoxis_eos_dir()

    # Seager2007 paths (also in the EOS_material_properties location)
    seager_dir = eos_base / 'EOS_Seager2007'
    if not seager_dir.exists():
        seager_dir = FWL_DATA_DIR / 'EOS_material_properties' / 'EOS_Seager2007'

    _seager_iron = {'eos_file': str(seager_dir / 'eos_seager07_iron.txt')}
    _seager_silicate = {'eos_file': str(seager_dir / 'eos_seager07_silicate.txt')}
    _seager_water = {'eos_file': str(seager_dir / 'eos_seager07_water.txt')}

    # Wolf & Bower 2018
    wb_dir = eos_base / 'EOS_WolfBower2018_1TPa'
    _wb_melted = {
        'eos_file': str(wb_dir / 'density_melt.dat'),
        'adiabat_grad_file': str(wb_dir / 'adiabat_temp_grad_melt.dat'),
    }
    _wb_solid = {'eos_file': str(wb_dir / 'density_solid.dat')}

    # RTPress 100 TPa
    rt_dir = eos_base / 'EOS_RTPress_melt_100TPa'
    _rt_melted = {
        'eos_file': str(rt_dir / 'density_melt.dat'),
        'adiabat_grad_file': str(rt_dir / 'adiabat_temp_grad_melt.dat'),
    }

    # PALEOS 2-phase MgSiO3 (separate solid/liquid, Zenodo 19680050).
    # 150 pts/decade (default) and 600 pts/decade (highres) variants.
    paleos2ph_dir = eos_base / 'EOS_PALEOS_MgSiO3'
    _paleos2ph_melted = {
        'eos_file': str(paleos2ph_dir / 'paleos_mgsio3_tables_pt_proteus_liquid.dat'),
        'format': 'paleos',
    }
    _paleos2ph_solid = {
        'eos_file': str(paleos2ph_dir / 'paleos_mgsio3_tables_pt_proteus_solid.dat'),
        'format': 'paleos',
    }
    _paleos2ph_melted_highres = {
        'eos_file': str(paleos2ph_dir / 'paleos_mgsio3_tables_pt_proteus_liquid_highres.dat'),
        'format': 'paleos',
    }
    _paleos2ph_solid_highres = {
        'eos_file': str(paleos2ph_dir / 'paleos_mgsio3_tables_pt_proteus_solid_highres.dat'),
        'format': 'paleos',
    }

    # PALEOS unified tables
    _paleos_iron = {
        'eos_file': str(eos_base / 'EOS_PALEOS_iron' / 'paleos_iron_eos_table_pt.dat'),
        'format': 'paleos_unified',
    }
    _paleos_mgsio3 = {
        'eos_file': str(
            eos_base / 'EOS_PALEOS_MgSiO3_unified' / 'paleos_mgsio3_eos_table_pt.dat'
        ),
        'format': 'paleos_unified',
    }
    _paleos_h2o = {
        'eos_file': str(eos_base / 'EOS_PALEOS_H2O' / 'paleos_water_eos_table_pt.dat'),
        'format': 'paleos_unified',
    }

    # Chabrier H/He
    _chabrier_h = {
        'eos_file': str(eos_base / 'EOS_Chabrier2021_HHe' / 'chabrier2021_H.dat'),
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


def _strip_fraction_tokens(component: str) -> str:
    """Strip trailing mass-fraction tokens from an EOS component string.

    Extended mantle EOS strings carry per-component fractions, e.g.
    ``'PALEOS:MgSiO3:0.9800'``; the registry key is the prefix without
    the numeric token. Only finite numbers count as fraction tokens:
    ``'nan'`` and ``'inf'`` parse as floats but are never written by
    the EOS extension, so they stay part of the identifier.
    """
    tokens = component.split(':')
    while tokens:
        try:
            value = float(tokens[-1])
        except ValueError:
            break
        if not math.isfinite(value):
            break
        tokens.pop()
    return ':'.join(tokens)


def check_zalmoxis_eos_files(layer_eos_config: dict, mat_dicts: dict) -> None:
    """Fail fast when a selected EOS table file is missing on disk.

    Walks the registry entries selected by ``layer_eos_config`` and
    collects every referenced table path that does not exist, then
    raises one actionable error. Without this check the solver emits
    one read error per shell and ends in a non-convergence failure that
    hides the real cause. Registry entries without an ``eos_file``
    (PALEOS-API live tabulation) generate their tables on demand and
    are skipped.

    Parameters
    ----------
    layer_eos_config : dict
        Per-layer EOS identifier strings (``'core'``, ``'mantle'``, ...).
    mat_dicts : dict
        EOS registry from :func:`load_zalmoxis_material_dictionaries`.

    Raises
    ------
    RuntimeError
        If any selected EOS table file is missing, naming every missing
        path and the command that downloads them.
    """
    missing: set[str] = set()
    for identifier in layer_eos_config.values():
        for component in str(identifier).split('+'):
            entry = mat_dicts.get(_strip_fraction_tokens(component))
            if entry is None:
                # Unknown identifiers fail later with a registry error.
                continue
            # Flat entries carry 'eos_file' directly; nested entries map
            # layer roles (core / melted_mantle / ...) to flat entries.
            subentries = [entry] if 'eos_file' in entry else list(entry.values())
            for sub in subentries:
                if not isinstance(sub, dict):
                    continue
                for field in ('eos_file', 'adiabat_grad_file'):
                    path = sub.get(field)
                    if path and not os.path.isfile(path):
                        missing.add(path)
    if missing:
        listing = '\n  '.join(sorted(missing))
        raise RuntimeError(
            f'Interior EOS table file(s) not found:\n  {listing}\n'
            'Download them with '
            '`proteus get interiordata --config-path <config.toml>`, '
            'or run `proteus start` once without --offline.'
        )


def resolve_2phase_mgsio3_paths(mantle_eos: str, mat_dicts: dict):
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
        Absolute filesystem paths if both tables exist, ``(None, None)``
        otherwise. Caller is responsible for treating ``None`` as "no
        2-phase tables available, fall back".
    """
    use_api = mantle_eos.startswith(('PALEOS-API:', 'PALEOS-API-2phase:'))
    if use_api:
        twophase_key = 'PALEOS-API-2phase:MgSiO3'
    elif mantle_eos == 'PALEOS-2phase:MgSiO3-highres':
        twophase_key = 'PALEOS-2phase:MgSiO3-highres'
    else:
        twophase_key = 'PALEOS-2phase:MgSiO3'
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

    For WolfBower2018/RTPress100TPa, loads SPIDER-format P-T files from FWL_DATA.
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
    _TDEP_PREFIXES = ('WolfBower2018', 'RTPress100TPa')
    if mantle_eos.startswith(_TDEP_PREFIXES):
        return get_zalmoxis_melting_curves(config)

    # PALEOS unified and PALEOS-2phase: both use the same analytic Belonoshko+2005 /
    # Fei+2021 melting curve (`PALEOS-liquidus`) as the basis for MgSiO3 phase
    # separation. The unified path uses it for in-table density interpolation;
    # the 2-phase path uses it inside `_compute_paleos_dtdp` to weight nabla_ad
    # across the solid/liquid blend (mixing.py:_compute_paleos_dtdp). The solidus
    # is derived as `liquidus * mushy_zone_factor` so the mushy band lines up with
    # the unified PALEOS density interpolation. Without these curves, the
    # 2-phase nabla_ad call fails and Zalmoxis structure solve diverges; the
    # unified path falls back to phi=0.5 everywhere in VolatileProfile.
    if mantle_eos.startswith(
        ('PALEOS:', 'PALEOS-2phase:', 'PALEOS-API:', 'PALEOS-API-2phase:')
    ):
        try:
            from zalmoxis.melting_curves import get_solidus_liquidus_functions

            _, liquidus_func = get_solidus_liquidus_functions(
                solidus_id='Stixrude14-solidus',  # required by API but unused; solidus is built below as mushy_zone_factor * liquidus (default 0.8 = the Stixrude 2014 solidus/liquidus ratio)
                liquidus_id='PALEOS-liquidus',
            )
            mzf = config.interior_struct.zalmoxis.mushy_zone_factor
            solidus_func = _make_derived_solidus(liquidus_func, mzf)
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

    Returns
    -------
    str
        Deterministic cache-identity string.
    """
    eos_identity = '|'.join(str(p) for p in (mantle_eos, eos_file, solid_eos, liquid_eos))
    eos_digest = hashlib.sha1(eos_identity.encode()).hexdigest()[:12]
    eos_name = re.sub(r'[^A-Za-z0-9]+', '-', str(mantle_eos)).strip('-')
    return (
        f'P_max={P_max:.6e}_nP={nP}_nS={nS}_mzf={mzf}'
        f'_layout={layout}_eos={eos_name}-{eos_digest}'
    )


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
       solid + liquid tables when those are present (see the unified branch
       below), so the densities stay resolved across the melting-curve
       discontinuity. The solidus is derived from ``mushy_zone_factor *
       liquidus`` (default 0.8, the Stixrude 2014 solidus/liquidus ratio); the
       liquidus is the analytic PALEOS Belonoshko+2005 / Fei+2021 curve.
    2. ``PALEOS-2phase:<solid>`` (e.g. ``PALEOS-2phase:MgSiO3``): separate
       solid + liquid PALEOS tables. Phase boundaries are sampled at the
       PALEOS-liquidus temperature from each phase table directly. The
       ``mushy_zone_factor`` config value is ignored (no analytic mushy
       zone exists for 2-phase; the gap between solid-table-top and
       liquid-table-bottom defines the latent heat).

    For non-PALEOS EOS types (WolfBower2018, RTPress100TPa), returns None
    and the caller is expected to fall back on pre-existing SPIDER tables.

    Parameters
    ----------
    config : Config
        Configuration object with struct.zalmoxis settings.
    outdir : str
        Output directory. Tables are written to ``outdir/data/spider_eos/``.

    Returns
    -------
    dict or None
        Keys ``'eos_dir'``, ``'solidus_path'``, ``'liquidus_path'`` with
        absolute paths. Returns None if the mantle EOS is not PALEOS.
    """
    from zalmoxis.eos_export import generate_spider_eos_tables, generate_spider_phase_boundaries
    from zalmoxis.melting_curves import get_solidus_liquidus_functions

    mantle_eos = config.interior_struct.zalmoxis.mantle_eos

    # Use FWL_DATA paths (not ZALMOXIS_ROOT) for EOS file lookup
    mat_dicts = load_zalmoxis_material_dictionaries()
    eos_entry = mat_dicts.get(mantle_eos)

    if eos_entry is None:
        log.info(
            'Mantle EOS %s not found in material dictionary; using pre-existing SPIDER tables.',
            mantle_eos,
        )
        return None

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
        log.info(
            'Mantle EOS %s is neither PALEOS unified nor PALEOS-2phase; '
            'using pre-existing SPIDER tables.',
            mantle_eos,
        )
        return None

    # Resolve unified file (if present) and 2-phase files (if present).
    eos_file = eos_entry.get('eos_file', '')
    eos_file = eos_file if eos_file and os.path.isfile(eos_file) else None

    if is_twophase:
        solid_eos = eos_entry['solid_mantle'].get('eos_file', '')
        liquid_eos = eos_entry['melted_mantle'].get('eos_file', '')
    else:
        # Unified mantle: also look for sibling 2-phase tables to harden
        # the property surfaces (avoids interpolation across the melting
        # curve discontinuity in the unified table). Use the API-aware
        # helper so PALEOS-API unified runs pull API 2-phase tables
        # rather than silently pulling shipped Zenodo ones.
        # Net effect for PALEOS:MgSiO3: the structure solve uses the unified
        # table, but the per-phase property/density surfaces are taken from
        # these two-phase tables when present. If they are absent the code
        # below falls back to the unified table alone (entropy near the
        # melting curve is then less reliable). The solidus stays synthetic
        # (mushy_zone_factor * liquidus) in both cases.
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
            log.warning(
                'No PALEOS EOS file available for %s '
                '(unified missing and 2-phase incomplete); skipping table gen.',
                mantle_eos,
            )
            return None

    if is_twophase and not (solid_eos and liquid_eos):
        log.warning(
            'PALEOS-2phase entry %s missing solid or liquid file; skipping.',
            mantle_eos,
        )
        return None

    # Phase boundaries: PALEOS-liquidus is the analytic Belonoshko+2005 /
    # Fei+2021 Simon-Glatzel curve. For PALEOS-2phase, mushy_zone_factor=1.0
    # collapses solidus = liquidus and the latent-heat gap is supplied by
    # the entropy difference between solid_table.s(P, T_liq) and
    # liquid_table.s(P, T_liq).
    _, liquidus_func = get_solidus_liquidus_functions(
        solidus_id='Stixrude14-solidus',  # unused, but API requires it
        liquidus_id='PALEOS-liquidus',
    )
    mzf = config.interior_struct.zalmoxis.mushy_zone_factor
    solidus_func = _make_derived_solidus(liquidus_func, mzf)
    if is_twophase:
        log.info(
            'PALEOS-2phase phase boundaries: solidus T = liquidus T '
            '(mushy_zone_factor=%.2f, latent heat from 2-phase tables)',
            mzf,
        )
    else:
        log.info(
            'PALEOS unified phase boundaries: solidus = liquidus * %.2f (mushy_zone_factor)',
            mzf,
        )

    # Determine pressure range from planet mass (higher mass needs wider range)
    mass_tot = config.planet.mass_tot or 1.0
    # P_max for the SPIDER P-S lookup grid. Must cover the actual P_cmb
    # of the planet; the 10 TPa cap covers very massive rocky planets
    # (mass_tot well above 2) without hitting the table edge. See
    # interior_energetics/aragog.py for the matching cap and the
    # comment on EOS / melting-curve calibration ranges.
    P_max = min(1.0e13, 150e9 * mass_tot + 200e9)

    if solid_eos and liquid_eos:
        log.info('Using PALEOS-2phase tables for entropy-IC table generation')

    # Table resolution from config
    nP = config.interior_struct.zalmoxis.lookup_nP
    nS = config.interior_struct.zalmoxis.lookup_nS

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

    # Table location. Default: per-run output/<run>/data/spider_eos. When
    # PROTEUS_PS_CACHE_DIR is set, the directory is keyed by cache_key so that
    # independent runs with the same planet mass, table resolution, and mantle
    # EOS reuse one generated table instead of each rebuilding the slow
    # full-resolution PALEOS P-S table. The cache_key encodes everything that
    # changes the table (P_max, nP, nS, mushy_zone_factor, layout, and the
    # resolved EOS identity), so reuse is exact.
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
        can take the Zalmoxis JAX dispatch (2-phase PALEOS mantle with
        solid and melted sub-tables plus a unified PALEOS core; see
        :func:`_zalmoxis_jax_structure_viable`), and the solve is dry
        (no ``VolatileProfile`` in play). In that case the
        external callable is withheld so the inner Picard converges on
        Zalmoxis' internal linear-T profile while the JAX RHS integrates
        against the arrays. For any other EOS configuration the solve
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
    if volatile_profile is not None:
        config_params['layer_eos_config']['mantle'] = extend_mantle_eos_with_volatiles(
            config_params['layer_eos_config']['mantle'], volatile_profile
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
    # 2-phase PALEOS mantle (solid_mantle + melted_mantle sub-tables) and
    # a unified PALEOS core. Only in that case is the external callable
    # withheld: the JAX RHS integrates against the arrays while the numpy
    # Picard helper converges quickly on Zalmoxis' internal linear-T
    # profile (passing the callable there lands Picard near PALEOS
    # phase-boundary clamps and costs roughly two orders of magnitude
    # more wall time at the same JAX arrays). For every other EOS
    # configuration the JAX dispatch declines and the numpy ODE runs;
    # that path consumes only the callable, so it must pass through for
    # the solve to follow the evolved T(r) instead of rebuilding the
    # internal temperature_mode profile from the hot initial anchor.
    # Wet solves (volatile_profile present) also keep the callable and
    # stay on the numpy path here: the arrays handoff and the post-solve
    # rebuild gated on _drop_callable recompute density from the bare dry
    # mantle EOS and would overwrite the volatile-blended column.
    _use_jax_active = bool(config_params.get('use_jax'))
    _jax_viable = _zalmoxis_jax_structure_viable(
        mat_dicts, config.interior_struct.zalmoxis.core_eos, mantle_eos
    )
    # Surface a JAX->numpy fallback once from the PROTEUS side (see helper).
    if _use_jax_active and not _jax_viable:
        _log_jax_nonviable_once(config.interior_struct.zalmoxis.core_eos, mantle_eos)
    _drop_callable = (
        _use_jax_active
        and temperature_arrays is not None
        and _jax_viable
        and volatile_profile is None
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
        # will actually consume temperature_arrays (2-phase PALEOS
        # mantle + unified core); on any other EOS configuration the
        # numpy path runs and the callable passes through. Warm-starts
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
    # runs.
    if _drop_callable:
        from zalmoxis.eos.dispatch import calculate_density as _calc_rho

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
        hf_row['T_cmb_initial'] = thermal['T_cmb']
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

    if config.interior_struct.zalmoxis.global_miscibility:
        R_solvus = hf_row.get('R_solvus')
        if R_solvus is not None and R_solvus < planet_radius:
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
