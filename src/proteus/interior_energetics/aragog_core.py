"""Aragog ``core_module`` core in the PROTEUS wrapper: settings, start state, snapshot, diagnostics."""

from __future__ import annotations

import logging
import os

import netCDF4 as nc
import numpy as np
from aragog.solver.entropy_solver import EXTRA_STATE_SLOTS

from proteus.utils.constants import secs_per_year
from proteus.utils.helper import snapshot_path_for_time

log = logging.getLogger('fwl.' + __name__)

# core_module settings that only the wrapper-side diagnostics read; the budget factory rejects them.
_DIAGNOSTIC_ONLY_KEYS = ('f_ohm', 'flux_geometry')
# core_module settings the solver reads itself; the budget factory rejects them.
SOLVER_ONLY_KEYS = ('q_radio', 'ra_crit_cmb')
# Scalars of the core profile and its CMB geometry, with their snapshot units.
_FROZEN_UNITS = {
    'rho_cen': 'kg m-3',
    'length_scale': 'm',
    'm_core': 'kg',
    'p_cen': 'Pa',
    'r_cmb': 'm',
    'p_cmb': 'Pa',
}
# Range of T'/T over the mass parcels of the core above which an update is reported.
ENTROPY_SPREAD_MAX = 1.0e-2
_MASS_NODES = 256


def frozen_profile(interior_o) -> dict[str, float | None]:
    """The core profile scalars of ``interior_o``, None where not set; the CMB radius and
    pressure are those of the solver's budget."""
    profile = {
        name: getattr(interior_o, f'_frozen_core_{name}', None) for name in _FROZEN_UNITS
    }
    budget = getattr(interior_o.aragog_solver, '_core_module_budget', None)
    if budget is not None:
        profile.update(r_cmb=float(budget.profiles.r_cmb), p_cmb=float(budget.profiles.p_cmb))
    return profile


def frozen_profile_fields(profile: dict | None):
    """Snapshot ``(name, value, units)`` rows of a frozen core profile."""
    return tuple(
        (f'core_module_{name}', (profile or {}).get(name), units)
        for name, units in _FROZEN_UNITS.items()
    )


def _snapshot_frozen_profile(output_dir: str, time: float) -> dict[str, float | None]:
    """The frozen core profile scalars stored in the snapshot at ``time``."""
    from proteus.interior_energetics.aragog import _snapshot_scalar

    return {
        name: _snapshot_scalar(output_dir, time, f'core_module_{name}')[0]
        for name in _FROZEN_UNITS
    }


def restore_frozen_profile(interior_o, output_dir: str, time: float) -> None:
    """Copy the frozen core profile scalars of a snapshot onto ``interior_o``, where stored."""
    for name, value in _snapshot_frozen_profile(output_dir, time).items():
        if value is not None:
            setattr(interior_o, f'_frozen_core_{name}', value)


def snapshot_shell(output_dir: str, time: float) -> np.ndarray | None:
    """Shell temperatures [K] of a stratified core from the snapshot at ``time``, or None."""
    fpath = snapshot_path_for_time(os.path.join(output_dir, 'data'), time, '_int.nc')
    with nc.Dataset(fpath) as ds:
        if 'core_T_shell_state' not in ds.variables:
            return None
        t_shell = np.asarray(ds['core_T_shell_state'][:], dtype=float)
    return t_shell if np.all(np.isfinite(t_shell)) else None


def core_temperature_state(solver, core_bc: str) -> float | None:
    """Core temperature state of a ``core_module`` or ``bower2018`` solve [K].

    Parameters
    ----------
    solver : EntropySolver
        Aragog solver after at least one solve.
    core_bc : str
        ``interior_energetics.aragog.core_bc``.

    Returns
    -------
    float or None
        T_core [K] for ``core_module`` or ``bower2018``, else None.
    """
    if core_bc not in ('core_module', 'bower2018') or not hasattr(
        solver, 'get_current_core_temperature'
    ):
        return None
    return solver.get_current_core_temperature()


def start_core_temperature(solver, core_bc: str) -> float | None:
    """Core temperature [K] in the start state of the next solve, or None when the mode
    or the solver holds none."""
    slots = EXTRA_STATE_SLOTS.get(core_bc, ())
    S0 = getattr(solver, '_S0', None)
    if 'T_core' not in slots or S0 is None or len(S0) < solver._n_stag + len(slots):
        return None
    return float(S0[solver._n_stag + slots.index('T_core')])


def core_module_params(config, hf_row: dict, interior_o, outdir: str) -> dict:
    """Aragog ``core_module_params`` for a solver setup.

    The settings of ``[interior_energetics.aragog.core_module]`` without the
    diagnostics-only keys, plus the core profile: on a resume the frozen profile of
    the snapshot, else the structure's core mass and central pressure, to which
    Aragog fits the profile.

    Raises
    ------
    ValueError
        When no frozen profile is restored and the structure gives no positive
        ``M_core`` and ``P_center``.
    """
    import attrs

    params = attrs.asdict(config.interior_energetics.aragog.core_module)
    for key in _DIAGNOSTIC_ONLY_KEYS:
        params.pop(key)

    if getattr(config.params, 'resume', False) is True and 'Time' in hf_row:
        snap = _snapshot_frozen_profile(outdir, hf_row['Time'])
        if snap['rho_cen'] is not None and snap['length_scale'] is not None:
            params.update(
                rho_cen=snap['rho_cen'], length_scale=snap['length_scale'], fit_profile=False
            )
            for name, value in snap.items():
                setattr(interior_o, f'_frozen_core_{name}', value)
            log.info(
                'Restored frozen core profile from snapshot: rho_cen=%.2f kg/m^3, length_scale=%.1f km',
                snap['rho_cen'],
                snap['length_scale'] / 1e3,
            )
            return params

    # Structure constraints from hf_row feed the Gaussian profile fit.
    m_core_val = float(hf_row.get('M_core', 0.0) or 0.0)
    p_cen_val = float(hf_row.get('P_center', 0.0) or 0.0)
    if m_core_val <= 0.0 or p_cen_val <= 0.0:
        struct_mod = config.interior_struct.module
        raise ValueError(
            f"core_bc='core_module' requires positive M_core and P_center from interior structure, "
            f"but interior_struct.module='{struct_mod}' provided M_core={m_core_val:.4e}, P_center={p_cen_val:.4e}"
        )
    params.update(m_core=m_core_val, p_cen=p_cen_val)
    interior_o._frozen_core_m_core = m_core_val
    interior_o._frozen_core_p_cen = p_cen_val
    log.info(
        'Aragog core_module structure constraints: M_core=%.4e kg, P_center=%.4e Pa',
        m_core_val,
        p_cen_val,
    )
    return params


def freeze_core_profile(interior_o) -> None:
    """Keep the profile Aragog fitted at its first setup for every later solver reset."""
    if getattr(interior_o, '_frozen_core_rho_cen', None) is not None:
        return
    solver = interior_o.aragog_solver
    budget = getattr(solver, '_core_module_budget', None)
    if budget is None:
        return
    fitted_rho = float(budget.profiles.rho_cen)
    fitted_len = float(budget.profiles.length_scale)
    interior_o._frozen_core_rho_cen = fitted_rho
    interior_o._frozen_core_length_scale = fitted_len
    bc_params = solver.parameters.boundary_conditions.core_module_params
    if bc_params is not None:
        bc_params['rho_cen'] = fitted_rho
        bc_params['length_scale'] = fitted_len
        bc_params['fit_profile'] = False
        bc_params.pop('m_core', None)
        bc_params.pop('p_cen', None)
    log.info(
        'Aragog core_module profile frozen: rho_cen=%.2f kg/m^3, length_scale=%.1f km',
        fitted_rho,
        fitted_len / 1e3,
    )


def restore_core_start(hf_row: dict, interior_o, solver) -> None:
    """Start a resumed ``core_module`` or ``bower2018`` core from its snapshot state.

    The core temperature comes from the snapshot, else from the resumed row's
    ``T_cmb``, which holds the core temperature in these modes, else Aragog starts
    it from the restored profile's bottom cell. A stratified core's shell comes
    from the snapshot, else it restarts on the core adiabat.
    """
    T_core = getattr(interior_o, '_last_T_core', None)
    if T_core is not None:
        log.info('Restored core temperature from snapshot: T_core=%.2f K', T_core)
    else:
        status = getattr(interior_o, '_last_T_core_status', 'absent')
        T_cmb = float(hf_row.get('T_cmb', np.nan))
        if np.isfinite(T_cmb) and T_cmb > 0:
            T_core = T_cmb
            log.warning(
                'Snapshot core temperature is %s; it restarts from the '
                'resumed row, T_cmb=%.2f K.',
                status,
                T_cmb,
            )
        else:
            log.warning(
                'Snapshot core temperature is %s and the resumed row has no '
                'usable T_cmb; it restarts from the temperature of the '
                "restored profile's bottom cell at the CMB.",
                status,
            )
    if T_core is not None:
        solver.set_initial_core_temperature(T_core)
    solver.set_initial_shell_temperature(getattr(interior_o, '_last_T_shell', None))


def isentropic_ratio(alpha, c_p, dP, rho, ln_shape=0.0, weights=None) -> float:
    """``T_cmb' / T_cmb`` that keeps the total entropy of a core of constant mass.

    The mass-weighted mean of ``c_p ln(T'/T) - alpha dP / rho`` over the parcels is
    zero. With the adiabat of each state written as its CMB temperature times a shape
    factor, ``ln(T_cmb'/T_cmb)`` is the mean of ``alpha dP / (rho c_p) - ln_shape``.

    Parameters
    ----------
    alpha, c_p : float
        Thermal expansion coefficient [K-1] and specific heat capacity [J kg-1 K-1].
    dP, rho : array_like
        Pressure change [Pa] and density [kg m-3] of each mass parcel.
    ln_shape : array_like, optional
        ``ln`` of the new over the old adiabat shape factor of each parcel.
    weights : array_like, optional
        Parcel masses; equal masses when None.
    """
    terms = alpha * np.asarray(dP, dtype=float) / (np.asarray(rho, dtype=float) * c_p)
    return float(np.exp(np.average(terms - ln_shape, weights=weights)))


def _parcels(old, new, q):
    """Pressure change, mean density and ``ln`` of the adiabat shape ratio of the parcels
    at mass fractions ``q``, between two profiles of one core mass."""

    def radius(profiles):
        r = np.linspace(0.0, profiles.r_cmb, 513)
        m = np.asarray(profiles.enclosed_mass(r), dtype=float)
        return np.interp(q * m[-1], m, r)

    r_old, r_new = radius(old), radius(new)
    dP = np.asarray(new.pressure(r_new), dtype=float) - np.asarray(old.pressure(r_old))
    rho = 0.5 * (np.asarray(old.density(r_old), dtype=float) + np.asarray(new.density(r_new)))
    shape = np.asarray(new.adiabat(r_new, 1.0), dtype=float) / np.asarray(
        old.adiabat(r_old, 1.0)
    )
    return dP, rho, np.log(shape)


def profile_key(budget) -> tuple:
    """The density profile and the CMB geometry of a core budget; Aragog builds a new
    budget object for the same core when its parameter set changes form."""
    profiles = budget.profiles
    return profiles.rho_cen, profiles.length_scale, profiles.r_cmb, profiles.p_cmb


def fitted_core_budget(solver, m_core: float, p_cen: float) -> tuple:
    """A core budget fitted to ``m_core`` [kg] and ``p_cen`` [Pa] on the solver's CMB radius
    and pressure, with its 2 profile scalars; the solver stays as it is.

    Raises
    ------
    ValueError
        From the profile fit or the budget factory.
    """
    from aragog.core import build_core_module_budget, fit_gaussian_core_profiles

    params = solver.parameters.boundary_conditions.core_module_params
    r_cmb, p_cmb = float(solver._r_basic_flat[0]), float(solver._P_basic_flat[0])
    fit = fit_gaussian_core_profiles(
        m_core=m_core,
        p_cen=p_cen,
        r_cmb=r_cmb,
        p_cmb=p_cmb,
        alpha=float(params['alpha']),
        c_p=float(params['c_p']),
    )
    profile = dict(rho_cen=float(fit.rho_cen), length_scale=float(fit.length_scale))
    budget_params = {
        k: v for k, v in {**params, **profile}.items() if k not in SOLVER_ONLY_KEYS
    }
    return build_core_module_budget(budget_params, r_cmb=r_cmb, p_cmb_fallback=p_cmb), profile


def install_core_profile(
    interior_o, solver, profile: dict, m_core: float, p_cen: float
) -> None:
    """Give the solver a fitted profile and keep its scalars for the snapshot.

    Raises
    ------
    ValueError
        When the rebuilt solver budget does not carry the profile.
    """
    solver.parameters.boundary_conditions.core_module_params.update(profile)
    solver._cache_bc_constants()
    built = solver._core_module_budget.profiles
    if not all(np.isclose(float(getattr(built, k)), v) for k, v in profile.items()):
        raise ValueError('the rebuilt solver budget does not carry the refitted core profile')
    interior_o._frozen_core_rho_cen = profile['rho_cen']
    interior_o._frozen_core_length_scale = profile['length_scale']
    interior_o._frozen_core_m_core, interior_o._frozen_core_p_cen = m_core, p_cen


def _inner_core_mass(budget, t_core: float) -> float:
    """Mass [kg] inside the inner core boundary of ``budget`` at ``t_core`` [K]."""
    r_icb = budget.r_icb_batch(np.array([float(t_core)]))[0]
    return float(budget.profiles.enclosed_mass(r_icb))


def _with_fusion_entropy(old, new, t_old: float, t_liquid: float, m_core: float) -> float:
    """T_core' [K] that keeps the total entropy when the update changes the inner core.

    ``t_liquid`` keeps the entropy of a core whose inner core mass does not change. The
    mass that freezes at the update gives up ``ds_fusion`` per kilogram, which the core
    takes up: ``m_core c_p ln(T'/t_liquid) = ds_fusion (M_ic'(T') - M_ic(t_old))``.
    """
    from scipy.optimize import brentq

    m_ic, scale = _inner_core_mass(old, t_old), new.ds_fusion / (m_core * new.profiles.c_p)

    def excess(t_core):
        return np.log(t_core / t_liquid) - scale * (_inner_core_mass(new, t_core) - m_ic)

    # ln(T'/t_liquid) is at most ds_fusion / c_p in size, the whole core frozen or melted.
    low, high = (
        t_liquid * np.exp(-1.01 * scale * m_core),
        t_liquid * np.exp(1.01 * scale * m_core),
    )
    if not excess(low) < 0.0 < excess(high):
        raise ValueError(
            'no core temperature keeps the entropy across the structure update between '
            f'{low:.1f} and {high:.1f} K (liquid value {t_liquid:.1f} K)'
        )
    return float(brentq(excess, low, high, xtol=1e-9))


def core_state(solver) -> tuple:
    """The core budget and the core temperature [K] the next solve starts from, None
    where the solver holds none; read before a reset."""
    t_core = getattr(solver, '_T_core_init', None)
    if t_core is None:
        t_core = solver.get_current_core_temperature()
    return getattr(solver, '_core_module_budget', None), t_core


def snapshot_core_state(interior_o, solver) -> tuple:
    """The core budget and temperature of the resumed snapshot when its CMB geometry is
    not the solver's, else ``(None, None)``."""
    from aragog.core import build_core_module_budget

    r_cmb = getattr(interior_o, '_frozen_core_r_cmb', None)
    p_cmb = getattr(interior_o, '_frozen_core_p_cmb', None)
    now = solver._core_module_budget.profiles
    if r_cmb is None or p_cmb is None or (r_cmb, p_cmb) == (now.r_cmb, now.p_cmb):
        return None, None
    params = dict(solver.parameters.boundary_conditions.core_module_params)
    for key in SOLVER_ONLY_KEYS:
        params.pop(key, None)
    old = build_core_module_budget(params, r_cmb=r_cmb, p_cmb_fallback=p_cmb)
    return old, getattr(solver, '_T_core_init', None)


def carry_core_across_update(hf_row: dict, interior_o, solver, before: tuple) -> None:
    """Carry the core across a structure update with its mass and its total entropy.

    The profile is fitted again to the core mass ``interior_o._frozen_core_m_core`` at
    the new CMB radius and pressure and the row's ``P_center``, and ``T_core`` takes the
    value that keeps the total entropy: ``isentropic_ratio`` over equal-mass parcels,
    and below the nucleation onset the entropy of fusion of the mass that the update
    freezes or melts. The change of the heat content, ``E_new(T_core') - E_old(T_core)``,
    is added to ``interior_o._core_struct_booked`` for ``step_dE_struct_core_J``.
    Nothing changes when the CMB radius and pressure of the mesh are those of the budget
    before, and the solver changes only after the new state is computed.

    Parameters
    ----------
    hf_row : dict
        Helpfile row of the step being solved; read only.
    interior_o : Interior_t
        Interior state; its profile scalars and its booking are updated.
    solver : EntropySolver
        The Aragog solver, already on the mesh of the updated structure.
    before : tuple
        The core budget and core temperature [K] before the update.

    Raises
    ------
    ValueError
        From the profile fit, from the heat content of a stratified budget, when the
        heat change is not finite, or when the solver does not take the profile.
    """
    old, t_old = before
    if old is None or t_old is None:
        return
    r_cmb, p_cmb = float(solver._r_basic_flat[0]), float(solver._P_basic_flat[0])
    if (r_cmb, p_cmb) == (old.profiles.r_cmb, old.profiles.p_cmb):
        return
    m_core = getattr(interior_o, '_frozen_core_m_core', None)
    if m_core is None:  # a snapshot without the core mass: the mass of the profile before
        m_core = float(old.profiles.enclosed_mass(old.profiles.r_cmb))
    p_cen = float(hf_row['P_center'])
    new, profile = fitted_core_budget(solver, float(m_core), p_cen)
    q = np.r_[0.0, (np.arange(_MASS_NODES) + 0.5) / _MASS_NODES, 1.0]
    dP, rho, ln_shape = _parcels(old.profiles, new.profiles, q)
    alpha, c_p = new.profiles.alpha, new.profiles.c_p
    ratio = isentropic_ratio(alpha, c_p, dP[1:-1], rho[1:-1], ln_shape[1:-1])
    # T'/T of each parcel alone; all agree when the new profile is the old one compressed.
    parcel = np.exp(alpha * dP / (rho * c_p) - ln_shape)
    spread = float(parcel.max() - parcel.min()) / ratio
    t_new = float(t_old) * ratio
    if t_old < old.t_onset or t_new < new.t_onset:
        t_new = _with_fusion_entropy(old, new, float(t_old), t_new, float(m_core))
    dE = float(new.heat_content(t_new) - old.heat_content(t_old))
    if not np.isfinite(dE):
        raise ValueError(f'core heat change across a structure update is not finite: {dE}')
    install_core_profile(interior_o, solver, profile, float(m_core), p_cen)
    solver.set_initial_core_temperature(t_new)
    if getattr(interior_o, '_core_t_end', None) is None:
        interior_o._core_t_end, interior_o._core_budget_end = float(t_old), old
    interior_o._core_struct_booked = getattr(interior_o, '_core_struct_booked', 0.0) + dE
    log.info(
        '    core across the structure update: T_core %.9f K -> %.9f K, heat content '
        '%+.9e J; P_cmb %.6e -> %.6e Pa, R_core %.6e -> %.6e m; liquid T_core ratio %.9f '
        '(centre parcel %.6f, CMB parcel %.6f, all parcels %.6f to %.6f); inner core mass '
        '%.4e -> %.4e kg; length scale %.1f -> %.1f km; profile mass / structure M_core '
        '- 1 = %+.2e, profile P_center / structure - 1 = %+.2e',
        t_old,
        t_new,
        dE,
        old.profiles.p_cmb,
        p_cmb,
        old.profiles.r_cmb,
        r_cmb,
        ratio,
        parcel[0],
        parcel[-1],
        parcel.min(),
        parcel.max(),
        _inner_core_mass(old, t_old),
        _inner_core_mass(new, t_new),
        old.profiles.length_scale / 1e3,
        new.profiles.length_scale / 1e3,
        float(m_core) / float(hf_row['M_core']) - 1.0,
        float(new.profiles.pressure(0.0)) / p_cen - 1.0,
    )
    if spread > ENTROPY_SPREAD_MAX:
        log.warning(
            'core profile after the structure update is not the old one compressed: its '
            'mass parcels differ by %.2e in T_core ratio (limit %.1e)',
            spread,
            ENTROPY_SPREAD_MAX,
        )


def write_core_diagnostics(runner, output: dict, dt_actual_yr: float = 0.0, out=None) -> None:
    """Fill the ``core_*`` helpfile columns from the core evolution budget.

    Evaluates the energy-side quantities on the solver's own
    ``CoreEnergyBudget`` and the entropy, dynamo, and stratification
    diagnostics on a wrapper-side ``CoreEntropyBudget`` built from the
    config's ``k_core`` / ``f_ohm`` / ``flux_geometry``. The entropy
    budget is kept on ``runner`` and rebuilt whenever the solver's budget
    object changes, keyed on object identity.

    The CMB heat flow driving the diagnostics is the step-averaged
    power ``step_dE_F_cmb_J / dt``; with no elapsed time (the init
    call) it is ``F_cmb`` times the CMB area.
    """
    budget = getattr(runner.aragog_solver, '_core_module_budget', None)
    if budget is None:
        return
    from aragog.core import CoreEntropyBudget, crystallization_regime

    cm_cfg = runner._config.interior_energetics.aragog.core_module
    if getattr(runner, '_core_entropy_for', None) is not budget:
        runner._core_entropy_budget = CoreEntropyBudget(
            budget,
            k_core=cm_cfg.k_core,
            f_ohm=cm_cfg.f_ohm,
            flux_geometry=cm_cfg.flux_geometry,
        )
        runner._core_entropy_for = budget
    ent = runner._core_entropy_budget

    t_cmb = float(output['T_cmb'])
    area = 4.0 * np.pi * float(budget.profiles.r_cmb) ** 2
    span_s = float(dt_actual_yr) * secs_per_year
    if span_s > 0.0:
        q_cmb = float(output['step_dE_F_cmb_J']) / span_s
    else:
        q_cmb = float(output['F_cmb']) * area
    # A stratified core carries its shell profile; the layer base bounds the light-element mixing.
    t_shell = getattr(out, 'core_T_shell', None)
    base = float(getattr(out, 'core_layer_base', np.nan)) if t_shell is not None else np.nan
    upper = {'gravitational_upper': base} if np.isfinite(base) else {}
    output['core_r_icb'] = float(budget.r_icb(t_cmb))
    output['core_C_eff'] = float(budget.effective_capacity(t_cmb, **upper))
    output['core_dynamo_margin'] = float(
        ent.entropy_margin(t_cmb, q_cmb, q_radio=cm_cfg.q_radio, t_shell=t_shell)
    )
    output['core_B_rms'] = float(ent.b_rms_core(t_cmb, q_cmb))
    output['core_regime'] = float(int(crystallization_regime(budget, t_cmb)))
    output['core_strat_depth'] = (
        float(budget.profiles.r_cmb) - base if np.isfinite(base) else 0.0
    )
    output['core_T_top'] = float(t_shell[-1]) if t_shell is not None else t_cmb


def write_core_columns(runner, output: dict, out, interior_o) -> None:
    """Fill the core diagnostics and the core ledger columns of a ``core_module`` call.

    It changes ``output`` only; ``commit_core_call`` moves the ledger state when the
    call returned.
    """
    from proteus.interior_energetics.aragog_core_impact import core_call_heat

    write_core_diagnostics(runner, output, dt_actual_yr=float(out.dt_actual), out=out)
    # A core refit at this step's reset books the previous impact's core heat here.
    booked = getattr(interior_o, '_core_impact_booked', None) or (0.0, 0.0)
    struct = getattr(interior_o, '_core_struct_booked', 0.0)
    output['step_dE_impact_core_J'], output['step_dE_impact_core_refit_J'] = booked
    output['step_dE_struct_core_J'] = struct
    output['step_dE_core_J'] = core_call_heat(out, interior_o, lift=sum(booked) + struct)


def commit_core_call(interior_o, out) -> None:
    """Move the core ledger state to the end of a call whose row is kept: the booked
    heat is used up and the next call measures its jump from this T_core and budget."""
    interior_o._core_impact_booked, interior_o._core_struct_booked = None, 0.0
    interior_o._core_t_end = float(out.T_core)
    interior_o._core_budget_end = getattr(interior_o.aragog_solver, '_core_module_budget', None)
