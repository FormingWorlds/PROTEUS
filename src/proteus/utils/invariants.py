"""Run-row physical invariant assertions and enforcement modes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from proteus.escape.wrapper import readable_total
from proteus.outgas.trapping import derived_total_elements, escapable_inventory
from proteus.utils.constants import (
    element_list,
    noble_gases,
    vol_element_list,
    vol_gas_list,
)
from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    import pandas as pd

    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

STATUS_ERROR_ENERGY_INVARIANT: int = 30
STATUS_ERROR_MASS_INVARIANT: int = 31
STATUS_ERROR_STATE_BOUNDS_INVARIANT: int = 32


@dataclass(frozen=True)
class InvariantViolation:
    """Single run-row invariant violation record.

    Parameters
    ----------
    name : str
        Identifier of the invariant or helpfile column that failed.
    status_code : int
        PROTEUS status code (30 for energy, 31 for mass/structure, 32 for state bounds).
    value : float
        Observed residual or out-of-bounds value.
    tolerance : float
        Configured tolerance or bound threshold.
    message : str
        Diagnostic message formatting both the observed value and threshold.
    report_only : bool, default False
        When True, the violation logs a warning and never stops the run even when
        ``config.invariants.mode == 'stop'``.
    """

    name: str
    status_code: int
    value: float
    tolerance: float
    message: str
    report_only: bool = False


class InvariantViolationError(RuntimeError):
    """Raised when a stopping run-row invariant fails under ``mode == 'stop'``.

    Parameters
    ----------
    primary : InvariantViolation
        First stopping violation that triggered run termination.
    violations : Sequence[InvariantViolation], optional
        All stopping violations observed on the same row.
    """

    def __init__(
        self,
        primary: InvariantViolation,
        violations: Sequence[InvariantViolation] | None = None,
    ) -> None:
        self.status_code: int = primary.status_code
        self.invariant_name: str = primary.name
        self.value: float = primary.value
        self.tolerance: float = primary.tolerance
        self.violations: list[InvariantViolation] = (
            list(violations) if violations is not None else [primary]
        )
        joined = '; '.join(v.message for v in self.violations)
        super().__init__(
            f'Run-row invariant violation (status {self.status_code}, '
            f'{self.invariant_name}): {joined}'
        )


def handle_violations(
    violations: Sequence[InvariantViolation],
    mode: str = 'report',
    directories: dict | None = None,
) -> list[InvariantViolation]:
    """Process run-row invariant violations according to the configured mode.

    Parameters
    ----------
    violations : Sequence[InvariantViolation]
        Violations detected on the current helpfile row.
    mode : str, default 'report'
        Enforcement mode: ``'off'``, ``'report'``, or ``'stop'``.
    directories : dict or None, optional
        PROTEUS run directories dictionary; when provided and a stopping violation
        occurs under ``mode == 'stop'``, writes ``primary.status_code`` to the run
        status file before raising :class:`InvariantViolationError`.

    Returns
    -------
    list[InvariantViolation]
        List of processed violations (empty when ``mode == 'off'``).

    Raises
    ------
    InvariantViolationError
        When ``mode == 'stop'`` and at least one violation has ``report_only=False``.
    """
    if mode == 'off' or not violations:
        return []

    if mode == 'report':
        for v in violations:
            log.warning('Invariant report [%s]: %s', v.name, v.message)
        return list(violations)

    if mode != 'stop':
        raise ValueError(
            f"Unknown invariants.mode {mode!r}; expected 'off', 'report', or 'stop'."
        )

    stopping: list[InvariantViolation] = []
    for v in violations:
        if v.report_only:
            log.warning('Invariant report [%s]: %s', v.name, v.message)
        else:
            log.error('Invariant violation [%s]: %s', v.name, v.message)
            stopping.append(v)

    if stopping:
        primary = stopping[0]
        if directories is not None and 'output' in directories:
            UpdateStatusfile(directories, primary.status_code)
        raise InvariantViolationError(primary, stopping)

    return list(violations)


def check_energy_invariants(
    hf_row: dict,
    config: Config,
    hf_all: pd.DataFrame | None = None,
) -> list[InvariantViolation]:
    """Evaluate coupled energy conservation invariants on ``hf_row`` (status 30).

    Checks gross-throughput-normalized coupled energy residual, per-step energy
    increment consistency, cumulative entropy-ODE solver residual, and core
    ledger residual. Energy violations are emitted with ``report_only=True``.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row dictionary after energy residuals are populated.
    config : Config
        PROTEUS configuration object.
    hf_all : pd.DataFrame or None, optional
        Accumulated helpfile table prior to appending ``hf_row``.

    Returns
    -------
    list[InvariantViolation]
        Detected energy invariant violations (all with ``report_only=True``).
    """
    if hf_all is None or len(hf_all) == 0:
        return []

    e_state_cons = float(hf_row.get('E_state_cons_J', 0.0))
    if not np.isfinite(e_state_cons) or e_state_cons == 0.0:
        return []

    time_yr = float(hf_row.get('Time', 0.0))
    f_int = float(hf_row.get('step_dE_F_int_J', 0.0))
    f_cmb = float(hf_row.get('step_dE_F_cmb_J', 0.0))
    q_rad = float(hf_row.get('step_dE_Q_radio_J', 0.0))
    q_tid = float(hf_row.get('step_dE_Q_tidal_J', 0.0))
    step_gross = (
        (abs(f_int) if np.isfinite(f_int) else 0.0)
        + (abs(f_cmb) if np.isfinite(f_cmb) else 0.0)
        + (abs(q_rad) if np.isfinite(q_rad) else 0.0)
        + (abs(q_tid) if np.isfinite(q_tid) else 0.0)
    )

    hist_gross = 0.0
    if 'step_dE_F_int_J' in hf_all:
        hist_gross += float(
            np.sum(
                np.abs(np.nan_to_num(hf_all['step_dE_F_int_J'].to_numpy(dtype=float), nan=0.0))
            )
        )
    if 'step_dE_F_cmb_J' in hf_all:
        hist_gross += float(
            np.sum(
                np.abs(np.nan_to_num(hf_all['step_dE_F_cmb_J'].to_numpy(dtype=float), nan=0.0))
            )
        )
    if 'step_dE_Q_radio_J' in hf_all:
        hist_gross += float(
            np.sum(
                np.abs(
                    np.nan_to_num(hf_all['step_dE_Q_radio_J'].to_numpy(dtype=float), nan=0.0)
                )
            )
        )
    if 'step_dE_Q_tidal_J' in hf_all:
        hist_gross += float(
            np.sum(
                np.abs(
                    np.nan_to_num(hf_all['step_dE_Q_tidal_J'].to_numpy(dtype=float), nan=0.0)
                )
            )
        )

    e_state_heat = float(hf_row.get('E_state_heat_cons_J', 0.0))
    abs_e_state_heat = abs(e_state_heat) if np.isfinite(e_state_heat) else 0.0
    denom_state = max(abs_e_state_heat, 1.0)
    denom_gross = max(abs_e_state_heat, hist_gross + step_gross, 1.0)

    inv_cfg = config.invariants
    dt_cfg = getattr(getattr(config, 'params', None), 'dt', None)
    is_coarse_dt = bool(
        dt_cfg is not None
        and (
            float(getattr(dt_cfg, 'rtol', 0.1)) >= 0.2
            or float(getattr(dt_cfg, 'minimum', 0.0)) >= 100.0
        )
    )
    eff_energy_rtol = (
        float(inv_cfg.energy_residual_coarse_dt_rtol)
        if is_coarse_dt
        else float(inv_cfg.energy_residual_rtol)
    )

    violations: list[InvariantViolation] = []
    e_res_j = float(hf_row.get('E_residual_cons_J', 0.0))
    e_res_frac_gross = abs(e_res_j) / denom_gross if np.isfinite(e_res_j) else float('nan')
    if not np.isfinite(e_res_frac_gross) or e_res_frac_gross > eff_energy_rtol:
        violations.append(
            InvariantViolation(
                name='E_residual_cons_frac_gross',
                status_code=STATUS_ERROR_ENERGY_INVARIANT,
                value=e_res_frac_gross,
                tolerance=eff_energy_rtol,
                message=(
                    f'Time={time_yr:.6e} yr: E_residual_cons_frac_gross={e_res_frac_gross:.6e} '
                    f'exceeds tol={eff_energy_rtol:.6e} '
                    f'(E_residual_cons_J={e_res_j:.6e} J, denom_gross={denom_gross:.6e} J)'
                ),
                report_only=True,
            )
        )

    prev_e_state_heat = 0.0
    if 'E_state_heat_cons_J' in hf_all:
        prev_val = float(hf_all['E_state_heat_cons_J'].iloc[-1])
        if np.isfinite(prev_val):
            prev_e_state_heat = prev_val

    if prev_e_state_heat != 0.0:
        dE_state_heat = float(hf_row.get('step_dE_state_heat_J', 0.0))
        dE_inc_cons = f_int + f_cmb + q_rad + q_tid
        step_diff = abs(dE_state_heat - dE_inc_cons)
        step_err = step_diff / denom_gross if np.isfinite(step_diff) else float('nan')
        t_surf = float(hf_row.get('T_surf', 1000.0))
        t_magma = float(hf_row.get('T_magma', 1000.0))
        phi_glob = float(hf_row.get('Phi_global', 1.0))
        rf_depth = float(hf_row.get('RF_depth', 0.0))
        on_phase_kink = (
            (np.isfinite(t_surf) and t_surf <= 301.0)
            or (np.isfinite(t_magma) and t_magma <= 301.0)
            or (np.isfinite(phi_glob) and 0.0 < phi_glob < 1.0)
            or (np.isfinite(rf_depth) and 0.0 < rf_depth < 1.0)
        )
        step_tol = (
            float(inv_cfg.step_energy_smin_rtol)
            if on_phase_kink
            else float(inv_cfg.step_energy_rtol)
        )
        if is_coarse_dt:
            step_tol = max(step_tol, float(inv_cfg.energy_residual_coarse_dt_rtol))
        if not np.isfinite(step_err) or step_err > step_tol:
            violations.append(
                InvariantViolation(
                    name='step_dE_inc_frac',
                    status_code=STATUS_ERROR_ENERGY_INVARIANT,
                    value=step_err,
                    tolerance=step_tol,
                    message=(
                        f'Time={time_yr:.6e} yr: step_dE_inc_frac={step_err:.6e} '
                        f'exceeds tol={step_tol:.6e} '
                        f'(step_dE_state_heat_J={dE_state_heat:.6e} J, '
                        f'dE_inc_cons={dE_inc_cons:.6e} J)'
                    ),
                    report_only=True,
                )
            )

    sol_res_j = float(hf_row.get('solver_residual_J', 0.0))
    sol_res_frac = abs(sol_res_j) / denom_state if np.isfinite(sol_res_j) else float('nan')
    sol_tol = float(inv_cfg.solver_residual_rtol)
    if not np.isfinite(sol_res_frac) or sol_res_frac > sol_tol:
        violations.append(
            InvariantViolation(
                name='solver_residual_frac',
                status_code=STATUS_ERROR_ENERGY_INVARIANT,
                value=sol_res_frac,
                tolerance=sol_tol,
                message=(
                    f'Time={time_yr:.6e} yr: solver_residual_frac={sol_res_frac:.6e} '
                    f'exceeds tol={sol_tol:.6e} (solver_residual_J={sol_res_j:.6e} J)'
                ),
                report_only=True,
            )
        )

    core_c_eff = float(hf_row.get('core_C_eff', 0.0))
    if np.isfinite(core_c_eff) and core_c_eff != 0.0:
        core_res_raw = float(hf_row.get('E_core_residual_frac', 0.0))
        core_res_frac = abs(core_res_raw) if np.isfinite(core_res_raw) else float('nan')
        core_tol = float(inv_cfg.core_residual_rtol)
        if not np.isfinite(core_res_frac) or core_res_frac > core_tol:
            violations.append(
                InvariantViolation(
                    name='E_core_residual_frac',
                    status_code=STATUS_ERROR_ENERGY_INVARIANT,
                    value=core_res_frac,
                    tolerance=core_tol,
                    message=(
                        f'Time={time_yr:.6e} yr: |E_core_residual_frac|={core_res_frac:.6e} '
                        f'exceeds tol={core_tol:.6e}'
                    ),
                    report_only=True,
                )
            )

    return violations


def check_structure_invariants(
    hf_row: dict,
    config: Config,
) -> list[InvariantViolation]:
    """Evaluate structure mass self-consistency and target invariants on ``hf_row`` (status 31).

    Checks ``struct_mass_desync_frac`` against the warning threshold (default ``5.0e-3``,
    ``report_only=True``) and stop threshold (default ``1.0e-2``, ``report_only=False``),
    interior mass ``M_int`` against ``M_int_target`` (when ``M_int_target > 0``), and
    core mass ``M_core`` against ``M_core_target`` (when ``M_core_target > 0``).

    Parameters
    ----------
    hf_row : dict
        Current helpfile row dictionary.
    config : Config
        PROTEUS configuration object.

    Returns
    -------
    list[InvariantViolation]
        Detected structure mass and target violations.
    """
    inv_cfg = config.invariants
    time_yr = float(hf_row.get('Time', 0.0))
    m_int = float(hf_row.get('M_int', 0.0))
    m_core = float(hf_row.get('M_core', 0.0))
    r_int = float(hf_row.get('R_int', 0.0))
    r_core = float(hf_row.get('R_core', 0.0))

    violations: list[InvariantViolation] = []
    desync = float(hf_row.get('struct_mass_desync_frac', 0.0))
    warn_tol = float(inv_cfg.struct_mass_desync_warn_rtol)
    stop_tol = float(inv_cfg.struct_mass_desync_rtol)
    if not np.isfinite(desync) or desync > stop_tol:
        violations.append(
            InvariantViolation(
                name='struct_mass_desync_frac',
                status_code=STATUS_ERROR_MASS_INVARIANT,
                value=desync,
                tolerance=stop_tol,
                message=(
                    f'Time={time_yr:.6e} yr: struct_mass_desync_frac={desync:.6e} '
                    f'exceeds stop tol={stop_tol:.6e} '
                    f'(M_int={m_int:.6e} kg, R_int={r_int:.6e} m, R_core={r_core:.6e} m)'
                ),
                report_only=False,
            )
        )
    elif desync > warn_tol:
        violations.append(
            InvariantViolation(
                name='struct_mass_desync_frac',
                status_code=STATUS_ERROR_MASS_INVARIANT,
                value=desync,
                tolerance=warn_tol,
                message=(
                    f'Time={time_yr:.6e} yr: struct_mass_desync_frac={desync:.6e} '
                    f'exceeds warn tol={warn_tol:.6e} '
                    f'(M_int={m_int:.6e} kg, R_int={r_int:.6e} m, R_core={r_core:.6e} m)'
                ),
                report_only=True,
            )
        )

    m_int_target = float(hf_row.get('M_int_target', 0.0))
    if np.isfinite(m_int_target) and m_int_target > 0.0:
        int_rel_err = abs(m_int / m_int_target - 1.0) if np.isfinite(m_int) else float('nan')
        int_tol = float(inv_cfg.interior_mass_target_rtol)
        if not np.isfinite(int_rel_err) or int_rel_err > int_tol:
            violations.append(
                InvariantViolation(
                    name='M_int_target_rel_err',
                    status_code=STATUS_ERROR_MASS_INVARIANT,
                    value=int_rel_err,
                    tolerance=int_tol,
                    message=(
                        f'Time={time_yr:.6e} yr: |M_int / M_int_target - 1|={int_rel_err:.6e} '
                        f'exceeds tol={int_tol:.6e} '
                        f'(M_int={m_int:.6e} kg, M_int_target={m_int_target:.6e} kg)'
                    ),
                    report_only=False,
                )
            )

    m_core_target = float(hf_row.get('M_core_target', 0.0))
    if np.isfinite(m_core_target) and m_core_target > 0.0:
        core_rel_err = (
            abs(m_core / m_core_target - 1.0) if np.isfinite(m_core) else float('nan')
        )
        core_tol = float(inv_cfg.core_mass_target_rtol)
        if not np.isfinite(core_rel_err) or core_rel_err > core_tol:
            violations.append(
                InvariantViolation(
                    name='M_core_target_rel_err',
                    status_code=STATUS_ERROR_MASS_INVARIANT,
                    value=core_rel_err,
                    tolerance=core_tol,
                    message=(
                        f'Time={time_yr:.6e} yr: |M_core / M_core_target - 1|={core_rel_err:.6e} '
                        f'exceeds tol={core_tol:.6e} '
                        f'(M_core={m_core:.6e} kg, M_core_target={m_core_target:.6e} kg)'
                    ),
                    report_only=False,
                )
            )

    return violations


def check_volatile_invariants(
    hf_row: dict,
    config: Config,
) -> list[InvariantViolation]:
    """Evaluate volatile inventory, gas-species sum, planet mass cap, and escape bounds (status 31).

    Parameters
    ----------
    hf_row : dict
        Current helpfile row dictionary.
    config : Config
        PROTEUS configuration object.

    Returns
    -------
    list[InvariantViolation]
        Detected volatile and atmospheric mass violations.
    """
    inv_cfg = config.invariants
    time_yr = float(hf_row.get('Time', 0.0))
    m_planet = float(hf_row.get('M_planet', 0.0))
    if np.isfinite(m_planet) and m_planet <= 0.0:
        return []

    m_atm = float(hf_row.get('M_atm', 0.0))
    m_vol_atm = float(hf_row.get('M_vol_atm', 0.0))
    m_vaps = float(hf_row.get('M_vaps', 0.0))
    vapourise = bool(getattr(getattr(config, 'outgas', None), 'vapourise', False))
    violations: list[InvariantViolation] = []

    cap_tol = float(inv_cfg.planet_mass_cap_rtol)
    if not (np.isfinite(m_planet) and np.isfinite(m_atm) and np.isfinite(m_vaps)):
        violations.append(
            InvariantViolation(
                name='M_atm_planet_cap',
                status_code=STATUS_ERROR_MASS_INVARIANT,
                value=float('nan'),
                tolerance=cap_tol,
                message=(
                    f'Time={time_yr:.6e} yr: non-finite planet/atmosphere mass '
                    f'(M_atm={m_atm:.6e} kg, M_planet={m_planet:.6e} kg, M_vaps={m_vaps:.6e} kg)'
                ),
                report_only=False,
            )
        )
    elif not vapourise or m_vaps <= 0.0:
        rel_excess = m_atm / m_planet - 1.0
        if rel_excess > cap_tol:
            violations.append(
                InvariantViolation(
                    name='M_atm_planet_cap',
                    status_code=STATUS_ERROR_MASS_INVARIANT,
                    value=rel_excess,
                    tolerance=cap_tol,
                    message=(
                        f'Time={time_yr:.6e} yr: M_atm / M_planet - 1={rel_excess:.6e} '
                        f'exceeds tol={cap_tol:.6e} '
                        f'(M_atm={m_atm:.6e} kg, M_planet={m_planet:.6e} kg)'
                    ),
                    report_only=False,
                )
            )
    else:
        unexplained_rel = (m_atm - m_planet - m_vaps) / m_planet
        if unexplained_rel > cap_tol:
            violations.append(
                InvariantViolation(
                    name='M_atm_planet_cap',
                    status_code=STATUS_ERROR_MASS_INVARIANT,
                    value=unexplained_rel,
                    tolerance=cap_tol,
                    message=(
                        f'Time={time_yr:.6e} yr: (M_atm - M_planet - M_vaps) / M_planet='
                        f'{unexplained_rel:.6e} exceeds tol={cap_tol:.6e} '
                        f'(M_atm={m_atm:.6e} kg, M_planet={m_planet:.6e} kg, '
                        f'M_vaps={m_vaps:.6e} kg)'
                    ),
                    report_only=True,
                )
            )

    gas_tol = float(inv_cfg.volatile_gas_sum_rtol)
    summed = sum(float(hf_row.get(f'{s}_kg_atm', 0.0)) for s in vol_gas_list)
    if not (np.isfinite(m_vol_atm) and np.isfinite(summed)):
        violations.append(
            InvariantViolation(
                name='M_vol_atm_species_sum',
                status_code=STATUS_ERROR_MASS_INVARIANT,
                value=float('nan'),
                tolerance=gas_tol,
                message=(
                    f'Time={time_yr:.6e} yr: non-finite volatile species sum '
                    f'(M_vol_atm={m_vol_atm:.6e} kg, sum_s={summed:.6e} kg)'
                ),
                report_only=False,
            )
        )
    else:
        scale = max(m_vol_atm, summed)
        if scale > 0.0:
            rel_gas = abs(summed - m_vol_atm) / scale
            if rel_gas > gas_tol:
                violations.append(
                    InvariantViolation(
                        name='M_vol_atm_species_sum',
                        status_code=STATUS_ERROR_MASS_INVARIANT,
                        value=rel_gas,
                        tolerance=gas_tol,
                        message=(
                            f'Time={time_yr:.6e} yr: |sum_s(s_kg_atm) - M_vol_atm| / scale='
                            f'{rel_gas:.6e} exceeds tol={gas_tol:.6e} '
                            f'(M_vol_atm={m_vol_atm:.6e} kg, sum_s={summed:.6e} kg)'
                        ),
                        report_only=False,
                    )
                )

    closure_rtol = float(inv_cfg.element_closure_rtol)
    closure_atol_kg = float(inv_cfg.element_closure_atol_kg)
    derived = set(derived_total_elements(config))
    if vapourise:
        derived.add('O')
    for e in element_list:
        if e in derived:
            continue
        total = float(hf_row.get(f'{e}_kg_total', 0.0))
        if not np.isfinite(total):
            violations.append(
                InvariantViolation(
                    name=f'{e}_reservoir_closure',
                    status_code=STATUS_ERROR_MASS_INVARIANT,
                    value=float('nan'),
                    tolerance=closure_rtol,
                    message=f'Time={time_yr:.6e} yr: {e}_kg_total={total} is not finite',
                    report_only=False,
                )
            )
            continue
        if total <= 0.0:
            continue
        parts = sum(float(hf_row.get(f'{e}_kg_{r}', 0.0)) for r in ('atm', 'liquid', 'solid'))
        if not np.isfinite(parts):
            violations.append(
                InvariantViolation(
                    name=f'{e}_reservoir_closure',
                    status_code=STATUS_ERROR_MASS_INVARIANT,
                    value=float('nan'),
                    tolerance=closure_rtol,
                    message=f'Time={time_yr:.6e} yr: {e} reservoir sum={parts} is not finite',
                    report_only=False,
                )
            )
            continue
        diff = abs(parts - total)
        rel_closure = diff / total
        if diff > max(closure_atol_kg, closure_rtol * total):
            violations.append(
                InvariantViolation(
                    name=f'{e}_reservoir_closure',
                    status_code=STATUS_ERROR_MASS_INVARIANT,
                    value=rel_closure,
                    tolerance=closure_rtol,
                    message=(
                        f'Time={time_yr:.6e} yr: |({e}_atm+{e}_liquid+{e}_solid) - {e}_total| / '
                        f'{e}_total={rel_closure:.6e} exceeds tol={closure_rtol:.6e} '
                        f'(total={total:.6e} kg, reservoirs={parts:.6e} kg)'
                    ),
                    report_only=False,
                )
            )

    m_init_raw = hf_row.get('M_vol_initial', None)
    try:
        m_init = float(m_init_raw) if m_init_raw is not None else 0.0
    except (TypeError, ValueError):
        m_init = 0.0
    if np.isfinite(m_init) and m_init > 0.0:
        mass_thresh = float(getattr(getattr(config, 'outgas', None), 'mass_thresh', 1.0e10))
        fO2_source = getattr(getattr(config, 'planet', None), 'fO2_source', 'user_constant')
        all_vol_depleted = all(
            escapable_inventory(hf_row, e) <= mass_thresh
            for e in vol_element_list + noble_gases
        )
        if fO2_source == 'from_O_budget' or all_vol_depleted:
            removed = float(hf_row.get('M_desiccated', 0.0))
            if not np.isfinite(removed):
                removed = 0.0
            cur_m_ele = readable_total(hf_row)
            lost = m_init - cur_m_ele - removed
            esc_cum = float(hf_row.get('esc_kg_cumulative', 0.0))
            esc_tol = float(inv_cfg.escape_bound_factor) * esc_cum + float(
                inv_cfg.escape_bound_atol_kg
            )
            if not np.isfinite(lost) or not np.isfinite(esc_tol) or lost > esc_tol:
                violations.append(
                    InvariantViolation(
                        name='escape_cumulative_bound',
                        status_code=STATUS_ERROR_MASS_INVARIANT,
                        value=lost,
                        tolerance=esc_tol,
                        message=(
                            f'Time={time_yr:.6e} yr: volatile loss={lost:.6e} kg exceeds '
                            f'escape bound={esc_tol:.6e} kg '
                            f'(M_vol_initial={m_init:.6e} kg, cur_M_ele={cur_m_ele:.6e} kg, '
                            f'esc_kg_cumulative={esc_cum:.6e} kg)'
                        ),
                        report_only=False,
                    )
                )

    return violations
