"""Run-row physical invariant assertions and enforcement modes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from proteus.escape.wrapper import readable_total
from proteus.outgas.trapping import derived_total_elements
from proteus.utils.constants import (
    element_list,
    noble_gases,
    vap_element_list,
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
STEP_GROSS_FLOOR_J: float = 1.0
# 0.5 K margin above T(P_surf, S_min) bounds low-pressure P-S linear interpolation error (<= 9.5e-3 K)
# and static-mesh vs row P_surf drift (<= 0.071 K on S_min-clipped corpus rows; < 0.5 K to 1,748 bar).
EOS_SMIN_CLIP_MARGIN_K: float = 0.5
LAVATMOS_OVERWRITTEN_ELEMENTS: tuple[str, ...] = (*vol_element_list, *noble_gases)
_SMIN_TABLE_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, float]] = {}


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


_VALUE_CHANGE_RELOG_NAMES: frozenset[str] = frozenset(
    {
        'struct_mass_desync_frac',
        'M_core_target_rel_err',
        'M_int_target_rel_err',
    }
)
_CROSSING_SUPPRESSED_NAMES: frozenset[str] = _VALUE_CHANGE_RELOG_NAMES | frozenset(
    {
        'E_residual_cons_frac_gross',
        'escape_cumulative_bound',
    }
)


@dataclass
class InvariantTracker:
    """Per-run tracker that suppresses repeated cumulative warnings and summarizes counts and maxima.

    Attributes
    ----------
    active_names : set[str]
        Invariant names violated on the immediately preceding row.
    active_state : dict[tuple[str, bool], float]
        Mapping from ``(name, report_only)`` to the violation value on the preceding row.
    row_counts : dict[str, int]
        Total number of rows in the violating state per invariant name (including
        carried-forward rows between Zalmoxis structure updates).
    max_values : dict[str, float]
        Maximum observed finite magnitude ``|value|`` (or ``|value|`` when all
        observed values are non-finite) for each invariant name.
    nonfinite_counts : dict[str, int]
        Number of violation records with non-finite ``value`` per invariant name.
    rows_with_violations : int
        Total number of rows with at least one invariant violation.
    """

    active_names: set[str] = field(default_factory=set)
    active_state: dict[tuple[str, bool], float] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    max_values: dict[str, float] = field(default_factory=dict)
    nonfinite_counts: dict[str, int] = field(default_factory=dict)
    rows_with_violations: int = 0

    def record_row(self, violations: Sequence[InvariantViolation]) -> set[str]:
        """Record ``violations`` for the current row and return rising-edge names."""
        if not violations:
            self.active_names.clear()
            self.active_state.clear()
            return set()
        self.rows_with_violations += 1
        current = {v.name for v in violations}
        rising: set[str] = set()
        next_state: dict[tuple[str, bool], float] = {}
        for v in violations:
            key = (v.name, v.report_only)
            next_state[key] = v.value
            if key not in self.active_state:
                rising.add(v.name)
            else:
                prev_val = self.active_state[key]
                same_val = bool(np.isnan(v.value) and np.isnan(prev_val)) or v.value == prev_val
                if not same_val and (
                    v.name in _VALUE_CHANGE_RELOG_NAMES or not np.isfinite(v.value)
                ):
                    rising.add(v.name)
        for name in current:
            self.row_counts[name] = self.row_counts.get(name, 0) + 1
        for v in violations:
            mag = abs(v.value)
            if np.isfinite(v.value):
                prev = self.max_values.get(v.name)
                if prev is None or not np.isfinite(prev) or mag > prev:
                    self.max_values[v.name] = mag
            else:
                self.nonfinite_counts[v.name] = self.nonfinite_counts.get(v.name, 0) + 1
                if v.name not in self.max_values:
                    self.max_values[v.name] = mag
        self.active_names = current
        self.active_state = next_state
        return rising

    def log_summary(self) -> None:
        """Emit a single end-of-run summary line with count and maximum value per invariant name."""
        if not self.row_counts:
            return
        parts: list[str] = []
        for name, cnt in self.row_counts.items():
            max_val = self.max_values.get(name, 0.0)
            nf = self.nonfinite_counts.get(name, 0)
            if nf > 0:
                parts.append(f'{name}={cnt} (max={max_val:.6e}, non_finite={nf})')
            else:
                parts.append(f'{name}={cnt} (max={max_val:.6e})')
        log.warning(
            'Invariant summary: %d row(s) with violations (%s)',
            self.rows_with_violations,
            ', '.join(parts),
        )


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


def _emit_report_warning(
    v: InvariantViolation,
    rising: set[str],
    tracker: InvariantTracker | None,
) -> None:
    """Emit a warning for ``v``, suppressing repeated cumulative rows while keeping per-row logs."""
    if v.name in _CROSSING_SUPPRESSED_NAMES:
        if v.name not in rising:
            return
        if tracker is not None:
            log.warning(
                'Invariant report [%s]: %s (count=%d)',
                v.name,
                v.message,
                tracker.row_counts.get(v.name, 1),
            )
            return
    log.warning('Invariant report [%s]: %s', v.name, v.message)


def handle_violations(
    violations: Sequence[InvariantViolation],
    mode: str = 'report',
    directories: dict | None = None,
    tracker: InvariantTracker | None = None,
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
    tracker : InvariantTracker or None, optional
        Optional per-run tracker suppressing duplicate cumulative log lines on consecutive rows.

    Returns
    -------
    list[InvariantViolation]
        List of processed violations (empty when ``mode == 'off'``).

    Raises
    ------
    InvariantViolationError
        When ``mode == 'stop'`` and at least one violation has ``report_only=False``.
    """
    if mode == 'off':
        return []
    if not violations:
        if tracker is not None:
            tracker.record_row(())
        return []

    if mode not in ('report', 'stop'):
        raise ValueError(
            f"Unknown invariants.mode {mode!r}; expected 'off', 'report', or 'stop'."
        )

    rising = (
        tracker.record_row(violations) if tracker is not None else {v.name for v in violations}
    )

    if mode == 'report':
        for v in violations:
            _emit_report_warning(v, rising, tracker)
        return list(violations)

    stopping: list[InvariantViolation] = []
    for v in violations:
        if v.report_only:
            _emit_report_warning(v, rising, tracker)
        else:
            log.error('Invariant violation [%s]: %s', v.name, v.message)
            stopping.append(v)

    if stopping:
        primary = stopping[0]
        if directories is not None and 'output' in directories:
            UpdateStatusfile(directories, primary.status_code)
        raise InvariantViolationError(primary, stopping)

    return list(violations)


_ENERGY_FINITE_INPUT_KEYS: tuple[str, ...] = (
    'E_state_cons_J',
    'E_state_heat_cons_J',
    'step_dE_state_heat_J',
    'step_dE_F_int_J',
    'step_dE_F_cmb_J',
    'step_dE_Q_radio_J',
    'step_dE_Q_tidal_J',
    'step_dE_core_J',
    'core_C_eff',
)


def _resolve_ps_table_path(directories: dict | None) -> Path | None:
    """Resolve the active run's ``temperature_solid.dat`` path from ``directories``."""
    if not directories:
        return None
    eos_dir = directories.get('spider_eos_dir')
    if eos_dir:
        cand = Path(eos_dir) / 'temperature_solid.dat'
        if cand.is_file():
            return cand.resolve()
    data_dir = directories.get('output/data')
    if not data_dir and directories.get('output'):
        data_dir = str(Path(directories['output']) / 'data')
    if data_dir:
        cand = Path(data_dir) / 'spider_eos' / 'temperature_solid.dat'
        if cand.is_file():
            return cand.resolve()
        ptr = Path(data_dir) / 'spider_eos_cache.txt'
        if ptr.is_file():
            target_dir = ptr.read_text(encoding='utf-8').strip()
            if target_dir:
                cand_ptr = Path(target_dir) / 'temperature_solid.dat'
                if cand_ptr.is_file():
                    return cand_ptr.resolve()
    return None


def _parse_smin_temp_block(table_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Parse ``temperature_solid.dat`` once and return ``(P_pa, T_k, S_min)`` for the ``S_min`` block.

    Parameters
    ----------
    table_path : Path
        Path to ``temperature_solid.dat``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        1D arrays of pressure in Pa and temperature in K on the ``S_min`` block,
        plus ``S_min`` in J/(kg K).

    Raises
    ------
    ValueError
        If the first ``n_P`` block does not hold the minimum entropy ``S_min`` of the table.
    """
    with open(table_path, encoding='utf-8') as f:
        hdr = f.readline().strip().lstrip('#').split()
        n_hdr, n_p = int(hdr[0]), int(hdr[1])
        scale_line = ''
        for _ in range(n_hdr - 1):
            scale_line = f.readline()
        scales = [float(x) for x in scale_line.strip().lstrip('#').split()]
    data = np.loadtxt(table_path, skiprows=n_hdr)
    first_block = data[:n_p]
    s_all = data[:, 1] * scales[1]
    s_first = first_block[:, 1] * scales[1]
    s_min = float(np.min(s_all))
    if not np.allclose(s_first, s_min, rtol=0.0, atol=1.0e-3):
        raise ValueError(
            f'First block in {table_path} has S={float(s_first[0]):.6f}, not S_min={s_min:.6f}'
        )
    p_pa = np.asarray(first_block[:, 0], dtype=float) * scales[0]
    t_k = np.asarray(first_block[:, 2], dtype=float) * scales[2]
    return p_pa, t_k, s_min


def _smin_clip_temp_k(hf_row: dict, directories: dict | None) -> float | None:
    """Evaluate ``T(P_surf, S_min) + EOS_SMIN_CLIP_MARGIN_K`` at the row's ``P_surf``."""
    table_path = _resolve_ps_table_path(directories)
    if table_path is None:
        return None
    stat = table_path.stat()
    cache_key = (str(table_path), stat.st_mtime_ns)
    cached = _SMIN_TABLE_CACHE.get(cache_key)
    if cached is None:
        cached = _parse_smin_temp_block(table_path)
        _SMIN_TABLE_CACHE[cache_key] = cached
    p_pa, t_k, _ = cached
    p_surf_bar = float(hf_row.get('P_surf', 1.0))
    if not np.isfinite(p_surf_bar):
        return None
    p_surf_pa = float(np.clip(p_surf_bar * 1.0e5, p_pa[0], p_pa[-1]))
    t_smin_k = float(np.interp(p_surf_pa, p_pa, t_k))
    return t_smin_k + EOS_SMIN_CLIP_MARGIN_K


def check_energy_invariants(
    hf_row: dict,
    config: Config,
    hf_all: pd.DataFrame | None = None,
    directories: dict | None = None,
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
    directories : dict or None, optional
        PROTEUS run directories dictionary used to locate and cache the run's P-S
        EOS table ``temperature_solid.dat`` for the ``S_min`` clip threshold.

    Returns
    -------
    list[InvariantViolation]
        Detected energy invariant violations (all with ``report_only=True``).
    """
    if hf_all is None or len(hf_all) == 0:
        return []

    e_state_cons = float(hf_row.get('E_state_cons_J', 0.0))
    if e_state_cons == 0.0:
        return []

    time_yr = float(hf_row.get('Time', 0.0))
    violations: list[InvariantViolation] = []
    for key in _ENERGY_FINITE_INPUT_KEYS:
        if key not in hf_row:
            continue
        raw_val = float(hf_row.get(key, 0.0))
        if not np.isfinite(raw_val):
            violations.append(
                InvariantViolation(
                    name=key,
                    status_code=STATUS_ERROR_ENERGY_INVARIANT,
                    value=raw_val,
                    tolerance=0.0,
                    message=f'Time={time_yr:.6e} yr: {key}={raw_val} is not finite',
                    report_only=True,
                )
            )

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
    for col in (
        'step_dE_F_int_J',
        'step_dE_F_cmb_J',
        'step_dE_Q_radio_J',
        'step_dE_Q_tidal_J',
    ):
        if col in hf_all:
            hist_gross += float(
                np.sum(np.abs(np.nan_to_num(hf_all[col].to_numpy(dtype=float), nan=0.0)))
            )

    e_state_heat = float(hf_row.get('E_state_heat_cons_J', 0.0))
    abs_e_state_heat = abs(e_state_heat) if np.isfinite(e_state_heat) else 0.0
    denom_gross = max(abs_e_state_heat, hist_gross + step_gross, 1.0)
    has_prior_throughput = hist_gross > 0.0

    inv_cfg = config.invariants
    eff_energy_rtol = float(inv_cfg.energy_residual_rtol)

    e_res_j = float(hf_row.get('E_residual_cons_J', 0.0))
    e_res_frac_gross = abs(e_res_j) / denom_gross if np.isfinite(e_res_j) else float('nan')
    if not np.isfinite(e_res_frac_gross) or (
        has_prior_throughput and e_res_frac_gross > eff_energy_rtol
    ):
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
        denom_step = max(step_gross, STEP_GROSS_FLOOR_J)
        step_err = step_diff / denom_step if np.isfinite(step_diff) else float('nan')
        t_magma = float(hf_row.get('T_magma', 1000.0))
        smin_clip_k = _smin_clip_temp_k(hf_row, directories)
        on_phase_kink = (
            smin_clip_k is not None and np.isfinite(t_magma) and t_magma <= smin_clip_k
        )
        step_tol = (
            float(inv_cfg.step_energy_smin_rtol)
            if on_phase_kink
            else float(inv_cfg.step_energy_rtol)
        )
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
                        f'dE_inc_cons={dE_inc_cons:.6e} J, step_gross={step_gross:.6e} J)'
                    ),
                    report_only=True,
                )
            )

    sol_res_j = float(hf_row.get('solver_residual_J', 0.0))
    sol_res_frac = abs(sol_res_j) / denom_gross if np.isfinite(sol_res_j) else float('nan')
    sol_tol = float(inv_cfg.solver_residual_rtol)
    if not np.isfinite(sol_res_frac) or (has_prior_throughput and sol_res_frac > sol_tol):
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
    if core_c_eff != 0.0:
        core_res_raw = float(hf_row.get('E_core_residual_frac', 0.0))
        core_res_frac = abs(core_res_raw) if np.isfinite(core_res_raw) else float('nan')
        core_tol = float(inv_cfg.core_residual_rtol)
        if not np.isfinite(core_res_frac) or (
            has_prior_throughput and core_res_frac > core_tol
        ):
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

    Enforces non-negative reservoir bounds (``{e}_kg_{r}_bounds``) and finite reservoir/total
    masses across all elements in ``element_list``. Element reservoir closure skips oxygen
    under ``planet.fO2_source = 'user_constant'`` and, when ``outgas.vapourise`` is enabled,
    skips ``vap_element_list`` (``Si, Mg, Fe, Na, Al, Ti, Ca, K, V``, whose totals stay zero
    while rock vapour enters the atmosphere) and ``LAVATMOS_OVERWRITTEN_ELEMENTS``
    (``H, C, N, S, O`` and ``noble_gases``, whose atmospheric element masses ``lavatmos``
    overwrites from gas-phase equilibrium without updating the element totals).

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
    m_atm = float(hf_row.get('M_atm', 0.0))
    m_vol_atm = float(hf_row.get('M_vol_atm', 0.0))
    m_vaps = float(hf_row.get('M_vaps', 0.0))
    vapourise = bool(getattr(getattr(config, 'outgas', None), 'vapourise', False))
    closure_rtol = float(inv_cfg.element_closure_rtol)
    closure_atol_kg = float(inv_cfg.element_closure_atol_kg)
    violations: list[InvariantViolation] = []

    if np.isfinite(m_planet) and m_planet <= 0.0:
        violations.append(
            InvariantViolation(
                name='M_planet_bounds',
                status_code=STATUS_ERROR_MASS_INVARIANT,
                value=m_planet,
                tolerance=0.0,
                message=(
                    f'Time={time_yr:.6e} yr: M_planet={m_planet:.6e} kg violates M_planet > 0.0 kg'
                ),
                report_only=False,
            )
        )

    if np.isfinite(m_atm) and m_atm < -closure_atol_kg:
        violations.append(
            InvariantViolation(
                name='M_atm_bounds',
                status_code=STATUS_ERROR_MASS_INVARIANT,
                value=m_atm,
                tolerance=closure_atol_kg,
                message=(
                    f'Time={time_yr:.6e} yr: M_atm={m_atm:.6e} kg < '
                    f'-element_closure_atol_kg ({-closure_atol_kg:.6e} kg)'
                ),
                report_only=False,
            )
        )

    if np.isfinite(m_vol_atm) and m_vol_atm < -closure_atol_kg:
        violations.append(
            InvariantViolation(
                name='M_vol_atm_bounds',
                status_code=STATUS_ERROR_MASS_INVARIANT,
                value=m_vol_atm,
                tolerance=closure_atol_kg,
                message=(
                    f'Time={time_yr:.6e} yr: M_vol_atm={m_vol_atm:.6e} kg < '
                    f'-element_closure_atol_kg ({-closure_atol_kg:.6e} kg)'
                ),
                report_only=False,
            )
        )

    cap_tol = float(inv_cfg.planet_mass_cap_rtol)
    m_atm_eff = 0.0 if (np.isfinite(m_atm) and abs(m_atm) <= closure_atol_kg) else m_atm
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
    elif m_planet > 0.0 and (not vapourise or m_vaps <= 0.0):
        rel_excess = m_atm_eff / m_planet - 1.0
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
    elif m_planet > 0.0:
        unexplained_rel = (m_atm_eff - m_planet - m_vaps) / m_planet
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
    summed = 0.0
    for s in vol_gas_list:
        s_mass = float(hf_row.get(f'{s}_kg_atm', 0.0))
        summed += s_mass
        if np.isfinite(s_mass) and s_mass < -closure_atol_kg:
            violations.append(
                InvariantViolation(
                    name=f'{s}_kg_atm_bounds',
                    status_code=STATUS_ERROR_MASS_INVARIANT,
                    value=s_mass,
                    tolerance=closure_atol_kg,
                    message=(
                        f'Time={time_yr:.6e} yr: {s}_kg_atm={s_mass:.6e} kg < '
                        f'-element_closure_atol_kg ({-closure_atol_kg:.6e} kg)'
                    ),
                    report_only=False,
                )
            )
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
        else:
            diff_gas = abs(summed - m_vol_atm)
            if diff_gas > closure_atol_kg:
                violations.append(
                    InvariantViolation(
                        name='M_vol_atm_species_sum',
                        status_code=STATUS_ERROR_MASS_INVARIANT,
                        value=diff_gas,
                        tolerance=closure_atol_kg,
                        message=(
                            f'Time={time_yr:.6e} yr: |sum_s(s_kg_atm) - M_vol_atm|='
                            f'{diff_gas:.6e} kg exceeds element_closure_atol_kg='
                            f'{closure_atol_kg:.6e} kg with non-positive scale={scale:.6e} kg '
                            f'(M_vol_atm={m_vol_atm:.6e} kg, sum_s={summed:.6e} kg)'
                        ),
                        report_only=False,
                    )
                )

    outgas_cfg = getattr(config, 'outgas', None)
    is_calliope = getattr(outgas_cfg, 'module', None) == 'calliope'
    calliope_mass_thresh = float(outgas_cfg.mass_thresh) if is_calliope else 0.0
    derived = set(derived_total_elements(config))
    if vapourise:
        derived.update(vap_element_list)
        derived.update(LAVATMOS_OVERWRITTEN_ELEMENTS)
    for e in element_list:
        for r in ('atm', 'liquid', 'solid'):
            r_mass = float(hf_row.get(f'{e}_kg_{r}', 0.0))
            if np.isfinite(r_mass) and r_mass < -closure_atol_kg:
                violations.append(
                    InvariantViolation(
                        name=f'{e}_kg_{r}_bounds',
                        status_code=STATUS_ERROR_MASS_INVARIANT,
                        value=r_mass,
                        tolerance=closure_atol_kg,
                        message=(
                            f'Time={time_yr:.6e} yr: {e}_kg_{r}={r_mass:.6e} kg < '
                            f'-element_closure_atol_kg ({-closure_atol_kg:.6e} kg)'
                        ),
                        report_only=False,
                    )
                )
        total = float(hf_row.get(f'{e}_kg_total', 0.0))
        if not np.isfinite(total):
            violations.append(
                InvariantViolation(
                    name=f'{e}_reservoir_closure',
                    status_code=STATUS_ERROR_MASS_INVARIANT,
                    value=total,
                    tolerance=closure_rtol,
                    message=f'Time={time_yr:.6e} yr: {e}_kg_total={total} is not finite',
                    report_only=False,
                )
            )
            continue
        parts = sum(float(hf_row.get(f'{e}_kg_{r}', 0.0)) for r in ('atm', 'liquid', 'solid'))
        if not np.isfinite(parts):
            violations.append(
                InvariantViolation(
                    name=f'{e}_reservoir_closure',
                    status_code=STATUS_ERROR_MASS_INVARIANT,
                    value=parts,
                    tolerance=closure_rtol,
                    message=f'Time={time_yr:.6e} yr: {e} reservoir sum={parts} is not finite',
                    report_only=False,
                )
            )
            continue
        if e in derived:
            continue
        if total < -closure_atol_kg:
            violations.append(
                InvariantViolation(
                    name=f'{e}_reservoir_closure',
                    status_code=STATUS_ERROR_MASS_INVARIANT,
                    value=total,
                    tolerance=closure_atol_kg,
                    message=(
                        f'Time={time_yr:.6e} yr: {e}_kg_total={total:.6e} kg < '
                        f'-element_closure_atol_kg ({-closure_atol_kg:.6e} kg)'
                    ),
                    report_only=False,
                )
            )
            continue
        diff = abs(parts - total)
        pos_total = max(total, 0.0)
        tol_e = (
            calliope_mass_thresh + closure_rtol * pos_total
            if is_calliope
            else max(closure_atol_kg, closure_rtol * pos_total)
        )
        if diff > tol_e:
            if total > 0.0:
                rel_closure = diff / total
                eff_rel_tol = tol_e / total
                violations.append(
                    InvariantViolation(
                        name=f'{e}_reservoir_closure',
                        status_code=STATUS_ERROR_MASS_INVARIANT,
                        value=rel_closure,
                        tolerance=eff_rel_tol,
                        message=(
                            f'Time={time_yr:.6e} yr: |({e}_atm+{e}_liquid+{e}_solid) - {e}_total| / '
                            f'{e}_total={rel_closure:.6e} exceeds tol={eff_rel_tol:.6e} '
                            f'(total={total:.6e} kg, reservoirs={parts:.6e} kg)'
                        ),
                        report_only=False,
                    )
                )
            else:
                violations.append(
                    InvariantViolation(
                        name=f'{e}_reservoir_closure',
                        status_code=STATUS_ERROR_MASS_INVARIANT,
                        value=diff,
                        tolerance=tol_e,
                        message=(
                            f'Time={time_yr:.6e} yr: |({e}_atm+{e}_liquid+{e}_solid) - {e}_total|='
                            f'{diff:.6e} kg exceeds tol={tol_e:.6e} kg '
                            f'(total={total:.6e} kg, reservoirs={parts:.6e} kg)'
                        ),
                        report_only=False,
                    )
                )

    m_init = float(hf_row.get('M_vol_initial', 0.0) or 0.0)
    if np.isfinite(m_init) and m_init > 0.0:
        fO2_source = getattr(getattr(config, 'planet', None), 'fO2_source', 'user_constant')
        if fO2_source == 'from_O_budget':
            removed = float(hf_row.get('M_desiccated', 0.0))
            cur_m_ele = readable_total(hf_row)
            lost = m_init - cur_m_ele - removed
            esc_cum = float(hf_row.get('esc_kg_cumulative', 0.0))
            esc_tol = float(inv_cfg.escape_bound_factor) * esc_cum + float(
                inv_cfg.escape_bound_atol_kg
            )
            if not np.isfinite(lost) or not np.isfinite(esc_tol) or lost > esc_tol:
                if not np.isfinite(removed):
                    msg = (
                        f'Time={time_yr:.6e} yr: M_desiccated={removed} is not finite '
                        f'in escape_cumulative_bound '
                        f'(M_vol_initial={m_init:.6e} kg, cur_M_ele={cur_m_ele:.6e} kg)'
                    )
                else:
                    msg = (
                        f'Time={time_yr:.6e} yr: volatile loss={lost:.6e} kg exceeds '
                        f'escape bound={esc_tol:.6e} kg '
                        f'(M_vol_initial={m_init:.6e} kg, cur_M_ele={cur_m_ele:.6e} kg, '
                        f'M_desiccated={removed:.6e} kg, esc_kg_cumulative={esc_cum:.6e} kg)'
                    )
                violations.append(
                    InvariantViolation(
                        name='escape_cumulative_bound',
                        status_code=STATUS_ERROR_MASS_INVARIANT,
                        value=lost,
                        tolerance=esc_tol,
                        message=msg,
                        report_only=True,
                    )
                )

    return violations


def check_state_bounds_invariants(
    hf_row: dict,
    config: Config,
) -> list[InvariantViolation]:
    """Evaluate physical state bounds on ``T_surf``, ``T_magma``, ``P_surf``, and ``Phi_global`` (status 32).

    Parameters
    ----------
    hf_row : dict
        Current helpfile row dictionary.
    config : Config
        PROTEUS configuration object.

    Returns
    -------
    list[InvariantViolation]
        Detected state-bounds violations.
    """
    inv_cfg = config.invariants
    time_yr = float(hf_row.get('Time', 0.0))
    violations: list[InvariantViolation] = []

    t_surf = float(hf_row.get('T_surf', float('nan')))
    if not (np.isfinite(t_surf) and t_surf > 0.0):
        violations.append(
            InvariantViolation(
                name='T_surf_bounds',
                status_code=STATUS_ERROR_STATE_BOUNDS_INVARIANT,
                value=t_surf,
                tolerance=0.0,
                message=f'Time={time_yr:.6e} yr: T_surf={t_surf:.6e} K violates T_surf > 0.0 K',
                report_only=False,
            )
        )

    t_magma = float(hf_row.get('T_magma', float('nan')))
    t_magma_max = float(inv_cfg.t_magma_max)
    if not (np.isfinite(t_magma) and 0.0 < t_magma < t_magma_max):
        violations.append(
            InvariantViolation(
                name='T_magma_bounds',
                status_code=STATUS_ERROR_STATE_BOUNDS_INVARIANT,
                value=t_magma,
                tolerance=t_magma_max,
                message=(
                    f'Time={time_yr:.6e} yr: T_magma={t_magma:.6e} K violates '
                    f'0.0 < T_magma < {t_magma_max:.6e} K'
                ),
                report_only=False,
            )
        )

    closure_atol_kg = float(inv_cfg.element_closure_atol_kg)
    m_atm = float(hf_row.get('M_atm', 0.0))
    p_surf = float(hf_row.get('P_surf', float('nan')))
    if not np.isfinite(m_atm) or m_atm < -closure_atol_kg:
        valid_p = False
        req = f'>= 0.0 bar with M_atm >= {-closure_atol_kg:.6e} kg'
    elif abs(m_atm) <= closure_atol_kg:
        valid_p = np.isfinite(p_surf) and p_surf >= 0.0
        req = '>= 0.0 bar'
    else:
        valid_p = np.isfinite(p_surf) and p_surf > 0.0
        req = '> 0.0 bar'
    if not valid_p:
        violations.append(
            InvariantViolation(
                name='P_surf_bounds',
                status_code=STATUS_ERROR_STATE_BOUNDS_INVARIANT,
                value=p_surf,
                tolerance=0.0,
                message=(
                    f'Time={time_yr:.6e} yr: P_surf={p_surf:.6e} bar violates '
                    f'P_surf {req} (M_atm={m_atm:.6e} kg)'
                ),
                report_only=False,
            )
        )

    phi_atol = float(inv_cfg.phi_global_atol)
    phi_global = float(hf_row.get('Phi_global', float('nan')))
    if not (np.isfinite(phi_global) and -phi_atol <= phi_global <= 1.0 + phi_atol):
        violations.append(
            InvariantViolation(
                name='Phi_global_bounds',
                status_code=STATUS_ERROR_STATE_BOUNDS_INVARIANT,
                value=phi_global,
                tolerance=phi_atol,
                message=(
                    f'Time={time_yr:.6e} yr: Phi_global={phi_global:.6e} violates '
                    f'[-{phi_atol:.6e}, {1.0 + phi_atol:.6e}]'
                ),
                report_only=False,
            )
        )

    return violations


def check_row_invariants(
    hf_row: dict,
    config: Config,
    hf_all: pd.DataFrame | None = None,
    directories: dict | None = None,
    tracker: InvariantTracker | None = None,
) -> list[InvariantViolation]:
    """Evaluate all run-row invariants on ``hf_row`` and dispatch according to ``config.invariants.mode``.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row dictionary before appending to ``hf_all``.
    config : Config
        PROTEUS configuration object.
    hf_all : pd.DataFrame or None, optional
        Accumulated helpfile table prior to appending ``hf_row``.
    directories : dict or None, optional
        PROTEUS run directories dictionary for status-file updates under ``mode == 'stop'``.
    tracker : InvariantTracker or None, optional
        Cross-row state tracker for rising-edge report logging and total-row counting.

    Returns
    -------
    list[InvariantViolation]
        Processed violations (empty when ``mode == 'off'``).
    """
    inv_cfg = getattr(config, 'invariants', None)
    mode = getattr(inv_cfg, 'mode', 'report') if inv_cfg is not None else 'report'
    if mode == 'off':
        return []

    violations: list[InvariantViolation] = []
    violations.extend(check_energy_invariants(hf_row, config, hf_all, directories=directories))
    violations.extend(check_structure_invariants(hf_row, config))
    violations.extend(check_volatile_invariants(hf_row, config))
    violations.extend(check_state_bounds_invariants(hf_row, config))
    return handle_violations(violations, mode=mode, directories=directories, tracker=tracker)
