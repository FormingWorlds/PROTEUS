"""Unit tests for run-row invariant assertions in ``proteus.utils.invariants``.

Covers status code registration (30, 31, 32), ``InvariantViolationError``,
mode handling (``'off'``, ``'report'``, ``'stop'``), and the individual
coupled energy, structure mass, volatile inventory, and physical state-bounds
invariant checks.
"""

from __future__ import annotations

import logging

import pytest

from proteus.utils.helper import CommentFromStatus
from proteus.utils.invariants import (
    STATUS_ERROR_ENERGY_INVARIANT,
    STATUS_ERROR_MASS_INVARIANT,
    STATUS_ERROR_STATE_BOUNDS_INVARIANT,
    InvariantViolation,
    InvariantViolationError,
    check_energy_invariants,
    handle_violations,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_comment_from_status_maps_invariant_codes_30_31_32():
    """Status codes 30, 31, and 32 map to their named invariant error descriptions."""
    assert STATUS_ERROR_ENERGY_INVARIANT == 30
    assert STATUS_ERROR_MASS_INVARIANT == 31
    assert STATUS_ERROR_STATE_BOUNDS_INVARIANT == 32
    assert CommentFromStatus(30) == 'Error (Energy conservation invariant)'
    assert CommentFromStatus(31) == 'Error (Mass conservation invariant)'
    assert CommentFromStatus(32) == 'Error (State bounds invariant)'


def test_handle_violations_off_mode_suppresses_logs_and_errors(tmp_path, caplog):
    """In ``'off'`` mode, violations produce no log messages, status updates, or errors."""
    violation = InvariantViolation(
        name='struct_mass_desync_frac',
        status_code=STATUS_ERROR_MASS_INVARIANT,
        value=0.025,
        tolerance=0.01,
        message='struct_mass_desync_frac=2.500000e-02 exceeds 1.000000e-02',
        report_only=False,
    )
    dirs = {'output': str(tmp_path)}
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        out = handle_violations([violation], mode='off', directories=dirs)
    assert out == []
    assert caplog.records == []
    assert not (tmp_path / 'status').exists()


def test_handle_violations_report_mode_warns_without_raising(tmp_path, caplog):
    """In ``'report'`` mode, violations log a ``WARNING`` with values and do not stop."""
    violation = InvariantViolation(
        name='struct_mass_desync_frac',
        status_code=STATUS_ERROR_MASS_INVARIANT,
        value=0.025,
        tolerance=0.01,
        message='struct_mass_desync_frac=2.500000e-02 exceeds 1.000000e-02',
        report_only=False,
    )
    dirs = {'output': str(tmp_path)}
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        out = handle_violations([violation], mode='report', directories=dirs)
    assert len(out) == 1
    assert out[0] == violation
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'WARNING'
    assert 'struct_mass_desync_frac' in caplog.records[0].message
    assert '2.500000e-02' in caplog.records[0].message
    assert not (tmp_path / 'status').exists()


def test_handle_violations_stop_mode_handles_report_only_vs_stopping(tmp_path, caplog):
    """In ``'stop'`` mode, ``report_only`` violations only warn while stopping ones raise."""
    energy_v = InvariantViolation(
        name='E_residual_cons_frac_gross',
        status_code=STATUS_ERROR_ENERGY_INVARIANT,
        value=4.0e-4,
        tolerance=2.6e-4,
        message='E_residual_cons_frac_gross=4.000000e-04 exceeds 2.600000e-04',
        report_only=True,
    )
    dirs = {'output': str(tmp_path)}
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        out_warn = handle_violations([energy_v], mode='stop', directories=dirs)
    assert out_warn == [energy_v]
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'WARNING'
    assert not (tmp_path / 'status').exists()

    caplog.clear()
    mass_v = InvariantViolation(
        name='struct_mass_desync_frac',
        status_code=STATUS_ERROR_MASS_INVARIANT,
        value=0.025,
        tolerance=0.01,
        message='struct_mass_desync_frac=2.500000e-02 exceeds 1.000000e-02',
        report_only=False,
    )
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        with pytest.raises(InvariantViolationError) as exc_info:
            handle_violations([energy_v, mass_v], mode='stop', directories=dirs)
    err = exc_info.value
    assert err.status_code == STATUS_ERROR_MASS_INVARIANT
    assert err.invariant_name == 'struct_mass_desync_frac'
    assert err.value == pytest.approx(0.025)
    assert err.tolerance == pytest.approx(0.01)
    assert (tmp_path / 'status').exists()
    status_lines = (tmp_path / 'status').read_text().splitlines()
    assert status_lines[0].strip() == '31'
    assert status_lines[1].strip() == 'Error (Mass conservation invariant)'


def _make_mock_config(
    *,
    mode: str = 'report',
    dt_rtol: float = 0.1,
    dt_minimum: float = 1.0,
    fO2_source: str = 'user_constant',
    vapourise: bool = False,
    solver_rtol: float = 1.0e-4,
    mass_thresh: float = 1.0e10,
    outgas_module: str = 'atmodeller',
):
    """Build a minimal config object carrying ``invariants``, ``params.dt``, ``planet``, ``outgas``."""
    from types import SimpleNamespace

    from proteus.config._invariants import Invariants

    return SimpleNamespace(
        invariants=Invariants(mode=mode),
        params=SimpleNamespace(dt=SimpleNamespace(rtol=dt_rtol, minimum=dt_minimum)),
        planet=SimpleNamespace(fO2_source=fO2_source),
        outgas=SimpleNamespace(
            module=outgas_module,
            vapourise=vapourise,
            solver_rtol=solver_rtol,
            mass_thresh=mass_thresh,
        ),
    )


def test_check_energy_invariants_row4_flux_cancellation_passes_under_gross_denom():
    """Opposite-sign F_int and F_cmb on row 4 pass when normalized by gross throughput."""
    import pandas as pd

    cfg = _make_mock_config(dt_rtol=0.1, dt_minimum=1.0)
    hf_all = pd.DataFrame(
        [
            {
                'Time': 1.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 0.0,
                'step_dE_F_int_J': 0.0,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    # Net state heat is 2.48e27 J while gross flux sum is 1.116e28 J;
    # raw |E_residual_cons_frac| = 7.98e-4 > 2.6e-4, but gross ratio is 1.77e-4 <= 2.6e-4.
    hf_row = {
        'Time': 2.0,
        'T_surf': 3340.0,
        'T_magma': 4268.0,
        'Phi_global': 1.0,
        'RF_depth': 0.0,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 2.48e27,
        'dE_predicted_cons_J': 2.47802e27,
        'E_residual_cons_J': 1.98e24,
        'E_residual_cons_frac': 1.98e24 / 2.48e27,
        'step_dE_state_heat_J': 2.48e27,
        'step_dE_F_int_J': -4.34e27,
        'step_dE_F_cmb_J': 6.82e27,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'solver_residual_J': 1.0e12,
        'core_C_eff': 0.0,
        'E_core_residual_frac': 0.0,
    }
    assert check_energy_invariants(hf_row, cfg, hf_all) == []


def test_check_energy_invariants_coupled_residual_and_coarse_dt_scaling(tmp_path):
    """Coupled energy residual fires above 2.6e-4 for all time-step settings and is report-only."""
    import pandas as pd

    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': -1.0e30,
                'step_dE_F_int_J': -1.0e30,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    hf_row = {
        'Time': 20.0,
        'T_surf': 2500.0,
        'T_magma': 3000.0,
        'Phi_global': 1.0,
        'RF_depth': 0.0,
        'E_state_cons_J': 9.0e30,
        'E_state_heat_cons_J': -2.0e30,
        'dE_predicted_cons_J': -2.0007e30,
        'E_residual_cons_J': 7.0e26,  # 7.0e26 / 2.0e30 = 3.5e-4
        'E_residual_cons_frac': -3.5e-4,
        'step_dE_state_heat_J': -1.0e30,
        'step_dE_F_int_J': -1.0e30,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
        'E_core_residual_frac': 0.0,
    }

    cfg_std = _make_mock_config(mode='stop', dt_rtol=0.1, dt_minimum=1.0)
    v_std = check_energy_invariants(hf_row, cfg_std, hf_all)
    assert len(v_std) == 1
    assert v_std[0].name == 'E_residual_cons_frac_gross'
    assert v_std[0].status_code == STATUS_ERROR_ENERGY_INVARIANT
    assert v_std[0].report_only is True
    assert v_std[0].value == pytest.approx(3.5e-4)
    assert v_std[0].tolerance == pytest.approx(2.6e-4)
    assert '3.500000e-04' in v_std[0].message
    assert '2.600000e-04' in v_std[0].message

    # Energy violations never stop a run even under mode == 'stop'
    dirs = {'output': str(tmp_path)}
    out = handle_violations(v_std, mode='stop', directories=dirs)
    assert out == v_std
    assert not (tmp_path / 'status').exists()

    # A minimal.toml row (dt_rtol=0.2, dt_minimum=100.0) at gross residual 4.7e-4
    # must be reported against the single 2.6e-4 report threshold.
    cfg_minimal = _make_mock_config(mode='stop', dt_rtol=0.2, dt_minimum=100.0)
    hf_row_minimal = dict(hf_row, E_residual_cons_J=9.4e26)  # 9.4e26 / 2.0e30 = 4.7e-4 > 2.6e-4
    v_minimal = check_energy_invariants(hf_row_minimal, cfg_minimal, hf_all)
    assert len(v_minimal) == 1
    assert v_minimal[0].name == 'E_residual_cons_frac_gross'
    assert v_minimal[0].value == pytest.approx(4.7e-4)
    assert v_minimal[0].tolerance == pytest.approx(2.6e-4)
    assert v_minimal[0].report_only is True


def test_check_energy_invariants_step_increment_outside_and_on_smin():
    """Step energy increment checks 1.0e-4 in smooth single-phase and 2.5e-4 on S_min."""
    import pandas as pd

    cfg = _make_mock_config(dt_rtol=0.1, dt_minimum=1.0)
    hf_all = pd.DataFrame(
        [
            {
                'Time': 100.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': -1.0e30,
                'step_dE_F_int_J': -1.0e30,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    # Step increment mismatch is 3.6e26 / 2.0e30 = 1.8e-4 (> 1.0e-4, <= 2.5e-4),
    # while cumulative E_residual_cons_J is kept at 1.8e-4 (<= 2.6e-4).
    hf_row_warm = {
        'Time': 200.0,
        'T_surf': 2200.0,
        'T_magma': 2800.0,
        'Phi_global': 1.0,
        'RF_depth': 0.0,
        'E_state_cons_J': 9.0e30,
        'E_state_heat_cons_J': -2.0e30,
        'dE_predicted_cons_J': -2.00036e30,
        'E_residual_cons_J': 3.6e26,
        'E_residual_cons_frac': -1.8e-4,
        'step_dE_state_heat_J': -1.0e30,
        'step_dE_F_int_J': -1.00036e30,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
        'E_core_residual_frac': 0.0,
    }
    v_warm = check_energy_invariants(hf_row_warm, cfg, hf_all)
    assert len(v_warm) == 1
    assert v_warm[0].name == 'step_dE_inc_frac'
    assert v_warm[0].status_code == STATUS_ERROR_ENERGY_INVARIANT
    assert v_warm[0].report_only is True
    assert v_warm[0].value == pytest.approx(3.6e26 / 2.00036e30)
    assert v_warm[0].tolerance == pytest.approx(1.0e-4)

    # On the S_min floor (T_magma == 300.488 K), 1.8e-4 <= 2.5e-4 passes:
    hf_row_smin = dict(hf_row_warm, T_surf=300.488, T_magma=300.488)
    assert check_energy_invariants(hf_row_smin, cfg, hf_all) == []

    # Exceeding 2.5e-4 on S_min (e.g. 2.55e-4 <= 2.6e-4) fires step_dE_inc_frac with tol 2.5e-4:
    hf_row_smin_bad = dict(
        hf_row_smin,
        step_dE_F_int_J=-1.00051e30,
        E_residual_cons_J=5.1e26,
    )
    v_smin = check_energy_invariants(hf_row_smin_bad, cfg, hf_all)
    assert len(v_smin) == 1
    assert v_smin[0].name == 'step_dE_inc_frac'
    assert v_smin[0].tolerance == pytest.approx(2.5e-4)
    assert v_smin[0].report_only is True


def test_check_energy_invariants_solver_and_core_ledger_residuals():
    """Entropy ODE solver residual (> 1e-12) and core ledger residual (> 1e-4) report."""
    import pandas as pd

    cfg = _make_mock_config(dt_rtol=0.1, dt_minimum=1.0)
    hf_all = pd.DataFrame(
        [
            {
                'Time': 100.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': -1.0e30,
                'step_dE_F_int_J': -1.0e30,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    hf_row = {
        'Time': 200.0,
        'T_surf': 2200.0,
        'T_magma': 2800.0,
        'Phi_global': 1.0,
        'RF_depth': 0.0,
        'E_state_cons_J': 9.0e30,
        'E_state_heat_cons_J': -2.0e30,
        'dE_predicted_cons_J': -2.0e30,
        'E_residual_cons_J': 0.0,
        'E_residual_cons_frac': 0.0,
        'step_dE_state_heat_J': -1.0e30,
        'step_dE_F_int_J': -1.0e30,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'solver_residual_J': 1.0e19,  # 1.0e19 / 2.0e30 = 5.0e-12 > 1.0e-12
        'core_C_eff': 1.0e27,
        'E_core_residual_frac': -2.0e-4,  # |-2.0e-4| > 1.0e-4
    }
    violations = check_energy_invariants(hf_row, cfg, hf_all)
    names = [v.name for v in violations]
    assert names == ['solver_residual_frac', 'E_core_residual_frac']
    for v in violations:
        assert v.status_code == STATUS_ERROR_ENERGY_INVARIANT
        assert v.report_only is True
    assert violations[0].value == pytest.approx(5.0e-12)
    assert violations[0].tolerance == pytest.approx(1.0e-12)
    assert violations[1].value == pytest.approx(2.0e-4)
    assert violations[1].tolerance == pytest.approx(1.0e-4)


def test_check_structure_invariants_clean_row_and_zero_targets():
    """Co-converged structure row within desync and target tolerances returns no violations."""
    from proteus.utils.invariants import check_structure_invariants

    cfg = _make_mock_config()
    hf_row = {
        'Time': 500.0,
        'struct_mass_desync_frac': 3.2e-3,
        'M_int': 5.9722e24,
        'M_int_target': 5.9720e24,
        'M_core': 1.960e24,
        'M_core_target': 1.940e24,
        'R_int': 6.371e6,
        'R_core': 3.480e6,
    }
    assert check_structure_invariants(hf_row, cfg) == []

    # When targets are 0.0 (non-Zalmoxis or dummy structure), target checks do not fire:
    hf_row_no_target = {
        'Time': 500.0,
        'struct_mass_desync_frac': 0.0,
        'M_int': 5.972e24,
        'M_int_target': 0.0,
        'M_core': 1.940e24,
        'M_core_target': 0.0,
    }
    assert check_structure_invariants(hf_row_no_target, cfg) == []


def test_check_structure_invariants_desync_warn_and_stop_thresholds(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    """Desync in (5e-3, 1e-2] warns with report_only=True; > 1e-2 stops with status 31."""
    from proteus.utils.invariants import check_structure_invariants

    cfg = _make_mock_config()
    hf_warn = {
        'Time': 1000.0,
        'struct_mass_desync_frac': 7.0e-3,
        'M_int': 5.972e24,
        'M_int_target': 5.972e24,
        'M_core': 1.940e24,
        'M_core_target': 1.940e24,
        'R_int': 6.371e6,
        'R_core': 3.480e6,
    }
    v_warn = check_structure_invariants(hf_warn, cfg)
    assert len(v_warn) == 1
    assert v_warn[0].name == 'struct_mass_desync_frac'
    assert v_warn[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert v_warn[0].value == pytest.approx(7.0e-3)
    assert v_warn[0].tolerance == pytest.approx(5.0e-3)
    assert v_warn[0].report_only is True

    dirs = {'output': str(tmp_path)}
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        out = handle_violations(v_warn, mode='stop', directories=dirs)
    assert len(out) == 1
    assert not (tmp_path / 'status').exists()

    hf_stop = dict(hf_warn, struct_mass_desync_frac=1.2e-2)
    v_stop = check_structure_invariants(hf_stop, cfg)
    assert len(v_stop) == 1
    assert v_stop[0].name == 'struct_mass_desync_frac'
    assert v_stop[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert v_stop[0].value == pytest.approx(1.2e-2)
    assert v_stop[0].tolerance == pytest.approx(1.0e-2)
    assert v_stop[0].report_only is False

    with pytest.raises(InvariantViolationError) as exc_info:
        handle_violations(v_stop, mode='stop', directories=dirs)
    assert exc_info.value.status_code == 31
    status_lines = (tmp_path / 'status').read_text(encoding='utf-8').splitlines()
    assert status_lines[0] == '31'
    assert status_lines[1] == 'Error (Mass conservation invariant)'

    hf_nan = dict(hf_warn, struct_mass_desync_frac=float('nan'))
    v_nan = check_structure_invariants(hf_nan, cfg)
    assert len(v_nan) == 1
    assert v_nan[0].report_only is False
    assert v_nan[0].tolerance == pytest.approx(1.0e-2)


def test_check_structure_invariants_interior_and_core_mass_targets():
    """Interior mass target (> 3e-3) and core mass target (> 2e-2) raise stopping status-31 violations."""
    from proteus.utils.invariants import check_structure_invariants

    cfg = _make_mock_config()
    hf_bad = {
        'Time': 1500.0,
        'struct_mass_desync_frac': 3.0e-3,
        'M_int': 5.996e24,  # |5.996 / 5.972 - 1| = 4.0188e-3 > 3.0e-3
        'M_int_target': 5.972e24,
        'M_core': 1.990e24,  # |1.990 / 1.940 - 1| = 2.5773e-2 > 2.0e-2
        'M_core_target': 1.940e24,
        'R_int': 6.371e6,
        'R_core': 3.480e6,
    }
    violations = check_structure_invariants(hf_bad, cfg)
    assert [v.name for v in violations] == ['M_int_target_rel_err', 'M_core_target_rel_err']
    for v in violations:
        assert v.status_code == STATUS_ERROR_MASS_INVARIANT
        assert v.report_only is False
    assert violations[0].value == pytest.approx(abs(5.996e24 / 5.972e24 - 1.0))
    assert violations[0].tolerance == pytest.approx(3.0e-3)
    assert violations[1].value == pytest.approx(abs(1.990e24 / 1.940e24 - 1.0))
    assert violations[1].tolerance == pytest.approx(2.0e-2)


def test_check_volatile_invariants_clean_row_and_fO2_buffered_oxygen():
    """Volatile closure skips O under user_constant fO2 and checks O under from_O_budget."""
    from proteus.utils.invariants import check_volatile_invariants

    hf_row = {
        'Time': 2000.0,
        'M_planet': 6.0e24,
        'M_int': 5.9e24,
        'M_ele': 1.0e23,
        'M_atm': 1.0e21,
        'M_vol_atm': 1.0e21,
        'M_vaps': 0.0,
        'H2O_kg_atm': 6.0e20,
        'CO2_kg_atm': 4.0e20,
        'H_kg_total': 1.0e20,
        'H_kg_atm': 6.0e19,
        'H_kg_liquid': 3.0e19,
        'H_kg_solid': 1.0e19,
        'C_kg_total': 2.0e20,
        'C_kg_atm': 1.1e20,
        'C_kg_liquid': 9.0e19,
        'C_kg_solid': 0.0,
        # Oxygen reservoirs differ by 20% from O_kg_total:
        'O_kg_total': 1.0e23,
        'O_kg_atm': 4.0e22,
        'O_kg_liquid': 4.0e22,
        'O_kg_solid': 0.0,
        'M_vol_initial': 0.0,
        'esc_kg_cumulative': 0.0,
    }
    # Under default user_constant fO2, O is skipped (Ruling 3):
    cfg_user = _make_mock_config(fO2_source='user_constant', vapourise=False)
    assert check_volatile_invariants(hf_row, cfg_user) == []

    # Under from_O_budget (without vapourise), O is authoritative and fails closure:
    cfg_obud = _make_mock_config(fO2_source='from_O_budget', vapourise=False)
    v_obud = check_volatile_invariants(hf_row, cfg_obud)
    assert len(v_obud) == 1
    assert v_obud[0].name == 'O_reservoir_closure'
    assert v_obud[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert v_obud[0].value == pytest.approx(0.20)
    assert v_obud[0].tolerance == pytest.approx(2.5e-4)
    assert v_obud[0].report_only is False

    # Pre-IC row with M_planet <= 0.0 skips checks cleanly:
    assert check_volatile_invariants(dict(hf_row, M_planet=0.0), cfg_obud) == []


def test_check_volatile_invariants_element_closure_and_gas_sum():
    """Unbuffered element reservoir closure (> 2.5e-4 and > 1 kg) and gas sum (> 1e-6) fail."""
    from proteus.utils.invariants import check_volatile_invariants

    cfg = _make_mock_config(fO2_source='user_constant', vapourise=False)
    base_row = {
        'Time': 2500.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e21,
        'M_vol_atm': 1.0e21,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e21,
        'H_kg_total': 1.0e20,
        'H_kg_atm': 5.0e19,
        'H_kg_liquid': 5.0e19,
        'H_kg_solid': 0.0,
    }
    # Perturb H_kg_atm by 3.0e-4 * H_kg_total (> 2.5e-4):
    row_h_bad = dict(base_row, H_kg_atm=5.0e19 + 3.0e16)
    v_h = check_volatile_invariants(row_h_bad, cfg)
    assert len(v_h) == 1
    assert v_h[0].name == 'H_reservoir_closure'
    assert v_h[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert v_h[0].value == pytest.approx(3.0e-4)
    assert v_h[0].tolerance == pytest.approx(2.5e-4)
    assert v_h[0].report_only is False

    # Trace mismatch <= 1.0 kg (e.g. 0.5 kg on a 100 kg total, where rel = 5e-3 > 2.5e-4) passes:
    row_trace = dict(
        base_row,
        H_kg_total=100.0,
        H_kg_atm=50.5,
        H_kg_liquid=50.0,
        H_kg_solid=0.0,
    )
    assert check_volatile_invariants(row_trace, cfg) == []

    # Perturb H2O_kg_atm so species sum differs from M_vol_atm by 5.0e-6 > 1.0e-6:
    row_gas_bad = dict(base_row, H2O_kg_atm=1.0e21 * (1.0 - 5.0e-6))
    v_gas = check_volatile_invariants(row_gas_bad, cfg)
    assert len(v_gas) == 1
    assert v_gas[0].name == 'M_vol_atm_species_sum'
    assert v_gas[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert v_gas[0].value == pytest.approx(5.0e-6)
    assert v_gas[0].tolerance == pytest.approx(1.0e-6)
    assert v_gas[0].report_only is False


def test_check_volatile_invariants_calliope_mass_thresh_vs_atmodeller_floor():
    """CALLIOPE element closure uses outgas.mass_thresh + rtol * total; atmodeller keeps max(atol_kg, rtol * total)."""
    from proteus.utils.invariants import check_volatile_invariants

    # For N_kg_total = 4.0e18 kg and rtol = 2.5e-4, rtol * total = 1.0e15 kg.
    # Under calliope with mass_thresh = 1.0e16 kg, tol_e = 1.0e16 + 1.0e15 = 1.1e16 kg.
    cfg_cal = _make_mock_config(
        outgas_module='calliope',
        mass_thresh=1.0e16,
        fO2_source='user_constant',
        vapourise=False,
    )
    base_n_row = {
        'Time': 2600.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e21,
        'M_vol_atm': 1.0e21,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e21,
        'N_kg_total': 4.0e18,
        'N_kg_atm': 4.0e18,
        'N_kg_liquid': 0.0,
        'N_kg_solid': 0.0,
    }
    row_just_below = dict(base_n_row, N_kg_atm=4.0e18 + 1.09e16)
    assert check_volatile_invariants(row_just_below, cfg_cal) == []
    row_exact_bound = dict(base_n_row, N_kg_atm=4.0e18 + 1.10e16)
    assert check_volatile_invariants(row_exact_bound, cfg_cal) == []

    row_just_above = dict(base_n_row, N_kg_atm=4.0e18 + 1.11e16)
    v_above = check_volatile_invariants(row_just_above, cfg_cal)
    assert len(v_above) == 1
    assert v_above[0].name == 'N_reservoir_closure'
    assert v_above[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert v_above[0].value == pytest.approx(1.11e16 / 4.0e18)
    assert v_above[0].tolerance == pytest.approx(1.1e16 / 4.0e18)
    assert 'exceeds tol=2.750000e-03' in v_above[0].message
    assert v_above[0].report_only is False

    # Changing outgas.mass_thresh to 2.0e16 kg raises tol_e to 2.1e16 kg:
    cfg_cal_wider = _make_mock_config(
        outgas_module='calliope',
        mass_thresh=2.0e16,
        fO2_source='user_constant',
        vapourise=False,
    )
    assert check_volatile_invariants(row_just_above, cfg_cal_wider) == []
    row_above_wider = dict(base_n_row, N_kg_atm=4.0e18 + 2.11e16)
    v_wider = check_volatile_invariants(row_above_wider, cfg_cal_wider)
    assert len(v_wider) == 1
    assert v_wider[0].name == 'N_reservoir_closure'
    assert v_wider[0].value == pytest.approx(2.11e16 / 4.0e18)
    assert v_wider[0].tolerance == pytest.approx(2.1e16 / 4.0e18)
    assert 'exceeds tol=5.250000e-03' in v_wider[0].message

    # Under atmodeller, outgas.mass_thresh is not added and the 1.0 kg floor governs:
    cfg_atm = _make_mock_config(
        outgas_module='atmodeller',
        mass_thresh=1.0e16,
        fO2_source='user_constant',
        vapourise=False,
    )
    v_atm_n = check_volatile_invariants(row_just_below, cfg_atm)
    assert len(v_atm_n) == 1
    assert v_atm_n[0].name == 'N_reservoir_closure'
    assert v_atm_n[0].tolerance == pytest.approx(2.5e-4)

    row_atm_floor_ok = dict(base_n_row, N_kg_total=100.0, N_kg_atm=100.9)
    assert check_volatile_invariants(row_atm_floor_ok, cfg_atm) == []
    row_atm_floor_bad = dict(base_n_row, N_kg_total=100.0, N_kg_atm=101.1)
    v_atm_floor = check_volatile_invariants(row_atm_floor_bad, cfg_atm)
    assert len(v_atm_floor) == 1
    assert v_atm_floor[0].name == 'N_reservoir_closure'
    assert v_atm_floor[0].value == pytest.approx(1.1 / 100.0)
    assert v_atm_floor[0].tolerance == pytest.approx(1.0 / 100.0)


def test_check_volatile_invariants_calliope_zero_parts_threshold():
    """Under calliope with parts = 0, elements below mass_thresh / (1 - rtol) pass and above are reported."""
    from proteus.utils.invariants import check_volatile_invariants

    cfg = _make_mock_config(
        outgas_module='calliope',
        mass_thresh=1.0e16,
        fO2_source='user_constant',
        vapourise=False,
    )
    mass_thresh = float(cfg.outgas.mass_thresh)
    rtol = float(cfg.invariants.element_closure_rtol)
    # Zero parts closure condition: total > mass_thresh + rtol * total <=> total > mass_thresh / (1 - rtol)
    total_crit = mass_thresh / (1.0 - rtol)

    base_row = {
        'Time': 100.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e21,
        'M_vol_atm': 1.0e21,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e21,
        'H_kg_atm': 0.0,
        'H_kg_liquid': 0.0,
        'H_kg_solid': 0.0,
    }

    row_just_below = dict(base_row, H_kg_total=total_crit * (1.0 - 1.0e-6))
    assert check_volatile_invariants(row_just_below, cfg) == []

    row_just_above = dict(base_row, H_kg_total=total_crit * (1.0 + 1.0e-6))
    v_above = check_volatile_invariants(row_just_above, cfg)
    assert len(v_above) == 1
    assert v_above[0].name == 'H_reservoir_closure'
    assert v_above[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert v_above[0].report_only is False


def test_check_volatile_invariants_planet_mass_cap_and_vapourise():
    """M_atm > M_planet * (1 + 1e-6) stops without vapourise and warns past M_vaps with vapourise."""
    from proteus.utils.invariants import check_volatile_invariants

    cfg_no_vap = _make_mock_config(vapourise=False)
    row_cap = {
        'Time': 3000.0,
        'M_planet': 1.0e24,
        'M_atm': 1.0e24 * (1.0 + 5.0e-6),
        'M_vol_atm': 1.0e21,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e21,
    }
    v_stop = check_volatile_invariants(row_cap, cfg_no_vap)
    assert len(v_stop) == 1
    assert v_stop[0].name == 'M_atm_planet_cap'
    assert v_stop[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert v_stop[0].value == pytest.approx(5.0e-6)
    assert v_stop[0].tolerance == pytest.approx(1.0e-6)
    assert v_stop[0].report_only is False

    # With vapourise=True and M_vaps = 1e22: excess <= M_vaps passes, excess > M_vaps warns:
    cfg_vap = _make_mock_config(vapourise=True)
    row_vap_ok = dict(row_cap, M_atm=1.005e24, M_vaps=1.0e22)
    assert check_volatile_invariants(row_vap_ok, cfg_vap) == []

    row_vap_warn = dict(row_cap, M_atm=1.015e24, M_vaps=1.0e22)
    v_warn = check_volatile_invariants(row_vap_warn, cfg_vap)
    assert len(v_warn) == 1
    assert v_warn[0].name == 'M_atm_planet_cap'
    assert v_warn[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert v_warn[0].value == pytest.approx((1.015e24 - 1.0e24 - 1.0e22) / 1.0e24)
    assert v_warn[0].tolerance == pytest.approx(1.0e-6)
    assert v_warn[0].report_only is True


def test_check_volatile_invariants_escape_cumulative_bound():
    """Unexplained volatile loss > 1.5 * esc_kg_cumulative + 1e3 kg raises stopping status-31 violation."""
    from proteus.utils.invariants import check_volatile_invariants

    cfg = _make_mock_config(fO2_source='from_O_budget', vapourise=False)
    row_bad_loss = {
        'Time': 3500.0,
        'M_planet': 6.0e24,
        'M_atm': 0.0,
        'M_vol_atm': 0.0,
        'M_vaps': 0.0,
        'M_vol_initial': 1.0e21,
        'M_desiccated': 0.0,
        'esc_kg_cumulative': 1.0e15,
        'H_kg_total': 0.0,
    }
    v_loss = check_volatile_invariants(row_bad_loss, cfg)
    assert len(v_loss) == 1
    assert v_loss[0].name == 'escape_cumulative_bound'
    assert v_loss[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert v_loss[0].value == pytest.approx(1.0e21)
    assert v_loss[0].tolerance == pytest.approx(1.5 * 1.0e15 + 1.0e3)
    assert v_loss[0].report_only is False

    # When cumulative escape accounts for the loss (lost <= 1.5 * esc + 1e3), no violation:
    row_ok_loss = dict(row_bad_loss, esc_kg_cumulative=1.0e21)
    assert check_volatile_invariants(row_ok_loss, cfg) == []

    # When M_desiccated credits the removed volatile mass (lost = 1e21 - 0 - 1e21 = 0 <= esc_tol), no violation:
    row_desiccated_credit = dict(row_bad_loss, M_desiccated=1.0e21)
    assert check_volatile_invariants(row_desiccated_credit, cfg) == []

    # Non-finite M_desiccated must raise a named escape_cumulative_bound violation rather than defaulting to 0.0:
    for bad_des in (float('nan'), float('inf'), float('-inf')):
        row_bad_des = dict(row_ok_loss, M_desiccated=bad_des)
        v_bad_des = check_volatile_invariants(row_bad_des, cfg)
        assert len(v_bad_des) == 1
        assert v_bad_des[0].name == 'escape_cumulative_bound'
        assert v_bad_des[0].status_code == STATUS_ERROR_MASS_INVARIANT
        assert v_bad_des[0].report_only is False
        assert 'M_desiccated' in v_bad_des[0].message
        assert 'not finite' in v_bad_des[0].message

    # Under default fO2_source='user_constant', M_vol_initial includes buffered O
    # exchanged with the mantle prior to desiccation, so escape_cumulative_bound
    # must not fire even when all volatile inventories are depleted.
    cfg_user = _make_mock_config(fO2_source='user_constant', vapourise=False)
    assert check_volatile_invariants(row_bad_loss, cfg_user) == []


def test_check_state_bounds_invariants_temperatures_pressure_and_melt_fraction():
    """State bounds checks enforce T_surf > 0, 0 < T_magma < 1e6, P_surf >= 0 (> 0 if M_atm > 0), and Phi_global in [-1e-6, 1+1e-6]."""
    from proteus.utils.invariants import check_state_bounds_invariants

    cfg = _make_mock_config()
    clean_row = {
        'Time': 4000.0,
        'T_surf': 1500.0,
        'T_magma': 2000.0,
        'P_surf': 100.0,
        'M_atm': 1.0e21,
        'Phi_global': 0.5,
    }
    assert check_state_bounds_invariants(clean_row, cfg) == []

    # P_surf == 0.0 passes when M_atm == 0.0, and fails when M_atm > 0.0:
    assert check_state_bounds_invariants(dict(clean_row, P_surf=0.0, M_atm=0.0), cfg) == []
    v_p_zero = check_state_bounds_invariants(dict(clean_row, P_surf=0.0, M_atm=1.0e20), cfg)
    assert len(v_p_zero) == 1
    assert v_p_zero[0].name == 'P_surf_bounds'
    assert v_p_zero[0].status_code == STATUS_ERROR_STATE_BOUNDS_INVARIANT
    assert v_p_zero[0].report_only is False

    # Negative P_surf fails even when M_atm == 0.0:
    v_p_neg = check_state_bounds_invariants(dict(clean_row, P_surf=-1.0e-9, M_atm=0.0), cfg)
    assert len(v_p_neg) == 1
    assert v_p_neg[0].name == 'P_surf_bounds'

    # T_surf <= 0 or NaN fails:
    for bad_t in (0.0, -50.0, float('nan')):
        v_ts = check_state_bounds_invariants(dict(clean_row, T_surf=bad_t), cfg)
        assert len(v_ts) == 1
        assert v_ts[0].name == 'T_surf_bounds'
        assert v_ts[0].status_code == STATUS_ERROR_STATE_BOUNDS_INVARIANT
        assert v_ts[0].report_only is False

    # T_magma <= 0, >= 1.0e6, or NaN fails:
    for bad_tm in (0.0, -10.0, 1.0e6, 2.0e6, float('nan')):
        v_tm = check_state_bounds_invariants(dict(clean_row, T_magma=bad_tm), cfg)
        assert len(v_tm) == 1
        assert v_tm[0].name == 'T_magma_bounds'
        assert v_tm[0].status_code == STATUS_ERROR_STATE_BOUNDS_INVARIANT
        assert v_tm[0].tolerance == pytest.approx(1.0e6)
        assert v_tm[0].report_only is False

    # Phi_global within [-1e-6, 1 + 1e-6] passes; outside or NaN fails:
    for ok_phi in (-1.0e-6, 0.0, 1.0, 1.0 + 1.0e-6):
        assert check_state_bounds_invariants(dict(clean_row, Phi_global=ok_phi), cfg) == []
    for bad_phi in (-2.0e-6, 1.0 + 2.0e-6, float('nan')):
        v_phi = check_state_bounds_invariants(dict(clean_row, Phi_global=bad_phi), cfg)
        assert len(v_phi) == 1
        assert v_phi[0].name == 'Phi_global_bounds'
        assert v_phi[0].status_code == STATUS_ERROR_STATE_BOUNDS_INVARIANT
        assert v_phi[0].tolerance == pytest.approx(1.0e-6)
        assert v_phi[0].report_only is False


def test_check_row_invariants_orchestrates_modes_and_status_file(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    """check_row_invariants evaluates all row checks and applies mode dispatch."""
    from proteus.utils.invariants import check_row_invariants

    dirs = {'output': str(tmp_path)}
    bad_row = {
        'Time': 5000.0,
        'T_surf': -10.0,  # State bounds status 32 (stopping)
        'T_magma': 2000.0,
        'P_surf': 100.0,
        'M_atm': 1.0e21,
        'M_planet': 6.0e24,
        'M_vol_atm': 1.0e21,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e21,
        'Phi_global': 0.5,
        'struct_mass_desync_frac': 3.0e-3,
        'M_int': 5.972e24,
        'M_int_target': 5.972e24,
        'M_core': 1.940e24,
        'M_core_target': 1.940e24,
    }

    # Mode 'off' returns [] without logging or writing status file:
    cfg_off = _make_mock_config(mode='off')
    assert check_row_invariants(bad_row, cfg_off, directories=dirs) == []
    assert not (tmp_path / 'status').exists()

    # Mode 'report' logs WARNING and returns the violation without raising:
    cfg_rep = _make_mock_config(mode='report')
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        v_rep = check_row_invariants(bad_row, cfg_rep, directories=dirs)
    assert len(v_rep) == 1
    assert v_rep[0].name == 'T_surf_bounds'
    assert not (tmp_path / 'status').exists()

    # Mode 'stop' writes status 32 and raises InvariantViolationError:
    cfg_stop = _make_mock_config(mode='stop')
    with pytest.raises(InvariantViolationError) as exc_info:
        check_row_invariants(bad_row, cfg_stop, directories=dirs)
    assert exc_info.value.status_code == 32
    status_lines = (tmp_path / 'status').read_text(encoding='utf-8').splitlines()
    assert status_lines[0] == '32'
    assert status_lines[1] == 'Error (State bounds invariant)'

    # Unknown mode raises ValueError:
    with pytest.raises(ValueError, match="Unknown invariants.mode 'invalid'"):
        handle_violations(v_rep, mode='invalid')

    # Non-finite volatile fields and zero E_state_cons_J:
    import pandas as pd

    from proteus.utils.invariants import check_volatile_invariants

    assert (
        check_energy_invariants({'E_state_cons_J': 0.0}, cfg_rep, pd.DataFrame([{'Time': 0.0}]))
        == []
    )
    v_nan_vol = check_volatile_invariants(
        {
            'Time': 5000.0,
            'M_planet': 6.0e24,
            'M_atm': float('nan'),
            'M_vol_atm': float('nan'),
            'M_vaps': 0.0,
            'H_kg_total': float('nan'),
            'C_kg_total': 1.0e20,
            'C_kg_atm': float('nan'),
        },
        cfg_rep,
    )
    assert [v.name for v in v_nan_vol] == [
        'M_atm_planet_cap',
        'M_vol_atm_species_sum',
        'H_reservoir_closure',
        'C_reservoir_closure',
    ]


def test_main_loop_populates_energy_residuals_before_invariant_check_and_logs_to_proteus_log(
    tmp_path, monkeypatch, caplog: pytest.LogCaptureFixture
):
    """Main loop must populate cumulative energy residuals on hf_row before check_row_invariants so proteus_00.log records row-4 report lines."""
    from pathlib import Path

    import proteus.interior_energetics.wrapper as interior_wrapper
    from proteus import Proteus

    out = tmp_path / 'run'
    dummy_toml = Path(__file__).resolve().parents[2] / 'input' / 'dummy.toml'
    text = dummy_toml.read_text(encoding='utf-8')
    cfg_path = tmp_path / 'loop_energy_report.toml'
    cfg_path.write_text(text.replace('path = "auto"', f'path = "{out}"', 1), encoding='utf-8')

    runner = Proteus(config_path=cfg_path)
    config = runner.config
    config.atmos_chem.module = None
    config.params.stop.solid.enabled = False
    config.params.stop.iters.minimum = 1
    config.params.stop.time.minimum = 0.0
    config.params.stop.time.maximum = 2.0
    config.params.out.plot_mod = None
    config.params.out.archive_mod = 'none'

    real_interior = interior_wrapper.run_interior

    def interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        if runner.init_stage:
            return
        hf_row['E_state_cons_J'] = 6.95e31
        hf_row['step_dE_state_heat_J'] = 1.3452580147953664e27
        hf_row['step_dE_F_int_J'] = -3.067054909352445e27
        hf_row['step_dE_F_cmb_J'] = 4.409505110494009e27
        hf_row['step_dE_Q_radio_J'] = 0.0
        hf_row['step_dE_Q_tidal_J'] = 0.0
        hf_row['core_C_eff'] = 1.0e26
        hf_row['step_dE_core_J'] = -4.407e27

    monkeypatch.setattr(interior_wrapper, 'run_interior', interior)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        runner.start(resume=False, offline=True)

    assert runner.hf_row['E_residual_cons_J'] == pytest.approx(2.807814e24, rel=1e-5)
    assert float(runner.hf_all['E_residual_cons_J'].iloc[-1]) == pytest.approx(
        2.807814e24, rel=1e-5
    )
    assert runner.hf_row['E_core_residual_frac'] == pytest.approx(5.681160e-4, rel=1e-5)
    assert float(runner.hf_all['E_core_residual_frac'].iloc[-1]) == pytest.approx(
        5.681160e-4, rel=1e-5
    )
    assert any(
        'Invariant report [E_residual_cons_frac_gross]' in rec.message
        and '3.755489e-04 exceeds tol=2.600000e-04' in rec.message
        for rec in caplog.records
    )
    logfile_text = (out / 'proteus_00.log').read_text(encoding='utf-8')
    assert (
        '[ WARNING ] Invariant report [E_residual_cons_frac_gross]: '
        'Time=2.000000e+00 yr: E_residual_cons_frac_gross=3.755489e-04 '
        'exceeds tol=2.600000e-04'
    ) in logfile_text
    assert (
        '[ WARNING ] Invariant report [E_core_residual_frac]: '
        'Time=2.000000e+00 yr: |E_core_residual_frac|=5.681160e-04 '
        'exceeds tol=1.000000e-04'
    ) in logfile_text


@pytest.mark.unit
def test_populate_energy_and_core_residuals_idempotent_across_double_call_and_extend_helpfile():
    """Calling _populate_energy_residual and _populate_core_residual twice on the same
    (current_hf, new_row) and then via ExtendHelpfile must leave every cumulative column
    bit-equal (repr).
    """
    from proteus.utils.coupler import (
        CreateHelpfileFromDict,
        ExtendHelpfile,
        ZeroHelpfileRow,
        _populate_core_residual,
        _populate_energy_residual,
    )

    prior_row = ZeroHelpfileRow()
    prior_row['Time'] = 100.0
    prior_row['E_state_cons_J'] = 6.90e31
    prior_row['E_state_heat_cons_J'] = -5.123456789012345e29
    prior_row['dE_predicted_cons_J'] = -5.123489012345678e29
    prior_row['E_residual_cons_J'] = (
        prior_row['E_state_heat_cons_J'] - prior_row['dE_predicted_cons_J']
    )
    prior_row['E_residual_cons_frac'] = prior_row['E_residual_cons_J'] / abs(
        prior_row['E_state_heat_cons_J']
    )
    prior_row['solver_residual_J'] = 1.234567890123456e14
    prior_row['core_C_eff'] = 1.15e26
    prior_row['step_dE_F_cmb_J'] = 8.765432109876543e28
    prior_row['step_dE_impact_core_J'] = 2.5e27
    prior_row['E_core_residual_J'] = -4.321098765432109e24
    prior_row['E_core_residual_frac'] = prior_row['E_core_residual_J'] / (
        prior_row['step_dE_F_cmb_J'] + prior_row['step_dE_impact_core_J']
    )
    current_hf = CreateHelpfileFromDict(prior_row)

    new_row = ZeroHelpfileRow()
    new_row['Time'] = 200.0
    new_row['E_state_cons_J'] = 6.88e31
    new_row['step_dE_state_heat_J'] = -2.345678901234567e28
    new_row['step_dE_F_int_J'] = -3.456789012345678e28
    new_row['step_dE_F_cmb_J'] = 1.101110110111011e28
    new_row['step_dE_Q_radio_J'] = 9.876543210987654e25
    new_row['step_dE_Q_tidal_J'] = 1.234567890123456e25
    new_row['step_dE_impact_J'] = 5.0e26
    new_row['step_solver_residual_J'] = -3.456789012345678e13
    new_row['core_C_eff'] = 1.15e26
    new_row['step_dE_core_J'] = -1.131110110111011e28
    new_row['step_dE_impact_core_J'] = -3.0e26
    new_row['step_dE_struct_core_J'] = 1.25e25

    cumulative_keys = (
        'E_state_heat_cons_J',
        'dE_predicted_cons_J',
        'E_residual_cons_J',
        'E_residual_cons_frac',
        'solver_residual_J',
        'E_core_residual_J',
        'E_core_residual_frac',
    )

    _populate_energy_residual(current_hf, new_row)
    _populate_core_residual(current_hf, new_row)
    first_pass = {k: repr(new_row[k]) for k in cumulative_keys}
    first_pass_float = {k: repr(float(new_row[k])) for k in cumulative_keys}
    first_pass_all = {k: repr(v) for k, v in new_row.items()}

    _populate_energy_residual(current_hf, new_row)
    _populate_core_residual(current_hf, new_row)
    second_pass = {k: repr(new_row[k]) for k in cumulative_keys}
    second_pass_float = {k: repr(float(new_row[k])) for k in cumulative_keys}
    second_pass_all = {k: repr(v) for k, v in new_row.items()}

    assert second_pass == first_pass
    assert second_pass_float == first_pass_float
    assert second_pass_all == first_pass_all

    extended_hf = ExtendHelpfile(current_hf, new_row)
    after_extend = {k: repr(new_row[k]) for k in cumulative_keys}
    df_last_row = {k: repr(float(extended_hf[k].iloc[-1])) for k in cumulative_keys}

    assert after_extend == first_pass
    assert df_last_row == first_pass_float
