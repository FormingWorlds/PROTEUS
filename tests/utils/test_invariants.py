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
):
    """Build a minimal config object carrying ``invariants``, ``params.dt``, ``planet``, ``outgas``."""
    from types import SimpleNamespace

    from proteus.config._invariants import Invariants

    return SimpleNamespace(
        invariants=Invariants(mode=mode),
        params=SimpleNamespace(dt=SimpleNamespace(rtol=dt_rtol, minimum=dt_minimum)),
        planet=SimpleNamespace(fO2_source=fO2_source),
        outgas=SimpleNamespace(vapourise=vapourise),
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
    """Coupled energy residual fires above 2.6e-4 (or 5.0e-4 for coarse dt) and is report-only."""
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

    # Ruling 1: energy violations never stop a run even under mode == 'stop'
    dirs = {'output': str(tmp_path)}
    out = handle_violations(v_std, mode='stop', directories=dirs)
    assert out == v_std
    assert not (tmp_path / 'status').exists()

    # Coarse dt (rtol >= 0.2) relaxes threshold to 5.0e-4
    cfg_coarse = _make_mock_config(mode='stop', dt_rtol=0.2, dt_minimum=100.0)
    assert check_energy_invariants(hf_row, cfg_coarse, hf_all) == []

    hf_row_high = dict(hf_row, E_residual_cons_J=1.2e27)  # 1.2e27 / 2.0e30 = 6.0e-4 > 5.0e-4
    v_coarse = check_energy_invariants(hf_row_high, cfg_coarse, hf_all)
    assert len(v_coarse) == 1
    assert v_coarse[0].tolerance == pytest.approx(5.0e-4)
    assert v_coarse[0].report_only is True


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
