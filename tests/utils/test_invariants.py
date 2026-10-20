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
    """Status codes 30 (reserved), 31, and 32 map to their named invariant error descriptions."""
    assert STATUS_ERROR_ENERGY_INVARIANT == 30
    assert STATUS_ERROR_MASS_INVARIANT == 31
    assert STATUS_ERROR_STATE_BOUNDS_INVARIANT == 32
    assert CommentFromStatus(30) == 'Error (Energy conservation invariant, reserved)'
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


def test_check_energy_invariants_flux_cancellation_passes_under_gross_denom():
    """Opposite-sign F_int and F_cmb pass when normalized by gross throughput."""
    import pandas as pd

    cfg = _make_mock_config(dt_rtol=0.1, dt_minimum=1.0)
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 1.24e27,
                'step_dE_F_int_J': -2.17e27,
                'step_dE_F_cmb_J': 3.41e27,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    # Net state heat is 2.48e27 J while gross flux sum is 1.116e28 J;
    # raw |E_residual_cons_frac| = 7.98e-4 > 2.6e-4, but gross ratio is 1.77e-4 <= 2.6e-4.
    hf_row = {
        'Time': 20.0,
        'T_surf': 3340.0,
        'T_magma': 4268.0,
        'Phi_global': 1.0,
        'RF_depth': 0.0,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 2.48e27,
        'dE_predicted_cons_J': 2.47802e27,
        'E_residual_cons_J': 1.98e24,
        'E_residual_cons_frac': 1.98e24 / 2.48e27,
        'step_dE_state_heat_J': 1.24e27,
        'step_dE_F_int_J': -2.17e27,
        'step_dE_F_cmb_J': 3.41e27,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'solver_residual_J': 1.0e12,
        'core_C_eff': 0.0,
        'E_core_residual_frac': 0.0,
    }
    assert check_energy_invariants(hf_row, cfg, hf_all) == []
    hf_row_large = dict(hf_row, E_residual_cons_J=5.0e24, E_residual_cons_frac=5.0e24 / 2.48e27)
    assert any(
        'E_residual_cons_frac' in v.name
        for v in check_energy_invariants(hf_row_large, cfg, hf_all)
    )


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


def test_check_energy_invariants_step_increment_outside_and_on_smin(tmp_path):
    """Step energy increment divides by own step_gross with 1 J floor and uses 5.0e-2 only on T <= T(P_surf, S_min) + 0.5 K."""
    import pandas as pd

    eos_dir = tmp_path / 'spider_eos'
    eos_dir.mkdir()
    (eos_dir / 'temperature_solid.dat').write_text(
        '# 5 3 2\n'
        '# Pressure [nondim], Entropy [nondim], Quantity [nondim]\n'
        '# column * scaling factor = SI units\n'
        '# scaling factors (constant) for each column given on line below\n'
        '# 1000000000.0 4824266.84604467 1000.0\n'
        '1.000000000000000048e-04 2.199633491194133387e-05 3.023416715158805945e-01\n'
        '2.595513713862120375e-01 2.199633491194133387e-05 3.030318461324150170e-01\n'
        '5.190027427724240860e-01 2.199633491194133387e-05 3.037198468914933946e-01\n'
        '1.000000000000000048e-04 3.860484461686001134e-05 3.023416715158805945e-01\n'
        '2.595513713862120375e-01 3.860484461686001134e-05 3.030318461324150170e-01\n'
        '5.190027427724240860e-01 3.860484461686001134e-05 3.037198468914933946e-01\n',
        encoding='utf-8',
    )
    dirs = {'spider_eos_dir': str(eos_dir)}
    clip_k = 302.3416715158805945 + 0.5

    cfg = _make_mock_config(dt_rtol=0.1, dt_minimum=1.0)
    # Give hf_all a large cumulative history (1.0e32 J) so cumulative dilution would hide a 2.9e-2 step error:
    hf_all = pd.DataFrame(
        [
            {
                'Time': 100.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': -1.0e32,
                'step_dE_F_int_J': -1.0e32,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    # Step gross is 1.03e30 J and step_diff is 3.0e28 J -> step_err = 3.0e28 / 1.03e30 = 2.912621e-2.
    # Above T(P_surf, S_min) + 0.5 K, check_energy_invariants uses step_energy_rtol = 1.0e-2.
    hf_row_warm = {
        'Time': 200.0,
        'P_surf': 1.0,
        'T_surf': 2200.0,
        'T_magma': 2800.0,
        'Phi_global': 0.5,
        'RF_depth': 0.3,
        'E_state_cons_J': 9.0e30,
        'E_state_heat_cons_J': -1.01e32,
        'dE_predicted_cons_J': -1.0103e32,
        'E_residual_cons_J': 0.0,
        'E_residual_cons_frac': 0.0,
        'step_dE_state_heat_J': -1.0e30,
        'step_dE_F_int_J': -1.03e30,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
        'E_core_residual_frac': 0.0,
    }
    v_warm = check_energy_invariants(hf_row_warm, cfg, hf_all, directories=dirs)
    assert len(v_warm) == 1
    assert v_warm[0].name == 'step_dE_inc_frac'
    assert v_warm[0].status_code == STATUS_ERROR_ENERGY_INVARIANT
    assert v_warm[0].report_only is True
    assert v_warm[0].value == pytest.approx(3.0e28 / 1.03e30)
    assert v_warm[0].tolerance == pytest.approx(1.0e-2)

    # On the S_min floor (T_magma <= clip_k), ~2.91e-2 <= 5.0e-2 passes; without a table, 1.0e-2 applies:
    hf_row_smin = dict(hf_row_warm, T_surf=302.342, T_magma=302.342)
    assert check_energy_invariants(hf_row_smin, cfg, hf_all, directories=dirs) == []
    assert len(check_energy_invariants(hf_row_smin, cfg, hf_all)) == 1
    assert (
        check_energy_invariants(dict(hf_row_warm, T_magma=260.0), cfg, hf_all, directories=dirs)
        == []
    )
    assert (
        check_energy_invariants(
            dict(hf_row_warm, T_magma=clip_k), cfg, hf_all, directories=dirs
        )
        == []
    )
    v_above_clip = check_energy_invariants(
        dict(hf_row_warm, T_magma=clip_k + 1.0e-6), cfg, hf_all, directories=dirs
    )
    assert len(v_above_clip) == 1
    assert v_above_clip[0].tolerance == pytest.approx(1.0e-2)
    hf_row_tmagma_only = dict(hf_row_warm, T_surf=1500.0, T_magma=302.342)
    assert check_energy_invariants(hf_row_tmagma_only, cfg, hf_all, directories=dirs) == []
    hf_row_tsurf_only = dict(hf_row_warm, T_surf=302.342, T_magma=1500.0)
    v_tsurf_only = check_energy_invariants(hf_row_tsurf_only, cfg, hf_all, directories=dirs)
    assert len(v_tsurf_only) == 1
    assert v_tsurf_only[0].name == 'step_dE_inc_frac'
    assert v_tsurf_only[0].tolerance == pytest.approx(1.0e-2)

    # Exceeding 5.0e-2 on T_magma-only S_min (8.0e28 / 1.08e30 > 5.0e-2) fires step_dE_inc_frac with tol 5.0e-2:
    hf_row_smin_bad = dict(
        hf_row_tmagma_only,
        step_dE_F_int_J=-1.08e30,
        E_residual_cons_J=0.0,
    )
    v_smin = check_energy_invariants(hf_row_smin_bad, cfg, hf_all, directories=dirs)
    assert len(v_smin) == 1
    assert v_smin[0].name == 'step_dE_inc_frac'
    assert v_smin[0].value == pytest.approx(8.0e28 / 1.08e30)
    assert v_smin[0].tolerance == pytest.approx(5.0e-2)
    assert v_smin[0].report_only is True

    # Zero-flux step uses the 1.0 J floor: step_diff == 0.0 passes, step_diff == 1.0 fails with value 1.0:
    hf_row_zero_ok = dict(
        hf_row_warm,
        E_residual_cons_J=0.0,
        step_dE_state_heat_J=0.0,
        step_dE_F_int_J=0.0,
    )
    assert check_energy_invariants(hf_row_zero_ok, cfg, hf_all) == []
    hf_row_zero_bad = dict(hf_row_zero_ok, step_dE_state_heat_J=1.0)
    v_zero_bad = check_energy_invariants(hf_row_zero_bad, cfg, hf_all)
    assert len(v_zero_bad) == 1
    assert v_zero_bad[0].name == 'step_dE_inc_frac'
    assert v_zero_bad[0].value == pytest.approx(1.0)


def test_check_energy_invariants_solver_and_core_ledger_residuals():
    """Entropy ODE solver residual (> 1e-12 vs denom_gross) and core ledger residual (> 1e-4) report after prior throughput."""
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

    # Flux cancellation where |E_state_heat_cons_J| = 1.0e27 J << denom_gross = 2.0e30 J:
    # solver_residual_J = 5.0e16 J has 5.0e16 / 1.0e27 = 5.0e-11 > 1.0e-12 under net state heat,
    # but 5.0e16 / 2.0e30 = 2.5e-14 <= 1.0e-12 under denom_gross and must pass:
    hf_row_cancel = dict(
        hf_row,
        E_state_heat_cons_J=-1.0e27,
        dE_predicted_cons_J=-1.0e27,
        solver_residual_J=5.0e16,
        core_C_eff=0.0,
        E_core_residual_frac=0.0,
    )
    assert check_energy_invariants(hf_row_cancel, cfg, hf_all) == []

    # First post-init row (where prior rows in hf_all have zero gross throughput) skips
    # cumulative fraction checks (E_residual_cons_frac_gross, solver_residual_frac, E_core_residual_frac):
    hf_all_init = pd.DataFrame(
        [
            {
                'Time': 0.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 0.0,
                'step_dE_F_int_J': 0.0,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
            for _ in range(4)
        ]
    )
    hf_row_first_step = dict(
        hf_row,
        Time=2.0,
        E_state_heat_cons_J=-1.0e30,
        E_residual_cons_J=3.5e26,  # 3.5e-4 > 2.6e-4
        solver_residual_J=5.0e18,  # 5.0e-12 > 1.0e-12
        core_C_eff=1.0e27,
        E_core_residual_frac=-2.0e-4,  # 2.0e-4 > 1.0e-4
    )
    assert check_energy_invariants(hf_row_first_step, cfg, hf_all_init) == []


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
    """Interior mass target (> 3e-3) and core mass target (> 2.5e-2) are stopping violations."""
    from proteus.utils.invariants import check_structure_invariants

    cfg = _make_mock_config()
    # A 2.06e-2 core mass difference (above 2.0e-2 and the 1.565e-2 Zalmoxis maximum, below 2.5e-2) passes:
    hf_ok_core = {
        'Time': 1500.0,
        'struct_mass_desync_frac': 3.0e-3,
        'M_int': 5.972e24,
        'M_int_target': 5.972e24,
        'M_core': 1.980e24,  # |1.980 / 1.940 - 1| = 2.0619e-2 <= 2.5e-2
        'M_core_target': 1.940e24,
        'R_int': 6.371e6,
        'R_core': 3.480e6,
    }
    assert check_structure_invariants(hf_ok_core, cfg) == []

    hf_bad = {
        'Time': 1500.0,
        'struct_mass_desync_frac': 3.0e-3,
        'M_int': 5.996e24,  # |5.996 / 5.972 - 1| = 4.0188e-3 > 3.0e-3
        'M_int_target': 5.972e24,
        'M_core': 1.990e24,  # |1.990 / 1.940 - 1| = 2.5773e-2 > 2.5e-2
        'M_core_target': 1.940e24,
        'R_int': 6.371e6,
        'R_core': 3.480e6,
    }
    violations = check_structure_invariants(hf_bad, cfg)
    assert [v.name for v in violations] == ['M_int_target_rel_err', 'M_core_target_rel_err']
    assert violations[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert violations[0].report_only is False
    assert violations[1].status_code == STATUS_ERROR_MASS_INVARIANT
    assert violations[1].report_only is False
    assert violations[0].value == pytest.approx(abs(5.996e24 / 5.972e24 - 1.0))
    assert violations[0].tolerance == pytest.approx(3.0e-3)
    assert violations[1].value == pytest.approx(abs(1.990e24 / 1.940e24 - 1.0))
    assert violations[1].tolerance == pytest.approx(2.5e-2)


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
    # Under default user_constant fO2, O is skipped:
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

    # Non-positive M_planet emits a named M_planet_bounds violation without skipping closure checks:
    v_zero_mp = check_volatile_invariants(dict(hf_row, M_planet=0.0), cfg_obud)
    assert [v.name for v in v_zero_mp] == ['M_planet_bounds', 'O_reservoir_closure']


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

    # Perturbing H2O_kg_atm relative to M_vol_atm is checked by assert_mass_conservation,
    # not duplicated in check_volatile_invariants:
    row_gas_bad = dict(base_row, H2O_kg_atm=1.0e21 * (1.0 - 5.0e-6))
    assert check_volatile_invariants(row_gas_bad, cfg) == []


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

    # Under atmodeller, CHONS (like N) use outgas.mass_thresh only when total <= mass_thresh;
    # at N_kg_total = 4.0e18 kg > mass_thresh, atmodeller enforces the strict rtol check:
    cfg_atm = _make_mock_config(
        outgas_module='atmodeller',
        mass_thresh=1.0e16,
        fO2_source='user_constant',
        vapourise=False,
    )
    v_atm_above_thresh = check_volatile_invariants(row_just_below, cfg_atm)
    assert [v.name for v in v_atm_above_thresh] == ['N_reservoir_closure']
    assert v_atm_above_thresh[0].tolerance == pytest.approx(2.5e-4)
    assert len(check_volatile_invariants(row_just_above, cfg_atm)) == 1
    row_atm_subthresh = dict(base_n_row, N_kg_total=5.0e15, N_kg_atm=0.0)
    assert check_volatile_invariants(row_atm_subthresh, cfg_atm) == []

    row_he_below = dict(
        base_n_row,
        N_kg_total=0.0,
        N_kg_atm=0.0,
        He_kg_total=4.0e18,
        He_kg_liquid=4.0e18 + 1.09e16,
    )
    v_atm_he = check_volatile_invariants(row_he_below, cfg_atm)
    assert len(v_atm_he) == 1
    assert v_atm_he[0].name == 'He_reservoir_closure'
    assert v_atm_he[0].tolerance == pytest.approx(2.5e-4)

    row_atm_floor_ok = dict(
        base_n_row, N_kg_total=0.0, N_kg_atm=0.0, He_kg_total=100.0, He_kg_liquid=100.9
    )
    assert check_volatile_invariants(row_atm_floor_ok, cfg_atm) == []
    row_atm_floor_bad = dict(
        base_n_row, N_kg_total=0.0, N_kg_atm=0.0, He_kg_total=100.0, He_kg_liquid=101.1
    )
    v_atm_floor = check_volatile_invariants(row_atm_floor_bad, cfg_atm)
    assert len(v_atm_floor) == 1
    assert v_atm_floor[0].name == 'He_reservoir_closure'
    assert v_atm_floor[0].value == pytest.approx(1.1 / 100.0)
    assert v_atm_floor[0].tolerance == pytest.approx(1.0 / 100.0)


def test_check_volatile_invariants_calliope_zero_parts_threshold():
    """Under calliope, zero-parts and zero-total closure compare against mass_thresh."""
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

    # Zero-total leak check: with mass_thresh = 1.0e10 kg under calliope,
    # 9.8e9 kg with total = 0 stays below mass_thresh and passes, while 5.0e19 kg fails:
    cfg_cal_1e10 = _make_mock_config(
        outgas_module='calliope',
        mass_thresh=1.0e10,
        fO2_source='user_constant',
        vapourise=False,
    )
    row_zero_tot_pass = dict(base_row, H_kg_total=0.0, H_kg_atm=9.8e9)
    assert check_volatile_invariants(row_zero_tot_pass, cfg_cal_1e10) == []

    row_zero_tot_fail = dict(base_row, H_kg_total=0.0, H_kg_atm=5.0e19)
    v_zero_tot = check_volatile_invariants(row_zero_tot_fail, cfg_cal_1e10)
    assert len(v_zero_tot) == 1
    assert v_zero_tot[0].name == 'H_reservoir_closure'
    assert v_zero_tot[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert v_zero_tot[0].value == pytest.approx(5.0e19)
    assert v_zero_tot[0].tolerance == pytest.approx(1.0e10)
    assert v_zero_tot[0].report_only is False

    # Under vapourise=True, rock-vapour elements have total=0 by design and do not fail closure,
    # whereas with vapourise=False a rock-vapour or volatile leak with total=0 stops:
    from proteus.utils.constants import vap_element_list

    cfg_vap_stop = _make_mock_config(
        mode='stop',
        outgas_module='calliope',
        mass_thresh=1.0e10,
        fO2_source='user_constant',
        vapourise=True,
    )
    for elem in vap_element_list:
        row_vap_elem = dict(
            base_row,
            M_atm=1.03e21,
            M_vaps=3.0e19,
            **{f'{elem}_kg_total': 0.0, f'{elem}_kg_atm': 3.0e19},
        )
        assert check_volatile_invariants(row_vap_elem, cfg_vap_stop) == []
        assert (
            handle_violations(
                check_volatile_invariants(row_vap_elem, cfg_vap_stop), mode='stop'
            )
            == []
        )

    row_novap_si_leak = dict(base_row, Si_kg_total=0.0, Si_kg_atm=5.0e19)
    v_novap_si = check_volatile_invariants(row_novap_si_leak, cfg_cal_1e10)
    assert [v.name for v in v_novap_si] == ['Si_reservoir_closure']
    with pytest.raises(InvariantViolationError) as exc_info:
        handle_violations(v_novap_si, mode='stop')
    assert exc_info.value.invariant_name == 'Si_reservoir_closure'

    # Under atmodeller, CHONS uses mass_thresh while noble gases compare against element_closure_atol_kg (1.0 kg):
    cfg_atm = _make_mock_config(
        outgas_module='atmodeller',
        mass_thresh=1.0e10,
        fO2_source='user_constant',
        vapourise=False,
    )
    assert check_volatile_invariants(row_zero_tot_pass, cfg_atm) == []
    assert (
        check_volatile_invariants(dict(base_row, He_kg_total=0.0, He_kg_atm=0.5), cfg_atm) == []
    )
    v_atm_he_zero = check_volatile_invariants(
        dict(base_row, He_kg_total=0.0, He_kg_atm=9.8e9), cfg_atm
    )
    assert len(v_atm_he_zero) == 1
    assert v_atm_he_zero[0].name == 'He_reservoir_closure'
    assert v_atm_he_zero[0].value == pytest.approx(9.8e9)
    assert v_atm_he_zero[0].tolerance == pytest.approx(1.0)


def test_assert_mass_conservation_writes_status_31_and_no_duplicate_cap_or_species_sum(
    tmp_path, monkeypatch
):
    """Proteus.start writes status 31 when assert_mass_conservation raises, and check_volatile_invariants does not duplicate M_atm_planet_cap or M_vol_atm_species_sum."""
    from pathlib import Path

    from proteus import Proteus
    from proteus.interior_energetics.common import Interior_t
    from proteus.utils.coupler import assert_mass_conservation
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
    assert check_volatile_invariants(row_cap, cfg_no_vap) == []

    with pytest.raises(RuntimeError, match='Mass conservation violation'):
        assert_mass_conservation(row_cap)
    assert not (tmp_path / 'status').exists()

    # Verify through Proteus.start that catching RuntimeError from assert_mass_conservation
    # writes status 31 before re-raising:
    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    out_dir = tmp_path / 'run_mass_cap_31'
    dummy_toml = Path(__file__).resolve().parents[2] / 'input' / 'dummy.toml'
    cfg_path = tmp_path / 'dummy_mass_cap.toml'
    cfg_path.write_text(
        dummy_toml.read_text(encoding='utf-8').replace(
            'path = "auto"', f'path = "{out_dir}"', 1
        ),
        encoding='utf-8',
    )
    runner = Proteus(config_path=cfg_path)
    runner.config.params.stop.solid.enabled = False
    runner.config.params.stop.iters.minimum = 1
    runner.config.params.stop.iters.maximum = 3
    runner.config.params.out.plot_mod = None
    runner.config.params.out.archive_mod = 'none'

    import proteus.interior_energetics.wrapper as int_wrap

    real_update = int_wrap.update_planet_mass

    def corrupt_mass(hf_row):
        real_update(hf_row)
        hf_row['M_atm'] = 1.0e30

    monkeypatch.setattr(int_wrap, 'update_planet_mass', corrupt_mass)
    with pytest.raises(RuntimeError, match='Mass conservation violation'):
        runner.start(resume=False, offline=True)

    status_lines = (out_dir / 'status').read_text(encoding='utf-8').splitlines()
    assert status_lines[0].strip() == '31'
    assert status_lines[1].strip() == 'Error (Mass conservation invariant)'


def test_check_volatile_invariants_escape_cumulative_bound(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    """Unexplained volatile loss > 1.5 * esc_kg_cumulative + 1e3 kg emits report_only=True warning."""
    from proteus.utils.invariants import check_volatile_invariants

    cfg = _make_mock_config(mode='stop', fO2_source='from_O_budget', vapourise=False)
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
    assert v_loss[0].report_only is True

    # Sub-threshold N (5.0e15 kg) floored to 0 after one escape step with esc_kg_cumulative = 3.16e13 kg
    # under from_O_budget reports a warning and does not stop in mode='stop':
    row_n_floored = {
        'Time': 2.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e20,
        'M_vol_initial': 1.0e20 + 5.0e15,
        'M_desiccated': 0.0,
        'esc_kg_cumulative': 3.16e13,
        'H_kg_total': 1.0e20,
        'H_kg_atm': 1.0e20,
        'N_kg_total': 0.0,
        'N_kg_atm': 0.0,
    }
    v_floored = check_volatile_invariants(row_n_floored, cfg)
    assert len(v_floored) == 1
    assert v_floored[0].name == 'escape_cumulative_bound'
    assert v_floored[0].report_only is True
    dirs = {'output': str(tmp_path)}
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        out_floored = handle_violations(v_floored, mode='stop', directories=dirs)
    assert out_floored == v_floored
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'WARNING'
    assert not (tmp_path / 'status').exists()

    # When cumulative escape accounts for the loss (lost <= 1.5 * esc + 1e3), no violation:
    row_ok_loss = dict(row_bad_loss, esc_kg_cumulative=1.0e21)
    assert check_volatile_invariants(row_ok_loss, cfg) == []

    # When M_desiccated credits the removed volatile mass (lost = 1e21 - 0 - 1e21 = 0 <= esc_tol), no violation:
    row_desiccated_credit = dict(row_bad_loss, M_desiccated=1.0e21)
    assert check_volatile_invariants(row_desiccated_credit, cfg) == []

    # Non-finite M_desiccated must emit a named escape_cumulative_bound violation rather than defaulting to 0.0:
    for bad_des in (float('nan'), float('inf'), float('-inf')):
        row_bad_des = dict(row_ok_loss, M_desiccated=bad_des)
        v_bad_des = check_volatile_invariants(row_bad_des, cfg)
        assert len(v_bad_des) == 1
        assert v_bad_des[0].name == 'escape_cumulative_bound'
        assert v_bad_des[0].status_code == STATUS_ERROR_MASS_INVARIANT
        assert v_bad_des[0].report_only is True
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
        'M_atm_bounds',
        'M_vol_atm_bounds',
        'H_reservoir_closure',
        'C_reservoir_closure',
    ]


def test_main_loop_populates_energy_residuals_before_invariant_check_and_logs_to_proteus_log(
    tmp_path, monkeypatch, caplog: pytest.LogCaptureFixture
):
    """Main loop populates cumulative energy residuals via ExtendHelpfile before check_row_invariants and skips the first post-init step where hist_gross is zero."""
    from pathlib import Path

    import proteus.interior_energetics.wrapper as interior_wrapper
    from proteus import Proteus
    from proteus.interior_energetics.common import Interior_t

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    out = tmp_path / 'run'
    dummy_toml = Path(__file__).resolve().parents[2] / 'input' / 'dummy.toml'
    text = dummy_toml.read_text(encoding='utf-8')
    cfg_path = tmp_path / 'loop_energy_report.toml'
    cfg_path.write_text(text.replace('path = "auto"', f'path = "{out}"', 1), encoding='utf-8')

    runner = Proteus(config_path=cfg_path)
    config = runner.config
    config.atmos_chem.module = None
    config.params.stop.solid.enabled = False
    config.params.stop.strict = False
    config.params.stop.iters.minimum = 1
    config.params.stop.time.minimum = 0.0
    config.params.stop.time.maximum = 100.0
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

    assert runner.hf_row['E_residual_cons_J'] == pytest.approx(2.0 * 2.807814e24, rel=1e-5)
    assert float(runner.hf_all['E_residual_cons_J'].iloc[-1]) == pytest.approx(
        2.0 * 2.807814e24, rel=1e-5
    )
    assert runner.hf_row['E_core_residual_frac'] == pytest.approx(5.681160e-4, rel=1e-5)
    assert float(runner.hf_all['E_core_residual_frac'].iloc[-1]) == pytest.approx(
        5.681160e-4, rel=1e-5
    )
    logfile_text = (out / 'proteus_00.log').read_text(encoding='utf-8')
    # First post-init step (Time=2.0 yr) has hist_gross == 0.0 and must not log cumulative fractions:
    assert 'Time=2.000000e+00 yr: E_residual_cons_frac_gross=' not in logfile_text
    assert 'Time=2.000000e+00 yr: |E_core_residual_frac|=' not in logfile_text
    # Second post-init step (Time=100.0 yr) has hist_gross > 0.0 and logs both:
    assert (
        '[ WARNING ] Invariant report [E_residual_cons_frac_gross]: '
        'Time=1.000000e+02 yr: E_residual_cons_frac_gross=3.755489e-04 '
        'exceeds tol=2.600000e-04'
    ) in logfile_text
    assert (
        '[ WARNING ] Invariant report [E_core_residual_frac]: '
        'Time=1.000000e+02 yr: |E_core_residual_frac|=5.681160e-04 '
        'exceeds tol=1.000000e-04'
    ) in logfile_text


@pytest.mark.unit
def test_populate_energy_and_core_residuals_idempotent_across_double_call_and_extend_helpfile():
    """ExtendHelpfile populates cumulative energy and core residuals in place on new_row without redundant calls in proteus.py."""
    from pathlib import Path

    import proteus as proteus_pkg
    from proteus.utils.coupler import (
        CreateHelpfileFromDict,
        ExtendHelpfile,
        ZeroHelpfileRow,
    )

    proteus_src = (Path(proteus_pkg.__file__).resolve().parent / 'proteus.py').read_text(
        encoding='utf-8'
    )
    assert '_populate_energy_residual' not in proteus_src
    assert '_populate_core_residual' not in proteus_src

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

    extended_hf = ExtendHelpfile(current_hf, new_row)
    after_extend = {k: repr(float(new_row[k])) for k in cumulative_keys}
    df_last_row = {k: repr(float(extended_hf[k].iloc[-1])) for k in cumulative_keys}

    assert after_extend == df_last_row
    assert new_row['E_residual_cons_J'] != 0.0
    assert new_row['E_core_residual_J'] != 0.0


def test_check_volatile_invariants_vapourise_exempts_oxygen_while_checking_c_n_s():
    """With outgas.vapourise=True, lavatmos-overwritten elements are exempt from closure; without vapourise, closure fires."""
    from proteus.utils.invariants import check_volatile_invariants, handle_violations

    row = {
        'Time': 2100.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e21 + 1.5e17,
        'M_vol_atm': 1.0e21 + 1.5e17,
        'M_vaps': 1.0e20,
        'H2O_kg_atm': 1.0e21,
        'H_kg_total': 1.0e20,
        'H_kg_atm': 1.5e20,
        'H_kg_liquid': 0.0,
        'H_kg_solid': 0.0,
        'O_kg_total': 1.0e23,
        'O_kg_atm': 4.0e22,
        'O_kg_liquid': 4.0e22,
        'O_kg_solid': 0.0,
        'C_kg_total': 4.666e21,
        'C_kg_atm': 2.040e21,
        'C_kg_liquid': 2.666e21,
        'C_kg_solid': 0.0,
        'N_kg_total': 1.0e19,
        'N_kg_atm': 1.01e19,
        'N_kg_liquid': 0.0,
        'N_kg_solid': 0.0,
        'S_kg_total': 5.0e19,
        'S_kg_atm': 5.05e19,
        'S_kg_liquid': 0.0,
        'S_kg_solid': 0.0,
        'He_kg_total': 1.0e17,
        'He_kg_atm': 1.5e17,
        'He_kg_liquid': 0.0,
        'He_kg_solid': 0.0,
        'M_vol_initial': 0.0,
        'esc_kg_cumulative': 0.0,
    }
    cfg_vap = _make_mock_config(mode='stop', fO2_source='from_O_budget', vapourise=True)
    v_vap = check_volatile_invariants(row, cfg_vap)
    assert v_vap == []
    assert handle_violations(v_vap, mode='stop') == []

    # Canary: without vapourise=True, H, O, C, N, S, and He closure violations fire:
    cfg_no_vap = _make_mock_config(fO2_source='from_O_budget', vapourise=False)
    v_no_vap = check_volatile_invariants(row, cfg_no_vap)
    assert [v.name for v in v_no_vap] == [
        'H_reservoir_closure',
        'O_reservoir_closure',
        'C_reservoir_closure',
        'N_reservoir_closure',
        'S_reservoir_closure',
        'He_reservoir_closure',
    ]
    assert all(not v.report_only for v in v_no_vap)


def test_check_volatile_and_state_bounds_negative_masses_and_zero_atm_psurf():
    """Negative masses below -element_closure_atol_kg, M_planet <= 0, and P_surf < 0 at |M_atm| <= atol fail."""
    from proteus.utils.invariants import (
        check_state_bounds_invariants,
        check_volatile_invariants,
    )

    cfg = _make_mock_config(fO2_source='user_constant', vapourise=False)
    clean_row = {
        'Time': 3000.0,
        'T_surf': 1500.0,
        'T_magma': 2000.0,
        'P_surf': 100.0,
        'Phi_global': 0.5,
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

    # |M_atm| <= element_closure_atol_kg (1.0 kg) permits P_surf >= 0.0 (including tiny positive P_surf):
    for m_atm_tiny in (-0.5, 0.0, 0.5):
        for p_surf_ok in (0.0, 1.0e-5):
            row_tiny_p = dict(
                clean_row,
                M_atm=m_atm_tiny,
                M_vol_atm=0.0,
                H2O_kg_atm=0.0,
                P_surf=p_surf_ok,
            )
            assert check_state_bounds_invariants(row_tiny_p, cfg) == []
            assert check_volatile_invariants(row_tiny_p, cfg) == []

    # P_surf < 0 with |M_atm| <= atol fails P_surf_bounds:
    v_psurf = check_state_bounds_invariants(dict(clean_row, M_atm=0.5, P_surf=-1.0e-5), cfg)
    assert [v.name for v in v_psurf] == ['P_surf_bounds']
    assert v_psurf[0].status_code == STATUS_ERROR_STATE_BOUNDS_INVARIANT
    assert v_psurf[0].report_only is False
    assert 'violates P_surf >= 0.0 bar' in v_psurf[0].message
    assert 'kg bar' not in v_psurf[0].message

    # Negative M_atm below -element_closure_atol_kg with P_surf == 0.0 fails state-bounds with clean message:
    v_neg_matm_sb = check_state_bounds_invariants(
        dict(clean_row, M_atm=-1.0e20, P_surf=0.0), cfg
    )
    assert [v.name for v in v_neg_matm_sb] == ['P_surf_bounds']
    assert 'violates P_surf >= 0.0 bar with M_atm >=' in v_neg_matm_sb[0].message
    assert 'kg bar' not in v_neg_matm_sb[0].message

    # Derived elements (O under user_constant or vapourise) are still bound-checked on reservoirs:
    v_o_bound = check_volatile_invariants(
        dict(clean_row, O_kg_liquid=-1.0e20, O_kg_total=1.0e23), cfg
    )
    assert [v.name for v in v_o_bound] == ['O_kg_liquid_bounds']

    # Negative M_atm, M_vol_atm, species, total, and reservoir masses below -element_closure_atol_kg fail:
    v_neg_vol = check_volatile_invariants(
        dict(
            clean_row,
            M_planet=-6.0e24,
            M_atm=-1.0e21,
            M_vol_atm=-1.0e21,
            H2O_kg_atm=-1.0e21,
            H_kg_total=-1.0e20,
            H_kg_atm=-1.0e20,
            C_kg_total=1.0e20,
            C_kg_atm=-5.0e19,
            C_kg_liquid=1.5e20,
            C_kg_solid=0.0,
        ),
        cfg,
    )
    names = [v.name for v in v_neg_vol]
    assert 'M_planet_bounds' in names
    assert 'M_atm_bounds' in names
    assert 'M_vol_atm_bounds' in names
    assert 'H2O_kg_atm_bounds' in names
    assert 'H_reservoir_closure' in names
    assert 'H_kg_atm_bounds' in names
    assert 'C_kg_atm_bounds' in names
    assert all(
        'element_closure_atol_kg' in v.message for v in v_neg_vol if v.name != 'M_planet_bounds'
    )

    # Negative species mass within element_closure_atol_kg (1.0 kg) passes, and below -1.0 kg emits H2O_kg_atm_bounds:
    assert (
        check_volatile_invariants(
            dict(clean_row, M_vol_atm=0.0, H2O_kg_atm=-0.5),
            cfg,
        )
        == []
    )
    v_neg_sum = check_volatile_invariants(
        dict(clean_row, M_vol_atm=0.0, H2O_kg_atm=-10.0),
        cfg,
    )
    assert [v.name for v in v_neg_sum] == ['H2O_kg_atm_bounds']
    assert v_neg_sum[0].value == pytest.approx(-10.0)
    assert v_neg_sum[0].tolerance == pytest.approx(1.0)


def test_check_energy_invariants_non_finite_inputs_emit_named_violations():
    """Non-finite E_state_cons_J, core_C_eff, and step_dE_* inputs emit named status-30 violations."""
    import pandas as pd

    cfg = _make_mock_config()
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
    base_row = {
        'Time': 200.0,
        'T_surf': 2200.0,
        'T_magma': 2800.0,
        'Phi_global': 1.0,
        'RF_depth': 0.0,
        'E_state_cons_J': 9.0e30,
        'E_state_heat_cons_J': -2.0e30,
        'E_residual_cons_J': 0.0,
        'step_dE_state_heat_J': -1.0e30,
        'step_dE_F_int_J': -1.0e30,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 1.0e26,
        'step_dE_core_J': 0.0,
        'E_core_residual_frac': 0.0,
    }

    v_nan_state = check_energy_invariants(
        dict(base_row, E_state_cons_J=float('nan')), cfg, hf_all
    )
    assert 'E_state_cons_J' in [v.name for v in v_nan_state]
    assert all(
        v.status_code == STATUS_ERROR_ENERGY_INVARIANT and v.report_only for v in v_nan_state
    )

    v_nan_core = check_energy_invariants(
        dict(base_row, core_C_eff=float('nan'), E_core_residual_frac=0.5),
        cfg,
        hf_all,
    )
    assert 'core_C_eff' in [v.name for v in v_nan_core]
    assert 'E_core_residual_frac' in [v.name for v in v_nan_core]

    v_nan_step_core = check_energy_invariants(
        dict(base_row, step_dE_core_J=float('nan')),
        cfg,
        hf_all,
    )
    assert 'step_dE_core_J' in [v.name for v in v_nan_step_core]


@pytest.mark.parametrize('write_mod', [1, 3])
def test_stop_mode_flushes_failing_row_and_prior_rows_regardless_of_write_mod(
    tmp_path, monkeypatch, caplog: pytest.LogCaptureFixture, write_mod: int
):
    """In mode == 'stop', the failing row is logged before CSV fill and flushed to runtime_helpfile.csv."""
    from pathlib import Path

    import pandas as pd

    import proteus.interior_energetics.wrapper as interior_wrapper
    from proteus import Proteus
    from proteus.interior_energetics.common import Interior_t

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    out = tmp_path / f'run_wm{write_mod}'
    dummy_toml = Path(__file__).resolve().parents[2] / 'input' / 'dummy.toml'
    text = dummy_toml.read_text(encoding='utf-8')
    cfg_path = tmp_path / f'stop_flush_{write_mod}.toml'
    cfg_path.write_text(text.replace('path = "auto"', f'path = "{out}"', 1), encoding='utf-8')

    runner = Proteus(config_path=cfg_path)
    config = runner.config
    config.invariants.mode = 'stop'
    config.atmos_chem.module = None
    config.params.stop.solid.enabled = False
    config.params.stop.iters.minimum = 1
    config.params.stop.time.minimum = 0.0
    config.params.stop.time.maximum = 500.0
    config.params.out.write_mod = write_mod
    config.params.out.plot_mod = None
    config.params.out.archive_mod = 'none'

    real_interior = interior_wrapper.run_interior

    # Trigger a stopping violation on total loop 5 (not divisible by write_mod=3):
    def interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        if runner.loops['total'] == 4:
            hf_row['T_magma'] = 2.0e6 if write_mod == 1 else float('nan')

    monkeypatch.setattr(interior_wrapper, 'run_interior', interior)
    with caplog.at_level(logging.ERROR, logger='fwl.proteus.proteus'):
        with pytest.raises(InvariantViolationError) as exc_info:
            runner.start(resume=False, offline=True)
    assert exc_info.value.status_code == 32
    assert exc_info.value.invariant_name == (
        'T_magma_bounds' if write_mod == 1 else 'T_surf_bounds'
    )

    flush_logs = [
        r.message for r in caplog.records if 'Failing hf_row on invariant stop' in r.message
    ]
    assert len(flush_logs) == 1
    if write_mod == 1:
        assert "'T_magma': 2000000.0" in flush_logs[0]
    else:
        assert "'T_magma': nan" in flush_logs[0]

    csv_path = out / 'runtime_helpfile.csv'
    assert csv_path.exists()
    df_disk = pd.read_csv(csv_path, sep=r'\s+')
    assert len(df_disk) == 5
    if write_mod == 1:
        assert float(df_disk['T_magma'].iloc[-1]) == pytest.approx(2.0e6)
    else:
        assert float(df_disk['T_magma'].iloc[-1]) == pytest.approx(0.0)


def test_invariant_tracker_reports_on_crossing_counts_rows_and_logs_summary(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    """Crossing suppression applies to cumulative/carried-forward checks; non-finite values do not hide finite maxima or print max=-inf."""
    from proteus.utils.invariants import _CROSSING_SUPPRESSED_NAMES, InvariantTracker

    assert 'E_residual_cons_frac_gross' in _CROSSING_SUPPRESSED_NAMES
    assert 'struct_mass_desync_frac' in _CROSSING_SUPPRESSED_NAMES
    assert 'M_core_target_rel_err' in _CROSSING_SUPPRESSED_NAMES

    tracker = InvariantTracker()
    v_energy_1 = InvariantViolation(
        name='E_residual_cons_frac_gross',
        status_code=STATUS_ERROR_ENERGY_INVARIANT,
        value=4.0e-4,
        tolerance=2.6e-4,
        message='Time=1.0e+02 yr: E_residual_cons_frac_gross=4.000000e-04 exceeds tol=2.600000e-04',
        report_only=True,
    )
    v_energy_2 = InvariantViolation(
        name='E_residual_cons_frac_gross',
        status_code=STATUS_ERROR_ENERGY_INVARIANT,
        value=6.0e-4,
        tolerance=2.6e-4,
        message='Time=2.0e+02 yr: E_residual_cons_frac_gross=6.000000e-04 exceeds tol=2.600000e-04',
        report_only=True,
    )
    v_step_1 = InvariantViolation(
        name='step_dE_inc_frac',
        status_code=STATUS_ERROR_ENERGY_INVARIANT,
        value=2.0e-2,
        tolerance=1.0e-2,
        message='Time=1.0e+02 yr: step_dE_inc_frac=2.000000e-02 exceeds tol=1.000000e-02',
        report_only=True,
    )
    v_step_2 = InvariantViolation(
        name='step_dE_inc_frac',
        status_code=STATUS_ERROR_ENERGY_INVARIANT,
        value=5.0e-1,
        tolerance=1.0e-2,
        message='Time=2.0e+02 yr: step_dE_inc_frac=5.000000e-01 exceeds tol=1.000000e-02',
        report_only=True,
    )
    v_desync_1 = InvariantViolation(
        name='struct_mass_desync_frac',
        status_code=STATUS_ERROR_MASS_INVARIANT,
        value=6.0e-3,
        tolerance=5.0e-3,
        message='Time=1.0e+02 yr: struct_mass_desync_frac=6.000000e-03 exceeds warn tol=5.000000e-03',
        report_only=True,
    )
    v_desync_2 = InvariantViolation(
        name='struct_mass_desync_frac',
        status_code=STATUS_ERROR_MASS_INVARIANT,
        value=6.0e-3,
        tolerance=5.0e-3,
        message='Time=2.0e+02 yr: struct_mass_desync_frac=6.000000e-03 exceeds warn tol=5.000000e-03',
        report_only=True,
    )
    v_bounds_1 = InvariantViolation(
        name='T_surf_bounds',
        status_code=STATUS_ERROR_STATE_BOUNDS_INVARIANT,
        value=0.0,
        tolerance=0.0,
        message='Time=1.0e+02 yr: T_surf=0.000000e+00 K violates T_surf > 0.0 K',
        report_only=False,
    )
    v_bounds_2 = InvariantViolation(
        name='T_surf_bounds',
        status_code=STATUS_ERROR_STATE_BOUNDS_INVARIANT,
        value=-5.0,
        tolerance=0.0,
        message='Time=2.0e+02 yr: T_surf=-5.000000e+00 K violates T_surf > 0.0 K',
        report_only=False,
    )
    dirs = {'output': str(tmp_path)}
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        # Row 1: crossing into violation -> logs E_residual_cons_frac_gross (count=1), struct_mass_desync_frac (count=1), step_dE_inc_frac (count=1), and T_surf_bounds
        handle_violations(
            [v_energy_1, v_desync_1, v_step_1, v_bounds_1],
            mode='report',
            directories=dirs,
            tracker=tracker,
        )
        # Rows 2 and 3: E_residual_cons_frac_gross, struct_mass_desync_frac, and step_dE_inc_frac suppressed, while T_surf_bounds logs on every row
        handle_violations(
            [v_energy_2, v_desync_2, v_step_2, v_bounds_2],
            mode='report',
            directories=dirs,
            tracker=tracker,
        )
        handle_violations(
            [v_energy_1, v_desync_2], mode='report', directories=dirs, tracker=tracker
        )
        # Row 4: clean -> resets active state
        handle_violations([], mode='report', directories=dirs, tracker=tracker)
        # Row 5: second crossing of E_residual_cons_frac_gross -> logs with (count=4)
        handle_violations([v_energy_1], mode='report', directories=dirs, tracker=tracker)
        tracker.log_summary()

    energy_msgs = [
        r.message
        for r in caplog.records
        if 'Invariant report [E_residual_cons_frac_gross]' in r.message
    ]
    desync_msgs = [
        r.message
        for r in caplog.records
        if 'Invariant report [struct_mass_desync_frac]' in r.message
    ]
    step_msgs = [
        r.message for r in caplog.records if 'Invariant report [step_dE_inc_frac]' in r.message
    ]
    bounds_msgs = [
        r.message for r in caplog.records if 'Invariant report [T_surf_bounds]' in r.message
    ]
    summary_msgs = [r.message for r in caplog.records if 'Invariant summary:' in r.message]
    assert len(energy_msgs) == 2
    assert '(count=1)' in energy_msgs[0]
    assert '(count=4)' in energy_msgs[1]
    assert len(desync_msgs) == 1
    assert '(count=1)' in desync_msgs[0]
    assert len(step_msgs) == 1
    assert '(count=1)' in step_msgs[0]
    assert len(bounds_msgs) == 2
    assert '-5.000000e+00' in bounds_msgs[1]
    assert tracker.row_counts['E_residual_cons_frac_gross'] == 4
    assert tracker.row_counts['struct_mass_desync_frac'] == 3
    assert tracker.row_counts['step_dE_inc_frac'] == 2
    assert tracker.row_counts['T_surf_bounds'] == 2
    assert tracker.rows_with_violations == 4
    assert len(summary_msgs) == 1
    assert '4 row(s) with violations' in summary_msgs[0]
    assert 'E_residual_cons_frac_gross=4 (max=6.000000e-04)' in summary_msgs[0]
    assert 'struct_mass_desync_frac=3 (max=6.000000e-03)' in summary_msgs[0]
    assert 'step_dE_inc_frac=2 (max=5.000000e-01)' in summary_msgs[0]
    assert 'T_surf_bounds=2 (max=5.000000e+00)' in summary_msgs[0]

    # Structure-solve value changes re-log: desync 6e-3, 6e-3, 5e-2 logs the 6e-3 warning once and logs the 5e-2 escalation row:
    caplog.clear()
    esc_tracker = InvariantTracker()
    v_desync_stop = InvariantViolation(
        name='struct_mass_desync_frac',
        status_code=STATUS_ERROR_MASS_INVARIANT,
        value=5.0e-2,
        tolerance=1.0e-2,
        message='Time=3.0e2 yr: struct_mass_desync_frac=5.000000e-02 exceeds stop tol=1.000000e-02',
        report_only=False,
    )
    v_core_1 = InvariantViolation(
        name='M_core_target_rel_err',
        status_code=STATUS_ERROR_MASS_INVARIANT,
        value=3.0e-2,
        tolerance=2.5e-2,
        message='Time=1.0e2 yr: |M_core / M_core_target - 1|=3.000000e-02 exceeds tol=2.500000e-02',
        report_only=False,
    )
    v_core_2 = InvariantViolation(
        name='M_core_target_rel_err',
        status_code=STATUS_ERROR_MASS_INVARIANT,
        value=3.5e-2,
        tolerance=2.5e-2,
        message='Time=3.0e2 yr: |M_core / M_core_target - 1|=3.500000e-02 exceeds tol=2.500000e-02',
        report_only=False,
    )
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        handle_violations(
            [v_desync_1, v_core_1], mode='report', directories=dirs, tracker=esc_tracker
        )
        handle_violations(
            [v_desync_2, v_core_1], mode='report', directories=dirs, tracker=esc_tracker
        )
        handle_violations(
            [v_desync_stop, v_core_2], mode='report', directories=dirs, tracker=esc_tracker
        )

    esc_desync_msgs = [
        r.message
        for r in caplog.records
        if 'Invariant report [struct_mass_desync_frac]' in r.message
    ]
    esc_core_msgs = [
        r.message
        for r in caplog.records
        if 'Invariant report [M_core_target_rel_err]' in r.message
    ]
    assert len(esc_desync_msgs) == 2
    assert '6.000000e-03' in esc_desync_msgs[0]
    assert '(count=1)' in esc_desync_msgs[0]
    assert '5.000000e-02' in esc_desync_msgs[1]
    assert '(count=3)' in esc_desync_msgs[1]
    assert len(esc_core_msgs) == 2
    assert '3.000000e-02' in esc_core_msgs[0]
    assert '(count=1)' in esc_core_msgs[0]
    assert '3.500000e-02' in esc_core_msgs[1]
    assert '(count=3)' in esc_core_msgs[1]

    stop_tracker = InvariantTracker()
    handle_violations([v_desync_1], mode='stop', directories=dirs, tracker=stop_tracker)
    handle_violations([v_desync_2], mode='stop', directories=dirs, tracker=stop_tracker)
    with pytest.raises(InvariantViolationError) as stop_exc:
        handle_violations([v_desync_stop], mode='stop', directories=dirs, tracker=stop_tracker)
    assert stop_exc.value.invariant_name == 'struct_mass_desync_frac'
    assert stop_exc.value.value == pytest.approx(5.0e-2)

    # Non-finite values (-inf, nan) must not print max=-inf or hide the finite maximum:
    caplog.clear()
    nf_tracker = InvariantTracker()
    for val in (float('-inf'), float('nan'), -0.25):
        nf_tracker.record_row(
            [
                InvariantViolation(
                    name='step_dE_core_J',
                    status_code=STATUS_ERROR_ENERGY_INVARIANT,
                    value=val,
                    tolerance=0.0,
                    message=f'step_dE_core_J={val}',
                    report_only=True,
                )
            ]
        )
    nf_tracker.record_row(
        [
            InvariantViolation(
                name='core_C_eff',
                status_code=STATUS_ERROR_ENERGY_INVARIANT,
                value=float('-inf'),
                tolerance=0.0,
                message='core_C_eff=-inf',
                report_only=True,
            )
        ]
    )
    assert nf_tracker.max_values['step_dE_core_J'] == pytest.approx(0.25)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        nf_tracker.log_summary()
    nf_summary = [r.message for r in caplog.records if 'Invariant summary:' in r.message]
    assert len(nf_summary) == 1
    assert 'max=-inf' not in nf_summary[0]
    assert '2.500000e-01' in nf_summary[0]
    assert 'non_finite=2' in nf_summary[0]


def test_main_loop_logs_invariant_summary_on_module_exception(
    tmp_path, monkeypatch, caplog: pytest.LogCaptureFixture
):
    """When a module raises mid-loop, try/finally around the main loop still emits the invariant summary."""
    from pathlib import Path

    import proteus.interior_energetics.wrapper as interior_wrapper
    from proteus import Proteus
    from proteus.interior_energetics.common import Interior_t

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    out = tmp_path / 'run_exc'
    dummy_toml = Path(__file__).resolve().parents[2] / 'input' / 'dummy.toml'
    text = dummy_toml.read_text(encoding='utf-8')
    cfg_path = tmp_path / 'exc_summary.toml'
    cfg_path.write_text(text.replace('path = "auto"', f'path = "{out}"', 1), encoding='utf-8')

    runner = Proteus(config_path=cfg_path)
    config = runner.config
    config.invariants.mode = 'report'
    config.atmos_chem.module = None
    config.params.stop.solid.enabled = False
    config.params.stop.iters.minimum = 1
    config.params.stop.time.minimum = 0.0
    config.params.stop.time.maximum = 500.0
    config.params.out.plot_mod = None
    config.params.out.archive_mod = 'none'

    real_interior = interior_wrapper.run_interior

    def interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        if runner.loops['total'] in (1, 2):
            hf_row['Phi_global'] = 1.5
        elif runner.loops['total'] == 3:
            raise RuntimeError('simulated interior solver failure')

    monkeypatch.setattr(interior_wrapper, 'run_interior', interior)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        with pytest.raises(RuntimeError, match='simulated interior solver failure'):
            runner.start(resume=False, offline=True)

    summary_msgs = [r.message for r in caplog.records if 'Invariant summary:' in r.message]
    assert len(summary_msgs) == 1
    assert '2 row(s) with violations' in summary_msgs[0]
    assert 'Phi_global_bounds=2 (max=1.500000e+00)' in summary_msgs[0]


def test_init_stage_structure_desync_reports_without_stopping_in_stop_mode(
    caplog: pytest.LogCaptureFixture,
):
    """Init-stage rows before the baseline structure solve report desync without stopping in stop mode."""
    from proteus.utils.invariants import (
        InvariantTracker,
        InvariantViolationError,
        check_row_invariants,
        check_structure_invariants,
    )

    cfg = _make_mock_config(mode='stop')
    tracker = InvariantTracker()
    row0 = {
        'Time': 0.0,
        'T_surf': 3500.0,
        'T_magma': 3500.0,
        'P_surf': 100.0,
        'Phi_global': 1.0,
        'M_atm': 1.0e21,
        'M_planet': 7.973078065748122e24,
        'M_vol_atm': 1.0e21,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e21,
        'M_int': 6.0320092652009745e24,
        'M_core': 1.9410688005471474e24,
        'R_int': 6.365225673558478e6,
        'R_core': 3.504850166865783e6,
        'struct_mass_desync_frac': 1.123330005012071e-2,
    }

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        try:
            v0 = check_row_invariants(row0, cfg, tracker=tracker)
        except InvariantViolationError as exc:
            pytest.fail(f'init-stage row 0 raised InvariantViolationError: {exc}')
    assert len(v0) == 1
    assert v0[0].name == 'struct_mass_desync_frac'
    assert v0[0].report_only is True
    report_lines = [
        r.message
        for r in caplog.records
        if 'Invariant report [struct_mass_desync_frac]' in r.message
    ]
    assert len(report_lines) == 1

    # Final init-stage iteration at Time=1.0 yr stays report_only before baseline structure solve
    row3 = dict(row0, Time=1.0)
    try:
        v3 = check_row_invariants(row3, cfg, tracker=tracker, init_stage=True)
        v3_none = check_row_invariants(row3, cfg, tracker=tracker, init_stage=None)
    except InvariantViolationError as exc:
        pytest.fail(f'init-stage row 3 raised InvariantViolationError: {exc}')
    assert len(v3) == 1
    assert v3[0].report_only is True
    assert len(v3_none) == 1
    assert v3_none[0].report_only is True

    # Just above Time=1.0 yr with init_stage=None defaults to post-baseline and stops:
    with pytest.raises(InvariantViolationError) as exc_after_1yr:
        check_row_invariants(dict(row0, Time=1.0 + 1.0e-9), cfg, init_stage=None)
    assert exc_after_1yr.value.invariant_name == 'struct_mass_desync_frac'

    # Post-baseline row 4 at Time=2.0 yr with resolved baseline structure passes cleanly
    row4 = dict(
        row0,
        Time=2.0,
        M_int=5.970155248957322e24,
        M_core=1.9409866756544637e24,
        R_int=6.371213878930453e6,
        R_core=3.504800733997288e6,
        struct_mass_desync_frac=1.7210924886631676e-3,
    )
    assert check_row_invariants(row4, cfg, tracker=tracker, init_stage=False) == []

    # The same 1.123e-2 desync on a post-baseline row stops in stop mode
    post_bad = dict(row0, Time=2.0)
    with pytest.raises(InvariantViolationError) as exc_info:
        check_row_invariants(post_bad, cfg, tracker=tracker, init_stage=False)
    assert exc_info.value.status_code == 31
    assert exc_info.value.invariant_name == 'struct_mass_desync_frac'

    # Target mismatches on an init-stage row are also report-only, while NaN still stops
    init_targets = dict(
        row0,
        M_int_target=5.970155248957322e24,
        M_core_target=1.800e24,
    )
    v_init_t = check_structure_invariants(init_targets, cfg, init_stage=True)
    assert [v.name for v in v_init_t] == [
        'struct_mass_desync_frac',
        'M_int_target_rel_err',
        'M_core_target_rel_err',
    ]
    assert all(v.report_only for v in v_init_t)


def test_atmodeller_mass_thresh_for_chons_and_strict_noble_gases():
    """Under atmodeller, CHONS use mass_thresh + rtol * total only when total <= mass_thresh while solved budgets and noble gases stay strict."""
    from proteus.utils.constants import noble_gases, vol_element_list
    from proteus.utils.invariants import check_volatile_invariants

    mass_thresh = 1.0e16
    cfg = _make_mock_config(
        mode='stop',
        outgas_module='atmodeller',
        mass_thresh=mass_thresh,
        fO2_source='from_O_budget',
        vapourise=False,
    )
    rtol = float(cfg.invariants.element_closure_rtol)

    base_row = {
        'Time': 100.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e21,
        'M_vol_atm': 1.0e21,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e21,
    }

    for elem in vol_element_list:
        # Sub-threshold (total <= mass_thresh): atmodeller skips partitioning, so 100% unpartitioned passes
        row_sub = dict(base_row, **{f'{elem}_kg_total': mass_thresh})
        assert check_volatile_invariants(row_sub, cfg) == []

        # Just above mass_thresh (1.00025e16 kg): a 100% leak must stop
        row_just_above = dict(base_row, **{f'{elem}_kg_total': 1.00025e16})
        v_just_above = check_volatile_invariants(row_just_above, cfg)
        assert [v.name for v in v_just_above] == [f'{elem}_reservoir_closure']
        assert v_just_above[0].value == pytest.approx(1.0)
        assert v_just_above[0].tolerance == pytest.approx(rtol)
        assert v_just_above[0].report_only is False

        # Solved budget at 10 x mass_thresh with a 5% leak stops
        total_10x = 10.0 * mass_thresh
        row_10x_leak = dict(
            base_row,
            **{
                f'{elem}_kg_total': total_10x,
                f'{elem}_kg_atm': 0.95 * total_10x,
            },
        )
        v_10x = check_volatile_invariants(row_10x_leak, cfg)
        assert [v.name for v in v_10x] == [f'{elem}_reservoir_closure']
        assert v_10x[0].value == pytest.approx(0.05)
        assert v_10x[0].tolerance == pytest.approx(rtol)
        assert v_10x[0].report_only is False

    for gas in noble_gases:
        # Noble gases remain strict even when 0 < total <= mass_thresh
        row_noble_sub = dict(base_row, **{f'{gas}_kg_total': 1.0e14})
        v_noble = check_volatile_invariants(row_noble_sub, cfg)
        assert [v.name for v in v_noble] == [f'{gas}_reservoir_closure']
        assert v_noble[0].value == pytest.approx(1.0)
        assert v_noble[0].tolerance == pytest.approx(rtol)
        assert v_noble[0].report_only is False

        row_noble_ok = dict(
            base_row,
            **{f'{gas}_kg_total': 1.0e14, f'{gas}_kg_atm': 1.0e14 * (1.0 - 0.5 * rtol)},
        )
        assert check_volatile_invariants(row_noble_ok, cfg) == []


def test_dummy_outgas_skips_noble_gas_reservoir_closure():
    """Under outgas.module = 'dummy', noble gases skip reservoir closure while CHONS and bounds remain active."""
    from proteus.utils.constants import noble_gases
    from proteus.utils.invariants import (
        InvariantViolationError,
        check_row_invariants,
        check_volatile_invariants,
    )

    cfg_dummy = _make_mock_config(
        mode='stop',
        outgas_module='dummy',
        fO2_source='user_constant',
        vapourise=False,
    )
    base_row = {
        'Time': 100.0,
        'T_surf': 2500.0,
        'T_magma': 2500.0,
        'P_surf': 100.0,
        'Phi_global': 0.5,
        'M_planet': 6.0e24,
        'M_atm': 1.0e21,
        'M_vol_atm': 1.0e21,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e21,
    }

    for gas in noble_gases:
        row_noble = dict(base_row, **{f'{gas}_kg_total': 1.0e14})
        assert check_volatile_invariants(row_noble, cfg_dummy) == []
        try:
            assert check_row_invariants(row_noble, cfg_dummy) == []
        except InvariantViolationError as exc:
            pytest.fail(
                f'dummy outgassing raised InvariantViolationError on {gas}_kg_total: {exc}'
            )

    # Partitioned volatile elements (H, C, N, S) still enforce reservoir closure under dummy
    for elem in ('H', 'C', 'N', 'S'):
        row_vol = dict(base_row, **{f'{elem}_kg_total': 1.0e14})
        v_vol = check_volatile_invariants(row_vol, cfg_dummy)
        assert [v.name for v in v_vol] == [f'{elem}_reservoir_closure']
        assert v_vol[0].report_only is False

    # Noble-gas reservoir lower bounds remain enforced under dummy
    row_neg_noble = dict(base_row, Ar_kg_total=1.0e14, Ar_kg_liquid=-5.0)
    v_neg = check_volatile_invariants(row_neg_noble, cfg_dummy)
    assert [v.name for v in v_neg] == ['Ar_kg_liquid_bounds']


def test_smin_clip_temp_k_one_row_and_two_column_tables_warn_once_without_raising(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    """A 1-row, 2-column, or n_P=0 P-S table gives no clip and one warning per path in report mode."""
    import pandas as pd

    from proteus.utils.invariants import _smin_clip_temp_k, check_row_invariants

    hdr_lines = (
        '# 5 2 1\n'
        '# Pressure [nondim], Entropy [nondim], Quantity [nondim]\n'
        '# column * scaling factor = SI units\n'
        '# scaling factors (constant) for each column given on line below\n'
        '# 1000000000.0 4824266.84604467 1000.0\n'
    )
    dir_1row = tmp_path / 'eos_1row'
    dir_1row.mkdir()
    (dir_1row / 'temperature_solid.dat').write_text(
        hdr_lines + '1.0e-04 2.199633e-05 3.023417e-01\n',
        encoding='utf-8',
    )

    dir_2col = tmp_path / 'eos_2col'
    dir_2col.mkdir()
    (dir_2col / 'temperature_solid.dat').write_text(
        hdr_lines + '1.0e-04 2.199633e-05\n2.595514e-01 2.199633e-05\n',
        encoding='utf-8',
    )

    dir_np0 = tmp_path / 'eos_np0'
    dir_np0.mkdir()
    (dir_np0 / 'temperature_solid.dat').write_text(
        hdr_lines.replace('# 5 2 1\n', '# 5 0 1\n', 1) + '1.0e-04 2.199633e-05 3.023417e-01\n',
        encoding='utf-8',
    )

    cfg = _make_mock_config(mode='report')
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
    step_gross = 1.0e28
    row = {
        'Time': 200.0,
        'P_surf': 1.0,
        'T_surf': 300.0,
        'T_magma': 300.0,
        'Phi_global': 0.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e21,
        'M_vol_atm': 1.0e21,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e21,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': -1.01e30,
        'E_residual_cons_J': 0.0,
        'step_dE_state_heat_J': -step_gross * (1.0 - 0.02),
        'step_dE_F_int_J': -step_gross,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
    }

    for label, eos_dir, expected_phrase in (
        ('1row', dir_1row, 'Expected 2D P-S table'),
        ('2col', dir_2col, 'Expected 2D P-S table'),
        ('np0', dir_np0, 'Invalid n_P=0'),
    ):
        caplog.clear()
        dirs = {'spider_eos_dir': str(eos_dir)}
        with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
            try:
                clip = _smin_clip_temp_k(row, dirs)
                v1 = check_row_invariants(row, cfg, hf_all=hf_all, directories=dirs)
                v2 = check_row_invariants(row, cfg, hf_all=hf_all, directories=dirs)
            except (IndexError, ValueError, OSError) as exc:
                pytest.fail(f'{label} table raised {type(exc).__name__} in report mode: {exc}')
        assert clip is None
        assert [v.name for v in v1] == ['step_dE_inc_frac']
        assert v1[0].tolerance == pytest.approx(cfg.invariants.step_energy_rtol)
        assert [v.name for v in v2] == ['step_dE_inc_frac']
        warn_msgs = [
            r.message for r in caplog.records if 'Failed to load P-S EOS table' in r.message
        ]
        assert len(warn_msgs) == 1
        assert expected_phrase in warn_msgs[0]


def test_structure_rollback_makes_target_errors_report_only_while_normal_rows_stop(
    tmp_path, monkeypatch
):
    """A rejected up-step or convergence rollback in update_structure_from_interior marks M_int_target_rel_err and M_core_target_rel_err report_only=True while normal rows stop."""
    import numpy as np

    import proteus.interior_energetics.wrapper as int_wrap
    import proteus.interior_struct.zalmoxis as zalmoxis_mod
    from proteus.config import Config
    from proteus.interior_energetics.common import Interior_t
    from proteus.utils.invariants import (
        InvariantViolationError,
        check_row_invariants,
        check_structure_invariants,
    )

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    cfg = Config()
    cfg.invariants.mode = 'stop'
    cfg.interior_struct.module = 'zalmoxis'
    cfg.interior_energetics.module = 'aragog'
    cfg.planet.temperature_mode = 'liquidus_super'
    cfg.interior_struct.zalmoxis.mantle_eos = 'PALEOS-2phase:MgSiO3'
    cfg.interior_struct.zalmoxis.dry_mantle = True

    data_dir = tmp_path / 'data'
    data_dir.mkdir(parents=True)
    (data_dir / 'zalmoxis_output.dat').write_text('0.0 0.0\n', encoding='utf-8')
    dirs = {'output': str(tmp_path), 'output/data': str(data_dir)}

    interior_o = Interior_t(10)
    interior_o.ic = 2
    interior_o.radius = np.linspace(3.48e6, 6.37e6, 11)
    interior_o.temp = np.linspace(4000.0, 2500.0, 10)

    base_row = {
        'Time': 200.0,
        'T_surf': 2000.0,
        'T_magma': 2500.0,
        'T_cmb': 4000.0,
        'P_surf': 100.0,
        'Phi_global': 0.8,
        'M_planet': 5.972e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e20,
        'R_int': 6.370e6,
        'R_core': 3.480e6,
        'M_int': 5.970e24,
        'M_int_target': 5.970e24,
        'M_core': 1.940e24,
        'M_core_target': 1.940e24,
        'M_mantle': 4.030e24,
        'P_center': 3.6e11,
        'P_cmb': 1.35e11,
        'gravity': 9.81,
        'rho_avg': 5515.0,
        'struct_mass_desync_frac': 1.0e-3,
    }

    # 1. Monotonic-radius up-step reject: zalmoxis_solver writes a new M_int_target (+0.6%)
    # and M_core_target (+4.0%) with a slightly larger R_int while cooling; the wrapper
    # restores _saved_structure (M_int=5.970e24, M_core=1.940e24) and leaves the new targets.
    def fake_upstep_solver(config, outdir, hf_row, **kwargs):
        hf_row['R_int'] = 6.370e6 * 1.005
        hf_row['M_int_target'] = 5.970e24 * 1.006
        hf_row['M_int'] = 5.970e24 * 1.006
        hf_row['M_core_target'] = 1.940e24 * 1.04
        hf_row['M_core'] = 1.940e24 * 1.04
        return 3.480e6, None

    monkeypatch.setattr(zalmoxis_mod, 'zalmoxis_solver', fake_upstep_solver)
    row_upstep = dict(base_row)
    int_wrap.update_structure_from_interior(
        dirs,
        cfg,
        row_upstep,
        interior_o,
        last_struct_time=0.0,
        last_Tmagma=2600.0,
        last_Phi=0.85,
        force=True,
    )
    assert row_upstep['M_int'] == pytest.approx(5.970e24)
    assert row_upstep['M_int_target'] == pytest.approx(5.970e24 * 1.006)
    assert row_upstep.get('_structure_rollback') is True
    assert interior_o.structure_rollback_active is True

    # In stop mode, the row following the rejected up-step continues like off mode (report_only=True),
    # whether read from hf_row['_structure_rollback'] alone or interior_o.structure_rollback_active alone:
    v_upstep = check_row_invariants(row_upstep, cfg, init_stage=False, interior_o=interior_o)
    assert [v.name for v in v_upstep] == ['M_int_target_rel_err', 'M_core_target_rel_err']
    assert all(v.report_only for v in v_upstep)
    v_hf_only = check_row_invariants(row_upstep, cfg, init_stage=False, interior_o=None)
    assert [v.name for v in v_hf_only] == ['M_int_target_rel_err', 'M_core_target_rel_err']
    assert all(v.report_only for v in v_hf_only)
    row_no_hf_flag = {k: v for k, v in row_upstep.items() if k != '_structure_rollback'}
    v_obj_only = check_row_invariants(
        row_no_hf_flag, cfg, init_stage=False, interior_o=interior_o
    )
    assert [v.name for v in v_obj_only] == ['M_int_target_rel_err', 'M_core_target_rel_err']
    assert all(v.report_only for v in v_obj_only)

    # 2. Convergence / mass-anchor RuntimeError rollback also marks target checks report_only=True:
    def fake_failing_solver(config, outdir, hf_row, **kwargs):
        hf_row['M_int_target'] = 5.970e24 * 1.008
        hf_row['M_int'] = 5.970e24 * 1.020
        hf_row['M_core_target'] = 1.940e24 * 1.05
        hf_row['M_core'] = 1.940e24 * 1.05
        return 3.480e6, None

    monkeypatch.setattr(zalmoxis_mod, 'zalmoxis_solver', fake_failing_solver)
    row_fail = dict(base_row)
    interior_o_fail = Interior_t(10)
    interior_o_fail.ic = 2
    interior_o_fail.radius = interior_o.radius
    interior_o_fail.temp = interior_o.temp
    int_wrap.update_structure_from_interior(
        dirs,
        cfg,
        row_fail,
        interior_o_fail,
        last_struct_time=0.0,
        last_Tmagma=2600.0,
        last_Phi=0.85,
        force=True,
    )
    assert row_fail.get('_structure_rollback') is True
    assert interior_o_fail.structure_rollback_active is True
    v_fail = check_row_invariants(row_fail, cfg, init_stage=False, interior_o=interior_o_fail)
    assert [v.name for v in v_fail] == ['M_int_target_rel_err', 'M_core_target_rel_err']
    assert all(v.report_only for v in v_fail)

    # 3. Subsequent accepted down-step clears the rollback marker, and a target error on a normal row stops:
    def fake_accepted_solver(config, outdir, hf_row, **kwargs):
        hf_row['R_int'] = 6.350e6
        hf_row['M_int_target'] = 5.970e24
        hf_row['M_int'] = 5.970e24
        hf_row['M_core_target'] = 1.940e24
        hf_row['M_core'] = 1.940e24
        return 3.480e6, None

    monkeypatch.setattr(zalmoxis_mod, 'zalmoxis_solver', fake_accepted_solver)
    int_wrap.update_structure_from_interior(
        dirs,
        cfg,
        row_upstep,
        interior_o,
        last_struct_time=0.0,
        last_Tmagma=2600.0,
        last_Phi=0.85,
        force=True,
    )
    assert row_upstep.get('_structure_rollback') is False
    assert interior_o.structure_rollback_active is False
    assert check_row_invariants(row_upstep, cfg, init_stage=False, interior_o=interior_o) == []

    row_normal_err = dict(row_upstep, M_int=5.970e24 * 1.005)
    v_normal = check_structure_invariants(
        row_normal_err, cfg, init_stage=False, interior_o=interior_o
    )
    assert [v.name for v in v_normal] == ['M_int_target_rel_err']
    assert v_normal[0].report_only is False
    with pytest.raises(InvariantViolationError) as exc_normal:
        check_row_invariants(row_normal_err, cfg, init_stage=False, interior_o=interior_o)
    assert exc_normal.value.invariant_name == 'M_int_target_rel_err'


def test_negative_element_total_checked_before_derived_skip():
    """Verify negative element total triggers reservoir closure even when element is in derived skip set."""
    from proteus.utils.invariants import check_volatile_invariants

    # 1. Negative O_kg_total under fO2_source = 'user_constant' (O in derived)
    cfg_o = _make_mock_config(fO2_source='user_constant')
    row_o = {'Time': 100.0, 'M_planet': 6.0e24, 'O_kg_total': -10.0}
    v_o = check_volatile_invariants(row_o, cfg_o)
    assert [v.name for v in v_o] == ['O_reservoir_closure']
    assert v_o[0].value == pytest.approx(-10.0)

    # 2. Negative rock-vapour total under vapourise = True (Si in derived)
    cfg_vap = _make_mock_config(vapourise=True)
    row_vap = {'Time': 100.0, 'M_planet': 6.0e24, 'Si_kg_total': -10.0}
    v_vap = check_volatile_invariants(row_vap, cfg_vap)
    assert [v.name for v in v_vap] == ['Si_reservoir_closure']
    assert v_vap[0].value == pytest.approx(-10.0)

    # 3. Negative noble-gas total under outgas.module = 'dummy' (He in derived)
    cfg_dummy = _make_mock_config(outgas_module='dummy')
    row_dummy = {'Time': 100.0, 'M_planet': 6.0e24, 'He_kg_total': -10.0}
    v_dummy = check_volatile_invariants(row_dummy, cfg_dummy)
    assert [v.name for v in v_dummy] == ['He_reservoir_closure']
    assert v_dummy[0].value == pytest.approx(-10.0)


def test_crossing_suppressed_names_log_once_across_n_identical_rows(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    """Feed N identical violating rows for every suppressed cumulative, step, and closure name and assert 1 log line per name."""
    from proteus.utils.constants import element_list
    from proteus.utils.invariants import _CROSSING_SUPPRESSED_NAMES, InvariantTracker

    expected_names = {
        'E_residual_cons_frac_gross',
        'solver_residual_frac',
        'E_core_residual_frac',
        'step_dE_inc_frac',
        'escape_cumulative_bound',
        'struct_mass_desync_frac',
        'M_core_target_rel_err',
        'M_int_target_rel_err',
        *(f'{e}_reservoir_closure' for e in element_list),
    }
    assert expected_names <= _CROSSING_SUPPRESSED_NAMES

    n_rows = 10
    dirs = {'output': str(tmp_path)}
    for name in sorted(expected_names):
        caplog.clear()
        tracker = InvariantTracker()
        v = InvariantViolation(
            name=name,
            status_code=STATUS_ERROR_ENERGY_INVARIANT
            if 'E_' in name or 'solver' in name
            else STATUS_ERROR_MASS_INVARIANT,
            value=0.05,
            tolerance=0.01,
            message=f'Time=100.0 yr: {name}=5.000000e-02 exceeds tol=1.000000e-02',
            report_only=True,
        )
        with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
            for _ in range(n_rows):
                handle_violations([v], mode='report', directories=dirs, tracker=tracker)
        msgs = [r.message for r in caplog.records if f'Invariant report [{name}]' in r.message]
        assert len(msgs) == 1, (
            f'Expected 1 log line for {name} across {n_rows} rows, got {len(msgs)}'
        )
        assert '(count=1)' in msgs[0]
        assert tracker.row_counts[name] == n_rows


def test_smin_clip_temp_k_uses_fixed_aragog_mesh_surface_pressure_with_p_surf_fallback(
    tmp_path,
):
    """Evaluate T(p, S_min) at the fixed Aragog mesh surface pressure, falling back to P_surf only when no solver state exists."""
    from types import SimpleNamespace

    import pandas as pd

    from proteus.utils.invariants import _smin_clip_temp_k, check_row_invariants

    eos_dir = tmp_path / 'eos'
    eos_dir.mkdir()
    table_text = (
        '# 5 2 1\n'
        '# Pressure [nondim], Entropy [nondim], Quantity [nondim]\n'
        '# column * scaling factor = SI units\n'
        '# scaling factors (constant) for each column given on line below\n'
        '# 1000000000.0 4824266.84604467 1000.0\n'
        '1.0e-04 2.199633e-05 3.000000e-01\n'
        '1.0e+01 2.199633e-05 4.000000e-01\n'
    )
    (eos_dir / 'temperature_solid.dat').write_text(table_text, encoding='utf-8')
    dirs = {'spider_eos_dir': str(eos_dir), 'output': str(tmp_path)}

    # Solver mesh surface pressure fixed at 1 bar (1e5 Pa -> T_smin = 300.0 K, clip = 300.5 K),
    # while row P_surf has evolved to 1e5 bar (1e10 Pa -> T_smin = 400.0 K, clip = 400.5 K).
    interior_o = SimpleNamespace(
        aragog_solver=SimpleNamespace(
            parameters=SimpleNamespace(mesh=SimpleNamespace(surface_pressure=1.0e5))
        )
    )
    row = {
        'Time': 200.0,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': -1.03e30,
        'step_dE_state_heat_J': -3.0e28,
        'step_dE_F_int_J': -1.0e30,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'T_magma': 350.0,
        'T_surf': 300.0,
        'P_surf': 1.0e5,
        'Phi_global': 0.0,
        'M_planet': 6.0e24,
        'M_int': 6.0e24,
        'M_int_target': 6.0e24,
        'M_core': 2.0e24,
        'M_core_target': 2.0e24,
        'M_atm': 1.0e21,
        'M_vol_atm': 1.0e21,
        'M_vaps': 0.0,
    }
    # With solver state present: evaluated at mesh surface_pressure = 1e5 Pa -> clip = 300.5 K
    assert _smin_clip_temp_k(row, dirs, interior_o=interior_o) == pytest.approx(300.5)
    # Without solver state: falls back to row P_surf = 1e5 bar (1e10 Pa) -> clip = 400.5 K
    assert _smin_clip_temp_k(row, dirs, interior_o=None) == pytest.approx(400.5)

    # Through check_row_invariants: T_magma = 350.0 K is above the 300.5 K mesh clip, so
    # step_energy_rtol (0.01) applies when interior_o is passed, whereas fallback (400.5 K)
    # would treat 350.0 K as on the S_min clip (step_energy_smin_rtol = 0.05).
    cfg = _make_mock_config(mode='report')
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
    # Set step_dE_state_heat_J = -9.7e29 J vs dE_inc_cons = -1.0e30 J -> step_err = 0.03 (between 0.01 and 0.05)
    row_step = dict(row, step_dE_state_heat_J=-9.7e29)
    viols_with_solver = check_row_invariants(
        row_step, cfg, hf_all=hf_all, directories=dirs, interior_o=interior_o
    )
    assert [v.name for v in viols_with_solver] == ['step_dE_inc_frac']
    assert viols_with_solver[0].tolerance == pytest.approx(cfg.invariants.step_energy_rtol)


def test_smin_table_cache_key_includes_size_mtime_and_head_bytes(tmp_path):
    """Verify _SMIN_TABLE_CACHE keys on (path_key, st_size, st_mtime_ns, head_bytes) and invalidates when size or mtime changes."""
    import os

    from proteus.utils import invariants as inv_mod

    inv_mod._SMIN_TABLE_CACHE.clear()
    inv_mod._SMIN_TABLE_WARNED_PATHS.clear()

    eos_dir = tmp_path / 'eos_cache_key'
    eos_dir.mkdir()
    table_file = eos_dir / 'temperature_solid.dat'
    base_table = (
        '# 5 2 2\n'
        '# Pressure [nondim], Entropy [nondim], Quantity [nondim]\n'
        '# column * scaling factor = SI units\n'
        '# scaling factors (constant) for each column given on line below\n'
        '# 1000000000.0 4824266.84604467 1000.0\n'
        '1.0e-04 2.0e-05 3.000000e-01\n'
        '1.0e+01 2.0e-05 3.100000e-01\n'
        '1.0e-04 3.0e-05 3.500000e-01\n'
        '1.0e+01 3.0e-05 3.600000e-01\n'
    )
    table_file.write_text(base_table, encoding='utf-8')
    st0 = table_file.stat()
    head0 = inv_mod._read_smin_block_bytes(table_file)
    dirs = {'spider_eos_dir': str(eos_dir)}

    clip0 = inv_mod._smin_clip_temp_k({'P_surf': 1.0}, dirs)
    assert clip0 == pytest.approx(300.5)
    expected_key = (str(table_file.resolve()), st0.st_size, st0.st_mtime_ns, head0)
    assert expected_key in inv_mod._SMIN_TABLE_CACHE

    # Changing the second block (beyond head_bytes) with a new mtime or size invalidates cache:
    bad_second_block = base_table.replace('3.0e-05', '1.0e-05')
    table_file.write_text(bad_second_block, encoding='utf-8')
    os.utime(table_file, ns=(st0.st_atime_ns, st0.st_mtime_ns + 1_000_000))
    assert inv_mod._smin_clip_temp_k({'P_surf': 1.0}, dirs) is None


def test_resolve_ps_table_path_uses_zalmoxis_ps_cache_pointer_name(tmp_path):
    """Verify _resolve_ps_table_path reads the pointer file written by zalmoxis._write_ps_cache_pointer."""
    from proteus.interior_struct.zalmoxis import PS_CACHE_POINTER_NAME, _write_ps_cache_pointer
    from proteus.utils import constants as const_mod
    from proteus.utils import invariants as inv_mod

    assert (
        const_mod.PS_CACHE_POINTER_NAME
        == inv_mod.PS_CACHE_POINTER_NAME
        == PS_CACHE_POINTER_NAME
        == 'spider_eos_cache.txt'
    )

    out_dir = tmp_path / 'run_out'
    cache_dir = tmp_path / 'shared_ps_cache'
    cache_dir.mkdir(parents=True)
    table_path = cache_dir / 'temperature_solid.dat'
    table_path.write_text('# 1 1 1\n0.0 0.0 300.0\n', encoding='utf-8')

    _write_ps_cache_pointer(str(out_dir), str(cache_dir))
    resolved = inv_mod._resolve_ps_table_path({'output': str(out_dir)})
    assert resolved == table_path.resolve()


def test_import_proteus_does_not_eagerly_load_aragog_zalmoxis_or_netcdf4(tmp_path):
    """Importing proteus.proteus and proteus.utils.invariants from repo src with juliacall stubbed does not load aragog, zalmoxis, or netCDF4."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_src = Path(__file__).resolve().parents[2] / 'src'
    code = (
        'import json, sys, types; '
        'sys.path.insert(0, %r); '
        "jl = types.ModuleType('juliacall'); jl.Main = types.SimpleNamespace(); sys.modules['juliacall'] = jl; "
        'import proteus, proteus.proteus, proteus.utils.invariants; '
        'mods = ["proteus.interior_energetics.aragog", "proteus.interior_struct.zalmoxis", "netCDF4"]; '
        'print(json.dumps({"file": proteus.__file__, "loaded": {m: (m in sys.modules) for m in mods}}))'
    ) % str(repo_src)
    env = dict(os.environ, PYTHONPATH=str(repo_src))
    proc = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(tmp_path),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout.strip())
    assert Path(payload['file']).resolve().is_relative_to(repo_src.resolve())
    loaded = payload['loaded']
    assert loaded == {
        'proteus.interior_energetics.aragog': False,
        'proteus.interior_struct.zalmoxis': False,
        'netCDF4': False,
    }
    assert not any(loaded.values())


def test_proteus_row_init_stage_helper_all_cases(tmp_path, monkeypatch):
    """Verify Proteus._row_init_stage across init iteration, Zalmoxis pre/post baseline, stale fallback, and non-Zalmoxis."""
    from pathlib import Path
    from types import SimpleNamespace

    from proteus import Proteus
    from proteus.interior_energetics.common import Interior_t

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    dummy_toml = Path(__file__).resolve().parents[2] / 'input' / 'dummy.toml'
    cfg_path = tmp_path / 'row_init_stage.toml'
    cfg_path.write_text(
        dummy_toml.read_text(encoding='utf-8').replace(
            'path = "auto"', f'path = "{tmp_path / "out"}"', 1
        ),
        encoding='utf-8',
    )
    runner = Proteus(config_path=cfg_path)
    runner.interior_o = SimpleNamespace(structure_stale=False)

    # 1. Init iteration is True regardless of structure module
    runner.config.interior_struct.module = 'dummy'
    runner._baseline_structure_done = True
    assert runner._row_init_stage(True) is True

    # 2. Zalmoxis before post-init baseline solve is True
    runner.config.interior_struct.module = 'zalmoxis'
    runner._baseline_structure_done = False
    runner.interior_o.structure_stale = False
    assert runner._row_init_stage(False) is True

    # 3. Zalmoxis after successful post-init baseline solve is False
    runner._baseline_structure_done = True
    runner.interior_o.structure_stale = False
    assert runner._row_init_stage(False) is False

    # 4. Zalmoxis when baseline solve failed and fell back (structure_stale=True) stays True
    runner.init_stage = False
    runner._baseline_structure_done = False
    runner.interior_o.structure_stale = True
    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper._use_superliquidus_adiabat_ic',
        lambda cfg: False,
    )
    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.update_structure_from_interior',
        lambda *a, **kw: (1.0, 2000.0, 0.5),
    )
    runner._solve_structure_baseline_if_needed()
    assert runner._baseline_structure_done is False
    assert runner._row_init_stage(False) is True

    # 5. Zalmoxis after baseline solve succeeded stays False even if a later solve sets structure_stale=True
    runner._baseline_structure_done = True
    runner.interior_o.structure_stale = True
    assert runner._row_init_stage(False) is False

    # 6. Non-Zalmoxis run after init iterations is False even with _baseline_structure_done=False
    runner.config.interior_struct.module = 'dummy'
    runner._baseline_structure_done = False
    runner.interior_o.structure_stale = False
    assert runner._row_init_stage(False) is False


def test_nan_species_and_nan_mplanet_stop_in_check_volatile_invariants():
    """NaN species mass emits {s}_kg_atm_bounds and NaN M_planet emits M_planet_bounds as stopping violations."""
    from proteus.utils.invariants import (
        InvariantViolationError,
        check_row_invariants,
        check_volatile_invariants,
    )

    cfg = _make_mock_config(mode='stop', fO2_source='user_constant', vapourise=False)
    base_row = {
        'Time': 100.0,
        'T_surf': 1500.0,
        'T_magma': 2000.0,
        'P_surf': 100.0,
        'Phi_global': 0.5,
        'M_planet': 6.0e24,
        'M_atm': 1.0e21,
        'M_vol_atm': 1.0e21,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e21,
    }
    v_sp = check_volatile_invariants(dict(base_row, H2O_kg_atm=float('nan')), cfg)
    assert [v.name for v in v_sp] == ['H2O_kg_atm_bounds']
    assert v_sp[0].report_only is False
    with pytest.raises(InvariantViolationError) as exc_sp:
        check_row_invariants(dict(base_row, H2O_kg_atm=float('nan')), cfg)
    assert exc_sp.value.invariant_name == 'H2O_kg_atm_bounds'

    v_mp = check_volatile_invariants(dict(base_row, M_planet=float('nan')), cfg)
    assert [v.name for v in v_mp] == ['M_planet_bounds']
    assert v_mp[0].report_only is False
    with pytest.raises(InvariantViolationError) as exc_mp:
        check_row_invariants(dict(base_row, M_planet=float('nan')), cfg)
    assert exc_mp.value.invariant_name == 'M_planet_bounds'


def test_ic_adiabat_rollback_resets_and_resume_rollback_persistence(
    tmp_path, monkeypatch, caplog
):
    """IC adiabat rollback restores M_int_target and M_core_target without setting _structure_rollback, wrapper resets clear it, and resume_structure_stale restores both stale and up-step rollback flags."""
    import builtins
    import logging
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd

    import proteus.interior_energetics.wrapper as int_wrap
    import proteus.interior_struct.zalmoxis as zalmoxis_mod
    import proteus.outgas.wrapper as outgas_wrap
    from proteus.config import Config
    from proteus.interior_energetics.common import (
        Interior_t,
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
        resume_baseline_structure_done,
        write_baseline_structure_done,
    )
    from proteus.utils.invariants import check_row_invariants

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    data_dir = tmp_path / 'data'
    data_dir.mkdir(parents=True)
    (data_dir / 'zalmoxis_output.dat').write_text('0.0 0.0\n', encoding='utf-8')
    dirs = {'output': str(tmp_path), 'output/data': str(data_dir)}

    cfg = Config()
    cfg.invariants.mode = 'stop'
    cfg.interior_struct.module = 'zalmoxis'
    cfg.interior_energetics.module = 'aragog'
    cfg.planet.temperature_mode = 'liquidus_super'
    cfg.interior_struct.zalmoxis.mantle_eos = 'PALEOS-2phase:MgSiO3'
    cfg.interior_struct.zalmoxis.dry_mantle = True
    cfg.interior_struct.zalmoxis.equilibrate_max_iter = 1
    cfg.interior_struct.zalmoxis.equilibrate_tol = 0.01

    assert 'M_int_target' in int_wrap._ADIABAT_IC_RESTORE_KEYS
    assert 'M_core_target' in int_wrap._ADIABAT_IC_RESTORE_KEYS

    # 1. _solve_structure_with_adiabat_or_rollback: rollback restores M_int, M_int_target, M_core, and M_core_target without setting _structure_rollback=True
    def failing_adiabat_solver(config, outdir, hf_row, **kwargs):
        hf_row['M_int'] = 6.20e24
        hf_row['M_int_target'] = 6.20e24
        hf_row['M_core'] = 2.50e24
        hf_row['M_core_target'] = 2.50e24
        hf_row['R_int'] = 6.50e6
        raise RuntimeError('simulated adiabat IC non-convergence')

    monkeypatch.setattr(zalmoxis_mod, 'zalmoxis_solver', failing_adiabat_solver)
    ic_row = {
        'R_int': 6.37e6,
        'R_core': 3.48e6,
        'M_int': 5.97e24,
        'M_int_target': 5.97e24,
        'M_core': 1.94e24,
        'M_core_target': 1.94e24,
        'M_mantle': 4.03e24,
        'P_surf': 100.0,
        'P_cmb': 1.35e11,
        'P_center': 3.6e11,
        'gravity': 9.81,
        'rho_avg': 5515.0,
    }
    mesh_rb, ok_rb = int_wrap._solve_structure_with_adiabat_or_rollback(
        cfg, str(tmp_path), ic_row, 0, lambda r, p: 3000.0
    )
    assert mesh_rb is None
    assert ok_rb is False
    assert ic_row['M_int'] == pytest.approx(5.97e24)
    assert ic_row['M_int_target'] == pytest.approx(5.97e24)
    assert ic_row['M_core'] == pytest.approx(1.94e24)
    assert ic_row['M_core_target'] == pytest.approx(1.94e24)
    assert ic_row.get('_structure_rollback', False) is False

    # 2. _solve_structure_with_adiabat_or_rollback: success clears _structure_rollback=False
    def succeeding_adiabat_solver(config, outdir, hf_row, **kwargs):
        hf_row['R_int'] = 6.40e6
        hf_row['M_int'] = 5.97e24
        hf_row['M_int_target'] = 5.97e24
        return 3.48e6, None

    monkeypatch.setattr(zalmoxis_mod, 'zalmoxis_solver', succeeding_adiabat_solver)
    ic_row_ok = dict(ic_row, _structure_rollback=True)
    _, ok_succ = int_wrap._solve_structure_with_adiabat_or_rollback(
        cfg, str(tmp_path), ic_row_ok, 0, lambda r, p: 3000.0
    )
    assert ok_succ is True
    assert ic_row_ok['_structure_rollback'] is False

    # 3. determine_interior_radius_with_zalmoxis clears _structure_rollback=False (including when IC adiabat rolls back)
    monkeypatch.setattr(
        zalmoxis_mod,
        'generate_spider_tables',
        lambda cfg, outdir: {
            'eos_dir': str(tmp_path),
            'solidus_path': str(tmp_path),
            'liquidus_path': str(tmp_path),
        },
    )
    monkeypatch.setattr(
        int_wrap,
        '_build_superliquidus_adiabat_tp',
        lambda cfg, row, pmax: (lambda r, p: 3000.0, np.array([1e5]), np.array([3000.0])),
    )
    call_idx = 0

    def linear_ok_then_adiabat_fail(config, outdir, hf_row, **kwargs):
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            hf_row['R_int'] = 6.37e6
            hf_row['R_core'] = 3.48e6
            hf_row['M_int'] = 5.97e24
            hf_row['M_int_target'] = 5.97e24
            hf_row['M_core'] = 1.94e24
            hf_row['M_core_target'] = 1.94e24
            hf_row['P_cmb'] = 1.35e11
            return 3.48e6, None
        raise RuntimeError('adiabat pass failed')

    monkeypatch.setattr(zalmoxis_mod, 'zalmoxis_solver', linear_ok_then_adiabat_fail)
    det_row = dict(ic_row, _structure_rollback=True)
    int_wrap.determine_interior_radius_with_zalmoxis(
        dirs, cfg, pd.DataFrame(), det_row, str(tmp_path), thermal_solve=False
    )
    assert det_row['_structure_rollback'] is False

    # 4. equilibrate_initial_state resets _structure_rollback=False on both fallback (ok=False) and linear (adiabat_tfunc=None) branches
    monkeypatch.setattr(outgas_wrap, 'calc_target_elemental_inventories', lambda *a, **kw: None)
    monkeypatch.setattr(outgas_wrap, 'run_outgassing', lambda *a, **kw: None)
    monkeypatch.setattr(
        int_wrap,
        '_solve_structure_with_adiabat_or_rollback',
        lambda *a, **kw: (None, False),
    )
    monkeypatch.setattr(zalmoxis_mod, 'zalmoxis_solver', succeeding_adiabat_solver)

    eq_row_fallback = dict(ic_row, _structure_rollback=True)
    int_wrap.equilibrate_initial_state(dirs, cfg, eq_row_fallback, str(tmp_path))
    assert eq_row_fallback['_structure_rollback'] is False

    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda c: False)
    eq_row_linear = dict(ic_row, _structure_rollback=True)
    int_wrap.equilibrate_initial_state(dirs, cfg, eq_row_linear, str(tmp_path))
    assert eq_row_linear['_structure_rollback'] is False
    monkeypatch.undo()

    # 5. resume_structure_stale restores missing-rollback-file fallback from structure_stale.dat,
    # and write_structure_stale writes structure_rollback.dat purely from structure_rollback_active:
    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    Path(get_file_structure_rollback(str(tmp_path))).unlink(missing_ok=True)
    Path(get_file_structure_stale(str(tmp_path))).write_text('1\n', encoding='utf-8')
    missing_rb_stale_reader = Interior_t(5)
    missing_rb_stale_reader.resume_structure_stale(str(tmp_path))
    assert missing_rb_stale_reader.structure_stale is True
    assert missing_rb_stale_reader.structure_rollback_active is True

    Path(get_file_structure_stale(str(tmp_path))).write_text('0\n', encoding='utf-8')
    missing_rb_fresh_reader = Interior_t(5)
    missing_rb_fresh_reader.structure_rollback_active = True
    missing_rb_fresh_reader.resume_structure_stale(str(tmp_path))
    assert missing_rb_fresh_reader.structure_stale is False
    assert missing_rb_fresh_reader.structure_rollback_active is False

    stale_writer = Interior_t(5)
    stale_writer.structure_stale = True
    stale_writer.structure_rollback_active = False
    stale_writer.write_structure_stale(str(tmp_path))
    assert (
        Path(get_file_structure_rollback(str(tmp_path))).read_text(encoding='utf-8').strip()
        == '0'
    )
    stale_reader = Interior_t(5)
    stale_reader.resume_structure_stale(str(tmp_path))
    assert stale_reader.structure_stale is True
    assert stale_reader.structure_rollback_active is False

    # 6. An existing but unparseable rollback file sets structure_rollback_active=True (report-only safe side)
    # even when structure_stale.dat holds 0, and an existing but unparseable baseline-done file returns False:
    Path(get_file_structure_stale(str(tmp_path))).write_text('0\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(tmp_path))).write_text('corrupt\n', encoding='utf-8')
    corrupt_rb_reader = Interior_t(5)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus'):
        corrupt_rb_reader.resume_structure_stale(str(tmp_path))
    assert corrupt_rb_reader.structure_stale is False
    assert corrupt_rb_reader.structure_rollback_active is True
    assert 'Could not parse structure-rollback flag file' in caplog.text

    caplog.clear()
    Path(get_file_baseline_structure_done(str(tmp_path))).write_text(
        'corrupt\n', encoding='utf-8'
    )
    with caplog.at_level(logging.WARNING, logger='fwl.proteus'):
        assert resume_baseline_structure_done(str(tmp_path)) is False
    assert 'Could not parse baseline-structure flag file' in caplog.text

    # 7. Flag files are written atomically via a same-directory temp file and os.replace,
    # so a write or replace error leaves the existing files intact and removes any temp file:
    atomic_writer = Interior_t(5)
    atomic_writer.structure_stale = True
    atomic_writer.structure_rollback_active = True
    atomic_writer.write_structure_stale(str(tmp_path))
    write_baseline_structure_done(str(tmp_path), True)
    for fpath in (
        get_file_structure_stale(str(tmp_path)),
        get_file_structure_rollback(str(tmp_path)),
        get_file_baseline_structure_done(str(tmp_path)),
    ):
        assert Path(fpath).read_text(encoding='utf-8').strip() == '1'

    real_open = builtins.open

    class _FailingWriteHandle:
        def __init__(self, raw_hdl):
            self._raw = raw_hdl

        def __enter__(self):
            self._raw.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._raw.__exit__(exc_type, exc, tb)

        def write(self, _data):
            raise OSError('simulated mid-write failure')

    def open_fails_on_write(file, mode='r', *args, **kwargs):
        hdl = real_open(file, mode, *args, **kwargs)
        if 'w' in mode:
            return _FailingWriteHandle(hdl)
        return hdl

    monkeypatch.setattr(builtins, 'open', open_fails_on_write)
    atomic_writer.structure_stale = False
    atomic_writer.structure_rollback_active = False
    atomic_writer.write_structure_stale(str(tmp_path))
    write_baseline_structure_done(str(tmp_path), False)
    monkeypatch.setattr(builtins, 'open', real_open)

    for fpath in (
        get_file_structure_stale(str(tmp_path)),
        get_file_structure_rollback(str(tmp_path)),
        get_file_baseline_structure_done(str(tmp_path)),
    ):
        assert Path(fpath).read_text(encoding='utf-8').strip() == '1'
    assert [p.name for p in data_dir.iterdir() if '.tmp' in p.name] == []

    def failing_replace(_src, _dst):
        raise OSError('simulated os.replace failure')

    monkeypatch.setattr(os, 'replace', failing_replace)
    atomic_writer.write_structure_stale(str(tmp_path))
    write_baseline_structure_done(str(tmp_path), False)
    monkeypatch.undo()
    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)

    for fpath in (
        get_file_structure_stale(str(tmp_path)),
        get_file_structure_rollback(str(tmp_path)),
        get_file_baseline_structure_done(str(tmp_path)),
    ):
        assert Path(fpath).read_text(encoding='utf-8').strip() == '1'
    assert [p.name for p in data_dir.iterdir() if '.tmp' in p.name] == []

    # 8. Row-index flag persistence and newer-than-resume-row detection vs legacy single-token files:
    row_writer = Interior_t(5)
    row_writer.structure_stale = False
    row_writer.structure_rollback_active = False
    row_writer.write_structure_stale(str(tmp_path), row_index=5)
    write_baseline_structure_done(str(tmp_path), True, row_index=5)
    assert (
        Path(get_file_structure_stale(str(tmp_path))).read_text(encoding='utf-8').strip()
        == '0 5'
    )
    assert (
        Path(get_file_structure_rollback(str(tmp_path))).read_text(encoding='utf-8').strip()
        == '0 5'
    )
    assert (
        Path(get_file_baseline_structure_done(str(tmp_path)))
        .read_text(encoding='utf-8')
        .strip()
        == '1 5'
    )

    normal_resume_reader = Interior_t(5)
    normal_resume_reader.resume_structure_stale(str(tmp_path), resume_row=5)
    assert normal_resume_reader.structure_stale is False
    assert normal_resume_reader.structure_rollback_active is False
    assert resume_baseline_structure_done(str(tmp_path), resume_row=5) is True

    newer_event_reader = Interior_t(5)
    newer_event_reader.resume_structure_stale(str(tmp_path), resume_row=4)
    assert newer_event_reader.structure_rollback_active is True
    assert resume_baseline_structure_done(str(tmp_path), resume_row=4) is False

    # Legacy single-token flag files without a row index keep their recorded values when resume_row is given:
    Path(get_file_structure_stale(str(tmp_path))).write_text('0\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(tmp_path))).write_text('0\n', encoding='utf-8')
    Path(get_file_baseline_structure_done(str(tmp_path))).write_text('1\n', encoding='utf-8')
    legacy_reader = Interior_t(5)
    legacy_reader.resume_structure_stale(str(tmp_path), resume_row=4)
    assert legacy_reader.structure_stale is False
    assert legacy_reader.structure_rollback_active is False
    assert resume_baseline_structure_done(str(tmp_path), resume_row=4) is True

    # Up-step reject through update_structure_from_interior persists structure_rollback_active=True while structure_stale=False:
    def fake_upstep(config, outdir, hf_row, **kwargs):
        hf_row['R_int'] = 6.37e6 * 1.005
        hf_row['M_int_target'] = 5.97e24 * 1.006
        hf_row['M_int'] = 5.97e24 * 1.006
        hf_row['M_core_target'] = 1.94e24 * 1.04
        hf_row['M_core'] = 1.94e24 * 1.04
        return 3.48e6, None

    monkeypatch.setattr(zalmoxis_mod, 'zalmoxis_solver', fake_upstep)
    upstep_int = Interior_t(10)
    upstep_int.ic = 2
    upstep_int.radius = np.linspace(3.48e6, 6.37e6, 11)
    upstep_int.temp = np.linspace(4000.0, 2500.0, 10)
    upstep_row = dict(
        ic_row,
        Time=200.0,
        T_surf=2000.0,
        T_magma=2500.0,
        T_cmb=4000.0,
        Phi_global=0.8,
        M_planet=5.972e24,
        M_atm=1.0e20,
        M_vol_atm=1.0e20,
        M_vaps=0.0,
        H2O_kg_atm=1.0e20,
        M_core_target=1.94e24,
        struct_mass_desync_frac=1.0e-3,
    )
    int_wrap.update_structure_from_interior(
        dirs,
        cfg,
        upstep_row,
        upstep_int,
        last_struct_time=0.0,
        last_Tmagma=2600.0,
        last_Phi=0.85,
        force=True,
    )
    assert upstep_int.structure_stale is False
    assert upstep_int.structure_rollback_active is True

    resumed_after_upstep = Interior_t(10)
    resumed_after_upstep.resume_structure_stale(str(tmp_path))
    assert resumed_after_upstep.structure_stale is False
    assert resumed_after_upstep.structure_rollback_active is True

    # On the resumed run, the schema row (without _structure_rollback) stays report_only=True in stop mode:
    resumed_row = {k: v for k, v in upstep_row.items() if k != '_structure_rollback'}
    v_resumed = check_row_invariants(
        resumed_row, cfg, init_stage=False, interior_o=resumed_after_upstep
    )
    assert [v.name for v in v_resumed] == ['M_int_target_rel_err', 'M_core_target_rel_err']
    assert all(v.report_only for v in v_resumed)
