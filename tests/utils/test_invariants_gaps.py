"""Unit tests addressing invariant assertion gaps and mutant survival edge cases.

Covers multi-family stop-mode orchestration, status file generation and Zalmoxis
flushing, non-zero Q_radio and Q_tidal energy flux terms, multi-row history gross
throughput, non-finite input key checks, configuration tolerance sensitivity and
swaps, zero-M_vaps vapourise stops, and exact inequality boundaries.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from proteus.config._invariants import Invariants
from proteus.utils.invariants import (
    _ENERGY_FINITE_INPUT_KEYS,
    STATUS_ERROR_MASS_INVARIANT,
    STATUS_ERROR_STATE_BOUNDS_INVARIANT,
    InvariantTracker,
    InvariantViolation,
    InvariantViolationError,
    check_energy_invariants,
    check_row_invariants,
    check_state_bounds_invariants,
    check_structure_invariants,
    check_volatile_invariants,
    handle_violations,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


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
    **invariants_kw,
):
    """Build a minimal config object carrying invariants and module options."""
    inv = Invariants(mode=mode, **invariants_kw) if invariants_kw else Invariants(mode=mode)
    return SimpleNamespace(
        invariants=inv,
        params=SimpleNamespace(dt=SimpleNamespace(rtol=dt_rtol, minimum=dt_minimum)),
        planet=SimpleNamespace(fO2_source=fO2_source),
        outgas=SimpleNamespace(
            module=outgas_module,
            vapourise=vapourise,
            solver_rtol=solver_rtol,
            mass_thresh=mass_thresh,
        ),
    )


# ---------------------------------------------------------------------------
# Group 1: Orchestrator Tests
# ---------------------------------------------------------------------------


def test_orchestrator_check_row_invariants_stop_mode_per_family(tmp_path: Path):
    """Verify check_row_invariants in stop mode raises and updates status per family."""
    dirs = {'output': str(tmp_path)}
    cfg_stop = _make_mock_config(mode='stop')

    # 1. Structure family stop (status 31)
    row_struct = {
        'Time': 100.0,
        'struct_mass_desync_frac': 0.05,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'P_surf': 10.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'H2O_kg_atm': 1.0e20,
        'M_vaps': 0.0,
        'Phi_global': 0.5,
    }
    try:
        check_row_invariants(row_struct, cfg_stop, directories=dirs)
        assert False, 'expected InvariantViolationError for structure family'
    except InvariantViolationError as exc_info:
        assert exc_info.status_code == STATUS_ERROR_MASS_INVARIANT
        assert exc_info.invariant_name == 'struct_mass_desync_frac'
    status_file = tmp_path / 'status'
    assert status_file.exists(), 'status file was not written'
    status_lines = status_file.read_text(encoding='utf-8').splitlines()
    assert status_lines[0].strip() == '31'
    assert status_lines[1].strip() == 'Error (Mass conservation invariant)'

    # 2. Volatile family stop (status 31)
    row_vol = {
        'Time': 100.0,
        'M_planet': -1.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'M_vaps': 0.0,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'P_surf': 10.0,
        'Phi_global': 0.5,
    }
    try:
        check_row_invariants(row_vol, cfg_stop, directories=dirs)
        assert False, 'expected InvariantViolationError for volatile family'
    except InvariantViolationError as exc_info:
        assert exc_info.status_code == STATUS_ERROR_MASS_INVARIANT
        assert exc_info.invariant_name == 'M_planet_bounds'
    assert status_file.exists(), 'status file was not written'
    status_lines = status_file.read_text(encoding='utf-8').splitlines()
    assert status_lines[0].strip() == '31'

    # 3. State bounds family stop (status 32)
    row_state = {
        'Time': 100.0,
        'T_surf': -10.0,
        'T_magma': 1500.0,
        'P_surf': 10.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'H2O_kg_atm': 1.0e20,
        'M_vaps': 0.0,
        'Phi_global': 0.5,
    }
    try:
        check_row_invariants(row_state, cfg_stop, directories=dirs)
        assert False, 'expected InvariantViolationError for state bounds family'
    except InvariantViolationError as exc_info:
        assert exc_info.status_code == STATUS_ERROR_STATE_BOUNDS_INVARIANT
        assert exc_info.invariant_name == 'T_surf_bounds'
    assert status_file.exists(), 'status file was not written'
    status_lines = status_file.read_text(encoding='utf-8').splitlines()
    assert status_lines[0].strip() == '32'
    assert status_lines[1].strip() == 'Error (State bounds invariant)'


def test_stop_mode_main_loop_reads_status_and_spies_zalmoxis_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """In stop mode, Proteus main loop writes out/status and executes zalmoxis save."""
    from pathlib import Path

    import proteus.atmos_clim.wrapper as atmos_wrapper
    from proteus import Proteus
    from proteus.interior_energetics.common import Interior_t

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    out = tmp_path / 'run_zalmoxis_spy'
    dummy_toml = Path(__file__).resolve().parents[2] / 'input' / 'dummy.toml'
    text = dummy_toml.read_text(encoding='utf-8')
    cfg_path = tmp_path / 'stop_spy.toml'
    cfg_path.write_text(text.replace('path = "auto"', f'path = "{out}"', 1), encoding='utf-8')

    runner = Proteus(config_path=cfg_path)
    config = runner.config
    config.invariants.mode = 'stop'
    config.atmos_chem.module = None
    config.params.stop.solid.enabled = False
    config.params.stop.iters.minimum = 1
    config.params.stop.time.minimum = 0.0
    config.params.stop.time.maximum = 500.0
    config.params.out.plot_mod = None
    config.params.out.archive_mod = 'none'

    zalmoxis_flushed: list[bool] = []
    real_save_zalmoxis = runner._save_zalmoxis_output

    def spy_save_zalmoxis():
        zalmoxis_flushed.append(True)
        real_save_zalmoxis()

    monkeypatch.setattr(runner, '_save_zalmoxis_output', spy_save_zalmoxis)

    import proteus.atmos_clim

    real_atmos = proteus.atmos_clim.run_atmosphere

    def atmos(*args, **kwargs):
        real_atmos(*args, **kwargs)
        if runner.loops['total'] >= 1:
            runner.hf_row['T_surf'] = -50.0

    monkeypatch.setattr(proteus.atmos_clim, 'run_atmosphere', atmos)
    monkeypatch.setattr(atmos_wrapper, 'run_atmosphere', atmos)

    try:
        runner.start(resume=False, offline=True)
        assert False, 'expected InvariantViolationError was not raised'
    except InvariantViolationError as exc_info:
        assert exc_info.status_code == STATUS_ERROR_STATE_BOUNDS_INVARIANT
    assert len(zalmoxis_flushed) >= 2
    status_file = out / 'status'
    assert status_file.exists()
    status_lines = status_file.read_text(encoding='utf-8').splitlines()
    assert status_lines[0].strip() == '32'
    assert status_lines[1].strip() == 'Error (State bounds invariant)'


# ---------------------------------------------------------------------------
# Group 2: Energy Terms and Flux Components
# ---------------------------------------------------------------------------


def test_energy_invariants_flux_terms_q_radio_and_q_tidal():
    """Verify Q_radio and Q_tidal contribute to dE_inc_cons and gross step flux."""
    cfg = _make_mock_config()
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 1.0e26,
                'step_dE_F_int_J': 1.0e24,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    f_int = 2.0e24
    f_cmb = 1.0e24
    q_rad = 3.0e24
    q_tid = 4.0e24
    total_inc = f_int + f_cmb + q_rad + q_tid  # 1.0e25
    hf_row = {
        'Time': 20.0,
        'T_surf': 2500.0,
        'T_magma': 3000.0,
        'Phi_global': 1.0,
        'RF_depth': 0.0,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 2.0e26,
        'step_dE_state_heat_J': total_inc,
        'step_dE_F_int_J': f_int,
        'step_dE_F_cmb_J': f_cmb,
        'step_dE_Q_radio_J': q_rad,
        'step_dE_Q_tidal_J': q_tid,
        'E_residual_cons_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
    }
    violations = check_energy_invariants(hf_row, cfg, hf_all)
    step_viols = [v for v in violations if v.name == 'step_dE_inc_frac']
    assert step_viols == []
    hf_row_mismatch = dict(hf_row, step_dE_state_heat_J=total_inc - q_rad)
    viols_mismatch = check_energy_invariants(hf_row_mismatch, cfg, hf_all)
    assert any(v.name == 'step_dE_inc_frac' for v in viols_mismatch)


def test_energy_invariants_history_gross_mixed_sign_abs():
    """Verify hist_gross uses absolute values of history fluxes across steps."""
    cfg = _make_mock_config()
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 1.0e26,
                'step_dE_F_int_J': -5.0e26,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            },
            {
                'Time': 20.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 1.0e26,
                'step_dE_F_int_J': 5.0e26,
                'step_dE_F_cmb_J': -1.0e26,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 2.0e26,
            },
        ]
    )
    # Under abs: hist_gross = 5e26 + 5e26 + 1e26 + 2e26 = 1.3e27 J.
    # 3.0e23 / 1.3e27 = 2.308e-4 <= 2.6e-4 (passes).
    hf_row = {
        'Time': 30.0,
        'T_surf': 2500.0,
        'T_magma': 3000.0,
        'Phi_global': 1.0,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 1.0e26,
        'step_dE_state_heat_J': 0.0,
        'step_dE_F_int_J': 0.0,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'E_residual_cons_J': 3.0e23,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
    }
    violations = check_energy_invariants(hf_row, cfg, hf_all)
    cum_viols = [v for v in violations if v.name == 'E_residual_cons_frac_gross']
    assert cum_viols == []
    hf_row_high = dict(hf_row, E_residual_cons_J=5.0e23)
    viols_high = check_energy_invariants(hf_row_high, cfg, hf_all)
    assert any(v.name == 'E_residual_cons_frac_gross' for v in viols_high)


def test_energy_invariants_solver_residual_uses_denom_gross_when_net_heat_small():
    """Verify solver_residual_frac uses denom_gross when |E_state_heat| << gross."""
    cfg = _make_mock_config()
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 1.0e25,
                'step_dE_F_int_J': 1.0e28,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    # denom_gross = 1.0e28 J, while |E_state_heat| = 1.0e25 J.
    # sol_res = 5.0e15 J -> sol_res / denom_gross = 5.0e-13 <= 1.0e-12 (passes).
    hf_row = {
        'Time': 20.0,
        'T_surf': 2500.0,
        'T_magma': 3000.0,
        'Phi_global': 1.0,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 1.0e25,
        'step_dE_state_heat_J': 0.0,
        'step_dE_F_int_J': 0.0,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'E_residual_cons_J': 0.0,
        'solver_residual_J': 5.0e15,
        'core_C_eff': 0.0,
    }
    violations = check_energy_invariants(hf_row, cfg, hf_all)
    sol_viols = [v for v in violations if v.name == 'solver_residual_frac']
    assert sol_viols == []
    hf_row_high = dict(hf_row, solver_residual_J=5.0e16)
    viols_high = check_energy_invariants(hf_row_high, cfg, hf_all)
    assert any(v.name == 'solver_residual_frac' for v in viols_high)


def test_energy_invariants_all_zero_energy_row_and_empty_hf_all():
    """All-zero fluxes and heat do not trigger ZeroDivisionError or IndexError."""
    cfg = _make_mock_config()
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 0.0,
                'step_dE_F_int_J': 0.0,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    hf_row_zeros = {
        'Time': 20.0,
        'T_surf': 2500.0,
        'T_magma': 3000.0,
        'Phi_global': 1.0,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 0.0,
        'step_dE_state_heat_J': 0.0,
        'step_dE_F_int_J': 0.0,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'E_residual_cons_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
    }
    try:
        viols = check_energy_invariants(hf_row_zeros, cfg, hf_all)
    except ZeroDivisionError as exc:
        assert False, f'denom_gross zero division when all heat zero: {exc}'
    assert viols == []

    # Empty hf_all guard
    try:
        viols_empty = check_energy_invariants(hf_row_zeros, cfg, pd.DataFrame())
    except ZeroDivisionError as exc:
        assert False, f'denom_gross zero division on empty hf_all: {exc}'
    assert viols_empty == []

    # Non-empty row with E_state_cons_J == 0.0 returns []
    hf_row_no_e_state = dict(hf_row_zeros, E_state_cons_J=0.0, solver_residual_J=1.0e20)
    assert check_energy_invariants(hf_row_no_e_state, cfg, hf_all) == []


# ---------------------------------------------------------------------------
# Group 3: Non-finite Input Guards
# ---------------------------------------------------------------------------


def test_energy_finite_input_keys_all_nine_keys():
    """Verify each key in _ENERGY_FINITE_INPUT_KEYS reports a non-finite violation."""
    assert len(_ENERGY_FINITE_INPUT_KEYS) == 9
    assert 'step_dE_Q_tidal_J' in _ENERGY_FINITE_INPUT_KEYS
    assert 'E_state_heat_cons_J' in _ENERGY_FINITE_INPUT_KEYS

    cfg = _make_mock_config()
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 1.0e26,
                'step_dE_F_int_J': 1.0e24,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    base_row = {
        'Time': 20.0,
        'T_surf': 2500.0,
        'T_magma': 3000.0,
        'Phi_global': 1.0,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 1.0e26,
        'step_dE_state_heat_J': 1.0e24,
        'step_dE_F_int_J': 1.0e24,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'step_dE_core_J': 0.0,
        'core_C_eff': 0.0,
        'E_residual_cons_J': 0.0,
        'solver_residual_J': 0.0,
    }
    for key in _ENERGY_FINITE_INPUT_KEYS:
        row = dict(base_row, **{key: float('nan')})
        viols = check_energy_invariants(row, cfg, hf_all)
        matching = [v for v in viols if v.name == key]
        assert len(matching) == 1
        assert 'not finite' in matching[0].message


@pytest.mark.parametrize(
    'target_field,bad_val,check_func,expected_name',
    [
        (
            'E_residual_cons_J',
            float('nan'),
            'energy',
            'E_residual_cons_frac_gross',
        ),
        ('solver_residual_J', float('nan'), 'energy', 'solver_residual_frac'),
        ('step_dE_state_heat_J', float('nan'), 'energy', 'step_dE_inc_frac'),
        (
            'E_core_residual_frac',
            float('nan'),
            'energy',
            'E_core_residual_frac',
        ),
        ('M_int', float('nan'), 'structure', 'M_int_target_rel_err'),
        ('M_core', float('nan'), 'structure', 'M_core_target_rel_err'),
        ('M_atm', float('nan'), 'volatile', 'M_atm_bounds'),
        ('esc_kg_cumulative', float('nan'), 'volatile', 'escape_cumulative_bound'),
        ('T_surf', float('nan'), 'state', 'T_surf_bounds'),
        ('T_magma', float('nan'), 'state', 'T_magma_bounds'),
        ('P_surf', float('nan'), 'state', 'P_surf_bounds'),
        ('Phi_global', float('nan'), 'state', 'Phi_global_bounds'),
    ],
)
def test_non_finite_guards_on_individual_checks(
    target_field, bad_val, check_func, expected_name
):
    """Verify non-finite inputs trigger violations across all invariant checks."""
    cfg = _make_mock_config(fO2_source='from_O_budget')
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 1.0e26,
                'step_dE_F_int_J': 1.0e24,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    row = {
        'Time': 20.0,
        'T_surf': 2500.0,
        'T_magma': 3000.0,
        'P_surf': 100.0,
        'Phi_global': 0.5,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 1.0e26,
        'step_dE_state_heat_J': 1.0e24,
        'step_dE_F_int_J': 1.0e24,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'E_residual_cons_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 1.0 if target_field == 'E_core_residual_frac' else 0.0,
        'E_core_residual_frac': 0.0,
        'M_int': 5.972e24,
        'M_int_target': 5.972e24,
        'M_core': 1.940e24,
        'M_core_target': 1.940e24,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'M_vaps': 0.0,
        'M_vol_initial': 1.0e21,
        'esc_kg_cumulative': 1.0e20,
        'H_kg_total': 1.0e21,
    }
    row[target_field] = bad_val
    if check_func == 'energy':
        viols = check_energy_invariants(row, cfg, hf_all)
    elif check_func == 'structure':
        viols = check_structure_invariants(row, cfg)
    elif check_func == 'volatile':
        viols = check_volatile_invariants(row, cfg)
    else:
        viols = check_state_bounds_invariants(row, cfg)

    matching = [v for v in viols if v.name == expected_name]
    assert len(matching) == 1
    assert matching[0].status_code in (30, 31, 32)


# ---------------------------------------------------------------------------
# Group 4: Config Fields, Sensitivity, and Swap Detection
# ---------------------------------------------------------------------------


def test_config_fields_ten_x_flips_outcome():
    """Verify each tolerance field at 10x its default flips the violation outcome."""
    cfg_def = _make_mock_config()

    # 1. phi_global_atol (default 1e-6)
    row_phi = {
        'Time': 10.0,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'P_surf': 10.0,
        'Phi_global': 1.0 + 5.0e-6,
    }
    cfg_10x_phi = _make_mock_config(phi_global_atol=1.0e-5)
    assert any(
        v.name == 'Phi_global_bounds' for v in check_state_bounds_invariants(row_phi, cfg_def)
    )
    assert not any(
        v.name == 'Phi_global_bounds'
        for v in check_state_bounds_invariants(row_phi, cfg_10x_phi)
    )

    # 4. t_magma_max (default 1e6)
    row_tmag = {
        'Time': 10.0,
        'T_surf': 1000.0,
        'T_magma': 5.0e6,
        'P_surf': 10.0,
        'Phi_global': 0.5,
    }
    cfg_10x_tmag = _make_mock_config(t_magma_max=1.0e7)
    assert any(
        v.name == 'T_magma_bounds' for v in check_state_bounds_invariants(row_tmag, cfg_def)
    )
    assert not any(
        v.name == 'T_magma_bounds'
        for v in check_state_bounds_invariants(row_tmag, cfg_10x_tmag)
    )

    # 5. core_mass_target_rtol (default 2.5e-2)
    row_mcore = {
        'Time': 10.0,
        'M_core': 2.0e24 * 1.08,
        'M_core_target': 2.0e24,
    }
    cfg_10x_core = _make_mock_config(core_mass_target_rtol=0.25)
    assert any(
        v.name == 'M_core_target_rel_err'
        for v in check_structure_invariants(row_mcore, cfg_def)
    )
    assert not any(
        v.name == 'M_core_target_rel_err'
        for v in check_structure_invariants(row_mcore, cfg_10x_core)
    )

    # 6. interior_mass_target_rtol (default 3.0e-3)
    row_mint = {
        'Time': 10.0,
        'M_int': 6.0e24 * 1.015,
        'M_int_target': 6.0e24,
    }
    cfg_10x_int = _make_mock_config(interior_mass_target_rtol=0.03)
    assert any(
        v.name == 'M_int_target_rel_err' for v in check_structure_invariants(row_mint, cfg_def)
    )
    assert not any(
        v.name == 'M_int_target_rel_err'
        for v in check_structure_invariants(row_mint, cfg_10x_int)
    )

    # 7. struct_mass_desync_rtol (default 1.0e-2)
    row_desync_stop = {'Time': 10.0, 'struct_mass_desync_frac': 0.03}
    cfg_10x_desync = _make_mock_config(struct_mass_desync_rtol=0.10)
    assert any(
        v.name == 'struct_mass_desync_frac' and not v.report_only
        for v in check_structure_invariants(row_desync_stop, cfg_def)
    )
    assert not any(
        v.name == 'struct_mass_desync_frac' and not v.report_only
        for v in check_structure_invariants(row_desync_stop, cfg_10x_desync)
    )

    # 8. struct_mass_desync_warn_rtol (default 5.0e-3)
    row_desync_warn = {'Time': 10.0, 'struct_mass_desync_frac': 0.008}
    cfg_10x_warn = _make_mock_config(
        struct_mass_desync_warn_rtol=0.05,
        struct_mass_desync_rtol=0.10,
    )
    assert any(
        v.name == 'struct_mass_desync_frac' and v.report_only
        for v in check_structure_invariants(row_desync_warn, cfg_def)
    )
    assert not any(
        v.name == 'struct_mass_desync_frac'
        for v in check_structure_invariants(row_desync_warn, cfg_10x_warn)
    )


def test_config_equal_default_swaps_detected():
    """Verify distinct tolerance fields are wired independently."""
    cfg_def = _make_mock_config(outgas_module='atmodeller', mass_thresh=0.0)
    cfg_phi_asym = _make_mock_config(
        phi_global_atol=1.0e-3,
        element_closure_atol_kg=10.0,
        outgas_module='atmodeller',
        mass_thresh=0.0,
    )
    row_phi = {
        'Time': 10.0,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'P_surf': 10.0,
        'Phi_global': 1.0 + 1.0e-5,
    }
    assert any(
        v.name == 'Phi_global_bounds' for v in check_state_bounds_invariants(row_phi, cfg_def)
    )
    assert not any(
        v.name == 'Phi_global_bounds'
        for v in check_state_bounds_invariants(row_phi, cfg_phi_asym)
    )
    row_atol = {
        'Time': 10.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'M_vaps': 0.0,
        'H_kg_total': 100.0,
        'H_kg_atm': 105.0,
    }
    assert any(
        v.name == 'H_reservoir_closure' for v in check_volatile_invariants(row_atol, cfg_def)
    )
    assert not any(
        v.name == 'H_reservoir_closure'
        for v in check_volatile_invariants(row_atol, cfg_phi_asym)
    )


@pytest.mark.parametrize('bad_mode', ['warn', 'Stop', 'STOP', ''])
def test_config_mode_near_misses_rejected(bad_mode):
    """Verify near-miss mode strings are rejected by Invariants validator."""
    with pytest.raises((ValueError, TypeError)) as exc_info:
        Invariants(mode=bad_mode)
    assert 'mode' in str(exc_info.value)


def test_config_non_negative_fields_accept_zero():
    """Verify non-negative tolerances accept 0.0."""
    try:
        inv = Invariants(
            escape_bound_atol_kg=0.0,
            phi_global_atol=0.0,
        )
    except ValueError as exc:
        assert False, f'non-negative tolerance rejected 0.0: {exc}'
    assert inv.escape_bound_atol_kg == 0.0
    assert inv.phi_global_atol == 0.0


# ---------------------------------------------------------------------------
# Group 5: Edge Cases, Exact Boundaries, and Clean Run Summary
# ---------------------------------------------------------------------------


def test_log_summary_emits_nothing_on_clean_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """InvariantTracker.log_summary emits no records when no violations occurred."""
    tracker = InvariantTracker()
    dirs = {'output': str(tmp_path)}
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        handle_violations([], mode='report', directories=dirs, tracker=tracker)
        handle_violations([], mode='report', directories=dirs, tracker=tracker)
        tracker.log_summary()

    inv_records = [r for r in caplog.records if 'Invariant' in r.message]
    assert inv_records == []
    assert tracker.row_counts == {}


def test_exact_threshold_boundary_rows_and_strict_inequalities(tmp_path):
    """Verify exact-threshold rows behave correctly under strict vs non-strict inequalities."""
    cfg = _make_mock_config()
    eos_dir = tmp_path / 'eos_kink'
    eos_dir.mkdir()
    (eos_dir / 'temperature_solid.dat').write_text(
        FIXTURE_PS_RUN_20261018_131253_EA68, encoding='utf-8'
    )
    dirs = {'spider_eos_dir': str(eos_dir)}
    exact_clip = 302.3416715158806 + 0.5

    # 1. Kink boundary: T_magma == T(1 bar, S_min) + 0.5 K uses step_energy_smin_rtol (0.05)
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 1.0e26,
                'step_dE_F_int_J': 1.0e24,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    row_kink_exact = {
        'Time': 20.0,
        'P_surf': 1.0,
        'T_surf': 2500.0,
        'T_magma': exact_clip,
        'Phi_global': 0.5,
        'M_planet': 6.0e24,
        'M_atm': 0.0,
        'M_vol_atm': 0.0,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 1.0e26,
        'step_dE_state_heat_J': 1.03e24,
        'step_dE_F_int_J': 1.0e24,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'E_residual_cons_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
    }
    viols_kink = check_energy_invariants(row_kink_exact, cfg, hf_all, directories=dirs)
    assert not any(v.name == 'step_dE_inc_frac' for v in viols_kink)
    viols_kink_orch = check_row_invariants(row_kink_exact, cfg, hf_all=hf_all, directories=dirs)
    assert not any(v.name == 'step_dE_inc_frac' for v in viols_kink_orch)

    # 2. Desync == stop_tol (0.01) does not trigger stopping violation
    row_desync_exact = {'Time': 10.0, 'struct_mass_desync_frac': 0.01}
    viols_desync = check_structure_invariants(row_desync_exact, cfg)
    assert not any(
        v.name == 'struct_mass_desync_frac' and not v.report_only for v in viols_desync
    )

    # 3. Desync == warn_tol (0.005) does not trigger warning violation
    row_desync_warn_exact = {'Time': 10.0, 'struct_mass_desync_frac': 0.005}
    viols_desync_warn = check_structure_invariants(row_desync_warn_exact, cfg)
    assert not any(v.name == 'struct_mass_desync_frac' for v in viols_desync_warn)

    # 5. T_magma == t_magma_max (1.0e6) strictly violates (0 < T_magma < t_magma_max)
    row_tmag_exact = {
        'Time': 10.0,
        'T_surf': 1000.0,
        'T_magma': 1.0e6,
        'P_surf': 10.0,
        'Phi_global': 0.5,
    }
    viols_tmag = check_state_bounds_invariants(row_tmag_exact, cfg)
    assert any(v.name == 'T_magma_bounds' for v in viols_tmag)


def test_psurf_bounds_when_matm_le_closure_atol():
    """Verify P_surf bounds behavior when |M_atm| <= element_closure_atol_kg."""
    cfg = _make_mock_config()
    # M_atm = 0.5 kg (<= 1.0 kg), P_surf = 1.0e-6 bar -> valid
    row_pos = {
        'Time': 10.0,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'M_atm': 0.5,
        'P_surf': 1.0e-6,
        'Phi_global': 0.5,
    }
    assert check_state_bounds_invariants(row_pos, cfg) == []

    # P_surf = 0.0 bar -> valid
    row_zero = {
        'Time': 10.0,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'M_atm': 0.5,
        'P_surf': 0.0,
        'Phi_global': 0.5,
    }
    assert check_state_bounds_invariants(row_zero, cfg) == []

    # P_surf = -1.0e-6 bar -> invalid
    row_neg = {
        'Time': 10.0,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'M_atm': 0.5,
        'P_surf': -1.0e-6,
        'Phi_global': 0.5,
    }
    viols_neg = check_state_bounds_invariants(row_neg, cfg)
    assert any(v.name == 'P_surf_bounds' for v in viols_neg)


def test_stop_mode_emits_error_log_line(caplog):
    """Verify that stopping violations emit an error log line (kills T05)."""
    v_stop = InvariantViolation(
        name='test_stop_err',
        status_code=31,
        value=1.0,
        tolerance=0.1,
        message='test error log message',
        report_only=False,
    )
    with (
        caplog.at_level(logging.ERROR, logger='fwl.proteus.utils.invariants'),
        pytest.raises(InvariantViolationError) as exc_info,
    ):
        handle_violations([v_stop], mode='stop')
    assert 'test_stop_err' in str(exc_info.value)
    assert any(
        r.levelno == logging.ERROR and 'Invariant violation [test_stop_err]' in r.getMessage()
        for r in caplog.records
    )


def test_orchestrator_default_mode_is_report_when_unspecified():
    """Verify that when config mode is unspecified, default is report (kills O07)."""
    inv = Invariants()
    d = {a.name: getattr(inv, a.name) for a in inv.__attrs_attrs__ if a.name != 'mode'}
    cfg_no_mode = SimpleNamespace(
        invariants=SimpleNamespace(**d),
        planet=SimpleNamespace(fO2_source='user_constant'),
        outgas=SimpleNamespace(
            module='atmodeller', vapourise=False, solver_rtol=1e-4, mass_thresh=1e10
        ),
    )
    row_viol = {
        'Time': 10.0,
        'T_surf': -100.0,
        'T_magma': 1500.0,
        'Phi_global': 0.5,
        'M_atm': 1.0,
        'P_surf': 1.0,
    }
    # Under report mode, check_row_invariants returns violations without raising.
    try:
        violations = check_row_invariants(row_viol, cfg_no_mode)
    except InvariantViolationError as exc:
        assert False, f'unspecified mode raised InvariantViolationError: {exc}'
    assert any(v.name == 'T_surf_bounds' for v in violations)


def test_energy_invariants_empty_hf_all_and_zero_state_guards():
    """Verify guards on empty hf_all and zero E_state_cons_J (kills E01, E02)."""
    cfg = _make_mock_config()
    row = {'Time': 10.0, 'E_state_cons_J': 1.0e31}
    # Test empty dataframe with columns to detect unindexed iloc calls.
    try:
        viols = check_energy_invariants(
            row, cfg, pd.DataFrame(columns=['Time', 'E_state_heat_cons_J'])
        )
    except IndexError as exc:
        assert False, f'empty hf_all raised IndexError: {exc}'
    assert viols == []

    hf_all = pd.DataFrame([{'Time': 10.0, 'E_state_cons_J': 1.0e31, 'step_dE_F_int_J': 1.0e25}])
    row_zero = {
        'Time': 20.0,
        'E_state_cons_J': 0.0,
        'E_residual_cons_J': 1.0e25,
        'step_dE_F_int_J': 1.0e20,
    }
    assert check_energy_invariants(row_zero, cfg, hf_all) == []


def test_energy_invariants_step_gross_includes_qtidal_and_qradio():
    """Verify Q_tidal and Q_radio contribute to gross throughput (kills E03, E04)."""
    cfg = _make_mock_config()
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 0.0,
                'step_dE_F_int_J': 1.0e20,
            }
        ]
    )
    row_tid = {
        'Time': 20.0,
        'T_surf': 2000.0,
        'T_magma': 2000.0,
        'Phi_global': 0.5,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 0.0,
        'step_dE_state_heat_J': 0.0,
        'step_dE_F_int_J': 0.0,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 1.0e26,
        'E_residual_cons_J': 2.0e22,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
    }
    # Gross throughput is 1e26 from q_tid; rel error 2e22 / 1e26 = 2.0e-4 <= 2.6e-4.
    assert not any(
        v.name == 'E_residual_cons_frac_gross'
        for v in check_energy_invariants(row_tid, cfg, hf_all)
    )

    row_rad = dict(row_tid, step_dE_Q_tidal_J=0.0, step_dE_Q_radio_J=1.0e26)
    assert not any(
        v.name == 'E_residual_cons_frac_gross'
        for v in check_energy_invariants(row_rad, cfg, hf_all)
    )


def test_energy_invariants_prev_heat_zero_and_core_ceff_zero_gates():
    """Verify gates when prev heat is zero or core C_eff is zero (kills E12, E24)."""
    cfg = _make_mock_config()
    hf_zero_heat = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 0.0,
                'step_dE_F_int_J': 1.0e25,
            }
        ]
    )
    row_heat = {
        'Time': 20.0,
        'T_surf': 2000.0,
        'T_magma': 2000.0,
        'Phi_global': 0.5,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 1.0e20,
        'step_dE_state_heat_J': 1.0e25,
        'step_dE_F_int_J': 1.0e20,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'E_residual_cons_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
    }
    assert not any(
        v.name == 'step_dE_inc_frac'
        for v in check_energy_invariants(row_heat, cfg, hf_zero_heat)
    )

    row_no_core = dict(
        row_heat,
        core_C_eff=0.0,
        E_core_residual_frac=1.0,
    )
    assert not any(
        v.name == 'E_core_residual_frac'
        for v in check_energy_invariants(row_no_core, cfg, hf_zero_heat)
    )


def test_exact_threshold_boundary_step_and_cap_and_escape():
    """Verify strict inequality boundaries at exact equality (kills E19, V04, V16)."""
    cfg = _make_mock_config()
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 1.0e26,
            }
        ]
    )
    # E19: step_err == step_tol (1.0e-2)
    step_tol = cfg.invariants.step_energy_rtol
    dE_heat = 1.0e20 * (1.0 + step_tol)
    row_step = {
        'Time': 20.0,
        'T_surf': 2000.0,
        'T_magma': 2000.0,
        'Phi_global': 0.5,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 1.0e26,
        'step_dE_state_heat_J': dE_heat,
        'step_dE_F_int_J': 1.0e20,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'E_residual_cons_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
    }
    assert not any(
        v.name == 'step_dE_inc_frac' for v in check_energy_invariants(row_step, cfg, hf_all)
    )

    # V16: lost == esc_tol
    cfg_esc = _make_mock_config(
        fO2_source='from_O_budget',
        escape_bound_factor=1.5,
        escape_bound_atol_kg=1.0e3,
    )
    esc_cum = 1.0e15
    esc_tol = 1.5 * esc_cum + 1.0e3
    m_init = 2.0e15
    row_v16 = {
        'Time': 10.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'M_vol_initial': m_init,
        'H_kg_total': m_init - esc_tol,
        'esc_kg_cumulative': esc_cum,
    }
    assert not any(
        v.name == 'escape_cumulative_bound' for v in check_volatile_invariants(row_v16, cfg_esc)
    )


def test_structure_invariants_one_sided_errors_and_inf_target():
    """Verify structure relative errors for both signs and inf target (kills S04, S05, S08)."""
    cfg = _make_mock_config()
    # S04: m_int < m_int_target
    row_s04 = {
        'Time': 10.0,
        'M_int': 0.5e24,
        'M_int_target': 1.0e24,
        'M_core': 2.0e24,
        'M_core_target': 2.0e24,
    }
    assert any(
        v.name == 'M_int_target_rel_err' for v in check_structure_invariants(row_s04, cfg)
    )

    # S05: m_core < m_core_target
    row_s05 = {
        'Time': 10.0,
        'M_int': 1.0e24,
        'M_int_target': 1.0e24,
        'M_core': 0.5e24,
        'M_core_target': 1.0e24,
    }
    assert any(
        v.name == 'M_core_target_rel_err' for v in check_structure_invariants(row_s05, cfg)
    )

    # S08: m_int_target = inf
    row_s08 = {'Time': 10.0, 'M_int': 1.0e24, 'M_int_target': float('inf')}
    assert not any(
        v.name == 'M_int_target_rel_err' for v in check_structure_invariants(row_s08, cfg)
    )


def test_volatile_invariants_defaults_and_scaling():
    """Verify volatile defaults for missing fO2_source."""
    cfg_no_fo2 = SimpleNamespace(
        invariants=Invariants(),
        planet=SimpleNamespace(),
        outgas=SimpleNamespace(
            module='atmodeller', vapourise=False, solver_rtol=1e-4, mass_thresh=1e10
        ),
    )
    row_lost = {
        'Time': 10.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'M_vol_initial': 1.0e20,
        'H_kg_total': 0.0,
        'esc_kg_cumulative': 1.0e10,
    }
    assert not any(
        v.name == 'escape_cumulative_bound'
        for v in check_volatile_invariants(row_lost, cfg_no_fo2)
    )
    cfg_from_o = _make_mock_config(fO2_source='from_O_budget')
    assert any(
        v.name == 'escape_cumulative_bound'
        for v in check_volatile_invariants(row_lost, cfg_from_o)
    )


def test_state_bounds_exact_closure_atol_and_missing_key_defaults():
    """Verify state bounds at exact closure atol and missing defaults (kills B03, B05-B08)."""
    cfg = _make_mock_config()
    # B03: exact equality abs(m_atm) == closure_atol_kg (1.0 kg) with P_surf = 0.0
    row_b03 = {
        'Time': 10.0,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'M_atm': 1.0,
        'P_surf': 0.0,
        'Phi_global': 0.5,
    }
    assert not any(
        v.name == 'P_surf_bounds' for v in check_state_bounds_invariants(row_b03, cfg)
    )

    # B05: missing P_surf defaults to nan (triggers violation)
    row_no_p = {
        'Time': 10.0,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'M_atm': 10.0,
        'Phi_global': 0.5,
    }
    assert any(v.name == 'P_surf_bounds' for v in check_state_bounds_invariants(row_no_p, cfg))

    # B06: missing T_surf defaults to nan (triggers violation)
    row_no_t = {
        'Time': 10.0,
        'T_magma': 1500.0,
        'M_atm': 10.0,
        'P_surf': 10.0,
        'Phi_global': 0.5,
    }
    assert any(v.name == 'T_surf_bounds' for v in check_state_bounds_invariants(row_no_t, cfg))

    # B07: missing Phi_global defaults to nan (triggers violation)
    row_no_phi = {
        'Time': 10.0,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'M_atm': 10.0,
        'P_surf': 10.0,
    }
    assert any(
        v.name == 'Phi_global_bounds' for v in check_state_bounds_invariants(row_no_phi, cfg)
    )

    # B08: missing T_magma defaults to nan (triggers violation)
    row_no_tmag = {
        'Time': 10.0,
        'T_surf': 1000.0,
        'M_atm': 10.0,
        'P_surf': 10.0,
        'Phi_global': 0.5,
    }
    assert any(
        v.name == 'T_magma_bounds' for v in check_state_bounds_invariants(row_no_tmag, cfg)
    )


def test_config_threshold_custom_values_override_defaults(tmp_path):
    """Verify that configured threshold overrides prevent false stops (kills W_hardcode_*)."""
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 1.0e26,
                'step_dE_F_int_J': 1.0e26,
            }
        ]
    )
    # Energy residual rtol override
    cfg_e = _make_mock_config(energy_residual_rtol=0.1)
    row_e = {
        'Time': 20.0,
        'P_surf': 1.0,
        'T_surf': 2000.0,
        'T_magma': 2000.0,
        'Phi_global': 0.5,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 1.0e26,
        'step_dE_state_heat_J': 1.0e26,
        'step_dE_F_int_J': 1.0e26,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'E_residual_cons_J': 1.0e25,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
    }
    assert not any(
        v.name == 'E_residual_cons_frac_gross'
        for v in check_energy_invariants(row_e, cfg_e, hf_all)
    )

    # Step energy rtol override
    cfg_step = _make_mock_config(step_energy_rtol=0.1, step_energy_smin_rtol=0.1)
    row_step = dict(row_e, step_dE_state_heat_J=1.03e26, E_residual_cons_J=0.0)
    assert not any(
        v.name == 'step_dE_inc_frac'
        for v in check_energy_invariants(row_step, cfg_step, hf_all)
    )

    # Step energy smin rtol override
    eos_dir = tmp_path / 'eos_smin_override'
    eos_dir.mkdir()
    (eos_dir / 'temperature_solid.dat').write_text(
        FIXTURE_PS_RUN_20261018_131253_EA68, encoding='utf-8'
    )
    cfg_smin = _make_mock_config(step_energy_smin_rtol=0.2)
    row_smin = dict(row_e, T_magma=300.0, step_dE_state_heat_J=1.08e26, E_residual_cons_J=0.0)
    assert not any(
        v.name == 'step_dE_inc_frac'
        for v in check_energy_invariants(
            row_smin, cfg_smin, hf_all, directories={'spider_eos_dir': str(eos_dir)}
        )
    )

    # Solver residual rtol override
    cfg_sol = _make_mock_config(solver_residual_rtol=1.0e-6)
    row_sol = dict(row_e, solver_residual_J=1.0e20, E_residual_cons_J=0.0)
    assert not any(
        v.name == 'solver_residual_frac'
        for v in check_energy_invariants(row_sol, cfg_sol, hf_all)
    )

    # Core residual rtol override
    cfg_core = _make_mock_config(core_residual_rtol=1.0e-2)
    row_core = dict(row_e, core_C_eff=1.0e20, E_core_residual_frac=0.005, E_residual_cons_J=0.0)
    assert not any(
        v.name == 'E_core_residual_frac'
        for v in check_energy_invariants(row_core, cfg_core, hf_all)
    )

    # Element closure rtol override
    cfg_clos = _make_mock_config(element_closure_rtol=0.01, outgas_module='atmodeller')
    row_clos = {
        'Time': 10.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'H_kg_total': 1.0e20,
        'H_kg_atm': 1.001e20,
        'H_kg_liquid': 0.0,
        'H_kg_solid': 0.0,
        'M_vaps': 0.0,
    }
    assert not any(
        v.name == 'H_reservoir_closure' for v in check_volatile_invariants(row_clos, cfg_clos)
    )

    # Element closure atol override
    cfg_atol = _make_mock_config(element_closure_atol_kg=1.0e6, outgas_module='atmodeller')
    row_atol = {
        'Time': 10.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'H_kg_total': 0.0,
        'H_kg_atm': 1.0e4,
        'H_kg_liquid': 0.0,
        'H_kg_solid': 0.0,
        'M_vaps': 0.0,
    }
    assert not any(
        v.name == 'H_reservoir_closure' for v in check_volatile_invariants(row_atol, cfg_atol)
    )

    # Escape bound factor override
    cfg_efac = _make_mock_config(fO2_source='from_O_budget', escape_bound_factor=5.0)
    row_efac = {
        'Time': 10.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'M_vol_initial': 1.0e20,
        'H_kg_total': 1.0e20 - 3.0e15,
        'esc_kg_cumulative': 1.0e15,
    }
    assert not any(
        v.name == 'escape_cumulative_bound'
        for v in check_volatile_invariants(row_efac, cfg_efac)
    )

    # Escape bound atol override
    cfg_eatol = _make_mock_config(
        fO2_source='from_O_budget',
        escape_bound_atol_kg=1.0e6,
    )
    row_eatol = {
        'Time': 10.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'M_vol_initial': 1.0e20,
        'H_kg_total': 1.0e20 - 5.0e5,
        'esc_kg_cumulative': 0.0,
    }
    assert not any(
        v.name == 'escape_cumulative_bound'
        for v in check_volatile_invariants(row_eatol, cfg_eatol)
    )


def test_proteus_main_loop_provides_hf_history_on_second_loop(tmp_path, monkeypatch):
    """Verify that proteus.start() provides hf_all history on loop 2 (kills P12)."""
    import proteus.proteus as proteus_mod
    from proteus import Proteus
    from proteus.interior_energetics.common import Interior_t

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    out = tmp_path / 'run_p12_spy'
    dummy_toml = Path(__file__).resolve().parents[2] / 'input' / 'dummy.toml'
    text = dummy_toml.read_text(encoding='utf-8')
    cfg_path = tmp_path / 'p12_spy.toml'
    cfg_path.write_text(text.replace('path = "auto"', f'path = "{out}"', 1), encoding='utf-8')

    runner = Proteus(config_path=cfg_path)
    config = runner.config
    config.params.stop.solid.enabled = False
    config.params.stop.iters.minimum = 1
    config.params.stop.iters.maximum = 2
    config.params.stop.time.minimum = 0.0
    config.params.stop.time.maximum = 500.0
    config.params.out.plot_mod = None
    config.params.out.archive_mod = 'none'

    history_by_loop = {}
    real_check = proteus_mod.check_row_invariants

    def spy_check(hf_row, cfg, hf_all=None, **kwargs):
        loop = runner.loops['total']
        history_by_loop[loop] = hf_all is not None and len(hf_all) >= 1
        return real_check(hf_row, cfg, hf_all=hf_all, **kwargs)

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_check)
    runner.start(resume=False, offline=True)

    assert history_by_loop.get(1) is False
    assert history_by_loop.get(2) is True


def test_energy_invariants_denom_gross_has_floor_one_when_all_heat_zero():
    """Verify denom_gross has 1.0 floor when all state and flux heat are zero (kills E10)."""
    cfg = _make_mock_config()
    hf_row = {
        'Time': 10.0,
        'T_surf': 2500.0,
        'T_magma': 3000.0,
        'Phi_global': 1.0,
        'E_state_cons_J': 0.0,
        'E_state_heat_cons_J': 0.0,
        'step_dE_state_heat_J': 0.0,
        'step_dE_F_int_J': 0.0,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'step_dE_core_J': 0.0,
        'core_C_eff': 0.0,
        'E_residual_cons_J': 1.0,
        'solver_residual_J': 0.0,
    }
    hf_all = pd.DataFrame([hf_row])
    try:
        viols = check_energy_invariants(hf_row, cfg, hf_all)
    except ZeroDivisionError as exc:
        assert False, f'denom_gross zero division when all heat zero: {exc}'
    assert len(viols) == 0


def test_energy_invariants_e_res_frac_gross_exact_boundary():
    """Verify e_res_frac_gross exact equality at eff_energy_rtol does not violate (kills E20)."""
    cfg = _make_mock_config(energy_residual_rtol=2.6e-4)
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 1.0e26,
                'step_dE_F_int_J': 1.0e26,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    denom = 1.0e26
    tol = 2.6e-4
    hf_row = {
        'Time': 20.0,
        'T_surf': 2500.0,
        'T_magma': 3000.0,
        'Phi_global': 1.0,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 0.0,
        'step_dE_state_heat_J': 0.0,
        'step_dE_F_int_J': 0.0,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'step_dE_core_J': 0.0,
        'core_C_eff': 0.0,
        'E_residual_cons_J': tol * denom,
        'solver_residual_J': 0.0,
    }
    viols = check_energy_invariants(hf_row, cfg, hf_all)
    assert not any(v.name == 'E_residual_cons_frac_gross' for v in viols)
    hf_row_viol = dict(hf_row, E_residual_cons_J=(tol * 1.01) * denom)
    viols_above = check_energy_invariants(hf_row_viol, cfg, hf_all)
    assert any(v.name == 'E_residual_cons_frac_gross' for v in viols_above)


def test_volatile_invariants_total_at_negative_closure_atol_boundary():
    """Verify total equal to -closure_atol_kg does not trigger negative total violation (kills V11)."""
    cfg = _make_mock_config(element_closure_atol_kg=1.0e10, element_closure_rtol=1.0e-4)
    row = {
        'Time': 10.0,
        'M_planet': 6.0e24,
        'M_atm': 0.0,
        'M_vol_atm': 0.0,
        'M_vol_initial': 1.0e20,
        'H_kg_total': -1.0e10,
        'H_kg_atm': -1.0e10,
        'H_kg_liquid': 0.0,
        'H_kg_solid': 0.0,
        'esc_kg_cumulative': 0.0,
    }
    viols = check_volatile_invariants(row, cfg)
    assert not any(v.name == 'H_reservoir_closure' for v in viols)
    row_viol = dict(row, H_kg_total=-1.0001e10, H_kg_atm=-1.0001e10)
    viols_beyond = check_volatile_invariants(row_viol, cfg)
    assert any(v.name == 'H_reservoir_closure' for v in viols_beyond)


def test_handle_violations_stop_mode_does_not_raise_on_report_only():
    """Verify report_only violations in stop mode do not raise (kills T08)."""
    v = InvariantViolation(
        name='E_residual_cons_frac_gross',
        status_code=30,
        value=1.0e-3,
        tolerance=2.6e-4,
        message='test report only',
        report_only=True,
    )
    try:
        res = handle_violations([v], mode='stop')
    except InvariantViolationError as exc:
        assert False, f'stop mode raised on report_only violation: {exc}'
    assert len(res) == 1


def test_handle_violations_report_mode_does_not_raise():
    """Verify report mode never raises InvariantViolationError (kills T09)."""
    v = InvariantViolation(
        name='struct_mass_desync_frac',
        status_code=31,
        value=1.0e-1,
        tolerance=1.0e-2,
        message='test stopping violation',
        report_only=False,
    )
    try:
        res = handle_violations([v], mode='report')
    except InvariantViolationError as exc:
        assert False, f'report mode raised InvariantViolationError: {exc}'
    assert len(res) == 1


def test_orchestrator_check_row_invariants_structure_family_stop():
    """Verify structure family violation stops in stop mode (kills O01)."""
    cfg_stop = _make_mock_config(mode='stop')
    row_bad_struct = {
        'Time': 100.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'struct_mass_desync_frac': 0.05,
        'M_vaps': 0.0,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'P_surf': 10.0,
        'Phi_global': 0.5,
    }
    try:
        check_row_invariants(row_bad_struct, cfg_stop)
        assert False, 'expected InvariantViolationError for structure family'
    except InvariantViolationError as exc:
        assert exc.status_code == 31


def test_orchestrator_check_row_invariants_volatile_family_stop():
    """Verify volatile family violation stops in stop mode (kills O02)."""
    cfg_stop = _make_mock_config(mode='stop')
    row_bad_vol = {
        'Time': 100.0,
        'M_planet': -1.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'M_vaps': 0.0,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'P_surf': 10.0,
        'Phi_global': 0.5,
    }
    try:
        check_row_invariants(row_bad_vol, cfg_stop)
        assert False, 'expected InvariantViolationError for volatile family'
    except InvariantViolationError as exc:
        assert exc.status_code == 31


def test_orchestrator_check_row_invariants_passes_directories_to_status(tmp_path):
    """Verify directories parameter is passed and status written (kills O06)."""
    cfg_stop = _make_mock_config(mode='stop')
    dirs = {'output': str(tmp_path)}
    row_bad_state = {
        'Time': 100.0,
        'T_surf': -10.0,
        'T_magma': 1500.0,
        'P_surf': 0.0,
        'M_planet': 6.0e24,
        'M_atm': 0.0,
        'M_vol_atm': 0.0,
        'Phi_global': 0.5,
    }
    try:
        check_row_invariants(row_bad_state, cfg_stop, directories=dirs)
    except InvariantViolationError:
        pass
    status_file = tmp_path / 'status'
    assert status_file.exists(), 'status file was not written (directories not passed)'
    assert status_file.read_text(encoding='utf-8').splitlines()[0] == '32'


def test_orchestrator_default_mode_does_not_stop():
    """Verify unspecified mode defaults to report and does not raise (kills O07)."""
    cfg = _make_mock_config()
    cfg.invariants = SimpleNamespace(
        **{k: getattr(Invariants(), k) for k in Invariants.__annotations__ if k != 'mode'}
    )
    row_bad = {
        'Time': 10.0,
        'T_surf': -100.0,
        'T_magma': 1500.0,
        'P_surf': 0.0,
        'M_planet': 6.0e24,
        'M_atm': 0.0,
        'M_vol_atm': 0.0,
        'Phi_global': 0.5,
    }
    try:
        viols = check_row_invariants(row_bad, cfg)
    except InvariantViolationError as exc:
        assert False, f'unspecified mode raised InvariantViolationError: {exc}'
    assert len(viols) >= 1


def test_energy_invariants_empty_hf_all_guard_does_not_raise_index_error():
    """Verify empty hf_all does not raise IndexError (kills E01)."""
    cfg = _make_mock_config()
    hf_row = {
        'Time': 10.0,
        'T_surf': 2500.0,
        'T_magma': 3000.0,
        'Phi_global': 1.0,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 1.0e26,
        'step_dE_state_heat_J': 0.0,
        'step_dE_F_int_J': 0.0,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'step_dE_core_J': 0.0,
        'core_C_eff': 0.0,
        'E_residual_cons_J': 0.0,
        'solver_residual_J': 0.0,
    }
    empty_df = pd.DataFrame(columns=['E_state_heat_cons_J', 'step_dE_F_int_J'])
    try:
        viols = check_energy_invariants(hf_row, cfg, empty_df)
    except IndexError as exc:
        assert False, f'empty hf_all raised IndexError: {exc}'
    assert viols == []


def test_proteus_start_reraises_invariant_violation_error(tmp_path, monkeypatch):
    """Verify Proteus.start() re-raises InvariantViolationError on stop (kills P07)."""
    import proteus.proteus as proteus_mod
    from proteus import Proteus
    from proteus.interior_energetics.common import Interior_t

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    out = tmp_path / 'run_p07_spy'
    dummy_toml = Path(__file__).resolve().parents[2] / 'input' / 'dummy.toml'
    text = dummy_toml.read_text(encoding='utf-8')
    cfg_path = tmp_path / 'p07_spy.toml'
    cfg_path.write_text(text.replace('path = "auto"', f'path = "{out}"', 1), encoding='utf-8')

    runner = Proteus(config_path=cfg_path)
    config = runner.config
    config.invariants.mode = 'stop'
    config.params.stop.solid.enabled = False
    config.params.stop.iters.minimum = 1
    config.params.stop.iters.maximum = 2
    config.params.stop.time.minimum = 0.0
    config.params.stop.time.maximum = 500.0
    config.params.out.plot_mod = None
    config.params.out.archive_mod = 'none'

    def failing_check(hf_row, cfg, **kwargs):
        raise InvariantViolationError(
            InvariantViolation(
                name='M_planet_bounds',
                status_code=31,
                value=0.0,
                tolerance=0.0,
                message='synthetic stop',
                report_only=False,
            )
        )

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', failing_check)
    try:
        runner.start(resume=False, offline=True)
        assert False, 'expected InvariantViolationError to be raised'
    except InvariantViolationError:
        pass
    except Exception as exc:
        assert False, f'expected InvariantViolationError but got {type(exc).__name__}: {exc}'


def test_config_validators_assertion_kills():
    """Verify Invariants validators fail with AssertionError when removed (kills C01-C06)."""
    from proteus.config._invariants import Invariants

    # C01: step_energy_smin_rtol >= step_energy_rtol
    try:
        Invariants(step_energy_rtol=0.05, step_energy_smin_rtol=0.01)
        assert False, 'expected ValueError: step_energy_smin_rtol must be >= step_energy_rtol'
    except ValueError:
        pass

    # C02: struct_mass_desync_rtol >= struct_mass_desync_warn_rtol
    try:
        Invariants(struct_mass_desync_warn_rtol=0.05, struct_mass_desync_rtol=0.01)
        assert False, (
            'expected ValueError: struct_mass_desync_rtol must be >= struct_mass_desync_warn_rtol'
        )
    except ValueError:
        pass

    # C03: mode must be in ('off', 'report', 'stop')
    try:
        Invariants(mode='warn')
        assert False, "expected ValueError: mode 'warn' must be rejected"
    except ValueError:
        pass

    # C04: numeric fields reject boolean True/False
    try:
        Invariants(energy_residual_rtol=True)
        assert False, 'expected TypeError/ValueError: boolean True rejected'
    except (TypeError, ValueError):
        pass

    # C05: escape_bound_factor >= 1.0
    try:
        Invariants(escape_bound_factor=0.5)
        assert False, 'expected ValueError: escape_bound_factor < 1.0 rejected'
    except ValueError:
        pass


FIXTURE_PS_RUN_20261018_131253_EA68 = (
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
    '5.190027427724240860e-01 3.860484461686001134e-05 3.037198468914933946e-01\n'
)

FIXTURE_PS_TUTORIAL_EARTH = (
    '# 5 3 2\n'
    '# Pressure [nondim], Entropy [nondim], Quantity [nondim]\n'
    '# column * scaling factor = SI units\n'
    '# scaling factors (constant) for each column given on line below\n'
    '# 1000000000.0 4824266.84604467 1000.0\n'
    '1.000000000000000048e-04 2.199633491194133387e-05 3.232078753841440277e-01\n'
    '2.595513713862120375e-01 2.199633491194133387e-05 3.239502207971426961e-01\n'
    '5.190027427724240860e-01 2.199633491194133387e-05 3.246735911791558959e-01\n'
    '1.000000000000000048e-04 3.620883727604810447e-05 3.232078753841440277e-01\n'
    '2.595513713862120375e-01 3.620883727604810447e-05 3.239502207971426961e-01\n'
    '5.190027427724240860e-01 3.620883727604810447e-05 3.246735911791558959e-01\n'
)

FIXTURE_PS_TUTORIAL_EARTH_MOON = (
    '# 5 3 2\n'
    '# Pressure, Entropy, Quantity\n'
    '# column * scaling factor should be SI units\n'
    '# scaling factors (constant) for each column given on line below\n'
    '# 1000000000.0 4824266.84604467 1.0\n'
    '0.000000000000000000e+00\t-1.757487587703751225e-05\t2.600000000000000000e+02\n'
    '5.000012580964521280e-01\t-1.757487634872620987e-05\t2.611294099185590767e+02\n'
    '9.999984283795089191e-01\t-1.757487513486408613e-05\t2.622503654916838514e+02\n'
    '0.000000000000000000e+00\t-4.306167780593163587e-06\t2.900000000000000000e+02\n'
    '5.000012580964521280e-01\t-4.306167780593163587e-06\t2.912634485376959219e+02\n'
    '9.999984283795089191e-01\t-4.306167780593163587e-06\t2.925174355485420392e+02\n'
)

FIXTURE_PS_LOW_T_SMIN_2599 = (
    '# 5 3 2\n'
    '# Pressure [nondim], Entropy [nondim], Quantity [nondim]\n'
    '# column * scaling factor = SI units\n'
    '# scaling factors (constant) for each column given on line below\n'
    '# 1000000000.0 4824266.84604467 1000.0\n'
    '1.000000000000000048e-04 5.387347119067780726e-06 2.851250000000000000e-01\n'
    '2.595513713862120375e-01 5.387347119067780726e-06 2.858150000000000000e-01\n'
    '5.190027427724240860e-01 5.387347119067780726e-06 2.865050000000000000e-01\n'
    '1.000000000000000048e-04 2.199633491194133387e-05 3.023416715158805945e-01\n'
    '2.595513713862120375e-01 2.199633491194133387e-05 3.030318461324150170e-01\n'
    '5.190027427724240860e-01 2.199633491194133387e-05 3.037198468914933946e-01\n'
)

FIXTURE_REAL_ROWS = {
    'run_20261018_131253_ea68': {
        'Time': 32.0,
        'M_planet': 5.971837965919076e24,
        'M_vaps': 0.0,
        'M_atm': 2.3927000045964503e21,
        'M_vol_atm': 2.3927000045964503e21,
        'M_vol_initial': 0.0,
        'esc_kg_cumulative': 0.0,
        'M_desiccated': 0.0,
        'H2O_kg_atm': 4.818153654011604e17,
        'CO2_kg_atm': 5.48133596233104e20,
        'O2_kg_atm': 1.6293822322839511e21,
        'H2_kg_atm': 3315127877816527.0,
        'CH4_kg_atm': 21571.34812841389,
        'CO_kg_atm': 2.1469904558611625e20,
        'H_kg_atm': 5.723287613365085e16,
        'H_kg_liquid': 7.74427671238664e19,
        'H_kg_total': 7.75e19,
        'O_kg_atm': 2.1509200320879812e21,
        'O_kg_liquid': 6.898789120972752e20,
        'O_kg_total': 2.840798944185256e21,
        'C_kg_atm': 2.416593578040179e20,
        'C_kg_liquid': 3.134064219568384e19,
        'C_kg_total': 2.73e20,
        'Si_kg_atm': 0.0,
        'Si_kg_liquid': 0.0,
        'Si_kg_solid': 0.0,
        'Si_kg_total': 0.0,
        'Mg_kg_atm': 0.0,
        'Mg_kg_liquid': 0.0,
        'Mg_kg_solid': 0.0,
        'Mg_kg_total': 0.0,
        'Fe_kg_atm': 0.0,
        'Fe_kg_liquid': 0.0,
        'Fe_kg_solid': 0.0,
        'Fe_kg_total': 0.0,
        'Na_kg_atm': 0.0,
        'Na_kg_liquid': 0.0,
        'Na_kg_solid': 0.0,
        'Na_kg_total': 0.0,
        'Al_kg_atm': 0.0,
        'Al_kg_liquid': 0.0,
        'Al_kg_solid': 0.0,
        'Al_kg_total': 0.0,
        'Ti_kg_atm': 0.0,
        'Ti_kg_liquid': 0.0,
        'Ti_kg_solid': 0.0,
        'Ti_kg_total': 0.0,
        'Ca_kg_atm': 0.0,
        'Ca_kg_liquid': 0.0,
        'Ca_kg_solid': 0.0,
        'Ca_kg_total': 0.0,
        'K_kg_atm': 0.0,
        'K_kg_liquid': 0.0,
        'K_kg_solid': 0.0,
        'K_kg_total': 0.0,
        'V_kg_atm': 0.0,
        'V_kg_liquid': 0.0,
        'V_kg_solid': 0.0,
        'V_kg_total': 0.0,
    },
    'tutorial_earth': {
        'Time': 12.0,
        'M_planet': 5.971876383413712e24,
        'M_vaps': 0.0,
        'M_atm': 2.4969900423922676e21,
        'M_vol_atm': 2.496990042392267e21,
        'M_vol_initial': 0.0,
        'esc_kg_cumulative': 0.0,
        'M_desiccated': 0.0,
        'H2O_kg_atm': 1.7758460177386754e19,
        'CO2_kg_atm': 5.485865377718051e20,
        'O2_kg_atm': 1.7155219666956764e21,
        'H2_kg_atm': 1.2201093232301552e17,
        'CH4_kg_atm': 28246699.50572676,
        'CO_kg_atm': 2.150010668150476e20,
        'H_kg_atm': 2.1092787717625777e18,
        'H_kg_liquid': 4.678907212282374e20,
        'H_kg_total': 4.6999999999999993e20,
        'O_kg_atm': 2.2529019355724757e21,
        'O_kg_liquid': 3.787871717267085e21,
        'O_kg_total': 6.040773652839561e21,
        'C_kg_atm': 2.419124825228833e20,
        'C_kg_liquid': 3.108751747710883e19,
        'C_kg_total': 2.7299999999999997e20,
        'Si_kg_atm': 0.0,
        'Si_kg_liquid': 0.0,
        'Si_kg_solid': 0.0,
        'Si_kg_total': 0.0,
        'Mg_kg_atm': 0.0,
        'Mg_kg_liquid': 0.0,
        'Mg_kg_solid': 0.0,
        'Mg_kg_total': 0.0,
        'Fe_kg_atm': 0.0,
        'Fe_kg_liquid': 0.0,
        'Fe_kg_solid': 0.0,
        'Fe_kg_total': 0.0,
        'Na_kg_atm': 0.0,
        'Na_kg_liquid': 0.0,
        'Na_kg_solid': 0.0,
        'Na_kg_total': 0.0,
        'Al_kg_atm': 0.0,
        'Al_kg_liquid': 0.0,
        'Al_kg_solid': 0.0,
        'Al_kg_total': 0.0,
        'Ti_kg_atm': 0.0,
        'Ti_kg_liquid': 0.0,
        'Ti_kg_solid': 0.0,
        'Ti_kg_total': 0.0,
        'Ca_kg_atm': 0.0,
        'Ca_kg_liquid': 0.0,
        'Ca_kg_solid': 0.0,
        'Ca_kg_total': 0.0,
        'K_kg_atm': 0.0,
        'K_kg_liquid': 0.0,
        'K_kg_solid': 0.0,
        'K_kg_total': 0.0,
        'V_kg_atm': 0.0,
        'V_kg_liquid': 0.0,
        'V_kg_solid': 0.0,
        'V_kg_total': 0.0,
    },
    'tutorial_earth_moon': {
        'Time': 11.0,
        'M_planet': 5.978726775281167e24,
        'M_vaps': 0.0,
        'M_atm': 2.6028961394114576e21,
        'M_vol_atm': 2.6028961394114576e21,
        'M_vol_initial': 0.0,
        'esc_kg_cumulative': 0.0,
        'M_desiccated': 0.0,
        'H2O_kg_atm': 1.5699182466617856e19,
        'CO2_kg_atm': 1.1374281849304044e21,
        'O2_kg_atm': 1.4252171112474018e20,
        'H2_kg_atm': 1.14425141213347e17,
        'CH4_kg_atm': 454031036.0429061,
        'CO_kg_atm': 4.353173105948544e20,
        'N2_kg_atm': 2.9606306505322195e20,
        'NH3_kg_atm': 31189492346513.355,
        'S2_kg_atm': 1.6825607814574755e17,
        'SO2_kg_atm': 5.755800552380738e20,
        'H2S_kg_atm': 3917594239721642.5,
        'H_kg_atm': 1.8714861935559204e18,
        'H_kg_liquid': 6.181285138064441e20,
        'H_kg_total': 6.2e20,
        'O_kg_atm': 1.5195642056240047e21,
        'O_kg_liquid': 5.206024191223649e21,
        'O_kg_total': 6.725588396847654e21,
        'C_kg_atm': 4.970904052362512e20,
        'C_kg_liquid': 1.229095947637486e20,
        'C_kg_total': 6.2e20,
        'N_kg_atm': 2.9606309070475264e20,
        'N_kg_liquid': 1.3936909295247268e19,
        'N_kg_total': 3.0999999999999993e20,
        'S_kg_atm': 2.8820459210580155e20,
        'S_kg_liquid': 3.3179540789419835e20,
        'S_kg_total': 6.199999999999999e20,
        'Si_kg_atm': 0.0,
        'Si_kg_liquid': 0.0,
        'Si_kg_solid': 0.0,
        'Si_kg_total': 0.0,
        'Mg_kg_atm': 0.0,
        'Mg_kg_liquid': 0.0,
        'Mg_kg_solid': 0.0,
        'Mg_kg_total': 0.0,
        'Fe_kg_atm': 0.0,
        'Fe_kg_liquid': 0.0,
        'Fe_kg_solid': 0.0,
        'Fe_kg_total': 0.0,
        'Na_kg_atm': 0.0,
        'Na_kg_liquid': 0.0,
        'Na_kg_solid': 0.0,
        'Na_kg_total': 0.0,
        'Al_kg_atm': 0.0,
        'Al_kg_liquid': 0.0,
        'Al_kg_solid': 0.0,
        'Al_kg_total': 0.0,
        'Ti_kg_atm': 0.0,
        'Ti_kg_liquid': 0.0,
        'Ti_kg_solid': 0.0,
        'Ti_kg_total': 0.0,
        'Ca_kg_atm': 0.0,
        'Ca_kg_liquid': 0.0,
        'Ca_kg_solid': 0.0,
        'Ca_kg_total': 0.0,
        'K_kg_atm': 0.0,
        'K_kg_liquid': 0.0,
        'K_kg_solid': 0.0,
        'K_kg_total': 0.0,
        'V_kg_atm': 0.0,
        'V_kg_liquid': 0.0,
        'V_kg_solid': 0.0,
        'V_kg_total': 0.0,
    },
    'run_20261018_072515_c63d': {
        'Time': 32.0,
        'M_planet': 5.971896581983356e24,
        'M_vaps': 0.0,
        'M_atm': 2.479770912985214e21,
        'M_vol_atm': 2.479770912985214e21,
        'M_vol_initial': 0.0,
        'esc_kg_cumulative': 0.0,
        'M_desiccated': 0.0,
        'H2O_kg_atm': 1.7376303638004652e19,
        'CO2_kg_atm': 5.485809317108935e20,
        'O2_kg_atm': 1.6987224597552077e21,
        'H2_kg_atm': 1.1942269468348742e17,
        'CH4_kg_atm': 27191898.159797955,
        'CO_kg_atm': 2.149717951863974e20,
        'H_kg_atm': 2.0639251505691924e18,
        'H_kg_liquid': 4.6293607484943073e20,
        'H_kg_total': 4.65e20,
        'O_kg_atm': 2.2357427726804543e21,
        'O_kg_liquid': 3.748584751416915e21,
        'O_kg_total': 5.984327524097369e21,
        'C_kg_atm': 2.4189840054296828e20,
        'C_kg_liquid': 3.1101599457031594e19,
        'C_kg_total': 2.7299999999999997e20,
        'Si_kg_atm': 0.0,
        'Si_kg_liquid': 0.0,
        'Si_kg_solid': 0.0,
        'Si_kg_total': 0.0,
        'Mg_kg_atm': 0.0,
        'Mg_kg_liquid': 0.0,
        'Mg_kg_solid': 0.0,
        'Mg_kg_total': 0.0,
        'Fe_kg_atm': 0.0,
        'Fe_kg_liquid': 0.0,
        'Fe_kg_solid': 0.0,
        'Fe_kg_total': 0.0,
        'Na_kg_atm': 0.0,
        'Na_kg_liquid': 0.0,
        'Na_kg_solid': 0.0,
        'Na_kg_total': 0.0,
        'Al_kg_atm': 0.0,
        'Al_kg_liquid': 0.0,
        'Al_kg_solid': 0.0,
        'Al_kg_total': 0.0,
        'Ti_kg_atm': 0.0,
        'Ti_kg_liquid': 0.0,
        'Ti_kg_solid': 0.0,
        'Ti_kg_total': 0.0,
        'Ca_kg_atm': 0.0,
        'Ca_kg_liquid': 0.0,
        'Ca_kg_solid': 0.0,
        'Ca_kg_total': 0.0,
        'K_kg_atm': 0.0,
        'K_kg_liquid': 0.0,
        'K_kg_solid': 0.0,
        'K_kg_total': 0.0,
        'V_kg_atm': 0.0,
        'V_kg_liquid': 0.0,
        'V_kg_solid': 0.0,
        'V_kg_total': 0.0,
    },
}


def test_smin_clip_per_run_ps_table_interpolation_and_caching(tmp_path, monkeypatch, caplog):
    """Verify per-row T(P_surf, S_min) + 0.5 K clip from cached P-S tables and no-table fallback.

    The 0.5 K margin (EOS_SMIN_CLIP_MARGIN_K) bounds the smooth low-pressure P-S linear
    interpolation error (<= 9.488e-03 K) and static-mesh vs row P_surf offset (-0.004201 K
    to +0.071245 K across S_min-clipped corpus rows; < 0.5 K for |P_mesh - P_surf| <= 1,748 bar),
    remaining well below the first unclipped entropy-grid step (+22.74 K to +30.00 K).
    """
    import proteus.utils.invariants as inv_mod

    assert inv_mod.EOS_SMIN_CLIP_MARGIN_K == pytest.approx(0.5)

    cfg = _make_mock_config()
    hf_all = pd.DataFrame(
        [
            {
                'Time': 10.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': 1.0e26,
                'step_dE_F_int_J': 1.0e24,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    base_row = {
        'Time': 20.0,
        'P_surf': 1.0,
        'T_surf': 2500.0,
        'Phi_global': 0.5,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': 1.0e26,
        'step_dE_state_heat_J': 1.03e24,
        'step_dE_F_int_J': 1.0e24,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'E_residual_cons_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
    }

    cases = [
        ('ea68', FIXTURE_PS_RUN_20261018_131253_EA68, 1.0, 302.3416715158806 + 0.5),
        ('tutorial_earth', FIXTURE_PS_TUTORIAL_EARTH, 1.0, 323.20787538414403 + 0.5),
        ('wb_moon', FIXTURE_PS_TUTORIAL_EARTH_MOON, 0.0, 260.0 + 0.5),
        ('low_t_2599', FIXTURE_PS_LOW_T_SMIN_2599, 1.0, 285.125 + 0.5),
    ]
    for label, table_text, p_surf_bar, clip_k in cases:
        eos_dir = tmp_path / f'eos_{label}'
        eos_dir.mkdir()
        (eos_dir / 'temperature_solid.dat').write_text(table_text, encoding='utf-8')
        dirs = {'spider_eos_dir': str(eos_dir)}

        v_below = check_energy_invariants(
            dict(base_row, P_surf=p_surf_bar, T_magma=clip_k - 1.0e-6),
            cfg,
            hf_all,
            directories=dirs,
        )
        assert not any(v.name == 'step_dE_inc_frac' for v in v_below)

        v_exact = check_energy_invariants(
            dict(base_row, P_surf=p_surf_bar, T_magma=clip_k),
            cfg,
            hf_all,
            directories=dirs,
        )
        assert not any(v.name == 'step_dE_inc_frac' for v in v_exact)

        v_above = check_energy_invariants(
            dict(base_row, P_surf=p_surf_bar, T_magma=clip_k + 1.0e-6),
            cfg,
            hf_all,
            directories=dirs,
        )
        step_viols = [v for v in v_above if v.name == 'step_dE_inc_frac']
        assert len(step_viols) == 1
        assert step_viols[0].tolerance == pytest.approx(0.01)

    # Evaluate T(P_surf, S_min) per row at the row's P_surf:
    dirs_ea68 = {'spider_eos_dir': str(tmp_path / 'eos_ea68')}
    # At P_surf = 1.0 bar, clip = 302.841672 K, so T_magma = 303.2 K violates (> clip).
    v_low_p = check_energy_invariants(
        dict(base_row, P_surf=1.0, T_magma=303.2), cfg, hf_all, directories=dirs_ea68
    )
    assert any(v.name == 'step_dE_inc_frac' for v in v_low_p)
    # At P_surf = 2595.513714 bar (2.595514e8 Pa), T(P_surf, S_min) = 303.031846 K and clip = 303.531846 K.
    v_high_p = check_energy_invariants(
        dict(base_row, P_surf=2595.5137138621204, T_magma=303.2),
        cfg,
        hf_all,
        directories=dirs_ea68,
    )
    assert not any(v.name == 'step_dE_inc_frac' for v in v_high_p)

    # Clamping P_surf below P_min (0 bar -> 1e5 Pa) and above P_max (1e4 bar -> 5.190027e8 Pa), and non-finite P_surf:
    assert inv_mod._smin_clip_temp_k({'P_surf': 0.0}, dirs_ea68) == pytest.approx(
        302.3416715158806 + 0.5
    )
    assert inv_mod._smin_clip_temp_k({'P_surf': 1.0e4}, dirs_ea68) == pytest.approx(
        303.7198468914934 + 0.5
    )
    assert inv_mod._smin_clip_temp_k({'P_surf': float('nan')}, dirs_ea68) is None
    assert inv_mod._smin_clip_temp_k({'P_surf': float('inf')}, dirs_ea68) is None

    # No-table case (directories=None or missing file): clip rule does not apply.
    v_no_table = check_energy_invariants(
        dict(base_row, T_magma=250.0), cfg, hf_all, directories=None
    )
    assert any(v.name == 'step_dE_inc_frac' for v in v_no_table)
    v_missing_dir = check_energy_invariants(
        dict(base_row, T_magma=250.0),
        cfg,
        hf_all,
        directories={'spider_eos_dir': str(tmp_path / 'nonexistent_eos')},
    )
    assert any(v.name == 'step_dE_inc_frac' for v in v_missing_dir)

    # Caching keyed by (path, head_bytes): reading two rows on the same file parses the file once.
    inv_mod._SMIN_TABLE_CACHE.clear()
    inv_mod._SMIN_TABLE_WARNED_PATHS.clear()
    parse_calls = []
    real_parse = inv_mod._parse_smin_temp_block

    def counting_parse(path):
        parse_calls.append(str(path))
        return real_parse(path)

    monkeypatch.setattr(inv_mod, '_parse_smin_temp_block', counting_parse)
    check_energy_invariants(
        dict(base_row, P_surf=1.0, T_magma=302.5), cfg, hf_all, directories=dirs_ea68
    )
    check_energy_invariants(
        dict(base_row, P_surf=100.0, T_magma=302.5), cfg, hf_all, directories=dirs_ea68
    )
    assert len(parse_calls) == 1

    # Fallback via output/data/spider_eos_cache.txt pointer:
    out_data = tmp_path / 'out_data'
    out_data.mkdir()
    (out_data / 'spider_eos_cache.txt').write_text(
        str(tmp_path / 'eos_tutorial_earth') + '\n', encoding='utf-8'
    )
    assert inv_mod._smin_clip_temp_k(
        {'P_surf': 1.0}, {'output/data': str(out_data)}
    ) == pytest.approx(323.20787538414403 + 0.5)
    assert len(parse_calls) == 2
    monkeypatch.setattr(inv_mod, '_parse_smin_temp_block', real_parse)

    # Resolver fallbacks with {'output/data': str(d)} only and {'output': str(run)} only:
    direct_data = tmp_path / 'direct_out_data'
    (direct_data / 'spider_eos').mkdir(parents=True)
    (direct_data / 'spider_eos' / 'temperature_solid.dat').write_text(
        FIXTURE_PS_RUN_20261018_131253_EA68, encoding='utf-8'
    )
    assert inv_mod._smin_clip_temp_k(
        {'P_surf': 1.0}, {'output/data': str(direct_data)}
    ) == pytest.approx(302.3416715158806 + 0.5)

    run_root = tmp_path / 'run_root_only'
    (run_root / 'data' / 'spider_eos').mkdir(parents=True)
    (run_root / 'data' / 'spider_eos' / 'temperature_solid.dat').write_text(
        FIXTURE_PS_TUTORIAL_EARTH_MOON, encoding='utf-8'
    )
    assert inv_mod._smin_clip_temp_k(
        {'P_surf': 0.0}, {'output': str(run_root)}
    ) == pytest.approx(260.0 + 0.5)

    # In-place rewrite with a new table and the same mtime (os.utime) returns the new clip:
    rewrite_eos = tmp_path / 'eos_rewrite'
    rewrite_eos.mkdir()
    rewrite_file = rewrite_eos / 'temperature_solid.dat'
    rewrite_file.write_text(FIXTURE_PS_RUN_20261018_131253_EA68, encoding='utf-8')
    orig_stat = rewrite_file.stat()
    assert inv_mod._smin_clip_temp_k(
        {'P_surf': 1.0}, {'spider_eos_dir': str(rewrite_eos)}
    ) == pytest.approx(302.3416715158806 + 0.5)
    rewrite_file.write_text(FIXTURE_PS_TUTORIAL_EARTH, encoding='utf-8')
    os.utime(rewrite_file, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))
    assert inv_mod._read_smin_block_bytes(rewrite_file) == ''.join(
        FIXTURE_PS_TUTORIAL_EARTH.splitlines(keepends=True)[:8]
    ).encode('utf-8')
    assert inv_mod._smin_clip_temp_k(
        {'P_surf': 1.0}, {'spider_eos_dir': str(rewrite_eos)}
    ) == pytest.approx(323.20787538414403 + 0.5)
    # Rewriting only the last (n_P-th) row of the S_min block with the same mtime also invalidates cache:
    te_modified_last_row = FIXTURE_PS_TUTORIAL_EARTH.replace(
        '3.246735911791558959e-01', '3.500000000000000000e-01', 1
    )
    rewrite_file.write_text(te_modified_last_row, encoding='utf-8')
    os.utime(rewrite_file, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))
    assert inv_mod._smin_clip_temp_k(
        {'P_surf': 1.0e4}, {'spider_eos_dir': str(rewrite_eos)}
    ) == pytest.approx(350.0 + 0.5)

    # 2 paths with the same mtime (os.utime) return their own clips:
    same_mtime_a = tmp_path / 'eos_same_mtime_a'
    same_mtime_b = tmp_path / 'eos_same_mtime_b'
    same_mtime_a.mkdir()
    same_mtime_b.mkdir()
    file_a = same_mtime_a / 'temperature_solid.dat'
    file_b = same_mtime_b / 'temperature_solid.dat'
    file_a.write_text(FIXTURE_PS_RUN_20261018_131253_EA68, encoding='utf-8')
    file_b.write_text(FIXTURE_PS_TUTORIAL_EARTH, encoding='utf-8')
    stat_a = file_a.stat()
    os.utime(file_b, ns=(stat_a.st_atime_ns, stat_a.st_mtime_ns))
    assert inv_mod._smin_clip_temp_k(
        {'P_surf': 1.0}, {'spider_eos_dir': str(same_mtime_a)}
    ) == pytest.approx(302.3416715158806 + 0.5)
    assert inv_mod._smin_clip_temp_k(
        {'P_surf': 1.0}, {'spider_eos_dir': str(same_mtime_b)}
    ) == pytest.approx(323.20787538414403 + 0.5)

    # First block must hold S_min = min(S) across the Wolf & Bower table (+64.0 J/(kg K) block spacing):
    bad_eos = tmp_path / 'eos_bad_order'
    bad_eos.mkdir()
    wb_lines = FIXTURE_PS_TUTORIAL_EARTH_MOON.splitlines(keepends=True)
    swapped_wb_text = ''.join(wb_lines[:5] + wb_lines[8:11] + wb_lines[5:8])
    bad_file = bad_eos / 'temperature_solid.dat'
    bad_file.write_text(swapped_wb_text, encoding='utf-8')
    with pytest.raises(ValueError, match='S_min'):
        inv_mod._parse_smin_temp_block(bad_file)

    # In report mode, a malformed/swapped table gives no clip (None), logs one warning per path, and does not raise:
    caplog.clear()
    row_report = dict(
        base_row,
        T_magma=250.0,
        M_planet=6.0e24,
        M_atm=1.0e20,
        M_vol_atm=1.0e20,
        H2O_kg_atm=1.0e20,
        M_vaps=0.0,
    )
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        try:
            assert (
                inv_mod._smin_clip_temp_k({'P_surf': 1.0}, {'spider_eos_dir': str(bad_eos)})
                is None
            )
            viols_bad_1 = check_row_invariants(
                row_report, cfg, hf_all, directories={'spider_eos_dir': str(bad_eos)}
            )
            viols_bad_2 = check_row_invariants(
                row_report, cfg, hf_all, directories={'spider_eos_dir': str(bad_eos)}
            )
        except ValueError as exc:
            pytest.fail(f'unexpected ValueError on bad table: {exc}')
    step_bad = [v for v in viols_bad_1 if v.name == 'step_dE_inc_frac']
    assert len(step_bad) == 1
    assert step_bad[0].tolerance == pytest.approx(0.01)
    assert len([v for v in viols_bad_2 if v.name == 'step_dE_inc_frac']) == 1
    table_warns = [r for r in caplog.records if str(bad_file.resolve()) in r.message]
    assert len(table_warns) == 1

    # Second path raising OSError also falls back to no clip and logs one warning for its own path:
    os_err_eos = tmp_path / 'eos_os_error'
    os_err_eos.mkdir()
    os_err_file = os_err_eos / 'temperature_solid.dat'
    os_err_file.write_text(FIXTURE_PS_RUN_20261018_131253_EA68, encoding='utf-8')
    real_read_head = inv_mod._read_smin_block_bytes

    def raising_oserror(path):
        if Path(path).resolve() == os_err_file.resolve():
            raise OSError('simulated table I/O failure')
        return real_read_head(path)

    monkeypatch.setattr(inv_mod, '_read_smin_block_bytes', raising_oserror)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        try:
            assert (
                inv_mod._smin_clip_temp_k({'P_surf': 1.0}, {'spider_eos_dir': str(os_err_eos)})
                is None
            )
            viols_os_1 = check_row_invariants(
                row_report, cfg, hf_all, directories={'spider_eos_dir': str(os_err_eos)}
            )
            viols_os_2 = check_row_invariants(
                row_report, cfg, hf_all, directories={'spider_eos_dir': str(os_err_eos)}
            )
        except OSError as exc:
            pytest.fail(f'unexpected OSError on unreadable table: {exc}')
    step_os = [v for v in viols_os_1 if v.name == 'step_dE_inc_frac']
    assert len(step_os) == 1
    assert step_os[0].tolerance == pytest.approx(0.01)
    assert len([v for v in viols_os_2 if v.name == 'step_dE_inc_frac']) == 1
    os_warns = [r for r in caplog.records if str(os_err_file.resolve()) in r.message]
    assert len(os_warns) == 1


def test_lavatmos_overwritten_elements_match_source_and_skip_under_vapourise():
    """Verify LAVATMOS_OVERWRITTEN_ELEMENTS matches lavatmos.py input_eles and skips closure under vapourise."""
    import ast

    from proteus.utils import invariants as inv_mod
    from proteus.utils.constants import noble_gases, vap_element_list, vol_element_list

    lavatmos_path = Path(inv_mod.__file__).resolve().parents[1] / 'outgas' / 'lavatmos.py'
    tree = ast.parse(lavatmos_path.read_text(encoding='utf-8'))
    found_input_eles = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'input_eles':
                    found_input_eles = node.value
                    break
    assert found_input_eles is not None
    # In lavatmos.py: input_eles = ['H', 'C', 'N', 'S', 'O'] + noble_gases
    assert isinstance(found_input_eles, ast.BinOp)
    assert isinstance(found_input_eles.right, ast.Name)
    assert found_input_eles.right.id == 'noble_gases'
    left_elts = [elt.value for elt in found_input_eles.left.elts]
    assert set(left_elts) == set(vol_element_list)
    assert set(inv_mod.LAVATMOS_OVERWRITTEN_ELEMENTS) == set(left_elts) | set(noble_gases)

    # Post-vapourise row: lavatmos.py:567-586 keeps O_kg_atm and overwrites H/C/N/S/noble_gases after line 555:
    cfg_vap_stop = _make_mock_config(mode='stop', fO2_source='from_O_budget', vapourise=True)
    row_vap = {
        'Time': 100.0,
        'P_surf': 100.0,
        'T_surf': 2500.0,
        'T_magma': 2500.0,
        'Phi_global': 0.8,
        'M_planet': 6.0e24,
        'M_vaps': 3.0e19,
        'M_vol_atm': 5.2e19,
        'M_atm': 8.2e19,
        'H2_kg_atm': 5.0e19,
        'Ar_kg_atm': 2.0e18,
        'SiO_kg_atm': 3.0e19,
        'H_kg_atm': 5.0e19,
        'H_kg_liquid': 0.0,
        'H_kg_solid': 0.0,
        'H_kg_total': 1.0e19,
        'Ar_kg_liquid': 0.0,
        'Ar_kg_solid': 0.0,
        'Ar_kg_total': 0.0,
        'Si_kg_atm': 3.0e19,
        'Si_kg_liquid': 0.0,
        'Si_kg_solid': 0.0,
        'Si_kg_total': 0.0,
    }
    assert check_volatile_invariants(row_vap, cfg_vap_stop) == []
    for elem in set(vap_element_list) | set(inv_mod.LAVATMOS_OVERWRITTEN_ELEMENTS):
        assert (
            check_volatile_invariants(
                dict(row_vap, **{f'{elem}_kg_total': 1.0e20}), cfg_vap_stop
            )
            == []
        )

    # Reservoir bounds checks ({e}_kg_{r}_bounds) remain active for overwritten elements:
    try:
        check_row_invariants(dict(row_vap, H_kg_atm=-5.0), cfg_vap_stop)
        assert False, 'expected InvariantViolationError on negative reservoir mass'
    except InvariantViolationError as exc_info:
        assert any(v.name == 'H_kg_atm_bounds' for v in exc_info.violations)


@pytest.mark.parametrize('vapourise', [False, True])
@pytest.mark.parametrize(
    ('bad_key', 'bad_val', 'expected_name'),
    [
        ('Si_kg_atm', float('nan'), 'Si_reservoir_closure'),
        ('Si_kg_atm', float('-inf'), 'Si_reservoir_closure'),
        ('Si_kg_total', float('nan'), 'Si_reservoir_closure'),
        ('Si_kg_total', float('inf'), 'Si_reservoir_closure'),
        ('Si_kg_total', float('-inf'), 'Si_reservoir_closure'),
        ('O_kg_liquid', float('nan'), 'O_reservoir_closure'),
    ],
)
def test_non_finite_excluded_element_masses_emit_status_31(
    vapourise, bad_key, bad_val, expected_name
):
    """Non-finite {e}_kg_{atm,liquid,solid,total} of any element in element_list is a status 31 violation before the derived skip."""
    cfg = _make_mock_config(mode='stop', fO2_source='user_constant', vapourise=vapourise)
    base_row = {
        'Time': 50.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e20,
        bad_key: bad_val,
    }
    viols = check_volatile_invariants(base_row, cfg)
    closure_viols = [v for v in viols if v.name == expected_name]
    assert len(closure_viols) == 1
    assert closure_viols[0].status_code == STATUS_ERROR_MASS_INVARIANT
    assert closure_viols[0].report_only is False
    if bad_val != bad_val:
        assert closure_viols[0].value != closure_viols[0].value
    else:
        assert closure_viols[0].value == bad_val


def test_rock_vapour_exclusion_gated_on_vapourise_and_real_rows_pass():
    """Verify rock-vapour exclusion applies only under outgas.vapourise and real-row false-stop replay passes."""
    from proteus.utils.constants import vap_element_list

    cfg_stop = _make_mock_config(
        mode='stop', fO2_source='user_constant', vapourise=False, outgas_module='calliope'
    )
    base_row = {
        'Time': 100.0,
        'P_surf': 100.0,
        'T_surf': 2000.0,
        'T_magma': 2000.0,
        'Phi_global': 0.5,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'M_vaps': 0.0,
        'H2O_kg_atm': 1.0e20,
    }

    # 1. Vapourise off, Si_kg_atm = 1e20, total = 0 -> stop mode stops with Si_reservoir_closure:
    with pytest.raises(InvariantViolationError) as exc_atm:
        check_row_invariants(dict(base_row, Si_kg_atm=1.0e20, Si_kg_total=0.0), cfg_stop)
    assert [v.name for v in exc_atm.value.violations] == ['Si_reservoir_closure']
    assert exc_atm.value.status_code == STATUS_ERROR_MASS_INVARIANT

    # 2. Vapourise off, rock mass in liquid or solid while atm = 0 and total = 0 -> closure fires:
    viols_liq = check_volatile_invariants(
        dict(base_row, Si_kg_liquid=1.0e18, Si_kg_atm=0.0, Si_kg_total=0.0), cfg_stop
    )
    assert [v.name for v in viols_liq] == ['Si_reservoir_closure']
    viols_sol = check_volatile_invariants(
        dict(base_row, Mg_kg_solid=1.0e18, Mg_kg_atm=0.0, Mg_kg_total=0.0), cfg_stop
    )
    assert [v.name for v in viols_sol] == ['Mg_reservoir_closure']

    # 3. Real-row false-stop replay across 1 script-copied row from each of the 4 real non-vapourise runs:
    rock_cols = [
        f'{e}_kg_{r}' for e in vap_element_list for r in ('atm', 'liquid', 'solid', 'total')
    ]
    assert len(rock_cols) == 36
    assert len(FIXTURE_REAL_ROWS) == 4
    for run_name, row_dict in FIXTURE_REAL_ROWS.items():
        assert all(row_dict[col] == 0.0 for col in rock_cols), run_name
        viols_real = check_volatile_invariants(row_dict, cfg_stop)
        assert viols_real == [], f'{run_name}: {viols_real}'


def test_tracker_relogs_finite_to_nonfinite_and_nan_inf_transitions(caplog):
    """Tracker re-logs transitions between finite and non-finite values and between NaN and +-inf."""
    tracker = InvariantTracker()
    seq = [3.0e-4, float('nan'), float('nan'), 5.0e-4]

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        for val in seq:
            v = InvariantViolation(
                name='E_residual_cons_frac_gross',
                status_code=30,
                value=val,
                tolerance=2.0e-4,
                message=f'E_residual_cons_frac_gross={val:.6e}',
                report_only=False,
            )
            handle_violations([v], mode='report', tracker=tracker)

    records = [r for r in caplog.records if 'E_residual_cons_frac_gross' in r.message]
    assert len(records) == 2
    assert '3.000000e-04' in records[0].message
    assert 'nan' in records[1].message.lower()

    # Verify NaN -> +inf -> +inf -> -inf transitions also re-log on change and suppress on exact repeat:
    caplog.clear()
    nonfinite_seq = [float('nan'), float('inf'), float('inf'), float('-inf')]
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.utils.invariants'):
        for val in nonfinite_seq:
            v = InvariantViolation(
                name='E_residual_cons_frac_gross',
                status_code=30,
                value=val,
                tolerance=2.0e-4,
                message=f'E_residual_cons_frac_gross={val}',
                report_only=False,
            )
            handle_violations([v], mode='report', tracker=tracker)

    nf_records = [r for r in caplog.records if 'E_residual_cons_frac_gross' in r.message]
    assert len(nf_records) == 3
    assert 'nan' in nf_records[0].message.lower()
    assert 'inf' in nf_records[1].message
    assert '-inf' in nf_records[2].message
