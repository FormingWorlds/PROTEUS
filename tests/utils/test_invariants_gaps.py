"""Unit tests addressing invariant assertion gaps and mutant survival edge cases.

Covers multi-family stop-mode orchestration, status file generation and Zalmoxis
flushing, non-zero Q_radio and Q_tidal energy flux terms, multi-row history gross
throughput, non-finite input key checks, configuration tolerance sensitivity and
swaps, zero-M_vaps vapourise stops, and exact inequality boundaries.
"""

from __future__ import annotations

import logging
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
        'M_planet': 1.0e24,
        'M_atm': 2.0e24,
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
        assert exc_info.invariant_name == 'M_atm_planet_cap'
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
        ('M_vaps', float('nan'), 'volatile', 'M_atm_planet_cap'),
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


# ---------------------------------------------------------------------------
# Group 4: Config Fields, Sensitivity, and Swap Detection
# ---------------------------------------------------------------------------


def test_config_fields_ten_x_flips_outcome():
    """Verify each tolerance field at 10x its default flips the violation outcome."""
    cfg_def = _make_mock_config()

    # 1. planet_mass_cap_rtol (default 1e-6)
    row_cap = {
        'Time': 10.0,
        'M_planet': 1.0e24,
        'M_atm': 1.0e24 * (1.0 + 5.0e-6),
        'M_vaps': 0.0,
    }
    cfg_10x_cap = _make_mock_config(planet_mass_cap_rtol=1.0e-5)
    assert any(
        v.name == 'M_atm_planet_cap' for v in check_volatile_invariants(row_cap, cfg_def)
    )
    assert not any(
        v.name == 'M_atm_planet_cap' for v in check_volatile_invariants(row_cap, cfg_10x_cap)
    )

    # 2. volatile_gas_sum_rtol (default 1e-6)
    row_gas = {
        'Time': 10.0,
        'M_planet': 1.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e20,
        'H2O_kg_atm': 1.0e20 * (1.0 + 5.0e-6),
        'M_vaps': 0.0,
    }
    cfg_10x_gas = _make_mock_config(volatile_gas_sum_rtol=1.0e-5)
    assert any(
        v.name == 'M_vol_atm_species_sum' for v in check_volatile_invariants(row_gas, cfg_def)
    )
    assert not any(
        v.name == 'M_vol_atm_species_sum'
        for v in check_volatile_invariants(row_gas, cfg_10x_gas)
    )

    # 3. phi_global_atol (default 1e-6)
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
    """Verify swapping fields with equal defaults is caught by distinct tests."""
    cfg_asym = _make_mock_config(planet_mass_cap_rtol=1.0e-3, volatile_gas_sum_rtol=1.0e-7)
    row = {
        'Time': 10.0,
        'M_planet': 1.0e24,
        'M_atm': 1.0e24 * (1.0 + 1.0e-5),
        'M_vol_atm': 1.0e20,
        'H2O_kg_atm': 1.0e20 * (1.0 + 1.0e-5),
        'M_vaps': 0.0,
    }
    viols = check_volatile_invariants(row, cfg_asym)
    assert not any(v.name == 'M_atm_planet_cap' for v in viols)
    assert any(v.name == 'M_vol_atm_species_sum' for v in viols)

    cfg_phi_asym = _make_mock_config(phi_global_atol=1.0e-3, planet_mass_cap_rtol=1.0e-7)
    row_phi = {
        'Time': 10.0,
        'T_surf': 1000.0,
        'T_magma': 1500.0,
        'P_surf': 10.0,
        'Phi_global': 1.0 + 1.0e-5,
    }
    viols_phi = check_state_bounds_invariants(row_phi, cfg_phi_asym)
    assert not any(v.name == 'Phi_global_bounds' for v in viols_phi)


@pytest.mark.parametrize('bad_mode', ['warn', 'Stop', 'STOP', ''])
def test_config_mode_near_misses_rejected(bad_mode):
    """Verify near-miss mode strings are rejected by Invariants validator."""
    try:
        Invariants(mode=bad_mode)
        assert False, f'expected ValueError/TypeError for mode {bad_mode}'
    except (ValueError, TypeError):
        pass


def test_config_non_negative_fields_accept_zero():
    """Verify non-negative tolerances accept 0.0."""
    try:
        inv = Invariants(
            planet_mass_cap_rtol=0.0,
            escape_bound_atol_kg=0.0,
            phi_global_atol=0.0,
        )
    except ValueError as exc:
        assert False, f'non-negative tolerance rejected 0.0: {exc}'
    assert inv.planet_mass_cap_rtol == 0.0
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


def test_vapourise_with_zero_mvaps_and_matm_exceeds_mplanet_stops():
    """When vapourise=True and M_vaps <= 0, M_atm > M_planet produces a stopping violation."""
    cfg = _make_mock_config(vapourise=True)
    row = {
        'Time': 100.0,
        'M_planet': 1.0e24,
        'M_atm': 1.5e24,
        'M_vaps': 0.0,
        'M_vol_atm': 1.0e20,
    }
    viols = check_volatile_invariants(row, cfg)
    matching = [v for v in viols if v.name == 'M_atm_planet_cap']
    assert len(matching) == 1
    assert matching[0].report_only is False


def test_exact_threshold_boundary_rows_and_strict_inequalities():
    """Verify exact-threshold rows behave correctly under strict vs non-strict inequalities."""
    cfg = _make_mock_config()

    # 1. Kink boundary: T_magma == 301.0 uses step_energy_smin_rtol (0.05)
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
        'T_surf': 2500.0,
        'T_magma': 301.0,
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
    viols_kink = check_energy_invariants(row_kink_exact, cfg, hf_all)
    assert not any(v.name == 'step_dE_inc_frac' for v in viols_kink)

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

    # 4. Mass cap == cap_tol (1.0e-6) does not trigger violation
    row_cap_exact = {
        'Time': 10.0,
        'M_planet': 1.0e24,
        'M_atm': 1.0e24 * (1.0 + 1.0e-6),
        'M_vaps': 0.0,
    }
    viols_cap = check_volatile_invariants(row_cap_exact, cfg)
    assert not any(v.name == 'M_atm_planet_cap' for v in viols_cap)

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
    with caplog.at_level(logging.ERROR, logger='fwl.proteus.utils.invariants'):
        try:
            handle_violations([v_stop], mode='stop')
        except InvariantViolationError:
            pass
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

    # V04: rel_excess == cap_tol (0.5)
    cfg_cap = _make_mock_config(planet_mass_cap_rtol=0.5)
    row_v04 = {
        'Time': 10.0,
        'M_planet': 1.0e24,
        'M_atm': 1.5e24,
        'M_vaps': 0.0,
        'M_vol_atm': 1.0e20,
    }
    assert not any(
        v.name == 'M_atm_planet_cap' for v in check_volatile_invariants(row_v04, cfg_cap)
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
    """Verify volatile defaults, species-sum scaling, and effective mass (kills V06, V19-V22)."""
    cfg = _make_mock_config()
    # V06: scale = max(m_vol_atm, summed)
    row_v06 = {
        'Time': 10.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'M_vol_atm': 1.0e18,
        'H2O_kg_atm': 2.0e18,
        'M_vaps': 0.0,
    }
    v_v06 = [
        v for v in check_volatile_invariants(row_v06, cfg) if v.name == 'M_vol_atm_species_sum'
    ][0]
    assert abs(v_v06.value - 0.5) < 1.0e-12

    # V19: missing fO2_source defaults to user_constant
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

    # V20: missing vapourise defaults to False
    cfg_no_vap = SimpleNamespace(
        invariants=Invariants(),
        planet=SimpleNamespace(fO2_source='user_constant'),
        outgas=SimpleNamespace(module='atmodeller', solver_rtol=1e-4, mass_thresh=1e10),
    )
    row_excess = {
        'Time': 10.0,
        'M_planet': 1.0e24,
        'M_atm': 1.2e24,
        'M_vaps': 0.5e24,
        'M_vol_atm': 1.0e20,
    }
    assert any(
        v.name == 'M_atm_planet_cap' for v in check_volatile_invariants(row_excess, cfg_no_vap)
    )

    # V21: missing M_vol_atm defaults to 0.0
    row_no_mvol = {
        'Time': 10.0,
        'M_planet': 6.0e24,
        'M_atm': 1.0e20,
        'H2O_kg_atm': 1.0e20,
        'M_vaps': 0.0,
    }
    v_no_mvol = [
        v
        for v in check_volatile_invariants(row_no_mvol, cfg)
        if v.name == 'M_vol_atm_species_sum'
    ][0]
    assert abs(v_no_mvol.value - 1.0) < 1.0e-12

    # V22: m_atm_eff when abs(m_atm) <= closure_atol_kg
    row_small = {
        'Time': 10.0,
        'M_planet': 0.1,
        'M_atm': 0.5,
        'M_vaps': 0.0,
        'M_vol_atm': 0.0,
    }
    assert not any(
        v.name == 'M_atm_planet_cap' for v in check_volatile_invariants(row_small, cfg)
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


def test_config_threshold_custom_values_override_defaults():
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
    cfg_smin = _make_mock_config(step_energy_smin_rtol=0.2)
    row_smin = dict(row_e, T_magma=300.0, step_dE_state_heat_J=1.08e26, E_residual_cons_J=0.0)
    assert not any(
        v.name == 'step_dE_inc_frac'
        for v in check_energy_invariants(row_smin, cfg_smin, hf_all)
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
        'M_planet': 1.0e24,
        'M_atm': 2.0e24,
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

    # C06: planet_mass_cap_rtol accepts 0.0
    try:
        cfg = Invariants(planet_mass_cap_rtol=0.0)
    except ValueError as exc:
        assert False, f'planet_mass_cap_rtol=0.0 was rejected: {exc}'
    assert cfg.planet_mass_cap_rtol == 0.0
