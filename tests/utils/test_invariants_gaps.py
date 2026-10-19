"""Unit tests addressing invariant assertion gaps and mutant survival edge cases.

Covers multi-family stop-mode orchestration, status file generation and Zalmoxis
flushing, non-zero Q_radio and Q_tidal energy flux terms, multi-row history gross
throughput, non-finite input key checks, configuration tolerance sensitivity and
swaps, zero-M_vaps vapourise stops, and exact inequality boundaries.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from proteus.config._invariants import Invariants
from proteus.utils.invariants import (
    _ENERGY_FINITE_INPUT_KEYS,
    STATUS_ERROR_MASS_INVARIANT,
    STATUS_ERROR_STATE_BOUNDS_INVARIANT,
    InvariantTracker,
    InvariantViolationError,
    check_energy_invariants,
    check_row_invariants,
    check_state_bounds_invariants,
    check_structure_invariants,
    check_volatile_invariants,
    handle_violations,
)

if TYPE_CHECKING:
    from pathlib import Path

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
    with pytest.raises(InvariantViolationError) as exc_info:
        check_row_invariants(row_struct, cfg_stop, directories=dirs)
    assert exc_info.value.status_code == STATUS_ERROR_MASS_INVARIANT
    assert exc_info.value.invariant_name == 'struct_mass_desync_frac'
    status_lines = (tmp_path / 'status').read_text(encoding='utf-8').splitlines()
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
    with pytest.raises(InvariantViolationError) as exc_info:
        check_row_invariants(row_vol, cfg_stop, directories=dirs)
    assert exc_info.value.status_code == STATUS_ERROR_MASS_INVARIANT
    assert exc_info.value.invariant_name == 'M_atm_planet_cap'
    status_lines = (tmp_path / 'status').read_text(encoding='utf-8').splitlines()
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
    with pytest.raises(InvariantViolationError) as exc_info:
        check_row_invariants(row_state, cfg_stop, directories=dirs)
    assert exc_info.value.status_code == STATUS_ERROR_STATE_BOUNDS_INVARIANT
    assert exc_info.value.invariant_name == 'T_surf_bounds'
    status_lines = (tmp_path / 'status').read_text(encoding='utf-8').splitlines()
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

    with pytest.raises(InvariantViolationError) as exc_info:
        runner.start(resume=False, offline=True)

    assert exc_info.value.status_code == STATUS_ERROR_STATE_BOUNDS_INVARIANT
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
    assert check_energy_invariants(hf_row_zeros, cfg, hf_all) == []

    # Empty hf_all guard
    assert check_energy_invariants(hf_row_zeros, cfg, pd.DataFrame()) == []

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
    with pytest.raises((ValueError, TypeError)):
        Invariants(mode=bad_mode)


def test_config_non_negative_fields_accept_zero():
    """Verify non-negative tolerances accept 0.0."""
    inv = Invariants(
        planet_mass_cap_rtol=0.0,
        escape_bound_atol_kg=0.0,
        phi_global_atol=0.0,
    )
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
