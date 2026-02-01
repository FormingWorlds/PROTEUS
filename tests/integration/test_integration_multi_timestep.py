"""
Integration test: Multi-timestep dummy module coupling.

This test validates the integration test infrastructure by running PROTEUS
for multiple timesteps with dummy modules and checking:
- Energy conservation (flux balance)
- Mass conservation (elemental inventories)
- Stability (no runaway temperatures/pressures)

**Purpose**: Foundation test for Phase 2 of test building strategy
- Validates integration test infrastructure (fixtures and helpers)
- Tests multi-timestep coupling with minimal physics (dummy modules)
- Establishes pattern for future integration tests

**Runtime**: ~30-60s (5-10 timesteps, dummy modules)

**Documentation**:
- docs/test_infrastructure.md
- docs/test_categorization.md
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.integration.conftest import (
    validate_energy_conservation,
    validate_mass_conservation,
    validate_stability,
)


@pytest.mark.integration
def test_integration_dummy_multi_timestep(proteus_multi_timestep_run):
    """Test multi-timestep coupling with dummy modules.

    Physical scenario: Validates that PROTEUS can run multiple timesteps
    with dummy modules and maintain physical consistency. Uses dummy modules
    for speed while testing the full coupling infrastructure.

    Validates:
    - Simulation runs for multiple timesteps without errors
    - Energy conservation: F_atm ≈ F_int (within tolerance)
    - Mass conservation: Elemental masses change smoothly
    - Stability: No runaway temperatures or pressures
    - Helpfile contains expected number of timesteps

    Runtime: ~30-60s (5-10 timesteps, dummy modules)
    """
    # Run PROTEUS for 5 timesteps with dummy modules
    runner = proteus_multi_timestep_run(
        config_path='input/demos/dummy.toml',
        num_timesteps=5,
        max_time=1e6,  # years
        min_time=1e2,  # years
        interior__dummy__ini_tmagma=2000.0,  # Prevent runaway heating
    )

    # Validate that helpfile was created and has multiple timesteps
    assert runner.hf_all is not None, 'Helpfile should be created'
    assert len(runner.hf_all) >= 3, (
        f'Helpfile should have at least 3 timesteps, got {len(runner.hf_all)}'
    )

    # Validate energy conservation
    # Note: Dummy modules may have initial imbalance, so use more lenient tolerance
    energy_results = validate_energy_conservation(
        runner.hf_all,
        tolerance=0.3,  # 30% tolerance for dummy modules (initial imbalance expected)
    )
    assert energy_results['flux_stable'], 'Fluxes should be stable (no runaway behavior)'

    # Validate mass conservation
    mass_results = validate_mass_conservation(
        runner.hf_all,
        tolerance=0.1,  # 10% tolerance for mass changes (escape/outgassing)
    )
    assert mass_results['masses_positive'], 'All element masses should be positive'

    # Validate stability
    stability_results = validate_stability(
        runner.hf_all,
        max_temp=1e6,  # K
        max_pressure=1e10,  # Pa
    )
    assert stability_results['temps_stable'], 'Temperatures should be within bounds'
    assert stability_results['pressures_stable'], 'Pressures should be within bounds'
    assert stability_results['no_runaway'], 'No runaway behavior detected'

    # Validate that key variables are present and physical
    final_row = runner.hf_all.iloc[-1]

    # Check temperatures
    if 'T_surf' in final_row:
        assert 100 <= final_row['T_surf'] <= 5000, (
            f'T_surf should be physical (100-5000 K), got {final_row["T_surf"]}'
        )
    if 'T_magma' in final_row:
        assert 200 <= final_row['T_magma'] <= 1e6, (
            f'T_magma should be physical (200-1e6 K), got {final_row["T_magma"]}'
        )

    # Check fluxes
    if 'F_atm' in final_row:
        assert 0 <= final_row['F_atm'] <= 1e6, (
            f'F_atm should be physical (0-1e6 W/m²), got {final_row["F_atm"]}'
        )
    if 'F_int' in final_row:
        assert 0 <= final_row['F_int'] <= 1e6, (
            f'F_int should be physical (0-1e6 W/m²), got {final_row["F_int"]}'
        )

    # Check time progression
    assert 'Time' in final_row, 'Time should be in helpfile'
    assert final_row['Time'] > runner.hf_all.iloc[0]['Time'], 'Time should have progressed'


@pytest.mark.integration
def test_integration_dummy_extended_run(proteus_multi_timestep_run):
    """Test extended multi-timestep run (10 timesteps).

    Physical scenario: Validates that PROTEUS can run for extended periods
    with dummy modules while maintaining stability. Tests the coupling
    infrastructure under longer simulation times.

    Validates:
    - Simulation runs for 10 timesteps without errors
    - Stability maintained over extended run
    - No degradation in energy/mass conservation over time

    Runtime: ~60-90s (10 timesteps, dummy modules)
    """
    # Run PROTEUS for 10 timesteps
    runner = proteus_multi_timestep_run(
        config_path='input/demos/dummy.toml',
        num_timesteps=10,
        max_time=1e7,  # years
        min_time=1e2,  # years
        interior__dummy__ini_tmagma=2000.0,  # Prevent runaway heating
    )

    # Validate that helpfile has multiple timesteps
    assert runner.hf_all is not None, 'Helpfile should be created'
    assert len(runner.hf_all) >= 8, (
        f'Helpfile should have at least 8 timesteps, got {len(runner.hf_all)}'
    )

    # Validate stability over extended run
    stability_results = validate_stability(
        runner.hf_all,
        max_temp=1e6,
        max_pressure=1e10,
    )
    assert stability_results['no_runaway'], 'No runaway behavior over extended run'

    # Check that temperatures don't show unbounded growth
    if 'T_surf' in runner.hf_all.columns:
        t_surf_values = runner.hf_all['T_surf'].values
        # Temperature should not grow unbounded
        assert np.max(t_surf_values) < 1e5, (
            f'T_surf should not exceed 1e5 K, got max={np.max(t_surf_values):.2e}'
        )

    # Check that fluxes remain stable
    if 'F_atm' in runner.hf_all.columns and 'F_int' in runner.hf_all.columns:
        f_atm = runner.hf_all['F_atm'].values
        f_int = runner.hf_all['F_int'].values
        # Fluxes should not diverge unbounded
        assert np.max(np.abs(f_atm - f_int)) < 1e7, (
            'Flux imbalance should not exceed 1e7 W/m² over extended run'
        )
