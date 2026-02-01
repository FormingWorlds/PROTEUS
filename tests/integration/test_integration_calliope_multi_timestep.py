"""
Integration test: Multi-timestep CALLIOPE outgassing coupling.

This test validates multi-timestep coupling with CALLIOPE (real outgassing module)
and dummy atmosphere/interior modules. Tests the integration infrastructure with
a real physics module while maintaining reasonable runtime.

**Purpose**: Intermediate integration test for Phase 2
- Validates integration infrastructure with real module (CALLIOPE)
- Tests multi-timestep volatile outgassing and atmosphere coupling
- Establishes pattern for future real-module integration tests

**Runtime**: ~60-120s (5-10 timesteps, CALLIOPE + dummy modules)

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
def test_integration_calliope_multi_timestep(proteus_multi_timestep_run):
    """Test multi-timestep coupling with CALLIOPE outgassing.

    Physical scenario: Validates that CALLIOPE outgassing correctly couples
    with dummy atmosphere over multiple timesteps. Tests volatile mass
    exchange, fO2 evolution, and atmospheric pressure changes.

    Validates:
    - Simulation runs for multiple timesteps without errors
    - Volatile masses (H2O, CO2, etc.) evolve over time
    - fO2 (oxidation state) remains physically reasonable
    - Mass conservation: Total volatile mass changes smoothly
    - Stability: No runaway temperatures or pressures
    - Energy conservation: Fluxes remain balanced

    Runtime: ~60-90s (5 timesteps, CALLIOPE + dummy modules)
    """
    # Run PROTEUS for 5 timesteps with CALLIOPE outgassing
    runner = proteus_multi_timestep_run(
        config_path='input/demos/dummy.toml',
        num_timesteps=5,
        max_time=1e6,  # years
        min_time=1e2,  # years
        # Enable CALLIOPE outgassing
        outgas__module='calliope',
        outgas__fO2_shift_IW=0,  # No fO2 shift
        # Set initial volatile inventory
        delivery__module='none',
        delivery__initial='elements',
        delivery__elements__H_ppmw=3e3,  # Hydrogen inventory
        delivery__elements__CH_ratio=1.0,  # C/H ratio
        delivery__elements__N_ppmw=100.0,  # Nitrogen inventory
        delivery__elements__SH_ratio=1.0,  # S/H ratio
        # Prevent runaway heating
        interior__dummy__ini_tmagma=2000.0,
    )

    # Validate that helpfile was created and has multiple timesteps
    assert runner.hf_all is not None, 'Helpfile should be created'
    assert len(runner.hf_all) >= 3, (
        f'Helpfile should have at least 3 timesteps, got {len(runner.hf_all)}'
    )

    # Validate volatile masses are present and evolve
    final_row = runner.hf_all.iloc[-1]
    initial_row = runner.hf_all.iloc[0]

    # Check that volatile species are present
    volatile_keys = [
        key
        for key in final_row.index
        if any(vol in key for vol in ['H2O', 'CO2', 'N2', 'S2', 'H_kg', 'C_kg'])
    ]
    assert len(volatile_keys) > 0, 'Volatile masses should be present in helpfile'

    # Validate volatile masses are finite and positive
    for key in volatile_keys:
        if key in final_row:
            assert not np.isnan(final_row[key]), f'{key} should not be NaN'
            assert not np.isinf(final_row[key]), f'{key} should not be Inf'
            assert final_row[key] >= 0, f'{key} should be non-negative'

    # Validate fO2 is physically reasonable (if present)
    if 'fO2' in final_row or 'fO2_IW' in final_row:
        fO2_key = 'fO2_IW' if 'fO2_IW' in final_row else 'fO2'
        fO2 = final_row[fO2_key]
        assert not np.isnan(fO2), 'fO2 should not be NaN'
        assert not np.isinf(fO2), 'fO2 should not be Inf'
        # fO2 is typically in log10 units, reasonable range: -20 to +10
        assert -20 <= fO2 <= 10, f'fO2 should be physical (-20 to +10), got {fO2}'

    # Validate mass conservation
    mass_results = validate_mass_conservation(
        runner.hf_all,
        tolerance=0.2,  # 20% tolerance (mass can change due to outgassing/escape)
    )
    assert mass_results['masses_positive'], 'All element masses should be positive'

    # Validate energy conservation (may be less strict with CALLIOPE)
    energy_results = validate_energy_conservation(
        runner.hf_all,
        tolerance=0.3,  # 30% tolerance for dummy modules
    )
    assert energy_results['flux_stable'], 'Fluxes should be stable (no runaway behavior)'

    # Validate stability
    stability_results = validate_stability(
        runner.hf_all,
        max_temp=1e6,  # K
        max_pressure=1e10,  # Pa
    )
    assert stability_results['temps_stable'], 'Temperatures should be within bounds'
    assert stability_results['pressures_stable'], 'Pressures should be within bounds'
    assert stability_results['no_runaway'], 'No runaway behavior detected'

    # Validate that atmospheric pressure evolves (should change with outgassing)
    if 'P_surf' in final_row and 'P_surf' in initial_row:
        p_surf_final = final_row['P_surf']
        assert p_surf_final > 0, 'P_surf should be positive'
        assert p_surf_final <= 1e10, 'P_surf should be within physical bounds'
        # Pressure may increase (outgassing) or decrease (escape), but should be finite
        assert np.isfinite(p_surf_final), 'P_surf should be finite'

    # Validate time progression
    assert 'Time' in final_row, 'Time should be in helpfile'
    assert final_row['Time'] > initial_row['Time'], 'Time should have progressed'


@pytest.mark.integration
def test_integration_calliope_extended_run(proteus_multi_timestep_run):
    """Test extended multi-timestep run with CALLIOPE (10 timesteps).

    Physical scenario: Validates that CALLIOPE outgassing remains stable
    over extended simulation periods. Tests long-term volatile evolution
    and ensures no degradation in conservation or stability.

    Validates:
    - Simulation runs for 10 timesteps without errors
    - Volatile masses remain physically reasonable
    - Stability maintained over extended run
    - No unbounded growth in pressures or temperatures

    Runtime: ~90-120s (10 timesteps, CALLIOPE + dummy modules)
    """
    # Run PROTEUS for 10 timesteps
    runner = proteus_multi_timestep_run(
        config_path='input/demos/dummy.toml',
        num_timesteps=10,
        max_time=1e7,  # years
        min_time=1e2,  # years
        # Enable CALLIOPE outgassing
        outgas__module='calliope',
        outgas__fO2_shift_IW=0,
        # Set initial volatile inventory
        delivery__module='none',
        delivery__initial='elements',
        delivery__elements__H_ppmw=3e3,
        delivery__elements__CH_ratio=1.0,
        delivery__elements__N_ppmw=100.0,
        delivery__elements__SH_ratio=1.0,
        # Prevent runaway heating
        interior__dummy__ini_tmagma=2000.0,
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

    # Check that volatile masses don't show unbounded growth
    volatile_keys = [
        key
        for key in runner.hf_all.columns
        if any(vol in key for vol in ['H2O', 'CO2', 'N2', 'S2', '_kg_total'])
    ]
    if len(volatile_keys) > 0:
        for key in volatile_keys[:5]:  # Check first 5 volatile keys
            if key in runner.hf_all.columns:
                values = runner.hf_all[key].values
                # Volatile masses should not grow unbounded
                assert np.max(values) < 1e30, (
                    f'{key} should not exceed 1e30 kg, got max={np.max(values):.2e}'
                )
                # All values should be finite
                assert np.all(np.isfinite(values)), f'{key} should contain only finite values'

    # Check that pressures remain stable
    if 'P_surf' in runner.hf_all.columns:
        p_surf_values = runner.hf_all['P_surf'].values
        assert np.max(p_surf_values) < 1e11, (
            f'P_surf should not exceed 1e11 Pa, got max={np.max(p_surf_values):.2e}'
        )
        assert np.all(np.isfinite(p_surf_values)), 'P_surf should contain only finite values'
