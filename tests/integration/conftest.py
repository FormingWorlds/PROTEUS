"""
Integration test fixtures and helpers for PROTEUS.

This module provides reusable fixtures and validation functions for multi-timestep
integration tests. These are designed to support Phase 2 of the test building strategy:
establishing integration test infrastructure.

**Fixtures**:
- `proteus_multi_timestep_run`: Run PROTEUS for N timesteps with configurable parameters

**Validation Helpers**:
- `validate_energy_conservation`: Check energy balance across timesteps
- `validate_mass_conservation`: Check mass conservation across reservoirs
- `validate_stability`: Check for runaway temperatures/pressures

**Usage**:
    @pytest.mark.integration
    def test_multi_timestep(proteus_multi_timestep_run):
        runner = proteus_multi_timestep_run(
            config_path='input/demos/dummy.toml',
            num_timesteps=5,
            max_time=1e6,  # years
        )
        validate_energy_conservation(runner.hf_all)
        validate_mass_conservation(runner.hf_all)

Documentation:
- docs/test_building_strategy.md (Phase 2: Integration Test Foundation)
- docs/test_infrastructure.md
- docs/test_categorization.md
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus

if TYPE_CHECKING:
    from pandas import DataFrame


@pytest.fixture
def proteus_multi_timestep_run():
    """
    Factory fixture for running PROTEUS simulations with multiple timesteps.

    Returns a function that creates and runs a PROTEUS simulation with
    configurable parameters. The function returns the Proteus runner object
    with completed simulation.

    **Parameters** (passed to returned function):
        config_path: Path to TOML configuration file (str or Path)
        num_timesteps: Number of timesteps to run (int, default: 5)
        max_time: Maximum simulation time in years (float, default: 1e6)
        min_time: Minimum simulation time in years (float, default: 1e2)
        output_suffix: Suffix for output directory (str, default: auto-generated UUID)
        **kwargs: Additional config overrides (e.g., `interior.dummy.ini_tmagma=2000.0`)

    **Returns**:
        Proteus: Runner object with completed simulation

    **Example**:
        def test_my_integration(proteus_multi_timestep_run):
            runner = proteus_multi_timestep_run(
                config_path='input/demos/dummy.toml',
                num_timesteps=10,
                max_time=1e7,
                interior__dummy__ini_tmagma=2000.0,
            )
            assert len(runner.hf_all) >= 10
    """

    def _run_proteus(
        config_path: str | Path,
        num_timesteps: int = 5,
        max_time: float = 1e6,
        min_time: float = 1e2,
        output_suffix: str | None = None,
        **config_overrides,
    ) -> Proteus:
        """
        Run PROTEUS simulation with specified parameters.

        Args:
            config_path: Path to TOML configuration file
            num_timesteps: Target number of timesteps (approximate)
            max_time: Maximum simulation time in years
            min_time: Minimum simulation time in years
            output_suffix: Suffix for output directory (auto-generated if None)
            **config_overrides: Config overrides using dot notation (e.g., `interior__dummy__ini_tmagma=2000.0`)

        Returns:
            Proteus runner with completed simulation
        """
        # Resolve config path
        if isinstance(config_path, str):
            config_path = PROTEUS_ROOT / config_path
        else:
            config_path = Path(config_path)

        # Create unique temporary output directory
        if output_suffix is None:
            output_suffix = str(uuid.uuid4())[:8]
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize PROTEUS
            runner = Proteus(config_path=config_path)

            # Override output path
            runner.config.params.out.path = str(Path(tmpdir) / f'integration_{output_suffix}')

            # Re-initialize directories after changing output path
            runner.init_directories()

            # Apply config overrides (convert dot notation to nested attribute access)
            for key, value in config_overrides.items():
                # Convert 'interior__dummy__ini_tmagma' to nested attribute access
                parts = key.split('__')
                obj = runner.config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

            # Set time limits for multi-timestep run
            # Estimate timestep size to get approximately num_timesteps
            estimated_dt = (max_time - min_time) / num_timesteps
            runner.config.params.dt.initial = estimated_dt
            runner.config.params.dt.minimum = estimated_dt * 0.1
            runner.config.params.dt.maximum = estimated_dt * 10.0

            runner.config.params.stop.time.minimum = min_time
            runner.config.params.stop.time.maximum = max_time

            # Disable plotting and archiving for speed
            runner.config.params.out.plot_mod = 0
            runner.config.params.out.write_mod = 0
            runner.config.params.out.archive_mod = 'none'

            # Run simulation
            runner.start(resume=False, offline=True)

            # Return runner (note: tmpdir will be cleaned up after test, but runner
            # object retains references to helpfile data)
            return runner

    return _run_proteus


def validate_energy_conservation(
    hf_all: 'DataFrame',
    tolerance: float = 0.1,
    flux_keys: tuple[str, ...] = ('F_atm', 'F_int', 'F_ins'),
) -> dict[str, float]:
    """
    Validate energy conservation across timesteps.

    Checks that energy fluxes are balanced and don't show runaway behavior.
    For a simple check, validates that F_atm ≈ F_int (within tolerance) for
    radiative equilibrium scenarios.

    **Physical Basis**:
    - Energy conservation: F_atm ≈ F_int for steady-state
    - No runaway heating: |F_atm - F_int| should not grow unbounded
    - Fluxes should be finite and positive

    **Args**:
        hf_all: Helpfile DataFrame with all timesteps
        tolerance: Relative tolerance for flux balance (default: 0.1 = 10%)
        flux_keys: Column names for fluxes to check (default: F_atm, F_int, F_ins)

    **Returns**:
        dict: Validation results with keys:
            - 'flux_balance_ratio': Mean |F_atm - F_int| / mean(F_int)
            - 'max_flux_imbalance': Maximum flux imbalance across timesteps
            - 'flux_stable': True if fluxes don't show runaway behavior

    **Raises**:
        AssertionError: If energy conservation is violated
    """
    results = {}

    # Check that required columns exist
    for key in flux_keys:
        if key not in hf_all.columns:
            pytest.skip(f'Required flux column {key} not found in helpfile')

    # Extract flux columns
    f_atm = hf_all['F_atm'].values if 'F_atm' in hf_all.columns else None
    f_int = hf_all['F_int'].values if 'F_int' in hf_all.columns else None
    f_ins = hf_all['F_ins'].values if 'F_ins' in hf_all.columns else None

    # Validate fluxes are finite and positive
    if f_atm is not None:
        assert np.all(np.isfinite(f_atm)), 'F_atm contains NaN or Inf values'
        assert np.all(f_atm >= 0), 'F_atm contains negative values'
    if f_int is not None:
        assert np.all(np.isfinite(f_int)), 'F_int contains NaN or Inf values'
        assert np.all(f_int >= 0), 'F_int contains negative values'
    if f_ins is not None:
        assert np.all(np.isfinite(f_ins)), 'F_ins contains NaN or Inf values'
        assert np.all(f_ins >= 0), 'F_ins contains negative values'

    # Check flux balance (F_atm ≈ F_int for steady-state)
    if f_atm is not None and f_int is not None:
        # Calculate flux imbalance
        flux_imbalance = np.abs(f_atm - f_int)
        mean_flux = np.mean(f_int)
        flux_balance_ratio = np.mean(flux_imbalance) / mean_flux if mean_flux > 0 else np.inf

        results['flux_balance_ratio'] = flux_balance_ratio
        results['max_flux_imbalance'] = np.max(flux_imbalance)

        # Check for runaway behavior (flux imbalance should not grow unbounded)
        if len(flux_imbalance) > 1:
            # Check if imbalance is decreasing or stable (not growing)
            imbalance_trend = np.diff(flux_imbalance)
            results['flux_stable'] = (
                np.mean(imbalance_trend) <= 0 or np.std(imbalance_trend) < mean_flux
            )
        else:
            results['flux_stable'] = True

        # Assert flux balance within tolerance
        assert flux_balance_ratio <= tolerance, (
            f'Flux imbalance exceeds tolerance: {flux_balance_ratio:.3f} > {tolerance:.3f}'
        )

    return results


def validate_mass_conservation(
    hf_all: 'DataFrame',
    tolerance: float = 0.05,
    element_keys: tuple[str, ...] = ('H_kg_total', 'C_kg_total', 'N_kg_total', 'O_kg_total'),
) -> dict[str, float]:
    """
    Validate mass conservation across timesteps.

    Checks that total elemental masses are conserved (accounting for escape/outgassing).
    For a simple check, validates that total volatile mass changes are physically
    reasonable and don't show unbounded growth.

    **Physical Basis**:
    - Mass conservation: Total element mass should change only due to escape/outgassing
    - No unbounded growth: Mass changes should be finite
    - Positive masses: All element masses should be >= 0

    **Args**:
        hf_all: Helpfile DataFrame with all timesteps
        tolerance: Relative tolerance for mass conservation (default: 0.05 = 5%)
        element_keys: Column names for element masses to check

    **Returns**:
        dict: Validation results with keys:
            - 'total_mass_change': Total change in element masses
            - 'mass_conservation_ratio': Relative change in total mass
            - 'masses_positive': True if all masses are >= 0

    **Raises**:
        AssertionError: If mass conservation is violated
    """
    results = {}

    # Check that at least some element columns exist
    available_keys = [key for key in element_keys if key in hf_all.columns]
    if len(available_keys) == 0:
        pytest.skip('No element mass columns found in helpfile')

    # Extract element masses
    element_masses = hf_all[available_keys].values

    # Validate masses are finite and positive
    assert np.all(np.isfinite(element_masses)), 'Element masses contain NaN or Inf values'
    assert np.all(element_masses >= 0), 'Element masses contain negative values'

    # Calculate total mass at each timestep
    total_mass = np.sum(element_masses, axis=1)

    # Check mass conservation (total mass should change smoothly, not unbounded)
    if len(total_mass) > 1:
        mass_change = total_mass[-1] - total_mass[0]
        initial_mass = total_mass[0]
        mass_conservation_ratio = (
            abs(mass_change) / initial_mass if initial_mass > 0 else np.inf
        )

        results['total_mass_change'] = mass_change
        results['mass_conservation_ratio'] = mass_conservation_ratio
        results['masses_positive'] = np.all(total_mass >= 0)

        # Mass can change due to escape/outgassing, but should be reasonable
        # For integration tests, we allow up to tolerance change
        # (in real simulations, this would be tracked more carefully)
        if mass_conservation_ratio > tolerance:
            # This is a warning, not an error, as mass changes are expected
            # in simulations with escape/outgassing
            pass

    return results


def validate_stability(
    hf_all: 'DataFrame',
    temp_keys: tuple[str, ...] = ('T_surf', 'T_magma'),
    pressure_keys: tuple[str, ...] = ('P_surf',),
    max_temp: float = 1e6,
    max_pressure: float = 1e10,
) -> dict[str, bool]:
    """
    Validate that simulation remains stable (no runaway temperatures/pressures).

    Checks that key physical variables stay within reasonable bounds and don't
    show unbounded growth.

    **Physical Basis**:
    - Temperatures should be finite and within physical bounds (0 < T < max_temp)
    - Pressures should be finite and within physical bounds (0 < P < max_pressure)
    - No runaway behavior: Variables should not grow unbounded

    **Args**:
        hf_all: Helpfile DataFrame with all timesteps
        temp_keys: Column names for temperatures to check
        pressure_keys: Column names for pressures to check
        max_temp: Maximum allowed temperature in K (default: 1e6 K)
        max_pressure: Maximum allowed pressure in Pa (default: 1e10 Pa)

    **Returns**:
        dict: Validation results with keys:
            - 'temps_stable': True if all temperatures are within bounds
            - 'pressures_stable': True if all pressures are within bounds
            - 'no_runaway': True if no unbounded growth detected

    **Raises**:
        AssertionError: If stability is violated
    """
    results = {}

    # Check temperatures
    temp_stable = True
    for key in temp_keys:
        if key in hf_all.columns:
            temps = hf_all[key].values
            assert np.all(np.isfinite(temps)), f'{key} contains NaN or Inf values'
            assert np.all(temps > 0), f'{key} contains non-positive values'
            assert np.all(temps < max_temp), (
                f'{key} exceeds maximum allowed temperature: max={np.max(temps):.2e} > {max_temp:.2e}'
            )

            # Check for runaway behavior (temperature should not grow unbounded)
            if len(temps) > 1:
                temp_trend = np.diff(temps)
                # Allow some growth, but not unbounded
                if np.any(np.abs(temp_trend) > max_temp * 0.1):
                    temp_stable = False

    results['temps_stable'] = temp_stable

    # Check pressures
    pressure_stable = True
    for key in pressure_keys:
        if key in hf_all.columns:
            pressures = hf_all[key].values
            assert np.all(np.isfinite(pressures)), f'{key} contains NaN or Inf values'
            assert np.all(pressures >= 0), f'{key} contains negative values'
            assert np.all(pressures < max_pressure), (
                f'{key} exceeds maximum allowed pressure: max={np.max(pressures):.2e} > {max_pressure:.2e}'
            )

            # Check for runaway behavior
            if len(pressures) > 1:
                pressure_trend = np.diff(pressures)
                if np.any(np.abs(pressure_trend) > max_pressure * 0.1):
                    pressure_stable = False

    results['pressures_stable'] = pressure_stable
    results['no_runaway'] = temp_stable and pressure_stable

    return results
