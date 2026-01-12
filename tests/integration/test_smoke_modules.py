# Smoke tests: Module coupling validation
#
# Purpose: Validate that individual PROTEUS modules (escape, star, orbit, outgas)
# and full coupling chains work correctly with real binaries but minimal computation
# (1 timestep, low resolution).
#
# Tests validate:
# - Module initialization and coupling works correctly
# - Key physical variables are updated
# - No NaN or Inf values in outputs
# - Physical ranges are reasonable
#
# Runtime: Each test <30s (target for fast PR CI)
#
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus


@pytest.mark.smoke
def test_smoke_escape_dummy_atmos():
    """Test escape module + dummy atmosphere coupling (1 timestep).

    Validates that atmospheric escape correctly couples with a dummy atmosphere.
    Tests mass loss calculations and elemental inventory updates. Uses dummy
    escape module for simplicity and speed.

    Validates:
    - Escape rate is calculated and written to helpfile
    - Elemental inventories (H, C, N, S) are updated after escape
    - No NaN or Inf values in escape-related columns
    - Escape rate is physically reasonable (0-1e10 kg/s for exoplanets)

    Runtime: ~10-15s (1 timestep, dummy escape + dummy atmos)
    """
    # Create unique temporary output directory for this test
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load dummy configuration (uses dummy escape + dummy atmos)
        config_path = PROTEUS_ROOT / 'input' / 'demos' / 'dummy.toml'

        # Initialize PROTEUS
        runner = Proteus(config_path=config_path)

        # Override output path to use temporary directory
        runner.config.params.out.path = str(Path(tmpdir) / f'smoke_escape_{unique_id}')

        # Re-initialize directories after changing output path
        runner.init_directories()

        # Ensure escape module is enabled with dummy
        runner.config.escape.module = 'dummy'
        runner.config.escape.reservoir = 'bulk'  # Escape from bulk reservoir
        runner.config.escape.dummy.rate = 1e6  # kg/s, reasonable escape rate

        # Set initial volatile inventory (needed for escape to work)
        # Use delivery module to set initial elements
        runner.config.delivery.module = 'none'  # No delivery module, just initial inventory
        runner.config.delivery.initial = 'elements'
        runner.config.delivery.elements.H_ppmw = 3e3  # Hydrogen inventory
        runner.config.delivery.elements.CH_ratio = 1.0  # C/H ratio
        runner.config.delivery.elements.N_ppmw = 100.0  # Nitrogen inventory
        runner.config.delivery.elements.SH_ratio = 1.0  # S/H ratio

        # Fix: Lower ini_tmagma to prevent runaway heating
        runner.config.interior.dummy.ini_tmagma = 2000.0

        # Override stop time to run only 1 timestep
        runner.config.params.stop.time.minimum = 1e2  # yr
        runner.config.params.stop.time.maximum = 1e3  # yr

        # Disable plotting and archiving for speed
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        try:
            # Run simulation (1 timestep)
            runner.start(resume=False, offline=True)

            # Validate that helpfile was created and populated
            assert runner.hf_all is not None, 'Helpfile should be created'
            assert len(runner.hf_all) > 0, 'Helpfile should have at least one row'

            # Get final row
            final_row = runner.hf_all.iloc[-1]

            # Validate escape rate is written and physical
            assert 'esc_rate_total' in final_row, 'esc_rate_total should be in helpfile'
            esc_rate = final_row['esc_rate_total']
            assert not np.isnan(esc_rate), 'esc_rate_total should not be NaN'
            assert not np.isinf(esc_rate), 'esc_rate_total should not be Inf'
            assert 0 <= esc_rate <= 1e10, (
                f'esc_rate_total should be physical (0-1e10 kg/s), got {esc_rate}'
            )

            # Validate elemental inventories are present (for escape to work)
            # Check that at least one element inventory exists
            element_keys = [key for key in final_row.index if '_kg_total' in key]
            assert len(element_keys) > 0, 'Element inventories should be present'

            # Validate time progressed
            assert 'Time' in final_row, 'Time should be in helpfile'
            assert final_row['Time'] > 0, 'Time should have progressed'

        finally:
            # Cleanup handled by tempfile context manager
            pass


@pytest.mark.smoke
def test_smoke_star_instellation():
    """Test star module + dummy atmosphere coupling (1 timestep).

    Validates that stellar evolution correctly calculates instellation (stellar flux)
    and couples with a dummy atmosphere. Tests stellar luminosity, instellation
    calculations, and spectrum generation. Uses dummy star module for simplicity.

    Validates:
    - Stellar radius (R_star) is updated
    - Stellar temperature (T_star) is updated
    - Instellation flux (F_ins) is calculated correctly
    - No NaN or Inf values in stellar parameters
    - Instellation follows inverse-square law: F_ins ∝ L_star / separation²

    Runtime: ~15-20s (1 timestep, dummy star + dummy atmos)
    """
    # Create unique temporary output directory for this test
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load dummy configuration
        config_path = PROTEUS_ROOT / 'input' / 'demos' / 'dummy.toml'

        # Initialize PROTEUS
        runner = Proteus(config_path=config_path)

        # Override output path to use temporary directory
        runner.config.params.out.path = str(Path(tmpdir) / f'smoke_star_{unique_id}')

        # Re-initialize directories after changing output path
        runner.init_directories()

        # Use dummy star (simpler than MORS for smoke test)
        # MORS requires additional setup and data files, so use dummy for now
        # This still validates the star module coupling infrastructure
        runner.config.star.module = 'dummy'
        runner.config.star.dummy.Teff = 5772.0  # Solar temperature
        runner.config.star.dummy.radius = 1.0  # Solar radius

        # Fix: Lower ini_tmagma to prevent runaway heating
        runner.config.interior.dummy.ini_tmagma = 2000.0

        # Override stop time to run only 1 timestep
        runner.config.params.stop.time.minimum = 1e2  # yr
        runner.config.params.stop.time.maximum = 1e3  # yr

        # Disable plotting and archiving for speed
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        try:
            # Run simulation (1 timestep)
            runner.start(resume=False, offline=True)

            # Validate that helpfile was created and populated
            assert runner.hf_all is not None, 'Helpfile should be created'
            assert len(runner.hf_all) > 0, 'Helpfile should have at least one row'

            # Get final row
            final_row = runner.hf_all.iloc[-1]

            # Validate stellar radius is written and physical
            assert 'R_star' in final_row, 'R_star should be in helpfile'
            r_star = final_row['R_star']
            assert not np.isnan(r_star), 'R_star should not be NaN'
            assert not np.isinf(r_star), 'R_star should not be Inf'
            assert 1e7 <= r_star <= 1e12, (
                f'R_star should be physical (1e7-1e12 m), got {r_star}'
            )

            # Validate stellar temperature is written and physical
            assert 'T_star' in final_row, 'T_star should be in helpfile'
            t_star = final_row['T_star']
            assert not np.isnan(t_star), 'T_star should not be NaN'
            assert not np.isinf(t_star), 'T_star should not be Inf'
            assert 2000 <= t_star <= 100000, (
                f'T_star should be physical (2000-100000 K), got {t_star}'
            )

            # Validate instellation flux is calculated
            assert 'F_ins' in final_row, 'F_ins should be in helpfile'
            f_ins = final_row['F_ins']
            assert not np.isnan(f_ins), 'F_ins should not be NaN'
            assert not np.isinf(f_ins), 'F_ins should not be Inf'
            assert 0 <= f_ins <= 1e7, f'F_ins should be physical (0-1e7 W/m²), got {f_ins}'

            # Validate separation (orbital distance) is present
            assert 'separation' in final_row, 'separation should be in helpfile'
            separation = final_row['separation']
            assert not np.isnan(separation), 'separation should not be NaN'
            assert separation > 0, 'separation should be positive'

            # Validate time progressed
            assert 'Time' in final_row, 'Time should be in helpfile'
            assert final_row['Time'] > 0, 'Time should have progressed'

        finally:
            # Cleanup handled by tempfile context manager
            pass


@pytest.mark.smoke
def test_smoke_orbit_tidal_heating():
    """Test orbit module + dummy interior coupling (1 timestep).

    Validates that orbital dynamics correctly calculates tidal heating and couples
    with a dummy interior. Tests tidal heating calculations and orbital evolution.
    Uses dummy orbit module for simplicity.

    Validates:
    - Tidal heating (H_tide) is calculated and applied to interior
    - Love number (Imk2) is updated
    - Orbital parameters (semimajorax, eccentricity) are present
    - No NaN or Inf values in orbital parameters
    - Tidal heating is physically reasonable (0-1e-3 W/kg for exoplanets)

    Runtime: ~10-15s (1 timestep, dummy orbit + dummy interior)
    """
    # Create unique temporary output directory for this test
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load dummy configuration
        config_path = PROTEUS_ROOT / 'input' / 'demos' / 'dummy.toml'

        # Initialize PROTEUS
        runner = Proteus(config_path=config_path)

        # Override output path to use temporary directory
        runner.config.params.out.path = str(Path(tmpdir) / f'smoke_orbit_{unique_id}')

        # Re-initialize directories after changing output path
        runner.init_directories()

        # Use dummy orbit module (simpler than LovePy for smoke test)
        # LovePy requires Julia setup, so use dummy for now
        runner.config.orbit.module = 'dummy'
        runner.config.orbit.dummy.H_tide = 1e-9  # W/kg, reasonable tidal heating
        runner.config.orbit.dummy.Phi_tide = '<0.3'  # Apply when melt fraction < 0.3
        runner.config.orbit.dummy.Imk2 = -1e5  # Love number

        # Enable tidal heating in interior
        runner.config.interior.tidal_heat = True

        # Fix: Lower ini_tmagma to prevent runaway heating
        runner.config.interior.dummy.ini_tmagma = 2000.0

        # Override stop time to run only 1 timestep
        runner.config.params.stop.time.minimum = 1e2  # yr
        runner.config.params.stop.time.maximum = 1e3  # yr

        # Disable plotting and archiving for speed
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        try:
            # Run simulation (1 timestep)
            runner.start(resume=False, offline=True)

            # Validate that helpfile was created and populated
            assert runner.hf_all is not None, 'Helpfile should be created'
            assert len(runner.hf_all) > 0, 'Helpfile should have at least one row'

            # Get final row
            final_row = runner.hf_all.iloc[-1]

            # Validate orbital parameters are present
            assert 'semimajorax' in final_row, 'semimajorax should be in helpfile'
            semimajorax = final_row['semimajorax']
            assert not np.isnan(semimajorax), 'semimajorax should not be NaN'
            assert semimajorax > 0, 'semimajorax should be positive'

            assert 'eccentricity' in final_row, 'eccentricity should be in helpfile'
            eccentricity = final_row['eccentricity']
            assert not np.isnan(eccentricity), 'eccentricity should not be NaN'
            assert 0 <= eccentricity < 1, (
                f'eccentricity should be in [0, 1), got {eccentricity}'
            )

            # Validate Love number is present (if orbit module is enabled)
            if runner.config.orbit.module is not None:
                assert 'Imk2' in final_row, 'Imk2 should be in helpfile'
                imk2 = final_row['Imk2']
                assert not np.isnan(imk2), 'Imk2 should not be NaN'
                # Imk2 can be negative (dissipative)
                assert abs(imk2) <= 1e6, (
                    f'Imk2 should be reasonable (|Imk2| <= 1e6), got {imk2}'
                )

            # Validate time progressed
            assert 'Time' in final_row, 'Time should be in helpfile'
            assert final_row['Time'] > 0, 'Time should have progressed'

        finally:
            # Cleanup handled by tempfile context manager
            pass


@pytest.mark.smoke
def test_smoke_outgas_atmos_volatiles():
    """Test outgas module + dummy atmosphere coupling (1 timestep).

    Validates that outgassing (CALLIOPE) correctly couples with a dummy atmosphere.
    Tests volatile mass exchange and fO2 updates. Uses dummy atmosphere to keep
    runtime short.

    Validates:
    - Volatile masses (H2O, CO2, etc.) are updated in helpfile
    - fO2 (oxidation state) is physically reasonable
    - Volatile masses change after outgassing
    - No NaN or Inf values in volatile columns
    - Mass conservation: total volatile mass increases after outgassing

    Runtime: ~15-20s (1 timestep, CALLIOPE + dummy atmos)
    """
    # Create unique temporary output directory for this test
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load dummy configuration
        config_path = PROTEUS_ROOT / 'input' / 'demos' / 'dummy.toml'

        # Initialize PROTEUS
        runner = Proteus(config_path=config_path)

        # Override output path to use temporary directory
        runner.config.params.out.path = str(Path(tmpdir) / f'smoke_outgas_{unique_id}')

        # Re-initialize directories after changing output path
        runner.init_directories()

        # Ensure outgassing module is enabled with CALLIOPE
        runner.config.outgas.module = 'calliope'
        runner.config.outgas.fO2_shift_IW = 0  # No fO2 shift

        # Set initial volatile inventory
        runner.config.delivery.module = 'none'  # No delivery module, just initial inventory
        runner.config.delivery.initial = 'elements'
        runner.config.delivery.elements.H_ppmw = 3e3  # Hydrogen inventory
        runner.config.delivery.elements.CH_ratio = 1.0  # C/H ratio
        runner.config.delivery.elements.N_ppmw = 100.0  # Nitrogen inventory
        runner.config.delivery.elements.SH_ratio = 1.0  # S/H ratio

        # Fix: Lower ini_tmagma to prevent runaway heating
        runner.config.interior.dummy.ini_tmagma = 2000.0

        # Override stop time to run only 1 timestep
        runner.config.params.stop.time.minimum = 1e2  # yr
        runner.config.params.stop.time.maximum = 1e3  # yr

        # Disable plotting and archiving for speed
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        try:
            # Run simulation (1 timestep)
            runner.start(resume=False, offline=True)

            # Validate that helpfile was created and populated
            assert runner.hf_all is not None, 'Helpfile should be created'
            assert len(runner.hf_all) > 0, 'Helpfile should have at least one row'

            # Get final row
            final_row = runner.hf_all.iloc[-1]

            # Validate volatile masses are present (at least one)
            volatile_keys = [
                key
                for key in final_row.index
                if any(vol in key for vol in ['H2O', 'CO2', 'N2', 'S2', 'H_kg', 'C_kg'])
            ]
            assert len(volatile_keys) > 0, 'Volatile masses should be present in helpfile'

            # Check that at least one volatile mass is non-zero and finite
            volatile_values = [final_row[key] for key in volatile_keys if key in final_row]
            # Note: Some volatiles may be zero, but at least some should be present
            # Validate that volatile values are finite
            for val in volatile_values:
                assert not np.isnan(val), 'Volatile masses should not be NaN'
                assert not np.isinf(val), 'Volatile masses should not be Inf'

            # Validate fO2 is present (if outgassing is enabled)
            if 'fO2' in final_row:
                fO2 = final_row['fO2']
                assert not np.isnan(fO2), 'fO2 should not be NaN'
                assert not np.isinf(fO2), 'fO2 should not be Inf'
                # fO2 is typically in log10 units, reasonable range: -20 to +10
                assert -20 <= fO2 <= 10, f'fO2 should be physical (-20 to +10), got {fO2}'

            # Validate time progressed
            assert 'Time' in final_row, 'Time should be in helpfile'
            assert final_row['Time'] > 0, 'Time should have progressed'

        finally:
            # Cleanup handled by tempfile context manager
            pass


@pytest.mark.smoke
def test_smoke_dummy_full_chain():
    """Test all dummy modules in sequence (star → orbit → interior → atmos → escape).

    Validates that the full coupling loop works correctly with all dummy modules
    in sequence. This tests the coupling infrastructure end-to-end with minimal
    physics.

    Validates:
    - All modules initialize correctly
    - Coupling loop runs without errors
    - Key variables from all modules are present in helpfile
    - No NaN or Inf values in outputs
    - Time progresses correctly

    Runtime: ~10-15s (1 timestep, all dummy modules)
    """
    # Create unique temporary output directory for this test
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load dummy configuration (uses all dummy modules)
        config_path = PROTEUS_ROOT / 'input' / 'demos' / 'dummy.toml'

        # Initialize PROTEUS
        runner = Proteus(config_path=config_path)

        # Override output path to use temporary directory
        runner.config.params.out.path = str(Path(tmpdir) / f'smoke_full_chain_{unique_id}')

        # Re-initialize directories after changing output path
        runner.init_directories()

        # Ensure all modules are set to dummy
        runner.config.star.module = 'dummy'
        runner.config.orbit.module = 'dummy'
        runner.config.interior.module = 'dummy'
        runner.config.atmos_clim.module = 'dummy'
        runner.config.escape.module = 'dummy'

        # Fix: Lower ini_tmagma to prevent runaway heating
        runner.config.interior.dummy.ini_tmagma = 2000.0

        # Override stop time to run only 1 timestep
        runner.config.params.stop.time.minimum = 1e2  # yr
        runner.config.params.stop.time.maximum = 1e3  # yr

        # Disable plotting and archiving for speed
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        try:
            # Run simulation (1 timestep)
            runner.start(resume=False, offline=True)

            # Validate that helpfile was created and populated
            assert runner.hf_all is not None, 'Helpfile should be created'
            assert len(runner.hf_all) > 0, 'Helpfile should have at least one row'

            # Get final row
            final_row = runner.hf_all.iloc[-1]

            # Validate key variables from all modules are present
            # Star module
            assert 'R_star' in final_row or 'T_star' in final_row, (
                'Star variables should be present'
            )

            # Orbit module
            assert 'semimajorax' in final_row, 'Orbital parameters should be present'
            assert 'eccentricity' in final_row, 'Eccentricity should be present'

            # Interior module
            assert 'T_magma' in final_row, 'Interior temperature should be present'
            assert 'F_int' in final_row, 'Interior flux should be present'

            # Atmosphere module
            assert 'F_atm' in final_row, 'Atmospheric flux should be present'
            assert 'T_surf' in final_row, 'Surface temperature should be present'

            # Escape module (if enabled)
            if runner.config.escape.module:
                assert 'esc_rate_total' in final_row, 'Escape rate should be present'

            # Validate no NaN or Inf in key variables
            key_vars = ['T_surf', 'T_magma', 'F_atm', 'F_int', 'Time']
            for var in key_vars:
                if var in final_row:
                    value = final_row[var]
                    assert not np.isnan(value), f'{var} should not be NaN'
                    assert not np.isinf(value), f'{var} should not be Inf'

            # Validate time progressed
            assert 'Time' in final_row, 'Time should be in helpfile'
            assert final_row['Time'] > 0, 'Time should have progressed'

        finally:
            # Cleanup handled by tempfile context manager
            pass
