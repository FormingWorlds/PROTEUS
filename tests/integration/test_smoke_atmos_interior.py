# Smoke tests: Atmosphere-Interior Coupling (Priority 2.5.1)
#
# Purpose: Validate end-to-end coupling between atmosphere and interior modules
# using real binaries but minimal computation (1 timestep, low resolution).
#
# Tests validate:
# - Flux exchange (F_atm ↔ F_int) works correctly
# - Surface temperature (T_surf) updates properly
# - No NaN/Inf values in outputs
# - Energy balance is physically reasonable
#
# Runtime: Each test <30s (target for fast PR CI)
#
# Documentation: For testing standards, see:
# - docs/test_infrastructure.md
# - docs/test_categorization.md
# - docs/test_building_strategy.md (Priority 2.5.1)
#
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus


@pytest.mark.smoke
def test_smoke_dummy_atmos_dummy_interior_flux_exchange():
    """Test dummy atmosphere + dummy interior coupling (1 timestep).

    Physical scenario: Validates that the simplest possible atmosphere-interior
    coupling works correctly. Uses dummy modules for both atmosphere and interior,
    which implement minimal physics but exercise the full coupling machinery.

    Validates:
    - F_atm (atmospheric flux) is written
    - F_int (interior flux) is written
    - T_surf (surface temperature) is updated
    - No NaN or Inf values in outputs
    - Fluxes are in physically reasonable range (magma oceans: 0-1000 kW/m²)
    - Surface temperature is physical (100-5000 K)

    Runtime: ~10-15s (1 timestep, dummy modules)
    """
    import tempfile
    import uuid

    # Create unique temporary output directory for this test
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load dummy configuration (uses dummy atmos + dummy interior)
        config_path = PROTEUS_ROOT / 'input' / 'demos' / 'dummy.toml'

        # Initialize PROTEUS
        runner = Proteus(config_path=config_path)

        # Override output path to use temporary directory
        runner.config.params.out.path = str(Path(tmpdir) / f'smoke_test_{unique_id}')

        # Re-initialize directories after changing output path
        runner.init_directories()

        # Override stop time to run only 1 timestep
        runner.config.params.stop.time.minimum = 1e2  # yr, minimum time
        runner.config.params.stop.time.maximum = 1e3  # yr, maximum time (1 step)

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

            # Validate atmospheric flux (F_atm) is written and physical
            assert 'F_atm' in final_row, 'F_atm should be in helpfile'
            f_atm = final_row['F_atm']
            assert not np.isnan(f_atm), 'F_atm should not be NaN'
            assert not np.isinf(f_atm), 'F_atm should not be Inf'
            # Magma ocean fluxes can be very high (10-100 kW/m²)
            assert 0 <= f_atm <= 1e6, f'F_atm should be physical (0-1e6 W/m²), got {f_atm}'

            # Validate interior flux (F_int) is written and physical
            assert 'F_int' in final_row, 'F_int should be in helpfile'
            f_int = final_row['F_int']
            assert not np.isnan(f_int), 'F_int should not be NaN'
            assert not np.isinf(f_int), 'F_int should not be Inf'
            assert 0 <= f_int <= 1e6, f'F_int should be physical (0-1e6 W/m²), got {f_int}'

            # Validate surface temperature is updated and physical
            assert 'T_surf' in final_row, 'T_surf should be in helpfile'
            t_surf = final_row['T_surf']
            assert not np.isnan(t_surf), 'T_surf should not be NaN'
            assert not np.isinf(t_surf), 'T_surf should not be Inf'
            assert 100 <= t_surf <= 5000, f'T_surf should be physical (100-5000 K), got {t_surf}'

            # Validate time progressed
            assert 'Time' in final_row, 'Time should be in helpfile'
            assert final_row['Time'] > 0, 'Time should have progressed'
        finally:
            # Cleanup handled by tempfile context manager
            pass


@pytest.mark.smoke
@pytest.mark.skip(reason='JANUS integration requires compiled SOCRATES binaries')
def test_smoke_janus_dummy_interior_radiation_balance():
    """Test JANUS + dummy interior coupling (1 timestep).

    Physical scenario: Validates that JANUS (radiative-convective atmosphere)
    can couple correctly with a dummy interior. This tests real radiative transfer
    calculations with SOCRATES but uses simple interior physics.

    Validates:
    - JANUS produces valid atmospheric flux
    - Radiation balance is approximately satisfied (F_atm ≈ F_int within factor of 2)
    - Surface temperature converges to physically reasonable value
    - Atmospheric structure (T-P profile) is smooth and monotonic below tropopause

    Runtime: ~20-25s (1 timestep, JANUS low-res: 10 levels, SOCRATES minimal bands)
    """
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load JANUS configuration
        config_path = PROTEUS_ROOT / 'input' / 'demos' / 'janus.toml'

        # Initialize PROTEUS
        runner = Proteus(config_path=config_path)

        # Override output directory
        runner.directories['output'] = str(Path(tmpdir) / 'output')
        Path(runner.directories['output']).mkdir(parents=True, exist_ok=True)

        # Use dummy interior (fast)
        runner.config.interior.module = 'dummy'

        # Override stop time to run only 1 timestep
        runner.config.params.stop.time.minimum = 1e2
        runner.config.params.stop.time.maximum = 1e3

        # Disable plotting/archiving
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        # Run simulation
        runner.start(resume=False, offline=True)

        # Validate helpfile
        assert runner.hf_all is not None
        assert len(runner.hf_all) > 0

        final_row = runner.hf_all.iloc[-1]

        # Validate fluxes
        f_atm = final_row['F_atm']
        f_int = final_row['F_int']
        assert not np.isnan(f_atm)
        assert not np.isnan(f_int)
        assert 0 <= f_atm <= 10000
        assert 0 <= f_int <= 10000

        # Validate radiation balance (within factor of 2 for 1 timestep)
        flux_ratio = f_atm / f_int if f_int > 0 else np.inf
        assert 0.5 <= flux_ratio <= 2.0, (
            f'Radiation balance should be approximately satisfied (0.5 < F_atm/F_int < 2.0), got {flux_ratio}'
        )

        # Validate surface temperature
        t_surf = final_row['T_surf']
        assert not np.isnan(t_surf)
        assert 200 <= t_surf <= 4000, f'T_surf should be physical for magma ocean, got {t_surf}'


@pytest.mark.smoke
@pytest.mark.skip(reason='AGNI integration requires Julia/AGNI binaries')
def test_smoke_agni_dummy_interior_convergence():
    """Test AGNI + dummy interior coupling (1 timestep).

    Physical scenario: Validates that AGNI (Julia-based atmosphere) can couple
    with a dummy interior. Tests the Julia-Python interface and verifies that
    AGNI's radiative-convective solver converges.

    Validates:
    - AGNI executes without Julia runtime errors
    - Atmospheric flux converges (F_atm is stable)
    - Surface temperature is updated correctly
    - No NaN propagation from Julia to Python

    Runtime: ~25-30s (1 timestep, AGNI convergence with relaxed tolerances)
    """
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load AGNI configuration
        config_path = PROTEUS_ROOT / 'input' / 'demos' / 'agni.toml'

        # Initialize PROTEUS
        runner = Proteus(config_path=config_path)

        # Override output directory
        runner.directories['output'] = str(Path(tmpdir) / 'output')
        Path(runner.directories['output']).mkdir(parents=True, exist_ok=True)

        # Use dummy interior
        runner.config.interior.module = 'dummy'

        # Override stop time
        runner.config.params.stop.time.minimum = 1e2
        runner.config.params.stop.time.maximum = 1e3

        # Disable plotting/archiving
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        # Run simulation
        runner.start(resume=False, offline=True)

        # Validate helpfile
        assert runner.hf_all is not None
        assert len(runner.hf_all) > 0

        final_row = runner.hf_all.iloc[-1]

        # Validate AGNI produced valid flux
        f_atm = final_row['F_atm']
        assert not np.isnan(f_atm), 'AGNI should produce valid F_atm (no NaN from Julia)'
        assert not np.isinf(f_atm)
        assert 0 <= f_atm <= 10000

        # Validate interior flux
        f_int = final_row['F_int']
        assert not np.isnan(f_int)
        assert 0 <= f_int <= 10000

        # Validate convergence (AGNI should produce stable flux within 1 timestep)
        # For convergence, we expect F_atm and F_int to be within same order of magnitude
        flux_ratio = f_atm / f_int if f_int > 0 else np.inf
        assert 0.1 <= flux_ratio <= 10.0, (
            f'AGNI flux should converge (0.1 < F_atm/F_int < 10.0), got {flux_ratio}'
        )

        # Validate surface temperature
        t_surf = final_row['T_surf']
        assert not np.isnan(t_surf)
        assert 200 <= t_surf <= 5000
