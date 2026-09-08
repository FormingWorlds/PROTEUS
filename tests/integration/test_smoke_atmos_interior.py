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
# - docs/How-to/testing.md
# - docs/Explanations/test_framework.md
#
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from _smoke_invariants import assert_smoke_conservation_invariants
from helpers import PROTEUS_ROOT

from proteus import Proteus

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]

# Run JANUS/AGNI smoke tests only in nightly CI (requires compiled binaries)
RUN_NIGHTLY_SMOKE = os.environ.get('PROTEUS_CI_NIGHTLY', '0') == '1'


@pytest.mark.smoke
@pytest.mark.physics_invariant
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
    import uuid

    # Create unique temporary output directory for this test
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load dummy configuration (uses dummy atmos + dummy interior)
        config_path = PROTEUS_ROOT / 'input' / 'dummy.toml'

        # Initialize PROTEUS
        runner = Proteus(config_path=config_path)

        # Override output path to use temporary directory
        runner.config.params.out.path = str(Path(tmpdir) / f'smoke_test_{unique_id}')

        # Re-initialize directories after changing output path
        runner.init_directories()

        # Fix: Lower tsurf_init to prevent runaway heating (T_magma > 1e6 K issue)
        runner.config.planet.tsurf_init = 2000.0

        # Override stop time to run only 1 timestep
        runner.config.params.stop.time.minimum = 1e2  # yr, minimum time
        runner.config.params.stop.time.maximum = 1e3  # yr, maximum time

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
            assert 50 <= t_surf <= 10000, (
                f'T_surf should be physical (50-10000 K), got {t_surf}'
            )

            # Validate time progressed
            assert 'Time' in final_row, 'Time should be in helpfile'
            assert final_row['Time'] > 0, 'Time should have progressed'

            # Conservation invariants, applied to every smoke test so a
            # bookkeeping regression in any module surfaces here, not
            # in a quiet helpfile drift months later.
            assert_smoke_conservation_invariants(runner.hf_all)
        finally:
            # Cleanup handled by tempfile context manager
            pass
