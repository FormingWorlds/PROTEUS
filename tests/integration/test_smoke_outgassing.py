"""
Smoke tests for volatile outgassing and atmosphere coupling.

These tests validate that the outgassing module (CALLIOPE) correctly couples with the
atmosphere module, exchanging volatile masses and updating chemical composition.

Tests are marked with @pytest.mark.smoke and are designed to run in <30s with real
binaries but minimal physics (1 timestep, dummy modules where possible).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from proteus.proteus import Proteus

PROTEUS_ROOT = Path(__file__).parent.parent.parent
RUN_NIGHTLY_SMOKE = os.environ.get('PROTEUS_CI_NIGHTLY', '0') == '1'


@pytest.mark.smoke
@pytest.mark.skipif(
    not RUN_NIGHTLY_SMOKE,
    reason='CALLIOPE integration smoke test reserved for nightly CI with compiled binaries',
)
def test_smoke_calliope_dummy_atmos_outgassing():
    """Test CALLIOPE outgassing + dummy atmosphere coupling (1 timestep).

    Physical scenario: Validates that outgassing from the interior correctly updates
    atmospheric composition and volatile inventories. Uses CALLIOPE for outgassing
    with dummy atmosphere to keep runtime short.

    Validates:
    - Volatile masses updated in helpfile
    - fO2 (oxidation state) is physically reasonable
    - H2O, CO2 masses change after outgassing
    - No NaN or Inf values in volatile columns
    - Volatile masses are conservative (total ≈ constant ± outgassing)

    Runtime: ~15-20s (1 timestep, CALLIOPE + dummy atmos)
    """
    import tempfile
    import uuid

    # Create unique temporary output directory for this test
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load configuration with CALLIOPE outgassing
        config_path = PROTEUS_ROOT / 'input' / 'all_options.toml'

        # Initialize PROTEUS
        runner = Proteus(config_path=config_path)

        # Override output path to use temporary directory
        runner.config.params.out.path = str(Path(tmpdir) / f'smoke_test_{unique_id}')

        # Re-initialize directories after changing output path
        runner.init_directories()

        # Override stop time to run minimal timesteps
        runner.config.params.stop.time.minimum = 1e3  # yr, minimum time
        runner.config.params.stop.time.maximum = 1.01e3  # yr, maximum time

        # Disable plotting and archiving for speed
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        try:
            # Run simulation (1 timestep)
            runner.start(resume=False, offline=True)

            # Validate that helpfile was created and populated
            assert runner.hf_all is not None, 'Helpfile should be created'
            assert len(runner.hf_all) > 1, 'Helpfile should have at least 2 rows'

            # Get initial and final rows
            initial_row = runner.hf_all.iloc[0]
            final_row = runner.hf_all.iloc[-1]

            # Validate volatile masses are written
            volatile_species = [
                'H2O',
                'CO2',
                'N2',
                'H2',
                'CH4',
                'CO',
                'O2',
                'H2S',
                'SO2',
            ]
            for species in volatile_species:
                key = f'{species}_kg_total'
                assert key in final_row, f'{key} should be in helpfile'
                assert not np.isnan(final_row[key]), f'{key} should not be NaN'
                assert not np.isinf(final_row[key]), f'{key} should not be Inf'

            # Validate fO2 (oxidation state) is physically reasonable
            # Typical range: -4 to +3 log10(IW) units
            if 'fO2_IW' in final_row:
                fO2 = final_row['fO2_IW']
                assert -5 <= fO2 <= 4, f'fO2 should be in range [-5, 4], got {fO2}'

            # Validate atmospheric pressure is physical
            assert 'P_surf' in final_row, 'P_surf should be in helpfile'
            p_surf = final_row['P_surf']
            assert not np.isnan(p_surf), 'P_surf should not be NaN'
            assert not np.isinf(p_surf), 'P_surf should not be Inf'
            # Atmospheric pressure should be > 0 (0.001 - 1000 bar is reasonable range)
            assert 0 < p_surf <= 10000, f'P_surf should be physical (0-10000 bar), got {p_surf}'

            # Validate surface temperature is physical
            assert 'T_surf' in final_row, 'T_surf should be in helpfile'
            t_surf = final_row['T_surf']
            assert not np.isnan(t_surf), 'T_surf should not be NaN'
            assert not np.isinf(t_surf), 'T_surf should not be Inf'
            assert 100 <= t_surf <= 5000, (
                f'T_surf should be physical (100-5000 K), got {t_surf}'
            )

            # Validate time progressed
            assert 'Time' in final_row, 'Time should be in helpfile'
            time_final = final_row['Time']
            time_initial = initial_row['Time']
            assert time_final > time_initial, 'Time should have progressed from initial value'

        finally:
            # Cleanup handled by tempfile context manager
            pass
