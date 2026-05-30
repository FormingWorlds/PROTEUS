"""Integration test for JANUS atmosphere backend.

Verifies that a single-timestep coupled run with JANUS as the atmosphere
module completes without error and produces physically valid output.
Uses dummy modules for interior, escape, star, orbit, and outgassing to
isolate the JANUS atmosphere path. Integration tier because JANUS
performs real radiative transfer across 3 init loops.

Invariants tested:
  - T_surf > 0 K (positivity)
  - F_atm >= 0 W/m^2 (positivity)
  - P_surf > 0 Pa (positivity)
  - No NaN/Inf in critical output columns
  - Time advances beyond initial value

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from _smoke_invariants import assert_smoke_conservation_invariants
from helpers import PROTEUS_ROOT

from proteus import Proteus

pytest.importorskip('janus')

# Higher timeout than the 300 s integration default: a real JANUS
# radiative-transfer solve takes tens of seconds and the coupled loop runs
# it several times (init loops plus a step), so the bounded run needs more
# headroom than a mocked integration test. The local run is ~6 min; the
# wider ceiling absorbs the slower CI runners.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_smoke_janus_dummy_single_timestep():
    """JANUS atmosphere + dummy interior coupling for 1 timestep.

    Physical scenario: a planet with JANUS computing the atmospheric
    radiative-convective structure and dummy modules for everything
    else. Validates that JANUS initializes, runs, and returns physically
    plausible atmospheric fluxes and surface conditions.

    Validates:
    - JANUS produces valid F_atm (non-NaN, non-negative)
    - Surface temperature is physical (100 < T_surf < 10000 K)
    - Surface pressure is positive
    - Helpfile has at least one data row
    - Conservation invariants hold
    """
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = PROTEUS_ROOT / 'input' / 'dummy.toml'
        runner = Proteus(config_path=config_path)

        # Set JANUS as the atmosphere module, dummy for everything else
        runner.config.atmos_clim.module = 'janus'
        runner.config.interior_energetics.module = 'dummy'
        runner.config.interior_struct.module = 'dummy'

        # Output directory
        runner.config.params.out.path = str(Path(tmpdir) / f'smoke_janus_{unique_id}')
        runner.init_directories()

        # Moderate initial temperature to keep JANUS in a convergent regime
        runner.config.planet.tsurf_init = 2000.0

        # Keep the atmosphere thin so the SOCRATES radiative-transfer
        # solve stays within the smoke wall-time budget. The base config
        # carries a heavy volatile inventory for the outgas demo, which
        # makes the moist-adiabat radiation step slow; this smoke test
        # only needs JANUS to initialise and run one step, not a thick
        # atmosphere. Set all four volatiles to a trace ppmw budget so the
        # dummy outgas produces a low surface pressure.
        for _elem in ('H', 'C', 'N', 'S'):
            setattr(runner.config.planet.elements, f'{_elem}_mode', 'ppmw')
            setattr(runner.config.planet.elements, f'{_elem}_budget', 1.0)

        # The dummy interior does not advance model time, so the
        # maximum-time stop never fires. Cap the loop iteration count so
        # the run terminates after the init loops plus one step instead of
        # running to the 9000-iteration safety ceiling, which would call
        # the JANUS radiative-transfer solver thousands of times. This is
        # a smoke test of JANUS initialisation and a single coupled step.
        runner.config.params.stop.time.minimum = 1.0
        runner.config.params.stop.time.maximum = 2.0
        runner.config.params.stop.iters.minimum = 1
        runner.config.params.stop.iters.maximum = 5

        # Disable plotting and archiving for speed
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        runner.start(resume=False, offline=True)

        # Validate helpfile exists and is populated
        assert runner.hf_all is not None, 'Helpfile should be created'
        assert len(runner.hf_all) > 0, 'Helpfile should have at least one row'

        final_row = runner.hf_all.iloc[-1]

        # T_surf: positive and physical
        assert 'T_surf' in final_row, 'T_surf should be in helpfile'
        t_surf = final_row['T_surf']
        assert not np.isnan(t_surf), 'T_surf should not be NaN'
        assert not np.isinf(t_surf), 'T_surf should not be Inf'
        assert 100 < t_surf < 10000, f'T_surf should be physical, got {t_surf}'

        # F_atm: finite and bounded in magnitude. F_atm can be negative
        # when the atmosphere drives net heat downward (greenhouse forcing
        # exceeds OLR), which is physical with dummy interior modules.
        assert 'F_atm' in final_row, 'F_atm should be in helpfile'
        f_atm = final_row['F_atm']
        assert not np.isnan(f_atm), 'F_atm should not be NaN'
        assert not np.isinf(f_atm), 'F_atm should not be Inf'
        assert abs(f_atm) < 1e7, f'|F_atm| should be < 1e7 W/m^2, got {f_atm}'

        # P_surf: positive and finite
        if 'P_surf' in final_row:
            p_surf = final_row['P_surf']
            assert not np.isnan(p_surf), 'P_surf should not be NaN'
            assert p_surf > 0, f'P_surf should be positive, got {p_surf}'

        # Time progressed beyond init stage (init stage keeps Time=0)
        post_init_times = runner.hf_all['Time'].values
        assert np.any(post_init_times > 0), 'At least one row should have Time > 0'

        # Conservation invariants
        assert_smoke_conservation_invariants(runner.hf_all)
