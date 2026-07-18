# AGNI grey-gas atmosphere coupled to the dummy interior.
#
# Integration tier: this drives the Julia/AGNI solver for a coupled step and
# needs the 300 s budget, where the smoke cases in test_smoke_atmos_interior.py
# finish in under 30 s. A file carries one tier, so that the tier filters
# select every test exactly once.
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

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

# Run the AGNI coupling only in nightly CI (requires compiled binaries)
RUN_NIGHTLY_SMOKE = os.environ.get('PROTEUS_CI_NIGHTLY', '0') == '1'


@pytest.mark.physics_invariant
@pytest.mark.skipif(
    not RUN_NIGHTLY_SMOKE,
    reason='AGNI coupling test requires Julia/AGNI binaries (nightly only)',
)
def test_agni_greygas_dummy_interior_coupling():
    """AGNI (grey gas) + dummy interior coupled step.

    Physical scenario: AGNI solves radiative-convective balance for a
    thick outgassed atmosphere above a magma-ocean surface held by the
    dummy interior. The grey-gas scheme needs no SOCRATES spectral file
    or stellar spectrum, so the test isolates the Julia-Python
    interface and the AGNI solver itself from the data pipeline.

    Validates:
    - AGNI boots and executes without Julia runtime errors
    - No NaN propagation from Julia to Python (F_atm, T_surf)
    - The transit photosphere sits above the surface (0 < p_obs <
      P_surf) and emits cooler than the magma surface (T_obs <
      T_surf); the dummy atmosphere copies the surface values into
      both observables, so the strict inequalities also guard against
      silently running the dummy module instead of AGNI
    - Conservation invariants across the coupled step

    Integration tier: the Julia runtime boot and AGNI package load
    dominate the wall time; the grey-gas solve itself is seconds.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = PROTEUS_ROOT / 'input' / 'dummy.toml'

        runner = Proteus(config_path=config_path)

        # AGNI atmosphere over dummy interior modules. Grey-gas mode
        # bypasses the spectral-file and stellar-spectrum machinery.
        # The module must be set before init_directories so the
        # SOCRATES directory (RAD_DIR) is registered for the banner.
        runner.config.atmos_clim.module = 'agni'
        runner.config.atmos_clim.agni.spectral_file = 'greygas'
        runner.config.interior_energetics.module = 'dummy'
        runner.config.interior_struct.module = 'dummy'

        # Override output path via config (proper way - triggers init_directories)
        runner.config.params.out.path = str(Path(tmpdir) / 'output')
        runner.init_directories()

        # Moderate initial temperature to keep AGNI in a convergent regime
        runner.config.planet.tsurf_init = 2000.0

        # The dummy interior does not advance model time, so cap the
        # iteration count to bound the number of AGNI solves (init
        # loops plus one coupled step) instead of relying on the
        # maximum-time stop.
        runner.config.params.stop.time.minimum = 1.0
        runner.config.params.stop.time.maximum = 2.0
        runner.config.params.stop.iters.minimum = 1
        runner.config.params.stop.iters.maximum = 5

        # Disable plotting/archiving
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        runner.start(resume=False, offline=True)

        # Validate helpfile
        assert runner.hf_all is not None
        assert len(runner.hf_all) > 0

        final_row = runner.hf_all.iloc[-1]

        # F_atm: finite and bounded. Negative values are physical when
        # greenhouse forcing exceeds OLR over a dummy interior.
        f_atm = final_row['F_atm']
        assert not np.isnan(f_atm), 'AGNI should produce valid F_atm (no NaN from Julia)'
        assert not np.isinf(f_atm)
        assert abs(f_atm) < 1e7, f'|F_atm| should be < 1e7 W/m^2, got {f_atm}'

        # F_int: the dummy interior mirrors F_atm, so it must be finite
        # whenever the coupled step completed.
        f_int = final_row['F_int']
        assert not np.isnan(f_int)
        assert abs(f_int) < 1e7

        # The transit photosphere must sit above the surface: AGNI
        # derives p_obs from the optical-depth profile. The dummy
        # atmosphere copies P_surf into p_obs exactly, so the strict
        # upper bound discriminates a real AGNI solve from a silent
        # fallback to the dummy module; the lower bound catches a NaN
        # or unwritten observable.
        p_obs = final_row['p_obs']
        p_surf = final_row['P_surf']
        assert 0.0 < p_obs < p_surf, (
            f'photosphere must sit above the surface, p_obs={p_obs}, P_surf={p_surf} bar'
        )

        # The emitting level is cooler than the magma surface beneath
        # a greenhouse atmosphere; the dummy module copies T_surf.
        assert final_row['T_obs'] < final_row['T_surf']

        # Outgoing longwave is positive for any atmosphere above 0 K.
        assert final_row['F_olr'] > 0.0

        # Surface temperature physical
        t_surf = final_row['T_surf']
        assert not np.isnan(t_surf)
        assert 50 <= t_surf <= 10000

        # Conservation invariants across the coupled step
        assert_smoke_conservation_invariants(runner.hf_all)
