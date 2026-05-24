"""Smoke test: 2-timestep run with dummy backends.

Verifies that PROTEUS can advance through two consecutive timesteps
using all-dummy modules. Runs on every PR (smoke tier), unlike the
integration-tier multi-timestep tests that exercise real backends.

Invariants tested:
  - Time monotonicity: second row's Time > first row's Time
  - Helpfile has at least 2 data rows
  - T_surf, P_surf, F_atm positive in both rows
  - No NaN in critical columns

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

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(120)]


@pytest.mark.smoke
@pytest.mark.physics_invariant
def test_smoke_dummy_two_timesteps():
    """Run PROTEUS for 2 timesteps with all-dummy backends.

    Physical scenario: validates that the coupling infrastructure can
    advance through two consecutive iterations, updating hf_all with
    monotonically increasing time and physically valid state variables.

    Validates:
    - hf_all has at least 2 data rows (excluding the IC row if present)
    - Time is strictly increasing between rows
    - T_surf > 0, P_surf >= 0, F_atm >= 0 in every row
    - No NaN or Inf values in critical columns
    - Conservation invariants hold across both timesteps
    """
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = PROTEUS_ROOT / 'input' / 'dummy.toml'
        runner = Proteus(config_path=config_path)

        # Output directory
        runner.config.params.out.path = str(Path(tmpdir) / f'smoke_multi_{unique_id}')
        runner.init_directories()

        # Moderate initial temperature
        runner.config.planet.tsurf_init = 2000.0

        # Time window that allows 2 timesteps
        runner.config.params.stop.time.minimum = 1e2
        runner.config.params.stop.time.maximum = 1e5

        # Small initial dt so the solver takes at least 2 steps
        runner.config.params.dt.initial = 1e3
        runner.config.params.dt.minimum = 1e2
        runner.config.params.dt.maximum = 1e4

        # Disable plotting and archiving
        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        runner.start(resume=False, offline=True)

        # Validate helpfile
        assert runner.hf_all is not None, 'Helpfile should be created'
        n_rows = len(runner.hf_all)
        assert n_rows >= 2, f'Helpfile should have >= 2 rows, got {n_rows}'

        # PROTEUS keeps Time=0.0 during the init stage (first init_loops
        # iterations), then advances time in the science stage. Filter to
        # post-init rows for the monotonicity check.
        times = runner.hf_all['Time'].values
        post_init = times[times > 0]
        assert len(post_init) >= 2, (
            f'Need at least 2 post-init rows with Time > 0, got {len(post_init)}'
        )
        for i in range(1, len(post_init)):
            assert post_init[i] > post_init[i - 1], (
                f'Time must be strictly increasing in post-init rows: '
                f'row {i - 1}={post_init[i - 1]}, row {i}={post_init[i]}'
            )

        # Check critical columns in every row
        critical_cols = ['T_surf', 'F_atm']
        for col in critical_cols:
            if col not in runner.hf_all.columns:
                continue
            vals = runner.hf_all[col].values
            assert np.all(np.isfinite(vals)), f'{col} contains NaN or Inf'
            assert np.all(vals >= 0), f'{col} contains negative values'

        # T_surf positivity (strict) in every row
        if 'T_surf' in runner.hf_all.columns:
            t_surf_vals = runner.hf_all['T_surf'].values
            assert np.all(t_surf_vals > 0), 'T_surf must be strictly positive in all rows'
            # Physical range: 50 to 10000 K for magma ocean era
            assert np.all(t_surf_vals < 10000), (
                f'T_surf exceeds 10000 K: max={np.max(t_surf_vals)}'
            )

        # P_surf non-negative in every row (can be zero for dummy outgas)
        if 'P_surf' in runner.hf_all.columns:
            p_surf_vals = runner.hf_all['P_surf'].values
            assert np.all(np.isfinite(p_surf_vals)), 'P_surf contains NaN or Inf'
            assert np.all(p_surf_vals >= 0), 'P_surf must be non-negative'

        # F_atm finite and non-negative in every row
        if 'F_atm' in runner.hf_all.columns:
            f_atm_vals = runner.hf_all['F_atm'].values
            assert np.all(np.isfinite(f_atm_vals)), 'F_atm contains NaN or Inf'
            assert np.all(f_atm_vals >= 0), 'F_atm must be non-negative'

        # Conservation invariants
        assert_smoke_conservation_invariants(runner.hf_all)
