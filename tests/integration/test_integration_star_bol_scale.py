"""Integration test of the bolometric scaling window across a dummy PROTEUS run."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

import proteus.star.dummy as star
from proteus import Proteus

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


@pytest.mark.physics_invariant
def test_bol_scale_window_applies_across_dummy_run():
    """A bol_scale window spanning the whole run must stay active for
    every post-init row, and the scaled F_ins must match an independently
    recomputed unscaled instellation times the configured factor.
    """
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = PROTEUS_ROOT / 'input' / 'dummy.toml'
        runner = Proteus(config_path=config_path)

        runner.config.params.out.path = str(Path(tmpdir) / f'bolscale_{unique_id}')
        runner.init_directories()

        runner.config.planet.tsurf_init = 2000.0

        # Short run: a couple of small timesteps.
        runner.config.params.stop.time.minimum = 1e2
        runner.config.params.stop.time.maximum = 1e5
        runner.config.params.dt.initial = 1e3
        runner.config.params.dt.minimum = 1e2
        runner.config.params.dt.maximum = 1e4

        # Bolometric-scaling window: opens exactly at age_ini and stays
        # open for far longer (2 Gyr) than the run can possibly advance
        # (stop.time.maximum=1e5 yr), so the window covers every row.
        runner.config.star.bol_scale = 2.0

        # Set duration before start: valid_bol_scale_start performs its checks
        # when bol_scale_start is assigned.
        runner.config.star.bol_scale_duration = 2.0
        runner.config.star.bol_scale_start = runner.config.star.age_ini

        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        runner.start(resume=False, offline=True)

        assert runner.hf_all is not None, 'Helpfile should be created'
        times = runner.hf_all['Time'].values
        post_init = runner.hf_all[times > 0]
        assert len(post_init) >= 2, (
            f'Need at least 2 post-init rows with Time > 0, got {len(post_init)}'
        )

        # The window covers the whole run: bol_scale must read the
        # configured 2.0 in every post-init row, not just the first.
        bol_scale_vals = post_init['bol_scale'].values
        np.testing.assert_allclose(bol_scale_vals, 2.0, rtol=1e-12)

        # Unscaled reference at the flux-weighted distance a (1 - e^2)^(1/4),
        # written out here, not the time-averaged separation a (1 + e^2 / 2).
        def _flux_distance(row):
            return row['semimajorax'] * (1.0 - row['eccentricity'] ** 2) ** 0.25

        for _, row in post_init.iterrows():
            s0_unscaled = star.calc_instellation(
                runner.config.star.dummy.Teff, row['R_star'], _flux_distance(row)
            )
            assert row['F_ins'] == pytest.approx(row['bol_scale'] * s0_unscaled, rel=1e-9)

        # Discrimination: the scaled flux must actually differ from unscaled
        last_row = post_init.iloc[-1]
        s0_last_unscaled = star.calc_instellation(
            runner.config.star.dummy.Teff, last_row['R_star'], _flux_distance(last_row)
        )
        assert last_row['F_ins'] > s0_last_unscaled
        # The run is eccentric, so the two averages genuinely disagree and
        # this check discriminates between them rather than coinciding.
        assert last_row['eccentricity'] > 0.0
        assert _flux_distance(last_row) < last_row['separation']
