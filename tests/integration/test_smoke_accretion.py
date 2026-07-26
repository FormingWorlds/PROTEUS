"""Smoke test: a giant impact applied inside the coupled loop.

Every other accretion test runs a helper in isolation with the structure solve,
the interior solver and the impact timeline mocked. This one enables the
accretion module and runs the real loop across a scheduled impact, so the
wiring those tests cannot reach is exercised: the timestep clamp landing on the
impact time, the handler firing once at that time, the ordering that puts the
atmospheric strip before escape and outgassing, and the runtime mass-closure
assertion seeing the grown planet.

The analytical accretion module is used rather than a timeline file or the
dynamical model, so the test needs no fixture data and no optional dependency
and still applies the full impact physics.

Invariants tested:
  - the planet's mass grows, by the impactor rock the timeline specifies
  - the impact lands once, inside the simulated interval
  - the accreted-rock ledger a resume reads back is written and monotonic
  - M_planet stays consistent with M_int + M_ele after the impact
  - the run does not trip the runtime M_atm <= M_planet assertion

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus
from proteus.utils.constants import M_earth

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(120)]


@pytest.mark.smoke
@pytest.mark.physics_invariant
def test_smoke_accretion_impact_lands_inside_the_coupled_loop():
    """A scheduled impact grows the planet while the run stays self-consistent.

    Physical scenario: an all-dummy planet accretes one giant impact partway
    through a short run. The impact adds rock, the structure is re-solved
    against the grown mass, and the mantle is re-melted, all inside the loop
    rather than in a helper called directly.

    Validates:
    - the interior mass anchor grows by the delivered rock, once
    - the impact time falls inside the simulated interval, so the schedule and
      the timestep clamp actually met
    - M_accreted_rock is written, non-decreasing, and ends at the delivered mass
    - M_planet equals M_int + M_ele on every row, including the impact row
    - no NaN reaches the mass columns
    """
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = PROTEUS_ROOT / 'input' / 'dummy.toml'
        runner = Proteus(config_path=config_path)

        runner.config.params.out.path = str(Path(tmpdir) / f'smoke_accretion_{unique_id}')
        runner.init_directories()

        runner.config.planet.tsurf_init = 2000.0

        # A window that comfortably brackets the single impact below. The
        # timestep floor is far smaller than the shortening the clamp needs, so
        # a step can land exactly on the impact time; leaving the floor above
        # that shortening would let the run overshoot and still look correct.
        runner.config.params.stop.time.minimum = 1e2
        runner.config.params.stop.time.maximum = 1e5
        runner.config.params.dt.initial = 1e3
        runner.config.params.dt.minimum = 1e0
        runner.config.params.dt.maximum = 1e4

        runner.config.params.out.plot_mod = 0
        runner.config.params.out.write_mod = 1
        runner.config.params.out.archive_mod = 'none'

        # One impact delivering 0.1 M_earth. num_impacts = 1 puts the whole
        # budget in that single impact, so the expected growth is exact rather
        # than a share of an exponential. The time sits early in the run: the
        # all-dummy planet solidifies and stops the run within about 1e4 yr, so
        # a later impact would never be reached.
        delivered = 0.1
        impact_time = 4.0e3
        runner.config.accretion.module = 'dummy'
        runner.config.accretion.dummy.num_impacts = 1
        runner.config.accretion.dummy.mass_accreted = delivered
        runner.config.accretion.dummy.time_last = impact_time
        runner.config.accretion.dummy.timescale = 3.0e3
        runner.config.accretion.dummy.eccentricity = 0.05
        runner.config.accretion.impactor_volatiles = 'dry'

        # Strip a fixed fraction of the atmosphere as well, so the ordering
        # against escape and outgassing is exercised rather than skipped. The
        # constant module is used because it needs no optional dependency.
        atmloss = 0.25
        runner.config.accretion.atmloss_module = 'constant'
        runner.config.accretion.atmloss_frac = atmloss

        mass_before = runner.config.planet.mass_tot

        runner.start(resume=False, offline=True)

        assert runner.hf_all is not None, 'Helpfile should be created'
        hf = runner.hf_all

        # The impact is inside the simulated interval, so the schedule and the
        # run actually overlapped. Without this the growth checks below could
        # pass vacuously on a run that ended before the impact.
        assert hf['Time'].max() > impact_time, (
            f'Run ended at {hf["Time"].max():.3e} yr, before the impact at '
            f'{impact_time:.3e} yr; the test would not have exercised anything'
        )

        # A step lands exactly on the impact time. The adaptive controller would
        # not choose that time on its own, so this is the timestep clamp doing
        # its job: without it the impact fires on whichever step first overshoots
        # and the planet grows at the wrong moment.
        times = hf['Time'].values
        assert np.any(np.isclose(times, impact_time, rtol=0, atol=1e-6)), (
            f'no step landed on the impact time {impact_time:.4e} yr; '
            f'nearest was {times[np.argmin(np.abs(times - impact_time))]:.6e} yr'
        )

        # A dry impactor delivers no volatiles, so every kilogram of the
        # impactor is rock and the anchor grows by exactly the delivered mass.
        assert runner.config.planet.mass_tot == pytest.approx(mass_before + delivered, rel=1e-6)

        # The ledger a resumed run reads back was written, never decreases, and
        # ends at the delivered rock. A handler that applied the impact twice
        # would overshoot it, and one that never fired would leave it at zero.
        assert 'M_accreted_rock' in hf.columns, (
            'the accreted-rock ledger must be persisted, or a resume cannot '
            'rebuild the planet the impacts grew'
        )
        ledger = hf['M_accreted_rock'].fillna(0.0).values
        assert np.all(np.diff(ledger) >= 0.0), 'the accreted-rock ledger must not decrease'
        assert ledger[-1] == pytest.approx(delivered * M_earth, rel=1e-6)
        # Discrimination: a double application would land at twice this value,
        # which is a hundred thousand times the tolerance away.
        assert abs(2.0 * delivered * M_earth - ledger[-1]) > 0.5 * ledger[-1]

        # The whole-planet mass agrees with its parts on every row, including
        # the impact row where the strip and the delivery change the budgets
        # after the structure solve has already written both.
        for column in ('M_planet', 'M_int', 'M_ele'):
            assert column in hf.columns, f'{column} missing from the helpfile'
            assert np.all(np.isfinite(hf[column].values)), f'{column} contains NaN or Inf'

        np.testing.assert_allclose(
            hf['M_planet'].values,
            hf['M_int'].values + hf['M_ele'].values,
            rtol=1e-9,
            err_msg='M_planet must equal M_int + M_ele on every row',
        )

        # The planet only ever gains mass here, so the interior mass is
        # non-decreasing and strictly larger at the end than at the start.
        m_int = hf['M_int'].values
        assert m_int[-1] > m_int[0], 'the interior mass must grow across the impact'

        # The atmospheric strip ran and was booked into the loss ledger the
        # desiccation criterion audits. Without this the ordering claim in this
        # file's docstring would be untested, because a strip of zero exercises
        # nothing about where the strip sits relative to escape and outgassing.
        assert 'esc_kg_cumulative' in hf.columns
        ledger = hf['esc_kg_cumulative'].fillna(0.0).values
        assert np.all(np.diff(ledger) >= 0.0), 'the loss ledger must not decrease'
        assert ledger[-1] > 0.0, (
            'the impact strip removed nothing, so the strip path was not exercised'
        )

        # The strip is bounded by the atmosphere it is drawn from: it can never
        # remove more than the whole atmosphere, whatever the fraction asks for.
        assert 'M_atm' in hf.columns
        assert np.all(hf['M_atm'].values >= 0.0), 'atmospheric mass must stay non-negative'
        assert np.all(hf['M_atm'].values <= hf['M_planet'].values), (
            'the atmosphere cannot outweigh the planet carrying it'
        )
