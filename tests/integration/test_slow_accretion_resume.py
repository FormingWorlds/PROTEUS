"""Slow test: a run stopped on a giant impact resumes without losing the re-melt.

The interior writes its snapshot while a step is solved, which is before the
impacts falling in that step are applied at the end of it. A run stopped on
such a step leaves a snapshot of the mantle from before the re-melt beside a
helpfile row that already carries the impact's mass and orbit. Resuming from
that pair would restore a mantle the impact had melted while treating the
impact as already applied, so the stale snapshot is discarded and the resume
walks back to the last step that has a complete pair, applying the impact
again in full.

Nothing lighter reaches this. Only Aragog writes an interior snapshot, so on
the dummy interior the discard returns immediately and
``select_resumable_snapshot`` has no interior half to walk back over: the
whole mechanism is invisible. This file therefore runs the real Aragog
interior across a scheduled impact, stops on it, and resumes, with nothing in
the resume path mocked. That is what puts it in the slow tier.

Contract clauses exercised:

- The step that lands the impact leaves no interior snapshot of its own,
  while the snapshot from the previous step survives to be resumed from.
- The resume is not told where to land. ``select_resumable_snapshot`` runs
  unmocked, finds the trailing row unbacked, and truncates the helpfile to
  the last complete pair.
- The impact is applied exactly once across the two legs, in the ledger the
  resume reads back and in the planet mass the configuration carries.
- The mantle the resumed run carries forward is the one the impact melted,
  not the cooler one the discarded snapshot held.

Scope. The interior runs its production backend; every other module is on its
dummy backend, which keeps two Aragog legs inside the tier budget and leaves
the interior snapshot as the only state channel under test. The atmosphere is
dummy and writes no ``_atm.nc``, so the resume's atmosphere half imposes no
constraint and the interior half is what decides where the run lands.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus
from proteus.utils.constants import M_earth
from proteus.utils.data import download_sufficient_data

# Slow tier. Two real Aragog legs, about 10 minutes locally, most of it in
# the two solves of the molten initial condition: one at the start and one
# for the grown planet at the impact. The 3600 s ceiling leaves room for
# slower CI runners.
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

CONFIG = PROTEUS_ROOT / 'input' / 'dummy.toml'

# Time of the single scheduled impact [yr]. The first leg stops here, so the
# impact lands on the last step the run takes and its snapshot is the one a
# resume would otherwise reach for.
IMPACT_TIME = 3.0e2

# Mass the impactor delivers [M_earth]. A dry impactor carries no volatiles,
# so all of it is rock and the expected ledger value is exact.
DELIVERED = 0.1

# Stop time for the resumed leg [yr]. Far enough past the impact for the
# mantle it melted to be carried forward over several steps, close enough to
# keep the second leg to a handful of Aragog solves.
LEG2_STOP_TIME = 5.0e2

# Step limits [yr]. The ceiling is a third of the impact time, so the run
# takes several steps to reach the impact and the step before it carries a
# snapshot of its own for the resume to land on.
DT_INITIAL = 1.0e2
DT_MINIMUM = 1.0e1
DT_MAXIMUM = 1.0e2

# Melting-curve folder in FWL_DATA. Required because the dummy structure
# module selects the shipped EOS rather than the PALEOS tables Zalmoxis
# generates, and Aragog refuses to guess a melting curve for it.
MELTING_DIR = 'Monteux-600'


# Scratch directories created by Proteus construction, cleared after the
# test. ``set_directories`` allocates one per runner and nothing in the
# framework removes it.
_RUNNER_TEMP_DIRS: list[str] = []


@pytest.fixture(autouse=True)
def _remove_runner_temp_dirs():
    """Delete the scratch directories this file's runners allocate."""
    _RUNNER_TEMP_DIRS.clear()
    yield
    for path in _RUNNER_TEMP_DIRS:
        shutil.rmtree(path, ignore_errors=True)
    _RUNNER_TEMP_DIRS.clear()


def _config_with_output_path(output_dir):
    """Write a copy of the dummy config that already names its output path.

    Building the runner from a config that carries the run directory avoids a
    second ``init_directories`` call, and with it a second scratch directory
    per runner.
    """
    text = CONFIG.read_text()
    patched = text.replace('path = "auto"', f'path = "{output_dir}"', 1)
    assert patched != text, 'dummy config no longer carries the auto output path'
    destination = output_dir.parent / f'{output_dir.name}_config.toml'
    destination.write_text(patched)
    return destination


def _make_runner(output_dir, stop_time):
    """Build an Aragog runner with one giant impact scheduled.

    Parameters
    ----------
    output_dir : pathlib.Path
        Run directory, shared by both legs so the second resumes the first.
    stop_time : float
        Maximum simulation time for this leg [yr].

    Returns
    -------
    Proteus
        Configured runner; the caller starts it.
    """
    runner = Proteus(config_path=_config_with_output_path(output_dir))
    _RUNNER_TEMP_DIRS.append(runner.directories['temp'])

    runner.config.interior_energetics.module = 'aragog'
    runner.config.interior_struct.melting_dir = MELTING_DIR
    # The re-melt re-applies the interior initial condition, so it only adds
    # heat if that condition is molten for the grown planet. This one is, at
    # any mass and on any melting curve, which is what makes the re-melt
    # visible as a warming step below.
    runner.config.planet.temperature_mode = 'liquidus_super'

    runner.config.params.stop.solid.enabled = False
    runner.config.params.stop.time.minimum = 0.0
    runner.config.params.stop.time.maximum = stop_time
    runner.config.params.stop.iters.minimum = 1
    runner.config.params.stop.iters.maximum = 200

    runner.config.params.dt.initial = DT_INITIAL
    runner.config.params.dt.minimum = DT_MINIMUM
    runner.config.params.dt.maximum = DT_MAXIMUM

    # A snapshot on every iteration, so the impact step writes one and the
    # step before it leaves the pair the resume falls back to.
    runner.config.params.out.write_mod = 1
    runner.config.params.out.dt_write_rel = 0.0
    # None, not 0: the schema reads 0 as "plot once at completion", and the
    # end-of-run block only skips plotting when this is None.
    runner.config.params.out.plot_mod = None
    # Loose files, not tar archives, so the snapshots this test reads stay
    # where the interior wrote them.
    runner.config.params.out.archive_mod = 'none'

    runner.config.accretion.module = 'dummy'
    runner.config.accretion.dummy.num_impacts = 1
    runner.config.accretion.dummy.mass_accreted = DELIVERED
    runner.config.accretion.dummy.time_last = IMPACT_TIME
    runner.config.accretion.dummy.timescale = 3.0e3
    runner.config.accretion.dummy.eccentricity = 0.05
    runner.config.accretion.impactor_volatiles = 'dry'

    return runner


def _snapshot_times(output_dir):
    """Simulation times of the interior snapshots on disk [yr], ascending."""
    return sorted(
        int(path.name.split('_int.nc')[0]) for path in (output_dir / 'data').glob('*_int.nc')
    )


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_a_run_stopped_on_an_impact_resumes_from_before_it(tmp_path):
    """A resumed run re-applies the impact its last snapshot no longer described.

    Physical scenario: a magma-ocean planet takes one giant impact, which
    delivers rock and re-melts the mantle, and the run stops on that step. A
    second process resumes the run from the output directory, as
    ``proteus start -r`` does.

    Verifies, on the leg that stops on the impact:
    - The last stored row is the impact row, carrying the delivered rock.
    - No interior snapshot remains for that time, because the one the step
      wrote describes the mantle from before the re-melt.
    - An earlier snapshot survives, so the run is left resumable rather than
      stripped of its interior state.
    - The re-melt heated the mantle, which is what makes the discarded
      snapshot stale rather than merely redundant.

    Verifies, on the resumed leg:
    - The run advances past the impact time and stores no duplicate times, so
      the truncated rows were recomputed rather than appended alongside.
    - The impact is applied exactly once across both legs, in the accreted
      rock ledger and in the planet mass the configuration carries.
    - The mantle carried past the impact is hotter than the mantle before it,
      in a run that cools on every other step. A resume that had loaded the
      discarded snapshot would continue from the cooler pre-re-melt state
      while the helpfile row claimed the impact had landed, so this is the
      check that the walk-back happened.
    - The resumed leg discards its own impact-step snapshot in turn, so the
      behaviour is a property of the step rather than of the first run.
    """
    outdir = tmp_path / 'accretion_resume'
    outdir.mkdir()

    leg1 = _make_runner(outdir, IMPACT_TIME)
    mass_before = float(leg1.config.planet.mass_tot)

    # Aragog needs its lookup tables and melting curves. Fetch them here
    # rather than during the run, which is started offline so a slow network
    # cannot stall the solve.
    leg1.config.params.offline = False
    try:
        download_sufficient_data(leg1.config, clean=False)
    finally:
        leg1.config.params.offline = True

    leg1.start(resume=False, offline=True)
    stored = leg1.hf_all.copy()

    # The first leg stopped on the impact, so the impact row is the last one
    # and the snapshot beside it is the stale one. Without this the checks
    # below would be about an ordinary trailing row.
    assert float(stored.iloc[-1]['Time']) == pytest.approx(IMPACT_TIME, rel=0, abs=1e-6), (
        f'first leg ended at {float(stored.iloc[-1]["Time"]):.6e} yr, not on the '
        f'impact at {IMPACT_TIME:.6e} yr; the resume would not start from an impact step'
    )
    delivered_kg = DELIVERED * M_earth
    assert float(stored.iloc[-1]['M_accreted_rock']) == pytest.approx(delivered_kg, rel=1e-6), (
        'the last stored row does not carry the impactor rock, so the impact '
        'did not land on the step the run stopped on'
    )

    # A resumable history needs more rows than the initialisation loops write.
    assert len(stored) > leg1.loops['init_loops'] + 1, (
        f'first leg produced only {len(stored)} rows, too short to resume from'
    )

    # The impact step left no snapshot, and an older one survived it.
    snapshots = _snapshot_times(outdir)
    assert int(IMPACT_TIME) not in snapshots, (
        f'the impact step kept its interior snapshot (times on disk: {snapshots}); '
        'a resume would load a mantle the impact had already melted'
    )
    assert any(time < IMPACT_TIME for time in snapshots), (
        f'no interior snapshot older than the impact survived (times on disk: '
        f'{snapshots}); the run has nothing to walk back to'
    )

    # The mantle cools into the impact and the impact warms it. That contrast
    # is what makes the discarded snapshot stale rather than merely redundant,
    # and it is the signal the resume checks below read. Only the three rows
    # before the impact are taken: the opening step settles the solver against
    # the coupled surface flux and can move either way before the cooling
    # trend sets in.
    t_magma = stored['T_magma'].to_numpy()
    cooling_before = t_magma[len(stored) - 4 : len(stored) - 1]
    assert np.all(np.diff(cooling_before) < 0.0), (
        f'the mantle was not cooling into the impact (last pre-impact values '
        f'{np.round(cooling_before, 1).tolist()} K), so a warming step is not '
        'the anomaly this test reads it as'
    )
    assert t_magma[-1] > t_magma[-2], (
        f'the impact step ended at {t_magma[-1]:.1f} K against {t_magma[-2]:.1f} K '
        'before it, so the re-melt added no heat and the resume checks below '
        'could not tell the two mantles apart'
    )
    pre_impact_T = float(t_magma[-2])

    # Resume. Nothing in the resume path is mocked: the walk-back is
    # select_resumable_snapshot reading what the first leg left on disk.
    leg2 = _make_runner(outdir, LEG2_STOP_TIME)
    leg2.start(resume=True, offline=True)
    resumed = leg2.hf_all

    times = resumed['Time'].to_numpy()
    assert times.max() > IMPACT_TIME, (
        f'the resumed run ended at {times.max():.6e} yr, no further than the '
        'impact; nothing was carried past the re-melt'
    )
    # The truncated rows were recomputed in place. A resume that appended
    # instead would leave two rows at the same time.
    evolution = times[times > 0.0]
    assert len(np.unique(evolution)) == len(evolution), (
        'the resumed helpfile stores a simulation time twice, so the rows the '
        'resume walked back over were appended rather than recomputed'
    )

    # The impact landed exactly once across both legs.
    ledger = resumed['M_accreted_rock'].fillna(0.0).to_numpy()
    assert np.all(np.diff(ledger) >= 0.0), 'the accreted-rock ledger must not decrease'
    assert ledger[-1] == pytest.approx(delivered_kg, rel=1e-6), (
        f'the run ended with {ledger[-1]:.6e} kg of accreted rock against the '
        f'{delivered_kg:.6e} kg one impact delivers; the resume applied it '
        'twice or not at all'
    )
    # Discrimination: applying it on both legs would double the ledger, which
    # is five orders of magnitude outside the tolerance above.
    assert abs(2.0 * delivered_kg - ledger[-1]) > 0.5 * delivered_kg
    assert float(leg2.config.planet.mass_tot) == pytest.approx(
        mass_before + DELIVERED, rel=1e-6
    ), (
        f'the resumed run carries {float(leg2.config.planet.mass_tot):.6f} M_earth '
        f'against the {mass_before + DELIVERED:.6f} M_earth one impact grows the '
        'planet to; the restored mass and the re-applied impact do not agree'
    )

    # The mantle carried past the impact is the one the impact melted. The
    # resumed run walked back to a state cooler than the pre-impact row and
    # re-solved forward, so every row it wrote would stay below that row if
    # the re-melt had been lost: nothing else here can warm the interior.
    # Read as a maximum over the whole post-impact stretch rather than at the
    # impact row alone, because the re-melt resets the solver at the end of
    # the step and the row it lands on is written before that reset.
    post_impact = resumed[resumed['Time'] >= IMPACT_TIME]
    assert len(post_impact) > 1, (
        'the resumed run stored no row past the impact step, so nothing was '
        'evolved from the mantle the impact melted'
    )
    warmest_after = float(post_impact['T_magma'].max())
    assert warmest_after > pre_impact_T, (
        f'the mantle never rose above {pre_impact_T:.1f} K after the impact '
        f'(warmest row {warmest_after:.1f} K), so the resumed run kept cooling '
        'the mantle from before the re-melt instead of the one the impact melted'
    )

    # The resumed leg discarded its own impact-step snapshot in turn, so the
    # discard is a property of any step that lands an impact rather than of
    # the first run.
    resumed_snapshots = _snapshot_times(outdir)
    assert int(IMPACT_TIME) not in resumed_snapshots, (
        f'the resumed run left a snapshot at the impact time (times on disk: '
        f'{resumed_snapshots}); a second restart would resume from a stale mantle'
    )
    assert any(time > IMPACT_TIME for time in resumed_snapshots), (
        f'the resumed run wrote no snapshot past the impact (times on disk: '
        f'{resumed_snapshots}), so a further restart would have to recompute '
        'the impact a third time'
    )
