"""Integration test: resuming a stopped run from its on-disk state.

Exercises the ``config.params.resume`` branch of ``Proteus.start()``
(``src/proteus/proteus.py``) against a real coupled run, with no mocking
anywhere in the call chain. A second ``Proteus`` object is pointed at the
output directory left behind by a completed run and started with
``resume=True``, which is what ``proteus start -r`` does.

Contract clauses exercised:

- The restored helpfile is extended, never rewritten: every row written by
  the first leg survives the resume unchanged, to the precision of the
  helpfile's own ``%.10e`` serialisation.
- Simulation time continues from the last stored row instead of restarting
  at t = 0, and stays strictly increasing across the seam.
- Interior, atmosphere and volatile state are read back from disk rather
  than rebuilt from the configuration's initial condition. The magma-ocean
  temperature is the discriminating quantity: the run has cooled thousands
  of Kelvin below ``planet.tsurf_init`` by the time the first leg stops, so
  a resume that silently re-initialised would be visible as a jump of that
  size at the seam.
- The resume-only state flags are set: interior initial-condition code 2,
  the init stage closed, the one-time structure baseline marked done, and
  the loop counter continued rather than reset.
- A run too short to carry a usable history is refused with a specific
  error and the configuration-error status code, rather than resuming from
  a partially initialised state.

Both legs run every module on its dummy backend, which keeps a multi-step
run plus a resume inside the integration-tier budget. The dummy interior
and dummy atmosphere write no ``_int.nc`` / ``_atm.nc`` snapshot pair, so
the trailing-snapshot selection performed by ``select_resumable_snapshot``
is not covered here; that path needs an interior or atmosphere module that
writes real snapshots, and every such module costs minutes per step while
a resumable history needs more than ``init_loops + 1`` rows.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus

# Integration tier: two full dummy-module runs plus a resume. Local wall
# time is ~40 s; the 300 s tier ceiling absorbs slower CI runners.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CONFIG = PROTEUS_ROOT / 'input' / 'dummy.toml'

# Time limits for the two legs, in years. The first leg stops early enough
# that the magma ocean is still evolving, so the resumed leg has a live
# trajectory to continue rather than a frozen end state.
LEG1_STOP_TIME = 2.0e5
LEG2_STOP_TIME = 6.0e5

# Fixed step limits shared by both legs, in years. Holding these equal
# across the seam means the first post-resume step is an ordinary step, so
# the seam continuity check below compares like with like.
DT_INITIAL = 1.0e4
DT_MINIMUM = 1.0e3
DT_MAXIMUM = 1.0e5

# Elements whose reservoirs are tracked in the helpfile by the dummy
# outgassing backend.
ELEMENTS = ('H', 'C', 'N', 'S', 'O')

# Calibration of the seam continuity check: how many steps before the seam
# set the reference cooling rate, and how much faster than that reference
# the seam step is allowed to be. Ten steps is long enough to average over
# the step-size controller's fluctuations and short enough to stay in the
# same cooling regime as the seam.
N_REFERENCE_STEPS = 10
RATE_SLACK = 5.0


def _make_runner(output_dir, stop_time, *, iters_max=None):
    """Build a Proteus runner writing into ``output_dir``.

    Parameters
    ----------
    output_dir : pathlib.Path
        Absolute run directory. Passed straight through to
        ``params.out.path`` so both legs share one directory.
    stop_time : float
        Maximum simulation time for this leg [yr].
    iters_max : int, optional
        Cap on the iteration count. Used to build a history too short to
        resume from.

    Returns
    -------
    Proteus
        Configured runner; the caller starts it.
    """
    runner = Proteus(config_path=CONFIG)
    runner.config.params.out.path = str(output_dir)
    runner.init_directories()

    runner.config.params.dt.initial = DT_INITIAL
    runner.config.params.dt.minimum = DT_MINIMUM
    runner.config.params.dt.maximum = DT_MAXIMUM

    runner.config.params.stop.time.minimum = DT_INITIAL
    runner.config.params.stop.time.maximum = stop_time
    # The dummy interior solidifies quickly at these temperatures; leaving
    # the solidification criterion on would end the first leg before it
    # accumulates a resumable history.
    runner.config.params.stop.solid.enabled = False
    if iters_max is not None:
        runner.config.params.stop.iters.minimum = 1
        runner.config.params.stop.iters.maximum = iters_max
        # The minimum-time criterion overrides the iteration cap: the run
        # keeps stepping until it is met, whatever the loop count. Drop it
        # so the iteration cap is what actually ends the run.
        runner.config.params.stop.time.minimum = 0.0

    # Write a data snapshot on every iteration so the resume reads the
    # richest state the active backends produce, and skip plotting and
    # archiving, which contribute nothing to the resume contract.
    runner.config.params.out.write_mod = 1
    runner.config.params.out.dt_write_rel = 0.0
    runner.config.params.out.plot_mod = 0
    runner.config.params.out.archive_mod = 'none'

    return runner


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_resume_continues_trajectory_from_disk_state(tmp_path):
    """A resumed run continues the stored trajectory instead of restarting.

    Physical scenario: a magma-ocean planet on all-dummy backends is
    evolved until it has cooled well below its initial surface temperature,
    the run is stopped, and a fresh process resumes it from the output
    directory with a later stop time.

    Verifies:
    - The resumed helpfile strictly extends the stored one, with the shared
      prefix unchanged in time, temperature and melt fraction.
    - Time is strictly increasing over the whole resumed trajectory and the
      first post-resume row is later than the last stored row.
    - The magma-ocean cooling rate is continuous across the seam: the seam
      step is no faster than the steps just before it, while the state at
      the seam sits thousands of Kelvin away from the configuration's
      initial condition. A resume that rebuilt state from the
      configuration would fail the first of those and not the second.
    - Melt fraction stays inside [0, 1] and temperature stays positive
      across the seam.
    - Per-element mass closure holds on both sides of the seam, so the
      volatile reservoirs were restored consistently and not double
      counted.
    - The resume-only state carries over: interior initial condition 2,
      structure baseline marked done, loop counter continued.

    The init-stage flag is deliberately not asserted here. The main loop
    closes the init stage on its first iteration whenever the loop counter
    already exceeds ``init_loops``, which is always true on resume, so its
    end-of-run value is fixed regardless of what the resume set it to. Its
    observable consequence, simulation time held at zero, is covered by
    the strict time monotonicity check instead.
    """
    outdir = tmp_path / 'resume_run'

    leg1 = _make_runner(outdir, LEG1_STOP_TIME)
    tsurf_init = float(leg1.config.planet.tsurf_init)
    leg1.start(resume=False, offline=True)

    stored = leg1.hf_all.copy()
    n_stored = len(stored)
    # A resumable history must exceed init_loops + 1 rows; the guard tested
    # in the sibling test below rejects anything shorter.
    assert n_stored > leg1.loops['init_loops'] + 1, (
        f'first leg produced only {n_stored} rows, too short to resume from'
    )

    # The first leg must have cooled substantially, otherwise the
    # discrimination against a configuration-initialised resume below has
    # no separation to work with.
    t_magma_last = float(stored.iloc[-1]['T_magma'])
    assert t_magma_last > 0.0, f'stored T_magma is not positive: {t_magma_last}'
    ic_separation = abs(t_magma_last - tsurf_init)
    assert ic_separation > 500.0, (
        f'first leg ended at T_magma={t_magma_last:.1f} K, only {ic_separation:.1f} K '
        f'from the initial condition {tsurf_init:.1f} K; the resume seam check '
        'cannot discriminate re-initialisation at this separation'
    )

    leg2 = _make_runner(outdir, LEG2_STOP_TIME)
    leg2.start(resume=True, offline=True)
    resumed = leg2.hf_all

    # The resumed run advanced past where the first leg stopped.
    assert len(resumed) > n_stored, (
        f'resume produced {len(resumed)} rows, no advance on the stored {n_stored}'
    )
    assert float(resumed.iloc[-1]['Time']) > float(stored.iloc[-1]['Time']), (
        'resumed run did not advance simulation time'
    )

    # The stored prefix survives unchanged. The helpfile is serialised with
    # '%.10e', so a round-trip through disk is exact to ~11 significant
    # digits; rtol=1e-9 sits clear of that floor and far below any physical
    # change a single step would make.
    for column in ('Time', 'T_magma', 'Phi_global'):
        np.testing.assert_allclose(
            resumed[column].to_numpy()[:n_stored],
            stored[column].to_numpy(),
            rtol=1e-9,
            atol=0.0,
            err_msg=f'resume rewrote stored {column} values instead of extending them',
        )

    # Time is strictly increasing once the run leaves the initialisation
    # stage, which writes several rows at t = 0 before the first step. A
    # restart from t = 0 appended to the stored history would show up here
    # as a non-positive step.
    times = resumed['Time'].to_numpy()
    first_evolution = int(np.argmax(times > 0.0))
    assert times[first_evolution] > 0.0, 'resumed trajectory never left t = 0'
    evolution_steps = np.diff(times[first_evolution:])
    assert np.all(evolution_steps > 0.0), (
        'resumed trajectory has a non-increasing time step after the init stage; '
        f'minimum step is {np.min(evolution_steps):.3e} yr'
    )

    # Seam continuity in the magma-ocean cooling rate. Rates rather than
    # raw temperature steps, because the step size is free to change at the
    # seam; the cooling rate is a smooth function of the state and is not.
    # The reference is the fastest rate over the last few steps of the
    # first leg, which brackets the physical rate the resumed run should
    # pick up at.
    t_magma = resumed['T_magma'].to_numpy()
    rates = np.abs(np.diff(t_magma[first_evolution:])) / evolution_steps
    seam_index = n_stored - 1 - first_evolution
    reference_rates = rates[max(0, seam_index - N_REFERENCE_STEPS) : seam_index]
    assert len(reference_rates) > 0, 'first leg has no evolution steps to calibrate against'
    seam_rate = float(rates[seam_index])
    rate_bound = RATE_SLACK * float(np.max(reference_rates))
    assert seam_rate <= rate_bound, (
        f'magma-ocean cooling rate jumps to {seam_rate:.3e} K/yr across the resume '
        f'seam, above the {rate_bound:.3e} K/yr bound set by the last '
        f'{N_REFERENCE_STEPS} steps of the first leg'
    )
    # Discrimination: a resume that rebuilt state from the configuration
    # would move T_magma by ic_separation over the seam step. Confirm the
    # bound above is tight enough to reject that.
    seam_dt = float(evolution_steps[seam_index])
    reinit_rate = ic_separation / seam_dt
    assert rate_bound < reinit_rate, (
        f'seam bound {rate_bound:.3e} K/yr does not reject re-initialisation, which '
        f'would show as {reinit_rate:.3e} K/yr over the {seam_dt:.3e} yr seam step'
    )

    # Boundedness and positivity across the seam.
    seam_rows = resumed.iloc[n_stored - 1 : n_stored + 1]
    assert np.all(seam_rows['T_magma'].to_numpy() > 0.0), (
        'non-positive magma-ocean temperature at the resume seam'
    )
    phi_seam = seam_rows['Phi_global'].to_numpy()
    assert np.all((phi_seam >= 0.0) & (phi_seam <= 1.0)), (
        f'melt fraction outside [0, 1] at the resume seam: {phi_seam}'
    )

    # Per-element mass closure on both sides of the seam. The restored
    # reservoirs must still sum to their totals; a resume that reloaded the
    # atmosphere without the dissolved reservoirs would break closure here.
    for _, row in seam_rows.iterrows():
        for element in ELEMENTS:
            total = float(row[f'{element}_kg_total'])
            if total <= 0.0:
                continue
            parts = [
                float(row[f'{element}_kg_atm']),
                float(row[f'{element}_kg_liquid']),
                float(row[f'{element}_kg_solid']),
            ]
            for name, value in zip(('atm', 'liquid', 'solid'), parts):
                assert value >= 0.0, (
                    f'{element}_kg_{name} negative at t={row["Time"]:.3e} yr: {value:.3e}'
                )
            assert sum(parts) == pytest.approx(total, rel=1e-6), (
                f'{element} reservoirs do not close at t={row["Time"]:.3e} yr: '
                f'sum={sum(parts):.6e} kg vs total={total:.6e} kg'
            )

    # Resume-only state flags.
    assert leg2.interior_o.ic == 2, (
        f'interior initial-condition code is {leg2.interior_o.ic}, expected the resume value 2'
    )
    assert leg2._baseline_structure_done is True, (
        'resumed run did not carry over the structure baseline, so it would '
        're-solve the initial structure mid-evolution'
    )
    assert leg2.loops['total'] >= n_stored, (
        f'loop counter restarted at {leg2.loops["total"]} instead of continuing '
        f'from the stored {n_stored} rows'
    )


@pytest.mark.integration
def test_resume_refuses_run_shorter_than_init_window(tmp_path):
    """A run without enough history is refused instead of half-resumed.

    Contract clause: resume needs more than ``init_loops + 1`` helpfile
    rows, because the rows written during the initialisation loops do not
    describe a converged evolving state. Below that the run is rejected
    with the configuration-error status code.

    Verifies:
    - The iteration-capped run really is short: its stored helpfile has at
      most ``init_loops + 1`` rows, so the refusal is attributable to
      length rather than to a failed run.
    - Starting with ``resume=True`` raises ``RuntimeError`` naming the
      length problem.
    - The status file records code 20, so an outside observer sees the
      refusal rather than a silently abandoned directory.
    - The helpfile is left on disk untouched by the refused resume, so a
      user can lengthen the run and try again.
    """
    outdir = tmp_path / 'short_run'

    short = _make_runner(outdir, LEG1_STOP_TIME, iters_max=2)
    short.start(resume=False, offline=True)

    init_loops = short.loops['init_loops']
    n_short = len(short.hf_all)
    assert n_short <= init_loops + 1, (
        f'iteration-capped run produced {n_short} rows, above the '
        f'{init_loops + 1}-row resume threshold; the refusal below would not '
        'be attributable to length'
    )
    assert n_short > 0, 'iteration-capped run wrote no helpfile rows at all'

    helpfile = outdir / 'runtime_helpfile.csv'
    assert helpfile.is_file(), 'first run left no helpfile on disk'
    stored_bytes = helpfile.read_bytes()

    resumed = _make_runner(outdir, LEG2_STOP_TIME)
    with pytest.raises(RuntimeError, match='too short to be resumed'):
        resumed.start(resume=True, offline=True)

    status = (outdir / 'status').read_text().splitlines()
    assert status[0].strip() == '20', (
        f'status file records {status[0].strip()!r}, expected the configuration-error code 20'
    )

    assert helpfile.read_bytes() == stored_bytes, (
        'refused resume modified the stored helpfile, so the run cannot be '
        'lengthened and retried'
    )
