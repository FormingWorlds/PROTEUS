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
- Interior and volatile state continue from the stored row rather than
  being rebuilt from the configuration's initial condition. The magma-ocean
  temperature is the discriminating quantity: the run has cooled well over
  a thousand Kelvin below ``planet.tsurf_init`` by the time the first leg
  stops, and re-initialising the interior part-way through the evolution
  reheats it, which shows up as a warming step in a trajectory that must
  cool monotonically.
- The structure-baseline flag and the loop counter carry over rather than
  reset.
- A run too short to carry a usable history is refused with a specific
  error and the configuration-error status code, rather than resuming from
  a partially initialised state, and a run one row longer is admitted. The
  threshold is pinned from both sides.
- A run resumed twice keeps advancing, so state restored on one resume is
  also re-persisted for the next.

Scope. Both legs run every module on its dummy backend, which keeps two
runs plus a resume inside the integration-tier budget. That fixes what this
file can and cannot show:

- The restored state is the final row of ``runtime_helpfile.csv``, and that
  is the only restoration channel under test. The dummy interior and dummy
  atmosphere write no ``_int.nc`` / ``_atm.nc``, and both backends rebuild
  themselves from helpfile fields, so no snapshot file is read back here.
- ``select_resumable_snapshot`` is called but is a structural no-op on
  these backends: ``require_atm`` is False for the dummy atmosphere and the
  dummy interior declares no snapshot filenames, so the first candidate row
  is always accepted and nothing is ever trimmed. Its trimming and
  fall-back behaviour is covered at the unit tier in
  ``tests/utils/test_coupler.py``.
- ``resume_structure_stale``, ``check_desiccation`` and ``extract_archives``
  run on this configuration but reach only their nothing-to-do branches
  (no stale-structure file, zero escape rate, ``archive_mod = 'none'``).

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

# Integration tier. Each test runs two or three real dummy-module
# simulations, so the file costs roughly 80 to 105 s locally and its slowest
# single test 26 to 36 s, depending on machine load. The 300 s tier ceiling is
# per test and leaves room for slower CI runners.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CONFIG = PROTEUS_ROOT / 'input' / 'dummy.toml'

# Time limits for the two legs, in years. The first leg stops while the
# magma ocean is still solidifying, so the resume seam lands on a state
# where temperature and melt fraction are both changing every step. On this
# configuration the mantle is fully solid by roughly 2.3e4 yr, after which
# melt fraction is pinned at zero and only the interior temperature moves;
# resuming there would exercise a far more static state.
LEG1_STOP_TIME = 1.0e4
LEG2_STOP_TIME = 3.0e4
LEG3_STOP_TIME = 5.0e4

# Absolute floor on run length, in years. Kept well below the first leg's
# stop time so the minimum-time criterion never competes with it.
MIN_STOP_TIME = 1.0e2

# Fixed step limits shared by both legs, in years. Note that the step
# adjacent to the seam is not an ordinary one: a leg that stops on
# simulation time has its final step clamped against the time remaining. The
# seam check is written on cooling rates rather than temperature differences
# so that clamp does not matter, since the dummy interior advances
# temperature as dTdt * dt and the rate is therefore independent of the step
# size. The corollary is that the rate check cannot see a step-size
# discontinuity at the seam; the parity test covers that.
DT_INITIAL = 1.0e2
DT_MINIMUM = 1.0e1
DT_MAXIMUM = 1.0e3

# Calibration of the seam continuity check: how many steps before the seam
# set the reference rate, and how much faster than that reference the seam
# step is allowed to be. Ten steps is long enough to average over the
# step-size controller's fluctuations and short enough to stay in the same
# cooling regime as the seam.
N_REFERENCE_STEPS = 10
RATE_SLACK = 2.0

# Absolute backstop on what the self-calibrated seam bound is allowed to
# tolerate, in Kelvin. The bound is derived from the run's own recent
# cooling rates, so without this a future change that slowed the dummy
# cooling law would loosen the check silently. The mantle spans 1000 K
# between the configured solidus and liquidus, so this is a tenth of the
# melting interval the seam sits in. The bound currently tolerates about
# 43 K, so this catches a regime change of roughly 2.3x or more without
# sitting close enough to fire on ordinary variation.
MAX_TOLERATED_SEAM_JUMP = 100.0

# The init-loop count hardcoded in Proteus.start(). Mirrored here so the
# refusal-threshold test can build runs that land exactly on and one row
# above the boundary; asserted against the runtime value so a change to it
# fails loudly instead of silently moving the boundary under the test.
EXPECTED_INIT_LOOPS = 3

# Iteration cap for the leg that is compared against an uninterrupted run.
# Any value that stops the first leg well inside the evolution works; 40
# leaves roughly a third of the trajectory ahead of the seam.
PARITY_ITERS = 40

# Relative tolerance for that comparison. The helpfile serialises at
# '%.10e', so a restored value carries about 5e-11 of relative error, and
# differencing two similar fluxes into F_net amplifies it: the largest
# observed disagreement across every numeric column is 2e-8. This sits
# about fifty times above that floor and many orders below any physical
# difference a genuine divergence would produce.
PARITY_RTOL = 1.0e-6

# Columns excluded from that comparison because they are not part of the
# simulated state.
PARITY_SKIP = ('runtime',)

# Backstop on the comparison's coverage, for a helpfile that collapsed
# outright. The screens below are asserted to drop nothing, so this fires only
# if the schema itself lost most of its columns, which both sides would lose
# together while still agreeing over the few that remained.
MIN_COLUMNS_COMPARED = 400


# Scratch directories created by Proteus construction, cleared after each
# test. ``set_directories`` allocates one unconditionally and nothing in the
# framework removes it, so a file that builds a dozen runners would otherwise
# leave a dozen empty directories behind on every run.
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
    """Write a copy of the dummy config with the output path already set.

    Building the runner straight from a config that names its output
    directory avoids a second ``init_directories`` call, and with it a second
    scratch directory per runner.
    """
    text = CONFIG.read_text()
    patched = text.replace('path = "auto"', f'path = "{output_dir}"', 1)
    assert patched != text, 'dummy config no longer carries the auto output path'
    destination = output_dir.parent / f'{output_dir.name}_config.toml'
    destination.write_text(patched)
    return destination


def _make_runner(output_dir, stop_time, *, iters_max=None, min_time=MIN_STOP_TIME):
    """Build a Proteus runner writing into ``output_dir``.

    Parameters
    ----------
    output_dir : pathlib.Path
        Absolute run directory. Passed straight through to
        ``params.out.path`` so successive legs share one directory.
    stop_time : float
        Maximum simulation time for this leg [yr].
    iters_max : int, optional
        Cap on the iteration count. The stored row count comes out as
        ``max(init_loops + 1, iters_max + 1)``: termination is only checked
        once the run leaves the init stage, so no run can stop before
        ``init_loops + 1`` rows however low the cap. Values below 2 are
        rejected by the configuration validator, because the minimum is set
        to 1 first and the maximum must exceed it.
    min_time : float, optional
        Minimum simulation time [yr]. Lower it to zero when the iteration
        cap must be what ends the run, since the minimum-time criterion
        otherwise keeps the run stepping past the cap.

    Returns
    -------
    Proteus
        Configured runner; the caller starts it.
    """
    runner = Proteus(config_path=_config_with_output_path(output_dir))
    _RUNNER_TEMP_DIRS.append(runner.directories['temp'])

    runner.config.params.dt.initial = DT_INITIAL
    runner.config.params.dt.minimum = DT_MINIMUM
    runner.config.params.dt.maximum = DT_MAXIMUM

    runner.config.params.stop.time.minimum = min_time
    runner.config.params.stop.time.maximum = stop_time
    # Run past solidification rather than stopping at it, so a leg can be
    # given a stop time anywhere in the evolution and the later legs have
    # room to advance. The criterion would otherwise end every leg at the
    # solidus, around 2.2e4 yr on this configuration.
    runner.config.params.stop.solid.enabled = False
    if iters_max is not None:
        runner.config.params.stop.iters.minimum = 1
        runner.config.params.stop.iters.maximum = iters_max

    # Write a data snapshot on every iteration so the resume reads the
    # richest state the active backends produce. write_mod, dt_write_rel and
    # archive_mod are pinned to their current defaults so a future schema
    # change cannot quietly alter what this test writes.
    runner.config.params.out.write_mod = 1
    runner.config.params.out.dt_write_rel = 0.0
    runner.config.params.out.archive_mod = 'none'
    # None, not 0: the schema reads 0 as "plot once at completion", and the
    # end-of-run block only skips plotting when this is None. Leaving it at 0
    # renders the whole plot suite after every leg, which is most of the wall
    # time and couples this test to every plotting module.
    runner.config.params.out.plot_mod = None

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
    - The magma ocean cools monotonically and the mantle never remelts.
      Nothing in this configuration can heat the interior, so a single
      warming or remelting step is unphysical. A resume that rebuilds
      interior state rather than restoring it injects heat at the seam and
      pushes the mantle back up its melting curve, which this catches.
    - The seam cooling rate is no faster than the steps just before it, and
      that bound is tight enough to reject a state rebuilt from the
      configuration's initial condition.
    - Melt fraction is strictly between 0 and 1 at the seam, which pins the
      resumed state to the actively solidifying regime rather than to a
      fully solid mantle where only temperature still moves.
    - The interior/atmosphere flux handshake survives the seam as an exact
      one-row lag identity, and the stellar clock advances by exactly the
      simulation step rather than resetting to the configured initial age.
    - The resume-only state carries over: interior initial condition 2,
      structure baseline marked done, loop counter continued.

    Per-element mass closure is deliberately not asserted. On the dummy
    outgassing backend each element's total is restored as an input and the
    atmospheric and dissolved reservoirs are derived from it by dividing
    and re-multiplying by the same stoichiometric mass fraction, so closure
    holds algebraically whatever the resume did, and the oxygen total under
    ``O_mode = "ic_chemistry"`` is defined as the sum of its own reservoirs.
    No elemental total varies over the run either, since dummy escape
    removes no mass. A closure assertion here could not fail.

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

    # Distance of the stored state from the configuration's initial
    # condition, used as the scale the seam bound has to be tighter than.
    # It needs no threshold of its own: the melt-fraction guard further down
    # forces the seam into the mushy zone, which bounds T_magma between the
    # configured solidus and liquidus and so bounds this separation well
    # away from zero.
    t_magma_last = float(stored.iloc[-1]['T_magma'])
    ic_separation = abs(t_magma_last - tsurf_init)

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
    for column in ('Time', 'T_magma', 'Phi_global', 'F_atm', 'P_surf', 'M_atm', 'H_kg_atm'):
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

    seam_index = n_stored - 1 - first_evolution
    t_magma = resumed['T_magma'].to_numpy()
    phi = resumed['Phi_global'].to_numpy()
    d_tmagma = np.diff(t_magma[first_evolution:])
    d_phi = np.diff(phi[first_evolution:])

    # The planet cools monotonically and the mantle never remelts. There is
    # no heat source in this configuration that can reverse either, so a
    # single warming or remelting step anywhere in the trajectory is
    # unphysical. This is the check that catches a resume which rebuilds
    # interior state instead of restoring it: re-initialising the interior
    # part-way through the evolution injects heat and pushes the mantle back
    # up its melting curve, which shows here as a positive step. The sign is
    # the discriminating quantity, so it is asserted separately from the
    # magnitude bound below; taking the absolute value first would hide
    # exactly this failure.
    warming = np.flatnonzero(d_tmagma >= 0.0)
    assert warming.size == 0, (
        f'magma-ocean temperature does not fall at step(s) {warming.tolist()} '
        f'(seam is at step {seam_index}); largest rise is '
        f'{np.max(d_tmagma):+.3f} K, and nothing in this configuration can heat '
        'the interior'
    )
    remelting = np.flatnonzero(d_phi > 0.0)
    assert remelting.size == 0, (
        f'mantle remelts at step(s) {remelting.tolist()} (seam is at step '
        f'{seam_index}); largest melt-fraction rise is {np.max(d_phi):+.6f}'
    )

    # Magnitude bound on the seam step, on top of the sign guard above: a
    # resume that restored a wrong but still-cooling state would pass the
    # sign check while cooling far too fast. Rates rather than raw
    # temperature differences, because the step size is free to change at
    # the seam while the cooling rate is a smooth function of the state.
    rates = np.abs(d_tmagma) / evolution_steps
    # The window stops one step short of the seam. A leg that ends on
    # simulation time has its final step clamped against the time remaining,
    # so the step immediately before the seam is set by where the stop time
    # fell rather than by the controller, and it should not be the one
    # calibrating the bound.
    reference_end = max(1, seam_index - 1)
    reference_rates = rates[max(0, reference_end - N_REFERENCE_STEPS) : reference_end]
    assert len(reference_rates) > 0, 'first leg has no evolution steps to calibrate against'
    seam_rate = float(rates[seam_index])
    rate_bound = RATE_SLACK * float(np.max(reference_rates))
    assert seam_rate <= rate_bound, (
        f'magma-ocean cooling rate jumps to {seam_rate:.3e} K/yr across the resume '
        f'seam, above the {rate_bound:.3e} K/yr bound set by the last '
        f'{N_REFERENCE_STEPS} steps of the first leg'
    )
    # Absolute backstop. The bound above is calibrated from the run itself,
    # so state in Kelvin what it actually tolerates and hold that against a
    # fixed ceiling; otherwise a future change to the dummy cooling law would
    # relax the check without anyone noticing.
    seam_dt = float(evolution_steps[seam_index])
    tolerated_jump = rate_bound * seam_dt
    assert tolerated_jump < MAX_TOLERATED_SEAM_JUMP, (
        f'the seam bound tolerates a restored-state error of {tolerated_jump:.1f} K, '
        f'above the {MAX_TOLERATED_SEAM_JUMP:.1f} K ceiling; the self-calibrated '
        'bound has drifted loose'
    )

    # Scale guard: confirm the bound is tight enough to reject a resume that
    # rebuilt state from the configuration, which would move T_magma by
    # ic_separation over the seam step.
    reinit_rate = ic_separation / seam_dt
    assert rate_bound < reinit_rate, (
        f'seam bound {rate_bound:.3e} K/yr does not reject re-initialisation, which '
        f'would show as {reinit_rate:.3e} K/yr over the {seam_dt:.3e} yr seam step'
    )
    # Continuity is two-sided. Without a floor, a resume that restores the
    # state but then fails to evolve it, a collapsed step size or an
    # interior left at its constructor defaults, drives the seam rate toward
    # zero and passes the ceiling above.
    rate_floor = float(np.min(reference_rates)) / RATE_SLACK
    assert seam_rate >= rate_floor, (
        f'magma-ocean cooling stalls to {seam_rate:.3e} K/yr across the resume seam, '
        f'below the {rate_floor:.3e} K/yr floor set by the last '
        f'{N_REFERENCE_STEPS} steps of the first leg'
    )

    # Regime guard on the seam. The first leg must stop while the mantle is
    # still solidifying, so the resumed state is one where the interior is
    # actively evolving rather than sitting at a fixed point. Once the
    # mantle is fully solid the melt fraction is pinned at zero, the
    # no-remelting check above is satisfied by a constant, and only the
    # temperature still moves, which is a much weaker thing to resume into.
    # No bound is asserted on the first resumed row: the dummy interior
    # derives melt fraction as a ratio clamped to [0, 1] at both ends, so a
    # boundedness check there cannot fail whatever the resume did.
    phi_seam_pre = float(phi[n_stored - 1])
    assert 0.0 < phi_seam_pre < 1.0, (
        f'melt fraction at the seam is {phi_seam_pre:.4f}, pinned at an endpoint; '
        'the first leg must stop while the mantle is still solidifying for the '
        'seam checks to bear on an evolving state'
    )

    # The interior/atmosphere flux handshake carries across the seam. The
    # dummy interior sets F_int from the previous row's F_atm, so this is an
    # exact one-row lag identity: it holds only if the first resumed
    # iteration read the atmospheric flux from the restored row rather than
    # from a re-initialised state. Asserted at the seam against a control at
    # the step before, so a backend that stopped honouring the lag fails
    # loudly here rather than silently weakening the check.
    f_atm = resumed['F_atm'].to_numpy()
    f_int = resumed['F_int'].to_numpy()
    assert f_int[n_stored] == pytest.approx(f_atm[n_stored - 1], rel=1e-9), (
        f'interior flux on the first resumed row is {f_int[n_stored]:.6e} W/m2 against '
        f'{f_atm[n_stored - 1]:.6e} W/m2 of atmospheric flux on the last stored row; '
        'the flux handshake did not survive the resume'
    )
    assert f_int[n_stored - 1] == pytest.approx(f_atm[n_stored - 2], rel=1e-9), (
        'the one-row flux lag does not hold within the first leg either, so the '
        'seam check above is not testing the resume'
    )

    # The stellar clock is rebased on restored simulation time rather than
    # reset to the configured initial age. A reset would rewind the stellar
    # evolution track while leaving the interior checks above untouched. The
    # increment is the invariant, not the offset: the init-stage rows hold
    # Time at zero while age_star already advances, so age_star - Time is
    # not constant over the run.
    age_star = resumed['age_star'].to_numpy()
    assert age_star[n_stored] - age_star[n_stored - 1] == pytest.approx(
        times[n_stored] - times[n_stored - 1], rel=1e-6
    ), (
        f'stellar age advances {age_star[n_stored] - age_star[n_stored - 1]:.4f} yr '
        f'across the seam against a simulation step of '
        f'{times[n_stored] - times[n_stored - 1]:.4f} yr'
    )

    # Resume-only state carried to the end of the run. The interior
    # initial-condition code is deliberately absent: the main loop assigns it
    # 2 on every iteration outside the init stage, so its end-of-run value is
    # the same whatever the resume set, exactly like the init-stage flag.
    # Setting it wrongly at resume is caught instead by the monotonic-cooling
    # check above, which is where the resulting reheating shows up.
    #
    # The structure baseline is a state carry-over check only. The behaviour
    # it guards, re-solving the initial structure mid-evolution, needs
    # ``interior_struct.module = 'zalmoxis'``; on the dummy structure module
    # used here the baseline solve returns early regardless.
    assert leg2._baseline_structure_done is True, (
        'resumed run did not carry over the structure baseline flag'
    )
    # The counter is set to the restored row count and incremented once per
    # iteration, and the helpfile grows one row per iteration, so the exact
    # identity holds. An inequality would let a counter restored too low pass.
    assert leg2.loops['total'] == len(resumed), (
        f'loop counter ended at {leg2.loops["total"]} against {len(resumed)} '
        'helpfile rows; the resumed counter did not continue the stored history'
    )


@pytest.mark.integration
def test_resume_threshold_sits_one_row_above_the_init_loops(tmp_path):
    """The resume length threshold is where the contract says it is.

    Contract clause: resume needs strictly more than ``init_loops + 1``
    helpfile rows, because the rows written during the initialisation loops
    do not describe a converged evolving state. At or below that count the
    run is refused with the configuration-error status code.

    The threshold is pinned from both sides, one row apart, so an off-by-one
    regression in the comparison is caught. Testing only the refusal would
    pass just as happily if the guard rejected every run.

    Verifies, on the refusing side (exactly ``init_loops + 1`` rows):
    - Starting with ``resume=True`` raises ``RuntimeError`` naming the
      length problem.
    - The status file records code 20, so an outside observer sees the
      refusal rather than a silently abandoned directory.
    - The helpfile is left byte-identical, so the run can be lengthened and
      retried.

    Verifies, on the accepting side (one row more):
    - The resume is admitted and advances both the row count and simulation
      time.
    """
    # Refusing side: an iteration cap that lands exactly on the threshold.
    refused_dir = tmp_path / 'at_threshold'
    at_threshold = _make_runner(
        refused_dir, LEG1_STOP_TIME, iters_max=EXPECTED_INIT_LOOPS, min_time=0.0
    )
    at_threshold.start(resume=False, offline=True)

    init_loops = at_threshold.loops['init_loops']
    assert init_loops == EXPECTED_INIT_LOOPS, (
        f'Proteus now runs {init_loops} init loops, not the {EXPECTED_INIT_LOOPS} this '
        'test builds its boundary runs against; the caps below no longer straddle '
        'the refusal threshold'
    )
    assert len(at_threshold.hf_all) == init_loops + 1, (
        f'setup produced {len(at_threshold.hf_all)} rows, not the '
        f'{init_loops + 1} needed to sit exactly on the refusal threshold'
    )

    helpfile = refused_dir / 'runtime_helpfile.csv'
    assert helpfile.is_file(), 'first run left no helpfile on disk'
    stored_bytes = helpfile.read_bytes()

    with pytest.raises(RuntimeError, match='too short to be resumed'):
        _make_runner(refused_dir, LEG2_STOP_TIME).start(resume=True, offline=True)

    status = (refused_dir / 'status').read_text().splitlines()
    assert status[0].strip() == '20', (
        f'status file records {status[0].strip()!r}, expected the configuration-error code 20'
    )
    assert helpfile.read_bytes() == stored_bytes, (
        'refused resume modified the stored helpfile, so the run cannot be '
        'lengthened and retried'
    )

    # Accepting side: one row above the threshold must be admitted.
    accepted_dir = tmp_path / 'above_threshold'
    above = _make_runner(
        accepted_dir, LEG1_STOP_TIME, iters_max=EXPECTED_INIT_LOOPS + 1, min_time=0.0
    )
    above.start(resume=False, offline=True)
    n_above = len(above.hf_all)
    assert n_above == init_loops + 2, (
        f'setup produced {n_above} rows, not the {init_loops + 2} needed to sit '
        'one row above the refusal threshold'
    )

    continued = _make_runner(accepted_dir, LEG2_STOP_TIME)
    continued.start(resume=True, offline=True)
    assert len(continued.hf_all) > n_above, (
        f'resume one row above the threshold produced {len(continued.hf_all)} rows, '
        f'no advance on the stored {n_above}'
    )
    assert float(continued.hf_all.iloc[-1]['Time']) > float(above.hf_all.iloc[-1]['Time']), (
        'resume one row above the threshold did not advance simulation time'
    )


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_resume_twice_keeps_advancing(tmp_path):
    """A run resumed a second time continues from the first resume.

    Physical scenario: the same magma-ocean planet is stopped and restarted
    twice, as a long campaign run does when it is repeatedly requeued.

    A single resume only shows that stored state can be read back. It cannot
    show that the resumed run writes state the next resume can read: a field
    restored on resume but never re-persisted, or a counter restored but not
    carried into the next helpfile write, works once and fails the second
    time. Resuming twice is the cheapest check that the restart cycle is
    closed rather than one-shot.

    Verifies:
    - Each leg strictly extends the previous helpfile and advances
      simulation time.
    - The trajectory still cools monotonically and never remelts across both
      seams, so neither restart injects heat.
    """
    outdir = tmp_path / 'twice'

    leg1 = _make_runner(outdir, LEG1_STOP_TIME)
    leg1.start(resume=False, offline=True)
    n_first = len(leg1.hf_all)

    leg2 = _make_runner(outdir, LEG2_STOP_TIME)
    leg2.start(resume=True, offline=True)
    n_second = len(leg2.hf_all)
    assert n_second > n_first, (
        f'first resume produced {n_second} rows, no advance on the stored {n_first}'
    )

    leg3 = _make_runner(outdir, LEG3_STOP_TIME)
    leg3.start(resume=True, offline=True)
    final = leg3.hf_all
    assert len(final) > n_second, (
        f'second resume produced {len(final)} rows, no advance on the {n_second} '
        'rows the first resume left behind'
    )
    assert float(final.iloc[-1]['Time']) > float(leg2.hf_all.iloc[-1]['Time']), (
        'second resume did not advance simulation time past the first'
    )

    # The prefix written before the second resume survives it intact.
    np.testing.assert_allclose(
        final['Time'].to_numpy()[:n_second],
        leg2.hf_all['Time'].to_numpy(),
        rtol=1e-9,
        atol=0.0,
        err_msg='second resume rewrote the times stored by the first',
    )

    # Both seams obey the same monotonic-cooling and no-remelting invariants
    # as the single-resume case, so neither restart reheated the interior.
    times = final['Time'].to_numpy()
    first_evolution = int(np.argmax(times > 0.0))
    d_tmagma = np.diff(final['T_magma'].to_numpy()[first_evolution:])
    d_phi = np.diff(final['Phi_global'].to_numpy()[first_evolution:])
    warming = np.flatnonzero(d_tmagma >= 0.0)
    assert warming.size == 0, (
        f'magma-ocean temperature does not fall at step(s) {warming.tolist()} across '
        f'two resumes (seams near steps {n_first - 1 - first_evolution} and '
        f'{n_second - 1 - first_evolution}); largest rise is {np.max(d_tmagma):+.3f} K'
    )
    remelting = np.flatnonzero(d_phi > 0.0)
    assert remelting.size == 0, (
        f'mantle remelts at step(s) {remelting.tolist()} across two resumes; '
        f'largest melt-fraction rise is {np.max(d_phi):+.6f}'
    )


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_resume_reproduces_uninterrupted_run(tmp_path):
    """Stopping and resuming reproduces the trajectory of one continuous run.

    Physical scenario: the same planet is evolved twice to the same stop
    time, once in a single process and once as a shorter run that is
    stopped at an iteration boundary and resumed. Restarting is an
    operational act, not a physical one, so the two trajectories must agree.

    This is the strongest statement available about resume, and it subsumes
    the continuity and boundedness checks made elsewhere in this file: if
    every stored quantity matches an uninterrupted run row for row, then
    nothing was lost, rebuilt or double counted at the seam.

    The first leg is stopped on the iteration cap rather than on simulation
    time, which is what makes the comparison exact. ``next_step`` clamps the
    final step of a time-terminated run against the time remaining, so a
    leg that stops on ``stop.time.maximum`` ends on a shortened step and its
    history is not a prefix of any uninterrupted run.

    Verifies:
    - The restarted run has the same number of rows as the uninterrupted
      one, so the restart neither dropped nor duplicated a step.
    - Every numeric column agrees to within the helpfile's serialisation
      floor, including the interior temperature, the fluxes, the surface
      pressure and every volatile reservoir.
    - The comparison is made against a first leg that really did stop early,
      so the test cannot pass by comparing a run against itself.
    """
    uninterrupted = _make_runner(tmp_path / 'uninterrupted', LEG2_STOP_TIME)
    uninterrupted.start(resume=False, offline=True)
    reference = uninterrupted.hf_all.copy()

    restarted_dir = tmp_path / 'restarted'
    first_leg = _make_runner(restarted_dir, LEG2_STOP_TIME, iters_max=PARITY_ITERS)
    first_leg.start(resume=False, offline=True)
    n_first = len(first_leg.hf_all)

    # The first leg must genuinely stop short, otherwise the comparison
    # below is a run against itself and proves nothing about resuming.
    assert n_first < len(reference), (
        f'first leg stored {n_first} rows against the uninterrupted run of '
        f'{len(reference)}; it did not stop early, so the comparison would be vacuous'
    )
    assert float(first_leg.hf_all.iloc[-1]['Time']) < float(reference.iloc[-1]['Time']), (
        'first leg reached the uninterrupted stop time, so it did not stop early'
    )

    resumed_run = _make_runner(restarted_dir, LEG2_STOP_TIME)
    resumed_run.start(resume=True, offline=True)
    restarted = resumed_run.hf_all

    assert len(restarted) == len(reference), (
        f'restarted run stored {len(restarted)} rows against {len(reference)} for the '
        'uninterrupted run; the restart changed the discretisation'
    )

    compared = 0
    non_numeric: list[str] = []
    non_finite: list[str] = []
    for column in reference.columns:
        if column in PARITY_SKIP:
            continue
        expected = reference[column].to_numpy()
        actual = restarted[column].to_numpy()
        if not (
            np.issubdtype(expected.dtype, np.number) and np.issubdtype(actual.dtype, np.number)
        ):
            non_numeric.append(column)
            continue
        # Only the reference side is screened. Screening the restarted side
        # too would skip precisely the columns a resume corrupted into NaN,
        # which is the signature this comparison exists to catch; leaving them
        # in makes the comparison below fail on them instead.
        if not np.all(np.isfinite(expected)):
            non_finite.append(column)
            continue
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=PARITY_RTOL,
            atol=0.0,
            err_msg=f'restarted run diverges from the uninterrupted run in {column}',
        )
        compared += 1

    # The screens above drop columns without comparing them, and most of what
    # this configuration writes holds still for the whole run, so a count of
    # compared columns cannot tell a covered helpfile from a sparse one.
    # Nothing here is non-numeric or non-finite, so anything dropped is a
    # change to look at rather than an expected loss.
    dropped = non_numeric + non_finite
    assert not dropped, (
        f'{len(non_numeric)} non-numeric and {len(non_finite)} non-finite columns were '
        f'dropped before comparison ({", ".join(dropped[:8])}); the parity check no '
        'longer covers them'
    )

    assert compared >= MIN_COLUMNS_COMPARED, (
        f'only {compared} of the {len(reference.columns)} columns written were '
        'compared; the parity check is not covering the helpfile'
    )
