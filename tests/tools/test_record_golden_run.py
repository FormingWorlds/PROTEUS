"""Unit tests for tools/record_golden_run.py (reference trajectory recording).

The script records the trajectory the golden-run parity check compares against,
and reports how the working tree differs from it. Both callers act on its exit
code, so what is tested here is the decision it reaches on each outcome rather
than the run it drives, which is mocked out.

The committed reference itself is checked here too, rather than beside the run
that compares against it. Reading two files costs nothing, so the pull-request
tier catches a configuration edited without recording the trajectory again, or
a reference that has lost its discriminating power, instead of leaving both to
the nightly run.

Contract clauses exercised:

- A dry run exits zero only when the trajectory it produces reproduces the
  recorded one against the configuration that reference was taken from.
- A reference taken from a different configuration is reported as that, ahead
  of the columns it makes differ, and fails the dry run.
- Recording overwrites the reference and states what moved against the one it
  replaces, so the change is visible without reading the diff.
- The committed reference belongs to the committed configuration, and records
  a run that evolves rather than one that sits at its initial condition.

Testing standards: docs/How-to/testing.md,
docs/Explanations/test_framework.md
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proteus.utils.trajectory import config_digest, read_reference, write_reference

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / 'tools'

# Floor on how many recorded columns move over the run. A column that holds its
# initial value throughout still catches a change that disturbs it, but it
# cannot discriminate between two trajectories, so this is the part of the
# helpfile that does the work. The recorded run moves about a hundred; with
# escape switched off it would move about twenty.
MIN_COLUMNS_VARYING = 80


def _load_record_golden_run():
    """Import tools/record_golden_run.py by path."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        'record_golden_run', TOOLS / 'record_golden_run.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recorder = _load_record_golden_run()


def _trajectory(t_magma) -> pd.DataFrame:
    """Build a short trajectory carrying one quantity that moves."""
    return pd.DataFrame(
        {
            'T_magma': np.asarray(t_magma, dtype=float),
            'M_planet': np.full(len(t_magma), 5.972e24),
        }
    )


def _count_varying(frame: pd.DataFrame) -> int:
    """Count the columns of ``frame`` that do not hold one value throughout.

    Counted against the column's first value rather than by comparing its
    extremes, because a column that is undefined throughout has undefined
    extremes and an undefined value does not equal itself, which would report
    it as varying. That is backwards for a count of how much of the helpfile
    can discriminate one trajectory from another.
    """
    return sum(
        1
        for column in frame.columns
        if not np.array_equal(
            frame[column].to_numpy(),
            np.full(len(frame), frame[column].to_numpy()[0]),
            equal_nan=True,
        )
    )


def test_a_column_that_is_never_defined_does_not_count_as_varying():
    """An undefined column carries no trajectory and is not counted as one.

    Contract clause: the count of varying columns is what certifies that the
    recorded run discriminates between trajectories, so a column that holds no
    information must not inflate it. An undefined value does not equal itself,
    so the obvious way of writing this count reports such a column as the most
    varying thing in the file.

    Verifies:
    - A column that is undefined at every row is counted as constant.
    - A column that becomes undefined part way through is counted as varying,
      since where it became undefined is itself part of the trajectory.
    - An ordinary constant and an ordinary series are counted as before, so
      the undefined case has not been special-cased into a wrong answer.
    """
    frame = pd.DataFrame(
        {
            'never_defined': np.full(4, np.nan),
            'becomes_undefined': np.asarray([1.0, 2.0, np.nan, 4.0]),
            'constant': np.full(4, 7.0),
            'series': np.asarray([1.0, 2.0, 3.0, 4.0]),
        }
    )
    assert _count_varying(frame) == 2, (
        'expected the two columns that move to be counted, and the undefined and '
        'constant ones to be left out'
    )
    assert _count_varying(frame[['never_defined']]) == 0
    assert _count_varying(frame[['becomes_undefined']]) == 1


@pytest.fixture
def stub_run(monkeypatch):
    """Replace the run the script drives with a trajectory chosen per test."""

    def _install(frame):
        monkeypatch.setattr(recorder, 'run_trajectory', lambda *args, **kwargs: frame)

    return _install


def test_dry_run_passes_only_while_the_trajectory_holds(tmp_path, stub_run, capsys):
    """A dry run reports the trajectory and exits on whether it reproduced.

    Contract clause: the dry run is what a refactor is checked with, so it has
    to distinguish a working tree that changed nothing from one that changed
    something, and say which columns moved when it did.

    Verifies:
    - An unchanged trajectory exits zero and reports that everything agreed.
    - A trajectory moved by one part in 1e3 exits non-zero and names the
      column that moved. Pinned from both sides, since a dry run that always
      passed and one that always failed would each satisfy only one half.
    - The reference file is left untouched either way, so a failing dry run
      does not quietly become a recording.
    """
    reference = tmp_path / 'reference.tsv'
    recorded = _trajectory([3000.0, 2800.0, 2600.0])
    write_reference(recorded, reference, config_path=recorder.CONFIG_PATH)
    stored_bytes = reference.read_bytes()

    stub_run(recorded)
    assert recorder.main(['--dry-run', '--reference', str(reference)]) == 0
    assert 'reproduce the reference' in capsys.readouterr().out

    stub_run(_trajectory([3000.0, 2800.0, 2600.0 * 1.001]))
    assert recorder.main(['--dry-run', '--reference', str(reference)]) == 1
    assert 'T_magma' in capsys.readouterr().out
    assert reference.read_bytes() == stored_bytes, 'a dry run rewrote the reference'


def test_dry_run_reports_a_reference_from_another_configuration(
    tmp_path, stub_run, capsys, monkeypatch
):
    """A reference taken from a different configuration is named as such.

    Contract clause: a configuration edited without recording the trajectory
    again makes every dependent column differ, and reporting those columns
    instead of the cause sends the reader looking in the code for a change
    that is in the configuration.

    Verifies:
    - The dry run fails, and says the reference came from a different
      configuration.
    - It says so instead of listing columns, so the cause is what the reader
      meets first.
    """
    reference = tmp_path / 'reference.tsv'
    recorded = _trajectory([3000.0, 2800.0, 2600.0])
    other_config = tmp_path / 'other.toml'
    other_config.write_text('mass_tot = 2.0\n')
    write_reference(recorded, reference, config_path=other_config)

    monkeypatch.setattr(recorder, 'CONFIG_PATH', tmp_path / 'golden.toml')
    (tmp_path / 'golden.toml').write_text('mass_tot = 1.0\n')

    stub_run(recorded)
    assert recorder.main(['--dry-run', '--reference', str(reference)]) == 1
    out = capsys.readouterr().out
    assert 'recorded from a different' in out
    assert 'reproduce the reference' not in out, (
        'the comparison was reported alongside the configuration mismatch, which '
        'invites the reader to treat the columns as the problem'
    )


def test_dry_run_without_a_reference_says_so(tmp_path, stub_run, capsys):
    """A dry run against a reference that does not exist explains itself.

    Contract clause: the first use of the check is on a tree that has no
    reference yet, and a bare failure there reads as a broken trajectory
    rather than a missing file.

    Verifies:
    - The exit code is non-zero, so the check cannot pass by having nothing to
      compare against.
    - The message names the missing path and the command that writes it.
    """
    stub_run(_trajectory([3000.0, 2800.0]))
    missing = tmp_path / 'absent.tsv'
    assert recorder.main(['--dry-run', '--reference', str(missing)]) == 1
    out = capsys.readouterr().out
    assert str(missing) in out
    assert '--dry-run' in out


def test_recording_overwrites_the_reference_and_states_what_moved(tmp_path, stub_run, capsys):
    """Recording replaces the reference and reports the change it makes.

    Contract clause: recording is how a deliberate change to the trajectory is
    accepted, so it has to say what it accepted rather than silently rewriting
    the file.

    Verifies:
    - The new trajectory is on disk afterwards and reads back as what the run
      produced.
    - The column that moved is named against the reference being replaced.
    - Recording where no reference exists yet writes one and reports no
      comparison, since there is nothing to compare against.
    """
    reference = tmp_path / 'reference.tsv'

    stub_run(_trajectory([3000.0, 2800.0, 2600.0]))
    assert recorder.main(['--reference', str(reference)]) == 0
    first = capsys.readouterr().out
    assert 'wrote' in first
    assert 'replaces' not in first, 'a first recording compared against nothing'

    moved = _trajectory([3000.0, 2800.0, 2500.0])
    stub_run(moved)
    assert recorder.main(['--reference', str(reference)]) == 0
    out = capsys.readouterr().out
    assert 'replaces' in out
    assert 'T_magma' in out
    assert 'M_planet' not in out, 'a column that held still was reported as moved'

    np.testing.assert_array_equal(
        read_reference(reference).frame['T_magma'].to_numpy(),
        moved['T_magma'].to_numpy(),
        err_msg='the reference on disk is not the trajectory that was recorded',
    )


def test_the_committed_reference_is_worth_comparing_against():
    """The recorded run evolves rather than sitting at its initial condition.

    Physical scenario: the fixed configuration behind the golden-run check,
    examined for whether the trajectory recorded from it covers a stretch of
    evolution where quantities are actually moving.

    A comparison against a run that never left its initial state would pass on
    almost any change to the physics, since there would be nothing in the
    trajectory for a change to disturb. This holds the recorded run to the
    regime the configuration was chosen for. It reads the recorded file rather
    than producing one, so it costs nothing and runs on every pull request.

    Verifies:
    - The reference belongs to the configuration committed beside it, so a
      configuration edited without recording the trajectory again is reported
      here rather than as an unexplained numerical difference in the nightly
      run.
    - The mantle crosses its entire melting interval, from wholly molten to
      wholly solid, so the trajectory spans both regimes and the transition
      between them rather than an endpoint where the melt fraction is pinned.
    - The magma ocean cools monotonically and the mantle never remelts.
      Nothing in this configuration can heat the interior, so a single warming
      or remelting step anywhere is unphysical.
    - Escape draws the volatile inventory down over the run, which is what
      puts the volatile columns in motion; with escape off they hold their
      initial values and four fifths of the helpfile is constants.
    - A large part of the helpfile moves over the run, so the comparison rests
      on quantities that discriminate between trajectories.
    """
    reference = read_reference(recorder.REFERENCE_PATH)
    assert reference.config_digest == config_digest(recorder.CONFIG_PATH), (
        f'{recorder.REFERENCE_PATH.name} was recorded from a different '
        f'{recorder.CONFIG_PATH.name} than the one committed beside it. Record it '
        'again with tools/record_golden_run.py, in the commit that changed the '
        'configuration.'
    )
    stored = reference.frame

    phi = stored['Phi_global'].to_numpy()
    assert phi[0] == pytest.approx(1.0), (
        f'the run starts at a melt fraction of {phi[0]:.4f} rather than a wholly '
        'molten mantle, so it does not cover the upper end of the melting interval'
    )
    assert phi[-1] == pytest.approx(0.0), (
        f'the run ends at a melt fraction of {phi[-1]:.4f} rather than a wholly '
        'solid mantle, so it stops short of solidification'
    )
    mushy = np.count_nonzero((phi > 0.0) & (phi < 1.0))
    assert mushy >= 10, (
        f'only {mushy} rows sit between a molten and a solid mantle, so the '
        'trajectory crosses the melting interval too fast to be discriminating there'
    )

    # The initialisation loops write several rows at t = 0 before the first
    # step is taken, holding the interior at its initial condition, so the
    # cooling checks below start where the evolution does.
    times = stored['Time'].to_numpy()
    first_evolution = int(np.argmax(times > 0.0))
    assert times[first_evolution] > 0.0, 'the recorded run never left t = 0'

    t_magma = stored['T_magma'].to_numpy()[first_evolution:]
    warming = np.flatnonzero(np.diff(t_magma) >= 0.0)
    assert warming.size == 0, (
        f'the magma ocean does not cool at step(s) {warming.tolist()}; nothing in '
        'this configuration can heat the interior'
    )
    remelting = np.flatnonzero(np.diff(phi[first_evolution:]) > 0.0)
    assert remelting.size == 0, (
        f'the mantle remelts at step(s) {remelting.tolist()}, which no heat source '
        'in this configuration can produce'
    )

    # Escape is what puts the volatile inventory in motion. Hydrogen is the
    # discriminating element: it carries the largest budget, so a drawdown
    # visible in it is well clear of the rounding in the recorded values.
    hydrogen = stored['H_kg_total'].to_numpy()
    assert hydrogen[-1] < hydrogen[0], (
        'the hydrogen inventory does not fall over the run, so escape is not '
        'acting and the volatile columns are constants'
    )
    escaped = stored['esc_kg_cumulative'].to_numpy()
    assert escaped[-1] > 0.0, 'no mass escaped over the run'
    assert np.all(np.diff(escaped) >= 0.0), 'cumulative escaped mass is not monotonic'

    varying = _count_varying(stored)
    assert varying >= MIN_COLUMNS_VARYING, (
        f'only {varying} of {len(stored.columns)} recorded columns move over the run, '
        f'below the {MIN_COLUMNS_VARYING} the configuration is chosen to put in '
        'motion; the comparison would rest mostly on constants'
    )
