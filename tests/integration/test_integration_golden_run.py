"""Integration test: a fixed run reproduces its recorded trajectory.

A refactor is meant to leave behaviour alone, and the only way to show that is
to run the same configuration before and after and find the same numbers. This
file holds one fixed configuration (``golden_run.toml``), the trajectory it
produced (``golden_run.tsv``), and the comparison between the two.

Nothing here is mocked. The run goes through the same ``Proteus.start()`` as a
real simulation, and every helpfile column it writes is compared against the
recorded one.

Contract clauses exercised:

- The run reproduces the recorded trajectory: same number of rows, and every
  column agreeing within the tolerance in ``tests/helpers/_trajectory.py``.
- The recorded trajectory was taken from the configuration in this directory,
  not from an earlier version of it.
- The comparison covers the whole helpfile rather than a handful of columns,
  and would fail on a change of ordinary size.

That the recorded trajectory is worth comparing against at all, meaning that
it belongs to this configuration and covers a stretch where quantities are
moving, is checked in ``tests/helpers/test_trajectory.py``. Reading the
recorded file needs no run, so those checks sit in the tier that runs on every
pull request rather than waiting for this one.

Scope. Every module runs on its dummy backend, which is what makes the run
cheap enough to repeat and reproducible to the last bit on one machine. That
fixes what this file can and cannot show:

- It covers the coupling framework: the main loop, the timestep controller,
  the module handshakes, the helpfile round-trip and everything the
  configuration reaches. A change to any of them moves the trajectory.
- It does not cover the physics modules themselves. AGNI, SPIDER, Aragog and
  the rest are not exercised here, and their own solvers are not bit
  reproducible in the way this comparison needs.
- The tolerance is loose enough to absorb the last-bit differences between one
  platform's maths library and another's, and far tighter than any change of
  behaviour. It is not a statement about physical accuracy.
- Nothing on this configuration's path iterates to a tolerance, which is what
  keeps that claim true: an iterative solve would put its own convergence
  tolerance into the trajectory, at a level the comparison resolves. The
  configuration says where it avoids one.

While working on a change, running this test reports which quantities moved,
at which row, and by how much relative to the tolerance::

    pytest tests/integration/test_integration_golden_run.py

Recording the trajectory again, once a change to it is the intended outcome::

    pytest tests/integration/test_integration_golden_run.py --record-golden

Do that in the same commit as whatever moved it, so the diff of
``golden_run.tsv`` shows which quantities changed and by how much.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import pytest
from _trajectory import check_against_reference, record_reference, run_trajectory
from helpers import PROTEUS_ROOT

# Integration tier. The run itself takes a few seconds on the dummy backends;
# the 300 s ceiling leaves room for slower CI runners.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CONFIG = PROTEUS_ROOT / 'tests' / 'integration' / 'golden_run.toml'
REFERENCE = PROTEUS_ROOT / 'tests' / 'integration' / 'golden_run.tsv'

# Floor on how much of the helpfile the comparison has to cover. The
# configuration writes about 757 columns; a filter or a schema change that
# started dropping most of them would otherwise leave the comparison passing
# over almost nothing, since a reference recorded from the same narrowed run
# would agree with it.
#
# Set to about 86% of what the configuration writes, so the floor still
# catches a collapse while leaving room for a schema change that retires a
# handful of columns. It has to be raised whenever the helpfile grows
# substantially, or it stops discriminating: at 400 it would accept losing
# nearly half of the columns written today.
MIN_COLUMNS_COMPARED = 650

# Column perturbed to show the comparison can fail on this data, and by how
# much. Stated as a size rather than as a multiple of the tolerance, so that
# loosening the tolerance fails this guard instead of moving it: what the
# check is worth depends on the smallest change it still reports, not on
# whether it reports changes larger than whatever it currently accepts.
#
# One part in 1e5 is far below any behavioural change and far above the
# last-bit drift the tolerance absorbs. It is also well inside what the check
# resolves in practice: scaling dTdt in interior_energetics/dummy.py by
# 1 + 1e-7 is reported at 4.9 times the tolerance and 1 + 1e-6 at 49 times,
# while 1 + 2e-8 sits inside it, so the threshold is just above 2e-8. The
# column that reports it is not the one perturbed, since a change to the
# interior is amplified on its way into the volatile partitioning.
GUARD_COLUMN = 'T_magma'
GUARD_RELATIVE_SIZE = 1.0e-5


def test_fixed_run_reproduces_the_recorded_trajectory(tmp_path, request):
    """The fixed configuration produces the trajectory recorded beside it.

    Physical scenario: a magma-ocean planet on all-dummy backends is evolved
    from a fully molten mantle to a fully solid one, losing volatiles to
    escape throughout, and every quantity the run records is compared against
    the same run recorded earlier.

    Verifies:
    - The recorded trajectory belongs to the configuration in this directory.
      A configuration edited without recording the trajectory again would
      otherwise fail below as an unexplained numerical difference.
    - The run stores the same number of rows as the recording, so no step was
      added, dropped or split.
    - Every helpfile column agrees within tolerance, and the comparison covers
      substantially the whole helpfile rather than a residue of it.
    - The comparison fails on this trajectory when a single column is moved by
      one part in 1e5, so a passing comparison is evidence rather than an
      artefact of comparing something to itself, and the tolerance cannot be
      loosened past that without this failing.

    Under ``--record-golden`` the run is recorded as the new reference and
    nothing is compared, so the test reports itself as skipped rather than as
    having held the run to anything.
    """
    frame = run_trajectory(CONFIG, tmp_path / 'golden')

    if request.config.getoption('--record-golden'):
        # Reported through the skip rather than printed, so that what moved is
        # visible under the suite's ordinary output capture. A print here is
        # swallowed unless the caller also passes -s, which would leave a
        # recording that says nothing about what it accepted.
        pytest.skip(record_reference(frame, REFERENCE, CONFIG))

    check = check_against_reference(frame, REFERENCE, CONFIG)
    assert check.reproduces, (
        f'the run no longer reproduces the recorded trajectory.\n{check.report}\n'
        'If the change was intended, record the trajectory again with '
        '--record-golden in the same commit, so the diff shows what moved.'
    )
    assert len(check.comparison.compared) >= MIN_COLUMNS_COMPARED, (
        f'only {len(check.comparison.compared)} columns were compared, below the '
        f'{MIN_COLUMNS_COMPARED} this configuration writes; the comparison is no '
        'longer covering the helpfile'
    )

    # The comparison has to be able to fail on this data, not only on
    # constructed inputs. Moving one column by one part in 1e5 is far smaller
    # than any behavioural change and has to be reported.
    perturbed = frame.copy()
    perturbed[GUARD_COLUMN] = perturbed[GUARD_COLUMN] * (1.0 + GUARD_RELATIVE_SIZE)
    guard = check_against_reference(perturbed, REFERENCE, CONFIG)
    assert not guard.reproduces, (
        f'moving {GUARD_COLUMN} by one part in {1.0 / GUARD_RELATIVE_SIZE:.0e} was not '
        'reported, so the comparison above cannot fail on this trajectory'
    )
    assert guard.comparison is not None, (
        f'the perturbed run was refused before any column was compared: {guard.report}'
    )
    guard_columns = [difference.column for difference in guard.comparison.differences]
    assert guard_columns == [GUARD_COLUMN], (
        'perturbing one column was reported on more than that column, so the '
        'comparison is not localising a difference to where it happened'
    )
