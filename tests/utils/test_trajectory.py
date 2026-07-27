"""Unit tests for trajectory comparison in `proteus.utils.trajectory`.

The module records a run's helpfile trajectory and compares a later run
against it, which is how a change that is meant to leave behaviour untouched
is shown to have left it untouched. Its whole value rests on the comparison
being able to fail, so most of what follows pins the boundary between agreement
and disagreement from both sides rather than checking that identical inputs
agree.

Contract clauses exercised:

- A recorded reference reads back as the trajectory that was written,
  including columns that hold one value throughout and columns that carry
  undefined values.
- A trajectory that cannot be recorded without losing information is refused
  rather than recorded incompletely, and a reference file that has been
  corrupted is refused rather than read as a shorter trajectory.
- Agreement is decided per column against a tolerance built from that
  column's own scale, so a quantity that passes through zero is judged on the
  same terms along its whole length.
- An undefined value counts as state: two undefined values agree, and an
  undefined value against a number is reported. A column corrupted into NaN
  is exactly what the comparison exists to catch.
- A column that appears or disappears is reported, as is a run of a different
  length, since neither can be judged value by value.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proteus.utils.trajectory import (
    DEFAULT_ATOL_SCALE,
    DEFAULT_RTOL,
    ReferenceFormatError,
    compare_trajectories,
    config_digest,
    read_reference,
    run_trajectory,
    write_reference,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _frame(**columns) -> pd.DataFrame:
    """Build a trajectory from column name to values."""
    return pd.DataFrame(
        {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    )


def _cooling_frame() -> pd.DataFrame:
    """Build a trajectory shaped like a short cooling run.

    Carries a quantity that falls over the run, one that holds a single value
    throughout, and one that crosses zero, which are the three shapes the
    comparison has to treat differently.
    """
    return _frame(
        T_magma=[3000.0, 2800.0, 2600.0, 2400.0],
        M_planet=[5.972e24] * 4,
        F_net=[1.0e3, 1.0e1, -1.0e1, -1.0e3],
        runtime=[0.5, 1.1, 1.7, 2.4],
    )


def test_reference_round_trips_constant_varying_and_undefined_columns(tmp_path):
    """A recorded reference reads back as the trajectory that was written.

    Contract clause: recording is lossless for every column shape the
    helpfile produces, so a comparison against the file means the same thing
    as a comparison against the run that produced it.

    Verifies:
    - A column that varies, a column that holds one value throughout and a
      column that is undefined throughout all read back unchanged.
    - A column that becomes undefined part way through keeps the position of
      the undefined value, rather than collapsing to a constant.
    - Values round-trip exactly rather than to a stated number of digits, so a
      comparison against the file means the same as one against the run.
    - The constant column is stored once rather than once per row, which is
      what keeps the file small enough to read in a diff.
    - Columns that record how the run was executed are left out.
    - Blank lines are tolerated, so a file that picked one up in an editor
      still reads rather than failing as corrupted.
    """
    # Values chosen with full mantissas rather than round numbers: a format
    # that truncated to a fixed number of digits would round-trip 3000.0
    # exactly and lose these, so round numbers alone would not show it.
    frame = _frame(
        T_magma=[3000.123456789012, 2800.987654321098, 2600.111111111111, 2400.0],
        M_planet=[5.9722000000001e24] * 4,
        undefined=[np.nan] * 4,
        partly_undefined=[1.0 / 3.0, np.nan, np.pi, 4.0],
        runtime=[0.5, 1.1, 1.7, 2.4],
    )
    path = tmp_path / 'reference.tsv'
    write_reference(frame, path)
    path.write_text(path.read_text() + '\n')

    restored = read_reference(path)
    assert restored.config_digest == '', 'no configuration was given, so none is recorded'
    assert 'runtime' not in restored.frame.columns, (
        'wall-clock time is not simulated state and must not be recorded'
    )
    for column in ('T_magma', 'M_planet', 'undefined', 'partly_undefined'):
        np.testing.assert_array_equal(
            restored.frame[column].to_numpy(),
            frame[column].to_numpy(),
            err_msg=f'{column} did not survive the round trip',
        )

    lines = {
        line.split('\t')[0]: line.split('\t')
        for line in path.read_text().splitlines()
        if not line.startswith('#')
    }
    # Three fields is a name, a kind and one value; the varying column carries
    # one field per row on top of the name and kind.
    assert len(lines['M_planet']) == 3, 'a constant column is stored more than once'
    assert len(lines['undefined']) == 3, 'an all-undefined column is stored more than once'
    assert len(lines['T_magma']) == 2 + len(frame), 'a varying column lost rows'
    assert len(lines['partly_undefined']) == 2 + len(frame), (
        'a column with one undefined value collapsed to a constant, which would '
        'lose where the value became undefined'
    )


def test_reference_records_the_configuration_it_was_taken_from(tmp_path):
    """The reference carries a digest of the configuration behind it.

    Contract clause: a reference is only meaningful against one configuration,
    so the file records which one, and a comparison can tell that the
    configuration has since been edited.

    Verifies:
    - The digest stored in the reference is the digest of the configuration
      file's bytes.
    - Editing the configuration changes that digest, so the two no longer
      agree. A digest that did not move on an edit would leave the guard
      passing over exactly the change it exists to catch.
    """
    config = tmp_path / 'run.toml'
    config.write_text('mass_tot = 1.0\n')
    original = config_digest(config)

    path = tmp_path / 'reference.tsv'
    write_reference(_cooling_frame(), path, config_path=config)
    assert read_reference(path).config_digest == original

    config.write_text('mass_tot = 2.0\n')
    assert config_digest(config) != original, (
        'the digest did not move when the configuration did, so it cannot detect '
        'a reference recorded against a different configuration'
    )


def test_write_reference_refuses_what_it_cannot_record(tmp_path):
    """Recording refuses a trajectory it would have to record incompletely.

    Contract clause: a reference that silently dropped part of a run would
    look complete while covering less than it claims, so both cases raise
    instead.

    Verifies:
    - A trajectory with no rows is refused, since there is nothing to compare
      against later.
    - A trajectory carrying a non-numeric column is refused, and the message
      names the offending column so the format can be extended deliberately.
    """
    with pytest.raises(ValueError, match='no rows'):
        write_reference(_frame(T_magma=[]), tmp_path / 'empty.tsv')

    frame = _cooling_frame()
    frame['status'] = ['ok', 'ok', 'ok', 'ok']
    with pytest.raises(ValueError, match='status'):
        write_reference(frame, tmp_path / 'mixed.tsv')
    assert not (tmp_path / 'mixed.tsv').exists(), (
        'a refused recording left a partial file behind, which a later comparison '
        'would read as a valid reference'
    )


@pytest.mark.parametrize(
    ('body', 'match'),
    [
        ('# rows = 2\nT_magma\tseries\t1.0\t2.0\t3.0\n', 'against the 2 rows'),
        ('# rows = 2\nT_magma\tconst\t1.0\t2.0\n', 'carries one value'),
        ('# rows = 2\nT_magma\tramp\t1.0\t2.0\n', 'unknown column kind'),
        ('# rows = 2\nT_magma\tseries\n', 'at least one'),
        ('T_magma\tseries\t1.0\t2.0\n', 'before the row count'),
        ('# columns = 1\n', 'no row count'),
        (
            '# rows = 2\n# columns = 3\nT_magma\tseries\t1.0\t2.0\n',
            'been edited since it was written',
        ),
        (
            '# rows = 2\n# columns = 1\nT_magma\tconst\t1.0\nP_surf\tconst\t2.0\n',
            'been edited since it was written',
        ),
        ('# rows = 2\nT_magma\tconst\t1.0\nT_magma\tconst\t2.0\n', 'appears twice'),
    ],
)
def test_read_reference_refuses_a_corrupted_file(tmp_path, body, match):
    """A reference that has lost its integrity is refused, not read partially.

    Contract clause: the comparison treats the reference as ground truth, so a
    file that has been truncated, hand-edited or written by an incompatible
    version must fail loudly rather than resolve to a shorter or emptier
    trajectory that a run could accidentally reproduce.

    Verifies, one case per corruption:
    - A column longer or shorter than the declared row count is refused.
    - A constant column carrying more than one value is refused.
    - A column kind the format does not define is refused.
    - A column line with no values at all is refused.
    - Column data appearing before the row count is refused, since the length
      of a constant column is undefined until the count is known.
    - A file with no row count at all is refused.
    - A file whose column count disagrees with its header is refused, in
      either direction, rather than read as a reference the run would then be
      measured against. A message written for one direction alone would be
      false in the other, so it names neither.
    - A column appearing twice is refused, since which of its two trajectories
      counts is undefined.
    """
    path = tmp_path / 'broken.tsv'
    path.write_text(body)
    with pytest.raises(ReferenceFormatError, match=match):
        read_reference(path)


def test_comparison_boundary_is_where_the_tolerance_says_it_is():
    """Agreement is decided at the stated tolerance, from both sides.

    Contract clause: a change smaller than the tolerance is not reported and a
    change larger than it is. Testing only one side would pass just as happily
    if the comparison reported everything, or nothing.

    Verifies:
    - A perturbation at a tenth of the relative tolerance is not reported.
    - A perturbation at ten times it is, and the report names the column, the
      row, and both values.
    - The reported deviation is the size of the disagreement in units of the
      tolerance allowed there, so a perturbation ten times the tolerance is
      reported as such rather than as a bare failure.
    """
    reference = _cooling_frame()

    inside = reference.copy()
    inside.loc[2, 'T_magma'] *= 1.0 + 0.1 * DEFAULT_RTOL
    assert compare_trajectories(reference, inside).agrees, (
        'a change well inside the tolerance was reported, so ordinary platform '
        'drift would fail the comparison'
    )

    outside = reference.copy()
    outside.loc[2, 'T_magma'] *= 1.0 + 10.0 * DEFAULT_RTOL
    comparison = compare_trajectories(reference, outside)
    assert not comparison.agrees
    assert len(comparison.differences) == 1, (
        f'one column was perturbed but {len(comparison.differences)} were reported'
    )
    difference = comparison.differences[0]
    assert difference.column == 'T_magma'
    assert difference.row == 2, f'the perturbed row is 2, reported as {difference.row}'
    assert difference.expected == pytest.approx(2600.0)
    assert difference.actual == pytest.approx(2600.0 * (1.0 + 10.0 * DEFAULT_RTOL))
    # Ten times the tolerance, to within the absolute term that is added to it.
    assert 9.0 < difference.deviation < 11.0, (
        f'a perturbation of ten tolerances was reported as {difference.deviation:.3g}'
    )
    assert 'T_magma' in comparison.report()
    assert 'M_planet' not in comparison.report(), 'an untouched column was reported'


def test_comparison_resolves_a_change_of_one_part_in_a_hundred_thousand():
    """The default tolerance sits in the band it is meant to sit in.

    Contract clause: the comparison has to report a change small enough that
    no real one hides under it, while accepting the last-bit differences
    between one platform's maths library and another's. Both ends are stated
    as sizes rather than as multiples of the tolerance, so the tolerance
    cannot be moved without one of them failing. A test written in terms of
    the constant follows it wherever it goes and pins nothing.

    Verifies:
    - A relative change of 1e-5 is reported, which bounds how far the
      tolerance can be loosened.
    - A relative change of 1e-9 is not, which bounds how far it can be
      tightened before ordinary platform drift starts failing runs that
      changed nothing.
    """
    reference = _cooling_frame()

    resolved = reference.copy()
    resolved.loc[1, 'T_magma'] *= 1.0 + 1.0e-5
    comparison = compare_trajectories(reference, resolved)
    assert not comparison.agrees, (
        'a relative change of 1e-5 went unreported, so the tolerance is loose '
        'enough to hide a change worth knowing about'
    )
    assert comparison.differences[0].column == 'T_magma'

    drift = reference.copy()
    drift.loc[1, 'T_magma'] *= 1.0 + 1.0e-9
    assert compare_trajectories(reference, drift).agrees, (
        'a relative change of 1e-9 was reported, so the tolerance is tight enough '
        'that platform drift alone would fail a run that changed nothing'
    )


def test_a_column_is_collapsed_only_when_every_value_is_the_same_bit_pattern(tmp_path):
    """A nearly-constant column is recorded row by row, not as one value.

    Contract clause: collapsing a column to a single value asserts that the
    column holds that value at every row. A column that merely stays close to
    one value carries a trajectory, and collapsing it would erase exactly the
    small drift a comparison against this file is meant to resolve.

    Verifies:
    - A column whose values differ in the last bits is stored as a series.
    - Reading it back returns those distinct values rather than one repeated
      value, so a later comparison still sees the drift.
    """
    nearly = 5.972e24
    frame = _frame(M_planet=[nearly, nearly * (1.0 + 1.0e-12), nearly])
    destination = tmp_path / 'reference.tsv'
    write_reference(frame, destination)

    line = next(
        row for row in destination.read_text().splitlines() if row.startswith('M_planet')
    )
    assert line.split('\t')[1] == 'series', (
        'a column that only stays close to one value was collapsed to that value, '
        'which erases the drift the comparison exists to resolve'
    )

    restored = read_reference(destination).frame['M_planet'].to_numpy()
    assert restored[1] != restored[0], 'the round trip flattened a varying column'
    np.testing.assert_array_equal(restored, frame['M_planet'].to_numpy())


def test_the_tolerance_follows_the_magnitude_of_the_quantity():
    """A column's tolerance scales with the column, not with a fixed number.

    Contract clause: the same relative change has to be judged the same way
    whether a quantity is measured in single Kelvin or in units of 1e23, so
    the band is proportional rather than absolute.

    Note on what this does not pin. The band is built from the reference
    rather than from the run, so it does not move with the trajectory under
    test. That choice is not observable here: the two formulations differ by
    the relative tolerance times the difference itself, so they disagree only
    where a difference sits within one part in a million of the threshold,
    and a test placed there would turn on the last bits of an arithmetic
    result rather than on the choice.

    Verifies:
    - A difference of one part in 1e6 is accepted at a magnitude of 1e6, where
      the same absolute difference at a magnitude of 1 is not.
    - A doubling is reported at either magnitude, so the proportional band has
      not simply accepted everything at the larger one.
    """
    large_reference = _frame(F_atm=[1.0e6, 1.0e6, 1.0e6])
    assert compare_trajectories(
        large_reference, _frame(F_atm=[1.0e6, 1.0e6 - 1.0, 1.0e6])
    ).agrees, 'one part in 1e6 was reported, so the band is narrower than the tolerance'
    assert not compare_trajectories(
        large_reference, _frame(F_atm=[1.0e6, 2.0e6, 1.0e6])
    ).agrees, 'a doubling was accepted at a magnitude of 1e6'

    small_reference = _frame(F_atm=[1.0, 1.0, 1.0])
    assert not compare_trajectories(small_reference, _frame(F_atm=[1.0, 2.0, 1.0])).agrees, (
        'the same absolute difference of 1.0 was accepted at a magnitude of 1, where it '
        'is a doubling; the band is not proportional'
    )


def test_the_absolute_term_is_a_fixed_fraction_of_the_column():
    """The floor under the tolerance is set by the column's own largest value.

    Contract clause: the absolute term exists so that a quantity passing
    through zero is judged on the same terms as the rest of its column, and it
    is sized as a fraction of what the column reaches. Both halves are stated
    as sizes rather than as multiples of the constant, so a term hard-coded to
    some other number fails one of them.

    Verifies:
    - At a column maximum of 1e6, a difference of 1e-7 at a zero-valued row is
      accepted, since the floor there is a millionth of a millionth of 1e6.
    - The same 1e-7 difference is reported when the column only reaches 1.0,
      where the floor is a millionth of a millionth of 1.0 and far below it.
    - A difference of 1e-3 is reported at the larger magnitude too, so the
      floor has not simply swallowed the comparison at the zero crossing.
    """
    large = _frame(F_net=[1.0e6, 0.0, -1.0e6])
    assert compare_trajectories(large, _frame(F_net=[1.0e6, 1.0e-7, -1.0e6])).agrees, (
        'a difference of 1e-7 at the zero crossing of a column reaching 1e6 was '
        'reported; the floor is smaller than a fraction of the column'
    )
    assert not compare_trajectories(large, _frame(F_net=[1.0e6, 1.0e-3, -1.0e6])).agrees, (
        'a difference of 1e-3 at the zero crossing was accepted; the floor is larger '
        'than a fraction of the column'
    )

    small = _frame(F_net=[1.0, 0.0, -1.0])
    assert not compare_trajectories(small, _frame(F_net=[1.0, 1.0e-7, -1.0])).agrees, (
        'the same 1e-7 difference was accepted where the column only reaches 1.0, so '
        'the floor is a fixed number rather than a fraction of the column'
    )


def test_comparison_ignores_how_long_the_run_took():
    """Wall-clock time is not part of the trajectory.

    Contract clause: two runs of identical code take different amounts of
    time, so a comparison that included the clock would report a difference
    every time and be worthless.

    Verifies:
    - A run whose recorded runtime is an order of magnitude different from the
      reference still agrees.
    - The runtime column is not among the columns reported as compared, so it
      is skipped rather than passing by luck.
    """
    reference = _cooling_frame()
    slower = reference.copy()
    slower['runtime'] = slower['runtime'] * 10.0

    comparison = compare_trajectories(reference, slower)
    assert comparison.agrees
    assert 'runtime' not in comparison.compared
    # The agreeing report states how much was compared rather than saying
    # nothing, so a run that covered almost none of the helpfile does not read
    # the same as one that covered all of it.
    assert comparison.report() == '3 columns over 4 rows reproduce the reference'


def test_comparison_treats_an_undefined_value_as_state():
    """An undefined value is compared, not skipped.

    Contract clause: a column a change corrupted into NaN is exactly what the
    comparison exists to catch, so undefined values are compared rather than
    filtered out. Screening non-finite values before comparing would skip
    precisely those columns.

    Verifies:
    - Two undefined values at the same row agree, so a quantity a
      configuration never defines is not reported every run.
    - An undefined value against a number is reported, in both directions.
    - Infinities agree with themselves and disagree with a number, so the
      overflow case is covered alongside the undefined one.
    """
    reference = _frame(a=[1.0, np.nan, 3.0], b=[np.inf, 1.0, 2.0])
    assert compare_trajectories(reference, reference.copy()).agrees, (
        'a trajectory does not reproduce itself once it carries undefined values'
    )

    corrupted = reference.copy()
    corrupted.loc[0, 'a'] = np.nan
    assert not compare_trajectories(reference, corrupted).agrees, (
        'a value that became undefined was not reported'
    )

    recovered = reference.copy()
    recovered.loc[1, 'a'] = 2.0
    assert not compare_trajectories(reference, recovered).agrees, (
        'a value that stopped being undefined was not reported'
    )

    overflowed = reference.copy()
    overflowed.loc[0, 'b'] = 1.0e308
    assert not compare_trajectories(reference, overflowed).agrees, (
        'a value that stopped being infinite was not reported'
    )


def test_comparison_tolerance_follows_the_scale_of_the_column():
    """A quantity that crosses zero is judged on its own scale.

    Contract clause: a relative tolerance alone demands exact agreement
    wherever a quantity passes through zero, which is where cancellation makes
    agreement least exact. The absolute term is scaled to the largest value
    the column reaches, so one meaning of unchanged holds along its length.

    Verifies:
    - A perturbation at the zero crossing that is negligible against the
      column's own scale is not reported.
    - A perturbation there that is large against that scale is reported, so
      the absolute term has not simply disabled the comparison near zero.
    """
    reference = _frame(F_net=[1.0e3, 1.0e1, 0.0, -1.0e3])
    scale = 1.0e3

    negligible = reference.copy()
    negligible.loc[2, 'F_net'] = 0.1 * DEFAULT_ATOL_SCALE * scale
    assert compare_trajectories(reference, negligible).agrees, (
        'a difference far below the scale of the column was reported at its zero '
        'crossing, so a relative tolerance is being applied to a zero'
    )

    substantial = reference.copy()
    substantial.loc[2, 'F_net'] = 1.0e-3 * scale
    comparison = compare_trajectories(reference, substantial)
    assert not comparison.agrees, (
        'a difference of a thousandth of the column scale was accepted at its zero '
        'crossing, so the absolute term has swallowed the comparison there'
    )
    assert comparison.differences[0].row == 2


def test_comparison_reports_the_worst_row_of_a_column():
    """The row a difference names is the largest one, not the first.

    Contract clause: a column is reported once, so the row it carries has to
    be the one that says the most about the disagreement.

    Verifies:
    - With two rows outside tolerance, the larger is reported.
    - Ordering across columns follows the same rule, so the first difference
      in the report is the worst one anywhere.
    """
    reference = _frame(a=[100.0, 100.0, 100.0], b=[10.0, 10.0, 10.0])
    actual = reference.copy()
    actual.loc[0, 'a'] = 100.0 * (1.0 + 10.0 * DEFAULT_RTOL)
    actual.loc[2, 'a'] = 100.0 * (1.0 + 1.0e3 * DEFAULT_RTOL)
    actual.loc[1, 'b'] = 10.0 * (1.0 + 1.0e2 * DEFAULT_RTOL)

    comparison = compare_trajectories(reference, actual)
    by_column = {difference.column: difference for difference in comparison.differences}
    assert by_column['a'].row == 2, (
        f'column a disagrees most at row 2, reported at row {by_column["a"].row}'
    )
    assert comparison.differences[0].column == 'a', (
        'the report does not lead with the largest disagreement'
    )


def test_comparison_reports_columns_that_appear_or_disappear():
    """A change to the set of recorded quantities is reported.

    Contract clause: a column that the run no longer produces, or produces for
    the first time, cannot be compared value by value, and passing over it
    silently would let a change to the helpfile schema go unnoticed.

    Verifies:
    - A column the reference holds and the run does not is reported as
      missing, and one the run holds and the reference does not as unexpected.
    - Neither appears among the compared columns, and both are reported ahead
      of ordinary value differences, since a schema change explains any number
      of them.
    """
    reference = _frame(kept=[1.0, 2.0], dropped=[3.0, 4.0])
    actual = _frame(kept=[1.0, 2.0], added=[5.0, 6.0])

    comparison = compare_trajectories(reference, actual)
    assert not comparison.agrees
    reasons = {difference.column: difference.reason for difference in comparison.differences}
    assert reasons == {'dropped': 'missing', 'added': 'unexpected'}
    assert comparison.compared == ('kept',)
    assert 'absent from the run' in comparison.report()
    assert 'absent from the reference' in comparison.report()


def test_comparison_reports_a_column_the_run_stopped_writing_as_a_number():
    """A recorded quantity that stops being a number is reported, not raised on.

    Contract clause: a reference holds numbers only, so a run that writes text
    or an object where a number was recorded has changed the helpfile schema.
    That is a difference to report, not a conversion failure to surface from
    inside the comparison, where it would name a numpy error rather than the
    column that changed.

    Verifies:
    - The comparison completes and reports the column rather than raising.
    - The report says what happened to it, and the untouched column beside it
      is still compared, so one changed column does not stop the rest.
    """
    reference = _frame(T_magma=[3000.0, 2800.0], P_surf=[1.0e5, 2.0e5])
    actual = reference.copy()
    actual['T_magma'] = ['hot', 'less hot']

    comparison = compare_trajectories(reference, actual)
    assert not comparison.agrees
    assert [difference.reason for difference in comparison.differences] == ['non_numeric']
    assert 'other than a number' in comparison.report()
    assert comparison.compared == ('P_surf',), (
        'a column that stopped being numeric prevented the rest from being compared'
    )


def test_comparison_reports_a_run_of_a_different_length():
    """Trajectories of different lengths are reported as such.

    Contract clause: a run with more or fewer iterations has taken different
    steps, and lining the two up row by row would compare unrelated states, so
    the length is reported on its own instead.

    Verifies:
    - A shorter run does not agree, and reports both lengths rather than a
      list of columns.
    - No column is reported as compared, so a caller counting compared columns
      cannot mistake a length mismatch for coverage.
    """
    reference = _frame(T_magma=[3000.0, 2800.0, 2600.0])
    shorter = _frame(T_magma=[3000.0, 2800.0])

    comparison = compare_trajectories(reference, shorter)
    assert not comparison.agrees
    assert comparison.compared == ()
    assert comparison.differences == ()
    report = comparison.report()
    assert '2 rows' in report and '3' in report


def test_report_truncates_a_long_list_of_differences():
    """A report of many differences stays readable.

    Contract clause: a change that moves the whole trajectory produces one
    difference per column, and a failure message listing hundreds of them
    buries the useful part.

    Verifies:
    - Only the requested number of differences is listed, worst first.
    - The count of those left out is stated, so the reader is not left
      thinking the listed ones are all of them.
    """
    columns = {f'c{index}': [1.0, 2.0] for index in range(10)}
    reference = _frame(**columns)
    actual = reference * (1.0 + 1.0e3 * DEFAULT_RTOL)

    comparison = compare_trajectories(reference, actual)
    assert len(comparison.differences) == 10
    report = comparison.report(limit=3)
    assert report.count('against a recorded') == 3
    assert 'and 7 more' in report


def test_run_trajectory_refuses_a_configuration_that_fixes_its_output(tmp_path):
    """A configuration naming a real output directory is not redirected.

    Contract clause: the helper redirects a configuration that has not chosen
    where to write. Redirecting one that has would send a run somewhere the
    caller did not ask for, and quietly overwrite whatever is there.

    Verifies:
    - A configuration whose output path is a real directory raises, and the
      message names the configuration.
    - No run is started, so the refusal costs nothing.
    """
    config = tmp_path / 'fixed.toml'
    config.write_text('[params.out]\n    path = "output/somewhere"\n')
    with pytest.raises(ValueError, match='0 times'):
        run_trajectory(config, tmp_path / 'run')

    # Two of them is refused for the same reason: which one the run would have
    # written to is not something the caller stated.
    ambiguous = tmp_path / 'twice.toml'
    ambiguous.write_text('[params.out]\n    path = "auto"\n[other]\n    path = "auto"\n')
    with pytest.raises(ValueError, match='2 times'):
        run_trajectory(ambiguous, tmp_path / 'run')


def _fake_proteus(monkeypatch, seen, frame, *, output, fails_in=None):
    """Install a stand-in for Proteus that records how it was called.

    Allocates its scratch directory under ``TMPDIR`` the way the framework
    does, and can be made to fail either while being constructed or while
    running, since those leave the caller with different amounts of the
    object to clean up after.
    """
    import proteus

    class FakeProteus:
        def __init__(self, *, config_path):
            seen['config_path'] = config_path
            seen['tmpdir'] = os.environ.get('TMPDIR')
            scratch = Path(os.environ['TMPDIR']) / 'proteus_1234567890'
            scratch.mkdir(parents=True, exist_ok=True)
            (scratch / 'workfile').write_text('scratch\n')
            seen['scratch'] = scratch
            if fails_in == 'construction':
                raise RuntimeError('the configuration was rejected')
            self.directories = {'temp': str(scratch), 'output': str(output)}
            self.hf_all = frame

        def start(self, *, resume, offline):
            seen['resume'] = resume
            seen['offline'] = offline
            if fails_in == 'start':
                raise RuntimeError('the run failed')

    monkeypatch.setattr(proteus, 'Proteus', FakeProteus)


def test_run_trajectory_redirects_the_output_and_returns_the_trajectory(tmp_path, monkeypatch):
    """The helper runs the copied configuration, not the original.

    Contract clause: the configuration under version control keeps its
    ``"auto"`` output path, and the run reads a copy pointed at the requested
    directory.

    Verifies:
    - The configuration the run is handed is a copy, not the original file.
    - That copy names the requested output directory, and the original is
      left untouched.
    - A directory name that looks like a regular-expression group reference
      lands in the copy literally, rather than being read as a reference to
      part of the pattern it replaced.
    - The run is started without resuming, and its trajectory is returned.
    - The scratch directory the run allocated is gone afterwards.
    """
    config = tmp_path / 'source.toml'
    config.write_text('[params.out]\n    path = "auto"\n')
    original = config.read_text()
    frame = _frame(T_magma=[3000.0, 2800.0])
    seen: dict[str, object] = {}

    # A backslash group reference in the directory name. Interpolated into a
    # replacement string it would expand to part of the matched text and
    # corrupt the configuration silently.
    output_dir = tmp_path / r'run\g<lead>\1'
    before = os.environ.get('TMPDIR')
    _fake_proteus(monkeypatch, seen, frame, output=output_dir)

    returned = run_trajectory(config, output_dir)

    assert seen['config_path'] != config, 'the run was handed the original configuration'
    assert f'path = "{output_dir}"' in Path(seen['config_path']).read_text(), (
        'the redirected output path is not the directory that was asked for'
    )
    assert config.read_text() == original, 'the original configuration was edited'
    assert seen['resume'] is False
    assert seen['offline'] is True
    assert not seen['scratch'].exists(), (
        'the scratch directory the run allocated was left behind'
    )
    assert os.environ.get('TMPDIR') == before, (
        'the redirected scratch location was left in the environment after the run'
    )
    pd.testing.assert_frame_equal(returned, frame)
    assert returned is not frame, "the caller was handed the runner's own frame to mutate"


def test_run_trajectory_leaves_an_unset_scratch_location_unset(tmp_path, monkeypatch):
    """The run does not introduce a scratch location that was not there.

    Contract clause: the helper redirects where the framework puts its scratch
    space and puts the environment back afterwards. Putting it back means
    absent when it was absent, not set to whatever value happened to be in
    use, since a later caller would then inherit a directory this run removed.

    Verifies:
    - With no scratch location set beforehand, there is none afterwards.
    - The run still worked, so the restoration is not covering for a run that
      never happened.
    """
    monkeypatch.delenv('TMPDIR', raising=False)
    config = tmp_path / 'source.toml'
    config.write_text('[params.out]\n    path = "auto"\n')
    frame = _frame(T_magma=[3000.0, 2800.0])
    seen: dict[str, object] = {}
    _fake_proteus(monkeypatch, seen, frame, output=tmp_path / 'run')

    returned = run_trajectory(config, tmp_path / 'run')

    assert 'TMPDIR' not in os.environ, (
        'the run left a scratch location behind in an environment that had none, '
        'pointing at a directory it has since removed'
    )
    assert seen['tmpdir'] is not None, 'the run was not given a scratch location at all'
    pd.testing.assert_frame_equal(returned, frame)


def test_run_trajectory_cleans_up_after_a_failed_run(tmp_path, monkeypatch):
    """A run that raises does not leave its scratch directory behind.

    Contract clause: the scratch directory belongs to the run, and the
    framework never removes it. A failing run is the case the harness exists
    to catch, so it is the one that must not leak.

    Verifies, for a failure while the run is starting and for one while the
    runner is being constructed:
    - The failure reaches the caller rather than being swallowed.
    - The scratch directory is gone even though the run raised.

    The construction case is the one that needs the scratch space to be
    reachable without a runner. The framework allocates it inside the
    constructor, before it has checked the configuration, so a configuration
    carrying one unrecognised key raises out of the constructor with the
    directory already on disk and nothing left holding a reference to it.
    """
    config = tmp_path / 'source.toml'
    config.write_text('[params.out]\n    path = "auto"\n')

    for stage, message in (('start', 'the run failed'), ('construction', 'was rejected')):
        seen: dict[str, object] = {}
        _fake_proteus(
            monkeypatch,
            seen,
            _frame(T_magma=[1.0]),
            output=tmp_path / 'run',
            fails_in=stage,
        )
        with pytest.raises(RuntimeError, match=message):
            run_trajectory(config, tmp_path / 'run')
        assert not seen['scratch'].exists(), (
            f'a run that failed during {stage} left its scratch directory behind'
        )


def test_run_trajectory_keeps_the_output_when_it_is_also_the_scratch_directory(
    tmp_path, monkeypatch
):
    """The results survive when the framework uses them as its scratch space.

    Contract clause: under DEBUG logging the framework points its scratch
    directory at the output directory itself. The results are what the caller
    asked for, so the cleanup must not be able to reach them whatever the
    framework decided to reuse.

    Verifies:
    - The output directory and a file written into it survive the run.
    - The trajectory is still returned, so this case is not turned into a
      failure.
    """
    config = tmp_path / 'source.toml'
    config.write_text('[params.out]\n    path = "auto"\n')
    shared = tmp_path / 'run'
    shared.mkdir()
    (shared / 'runtime_helpfile.csv').write_text('results\n')
    frame = _frame(T_magma=[3000.0, 2800.0])
    seen: dict[str, object] = {}
    _fake_proteus(monkeypatch, seen, frame, output=shared)

    returned = run_trajectory(config, shared)

    assert (shared / 'runtime_helpfile.csv').is_file(), (
        'the run output was deleted along with the scratch space'
    )
    assert shared.is_dir(), 'the output directory itself was removed'
    pd.testing.assert_frame_equal(returned, frame)
