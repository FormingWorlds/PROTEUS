"""Recording and comparison of run trajectories, for the golden-run check.

A change that is meant to leave behaviour alone can only be shown to have left
it alone by running the same configuration before and after and finding the
same numbers. This module holds everything that decides what "the same
trajectory" means: running a configuration and collecting its helpfile, the
reference file format, the tolerance model, and the column-wise comparison.

Its one consumer is ``tests/integration/test_integration_golden_run.py``, which
compares a fixed configuration against the trajectory recorded beside it, and
records that trajectory again when run with ``--record-golden``.

The tolerance is relative with an absolute term scaled to each column's own
range, so that "unchanged" keeps one meaning along a column that passes through
zero. An undefined value is treated as state rather than screened out: two NaNs
at the same row agree, a NaN against a number does not.

Testing standards: docs/How-to/testing.md,
docs/Explanations/test_framework.md
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from proteus import Proteus

if TYPE_CHECKING:
    from os import PathLike

# Matches the output path of a configuration that has not fixed one. Anchored
# on the literal 'auto' so a configuration naming a real directory is left
# alone rather than silently redirected.
_AUTO_PATH = re.compile(r'^(?P<lead>[ \t]*path[ \t]*=[ \t]*)"auto"', re.MULTILINE)

# Columns that record how the run was executed rather than what it simulated.
# Wall-clock time differs between two runs of identical code, so comparing it
# would report a difference on every run.
NON_STATE_COLUMNS = ('runtime',)

# Relative tolerance for a trajectory comparison. Values round-trip through
# the reference file exactly, so the file contributes nothing to the floor of
# the comparison; what does is the platform, where one maths library differs
# from another by an ulp or so and a coupled loop carries that forward. This
# sits far above such drift and far below the change any behavioural
# difference makes.
DEFAULT_RTOL = 1.0e-6

# Absolute tolerance, as a fraction of the largest magnitude the reference
# column reaches. A relative tolerance alone is unusable where a quantity
# passes through zero, since it demands exact agreement there while accepting
# a wide band a few rows later. Scaling to the column's own range keeps one
# meaning of "unchanged" along the whole column.
#
# A column that is zero at every row therefore gets no tolerance at all, and
# has to stay exactly zero. That is the behaviour to want: such a column is a
# quantity the configuration never switches on, held at a literal zero rather
# than arrived at by cancellation, so any value at all in it is a change.
DEFAULT_ATOL_SCALE = 1.0e-12

_HEADER_PREFIX = '#'
_KIND_SERIES = 'series'
_KIND_CONST = 'const'


class ReferenceFormatError(ValueError):
    """Raised when a stored reference trajectory cannot be read."""


@dataclass(frozen=True)
class Reference:
    """A recorded trajectory and the configuration it was recorded from.

    Attributes
    ----------
    frame : pandas.DataFrame
        The recorded trajectory, one row per iteration.
    config_digest : str
        SHA-256 of the configuration file the trajectory was recorded from,
        as a hex string. Empty when the reference records no digest.
    """

    frame: pd.DataFrame
    config_digest: str


@dataclass(frozen=True)
class ColumnDifference:
    """One column on which two trajectories disagree.

    Attributes
    ----------
    column : str
        Helpfile column name.
    reason : str
        ``'value'`` when the column exists on both sides and disagrees,
        ``'missing'`` when the run did not produce a recorded column,
        ``'unexpected'`` when the run produced a column the reference has no
        record of, ``'non_numeric'`` when the run produced a recorded column
        as something other than a number.
    row : int
        Row index of the largest disagreement; -1 when no row applies.
    expected : float
        Reference value at that row; NaN when no row applies.
    actual : float
        Run value at that row; NaN when no row applies.
    deviation : float
        Size of the disagreement at that row relative to the tolerance
        allowed there. Greater than 1 by construction for a reported
        difference; NaN when no row applies.
    """

    column: str
    reason: str
    row: int = -1
    expected: float = float('nan')
    actual: float = float('nan')
    deviation: float = float('nan')

    def describe(self) -> str:
        """Return a one-line description of this difference."""
        if self.reason == 'missing':
            return f'{self.column}: recorded in the reference, absent from the run'
        if self.reason == 'unexpected':
            return f'{self.column}: produced by the run, absent from the reference'
        if self.reason == 'non_numeric':
            return f'{self.column}: produced by the run as something other than a number'
        return (
            f'{self.column}: row {self.row} is {self.actual:.10e} against a recorded '
            f'{self.expected:.10e}, {self.deviation:.3g}x the tolerance'
        )


@dataclass(frozen=True)
class TrajectoryComparison:
    """Outcome of comparing a run against a recorded reference.

    Attributes
    ----------
    differences : tuple of ColumnDifference
        Every column that disagrees, worst first.
    compared : tuple of str
        Columns that were compared value by value.
    rows_expected : int
        Row count of the reference.
    rows_actual : int
        Row count of the run.
    """

    differences: tuple[ColumnDifference, ...]
    compared: tuple[str, ...]
    rows_expected: int
    rows_actual: int

    @property
    def agrees(self) -> bool:
        """Whether the run reproduces the reference."""
        return not self.differences and self.rows_expected == self.rows_actual

    def report(self, limit: int = 12) -> str:
        """Return a human-readable summary of the comparison.

        Parameters
        ----------
        limit : int, optional
            Maximum number of differing columns to list.
        """
        if self.rows_expected != self.rows_actual:
            return (
                f'the run stored {self.rows_actual} rows against {self.rows_expected} '
                'in the reference, so the two trajectories are not the same length '
                'and cannot be compared row by row'
            )
        if not self.differences:
            return (
                f'{len(self.compared)} columns over {self.rows_actual} rows reproduce '
                'the reference'
            )
        shown = self.differences[:limit]
        lines = [
            f'{len(self.differences)} column(s) differ from the reference over '
            f'{self.rows_actual} rows, {len(self.compared)} agree:'
        ]
        lines.extend(f'  {difference.describe()}' for difference in shown)
        if len(self.differences) > limit:
            lines.append(f'  ... and {len(self.differences) - limit} more')
        return '\n'.join(lines)


def config_digest(path: str | PathLike) -> str:
    """Return the SHA-256 of a configuration file, as a hex string.

    Parameters
    ----------
    path : str or path-like
        File to digest.

    Returns
    -------
    str
        Hex digest of the file's bytes.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_columns(frame: pd.DataFrame, skip: tuple[str, ...]) -> list[str]:
    """Return the columns of ``frame`` that describe simulated state."""
    return [column for column in frame.columns if column not in skip]


def run_trajectory(
    config_path: str | PathLike,
    output_dir: str | PathLike,
    *,
    offline: bool = True,
) -> pd.DataFrame:
    """Run a configuration into a given directory and return its trajectory.

    The configuration is copied with its output path redirected, so one file
    can be run repeatedly into separate directories without editing it. The
    copy is what the run reads, which keeps the resolved output directory out
    of the configuration under version control.

    Parameters
    ----------
    config_path : str or path-like
        Configuration to run. Its output path must be ``"auto"``, which is
        what marks a configuration as not having fixed a destination.
    output_dir : str or path-like
        Directory the run writes into. Its parent receives the redirected
        copy of the configuration, and a scratch directory that is removed
        before returning.
    offline : bool, optional
        Run without checking for reference data.

    Notes
    -----
    ``TMPDIR`` is redirected for the duration of the run and restored
    afterwards, which is what makes the framework's scratch space removable
    however the run ended. Anything else in the same process that allocates a
    temporary file while this is running gets it under the same directory.

    Returns
    -------
    pandas.DataFrame
        The run's ``hf_all``.

    Raises
    ------
    ValueError
        If the configuration does not carry exactly one ``"auto"`` output path,
        since the run would then write somewhere the caller did not ask for.
    """
    output_dir = Path(output_dir)
    text = Path(config_path).read_text()
    # A function replacement, so a directory name containing a backslash escape
    # is inserted literally instead of being read as a group reference. No
    # count, so the substitution count below is the number of matches rather
    # than being capped at the limit.
    patched, substitutions = _AUTO_PATH.subn(
        lambda match: f'{match.group("lead")}"{output_dir}"', text
    )
    if substitutions != 1:
        raise ValueError(
            f'{config_path} sets its output path to "auto" {substitutions} times, and '
            f'exactly one is needed to redirect it into {output_dir}'
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    config_copy = output_dir.parent / f'{output_dir.name}_config.toml'
    config_copy.write_text(patched)

    # The framework allocates a scratch directory per run and never removes
    # it, so a caller that runs a configuration a dozen times would leave a
    # dozen behind. It allocates that directory under TMPDIR, and it does so
    # while the runner is being constructed, before the configuration has been
    # checked: a configuration with one unrecognised key raises out of the
    # constructor with the directory already on disk and no reference to it
    # anywhere. Pointing TMPDIR at a directory of our own is what makes the
    # scratch space removable whatever went wrong, rather than only when the
    # run got far enough to hand back a runner.
    scratch_root = output_dir.parent / f'{output_dir.name}_scratch'
    scratch_root.mkdir(parents=True, exist_ok=True)
    previous_tmpdir = os.environ.get('TMPDIR')
    os.environ['TMPDIR'] = str(scratch_root)
    try:
        runner = Proteus(config_path=config_copy)
        runner.start(resume=False, offline=offline)
        return runner.hf_all.copy()
    finally:
        if previous_tmpdir is None:
            os.environ.pop('TMPDIR', None)
        else:
            os.environ['TMPDIR'] = previous_tmpdir
        # The run's own output is a sibling of this directory, never inside
        # it, including under DEBUG logging where the framework uses the
        # output directory as its scratch space and allocates nothing here.
        shutil.rmtree(scratch_root, ignore_errors=True)


def write_reference(
    frame: pd.DataFrame,
    path: str | PathLike,
    *,
    config_path: str | PathLike | None = None,
    skip: tuple[str, ...] = NON_STATE_COLUMNS,
) -> None:
    """Write a trajectory to disk as a reference.

    One line per helpfile column, so a quantity that moves shows up as a
    single changed line rather than as a change to every row. A column that
    holds one value for the whole run is stored as that value alone, which is
    most of the helpfile and most of the file size.

    Parameters
    ----------
    frame : pandas.DataFrame
        Trajectory to record, normally ``Proteus.hf_all``.
    path : str or path-like
        Destination file.
    config_path : str or path-like, optional
        Configuration the trajectory was produced from. Its digest is stored
        in the header so a later comparison can tell that the configuration
        has moved on from the reference.
    skip : tuple of str, optional
        Columns to leave out, defaulting to those that do not describe
        simulated state.

    Raises
    ------
    ValueError
        If the frame holds no rows, or carries a column that is not numeric.
        Non-numeric state would need an encoding this format does not define,
        and silently dropping it would leave a reference that looks complete.
    """
    if len(frame) == 0:
        raise ValueError('cannot record a reference from a trajectory with no rows')

    columns = _state_columns(frame, skip)
    non_numeric = [
        column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(
            'reference trajectories hold numeric columns only, but the run produced '
            f'{", ".join(sorted(non_numeric))} as non-numeric; the format needs '
            'extending before this trajectory can be recorded'
        )

    digest = config_digest(config_path) if config_path is not None else ''
    lines = [
        f'{_HEADER_PREFIX} PROTEUS reference trajectory',
        f'{_HEADER_PREFIX} rows = {len(frame)}',
        f'{_HEADER_PREFIX} columns = {len(columns)}',
        f'{_HEADER_PREFIX} config_digest = {digest}',
    ]
    for column in columns:
        values = np.asarray(frame[column], dtype=float)
        first = values[0]
        # equal_nan so an all-NaN column collapses like any other constant
        # rather than being written out row by row.
        if np.array_equal(values, np.full(values.shape, first), equal_nan=True):
            lines.append(f'{column}\t{_KIND_CONST}\t{repr(float(first))}')
        else:
            body = '\t'.join(repr(float(value)) for value in values)
            lines.append(f'{column}\t{_KIND_SERIES}\t{body}')

    Path(path).write_text('\n'.join(lines) + '\n')


def read_reference(path: str | PathLike) -> Reference:
    """Read a reference trajectory written by :func:`write_reference`.

    Parameters
    ----------
    path : str or path-like
        File to read.

    Returns
    -------
    Reference
        The recorded trajectory, with constant columns expanded back to full
        length so callers see one uniform frame.

    Raises
    ------
    ReferenceFormatError
        If the file carries no row count, a line the format does not define,
        or a column whose length disagrees with the recorded row count.
    """
    text = Path(path).read_text()

    rows: int | None = None
    declared_columns: int | None = None
    digest = ''
    data: dict[str, np.ndarray] = {}

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(_HEADER_PREFIX):
            body = stripped.lstrip(_HEADER_PREFIX).strip()
            key, _, value = body.partition('=')
            if key.strip() == 'rows':
                rows = int(value.strip())
            elif key.strip() == 'columns':
                declared_columns = int(value.strip())
            elif key.strip() == 'config_digest':
                digest = value.strip()
            continue

        fields = line.rstrip('\n').split('\t')
        if len(fields) < 3:
            raise ReferenceFormatError(
                f'{path}:{number}: expected a column name, a kind and at least one '
                f'value, found {len(fields)} field(s)'
            )
        column, kind, values = fields[0], fields[1], fields[2:]
        if rows is None:
            raise ReferenceFormatError(
                f'{path}:{number}: column data appears before the row count header, '
                'so the length of a constant column is undefined'
            )
        if column in data:
            raise ReferenceFormatError(
                f'{path}:{number}: column {column} appears twice, so which of the two '
                'trajectories it records is undefined'
            )
        if kind == _KIND_CONST:
            if len(values) != 1:
                raise ReferenceFormatError(
                    f'{path}:{number}: a constant column carries one value, found {len(values)}'
                )
            data[column] = np.full(rows, float(values[0]))
        elif kind == _KIND_SERIES:
            if len(values) != rows:
                raise ReferenceFormatError(
                    f'{path}:{number}: column {column} carries {len(values)} values '
                    f'against the {rows} rows declared in the header'
                )
            data[column] = np.asarray([float(value) for value in values], dtype=float)
        else:
            raise ReferenceFormatError(
                f'{path}:{number}: unknown column kind {kind!r}, expected '
                f'{_KIND_CONST!r} or {_KIND_SERIES!r}'
            )

    if rows is None:
        raise ReferenceFormatError(f'{path}: no row count in the header')
    if declared_columns is not None and declared_columns != len(data):
        raise ReferenceFormatError(
            f'{path}: {len(data)} columns follow the header, which declares '
            f'{declared_columns}; the file has been edited since it was written, and '
            'the comparison would report the difference against the run rather than '
            'against the file'
        )

    return Reference(frame=pd.DataFrame(data), config_digest=digest)


def compare_trajectories(
    reference: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    rtol: float = DEFAULT_RTOL,
    atol_scale: float = DEFAULT_ATOL_SCALE,
    skip: tuple[str, ...] = NON_STATE_COLUMNS,
) -> TrajectoryComparison:
    """Compare a run against a reference trajectory, column by column.

    Parameters
    ----------
    reference : pandas.DataFrame
        Recorded trajectory.
    actual : pandas.DataFrame
        Trajectory produced by the run under test.
    rtol : float, optional
        Relative tolerance.
    atol_scale : float, optional
        Absolute tolerance for each column, as a fraction of the largest
        magnitude that column reaches in the reference.
    skip : tuple of str, optional
        Columns to leave out of the comparison.

    Returns
    -------
    TrajectoryComparison
        Every disagreement, worst first. A row-count mismatch is reported on
        its own, since trajectories of different lengths cannot be compared
        row by row.

    Notes
    -----
    Two NaNs at the same position count as agreement, so a quantity the run
    leaves undefined throughout is not reported as a difference. A NaN against
    a number is a difference: a column that a change corrupted into NaN is
    exactly what a comparison like this exists to catch, and screening
    non-finite values out first would hide it.
    """
    reference_columns = _state_columns(reference, skip)
    actual_columns = _state_columns(actual, skip)

    if len(reference) != len(actual):
        return TrajectoryComparison(
            differences=(),
            compared=(),
            rows_expected=len(reference),
            rows_actual=len(actual),
        )

    differences: list[ColumnDifference] = []
    for column in sorted(set(reference_columns) - set(actual_columns)):
        differences.append(ColumnDifference(column=column, reason='missing'))
    for column in sorted(set(actual_columns) - set(reference_columns)):
        differences.append(ColumnDifference(column=column, reason='unexpected'))

    compared: list[str] = []
    for column in reference_columns:
        if column not in actual_columns:
            continue
        if not pd.api.types.is_numeric_dtype(actual[column]):
            # Nothing the run writes today is non-numeric, and a reference
            # cannot record such a column at all, so this is a schema change
            # rather than a difference in value. Reported as one rather than
            # left to fail as a conversion error deep in the comparison.
            differences.append(ColumnDifference(column=column, reason='non_numeric'))
            continue

        expected = np.asarray(reference[column], dtype=float)
        found = np.asarray(actual[column], dtype=float)

        # Both terms are built from the reference rather than from the run, so
        # the band a run is judged against is a property of the recording and
        # does not move with whatever the run produced.
        largest = np.max(np.abs(expected[np.isfinite(expected)]), initial=0.0)
        atol = atol_scale * largest
        tolerance = atol + rtol * np.abs(expected)

        close = np.isclose(found, expected, rtol=rtol, atol=atol, equal_nan=True)
        if close.all():
            compared.append(column)
            continue

        # Rank the offending rows by how far past their own tolerance they
        # sit, so the reported row is the worst one rather than the first.
        # A zero tolerance means any difference at all, which ranks above
        # every finite excess.
        gap = np.abs(found - expected)
        with np.errstate(divide='ignore', invalid='ignore'):
            excess = np.where(tolerance > 0.0, gap / tolerance, np.inf)
        excess = np.where(close, -np.inf, excess)
        # A NaN on exactly one side leaves the subtraction undefined, which
        # would otherwise sort below every real excess and hide the row.
        excess = np.where(np.isnan(excess), np.inf, excess)
        row = int(np.argmax(excess))

        differences.append(
            ColumnDifference(
                column=column,
                reason='value',
                row=row,
                expected=float(expected[row]),
                actual=float(found[row]),
                deviation=float(excess[row]),
            )
        )

    differences.sort(key=lambda difference: -_sort_key(difference))
    return TrajectoryComparison(
        differences=tuple(differences),
        compared=tuple(compared),
        rows_expected=len(reference),
        rows_actual=len(actual),
    )


def _sort_key(difference: ColumnDifference) -> float:
    """Rank differences so the largest deviation is reported first.

    A column that appears or disappears sorts above every value difference: it
    explains any number of them, so it is the first thing a reader needs.
    """
    if difference.reason != 'value':
        return float('inf')
    return float(difference.deviation)


@dataclass(frozen=True)
class ReferenceCheck:
    """Outcome of holding a run against the reference recorded beside it.

    Attributes
    ----------
    reproduces : bool
        Whether the run reproduces the reference.
    report : str
        What to tell the reader, whether it reproduced or not.
    comparison : TrajectoryComparison or None
        The column-by-column comparison, where one was made. ``None`` when the
        reference is missing or belongs to another configuration, since
        neither leaves anything worth comparing against.
    """

    reproduces: bool
    report: str
    comparison: TrajectoryComparison | None = None


def check_against_reference(
    frame: pd.DataFrame,
    reference_path: str | PathLike,
    config_path: str | PathLike,
) -> ReferenceCheck:
    """Hold a trajectory against a recorded reference and say what happened.

    Parameters
    ----------
    frame : pandas.DataFrame
        Trajectory the run produced.
    reference_path : str or path-like
        Reference file to compare against.
    config_path : str or path-like
        Configuration the run was made from.

    Returns
    -------
    ReferenceCheck
        Whether the run reproduced the reference, and the text describing it.

    Notes
    -----
    A reference recorded from a different configuration is reported as that,
    on its own, ahead of the columns it makes differ. Such a reference explains
    any number of differing columns, and a reader who meets the column list
    first will look for the cause in the code rather than in the configuration.
    """
    reference_path = Path(reference_path)
    if not reference_path.is_file():
        return ReferenceCheck(
            reproduces=False,
            report=(
                f'no reference trajectory at {reference_path}; record one by running '
                'this test with --record-golden'
            ),
        )

    stored = read_reference(reference_path)
    if stored.config_digest != config_digest(config_path):
        return ReferenceCheck(
            reproduces=False,
            report=(
                f'{reference_path.name} was recorded from a different '
                f'{Path(config_path).name} than the one committed beside it; record it '
                'again with --record-golden, in the commit that changed the '
                'configuration'
            ),
        )

    comparison = compare_trajectories(stored.frame, frame)
    return ReferenceCheck(
        reproduces=comparison.agrees,
        report=comparison.report(),
        comparison=comparison,
    )


def record_reference(
    frame: pd.DataFrame,
    reference_path: str | PathLike,
    config_path: str | PathLike,
) -> str:
    """Record a trajectory as the reference, and say what that changed.

    Parameters
    ----------
    frame : pandas.DataFrame
        Trajectory to record.
    reference_path : str or path-like
        Reference file to write.
    config_path : str or path-like
        Configuration the trajectory was produced from, whose digest is stored
        alongside it.

    Returns
    -------
    str
        What was written, and how it differs from the reference it replaced.
        Recording is how a deliberate change to the trajectory is accepted, so
        it states what it accepted rather than rewriting the file silently.
    """
    reference_path = Path(reference_path)
    previous = read_reference(reference_path) if reference_path.is_file() else None

    write_reference(frame, reference_path, config_path=config_path)
    lines = [f'wrote {reference_path} ({len(frame)} rows)']
    if previous is not None:
        lines.append('against the reference it replaces:')
        lines.append(compare_trajectories(previous.frame, frame).report())
    return '\n'.join(lines)
