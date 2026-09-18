"""Recording and reporting failing inference evaluations, and reporting outcomes
excluded through `failure_codes`.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from proteus.utils.helper import STATUS_MISSING, CommentFromStatus

log = logging.getLogger('fwl.' + __name__)

# Whether a failed child run aborts the study or scores a bad objective value.
_ABORT_ON_FAILURE_ENV = 'PROTEUS_INFERENCE_ABORT_ON_FAILURE'

# Suffix for the file holding whatever a child wrote to its console.
CHILD_CONSOLE_SUFFIX = '_console.log'

# How an evaluation that failed is classified. A run that
# crashed, or stopped in an error state, did not produce a result at all. A run
# that completed normally but ended on a status listed in the study's
# `failure_codes` did produce a result, but the study does not fit against
# that outcome. Only the first is a fault.
CATEGORY_FAILURE = 'failure'
CATEGORY_EXCLUDED = 'excluded'

# Table inside the study output holding one row per unscored evaluation.
# Appended by the workers as they fail and read back once at the end, so that
# the summary covers initial sampling and optimisation alike without the two
# paths having to share any state while they run.
FAILURE_CSV = 'failures.csv'

# Fixed columns of that table, in order. The swept parameter values follow, one
# column each. The two paths are what the user opens after:
# the logfile for a run that got far enough to configure its logger,
# the console capture for one that did not.
_FAILURE_COLUMNS = (
    'worker',
    'iter',
    'category',
    'status',
    'status_desc',
    'exit_code',
    'reason',
    'out_dir',
    'log_path',
    'console_path',
)

# Fraction of evaluations that may fail before the summary escalates from a
# report to a warning. Above this, the sampled region is mostly unrunnable and
# the posterior is built on too few real evaluations to mean much.
FAILURE_FRACTION_WARN = 0.5


@dataclass(eq=False)
class ProteusRunFailure(RuntimeError):
    """A single child PROTEUS run that did not produce a usable result.

    Carries everything needed to diagnose the run without opening the study
    by hand: which evaluation it was, where its output landed, how it died,
    what PROTEUS recorded in its status file, and the parameter values that
    produced it. `category` separates a genuine fault from a run with an
    excluded status. Both score the failure value, but only the first is
    reported as something having gone wrong.

    Faults that would affect every evaluation (eg no `proteus` on PATH)
    stay as ordinary exceptions so they abort the study instead of being
    scored as a bad sample.
    """

    reason: str
    worker: int
    iter: int
    out_dir: str
    exit_code: int | None = None
    status: int = STATUS_MISSING
    log_path: str | None = None
    console_path: str | None = None
    parameters: dict = field(default_factory=dict)
    category: str = CATEGORY_FAILURE

    @property
    def status_desc(self) -> str:
        """Human-readable form of the PROTEUS status code."""
        if self.status == STATUS_MISSING:
            return 'no readable status file (died during start-up)'
        return CommentFromStatus(self.status)

    def summary(self) -> str:
        """Single-line description naming the outcome and where to look next."""
        verb = 'excluded' if self.category == CATEGORY_EXCLUDED else 'failed'
        parts = [
            f'PROTEUS run {verb} for worker={self.worker} iter={self.iter}: {self.reason}',
            f'status {self.status} ({self.status_desc})',
        ]
        # A zero exit code is the norm for every path except a crash, where it
        # is the one number that says which signal or error ended the run.
        if self.exit_code:
            parts.append(f'exit code {self.exit_code}')
        parts.append(f'output {self.out_dir}')
        return '; '.join(parts)

    def report(self) -> str:
        """The summary plus the detail that is too long to log on one line."""
        lines = [self.summary()]
        if self.log_path:
            lines.append(f'    logfile    = {self.log_path}')
        # Named whether or not a logfile exists: a run that died before its
        # logger was configured left nothing else behind to read.
        if self.console_path:
            lines.append(f'    console    = {self.console_path}')
        if self.parameters:
            pretty = ', '.join(f'{k}={v:g}' for k, v in sorted(self.parameters.items()))
            lines.append(f'    parameters = {pretty}')
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.report()

    def __reduce__(self):
        # A failure raised inside a pool worker is pickled to be re-raised in
        # the parent. BaseException.__reduce__ rebuilds from `self.args`,
        # which a dataclass __init__ leaves empty, so the default would fail
        # to reconstruct this class. Rebuild from the fields instead.
        return (
            self.__class__,
            (
                self.reason,
                self.worker,
                self.iter,
                self.out_dir,
                self.exit_code,
                self.status,
                self.log_path,
                self.console_path,
                self.parameters,
                self.category,
            ),
        )


def set_abort_on_failure(abort: bool = False) -> None:
    """Record whether a failed child run should abort the whole study.

    Stored in the environment so it is visible to the main process and to any
    spawned pool workers, matching how the child timeout is plumbed.
    """
    os.environ[_ABORT_ON_FAILURE_ENV] = '1' if abort else '0'


def abort_on_failure() -> bool:
    """Return whether a failed child run should abort the whole study.

    Defaults to False: an inference sweep is expected to visit parameter
    combinations the simulator cannot integrate, and treating those as fatal
    would make most studies unrunnable. Set the inference config field
    `abort_on_failure` to stop at the first failure instead.
    """
    return os.environ.get(_ABORT_ON_FAILURE_ENV, '0') == '1'


def find_run_logfile(out_abs: Path | str) -> str | None:
    """Return the newest PROTEUS logfile in a run's output folder, if any.

    PROTEUS captures uncaught exceptions into this file, so it usually holds
    the traceback for a crashed run. It does not exist for a run that failed
    before the logger was configured.
    """
    logs = sorted(Path(out_abs).glob('proteus_*.log'))
    return str(logs[-1]) if logs else None


def record_failure(study_abs: Path | str, failure: ProteusRunFailure) -> str | None:
    """Append one row to the study's failure table.

    Workers are separate processes with no shared state, so each appends its
    own row rather than handing the failure back to the parent. The first
    worker to fail creates the file with its header through an exclusive
    create, which exactly one caller can win, and every later row is a single
    append. A row is one `write` call of well under a pipe buffer, which the
    kernel adds whole, so no lock is needed on a local filesystem.

    Parameters
    ----------
    - study_abs (Path | str): Absolute path to the study output folder.
    - failure (ProteusRunFailure): The failure to record.

    Returns
    ----------
    - str | None: Path written, or None if the row could not be written.
      Recording is best-effort: a study must not be brought down by a fault in
      its own bookkeeping, so the failure being reported still reaches the log.
    """
    row = {key: getattr(failure, key) for key in _FAILURE_COLUMNS if key != 'status_desc'}
    row['status_desc'] = failure.status_desc
    row.update(failure.parameters)
    ordered = {key: row[key] for key in (*_FAILURE_COLUMNS, *failure.parameters)}

    target = Path(study_abs) / FAILURE_CSV
    line = _csv_row(ordered.values())
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(target, 'x') as f:
                f.write(_csv_row(ordered.keys()) + line)
        except FileExistsError:
            with open(target, 'a') as f:
                f.write(line)
    except OSError as err:
        log.warning(
            f'Could not record the failure of worker={failure.worker} '
            f'iter={failure.iter}: {err}'
        )
        return None
    return str(target)


def _csv_row(values) -> str:
    """Render one CSV line, quoting the fields that need it."""
    fields = []
    for value in values:
        text = '' if value is None else str(value)
        if any(c in text for c in ',"\n'):
            text = '"' + text.replace('"', '""') + '"'
        fields.append(text)
    return ','.join(fields) + '\n'


def read_failure_records(study_abs: Path | str) -> list[dict]:
    """Read back the failure table written during a study.

    Parameters
    ----------
    - study_abs (Path | str): Absolute path to the study output folder.

    Returns
    ----------
    - list[dict]: One entry per unscored evaluation, ordered by worker then
      iteration. An unreadable table is reported and treated as empty rather
      than aborting the summary it feeds.
    """
    target = Path(study_abs) / FAILURE_CSV
    if not target.is_file():
        return []
    try:
        table = pd.read_csv(target)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as err:
        log.warning(f'Skipping unreadable failure table {target}: {err}')
        return []
    table = table.sort_values(['worker', 'iter'], kind='stable')
    # An absent exit code or logfile reads back as NaN, which would print as
    # 'nan' in the summary and compare equal to nothing.
    return table.astype(object).where(table.notna(), None).to_dict('records')


def summarise_failures(output: str, n_attempted: int) -> int:
    """Report on the study's unscored evaluations.

    The breakdown by cause, and the per-run paths are reported together at the end
    of the inference run. Runs that failed and runs that were excluded are counted apart.

    Parameters
    ----------
    - output (str): Absolute path to the study output folder.
    - n_attempted (int): Total evaluations attempted, initial samples included.

    Returns
    ----------
    - int: Number of evaluations that carry the failure score.
    """
    records = read_failure_records(output)
    n_unscored = len(records)

    log.info('-----------------------------------')
    if not n_unscored:
        log.info(f'Unscored evaluations: none, all {n_attempted} evaluations were usable')
        log.info('-----------------------------------')
        return 0

    # A record with no category describes a genuine fault
    n_excluded = sum(1 for r in records if r.get('category') == CATEGORY_EXCLUDED)
    n_failed = n_unscored - n_excluded

    frac = n_unscored / max(n_attempted, 1)
    log.info(
        f'Unscored evaluations: {n_unscored} of {n_attempted} evaluations '
        f'({100 * frac:.1f}%, initial samples included) carry the failure score '
        'rather than a fit quality'
    )
    log.info(f'    {n_failed} did not produce a usable result')
    log.info(f'    {n_excluded} completed on a status this study excludes')

    # Grouped by cause, and labelled so that an excluded outcome is not read as
    # something having gone wrong in the run that reached it.
    log.info(f'{"Cause":52s} | Count')
    for (category, desc), count in Counter(
        (r.get('category') or CATEGORY_FAILURE, r.get('status_desc') or 'unknown')
        for r in records
    ).most_common():
        label = f'{desc} [excluded]' if category == CATEGORY_EXCLUDED else str(desc)
        log.info(f'{label:52s}   {count}')
    # A few concrete places to look. A run that died before configuring its logger
    # has no logfile, and its console capture is then the only record of why it refused to start.
    sample = [rec.get('log_path') or rec.get('console_path') for rec in records]
    sample = [path for path in sample if path][:3]
    if sample:
        log.info(f'Logfiles ({len(sample)} of {n_unscored} shown):')
        for log_path in sample:
            log.info(f'    {log_path}')
    log.info(f'Full list: {Path(output) / FAILURE_CSV}')

    if frac > FAILURE_FRACTION_WARN:
        log.warning(
            f'More than {100 * FAILURE_FRACTION_WARN:.0f}% of evaluations failed or were'
            f'excluded, so the result rests on {n_attempted - n_unscored} real '
            'evaluations.'
        )
    log.info('-----------------------------------')

    return n_unscored
