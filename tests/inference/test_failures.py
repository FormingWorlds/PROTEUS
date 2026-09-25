"""
Unit tests for recording and reporting unscored inference evaluations.

Covers `proteus.inference.failures`: what a failed or excluded evaluation
carries, how one is appended to the study's failure table without workers
contending for it, how that table is read back, and the end-of-study tally.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import logging
import pickle
import re

import pandas as pd
import pytest

import proteus.inference.failures as failures_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _counts(line: str) -> list[str]:
    """Every number in a log line, in order.

    The tally lines are prose around a handful of counts. Asserting on the
    numbers keeps a test pinned to what the reader has to get right, and lets
    the wording be changed without a test failing for no reason. Order is kept,
    so a line that swapped the failed and excluded counts still fails.
    """
    return re.findall(r'\d+(?:\.\d+)?', line)


@pytest.mark.unit
def test_failure_summary_is_one_line_and_names_where_the_detail_is_kept():
    """The line a study logs for each unscored run identifies the run, names
    the status code and points at the output folder, and stays on one line
    however many parameters the study sweeps. The swept values and the paths to
    open next belong to the fuller report, not to the one-liner.
    """
    # Edge case: a wide sweep is the situation the one-liner exists for. Twenty
    # parameters rendered inline would run past any terminal width.
    swept = {f'planet.param_{i}': float(i) for i in range(20)}
    swept['planet.mass_tot'] = 1.25
    failure = failures_mod.ProteusRunFailure(
        reason='the simulator exited with an error',
        worker=2,
        iter=16,
        out_dir='/study/workers/w_2/i_16',
        exit_code=1,
        status=22,
        log_path='/study/workers/w_2/i_16/proteus_00.log',
        console_path='/study/workers/w_2/i_16_console.log',
        parameters=swept,
    )
    line = failure.summary()

    assert '\n' not in line
    assert 'planet.mass_tot' not in line
    assert 'proteus_00.log' not in line
    # What has to survive the trim: who failed, what the status was, and the
    # folder holding the logfile and the console capture.
    assert 'worker=2 iter=16' in line
    assert 'status 22' in line
    assert 'Atmosphere' in line
    assert 'exit code 1' in line
    assert '/study/workers/w_2/i_16' in line
    # Discrimination: the detail is not lost, only moved. A regression that
    # trimmed `report` instead of adding a second renderer would fail here.
    rendered = failure.report()
    assert 'planet.mass_tot=1.25' in rendered
    assert 'proteus_00.log' in rendered
    assert 'i_16_console.log' in rendered
    # The report opens with the same one-liner, so nothing the summary names is
    # dropped on the way to the fuller form.
    assert rendered.splitlines()[0] == line

    # Limit input: an excluded run has nothing to report as a fault, so it is
    # named as excluded and its exit code, always zero on that path, is left
    # out rather than read as a crash code.
    excluded = failures_mod.ProteusRunFailure(
        reason='completed on a status this study excludes',
        worker=0,
        iter=10,
        out_dir='/study/workers/w_0/i_10',
        exit_code=0,
        status=11,
        category=failures_mod.CATEGORY_EXCLUDED,
    )
    excluded_line = excluded.summary()
    assert '\n' not in excluded_line
    assert 'excluded for worker=0 iter=10' in excluded_line
    assert 'failed for worker=0' not in excluded_line
    assert 'exit code' not in excluded_line
    assert 'status 11' in excluded_line


@pytest.mark.unit
def test_proteus_run_failure_survives_the_trip_back_from_a_pool_worker():
    """A failure raised inside a pool worker is pickled and re-raised in the
    parent process. Every reported field must survive that round trip, or the
    parent sees a reconstruction error in place of the diagnosis.
    """
    original = failures_mod.ProteusRunFailure(
        reason='the simulator exited with an error',
        worker=2,
        iter=7,
        out_dir='/study/workers/w_2/i_7',
        exit_code=1,
        status=27,
        log_path='/study/workers/w_2/i_7/proteus_00.log',
        console_path='/study/workers/w_2/i_7_console.log',
        parameters={'planet.mass_tot': 2.0},
    )
    restored = pickle.loads(pickle.dumps(original))

    assert isinstance(restored, failures_mod.ProteusRunFailure)
    assert restored.report() == original.report()
    # Field-level guard: an equal report could still hide a dropped field that
    # the renderer omits when empty, so pin the values that steer diagnosis.
    assert restored.status == 27
    assert restored.worker == 2 and restored.iter == 7
    assert restored.parameters == {'planet.mass_tot': pytest.approx(2.0)}
    assert restored.log_path == original.log_path
    assert restored.console_path == original.console_path
    assert restored.category == failures_mod.CATEGORY_FAILURE

    # The category rides along in the same tuple, and it decides whether the
    # parent calls the run a fault. A field dropped from the reconstruction
    # would fall back to the 'failure' default and go unnoticed on a failure,
    # so the round trip is checked on the other value too.
    excluded = failures_mod.ProteusRunFailure(
        reason='completed on a status this study excludes',
        worker=2,
        iter=7,
        out_dir='/study/workers/w_2/i_7',
        exit_code=0,
        status=11,
        category=failures_mod.CATEGORY_EXCLUDED,
    )
    restored_excluded = pickle.loads(pickle.dumps(excluded))
    assert restored_excluded.category == failures_mod.CATEGORY_EXCLUDED
    assert 'excluded for worker=2' in restored_excluded.report()
    assert 'failed for worker=2' not in restored_excluded.report()


@pytest.mark.unit
def test_failure_records_round_trip_into_one_table(tmp_path):
    """Each worker appends its own row to the study's failure table, and the
    rows are read back ordered by worker then iteration whatever order they
    arrived in. The status description is stored rather than recomputed, so the
    summary does not have to re-derive it from the code.
    """
    first = failures_mod.ProteusRunFailure(
        reason='the simulator exited with an error',
        worker=0,
        iter=5,
        out_dir='/study/workers/w_0/i_5',
        exit_code=1,
        status=21,
        parameters={'planet.mass_tot': 3.0},
    )
    # Same iteration, different worker: two rows, not one overwriting the other.
    second = failures_mod.ProteusRunFailure(
        reason='exceeded the 3600.0 s timeout',
        worker=1,
        iter=5,
        out_dir='/study/workers/w_1/i_5',
        status=failures_mod.STATUS_MISSING,
        parameters={'planet.mass_tot': 4.0},
    )

    # Written out of order: the second worker fails first.
    assert failures_mod.record_failure(tmp_path, second) is not None
    assert failures_mod.record_failure(tmp_path, first) is not None
    table = tmp_path / failures_mod.FAILURE_CSV
    # One header however many workers append, so the table parses as one frame.
    assert table.read_text().count('worker,iter,') == 1

    records = failures_mod.read_failure_records(tmp_path)
    # Ordering guard: written second-then-first, read back in worker order.
    assert [r['worker'] for r in records] == [0, 1]
    assert records[0]['status'] == 21
    assert records[0]['status_desc'] == first.status_desc
    # Swept values are columns of their own, so the table can be sorted on a
    # parameter to see which region of the box fails.
    assert records[0]['planet.mass_tot'] == pytest.approx(3.0)
    assert records[1]['planet.mass_tot'] == pytest.approx(4.0)
    # A run that never wrote a status file is stored as such, not as a generic
    # error, so the summary can separate start-up deaths from model faults.
    assert records[1]['status'] == failures_mod.STATUS_MISSING
    assert 'no readable status file' in records[1]['status_desc']
    # An absent exit code reads back as None, not as the string 'nan', which
    # would print into the summary as though it were a code the child returned.
    assert records[1]['exit_code'] is None

    # Edge case: a field holding the delimiter is quoted on the way out, or it
    # would shift every later column of that row by one.
    comma = failures_mod.ProteusRunFailure(
        reason='the simulator exited with an error, code 3',
        worker=2,
        iter=0,
        out_dir='/study/workers/w_2/i_0',
        exit_code=3,
        status=21,
        parameters={'planet.mass_tot': 5.0},
    )
    assert failures_mod.record_failure(tmp_path, comma) is not None
    reread = failures_mod.read_failure_records(tmp_path)[-1]
    assert reread['reason'] == 'the simulator exited with an error, code 3'
    assert reread['planet.mass_tot'] == pytest.approx(5.0)

    # Edge case: a table that cannot be parsed is reported and treated as empty
    # rather than aborting the summary it feeds. A zero-length file is the
    # reachable form of this: a worker killed between creating the table and
    # writing its first row leaves exactly that behind.
    table.write_text('')
    assert failures_mod.read_failure_records(tmp_path) == []


@pytest.mark.unit
def test_recording_a_failure_never_masks_the_failure_it_records(tmp_path):
    """Bookkeeping must not bring down a study. When the record cannot be
    written the writer reports that it could not, and the caller still has the
    failure in hand to log and to score.
    """
    blocked = tmp_path / 'not_a_directory'
    blocked.write_text('this is a file, so no folder can be made beneath it')
    failure = failures_mod.ProteusRunFailure(
        reason='the simulator exited with an error',
        worker=0,
        iter=0,
        out_dir=str(tmp_path),
        exit_code=1,
        status=21,
    )

    assert failures_mod.record_failure(blocked, failure) is None
    # Discrimination: the same failure records fine against a usable folder, so
    # the None above came from the blocked path and not from a writer that
    # always fails.
    assert failures_mod.record_failure(tmp_path / 'study', failure) is not None
    # Reading a study that never created the folder is empty, not an error.
    assert failures_mod.read_failure_records(tmp_path / 'never_ran') == []


@pytest.mark.unit
def test_summarise_failures_tabulates_causes_and_warns_on_every_real_failure(tmp_path, caplog):
    """The end-of-study tally turns the per-run rows into one table and one
    breakdown by cause, and warns whenever a run produced nothing usable. The
    warning does not wait for a fraction of the study to fail: a sweep can lose
    a tenth of its evaluations and still fit well, so the count is put in front
    of the reader to weigh rather than compared against a threshold.
    """

    # Two runs that died the same way and one that died differently, so the
    # breakdown has something to group.
    for worker, status in ((0, 21), (1, 21), (2, 24)):
        failures_mod.record_failure(
            tmp_path,
            failures_mod.ProteusRunFailure(
                reason='the simulator exited with an error',
                worker=worker,
                iter=0,
                out_dir=f'/study/workers/w_{worker}/i_0',
                exit_code=1,
                status=status,
                parameters={'planet.mass_tot': 1.0 + worker},
            ),
        )

    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.failures'):
        n_failed = failures_mod.summarise_failures(str(tmp_path), n_attempted=20)

    assert n_failed == 3
    messages = '\n'.join(r.message for r in caplog.records)
    assert '3 of 20' in messages
    # Grouped by cause, so two runs that died the same way count as one line.
    assert 'Interior model' in messages

    # 3 of 20 is 15%, well under the half-the-study line the old threshold drew,
    # and it is raised to a warning anyway: the count is what the reader weighs.
    # One record carries it, so the level changes rather than a second line
    # repeating the counts the report already gave.
    warnings = [r for r in caplog.records if r.levelname == 'WARNING']
    assert len(warnings) == 1
    # unscored, attempted, percent, failed, excluded.
    assert _counts(warnings[0].message) == ['3', '20', '15.0', '3', '0']

    # The table carries the swept parameter alongside the diagnosis, so the
    # failing region can be located without opening each run folder.
    table = pd.read_csv(tmp_path / 'failures.csv')
    assert len(table) == 3
    assert list(table['worker']) == [0, 1, 2]
    assert sorted(table['status']) == [21, 21, 24]
    assert table['planet.mass_tot'].max() == pytest.approx(3.0)

    # The percentage tracks the study size rather than being a fixed string:
    # the same three failures against a smaller study report a larger share.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.failures'):
        failures_mod.summarise_failures(str(tmp_path), n_attempted=4)
    warnings = [r for r in caplog.records if r.levelname == 'WARNING']
    assert len(warnings) == 1
    assert _counts(warnings[0].message) == ['3', '4', '75.0', '3', '0']


@pytest.mark.unit
def test_summarise_failures_counts_excluded_outcomes_apart_from_failures(tmp_path, caplog):
    """A run that completed on a status the study excludes is tallied, but not
    as a fault. Folding the two together would tell the user that a study whose
    runs all reached their clock limit, exactly as configured, is a study full
    of broken simulations.
    """

    failures_mod.record_failure(
        tmp_path,
        failures_mod.ProteusRunFailure(
            reason='the simulator exited with an error',
            worker=0,
            iter=0,
            out_dir='/study/workers/w_0/i_0',
            exit_code=1,
            status=21,
            parameters={'planet.mass_tot': 1.0},
        ),
    )
    for worker in (1, 2):
        failures_mod.record_failure(
            tmp_path,
            failures_mod.ProteusRunFailure(
                reason='completed on a status this study excludes',
                worker=worker,
                iter=0,
                out_dir=f'/study/workers/w_{worker}/i_0',
                exit_code=0,
                status=11,
                parameters={'planet.mass_tot': 1.0 + worker},
                category=failures_mod.CATEGORY_EXCLUDED,
            ),
        )

    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.failures'):
        n_unscored = failures_mod.summarise_failures(str(tmp_path), n_attempted=20)

    # Both kinds are unscored, so both count toward how much of the study was
    # real, but the breakdown names them apart.
    assert n_unscored == 3
    messages = '\n'.join(r.message for r in caplog.records)
    tally = next(r for r in caplog.records if 'Unscored evaluations' in r.message)
    # unscored, attempted, percent, failed, excluded: the one fault is counted
    # apart from the two runs that completed on an excluded status.
    assert _counts(tally.message) == ['3', '20', '15.0', '1', '2']
    # The clock-limit outcome is labelled in the cause table rather than being
    # listed beside the interior-model error as if it were one.
    assert 'Completed (maximum clock runtime) [excluded]' in messages
    assert 'Error (Interior model) [excluded]' not in messages

    # Carried into the table too, so the excluded rows can be filtered out when
    # looking for the region that actually breaks the simulator.
    table = pd.read_csv(tmp_path / 'failures.csv')
    assert sorted(table['category']) == ['excluded', 'excluded', 'failure']
    assert sorted(table.loc[table['category'] == 'excluded', 'status']) == [11, 11]

    # Raised to a warning by the one genuine fault. The counts were pinned
    # above; what matters here is that the line carrying them is the warning.
    assert [r.levelname for r in caplog.records if r.levelname == 'WARNING'] == ['WARNING']
    assert 'Unscored evaluations' in tally.message and tally.levelname == 'WARNING'

    # Limit input: a study whose runs were *all* excluded did exactly what it
    # was configured to do, so it is tallied without any warning at all. Keying
    # the warning on the unscored total would have flagged it as broken.
    caplog.clear()
    (tmp_path / 'failures.csv').unlink()
    for worker in (0, 1):
        failures_mod.record_failure(
            tmp_path,
            failures_mod.ProteusRunFailure(
                reason='completed on a status this study excludes',
                worker=worker,
                iter=0,
                out_dir=f'/study/workers/w_{worker}/i_0',
                exit_code=0,
                status=11,
                parameters={'planet.mass_tot': 1.0 + worker},
                category=failures_mod.CATEGORY_EXCLUDED,
            ),
        )
    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.failures'):
        assert failures_mod.summarise_failures(str(tmp_path), n_attempted=4) == 2
    assert not [r for r in caplog.records if r.levelname == 'WARNING']
    # Discrimination: 2 of 4 is half the study, which the old fraction rule
    # would have reported as a study mostly not worth trusting.
    assert '2 of 4' in '\n'.join(r.message for r in caplog.records)


@pytest.mark.unit
def test_summarise_failures_reports_a_clean_study_without_writing_a_table(tmp_path, caplog):
    """A study in which nothing failed says so and writes no table. An empty
    failures.csv would suggest the accounting had run and found nothing to
    say about a study that in fact had nothing to report.
    """

    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.failures'):
        n_failed = failures_mod.summarise_failures(str(tmp_path), n_attempted=12)

    assert n_failed == 0
    assert not (tmp_path / 'failures.csv').exists()
    messages = '\n'.join(r.message for r in caplog.records)
    assert 'none' in messages and '12' in _counts(messages)
    assert not [r for r in caplog.records if r.levelname in ('WARNING', 'ERROR')]


@pytest.mark.unit
def test_summarise_failures_labels_the_logfile_sample_and_counts_the_whole_study(
    tmp_path, caplog
):
    """The tally covers every evaluation attempted, initial samples included,
    while the warning raised alongside the best fit covers the optimisation
    steps alone. The logfile lines are a sample of at most three, so they are
    labelled with how many of the total they show and printed above the pointer
    to the full table; unlabelled, three paths below a "Full list" line read as
    the complete set.
    """

    # Four records, the first of which has no logfile: the run died before the
    # child wrote one. The sample must skip it and still offer three paths.
    for worker in range(4):
        failures_mod.record_failure(
            tmp_path,
            failures_mod.ProteusRunFailure(
                reason='the simulator exited with an error',
                worker=worker,
                iter=0,
                out_dir=f'/study/workers/w_{worker}/i_0',
                exit_code=1,
                status=21,
                log_path=None if worker == 0 else f'/study/w_{worker}/proteus_00.log',
                parameters={'planet.mass_tot': 1.0 + worker},
            ),
        )

    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.failures'):
        failures_mod.summarise_failures(str(tmp_path), n_attempted=20)

    lines = [r.message for r in caplog.records]
    messages = '\n'.join(lines)
    # The denominator of the tally is the whole study, stated in the line
    # itself so it cannot be confused with the optimisation-only warning.
    assert '4 of 20' in messages
    # Three shown out of four unscored, not four out of four.
    assert '3 of 4 shown' in messages
    shown = [line.strip() for line in lines if line.strip().endswith('proteus_00.log')]
    assert len(shown) == 3
    # The record without a logfile is skipped rather than truncating the
    # sample to the two paths that follow it in the first three records.
    assert '/study/w_1/proteus_00.log' in shown
    assert '/study/w_3/proteus_00.log' in shown

    # The pointer to the complete table comes after the sample, so the sample
    # cannot be read as a continuation of it.
    i_sample = next(i for i, line in enumerate(lines) if line.startswith('Logfiles ('))
    i_full = next(i for i, line in enumerate(lines) if line.startswith('Full list:'))
    assert i_sample < i_full

    # Discrimination: with no logfile recorded anywhere, no sample block is
    # emitted at all, so the label tracks the data rather than always printing.
    caplog.clear()
    (tmp_path / 'failures.csv').rename(tmp_path / 'failures_old.csv')
    failures_mod.record_failure(
        tmp_path,
        failures_mod.ProteusRunFailure(
            reason='the simulator exited with an error',
            worker=0,
            iter=1,
            out_dir='/study/workers/w_0/i_1',
            exit_code=1,
            status=21,
            parameters={'planet.mass_tot': 1.0},
        ),
    )
    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.failures'):
        failures_mod.summarise_failures(str(tmp_path), n_attempted=20)
    assert not [r for r in caplog.records if r.message.startswith('Logfiles (')]
