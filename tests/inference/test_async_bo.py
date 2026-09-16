"""
Unit tests for asynchronous Bayesian optimization orchestration helpers.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

# The Bayesian-optimisation stack ships as the optional `inference` extra,
# which installs all three together. Guarding the whole stack keeps a
# partial environment skipping rather than failing collection.
torch = pytest.importorskip('torch')
pytest.importorskip('botorch')
pytest.importorskip('gpytorch')

import proteus.inference.async_BO as async_mod  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


class _DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.mark.unit
def test_checkpoint_writes_expected_files(tmp_path):
    """``async_BO.checkpoint`` writes the three expected files
    (``data.csv``, ``logs.csv``, ``Ts.csv``) and the timing CSV has the
    canonical ``elapsed_s`` column header.
    """
    D = {
        'X': torch.tensor([[0.1]], dtype=torch.double),
        'Y': torch.tensor([[0.2]], dtype=torch.double),
    }
    logs = [{'worker': 0, 'task_id': 0, 'y_value': 0.2}]
    Ts = [0.5]

    async_mod.checkpoint(D, logs, Ts, str(tmp_path))

    assert (tmp_path / 'data.csv').is_file()
    assert (tmp_path / 'logs.csv').is_file()
    assert (tmp_path / 'Ts.csv').is_file()
    assert list(pd.read_csv(tmp_path / 'Ts.csv').columns) == ['elapsed_s']


@pytest.mark.unit
def test_worker_updates_shared_data_and_logs(monkeypatch, tmp_path):
    """A single worker iteration appends one row to the shared X/Y
    tensors, one entry to the timing list, one log record, and triggers
    exactly one checkpoint snapshot.
    """
    D_shared = {
        'X': torch.tensor([[0.1]], dtype=torch.double),
        'Y': torch.tensor([[0.2]], dtype=torch.double),
    }
    B = {}
    T = []
    logs = []
    snapshots = []

    def fake_build_obj(**kwargs):
        return lambda x: x

    def fake_process_fun(**kwargs):
        return (
            torch.tensor([[0.4]], dtype=torch.double),
            torch.tensor([[0.9]], dtype=torch.double),
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        )

    monkeypatch.setattr(
        async_mod,
        'checkpoint',
        lambda D, log_list, Ts, output_dir: snapshots.append((D, log_list, Ts, output_dir)),
    )

    async_mod.worker(
        process_fun=fake_process_fun,
        build_obj=fake_build_obj,
        D_shared=D_shared,
        B=B,
        T=T,
        T0=0.0,
        x_init=torch.tensor([[0.3]], dtype=torch.double),
        n_init=1,
        lock=_DummyLock(),
        max_len=2,
        worker_id=0,
        log_list=logs,
        output_dir=str(tmp_path),
    )

    assert D_shared['X'].shape == (2, 1)
    assert D_shared['Y'].shape == (2, 1)
    assert len(T) == 1
    assert len(logs) == 1
    assert logs[0]['worker'] == 0
    assert len(snapshots) == 1


@pytest.mark.unit
def test_parallel_process_rejects_unknown_kernel():
    """``parallel_process`` raises ValueError with a 'Unknown kernel'
    message when called with a kernel name outside the supported set.
    """
    with pytest.raises(ValueError, match='Unknown kernel'):
        async_mod.parallel_process(
            objective_builder=lambda **kwargs: None,
            kernel='UNKNOWN',
            acqf='LogEI',
            n_workers=1,
            max_len=3,
            output='dummy',
            seed=1,
            ref_config='ref.toml',
            observables={'obs': 1.0},
            parameters={'a': [0.0, 1.0]},
            failure_codes=[],
        )
    # Discrimination: the error message must surface the valid choices so
    # callers can correct the misconfiguration; this guards against a
    # regression that left only a bare "Unknown kernel" string with no
    # remediation hint.
    with pytest.raises(ValueError, match='RBF'):
        async_mod.parallel_process(
            objective_builder=lambda **kwargs: None,
            kernel='UNKNOWN',
            acqf='LogEI',
            n_workers=1,
            max_len=3,
            output='dummy',
            seed=1,
            ref_config='ref.toml',
            observables={'obs': 1.0},
            parameters={'a': [0.0, 1.0]},
            failure_codes=[],
        )
    # Discrimination: the error message must surface the valid choices so
    # callers can correct the misconfiguration; this guards against a
    # regression that left only a bare "Unknown kernel" string with no
    # remediation hint.
    with pytest.raises(ValueError, match='RBF'):
        async_mod.parallel_process(
            objective_builder=lambda **kwargs: None,
            kernel='UNKNOWN',
            acqf='LogEI',
            n_workers=1,
            max_len=3,
            output='dummy',
            seed=1,
            ref_config='ref.toml',
            observables={'obs': 1.0},
            parameters={'a': [0.0, 1.0]},
            failure_codes=[],
        )


@pytest.mark.unit
def test_parallel_process_raises_when_init_dataset_missing(monkeypatch, tmp_path):
    """``parallel_process`` raises FileNotFoundError when the initial
    dataset (``D_init``) is missing from the output directory.
    """
    monkeypatch.setattr(
        async_mod, 'get_proteus_directories', lambda _output: {'output': str(tmp_path)}
    )
    monkeypatch.setattr(async_mod, 'Manager', lambda: object())

    with pytest.raises(FileNotFoundError, match='Cannot find D_init'):
        async_mod.parallel_process(
            objective_builder=lambda **kwargs: None,
            kernel='RBF',
            acqf='LogEI',
            n_workers=1,
            max_len=3,
            output='dummy',
            seed=1,
            ref_config='ref.toml',
            observables={'obs': 1.0},
            parameters={'a': [0.0, 1.0]},
            failure_codes=[],
        )
    # Discrimination: the missing-init guard must fire only when the file
    # is actually absent. Confirm the dataset filename is not on disk so
    # the FileNotFoundError above can only have come from this guard.
    assert not (tmp_path / 'init.csv').exists()


@pytest.mark.unit
def test_parallel_process_happy_path_with_mocked_manager(monkeypatch, tmp_path):
    """With a mocked multiprocessing Manager and Process, ``parallel_process``
    spawns one Process per worker, returns the final dataset, the per-worker
    logs, and the elapsed-time list. Pins the orchestration contract
    without invoking real subprocesses.
    """

    class FakeManager:
        def dict(self, data=None):
            return {} if data is None else dict(data)

        def list(self, data=None):
            return [] if data is None else list(data)

        def Lock(self):
            return _DummyLock()

    created_processes = []

    class FakeProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.exitcode = 0
            created_processes.append(self)

        def start(self):
            # A worker that runs contributes at least one evaluation. Standing
            # in for that keeps the dataset past the initial samples, which is
            # what distinguishes a study that ran from one that did not.
            shared = self.args[2]
            shared['X'] = torch.cat(
                (shared['X'], torch.tensor([[0.5]], dtype=torch.double)), dim=0
            )
            shared['Y'] = torch.cat(
                (shared['Y'], torch.tensor([[0.7]], dtype=torch.double)), dim=0
            )

        def join(self):
            return None

    (tmp_path / 'init.csv').write_text('x_0,y\n0.1,0.2\n', encoding='utf-8')
    monkeypatch.setattr(
        async_mod, 'get_proteus_directories', lambda _output: {'output': str(tmp_path)}
    )
    monkeypatch.setattr(async_mod, 'Manager', FakeManager)
    monkeypatch.setattr(async_mod, 'Process', FakeProcess)
    monkeypatch.setattr(
        async_mod,
        'load_dataset_csv',
        lambda _path: {
            'X': torch.tensor([[0.1]], dtype=torch.double),
            'Y': torch.tensor([[0.2]], dtype=torch.double),
        },
    )
    monkeypatch.setattr(
        async_mod,
        'init_locs',
        lambda n_workers, _D_shared, acqf='LogEI': torch.tensor(
            [[0.2], [0.8]], dtype=torch.double
        )[:n_workers],
    )
    monkeypatch.setattr(async_mod, 'get_kernel', lambda *args, **kwargs: object())

    D_final, logs, elapsed = async_mod.parallel_process(
        objective_builder=lambda **kwargs: lambda x: x,
        kernel='MAT3/2',
        acqf='LogEI',
        n_workers=2,
        max_len=3,
        output='dummy',
        seed=1,
        ref_config='ref.toml',
        observables={'obs': 1.0},
        parameters={'a': [0.0, 1.0]},
        failure_codes=[],
    )

    assert len(created_processes) == 2
    # One initial sample plus one evaluation from each of the two workers.
    assert D_final['X'].shape == (3, 1)
    assert D_final['Y'].shape == (3, 1)
    assert logs == [None]
    assert elapsed == []


# ============================================================================
# Reporting workers that stop before the evaluation budget is reached
# ============================================================================


def _mocked_parallel_process_env(monkeypatch, tmp_path, fake_process_cls, n_init_rows=1):
    """Wire ``parallel_process`` to in-process fakes for the shared state.

    Returns nothing; the caller supplies the Process stand-in whose exit codes
    and side effects define the scenario under test.
    """

    class FakeManager:
        def dict(self, data=None):
            return {} if data is None else dict(data)

        def list(self, data=None):
            return [] if data is None else list(data)

        def Lock(self):
            return _DummyLock()

    (tmp_path / 'init.csv').write_text('x_0,y\n0.1,0.2\n', encoding='utf-8')
    monkeypatch.setattr(
        async_mod, 'get_proteus_directories', lambda _output: {'output': str(tmp_path)}
    )
    monkeypatch.setattr(async_mod, 'Manager', FakeManager)
    monkeypatch.setattr(async_mod, 'Process', fake_process_cls)
    monkeypatch.setattr(
        async_mod,
        'load_dataset_csv',
        lambda _path: {
            'X': torch.tensor([[0.1]] * n_init_rows, dtype=torch.double),
            'Y': torch.tensor([[0.2]] * n_init_rows, dtype=torch.double),
        },
    )
    monkeypatch.setattr(
        async_mod,
        'init_locs',
        lambda n_workers, _D_shared, acqf='LogEI': torch.tensor(
            [[0.2], [0.8]], dtype=torch.double
        )[:n_workers],
    )
    monkeypatch.setattr(async_mod, 'get_kernel', lambda *args, **kwargs: object())


@pytest.mark.unit
def test_parallel_process_reports_a_worker_that_stopped_early(monkeypatch, tmp_path, caplog):
    """A worker that dies mid-study leaves the run looking complete: the others
    carry on and the results are saved. The shortfall is reported by worker id
    and evaluation count so the summary that follows is not read as a full
    sweep of the requested budget.
    """

    class FakeProcess:
        _next = [0]

        def __init__(self, target, args):
            self.args = args
            self.worker_id = FakeProcess._next[0]
            FakeProcess._next[0] += 1
            # Worker 1 is killed by a signal, as an out-of-memory kill does,
            # which reports a negative code rather than a positive one.
            # Worker 0 contributes one evaluation.
            self.exitcode = -9 if self.worker_id == 1 else 0

        def start(self):
            if self.exitcode == 0:
                shared = self.args[2]
                shared['X'] = torch.cat(
                    (shared['X'], torch.tensor([[0.5]], dtype=torch.double)), dim=0
                )
                shared['Y'] = torch.cat(
                    (shared['Y'], torch.tensor([[0.7]], dtype=torch.double)), dim=0
                )

        def join(self):
            return None

    _mocked_parallel_process_env(monkeypatch, tmp_path, FakeProcess)

    with caplog.at_level('ERROR'):
        D_final, _logs, _elapsed = async_mod.parallel_process(
            objective_builder=lambda **kwargs: lambda x: x,
            kernel='MAT3/2',
            acqf='LogEI',
            n_workers=2,
            max_len=6,
            output='dummy',
            seed=1,
            ref_config='ref.toml',
            observables={'obs': 1.0},
            parameters={'a': [0.0, 1.0]},
            failure_codes=[],
        )

    # The partial study is still returned, so the evaluations that did complete
    # are not thrown away.
    assert D_final['X'].shape == (2, 1)
    reported = '\n'.join(record.getMessage() for record in caplog.records)
    # Identity guard: the dead worker is named, not merely counted. A
    # regression that reported "1 worker failed" without the id would leave
    # the user with nowhere to look.
    assert 'workers 1' in reported
    assert '1 of 2 workers' in reported
    # The signal code is reported as it stands. A guard written as
    # "exitcode > 0" would miss a killed worker entirely.
    assert '-9' in reported
    # Budget guard: the count actually achieved is contrasted with the count
    # requested, which is what makes the shortfall visible.
    assert '2 evaluations' in reported and '6 requested' in reported


@pytest.mark.unit
def test_parallel_process_stays_silent_when_every_worker_completes(
    monkeypatch, tmp_path, caplog
):
    """A study in which no worker died reports no shortfall. Without this, the
    failure message above would be indistinguishable from routine noise.
    """

    class FakeProcess:
        def __init__(self, target, args):
            self.args = args
            self.exitcode = 0

        def start(self):
            shared = self.args[2]
            shared['X'] = torch.cat(
                (shared['X'], torch.tensor([[0.5]], dtype=torch.double)), dim=0
            )
            shared['Y'] = torch.cat(
                (shared['Y'], torch.tensor([[0.7]], dtype=torch.double)), dim=0
            )

        def join(self):
            return None

    _mocked_parallel_process_env(monkeypatch, tmp_path, FakeProcess)

    with caplog.at_level('ERROR'):
        D_final, _logs, _elapsed = async_mod.parallel_process(
            objective_builder=lambda **kwargs: lambda x: x,
            kernel='MAT3/2',
            acqf='LogEI',
            n_workers=2,
            max_len=6,
            output='dummy',
            seed=1,
            ref_config='ref.toml',
            observables={'obs': 1.0},
            parameters={'a': [0.0, 1.0]},
            failure_codes=[],
        )

    assert D_final['X'].shape == (3, 1)
    assert [r for r in caplog.records if r.levelname == 'ERROR'] == []


@pytest.mark.unit
def test_parallel_process_refuses_a_study_with_no_completed_steps(monkeypatch, tmp_path):
    """When the dataset never grows past the initial samples there is no
    optimisation to report, and the best-fit summary downstream would describe
    the initial design while presenting it as an inference result. The study
    stops instead, naming the reason.
    """

    class FakeProcess:
        def __init__(self, target, args):
            self.args = args
            self.exitcode = 1

        def start(self):
            return None

        def join(self):
            return None

    _mocked_parallel_process_env(monkeypatch, tmp_path, FakeProcess)

    with pytest.raises(RuntimeError) as excinfo:
        async_mod.parallel_process(
            objective_builder=lambda **kwargs: lambda x: x,
            kernel='MAT3/2',
            acqf='LogEI',
            n_workers=2,
            max_len=6,
            output='dummy',
            seed=1,
            ref_config='ref.toml',
            observables={'obs': 1.0},
            parameters={'a': [0.0, 1.0]},
            failure_codes=[],
        )
    message = str(excinfo.value)
    assert 'No optimisation steps completed' in message
    # The cause is attributed to the workers, not to the evaluation budget,
    # because they reported non-zero exit codes.
    assert '2 of 2 workers stopped early' in message


@pytest.mark.unit
def test_worker_releases_its_busy_point_and_records_why_it_stopped(tmp_path, caplog):
    """A worker that fails records the cause in the study log before it dies,
    and releases the point it had claimed. Neither happens on its own:
    multiprocessing prints a dead worker's traceback straight to the parent's
    stderr without consulting the logging configuration, and a claimed point
    left in place steers the surviving workers away from a region nothing is
    exploring.
    """
    D_shared = {
        'X': torch.tensor([[0.1]], dtype=torch.double),
        'Y': torch.tensor([[0.2]], dtype=torch.double),
    }
    B = {0: torch.tensor([[0.3]], dtype=torch.double)}

    def exploding_process_fun(**_kwargs):
        raise RuntimeError('objective evaluation failed')

    with caplog.at_level('ERROR'):
        with pytest.raises(RuntimeError, match='objective evaluation failed'):
            async_mod.worker(
                process_fun=exploding_process_fun,
                build_obj=lambda **kwargs: lambda x: x,
                D_shared=D_shared,
                B=B,
                T=[],
                T0=0.0,
                x_init=torch.tensor([[0.3]], dtype=torch.double),
                n_init=1,
                lock=_DummyLock(),
                max_len=4,
                worker_id=0,
                log_list=[],
                output_dir=str(tmp_path),
            )

    # The claimed point is released.
    assert 0 not in B
    # The cause reached the study log, with a traceback attached.
    records = [r for r in caplog.records if r.levelname == 'ERROR']
    assert any('Worker 0 stopped early' in r.getMessage() for r in records)
    assert any(r.exc_info is not None for r in records)
    # Nothing was appended to the shared dataset, so a failed evaluation
    # cannot masquerade as a completed one.
    assert D_shared['X'].shape == (1, 1)


@pytest.mark.unit
def test_worker_releases_its_busy_point_after_a_normal_finish(tmp_path):
    """A worker that reaches the evaluation budget also releases its claimed
    point. Left behind, it would bias the acquisition for every worker still
    running through the tail of the study.
    """
    D_shared = {
        'X': torch.tensor([[0.1], [0.2]], dtype=torch.double),
        'Y': torch.tensor([[0.3], [0.4]], dtype=torch.double),
    }
    B = {
        0: torch.tensor([[0.5]], dtype=torch.double),
        1: torch.tensor([[0.6]], dtype=torch.double),
    }

    # max_len is already reached, so the loop exits without an evaluation.
    async_mod.worker(
        process_fun=lambda **_kwargs: pytest.fail('no evaluation should run'),
        build_obj=lambda **kwargs: lambda x: x,
        D_shared=D_shared,
        B=B,
        T=[],
        T0=0.0,
        x_init=torch.tensor([[0.5]], dtype=torch.double),
        n_init=2,
        lock=_DummyLock(),
        max_len=2,
        worker_id=0,
        log_list=[],
        output_dir=str(tmp_path),
    )

    assert 0 not in B
    # Only this worker's claim is released; the other worker is still running.
    assert 1 in B


@pytest.mark.unit
def test_parallel_process_names_the_real_step_budget_when_no_worker_failed(
    monkeypatch, tmp_path
):
    """A study configured with fewer optimisation steps than workers finishes
    without any worker failing and without any step being taken. The refusal
    must quote the row count the workers actually stop at, which is the
    requested budget less one per worker beyond the first, or the advice reads
    as false against the numbers the user set.
    """

    class FakeProcess:
        def __init__(self, target, args):
            self.args = args
            self.exitcode = 0

        def start(self):
            # Every worker sees the budget already met and exits at once,
            # contributing nothing to the dataset.
            return None

        def join(self):
            return None

    # Six initial samples against a budget of six with two workers: the worker
    # threshold is 6 - (2 - 1) = 5, which the initial samples already exceed.
    _mocked_parallel_process_env(monkeypatch, tmp_path, FakeProcess, n_init_rows=6)

    with pytest.raises(RuntimeError) as excinfo:
        async_mod.parallel_process(
            objective_builder=lambda **kwargs: lambda x: x,
            kernel='MAT3/2',
            acqf='LogEI',
            n_workers=2,
            max_len=6,
            output='dummy',
            seed=1,
            ref_config='ref.toml',
            observables={'obs': 1.0},
            parameters={'a': [0.0, 1.0]},
            failure_codes=[],
        )
    message = str(excinfo.value)
    assert 'No worker failed' in message
    # The threshold the workers actually apply.
    assert 'reaches 5 rows' in message
    # Discrimination: quoting the requested budget here instead would state
    # that six initial samples satisfy a six-row threshold, which is false.
    assert 'reaches 6 rows' not in message
    # The requested budget is still named, so the two numbers can be related.
    assert '6 requested' in message
    assert 'Raise n_steps to at least n_workers (2)' in message


@pytest.mark.unit
def test_worker_writes_its_traceback_to_the_study_logfile(tmp_path):
    """A worker started with the 'spawn' method inherits no logging
    configuration, so the report of its death would go to stderr and never
    reach the study logfile. Given the logfile path, the worker reopens it
    and the traceback lands where the study is read from.

    The 'fwl' logger is emptied here to stand in for a spawned process, which
    is what the parent's handlers are absent in; pytest's own capture would
    otherwise hide the gap this covers.
    """
    logger = logging.getLogger('fwl')
    saved_handlers, saved_level = list(logger.handlers), logger.level
    logger.handlers.clear()

    logpath = tmp_path / 'infer.log'
    logpath.write_text('[ INFO  ] study started\n', encoding='utf-8')

    D_shared = {
        'X': torch.tensor([[0.1]], dtype=torch.double),
        'Y': torch.tensor([[0.2]], dtype=torch.double),
    }
    # Keyed by this worker's own id, so the release below is a real check.
    B = {3: torch.tensor([[0.3]], dtype=torch.double)}

    def exploding_process_fun(**_kwargs):
        raise RuntimeError('objective evaluation failed')

    try:
        with pytest.raises(RuntimeError, match='objective evaluation failed'):
            async_mod.worker(
                process_fun=exploding_process_fun,
                build_obj=lambda **kwargs: lambda x: x,
                D_shared=D_shared,
                B=B,
                T=[],
                T0=0.0,
                x_init=torch.tensor([[0.3]], dtype=torch.double),
                n_init=1,
                lock=_DummyLock(),
                max_len=4,
                worker_id=3,
                log_list=[],
                output_dir=str(tmp_path),
                logpath=str(logpath),
                log_level=logging.INFO,
            )
        for handler in logging.getLogger('fwl').handlers:
            handler.flush()
        text = logpath.read_text(encoding='utf-8')
    finally:
        logger.handlers.clear()
        logger.handlers.extend(saved_handlers)
        logger.setLevel(saved_level)

    assert 'Worker 3 stopped early' in text
    # The cause, not just the headline: a report without the traceback body
    # would leave the study with no more than the fact that something failed.
    assert 'RuntimeError: objective evaluation failed' in text
    # Appended, never recreated: the lines written before the worker started
    # are what place the failure in the run.
    assert 'study started' in text
    # The busy point is still released on the way out, so the logfile change
    # has not displaced the behaviour the failure path already had.
    assert B == {}


@pytest.mark.unit
def test_worker_without_a_logfile_path_leaves_logging_untouched(tmp_path, caplog):
    """Under 'fork' the parent's handlers are inherited, so `parallel_process`
    passes no path and the worker must not attach one of its own; a second
    handler on the same file would double every line. The failure is still
    reported through whatever configuration the process already has.
    """
    logger = logging.getLogger('fwl')
    before = list(logger.handlers)

    D_shared = {
        'X': torch.tensor([[0.1]], dtype=torch.double),
        'Y': torch.tensor([[0.2]], dtype=torch.double),
    }
    B = {0: torch.tensor([[0.3]], dtype=torch.double)}

    def exploding_process_fun(**_kwargs):
        raise RuntimeError('objective evaluation failed')

    with caplog.at_level('ERROR'):
        with pytest.raises(RuntimeError, match='objective evaluation failed'):
            async_mod.worker(
                process_fun=exploding_process_fun,
                build_obj=lambda **kwargs: lambda x: x,
                D_shared=D_shared,
                B=B,
                T=[],
                T0=0.0,
                x_init=torch.tensor([[0.3]], dtype=torch.double),
                n_init=1,
                lock=_DummyLock(),
                max_len=4,
                worker_id=0,
                log_list=[],
                output_dir=str(tmp_path),
            )

    # No handler added, and none taken away.
    assert list(logger.handlers) == before
    # No stray logfile created beside the run output.
    assert not (tmp_path / 'infer.log').exists()
    reported = '\n'.join(record.getMessage() for record in caplog.records)
    assert 'Worker 0 stopped early' in reported
