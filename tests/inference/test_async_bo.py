"""
Unit tests for asynchronous Bayesian optimization orchestration helpers.

References:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

import proteus.inference.async_BO as async_mod


class _DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.mark.unit
def test_checkpoint_writes_expected_files(tmp_path):
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


@pytest.mark.unit
def test_parallel_process_raises_when_init_dataset_missing(monkeypatch, tmp_path):
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


@pytest.mark.unit
def test_parallel_process_happy_path_with_mocked_manager(monkeypatch, tmp_path):
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
            created_processes.append(self)

        def start(self):
            return None

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
        lambda n_workers, _D_shared: torch.tensor([[0.2], [0.8]], dtype=torch.double)[
            :n_workers
        ],
    )
    monkeypatch.setattr(async_mod, 'get_kernel_w_prior', lambda **kwargs: object())

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
    assert D_final['X'].shape == (1, 1)
    assert D_final['Y'].shape == (1, 1)
    assert logs == [None]
    assert elapsed == []
