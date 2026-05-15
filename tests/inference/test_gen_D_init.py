"""
Unit tests for inference initial-dataset generation utilities.

References:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import toml
import torch

import proteus.inference.gen_D_init as init_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_create_init_rejects_small_sample_count():
    """``create_init`` rejects ``init_samps < 2`` because the GP needs at
    least two distinct samples for a meaningful prior fit.
    """
    config = {'init_grid': 'none', 'init_samps': 1}
    with pytest.raises(ValueError, match='must contain >1 sample'):
        init_mod.create_init(config)


@pytest.mark.unit
def test_create_init_routes_to_sample_from_bounds(monkeypatch):
    """``create_init`` with ``init_grid='none'`` dispatches to
    ``sample_from_bounds`` (Halton-sequence sampling of the parameter
    box), not to ``sample_from_grid``.
    """
    config = {
        'init_grid': 'none',
        'init_samps': 4,
        'output': 'out',
        'ref_config': 'ref.toml',
        'parameters': {'planet.mass_tot': [0.7, 3.0]},
        'observables': {'R_obs': 1.0},
        'seed': 1,
        'n_workers': 2,
    }
    monkeypatch.setattr(init_mod, 'sample_from_bounds', lambda *args, **kwargs: 4)
    assert init_mod.create_init(config) == 4


@pytest.mark.unit
def test_create_init_routes_to_sample_from_grid(monkeypatch, tmp_path):
    """``create_init`` with a non-'none' ``init_grid`` dispatches to
    ``sample_from_grid`` and resolves the grid path via
    ``proteus_directories.output``.
    """
    observed = {}
    monkeypatch.setattr(init_mod, 'get_proteus_directories', lambda: {'proteus': str(tmp_path)})

    def fake_sample_from_grid(output, params, observables, grid_dir):
        observed['grid_dir'] = grid_dir
        return 6

    monkeypatch.setattr(init_mod, 'sample_from_grid', fake_sample_from_grid)
    config = {
        'init_grid': 'my_grid',
        'init_samps': 3,
        'output': 'out',
        'parameters': {'planet.mass_tot': [0.7, 3.0]},
        'observables': {'R_obs': 1.0},
    }

    assert init_mod.create_init(config) == 6
    assert observed['grid_dir'] == str(tmp_path / 'output' / 'my_grid')


@pytest.mark.unit
def test_sample_from_grid_builds_and_saves_dataset(monkeypatch, tmp_path):
    """``sample_from_grid`` walks each ``case_N/`` subdirectory, reads
    the case parameters from ``init_coupler.toml``, reads the observable
    from ``runtime_helpfile.csv``, and saves the combined dataset as
    ``init.csv`` with canonical columns ``x_0, y``.
    """
    grid_dir = tmp_path / 'grid'
    output_dir = tmp_path / 'output'
    output_dir.mkdir(parents=True)
    for i, mass in enumerate([1.0, 2.0]):
        case = grid_dir / f'case_{i}'
        case.mkdir(parents=True)
        pd.DataFrame([{'R_obs': 1.5 + i}]).to_csv(
            case / 'runtime_helpfile.csv', sep=' ', index=False
        )
        (case / 'init_coupler.toml').write_text(
            toml.dumps({'planet': {'mass_tot': mass}}), encoding='utf-8'
        )

    monkeypatch.setattr(
        init_mod, 'get_proteus_directories', lambda _output: {'output': str(output_dir)}
    )

    n = init_mod.sample_from_grid(
        output='ignored',
        params={'planet.mass_tot': [0.0, 10.0]},
        observables={'R_obs': 1.0},
        grid_dir=str(grid_dir),
    )

    data = pd.read_csv(output_dir / 'init.csv')
    assert n == 2
    assert list(data.columns) == ['x_0', 'y']
    assert len(data) == 2


@pytest.mark.unit
def test_sample_from_bounds_rejects_invalid_worker_count():
    """``sample_from_bounds`` rejects ``n_workers < 1`` with an
    'at least 1' message, so a misconfigured worker pool fails loudly
    rather than silently producing zero samples.
    """
    with pytest.raises(ValueError, match='at least 1'):
        init_mod.sample_from_bounds(
            output='out',
            ref_config='ref.toml',
            params={'a': [0.0, 1.0]},
            observables={'obs': 1.0},
            nsamp=2,
            seed=1,
            n_workers=0,
        )


@pytest.mark.unit
def test_sample_from_bounds_caps_workers_and_saves(monkeypatch, tmp_path):
    """``sample_from_bounds`` caps the worker pool at ``cpu_count - 1``
    (3 in this test, with cpu_count=4 mocked) regardless of the user's
    request, uses Halton sequences for the initial design, and saves
    the resulting (X, Y) dataset to ``init.csv``.
    """
    captured = {}

    class FakeHalton:
        def __init__(self, d, seed, scramble):
            captured['dims'] = d
            captured['scramble'] = scramble

        def random(self, n):
            return np.array([[0.1], [0.9]])[:n]

    class FakePool:
        def __init__(self, processes):
            captured['processes'] = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starmap(self, func, args):
            captured['task_count'] = len(args)
            return [torch.tensor([[0.2]], dtype=torch.double) for _ in args]

    monkeypatch.setattr(init_mod.os, 'cpu_count', lambda: 4)
    monkeypatch.setattr(init_mod, 'Halton', FakeHalton)
    monkeypatch.setattr(init_mod, 'Pool', FakePool)
    monkeypatch.setattr(
        init_mod, 'get_proteus_directories', lambda _output: {'output': str(tmp_path)}
    )

    def fake_save_dataset_csv(X, Y, fpath):
        captured['saved_shape'] = (tuple(X.shape), tuple(Y.shape))
        captured['saved_path'] = fpath

    monkeypatch.setattr(init_mod, 'save_dataset_csv', fake_save_dataset_csv)

    n = init_mod.sample_from_bounds(
        output='out',
        ref_config='ref.toml',
        params={'a': [0.0, 1.0]},
        observables={'obs': 1.0},
        nsamp=2,
        seed=11,
        n_workers=10,
    )

    assert n == 2
    assert captured['processes'] == 3
    assert captured['task_count'] == 2
    assert captured['saved_shape'] == ((2, 1), (2, 1))
    assert captured['saved_path'].endswith('init.csv')
