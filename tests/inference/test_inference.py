"""
Tests for the inference pipeline entrypoints.

References:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import filecmp
import multiprocessing as mp
import os

import pandas as pd
import pytest
import toml
from helpers import PROTEUS_ROOT

import proteus.inference.inference as inference_mod
from proteus.inference.inference import infer_from_config

# Pytest can hang on process completion when using multiprocessing by default.
mp.set_start_method('spawn', force=True)

OUT_DIR = PROTEUS_ROOT / 'output' / 'dummy_inference'
INFER_CONFIG = PROTEUS_ROOT / 'tests' / 'inference' / 'dummy.infer.toml'
BASE_CONFIG = PROTEUS_ROOT / 'tests' / 'inference' / 'base.toml'


@pytest.fixture(scope='module')
def inference_run():
    infer_from_config(INFER_CONFIG)


@pytest.mark.smoke
def test_inference_smoke_run(inference_run):
    assert inference_run is None


@pytest.mark.smoke
def test_inference_smoke_config(inference_run):
    assert os.path.isfile(OUT_DIR / 'copy.infer.toml')
    assert filecmp.cmp(OUT_DIR / 'ref_config.toml', BASE_CONFIG, shallow=False)


@pytest.mark.smoke
def test_inference_smoke_init(inference_run):
    assert os.path.isfile(OUT_DIR / 'init.csv')
    data = pd.read_csv(OUT_DIR / 'init.csv')
    assert 'y' in data.columns
    assert any(col.startswith('x_') for col in data.columns)


@pytest.mark.smoke
def test_inference_smoke_output(inference_run):
    assert os.path.isfile(OUT_DIR / 'data.csv')

    data = pd.read_csv(OUT_DIR / 'data.csv')
    assert 'y' in data.columns
    assert any(col.startswith('x_') for col in data.columns)
    assert len(data['y']) > 2

    assert os.path.isfile(OUT_DIR / 'plots' / 'result_correlation.png')
    assert os.path.isfile(OUT_DIR / 'plots' / 'result_objective.png')


@pytest.mark.unit
def test_run_inference_rejects_too_many_workers(monkeypatch, tmp_path):
    config = {
        'output': 'unit_inference',
        'logging': 'INFO',
        'n_workers': 4,
        'ref_config': 'tests/inference/base.toml',
        'n_steps': 1,
        'kernel': 'MAT3/2',
        'acqf': 'LogEI',
        'seed': 1,
        'observables': {'P_surf': 1.0},
        'parameters': {'struct.mass_tot': [0.7, 3.0]},
    }
    output_root = tmp_path / 'output'

    monkeypatch.setattr(
        inference_mod,
        'get_proteus_directories',
        lambda _output: {'output': str(output_root), 'proteus': str(tmp_path)},
    )
    monkeypatch.setattr(inference_mod, 'safe_rm', lambda _path: None)
    monkeypatch.setattr(inference_mod, 'setup_logger', lambda **_kwargs: None)
    monkeypatch.setattr(inference_mod, 'str_time', lambda: '2026-04-30 00:00:00 UTC')
    monkeypatch.setattr(inference_mod.os, 'cpu_count', lambda: 4)

    with pytest.raises(RuntimeError, match='Not enough CPU cores'):
        inference_mod.run_inference(config)


@pytest.mark.unit
def test_run_inference_raises_for_missing_reference_config(monkeypatch, tmp_path):
    config = {
        'output': 'unit_inference',
        'logging': 'INFO',
        'n_workers': 1,
        'ref_config': 'missing.toml',
        'n_steps': 1,
        'kernel': 'MAT3/2',
        'acqf': 'LogEI',
        'seed': 1,
        'observables': {'P_surf': 1.0},
        'parameters': {'struct.mass_tot': [0.7, 3.0]},
    }
    output_root = tmp_path / 'output'

    monkeypatch.setattr(
        inference_mod,
        'get_proteus_directories',
        lambda _output: {'output': str(output_root), 'proteus': str(tmp_path)},
    )
    monkeypatch.setattr(inference_mod, 'safe_rm', lambda _path: None)
    monkeypatch.setattr(inference_mod, 'setup_logger', lambda **_kwargs: None)
    monkeypatch.setattr(inference_mod, 'str_time', lambda: '2026-04-30 00:00:00 UTC')
    monkeypatch.setattr(inference_mod.os, 'cpu_count', lambda: 8)
    monkeypatch.setattr(inference_mod.os.path, 'isfile', lambda _path: False)

    with pytest.raises(FileNotFoundError, match='Cannot find reference config'):
        inference_mod.run_inference(config)


@pytest.mark.unit
def test_infer_from_config_loads_toml_and_dispatches(monkeypatch, tmp_path):
    config_path = tmp_path / 'inference.toml'
    expected = {'output': 'dummy', 'n_workers': 1}
    config_path.write_text(toml.dumps(expected), encoding='utf-8')

    observed = {}

    def fake_run_inference(cfg):
        observed['config'] = cfg

    monkeypatch.setattr(inference_mod, 'run_inference', fake_run_inference)

    inference_mod.infer_from_config(str(config_path))

    assert observed['config'] == expected
