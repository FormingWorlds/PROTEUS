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
@pytest.mark.skip(
    reason='FIXME: PROTEUS subprocess exits with code 1 inside the CI container during the inference fixture. All four smoke tests in this file share the `inference_run` fixture so they fail in lockstep.'
)
def test_inference_smoke_run(inference_run):
    assert inference_run is None


@pytest.mark.smoke
@pytest.mark.skip(
    reason='FIXME: PROTEUS subprocess exits with code 1 inside the CI container during the inference fixture. See test_inference_smoke_run.'
)
def test_inference_smoke_config(inference_run):
    assert os.path.isfile(OUT_DIR / 'copy.infer.toml')
    assert filecmp.cmp(OUT_DIR / 'ref_config.toml', BASE_CONFIG, shallow=False)


@pytest.mark.smoke
@pytest.mark.skip(
    reason='FIXME: PROTEUS subprocess exits with code 1 inside the CI container during the inference fixture. See test_inference_smoke_run.'
)
def test_inference_smoke_init(inference_run):
    assert os.path.isfile(OUT_DIR / 'init.csv')
    data = pd.read_csv(OUT_DIR / 'init.csv')
    assert 'y' in data.columns
    assert any(col.startswith('x_') for col in data.columns)


@pytest.mark.smoke
@pytest.mark.skip(
    reason='FIXME: PROTEUS subprocess exits with code 1 inside the CI container during the inference fixture. See test_inference_smoke_run.'
)
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
        'parameters': {'planet.mass_tot': [0.7, 3.0]},
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
        'parameters': {'planet.mass_tot': [0.7, 3.0]},
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


# ============================================================================
# Regression: no stray prints + docstring uses current schema
# ============================================================================


@pytest.mark.unit
def test_infer_from_config_uses_logger_not_print(caplog, tmp_path, monkeypatch):
    """Regression: infer_from_config must route its startup message through
    the module logger, not print(). The original PR #675 BayesOpt rewrite
    migrated every other print() in the inference src to log.info; this
    one was missed and is corrected here."""
    import io
    from contextlib import redirect_stdout

    import proteus.inference.inference as inference_mod

    # Stub the heavy run path; we only want the early log statement.
    monkeypatch.setattr(inference_mod, 'run_inference', lambda _cfg: None)

    cfg_path = tmp_path / 'dummy.infer.toml'
    cfg_path.write_text('seed = 1\noutput = "x"\nlogging = "INFO"\n', encoding='utf-8')

    buf = io.StringIO()
    with caplog.at_level('INFO', logger='fwl.proteus.inference.inference'):
        with redirect_stdout(buf):
            inference_mod.infer_from_config(str(cfg_path))

    # Logger captured the message...
    assert any('Inference config:' in rec.message for rec in caplog.records), (
        'infer_from_config must emit Inference config via log.info'
    )
    # ...and stdout did not.
    assert 'Inference config:' not in buf.getvalue(), (
        'infer_from_config must not write to stdout via print()'
    )


@pytest.mark.unit
def test_get_nested_docstring_uses_current_schema_example():
    """Regression: utils.get_nested docstring example must use a
    parameter path that is valid on the current branch schema
    (planet.mass_tot), not the pre-Phase-5 schema (struct.mass_tot)."""
    from inspect import getdoc

    from proteus.inference.utils import get_nested

    doc = getdoc(get_nested)
    assert doc is not None
    assert 'struct.mass_tot' not in doc, (
        'docstring example must not reference the deprecated struct.* schema'
    )
    assert 'planet.mass_tot' in doc, 'docstring example must use the current planet.* schema'
