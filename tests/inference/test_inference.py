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

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Mixed-tier file: 5 unit tests + 3 slow tests (subprocess-driven). The
# slow tests carry @pytest.mark.slow per-function and run only in the
# nightly tier; the fast PR filter "unit and not slow" selects only
# the 5 unit tests. Do not move a slow test to the unit tier without
# refitting it to the 100 ms / 30 s wall budget.


# Pytest can hang on process completion when using multiprocessing by default.
mp.set_start_method('spawn', force=True)

OUT_DIR = PROTEUS_ROOT / 'output' / 'dummy_inference'
INFER_CONFIG = PROTEUS_ROOT / 'tests' / 'inference' / 'dummy.infer.toml'
BASE_CONFIG = PROTEUS_ROOT / 'tests' / 'inference' / 'base.toml'


@pytest.fixture(scope='module')
def inference_run():
    infer_from_config(INFER_CONFIG)


# The three tests below run the full inference pipeline as a child PROTEUS
# subprocess. Each evaluation takes ~75 s wall (~5 min for 4 evaluations),
# which exceeds the smoke tier's per-test budget. Tagged `slow` so they
# run on-demand without gating PR CI; the unit-tier test below
# (test_run_inference_rejects_too_many_workers) covers the fast-feedback
# portion of the inference contract.
#
# The earlier `test_inference_smoke_run` was removed: its only assertion
# was `assert inference_run is None`, which is trivially true because the
# fixture returns None implicitly. The three remaining tests check actual
# output files, which is the meaningful evidence that the run completed.


@pytest.mark.slow
def test_inference_smoke_config(inference_run):
    """A finished inference run leaves a copy of its config TOML
    (``copy.infer.toml``) and a bit-identical copy of the reference base
    config under the output directory. Reproducibility hook.
    """
    assert os.path.isfile(OUT_DIR / 'copy.infer.toml')
    assert filecmp.cmp(OUT_DIR / 'ref_config.toml', BASE_CONFIG, shallow=False)


@pytest.mark.slow
def test_inference_smoke_init(inference_run):
    """The inference run produces an ``init.csv`` containing the
    Halton-sampled initial design, with the canonical ``y`` objective
    column and at least one ``x_*`` parameter column.
    """
    assert os.path.isfile(OUT_DIR / 'init.csv')
    data = pd.read_csv(OUT_DIR / 'init.csv')
    assert 'y' in data.columns
    assert any(col.startswith('x_') for col in data.columns)


@pytest.mark.slow
def test_inference_smoke_output(inference_run):
    """The inference run produces a ``data.csv`` with at least three rows
    (initial design + BO iterations) and writes the two canonical result
    plots (``result_correlation.png``, ``result_objective.png``).
    """
    assert os.path.isfile(OUT_DIR / 'data.csv')

    data = pd.read_csv(OUT_DIR / 'data.csv')
    assert 'y' in data.columns
    assert any(col.startswith('x_') for col in data.columns)
    assert len(data['y']) > 2

    assert os.path.isfile(OUT_DIR / 'plots' / 'result_correlation.png')
    assert os.path.isfile(OUT_DIR / 'plots' / 'result_objective.png')


@pytest.mark.unit
def test_run_inference_rejects_too_many_workers(monkeypatch, tmp_path):
    """``run_inference`` rejects ``n_workers >= cpu_count`` with a
    'Not enough CPU cores' error, so a misconfigured job fails at
    config-load rather than DoSing the host machine.
    """
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

    create_init_calls: list = []
    monkeypatch.setattr(
        inference_mod, 'create_init', lambda *a, **kw: create_init_calls.append((a, kw))
    )
    with pytest.raises(RuntimeError, match='Not enough CPU cores'):
        inference_mod.run_inference(config)
    # Discrimination: the CPU-count guard must fire before any expensive
    # initial-design dispatch. A regression that allowed init sampling to
    # start and only raised at parallel_process would have a non-empty
    # call list here.
    assert create_init_calls == []


@pytest.mark.unit
def test_run_inference_raises_for_missing_reference_config(monkeypatch, tmp_path):
    """``run_inference`` raises FileNotFoundError when ``ref_config`` does
    not point to an existing file on disk, naming the missing path.
    """
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

    create_init_calls: list = []
    monkeypatch.setattr(
        inference_mod, 'create_init', lambda *a, **kw: create_init_calls.append((a, kw))
    )
    with pytest.raises(FileNotFoundError, match='Cannot find reference config'):
        inference_mod.run_inference(config)
    # Discrimination: the missing-config guard must fire before initial
    # design generation. A regression that built the init dataset and
    # only failed later would have a non-empty call list here.
    assert create_init_calls == []


@pytest.mark.unit
def test_infer_from_config_loads_toml_and_dispatches(monkeypatch, tmp_path):
    """``infer_from_config(path)`` parses the TOML and forwards the
    resulting dict verbatim to ``run_inference``; no field is dropped or
    renamed in the dispatch step.
    """
    config_path = tmp_path / 'inference.toml'
    expected = {'output': 'dummy', 'n_workers': 1}
    config_path.write_text(toml.dumps(expected), encoding='utf-8')

    observed = {}

    def fake_run_inference(cfg):
        observed['config'] = cfg

    monkeypatch.setattr(inference_mod, 'run_inference', fake_run_inference)

    inference_mod.infer_from_config(str(config_path))

    assert observed['config'] == expected
    # Discrimination: a regression that mutated the dispatched config in
    # place would leave a still-equal-to-expected dict but with extra keys
    # injected. Pin the exact key set.
    assert set(observed['config'].keys()) == set(expected.keys())


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
    (planet.mass_tot). The deprecated struct.mass_tot path must not
    appear; it would mislead any reader who copies the docstring example
    into a working config."""
    from inspect import getdoc

    from proteus.inference.utils import get_nested

    doc = getdoc(get_nested)
    assert doc is not None
    assert 'struct.mass_tot' not in doc, (
        'docstring example must not reference the deprecated struct.* schema'
    )
    assert 'planet.mass_tot' in doc, 'docstring example must use the current planet.* schema'
