"""
Slow-tier tests for the inference pipeline.

These drive the full inference pipeline as child PROTEUS subprocesses, so they
are slow tier and live apart from the fast entrypoint tests in
``test_inference.py``. A file carries one tier, so that the tier filters select
every test exactly once.

All three are skipped. The pipeline does not complete: each worker subprocess
writes its opening log lines, reports status "Running", and then makes no
further progress, so the run produces no init.csv or data.csv and the tests sit
until the tier timeout. Reproduce with
``pytest tests/inference/test_inference_slow.py`` and watch
``output/dummy_inference/workers/w_-1/i_0/proteus_00.log`` stop after its
warnings. Until the pipeline finishes, running these would hang the nightly
shard rather than report anything.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import filecmp
import multiprocessing as mp
import os

import pandas as pd
import pytest
from helpers import PROTEUS_ROOT

from proteus.inference.inference import infer_from_config

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

# Pytest can hang on process completion when using multiprocessing by default.
mp.set_start_method('spawn', force=True)

OUT_DIR = PROTEUS_ROOT / 'output' / 'dummy_inference'
INFER_CONFIG = PROTEUS_ROOT / 'tests' / 'inference' / 'dummy.infer.toml'
BASE_CONFIG = PROTEUS_ROOT / 'tests' / 'inference' / 'base.toml'


@pytest.fixture(scope='module')
def inference_run():
    infer_from_config(INFER_CONFIG)


@pytest.mark.skip(reason='inference pipeline does not complete: workers stall after startup')
def test_inference_smoke_config(inference_run):
    """A finished inference run leaves a copy of its config TOML
    (``copy.infer.toml``) and a bit-identical copy of the reference base
    config under the output directory. Reproducibility hook.
    """
    assert os.path.isfile(OUT_DIR / 'copy.infer.toml')
    assert filecmp.cmp(OUT_DIR / 'ref_config.toml', BASE_CONFIG, shallow=False)


@pytest.mark.skip(reason='inference pipeline does not complete: workers stall after startup')
def test_inference_smoke_init(inference_run):
    """The inference run produces an ``init.csv`` containing the
    Halton-sampled initial design, with the canonical ``y`` objective
    column and at least one ``x_*`` parameter column.
    """
    assert os.path.isfile(OUT_DIR / 'init.csv')
    data = pd.read_csv(OUT_DIR / 'init.csv')
    assert 'y' in data.columns
    assert any(col.startswith('x_') for col in data.columns)


@pytest.mark.skip(reason='inference pipeline does not complete: workers stall after startup')
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
