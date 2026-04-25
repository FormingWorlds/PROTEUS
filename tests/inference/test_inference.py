# This test runs the BO inference scheme with PROTEUS for a few evaluations
# ruff: noqa: E402, I001
from __future__ import annotations

import filecmp

# Pytest will hang on process completion when using multiprocessing, by default
#    Add these lines to ensure that the processes do not block
#    See issue: https://github.com/pytest-dev/pytest/issues/11174#issuecomment-1921876937
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

import os

import pandas as pd
import pytest
from helpers import PROTEUS_ROOT

from proteus.inference.inference import infer_from_config

OUT_DIR = PROTEUS_ROOT / 'output' / 'dummy_inference'

INFER_CONFIG = PROTEUS_ROOT / 'tests' / 'inference' / 'dummy.infer.toml'
BASE_CONFIG = PROTEUS_ROOT / 'tests' / 'inference' / 'base.toml'


@pytest.fixture(scope='module')
def inference_run():
    infer_from_config(INFER_CONFIG)


@pytest.mark.integration
def test_inference_run(inference_run):
    # Call fixture to ensure that it has run without error
    pass


@pytest.mark.integration
def test_inference_config(inference_run):
    # Copy of grid's config exists in output dir
    assert os.path.isfile(OUT_DIR / 'copy.infer.toml')

    # Copy of base config is identical to base config
    assert filecmp.cmp(OUT_DIR / 'ref_config.toml', BASE_CONFIG, shallow=False)


@pytest.mark.integration
def test_inference_init(inference_run):
    # Check that init data was written
    assert os.path.isfile(OUT_DIR / 'init.csv')

    # Check init data format
    data = pd.read_csv(OUT_DIR / 'init.csv')
    assert 'y' in data.columns
    assert any(col.startswith('x_') for col in data.columns)


@pytest.mark.integration
def test_inference_output(inference_run):
    # Check that output results exist
    assert os.path.isfile(OUT_DIR / 'data.csv')

    # Check output result format
    data = pd.read_csv(OUT_DIR / 'data.csv')
    assert 'y' in data.columns
    assert any(col.startswith('x_') for col in data.columns)
    assert len(data['y']) > 2

    # Check plots exist
    assert os.path.isfile(OUT_DIR / 'plots' / 'result_correlation.png')
    assert os.path.isfile(OUT_DIR / 'plots' / 'result_objective.png')
