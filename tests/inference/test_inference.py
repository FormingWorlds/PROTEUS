# This test runs the BO inference scheme with PROTEUS for a few evaluations
from __future__ import annotations

import filecmp
import os
import pickle

import pytest
from helpers import PROTEUS_ROOT

from proteus.inference.inference import infer_from_config

OUT_DIR = PROTEUS_ROOT / 'output' / 'dummy_inference'

INFER_CONFIG = PROTEUS_ROOT / 'tests' / 'inference' / 'dummy.infer.toml'
BASE_CONFIG = PROTEUS_ROOT / 'tests' / 'inference' / 'base.toml'

@pytest.fixture(scope="module")
def inference_run():
    infer_from_config(INFER_CONFIG)

def test_inference_run(inference_run):
    # Call fixture to ensure that it has run without error
    pass

def test_inference_config(inference_run):
    # Copy of grid's config exists in output dir
    assert os.path.isfile(OUT_DIR / 'copy.infer.toml')

    # Copy of base config is identical to base config
    assert filecmp.cmp(OUT_DIR / 'ref_config.toml', BASE_CONFIG, shallow=False)

def test_inference_init(inference_run):
    # Check that init data was written
    assert os.path.isfile(OUT_DIR / 'init.pkl')

    # Chec init data format
    with open(OUT_DIR / 'data.pkl', 'rb') as hdl:
        data = pickle.load(hdl)
    assert "X" in data.keys()
    assert "Y" in data.keys()

def test_inference_output(inference_run):
    # Check that output results exist
    assert os.path.isfile(OUT_DIR / 'data.pkl')

    # Check output result format
    with open(OUT_DIR / 'data.pkl', 'rb') as hdl:
        data = pickle.load(hdl)
    assert "X" in data.keys()
    assert "Y" in data.keys()
    assert len(data["Y"]) > 2

    # Check plots exist
    assert os.path.isfile(OUT_DIR / 'plots' / 'result_correlation.png')
    assert os.path.isfile(OUT_DIR / 'plots' / 'result_objective.png')
