from __future__ import annotations

import filecmp
import os

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

    # Check that case config files have been written
    # assert os.path.isfile(OUT_DIR / 'cfgs' / 'case_000000.toml')
