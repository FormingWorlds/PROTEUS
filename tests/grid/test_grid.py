# This test runs the PROTEUS grid functionality, for a small grid of simple models
from __future__ import annotations

import filecmp
import os

import pytest
from helpers import PROTEUS_ROOT

from proteus.grid.manage import grid_from_config
from proteus.grid.summarise import summarise as gsummarise

OUT_DIR = PROTEUS_ROOT / 'output' / 'dummy_grid'

GRID_CONFIG = PROTEUS_ROOT / 'tests' / 'grid' / 'dummy.grid.toml'
BASE_CONFIG = PROTEUS_ROOT / 'tests' / 'grid' / 'base.toml'

@pytest.fixture(scope="module")
def grid_run():
    grid_from_config(GRID_CONFIG, test_run=True)

def test_grid_run(grid_run):
    # Call fixture to ensure that it has run without error
    pass

def test_grid_config(grid_run):
    # Copy of grid's config exists in output dir
    assert os.path.isfile(OUT_DIR / 'copy.grid.toml')

    # Copy of base config is identical to base config
    assert filecmp.cmp(OUT_DIR / 'ref_config.toml', BASE_CONFIG, shallow=False)

    # Check that case config files have been written
    assert os.path.isfile(OUT_DIR / 'cfgs' / 'case_000000.toml')

def test_grid_log(grid_run):
    # Read logfile and check for expected statements
    with open(OUT_DIR / 'manager.log', 'r') as hdl:
        lines = hdl.read()
    assert 'Flattened grid points' in lines
    assert 'values   : [1000.0, 2000.0]' in lines
    assert 'All cases have exited' in lines

def test_grid_summarise(grid_run):
    # Test running grid-summarise command
    gsummarise(OUT_DIR)
