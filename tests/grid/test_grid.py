# This test runs the PROTEUS grid functionality, for a small grid of simple models
from __future__ import annotations

import filecmp
import os

import pytest
from helpers import PROTEUS_ROOT

from proteus.grid.manage import grid_from_config
from proteus.grid.pack import pack as gpack
from proteus.grid.summarise import summarise as gsummarise

OUT_DIR = PROTEUS_ROOT / 'output' / 'dummy_grid'

GRID_CONFIG = PROTEUS_ROOT / 'tests' / 'grid' / 'dummy.grid.toml'
BASE_CONFIG = PROTEUS_ROOT / 'tests' / 'grid' / 'base.toml'

@pytest.fixture(scope="module")
def grid_run():
    grid_from_config(GRID_CONFIG, test_run=False, check_interval=1)

def test_grid_run(grid_run):
    # Call fixture to ensure that it has run without error
    pass

def test_grid_config(grid_run):
    # Copy of grid's config exists in output dir
    assert os.path.isfile(OUT_DIR / 'copy.grid.toml')

    # Copy of base config is identical to base config
    assert filecmp.cmp(OUT_DIR / 'ref_config.toml', BASE_CONFIG, shallow=False)

    # Check that case config files have been written
    assert os.path.isfile(OUT_DIR / 'cfgs' / f'case_{0:06d}.toml')

def test_grid_log(grid_run):
    # Read logfile and check for expected statements
    with open(OUT_DIR / 'manager.log', 'r') as hdl:
        lines = hdl.read()
    assert 'Flattened grid points' in lines
    assert 'values   : [1000.0, 2000.0]' in lines
    assert 'All cases have exited' in lines

def test_grid_summarise(grid_run):
    # Test running grid-summarise command
    assert gsummarise(OUT_DIR)
    assert gsummarise(OUT_DIR, "completed")
    assert gsummarise(OUT_DIR, "status=11")

def test_grid_pack(grid_run):
    # Test running grid-pack command
    assert gpack(OUT_DIR, plots=True, zip=True, rmdir_pack=False)

    # check pack folder exists
    assert os.path.isdir(OUT_DIR / 'pack')

    # check manager.log exists in pack folder
    assert os.path.isfile(OUT_DIR / 'pack' / 'manager.log')

    # check helpfile exists for case 0
    assert os.path.isfile(OUT_DIR / 'pack' / f'case_{0:06d}' / 'runtime_helpfile.csv')

    # check zip exists
    assert os.path.isfile(OUT_DIR / 'pack.zip')
