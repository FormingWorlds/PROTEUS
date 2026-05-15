# This test runs the PROTEUS grid functionality, for a small grid of simple models
from __future__ import annotations

import filecmp
import os

import pytest
from helpers import PROTEUS_ROOT

from proteus.grid.manage import grid_from_config
from proteus.grid.pack import pack as gpack
from proteus.grid.summarise import summarise as gsummarise

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


OUT_DIR = PROTEUS_ROOT / 'output' / 'dummy_grid'

GRID_CONFIG = PROTEUS_ROOT / 'tests' / 'grid' / 'dummy.grid.toml'
BASE_CONFIG = PROTEUS_ROOT / 'tests' / 'grid' / 'base.toml'


@pytest.fixture(scope='module')
def grid_run():
    grid_from_config(GRID_CONFIG, test_run=False, check_interval=1)


@pytest.mark.integration
def test_grid_run(grid_run):
    """A small dummy-backend grid completes without raising. The fixture
    runs the whole grid; this test pins that the fixture succeeded by
    asserting the per-grid output directory exists.
    """
    # Discriminating post-state: the grid manager creates OUT_DIR as part of
    # its normal completion path. A fixture that raised partway would leave
    # this assertion to fire instead of swallowing the failure silently.
    assert OUT_DIR.exists()
    assert OUT_DIR.is_dir()


@pytest.mark.integration
def test_grid_config(grid_run):
    """The grid run copies its grid TOML and the base config TOML into the
    output directory bit-identically, and writes a per-case config under
    ``cfgs/case_<id>.toml``. Reproducibility hook for the grid manager.
    """
    assert os.path.isfile(OUT_DIR / 'copy.grid.toml')

    # Copy of base config is identical to base config
    assert filecmp.cmp(OUT_DIR / 'ref_config.toml', BASE_CONFIG, shallow=False)

    # Check that case config files have been written
    assert os.path.isfile(OUT_DIR / 'cfgs' / f'case_{0:06d}.toml')


@pytest.mark.integration
def test_grid_log(grid_run):
    """``manager.log`` records the flattened parameter grid (values list),
    the case-completion summary line, and the explored sweep values
    (1000, 2000 in the test grid).
    """
    with open(OUT_DIR / 'manager.log', 'r') as hdl:
        lines = hdl.read()
    assert 'Flattened grid points' in lines
    assert 'values   : [1000.0, 2000.0]' in lines
    assert 'All cases have exited' in lines


@pytest.mark.integration
def test_grid_summarise(grid_run):
    """``proteus grid summarise`` produces a non-empty summary in three
    modes: default, ``completed`` filter, and ``status=11`` filter.
    """
    assert gsummarise(OUT_DIR)
    assert gsummarise(OUT_DIR, 'completed')
    assert gsummarise(OUT_DIR, 'status=11')


@pytest.mark.integration
def test_grid_pack(grid_run):
    """``proteus grid pack`` produces a packed directory and zip archive
    containing manager log, per-case helpfiles, and plots. Verifies the
    pack tree structure that downstream users rely on.
    """
    assert gpack(OUT_DIR, plots=True, zip=True, rmdir_pack=False)

    # check pack folder exists
    assert os.path.isdir(OUT_DIR / 'pack')

    # check manager.log exists in pack folder
    assert os.path.isfile(OUT_DIR / 'pack' / 'manager.log')

    # check helpfile exists for case 0
    assert os.path.isfile(OUT_DIR / 'pack' / f'case_{0:06d}' / 'runtime_helpfile.csv')

    # check zip exists
    assert os.path.isfile(OUT_DIR / 'pack.zip')
