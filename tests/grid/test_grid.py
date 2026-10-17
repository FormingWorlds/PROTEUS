# This test runs the PROTEUS grid functionality, for a small grid of simple models
from __future__ import annotations

import filecmp
import os
import tomllib

import numpy as np
import pandas as pd
import pytest
from helpers import PROTEUS_ROOT

from proteus.grid.analyse import analyse_grid
from proteus.grid.manage import grid_from_config
from proteus.grid.pack import pack as gpack
from proteus.grid.summarise import status_category
from proteus.grid.summarise import summarise as gsummarise
from proteus.utils.helper import CommentFromStatus

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


OUT_DIR = PROTEUS_ROOT / 'output' / 'dummy_grid'

GRID_CONFIG = PROTEUS_ROOT / 'tests' / 'grid' / 'dummy.grid.toml'
BASE_CONFIG = PROTEUS_ROOT / 'tests' / 'grid' / 'base.toml'


@pytest.fixture(scope='module')
def grid_run():
    grid_from_config(GRID_CONFIG, test_run=False, check_interval=1)


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


def test_grid_summarise(grid_run):
    """``proteus grid summarise`` produces a non-empty summary in three
    modes: default, ``completed`` filter, and ``status=11`` filter.
    """
    assert gsummarise(OUT_DIR)
    assert gsummarise(OUT_DIR, 'completed')
    assert gsummarise(OUT_DIR, 'status=11')


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


def test_grid_post_process(grid_run):
    """``proteus grid-analyse`` on the completed dummy grid writes both plots and
    three CSVs whose rows match the case folders: every swept combination once,
    the status and category of each case's status file, the last row of its
    helpfile, and a solidification time recomputed here from its helpfile and
    init_coupler.toml. Runs in the slow tier because it depends on the grid run.
    """
    analyse_grid(GRID_CONFIG)

    extracted = OUT_DIR / 'post_processing' / 'extracted_data'
    plots = OUT_DIR / 'post_processing' / 'grid_plots'
    assert os.path.isfile(plots / 'summary_grid_statuses.png')
    assert os.path.isfile(plots / 'ecdf_grid_plot.png')
    read = {
        mode: pd.read_csv(extracted / f'final_extracted_data_{mode}.csv', sep='\t')
        for mode in ('all', 'completed', 'running_error')
    }
    summary = read['all'].set_index('case_number')

    # One row per case of the 1 x 2 x 2 grid, each swept combination exactly once.
    assert sorted(summary.index) == [0, 1, 2, 3]
    assert summary['planet.mass_tot'].tolist() == pytest.approx([0.7] * 4)
    combos = sorted(
        zip(summary['planet.elements.H_budget'], summary['interior_struct.core_frac'])
    )
    np.testing.assert_allclose(combos, [[1000, 0.35], [1000, 0.95], [2000, 0.35], [2000, 0.95]])

    for index, row in summary.iterrows():
        case_dir = OUT_DIR / f'case_{index:06d}'
        code = int((case_dir / 'status').read_text().split()[0])
        assert row['status'] == CommentFromStatus(code)
        assert row['status_category'] == status_category(code)
        assert row['config_source'] == 'init_coupler.toml'

        helpfile = pd.read_csv(case_dir / 'runtime_helpfile.csv', sep='\t')
        for column in ('Time', 'T_surf', 'P_surf', 'Phi_global'):
            assert row[column] == pytest.approx(helpfile[column].iloc[-1], rel=1e-12)
        assert row['T_surf'] > 0.0
        assert 0.0 <= row['Phi_global'] <= 1.0

        with open(case_dir / 'init_coupler.toml', 'rb') as f:
            solid = tomllib.load(f)['params']['stop']['solid']
        crossed = helpfile['Phi_global'] <= solid['phi_crit']
        if solid['enabled'] and crossed.any():
            assert row['solidification_time'] == pytest.approx(
                helpfile.loc[crossed.idxmax(), 'Time'], rel=1e-12
            )
        else:
            assert np.isnan(row['solidification_time'])

    # completed and running_error hold exactly the Completed and the other cases.
    is_completed = summary['status_category'] == 'Completed'
    assert set(read['completed']['case_number']) == set(summary.index[is_completed])
    assert set(read['running_error']['case_number']) == set(summary.index[~is_completed])
