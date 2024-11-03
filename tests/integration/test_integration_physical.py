from __future__ import annotations

import filecmp
from pathlib import Path

import pytest
from helpers import PROTEUS_ROOT
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose

from proteus import Proteus
from proteus.utils.coupler import ReadHelpfileFromCSV
from proteus.atmos_clim.common import read_ncdf_profile
from proteus.utils.plot import sample_output

out_dir = PROTEUS_ROOT / 'output' / 'physical'
ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'physical'
config_path = PROTEUS_ROOT /'tests' / 'data' / 'integration' / 'physical' / 'physical.toml'

@pytest.fixture(scope="module")
def physical_run():
    runner = Proteus(config_path=config_path)
    runner.start()

    return runner

def test_physical_run(physical_run):
    hf_all = ReadHelpfileFromCSV(out_dir)
    hf_all_ref = ReadHelpfileFromCSV(ref_dir)

    # Check helpfile
    assert_frame_equal(hf_all, hf_all_ref, rtol=5e-3)

    # Check stellar spectra
    assert filecmp.cmp(out_dir / 'data' / '0.sflux', ref_dir / '0.sflux', shallow=False)

def test_physical_atmosphere(physical_run):
    # Keys to load and test
    fields = ["t", "p", "z", "fl_U_LW", "fl_D_LW", "fl_U_SW", "fl_D_SW"]

    # Load atmosphere output
    atm_out = read_ncdf_profile(out_dir / 'data' / '2002_atm.nc', extra_keys=fields)

    # Compare to config
    assert len(atm_out["t"]) == physical_run.config.atmos_clim.janus.num_levels*2+1

    # Load atmosphere reference
    atm_ref = read_ncdf_profile(ref_dir / '2002_atm.nc', extra_keys=fields)

    # Compare to expected array values.
    # Cannot simply compare the files as black-boxes, because they contain date information
    for key in fields:
        assert_allclose(atm_out[key], atm_ref[key], rtol=5e-3)
