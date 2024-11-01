from __future__ import annotations

import filecmp
from pathlib import Path

import pytest
from helpers import PROTEUS_ROOT
from pandas.testing import assert_frame_equal

from proteus import Proteus
from proteus.utils.coupler import ReadHelpfileFromCSV
from proteus.atmos_clim.common import read_atmosphere_data
from proteus.interior.wrapper import read_interior_data
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

    # Check statusfiles
    assert filecmp.cmp(out_dir / 'status', ref_dir / 'status', shallow=False)

    # Check stellar spectra
    assert filecmp.cmp(out_dir / 'data' / '0.sflux', ref_dir / '0.sflux', shallow=False)

def test_physical_atmosphere(physical_run):
    # Get output times
    print(physical_run)
    times,_ = sample_output(physical_run, extension="_atm.nc")

    # Check JANUS output
    atm_data = read_atmosphere_data(out_dir,times)

def test_physical_interior(physical_run):
    # Get output times
    print(physical_run)
    times,_ = sample_output(physical_run, extension="_atm.nc")

    # Check Aragog output
    int_data = read_interior_data(out_dir,times)
