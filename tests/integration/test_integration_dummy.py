from __future__ import annotations

import filecmp

import pytest
from helpers import PROTEUS_ROOT
from pandas.testing import assert_frame_equal

from proteus import Proteus
from proteus.utils.coupler import ReadHelpfileFromCSV


@pytest.fixture(scope="module")
def dummy_run():
    config_path = PROTEUS_ROOT /'tests' / 'integration' / 'dummy.toml'

    runner = Proteus(config_path=config_path)

    runner.start()

def test_dummy_run(dummy_run):
    out_dir = PROTEUS_ROOT / 'output' / 'dummy'
    ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy'

    hf_all = ReadHelpfileFromCSV(out_dir)
    hf_ref = ReadHelpfileFromCSV(ref_dir)

    # Neglect these columns
    neglect = ["CH4_mol_atm", "CH4_mol_solid", "CH4_mol_liquid", "CH4_mol_total"]
    hf_all = hf_all.drop(columns=neglect)
    hf_ref = hf_ref.drop(columns=neglect)

    # Check helpfile
    assert_frame_equal(hf_all, hf_ref, rtol=5e-3)

    # Check statusfiles
    assert filecmp.cmp(out_dir / 'status', ref_dir / 'status', shallow=False)

    # Check stellar spectra
    assert filecmp.cmp(out_dir / 'data' / '0.sflux', ref_dir / '0.sflux', shallow=False)
