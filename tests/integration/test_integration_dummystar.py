from __future__ import annotations

import filecmp

import pytest
from helpers import PROTEUS_ROOT
from pandas.testing import assert_frame_equal

from proteus import Proteus
from proteus.utils.coupler import ReadHelpfileFromCSV


@pytest.fixture(scope="module")
def dummy_run():
    config_path = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummystar' / 'cfg.toml'

    runner = Proteus(config_path=config_path)

    runner.start()

def test_dummy_run(dummy_run):
    out_dir = PROTEUS_ROOT / 'output' / 'dummystar'
    ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummystar'

    # Check helpfiles
    hf_all = ReadHelpfileFromCSV(out_dir)
    hf_all_ref = ReadHelpfileFromCSV(ref_dir)
    assert_frame_equal(hf_all, hf_all_ref, rtol=5e-3)

    # Check statusfiles
    assert filecmp.cmp(out_dir / 'status', ref_dir / 'status', shallow=False)

    # Check stellar spectra
    assert filecmp.cmp(out_dir / 'data' / '9002.sflux',
                       ref_dir / '9002.sflux', shallow=False)
