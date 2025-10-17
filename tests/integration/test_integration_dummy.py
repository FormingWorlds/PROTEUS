# This test runs PROTEUS using only the dummy modules
from __future__ import annotations

import filecmp

import pytest
from helpers import NEGLECT, PROTEUS_ROOT, df_intersect
from pandas.testing import assert_frame_equal

from proteus import Proteus
from proteus.utils.coupler import ReadHelpfileFromCSV

out_dir = PROTEUS_ROOT / 'output' / 'dummy'
ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy'


@pytest.fixture(scope="module")
def dummy_run():
    config_path = PROTEUS_ROOT /'tests' / 'integration' / 'dummy.toml'

    runner = Proteus(config_path=config_path)

    runner.start(offline=True)

# run the integration
def test_dummy_run(dummy_run):
    assert filecmp.cmp(out_dir / 'status', ref_dir / 'status', shallow=False)

# check result
def test_dummy_helpfile(dummy_run):

    hf_all = ReadHelpfileFromCSV(out_dir)
    hf_ref = ReadHelpfileFromCSV(ref_dir)

    # Get intersection
    hf_all, hf_ref = df_intersect(hf_all, hf_ref)

    # Neglect some columns
    hf_all = hf_all.drop(columns=NEGLECT, errors='ignore')
    hf_ref = hf_ref.drop(columns=NEGLECT, errors='ignore')

    # Check helpfile
    assert_frame_equal(hf_all, hf_ref, rtol=5e-3)

# Check stellar spectra
def test_dummy_stellar(dummy_run):
    assert filecmp.cmp(out_dir / 'data' / '0.sflux', ref_dir / '0.sflux', shallow=False)

# Check physics
def test_dummy_physics(dummy_run):
    hf_all = ReadHelpfileFromCSV(out_dir)
    row_0 = hf_all.iloc[3]
    row_1 = hf_all.iloc[-1]

    # planet cools down
    assert row_0["F_atm"] > 0

    # eccentricity should decrease
    assert row_1["eccentricity"] < row_0["eccentricity"]

    # reasonable surface temperatures
    assert row_1["T_surf"] < row_0["T_surf"]
    assert row_1["T_surf"] > 100.0
