# This test runs PROTEUS using only the dummy modules
from __future__ import annotations

import pytest
from helpers import NEGLECT, PROTEUS_ROOT, df_intersect
from pandas.testing import assert_frame_equal

from proteus import Proteus
from proteus.utils.coupler import ReadHelpfileFromCSV

out_dir = PROTEUS_ROOT / 'output' / 'albedo_lookup'
ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'albedo_lookup'


@pytest.fixture(scope="module")
def albedo_run():
    config_path = ref_dir /  'config.toml'

    runner = Proteus(config_path=config_path)

    runner.start(offline=True)

# check result
def test_albedo_helpfile(albedo_run):

    hf_all = ReadHelpfileFromCSV(out_dir)
    hf_ref = ReadHelpfileFromCSV(ref_dir)

    # Get intersection
    hf_all, hf_ref = df_intersect(hf_all, hf_ref)

    # Neglect some columns
    hf_all = hf_all.drop(columns=NEGLECT, errors='ignore')
    hf_ref = hf_ref.drop(columns=NEGLECT, errors='ignore')

    # Check helpfile
    assert_frame_equal(hf_all, hf_ref, rtol=5e-3)

# Check physics
def test_albedo_physics(albedo_run):
    hf_all = ReadHelpfileFromCSV(out_dir)
    row_0 = hf_all.iloc[3]
    row_1 = hf_all.iloc[-1]

    # planet cools down
    assert row_0["F_atm"] > 0

    # albedo in range
    assert 0 < row_0["albedo_pl"] < 1
    assert 0 < row_1["albedo_pl"] < 1

    # reasonable surface temperatures
    assert row_1["T_surf"] < row_0["T_surf"]
    assert row_1["T_surf"] > 100.0
