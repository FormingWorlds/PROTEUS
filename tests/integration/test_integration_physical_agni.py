from __future__ import annotations

import numpy as np
import pytest
from helpers import NEGLECT, PROTEUS_ROOT, df_intersect
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from proteus import Proteus
from proteus.atmos_clim.common import read_ncdf_profile as read_atmosphere
from proteus.plot.cpl_atmosphere_cbar import plot_atmosphere_cbar_entry
from proteus.utils.coupler import ReadHelpfileFromCSV

out_dir = PROTEUS_ROOT / 'output' / 'physical_agni'
ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'physical_agni'
config_path = PROTEUS_ROOT /'tests' / 'integration' / 'physical_agni.toml'


@pytest.fixture(scope="module")
def agni_run():

    # Run simulation
    runner = Proteus(config_path=config_path)
    runner.start()

    # Make extra plots
    plot_atmosphere_cbar_entry(runner)

    return runner

def test_agni_run(agni_run):
    hf_all = ReadHelpfileFromCSV(out_dir)
    hf_ref = ReadHelpfileFromCSV(ref_dir)

    # Get intersection
    hf_all, hf_ref = df_intersect(hf_all, hf_ref)

    # Neglect some columns
    hf_all = hf_all.drop(columns=NEGLECT, errors='ignore')
    hf_ref = hf_ref.drop(columns=NEGLECT, errors='ignore')

    # Check helpfile
    assert_frame_equal(hf_all, hf_ref, rtol=5e-3)

def test_agni_atmosphere(agni_run):
    # Keys to load and test
    _out   = out_dir / 'data' / '99002_atm.nc'
    _ref   = ref_dir / '99002_atm.nc'
    fields = ["tmpl", "pl", "rl", "fl_U_LW", "fl_D_SW", "fl_cnvct", "Kzz"]

    # Load atmosphere output
    out = read_atmosphere(_out, extra_keys=fields)

    # Compare to config
    assert len(out["tmpl"]) == agni_run.config.atmos_clim.agni.num_levels+1
    assert np.all(out["Kzz"] >= 0)
    assert np.all(out["rl"][:-1] - out["rl"][1:] > 0)

    # Load atmosphere reference
    ref = read_atmosphere(_ref, extra_keys=fields)

    # Compare to expected array values.
    for key in fields:
        assert_allclose(out[key], ref[key], rtol=1e-3,
                        err_msg=f"Key {key} does not match reference data")


def test_agni_offchem(agni_run):

    # run offline chemistry and load result
    df = agni_run.offline_chemistry()

    # Print result, captured if assertions fail
    print(df)

    # validate output
    assert df is not None
    assert "Kzz" in df.columns
    assert "H2O" in df.columns
    assert np.all(df["H2O"].values >= 0)
