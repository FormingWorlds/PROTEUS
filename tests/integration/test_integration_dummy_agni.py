# This test runs PROTEUS with dummy interior and AGNI atmosphere, then chemistry with VULCAN
from __future__ import annotations

import os

import numpy as np
import pytest
from helpers import NEGLECT, PROTEUS_ROOT, df_intersect
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from proteus import Proteus
from proteus.atmos_chem.common import read_result
from proteus.atmos_clim.common import read_ncdf_profile as read_atmosphere
from proteus.plot.cpl_atmosphere_cbar import plot_atmosphere_cbar_entry
from proteus.plot.cpl_chem_atmosphere import plot_chem_atmosphere
from proteus.utils.coupler import ReadHelpfileFromCSV

out_dir = PROTEUS_ROOT / 'output' / 'dummy_agni'
ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy_agni'
config_path = PROTEUS_ROOT /'tests' / 'integration' / 'dummy_agni.toml'


@pytest.fixture(scope="module")
def dummy_agni_run():

    # Run simulation
    runner = Proteus(config_path=config_path)
    runner.start()

    # Make extra plots
    plot_atmosphere_cbar_entry(runner)

    return runner

def test_dummy_agni_run(dummy_agni):
    '''
    Test that the AGNI run completes without error and produces correct output
    '''

    hf_all = ReadHelpfileFromCSV(out_dir)
    hf_ref = ReadHelpfileFromCSV(ref_dir)

    # Get intersection
    hf_all, hf_ref = df_intersect(hf_all, hf_ref)

    # Neglect some columns
    hf_all = hf_all.drop(columns=NEGLECT, errors='ignore')
    hf_ref = hf_ref.drop(columns=NEGLECT, errors='ignore')

    # Check helpfile
    assert_frame_equal(hf_all, hf_ref, rtol=5e-3)

@pytest.mark.dependency()
def test_dummy_agni_archive(dummy_agni):
    '''
    Test that the archive files are able to be extracted and created again

    The archive files were automatically created during the run.
    '''

    data_dir = dummy_agni.directories["output/data"]

    # Check that the archive file exists
    assert os.path.isfile(os.path.join(data_dir, "data.tar"))

    # Check that the already-archived files do not exist loosely in the data directory
    assert not os.path.isfile(os.path.join(data_dir, "0_atm.nc"))

    # Extract archives
    dummy_agni.extract_archives()

    # Check that the extracted files now exist
    assert os.path.isfile(os.path.join(data_dir, "0_atm.nc"))

@pytest.mark.dependency(depends=["test_dummy_agni_archive"])
def test_dummy_agni_atmosphere(dummy_agni):
    '''
    Test that the modelled profiles match the reference data
    '''

    # Keys to load and test
    _out   = out_dir / 'data' / '99002_atm.nc'
    _ref   = ref_dir / '99002_atm.nc'
    fields = ["tmpl", "pl", "rl", "fl_U_LW", "fl_D_SW", "fl_cnvct", "Kzz"]

    # Load atmosphere output
    out = read_atmosphere(_out, extra_keys=fields)

    # Compare to config
    assert len(out["tmpl"]) == dummy_agni.config.atmos_clim.agni.num_levels+1
    assert np.all(out["Kzz"] >= 0)
    assert np.all(out["rl"][:-1] - out["rl"][1:] > 0)

    # Load atmosphere reference
    ref = read_atmosphere(_ref, extra_keys=fields)

    # Compare to expected array values.
    for key in fields:
        assert_allclose(out[key], ref[key], rtol=1e-3,
                        err_msg=f"Key {key} does not match reference data")

@pytest.mark.dependency(depends=["test_dummy_agni_archive"])
def test_dummy_agni_offchem(dummy_agni):
    '''
    Test that the offline chemistry is working and matches reference data
    '''

    # run offline chemistry and load result
    df_out = dummy_agni.offline_chemistry()

    # Print result, captured if assertions fail
    print(df_out)

    # validate output
    assert df_out is not None
    assert "Kzz" in df_out.columns
    assert "H2O" in df_out.columns
    assert np.all(df_out["H2O"].values >= 0)

    # load reference data
    module = dummy_agni.config.atmos_chem.module
    df_ref = read_result(out_dir, module)

    # validate dataframes
    assert_frame_equal(df_out, df_ref, rtol=5e-3)

    # make plot
    plot_chem_atmosphere(out_dir, module, plot_format=dummy_agni.config.params.out.plot_fmt)
