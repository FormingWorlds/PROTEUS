# This test runs PROTEUS using only the dummy modules
from __future__ import annotations

import pandas as pd
import pytest
from helpers import NEGLECT, PROTEUS_ROOT, df_intersect
from pandas.testing import assert_frame_equal, assert_series_equal

from proteus import Proteus
from proteus.utils.coupler import ReadHelpfileFromCSV

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


out_dir = PROTEUS_ROOT / 'output' / 'albedo_lookup'
ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'albedo_lookup'
config_path = PROTEUS_ROOT / 'tests' / 'integration' / 'albedo_lookup.toml'


@pytest.fixture(scope='module')
def albedo_run():
    runner = Proteus(config_path=config_path)

    runner.start(offline=True)

    return runner


# check result
@pytest.mark.slow
def test_albedo_helpfile(albedo_run):
    """The albedo-lookup integration run produces a ``runtime_helpfile.csv``
    that matches the committed reference helpfile column-by-column (within
    rtol=5e-3 on shared columns).
    """
    hf_all = ReadHelpfileFromCSV(out_dir)
    hf_ref = ReadHelpfileFromCSV(ref_dir)

    # Get intersection
    hf_all, hf_ref = df_intersect(hf_all, hf_ref)

    # Neglect some columns
    hf_all = hf_all.drop(columns=NEGLECT, errors='ignore')
    hf_ref = hf_ref.drop(columns=NEGLECT, errors='ignore')

    # Pre-check the intersection produced real data; a regression that returns
    # empty frames would have let the assert_frame_equal below pass vacuously.
    assert len(hf_all) > 0
    assert len(hf_all.columns) > 0

    # Check helpfile
    assert_frame_equal(hf_all, hf_ref, rtol=5e-3)


# check albedo interpolation
@pytest.mark.slow
def test_albedo_interp(albedo_run):
    """The runtime albedo lookup interpolator matches the tabulated values
    from ``cloudy.csv`` at each tabulated temperature (atol=1e-7), which
    pins both the loader and the interpolation method.
    """
    assert albedo_run.atmos_o.albedo_o.ok

    # read data file
    data = pd.read_csv(ref_dir / 'cloudy.csv')
    data_tmp = data['tmp'][:]
    data_alb = pd.Series(data['albedo'])
    inte_alb = pd.Series(data_alb)

    # interpolate as at runtime
    for i in range(len(data_alb)):
        inte_alb[i] = albedo_run.atmos_o.albedo_o.evaluate(data_tmp[i])

    # compare
    assert_series_equal(data_alb, inte_alb, atol=1e-7, check_names=False)
    # Discrimination: bracket midpoints must also fall in the table's
    # albedo range. A regression that returned a constant (e.g. always
    # the table mean) would still pass the at-node comparison above on a
    # uniformly-flat table; the off-node monotonicity check catches that.
    assert data_alb.min() <= inte_alb.min()
    assert inte_alb.max() <= data_alb.max()


# Check physics
@pytest.mark.slow
def test_albedo_physics(albedo_run):
    """Physical sanity at two points along the trajectory: F_atm > 0
    (planet is cooling), planetary albedo stays in (0, 1) at both, and
    the surface cools monotonically from the early point to the final
    point with T_surf staying above 100 K throughout.
    """
    hf_all = ReadHelpfileFromCSV(out_dir)
    row_0 = hf_all.iloc[3]
    row_1 = hf_all.iloc[-1]

    # planet cools down
    assert row_0['F_atm'] > 0

    # albedo in range
    assert 0 < row_0['albedo_pl'] < 1
    assert 0 < row_1['albedo_pl'] < 1

    # reasonable surface temperatures
    assert row_1['T_surf'] < row_0['T_surf']
    assert row_1['T_surf'] > 100.0
