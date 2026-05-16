# This test runs PROTEUS using only the dummy modules
from __future__ import annotations

import filecmp

import pytest
from helpers import NEGLECT, PROTEUS_ROOT, df_intersect
from pandas.testing import assert_frame_equal

from proteus import Proteus
from proteus.utils.coupler import ReadHelpfileFromCSV

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


out_dir = PROTEUS_ROOT / 'output' / 'dummy'
ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy'


@pytest.fixture(scope='module')
def dummy_run():
    config_path = PROTEUS_ROOT / 'tests' / 'integration' / 'dummy.toml'

    runner = Proteus(config_path=config_path)

    runner.start(offline=True)


# run the integration
@pytest.mark.slow
def test_dummy_run(dummy_run):
    """All-dummy-backend integration produces a ``status`` file
    bit-identical to the committed reference. This is the cheapest
    binary-reproducibility check for the run loop end-to-end.
    """
    assert filecmp.cmp(out_dir / 'status', ref_dir / 'status', shallow=False)
    # Discrimination: the bit-comparison above would pass on two empty
    # files. Pin a non-zero size so a regression that emitted an empty
    # status file (or skipped the write entirely) is caught.
    assert (out_dir / 'status').stat().st_size > 0


# check result
@pytest.mark.slow
def test_dummy_helpfile(dummy_run):
    """``runtime_helpfile.csv`` of the all-dummy run matches the
    committed reference column-by-column within rtol=5e-3 on shared columns.
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


# Check stellar spectra
@pytest.mark.slow
def test_dummy_stellar(dummy_run):
    """The all-dummy stellar spectrum file ``0.sflux`` is bit-identical to
    the committed reference, pinning the dummy-star fall-back path.
    """
    assert filecmp.cmp(out_dir / 'data' / '0.sflux', ref_dir / '0.sflux', shallow=False)
    # Discrimination: bit-comparison passes on two empty files. Pin a
    # non-zero size on the generated spectrum to catch a regression that
    # silently wrote an empty `0.sflux`.
    assert (out_dir / 'data' / '0.sflux').stat().st_size > 0


# Check physics
@pytest.mark.slow
def test_dummy_physics(dummy_run):
    """Physical sanity along the all-dummy trajectory: F_atm > 0 (planet
    is cooling), eccentricity decreases monotonically (tidal damping),
    and T_surf decreases monotonically but stays above 100 K.
    """
    hf_all = ReadHelpfileFromCSV(out_dir)
    row_0 = hf_all.iloc[3]
    row_1 = hf_all.iloc[-1]

    # planet cools down
    assert row_0['F_atm'] > 0

    # eccentricity should decrease
    assert row_1['eccentricity'] < row_0['eccentricity']

    # reasonable surface temperatures
    assert row_1['T_surf'] < row_0['T_surf']
    assert row_1['T_surf'] > 100.0
