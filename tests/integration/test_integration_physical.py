from __future__ import annotations

import filecmp
from pathlib import Path

import pytest
from helpers import PROTEUS_ROOT, resize_to_match
from matplotlib.testing.compare import compare_images
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from proteus import Proteus
from proteus.atmos_clim.common import read_ncdf_profile as read_atmosphere
from proteus.interior.aragog import read_ncdf as read_interior
from proteus.utils.coupler import ReadHelpfileFromCSV

out_dir = PROTEUS_ROOT / 'output' / 'physical'
ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'physical'
config_path = PROTEUS_ROOT /'tests' / 'integration' / 'physical.toml'

IMAGE_LIST = (
        "plot_atmosphere.png",
        "plot_escape.png",
        "plot_global_log.png",
        "plot_observables.png",
        "plot_stacked.png",
        "plot_elements.png",
        "plot_fluxes_atmosphere.png",
        "plot_interior_cmesh.png",
        "plot_sflux_cross.png",
        "plot_emission.png",
        "plot_interior.png",
        "plot_sflux.png",
        )

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

def test_physical_spectrum(physical_run):
    # Check stellar spectrum
    assert filecmp.cmp(out_dir / 'data' / '2002.sflux',
                       ref_dir / '2002.sflux', shallow=False)

def test_physical_atmosphere(physical_run):
    # Keys to load and test
    fields = ["t", "p", "z", "fl_U_LW", "fl_D_SW"]

    # Load atmosphere output
    out = read_atmosphere(out_dir / 'data' / '2002_atm.nc', extra_keys=fields)

    # Compare to config
    assert len(out["t"]) == physical_run.config.atmos_clim.janus.num_levels*2+1

    # Load atmosphere reference
    ref = read_atmosphere(ref_dir / '2002_atm.nc', extra_keys=fields)

    # Compare to expected array values.
    # Cannot simply compare the files as black-boxes, because they contain date information
    for key in fields:
        assert_allclose(out[key], ref[key], rtol=1e-3)

def test_physical_interior(physical_run):
    # Keys to load and test
    fields = ["radius_b", "pres_b", "temp_b", "phi_b"]

    # Load interior output
    out = read_interior(out_dir / 'data' / '2002_int.nc')

    # Compare to config
    assert len(out["radius_b"]) == physical_run.config.interior.aragog.num_levels

    # Load interior reference
    ref = read_interior(ref_dir / '2002_int.nc')

    # Compare to expected array values.
    # Cannot simply compare the files as black-boxes, because they contain date information
    for key in fields:
        assert_allclose(out[key], ref[key], rtol=1e-3)


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("image", IMAGE_LIST)
def test_physical_plot(physical_run, image):

    out_img = out_dir / image
    ref_img = ref_dir / image
    tolerance = 3

    # Resize images if needed
    out_img, ref_img = resize_to_match(out_img, ref_img)

    results_dir = Path('result_images')
    results_dir.mkdir(exist_ok=True, parents=True)

    actual = results_dir / image
    expected = results_dir / f'{actual.stem}-expected{actual.suffix}'

    # Save resized images to temporary paths
    out_img.save(actual)
    ref_img.save(expected)

    # Use compare_images to compare the resized files
    result = compare_images(actual=str(actual), expected=str(expected), tol=tolerance)

    assert result is None, f"The two PNG files {image} differ more than the allowed tolerance: {result}"