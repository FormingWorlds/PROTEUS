from __future__ import annotations

import filecmp
from pathlib import Path

import pytest
from helpers import PROTEUS_ROOT, resize_to_match
from matplotlib.testing.compare import compare_images
from pandas.testing import assert_frame_equal

from proteus import Proteus
from proteus.utils.coupler import ReadHelpfileFromCSV

IMAGE_LIST = (
        "plot_elements.png",
        "plot_escape.png",
        "plot_fluxes_global.png",
        "plot_global_lin.png",
        "plot_global_log.png",
        "plot_observables.png",
        "plot_sflux.png",
        "plot_population_mass_radius.png",
        "plot_population_time_density.png",
        )

@pytest.fixture(scope="module")
def dummy_run():
    config_path = PROTEUS_ROOT /'tests' / 'data' / 'integration' / 'dummy' / 'dummy.toml'

    runner = Proteus(config_path=config_path)

    runner.start()

def test_dummy_run(dummy_run):
    out_dir = PROTEUS_ROOT / 'output' / 'dummy'
    ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy'

    hf_all = ReadHelpfileFromCSV(out_dir)
    hf_all_ref = ReadHelpfileFromCSV(ref_dir)

    # Check helpfile
    assert_frame_equal(hf_all, hf_all_ref, rtol=5e-3)

    # Check statusfiles
    assert filecmp.cmp(out_dir / 'status', ref_dir / 'status', shallow=False)

    # Check stellar spectra
    assert filecmp.cmp(out_dir / 'data' / '0.sflux', ref_dir / '0.sflux', shallow=False)


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("image", IMAGE_LIST)
def test_plot_dummy_integration(dummy_run, image):

    out_dir = PROTEUS_ROOT / 'output' / 'dummy'
    ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy'

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
