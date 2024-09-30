from __future__ import annotations

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
        )

@pytest.fixture(scope="module")
def dummy_run():
    config_path = PROTEUS_ROOT / 'input' / 'dummy.toml'

    runner = Proteus(config_path=config_path)

    runner.start()

def test_dummy_run(dummy_run):
    out_dir = PROTEUS_ROOT / 'output' / 'dummy'
    ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy'

    hf_all = ReadHelpfileFromCSV(out_dir)
    hf_all_ref = ReadHelpfileFromCSV(ref_dir)

    assert_frame_equal(hf_all, hf_all_ref, rtol=5e-3)

@pytest.mark.xfail
@pytest.mark.parametrize("image", IMAGE_LIST)
def test_plot_dummy_integration(dummy_run, image):

    out_dir = PROTEUS_ROOT / 'output' / 'dummy'
    ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy'

    out_img = out_dir / image
    ref_img = ref_dir / image
    tolerance = 3

    # Resize images if needed
    out_tmp, ref_tmp = resize_to_match(out_img, ref_img)

    # Save resized images to temporary paths
    out_tmp.save('test.png')
    ref_tmp.save('ref.png')

    # Use compare_images to compare the resized files
    result = compare_images('test.png', 'ref.png', tol=tolerance)

    assert result is None, f"The two PNG files {image} differ more than the allowed tolerance: {result}"
