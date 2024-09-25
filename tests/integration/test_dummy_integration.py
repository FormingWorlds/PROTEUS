from __future__ import annotations

import os
from pathlib import Path

import pytest
from helpers import PROTEUS_ROOT, resize_to_match
from pandas.testing import assert_frame_equal
from matplotlib.testing.compare import compare_images

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

@pytest.fixture
def test_dummy_run():
    config_path = PROTEUS_ROOT / 'input' / 'dummy.toml'
    ref_output = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy'

    runner = Proteus(config_path=config_path)

    runner.start()

    output = Path(runner.directories['output'])

    hf_all = ReadHelpfileFromCSV(output)
    hf_all_ref = ReadHelpfileFromCSV(ref_output)

    assert_frame_equal(hf_all, hf_all_ref, rtol=1e-2)

def test_plot_dummy_integration(test_dummy_run):

    out_dir = PROTEUS_ROOT / 'output' / 'dummy'
    ref_dir = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy'

    for image in IMAGE_LIST:
        out_img = out_dir / image
        ref_img = ref_dir / image
        tolerance = 14

        # Resize images if needed
        out_tmp, ref_tmp = resize_to_match(out_img, ref_img)

        # Save resized images to temporary paths
        out_tmp.save('test.png')
        ref_tmp.save('ref.png')

        # Use compare_images to compare the resized files
        result = compare_images('test.png', 'ref.png', tol=tolerance)

        assert result is None, f"The two PNG files {image} differ more than the allowed tolerance: {result}"
