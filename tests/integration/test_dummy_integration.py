from __future__ import annotations

import os
from pathlib import Path

import pytest
from helpers import PROTEUS_ROOT, resize_to_match
from pandas.testing import assert_frame_equal
from matplotlib.testing.compare import compare_images

from proteus import Proteus
from proteus.utils.coupler import ReadHelpfileFromCSV

@pytest.fixture
def test_dummy_run():
    config_path = PROTEUS_ROOT / 'input' / 'dummy.toml'
    ref_output = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy'

    runner = Proteus(config_path=config_path)

    runner.start()

    output = Path(runner.directories['output'])

    hf_all = ReadHelpfileFromCSV(output)
    hf_all_ref = ReadHelpfileFromCSV(ref_output)

    assert_frame_equal(hf_all, hf_all_ref, rtol=1e-3)

def test_plot_dummy_integration(test_dummy_run):

    test_plot = PROTEUS_ROOT / 'output' / 'dummy' / 'plot_global_lin.png'
    ref_plot = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy' / 'plot_global_lin.png'
    tolerance = 4

    # Resize images if needed
    test, ref = resize_to_match(test_plot, ref_plot)

    # Save resized images to temporary paths
    test.save('test.png')
    ref.save('ref.png')

    # Use compare_images to compare the resized files
    result = compare_images('test.png', 'ref.png', tol=tolerance)

    # compare_images returns None if the images are the same within the tolerance
    assert result is None, f"The two PNG files differ more than the allowed tolerance: {result}"
