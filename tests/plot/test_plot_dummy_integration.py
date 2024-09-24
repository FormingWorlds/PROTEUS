from __future__ import annotations

import os
from pathlib import Path

import pytest
from helpers import PROTEUS_ROOT
from matplotlib.testing.compare import compare_images

def test_plot_dummy_integration(test_dummy_run):

    test_plot = PROTEUS_ROOT / 'output' / 'dummy' / 'plot_global_lin.png'
    ref_plot = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy' / 'plot_global_lin.png'
    tolerance = 5

    result = compare_images(test_plot, ref_plot, tol=tolerance)

    # compare_images returns None if the images are the same within the tolerance
    assert result is None, f"The two PNG files differ more than the allowed tolerance: {result}"
