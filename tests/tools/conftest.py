"""Fixtures for the tests of the CHILI comparison plot script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import pytest


@pytest.fixture(scope='module')
def chili_tool():
    """The co-located ``plot_chili_comparison.py``, loaded as a module."""
    script_path = Path(__file__).resolve().parents[2] / 'tools' / 'plot_chili_comparison.py'
    spec = importlib.util.spec_from_file_location('plot_chili_under_test', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def chili_figures(chili_tool):
    """Each test starts from the default style and leaves no open figure."""
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        yield
    chili_tool.plt.close('all')


@pytest.fixture(params=['light', 'dark'])
def plot_style(request):
    """A proteus-mpl style, active for the test."""
    import proteus_mpl

    proteus_mpl.use(request.param)
    return request.param
