"""Fixtures for the tests of the CHILI comparison plot script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import pytest

from ._plot_chili_helpers import DARK_BY_HAND


@pytest.fixture(scope='module')
def tool():
    """The co-located ``plot_chili_comparison.py``, loaded as a module."""
    script_path = Path(__file__).resolve().parents[2] / 'tools' / 'plot_chili_comparison.py'
    spec = importlib.util.spec_from_file_location('plot_chili_under_test', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def close_figures(tool):
    """Each test starts from the default style and leaves no open figure."""
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        yield
    tool.plt.close('all')


@pytest.fixture(params=['dark_by_hand', 'light', 'dark'])
def style(request):
    """A style active for the test: dark colours set by hand, or a proteus-mpl style."""
    if request.param == 'dark_by_hand':
        matplotlib.rcParams.update(DARK_BY_HAND)
    else:
        pytest.importorskip('proteus_mpl').use(request.param)
    return request.param
