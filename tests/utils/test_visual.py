"""Unit tests for ``proteus.utils.visual`` helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from proteus.utils.visual import cmf, interp_spec

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_interp_spec_single_point_returns_constant():
    """A single-point input spectrum interpolates to a constant value on
    the CMF wavelength grid. This is the greygas RT path where the input
    is effectively flat.
    """
    wl = np.array([500.0])
    fl = np.array([42.0])
    out = interp_spec(wl, fl)
    assert np.all(out == 42.0)
    assert out.shape == cmf[:, 0].shape


@pytest.mark.unit
def test_interp_spec_multi_point_interpolates_to_cmf_grid():
    """A multi-point spectrum is interpolated onto the CMF wavelength
    grid. The output is finite at every CMF wavelength and matches the
    input endpoints at the grid edges (380 nm and 780 nm here).
    """
    wl = np.array([380.0, 500.0, 780.0])
    fl = np.array([1.0, 2.0, 4.0])

    # interpolate onto wavelength grid used for colormatching functions
    out = interp_spec(wl, fl)
    assert out.shape == cmf[:, 0].shape

    # check CMF mapping works ok
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(1.0)
    assert out[-1] == pytest.approx(4.0)
