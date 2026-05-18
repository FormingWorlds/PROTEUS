"""Unit tests for ``proteus.utils.visual`` helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from proteus.utils.visual import cmf, interp_spec


@pytest.mark.unit
def test_interp_spec_single_point_returns_constant():
    # handles case when using greygas RT, for example
    wl = np.array([500.0])
    fl = np.array([42.0])
    out = interp_spec(wl, fl)
    assert np.all(out == 42.0)
    assert out.shape == cmf[:, 0].shape


@pytest.mark.unit
def test_interp_spec_multi_point_interpolates_to_cmf_grid():
    # dummy wl and fl arrays
    wl = np.array([380.0, 500.0, 780.0])
    fl = np.array([1.0, 2.0, 4.0])

    # interpolate onto wavelength grid used for colormatching functions
    out = interp_spec(wl, fl)
    assert out.shape == cmf[:, 0].shape

    # check CMF mapping works ok
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(1.0)
    assert out[-1] == pytest.approx(4.0)
