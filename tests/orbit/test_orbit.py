"""Unit tests for the dummy orbit module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from proteus.interior.common import Interior_t
from proteus.orbit.dummy import run_dummy_orbit


def _make_config(phi_tide: str, h_tide: float, imk2: float) -> Any:
    dummy = SimpleNamespace(Phi_tide=phi_tide, H_tide=h_tide, Imk2=imk2)
    orbit = SimpleNamespace(dummy=dummy)
    return cast(Any, SimpleNamespace(orbit=orbit))


@pytest.mark.unit
def test_less_than_threshold_heating_scales_with_melt_fraction():
    """Heating applies where phi < threshold and scales linearly with melt fraction."""
    config = _make_config('<0.5', h_tide=10.0, imk2=1e-2)
    interior = Interior_t(nlev_b=4)
    interior.phi = np.array([0.1, 0.4, 0.6])

    result = run_dummy_orbit(config, interior)

    assert result == pytest.approx(1e-2)
    assert interior.tides == pytest.approx(
        [10.0 * (1 - 0.1 / 0.5), 10.0 * (1 - 0.4 / 0.5), 0.0]
    )


@pytest.mark.unit
def test_greater_than_threshold_heating_scales_linearly():
    """Heating applies where phi > threshold and increases toward fully liquid."""
    config = _make_config('>0.25', h_tide=5.0, imk2=2e-3)
    interior = Interior_t(nlev_b=5)
    interior.phi = np.array([0.1, 0.25, 0.75, 1.0])

    result = run_dummy_orbit(config, interior)

    assert result == pytest.approx(2e-3)
    expected = [0.0, 0.0, 5.0 * (0.75 - 0.25) / (1 - 0.25), 5.0 * (1.0 - 0.25) / (1 - 0.25)]
    assert interior.tides == pytest.approx(expected)


@pytest.mark.unit
def test_equal_to_threshold_produces_zero_heating():
    """Boundary equality does not trigger heating (strict inequality)."""
    config = _make_config('<0.3', h_tide=7.5, imk2=0.0)
    interior = Interior_t(nlev_b=3)
    interior.phi = np.array([0.3, 0.3])

    run_dummy_orbit(config, interior)

    assert interior.tides == pytest.approx([0.0, 0.0])


@pytest.mark.unit
def test_no_cells_meet_condition_keeps_zero_heating():
    """If no layer meets inequality, tides remain zero everywhere."""
    config = _make_config('<0.1', h_tide=9.0, imk2=0.5)
    interior = Interior_t(nlev_b=3)
    interior.phi = np.array([0.5, 0.9])

    run_dummy_orbit(config, interior)

    assert interior.tides == pytest.approx([0.0, 0.0])


@pytest.mark.unit
def test_phi_array_unchanged_by_heating_calculation():
    """run_dummy_orbit updates tides but leaves melt fractions untouched."""
    config = _make_config('>0.2', h_tide=4.0, imk2=1.0)
    interior = Interior_t(nlev_b=4)
    interior.phi = np.array([0.1, 0.2, 0.3])
    phi_before = interior.phi.copy()

    run_dummy_orbit(config, interior)

    assert interior.phi == pytest.approx(phi_before)


@pytest.mark.unit
def test_returns_imk2_from_config():
    """Function returns Imk2 value provided in config without modification."""
    config = _make_config('<0.9', h_tide=1.0, imk2=3.21)
    interior = Interior_t(nlev_b=2)
    interior.phi = np.array([0.5])

    result = run_dummy_orbit(config, interior)

    assert result == pytest.approx(3.21)


@pytest.mark.unit
def test_single_level_interpolates_correctly():
    """Single-layer interior still applies heating logic and preserves array shapes."""
    config = _make_config('>0.5', h_tide=2.0, imk2=0.7)
    interior = Interior_t(nlev_b=2)
    interior.phi = np.array([0.8])

    run_dummy_orbit(config, interior)

    assert interior.tides.shape == (1,)
    assert interior.tides == pytest.approx([2.0 * (0.8 - 0.5) / (1 - 0.5)])
