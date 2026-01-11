"""Unit tests for the dummy star module.

Exercises stellar radius calculation (empirical scaling laws and direct config),
blackbody spectrum generation, bolometric luminosity, and planetary instellation.
Follows PROTEUS testing standards (see docs/test_infrastructure.md).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import proteus.star.dummy as star
from proteus.utils.constants import AU, R_sun, Teff_sun


def _cfg(*, teff: float = 5778.0, radius: float = 1.0, calculate_radius: bool = False) -> Any:
    """Build a minimal config for star tests."""
    dummy = SimpleNamespace(Teff=teff, radius=radius, calculate_radius=calculate_radius)
    return SimpleNamespace(star=SimpleNamespace(dummy=dummy))


@pytest.mark.unit
def test_get_star_radius_from_config_direct():
    """Radius from config is returned when calculate_radius=False."""
    cfg = _cfg(radius=1.5, calculate_radius=False)
    assert star.get_star_radius(cfg) == pytest.approx(1.5)


@pytest.mark.unit
def test_get_star_radius_solar():
    """Solar Teff yields solar radius via scaling law."""
    cfg = _cfg(teff=Teff_sun, calculate_radius=True)
    r = star.get_star_radius(cfg)
    assert r == pytest.approx(1.0, rel=1e-2)


@pytest.mark.unit
def test_get_star_radius_scaling_hotter_star():
    """Hotter star scales to larger radius via mass-radius relation."""
    cfg_sun = _cfg(teff=Teff_sun, calculate_radius=True)
    cfg_hot = _cfg(teff=Teff_sun * 1.5, calculate_radius=True)
    r_sun = star.get_star_radius(cfg_sun)
    r_hot = star.get_star_radius(cfg_hot)
    assert r_hot > r_sun


@pytest.mark.unit
def test_generate_spectrum_shape():
    """Spectrum returns equal-length wavelength and flux arrays."""
    wl, fl = star.generate_spectrum(5778.0, 1.0)
    assert len(wl) == len(fl)
    assert len(wl) > 0


@pytest.mark.unit
def test_generate_spectrum_zero_temp():
    """Zero temperature produces all-zero flux (star is off)."""
    wl, fl = star.generate_spectrum(0.0, 1.0)
    assert all(f == 0.0 for f in fl)
    assert len(wl) == len(fl)


@pytest.mark.unit
def test_generate_spectrum_below_min_temp():
    """Below PLANCK_MIN_TEMPERATURE produces all-zero flux."""
    wl, fl = star.generate_spectrum(0.01, 1.0)
    assert all(f == 0.0 for f in fl)


@pytest.mark.unit
def test_generate_spectrum_increases_with_temp():
    """Hotter star produces higher peak flux."""
    wl_cool, fl_cool = star.generate_spectrum(4000.0, 1.0)
    wl_hot, fl_hot = star.generate_spectrum(6000.0, 1.0)
    assert max(fl_hot) > max(fl_cool)


@pytest.mark.unit
def test_generate_spectrum_increases_with_radius():
    """Larger radius at fixed Teff produces higher flux (more surface area)."""
    _, fl_small = star.generate_spectrum(5778.0, 1.0)
    _, fl_large = star.generate_spectrum(5778.0, 2.0)
    assert max(fl_large) > max(fl_small)


@pytest.mark.unit
def test_calc_star_luminosity_solar():
    """Solar Teff and radius yields ~1 solar luminosity."""
    l = star.calc_star_luminosity(Teff_sun, 1.0)
    l_sun = 3.828e26  # watts
    assert l == pytest.approx(l_sun, rel=0.01)


@pytest.mark.unit
def test_calc_star_luminosity_zero_temp():
    """Zero temperature yields zero luminosity."""
    l = star.calc_star_luminosity(0.0, 1.0)
    assert l == pytest.approx(0.0)


@pytest.mark.unit
def test_calc_star_luminosity_below_min_temp():
    """Below PLANCK_MIN_TEMPERATURE yields zero luminosity."""
    l = star.calc_star_luminosity(0.01, 1.0)
    assert l == pytest.approx(0.0)


@pytest.mark.unit
def test_calc_star_luminosity_scales_with_temp():
    """Luminosity increases steeply with temperature (T^4)."""
    l1 = star.calc_star_luminosity(5000.0, 1.0)
    l2 = star.calc_star_luminosity(6000.0, 1.0)
    assert l2 > l1
    assert l2 / l1 > 1.5


@pytest.mark.unit
def test_calc_instellation_inverse_square_law():
    """Instellation follows 1/r^2 with separation."""
    sep1 = 1 * AU
    sep2 = 2 * AU
    s1 = star.calc_instellation(Teff_sun, 1.0 * R_sun, sep1)
    s2 = star.calc_instellation(Teff_sun, 1.0 * R_sun, sep2)
    assert s1 / s2 == pytest.approx(4.0, rel=1e-6)


@pytest.mark.unit
def test_calc_instellation_earth_like():
    """Earth-like planet receives ~1361 W/m^2 at 1 AU."""
    s = star.calc_instellation(Teff_sun, 1.0 * R_sun, 1.0 * AU)
    assert s == pytest.approx(1361.0, rel=0.01)
