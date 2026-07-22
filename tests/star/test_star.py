"""Unit tests for the dummy star module.

Exercises stellar radius calculation (empirical scaling laws and direct config),
blackbody spectrum generation, bolometric luminosity, and planetary instellation.
Follows PROTEUS testing standards (see docs/How-to/testing.md).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import proteus.star.dummy as star
from proteus.utils.constants import AU, R_sun, Teff_sun

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _cfg(*, teff: float = 5778.0, radius: float = 1.0, calculate_radius: bool = False) -> Any:
    """Build a minimal config for star tests."""
    dummy = SimpleNamespace(Teff=teff, radius=radius, calculate_radius=calculate_radius)
    return SimpleNamespace(star=SimpleNamespace(dummy=dummy))


@pytest.mark.unit
def test_get_star_radius_from_config_direct():
    """Radius from config is returned when calculate_radius=False."""
    cfg = _cfg(radius=1.5, calculate_radius=False)
    assert star.get_star_radius(cfg) == pytest.approx(1.5)
    # Pass-through invariant: Teff must NOT influence the result when
    # calculate_radius is False. A regression that wired Teff into the
    # direct branch would change r when Teff changes.
    cfg_hot = _cfg(teff=10000.0, radius=1.5, calculate_radius=False)
    assert star.get_star_radius(cfg_hot) == pytest.approx(1.5)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_get_star_radius_solar():
    """Solar Teff yields solar radius via the Demircan 1991 / Eker 2015
    mass-radius / mass-luminosity scaling. At Teff = Teff_sun the
    scaling identity (Teff/Teff_sun)**exponent = 1 must hold for any
    finite exponent."""
    cfg = _cfg(teff=Teff_sun, calculate_radius=True)
    r = star.get_star_radius(cfg)
    assert r == pytest.approx(1.0, rel=1e-2)
    # Positivity invariant: stellar radius is a strictly positive
    # length and must equal exactly 1.0 R_sun at Teff = Teff_sun
    # (the scaling identity 1**x = 1 holds regardless of x).
    assert r == pytest.approx(1.0, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_get_star_radius_scaling_hotter_star():
    """Hotter star scales to larger radius via mass-radius relation.

    With a = 0.945 and b = 4.04 the exponent is 4 / (b/a - 2) ~ 1.74;
    a 1.5x Teff ratio must give a 1.5**1.74 ~ 2.0x radius ratio.
    """
    cfg_sun = _cfg(teff=Teff_sun, calculate_radius=True)
    cfg_hot = _cfg(teff=Teff_sun * 1.5, calculate_radius=True)
    r_sun = star.get_star_radius(cfg_sun)
    r_hot = star.get_star_radius(cfg_hot)
    assert r_hot > r_sun
    # Pinned-value invariant: exponent = 4 / (4.04/0.945 - 2). Mid-K
    # to mid-A regime, the ratio lands at ~2.0. A regression that
    # dropped one of the prefactors would land outside [1.5, 2.5].
    expected = 1.5 ** (4 / (4.04 / 0.945 - 2))
    assert r_hot / r_sun == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_generate_spectrum_shape():
    """Spectrum returns equal-length wavelength and flux arrays."""
    wl, fl = star.generate_spectrum(5778.0, 1.0)
    assert len(wl) == len(fl)
    assert len(wl) > 0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_generate_spectrum_zero_temp():
    """Zero temperature produces all-zero flux (star is off)."""
    wl, fl = star.generate_spectrum(0.0, 1.0)
    np.testing.assert_allclose(fl, 0.0, atol=1e-12)
    assert len(wl) == len(fl)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_generate_spectrum_below_min_temp():
    """Below PLANCK_MIN_TEMPERATURE produces all-zero flux."""
    wl, fl = star.generate_spectrum(0.01, 1.0)
    np.testing.assert_allclose(fl, 0.0, atol=1e-12)
    # Wavelength array is still populated even when the gate cuts off
    # the flux. A regression that returned an empty wavelength grid on
    # the below-min branch would fail this length check.
    assert len(wl) == 600


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_generate_spectrum_increases_with_temp():
    """Hotter star produces higher peak flux. Per Stefan-Boltzmann
    the wavelength-integrated flux scales as T**4, so a 1.5x
    temperature ratio gives a >5x peak-flux ratio. A regression that
    swapped T**4 for T**3 would still pass the monotonicity check but
    yield a ratio < 4."""
    wl_cool, fl_cool = star.generate_spectrum(4000.0, 1.0)
    wl_hot, fl_hot = star.generate_spectrum(6000.0, 1.0)
    assert max(fl_hot) > max(fl_cool)
    # Wien's law: peak flux scales steeply with T. The ratio of peak
    # flux for 6000/4000 is well above 1.5 (the ratio if peak only
    # scaled linearly).
    assert max(fl_hot) / max(fl_cool) > 4.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_generate_spectrum_increases_with_radius():
    """Larger radius at fixed Teff produces higher flux (surface area
    scales as R**2).
    """
    _, fl_small = star.generate_spectrum(5778.0, 1.0)
    _, fl_large = star.generate_spectrum(5778.0, 2.0)
    assert max(fl_large) > max(fl_small)
    # R**2 scaling at fixed Teff: doubling R must multiply peak flux
    # by 4. A regression to linear R scaling would give 2; a swap to
    # R**4 would give 16.
    assert max(fl_large) / max(fl_small) == pytest.approx(4.0, rel=1e-6)


@pytest.mark.unit
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_calc_star_luminosity_solar():
    """Solar Teff and radius yields ~1 solar luminosity. Pin against
    the IAU 2015 nominal solar luminosity L_sun = 3.828e26 W
    (Resolution B3) via the Stefan-Boltzmann law F = sigma T**4."""
    l = star.calc_star_luminosity(Teff_sun, 1.0 * R_sun)
    l_sun = 3.828e26  # watts
    assert l == pytest.approx(l_sun, rel=0.01)
    # Scale guard: 4 pi R_sun**2 sigma T_sun**4 lands at ~3.85e26 W.
    # A R_sun-in-km vs R_sun-in-m unit slip would give 3.85e20 W; a
    # CGS-vs-SI slip on sigma would land 1e3 above or below.
    assert 1e26 < l < 1e27


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_calc_star_luminosity_zero_temp():
    """Zero temperature yields zero luminosity (Stefan-Boltzmann
    sigma T**4 vanishes at T = 0; the source also gates on
    PLANCK_MIN_TEMPERATURE)."""
    l = star.calc_star_luminosity(0.0, 1.0 * R_sun)
    assert l == pytest.approx(0.0)
    # Limit-input invariant: T=0 must zero L regardless of R. A
    # regression that put a stray additive term in the formula would
    # leak through at a different R.
    l_big = star.calc_star_luminosity(0.0, 100.0 * R_sun)
    assert l_big == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_calc_star_luminosity_below_min_temp():
    """Below PLANCK_MIN_TEMPERATURE = 0.1 K, the gate forces L to
    zero. The threshold is a hard floor, not a soft taper."""
    l = star.calc_star_luminosity(0.01, 1.0 * R_sun)
    assert l == pytest.approx(0.0)
    # Threshold-flip discrimination: lifting tmp above the 0.1 K
    # gate must produce a non-zero luminosity. A regression that
    # left the gate clamped on or used a higher threshold would keep
    # the second call zero too.
    l_above = star.calc_star_luminosity(1.0, 1.0 * R_sun)
    assert l_above > 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_calc_star_luminosity_scales_with_temp():
    """Luminosity increases steeply with temperature (T^4)."""
    l1 = star.calc_star_luminosity(5000.0, 1.0 * R_sun)
    l2 = star.calc_star_luminosity(6000.0, 1.0 * R_sun)
    assert l2 > l1
    assert l2 / l1 > 1.5


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_calc_instellation_inverse_square_law():
    """Instellation follows 1/r^2 with separation. Doubling r must
    quarter the instellation (ratio 4); a regression to 1/r would
    give 2, a regression to 1/r^3 would give 8."""
    sep1 = 1 * AU
    sep2 = 2 * AU
    s1 = star.calc_instellation(Teff_sun, 1.0 * R_sun, sep1)
    s2 = star.calc_instellation(Teff_sun, 1.0 * R_sun, sep2)
    ratio = s1 / s2
    assert ratio == pytest.approx(4.0, rel=1e-6)
    # Exponent guards: inverse-linear would give 2; inverse-cubic
    # would give 8. Both wrong-exponent landings are well separated
    # from 4.
    assert abs(ratio - 2.0) > 1.0
    assert abs(ratio - 8.0) > 3.0


@pytest.mark.unit
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_calc_instellation_earth_like():
    """Earth-like planet receives ~1361 W/m^2 at 1 AU, the IAU 2015
    nominal total solar irradiance (Resolution B3, S_0 = 1361 W/m^2).
    """
    s = star.calc_instellation(Teff_sun, 1.0 * R_sun, 1.0 * AU)
    assert s == pytest.approx(1361.0, rel=0.01)
    # Scale guard: instellation is in W/m^2 at SI units. An R-in-km
    # vs R-in-m slip would land at ~1e-9 W/m^2; an AU-in-km vs AU-in-m
    # slip would land at ~1e3x the correct value. The bracket
    # discriminates either slip.
    assert 1e3 < s < 2e3
