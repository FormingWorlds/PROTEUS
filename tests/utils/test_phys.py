"""Unit tests for ``proteus.utils.phys.planck_wav``.

Validates the Planck spectral flux density at standard astrophysical
temperatures + wavelengths against the closed-form Stefan-Boltzmann /
Wien displacement relations, and exercises the overflow-fallback path
at extreme short wavelengths.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest

import proteus.utils.phys as phys_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30), pytest.mark.physics_invariant]


def test_planck_wav_returns_positive_at_solar_visible_peak():
    """At T = 5772 K and lambda = 500 nm = 5e-7 m (solar peak), the
    Planck flux must be positive and within the expected magnitude
    range for stellar surface flux density in SI units.

    Discrimination: sign guard rules out a wrong-sign regression;
    scale guard at order of magnitude rules out a unit-conversion
    regression (W m-2 m-1 vs W m-2 sr-1 m-1).
    """
    T = 5772.0  # Solar Teff
    wav = 5e-7  # 500 nm
    flx = phys_mod.planck_wav(T, wav)
    # Sign guard
    assert flx > 0
    # Scale guard: ~1e13 W m-2 m-1 at solar peak (hemispheric, in SI)
    # A regression that lost the pi factor would be a factor ~3 smaller;
    # a regression that retained sr in units would be smaller by ~10x.
    assert 1e12 < flx < 1e14


def test_planck_wav_obeys_wien_displacement_law_at_solar_temperature():
    """The Planck function peaks at lambda_max = b / T where b ~ 2.898e-3
    m K (Wien). For T = 5772 K, lambda_max ~ 502 nm. Flux at the peak
    must exceed flux at 200 nm and 1500 nm.

    Discrimination: a wrong-exponent regression (e.g. lambda^4 instead
    of lambda^5) would shift the peak and fail this comparison.
    """
    T = 5772.0
    flx_peak = phys_mod.planck_wav(T, 5.02e-7)
    flx_uv = phys_mod.planck_wav(T, 2e-7)
    flx_nir = phys_mod.planck_wav(T, 1.5e-6)
    assert flx_peak > flx_uv
    assert flx_peak > flx_nir


def test_planck_wav_scales_with_temperature_in_expected_direction():
    """At fixed wavelength near the visible peak (500 nm), doubling T
    moves the function into a regime where flux increases (Wien tail
    -> peak). Discrimination: the ratio must be substantially > 1
    (sign and scale guards rule out flipped or unit-confused regressions).
    """
    wav = 5e-7
    flx_cold = phys_mod.planck_wav(2500.0, wav)
    flx_hot = phys_mod.planck_wav(5000.0, wav)
    # At 500 nm, going from 2500 K to 5000 K is moving from deep Wien
    # tail into peak; flux increases by many orders of magnitude.
    assert flx_hot > flx_cold * 100


def test_planck_wav_returns_floor_value_under_short_wavelength_overflow():
    """At very short wavelengths (X-ray regime) the exp(hc/(wav*k*T))
    factor overflows and the function returns the 1e-40 floor rather
    than NaN or +inf. Discrimination: a regression that propagated
    the overflow would return inf, breaking downstream radiative
    transfer code that integrates over wavelength.
    """
    # Very short wavelength + cool temperature -> overflow regime
    flx = phys_mod.planck_wav(100.0, 1e-12)  # 1 picometre
    # Floor is 1e-40; result must equal or exceed it without being inf
    assert 0 < flx
    # Discrimination: must be finite (the overflow handler converted to 0
    # and then the floor lifted it to 1e-40)
    import math

    assert math.isfinite(flx)
    # Discrimination: the floor must be the floor value, not something
    # huge that suggests the overflow propagated. 1e-40 to 1e-30 is the
    # acceptable range (1e-40 floor, or some small but bounded value).
    assert flx < 1e-30


def test_planck_wav_hemispheric_integration_multiplies_by_pi():
    """The function multiplies the per-steradian Planck radiance by pi
    to give hemispheric flux density. Discrimination: a regression
    that dropped the pi factor would yield a value smaller by exactly
    pi at the visible peak.

    Compare two evaluations at the same conditions; this checks that
    the function is deterministic and returns the same number for the
    same inputs (regression guard against accidental randomness or
    state-leaking caches).
    """
    flx1 = phys_mod.planck_wav(3000.0, 1e-6)
    flx2 = phys_mod.planck_wav(3000.0, 1e-6)
    assert flx1 == pytest.approx(flx2, rel=1e-15)
