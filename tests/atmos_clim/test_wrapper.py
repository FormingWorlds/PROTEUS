"""Unit tests for the pure-Python helpers in ``proteus.atmos_clim.wrapper``.

Covers ``update_wtg_surf`` (weak-temperature-gradient surface parameter)
and ``update_bolometry`` (transit + eclipse depth closed-form
relations). Heavy dispatch helpers like ``run_atmosphere`` and
``ShallowMixedOceanLayer`` are exercised by integration tests in
nightly tier.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import math

import pytest

import proteus.atmos_clim.wrapper as atmos_wrapper
from proteus.utils.constants import const_R

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30), pytest.mark.physics_invariant]


# ---------------------------------------------------------------------------
# update_wtg_surf: weak-temperature-gradient surface scaling
# ---------------------------------------------------------------------------


def test_update_wtg_surf_closed_form_at_unit_inputs():
    """``update_wtg_surf`` computes wtg_surf = sqrt(R_mix * T_surf) /
    (omega * R_int) where omega = 2*pi / axial_period. With chosen
    inputs the closed-form value can be pinned to high precision.

    Discrimination: a regression that dropped the sqrt would land at
    R_mix * T_surf / (omega * R_int) which is much larger; a regression
    that flipped the period to omega would land at axial_period /
    (2*pi * R_int) * sqrt(R_mix * T_surf), much smaller.
    """
    hf_row = {
        'axial_period': 86400.0,  # 1 day in seconds
        'atm_kg_per_mol': 0.029,  # Earth-like mean molecular weight
        'T_surf': 300.0,  # K
        'R_int': 6.371e6,  # Earth radius in m
    }
    atmos_wrapper.update_wtg_surf(hf_row)

    omega = 2.0 * math.pi / 86400.0
    R_mix = const_R / 0.029
    expected = math.sqrt(R_mix * 300.0) / (omega * 6.371e6)

    assert hf_row['wtg_surf'] == pytest.approx(expected, rel=1e-12)
    # Discrimination: positivity guard rules out a sign flip
    assert hf_row['wtg_surf'] > 0
    # Discrimination: scale guard. For Earth-like conditions wtg_surf
    # is order ~0.05 (dimensionless WTG parameter). A regression that
    # dropped the sqrt would yield ~3e7; one that kept R_int^2 would
    # be ~1e-14.
    assert 1e-3 < hf_row['wtg_surf'] < 10.0


def test_update_wtg_surf_scales_inversely_with_planet_rotation_rate():
    """A slower-rotating planet (larger axial_period -> smaller omega)
    has a LARGER wtg_surf, because the WTG approximation is more valid
    when rotation is slow.

    Discrimination: a regression that put omega in the numerator would
    invert this scaling.
    """
    hf_fast = {
        'axial_period': 86400.0,
        'atm_kg_per_mol': 0.029,
        'T_surf': 300.0,
        'R_int': 6.371e6,
    }
    hf_slow = {
        'axial_period': 86400.0 * 10.0,
        'atm_kg_per_mol': 0.029,
        'T_surf': 300.0,
        'R_int': 6.371e6,
    }
    atmos_wrapper.update_wtg_surf(hf_fast)
    atmos_wrapper.update_wtg_surf(hf_slow)

    # Slower rotation -> larger wtg
    assert hf_slow['wtg_surf'] > hf_fast['wtg_surf']
    # Discrimination: 10x slower rotation means 10x larger wtg_surf
    # (linear in axial_period via omega in denominator)
    ratio = hf_slow['wtg_surf'] / hf_fast['wtg_surf']
    assert ratio == pytest.approx(10.0, rel=1e-12)


# ---------------------------------------------------------------------------
# update_bolometry: transit + eclipse depth
# ---------------------------------------------------------------------------


def test_update_bolometry_transit_depth_is_ratio_of_radii_squared():
    """Transit depth = (R_obs / R_star)^2. With Earth-like geometry
    (R_obs ~ 6e6 m, R_star ~ 7e8 m), the depth is ~(6e6/7e8)^2 ~ 7e-5
    (about 73 ppm).

    Discrimination: a regression that used (R_obs/R_star) instead of
    squaring would yield ~8e-3 (1000x larger); a regression that
    flipped the ratio would yield ~1.3e4 (impossibly large).
    """
    hf_row = {
        'R_obs': 6.371e6,
        'R_star': 6.96e8,  # Solar radius in m
        'F_olr': 200.0,
        'F_sct': 100.0,
        'F_ins': 1361.0,
        'separation': 1.5e11,
    }
    atmos_wrapper.update_bolometry(hf_row)

    expected_transit = (6.371e6 / 6.96e8) ** 2
    assert hf_row['transit_depth'] == pytest.approx(expected_transit, rel=1e-12)
    # Discrimination: positivity + scale guards
    assert hf_row['transit_depth'] > 0
    # Earth-like transit depth is ~8e-5; a regression that dropped the
    # square would give ~9e-3, two orders of magnitude bigger.
    assert 1e-5 < hf_row['transit_depth'] < 1e-3


def test_update_bolometry_eclipse_depth_is_flux_ratio_times_radius_ratio_squared():
    """Eclipse depth = ((F_olr + F_sct) / F_ins) * (R_obs / separation)^2.
    The F_olr + F_sct is the planet's thermal+scattered flux at TOA;
    F_ins is the incoming stellar flux at TOA; the (R_obs/separation)^2
    factor accounts for the inverse-square attenuation from the planet
    to the star.

    Discrimination: a regression that dropped the (R_obs/separation)^2
    would yield a much larger depth (~order unity).
    """
    hf_row = {
        'R_obs': 6.371e6,
        'R_star': 6.96e8,
        'F_olr': 200.0,
        'F_sct': 100.0,
        'F_ins': 1361.0,
        'separation': 1.5e11,  # 1 AU
    }
    atmos_wrapper.update_bolometry(hf_row)

    expected_eclipse = (300.0 / 1361.0) * (6.371e6 / 1.5e11) ** 2
    assert hf_row['eclipse_depth'] == pytest.approx(expected_eclipse, rel=1e-12)
    # Discrimination: positivity + scale guards
    assert hf_row['eclipse_depth'] > 0
    # Earth-like eclipse depth is ~4e-10 (1 AU separation, Earth radius);
    # a regression that dropped the (R_obs/separation)^2 geometric factor
    # would land near 0.22 (the flux ratio alone), 9 orders larger.
    assert 1e-12 < hf_row['eclipse_depth'] < 1e-7


def test_update_bolometry_eclipse_depth_scales_with_flux_excess():
    """At fixed geometry, doubling (F_olr + F_sct) doubles the eclipse
    depth (linear in planet's emission flux). Discrimination: a
    regression that squared or exponentiated the flux ratio would not
    show this 2x scaling.
    """
    base_geometry = {
        'R_obs': 6.371e6,
        'R_star': 6.96e8,
        'F_ins': 1361.0,
        'separation': 1.5e11,
    }
    hf_base = {**base_geometry, 'F_olr': 200.0, 'F_sct': 100.0}
    hf_hot = {**base_geometry, 'F_olr': 400.0, 'F_sct': 200.0}

    atmos_wrapper.update_bolometry(hf_base)
    atmos_wrapper.update_bolometry(hf_hot)

    ratio = hf_hot['eclipse_depth'] / hf_base['eclipse_depth']
    assert ratio == pytest.approx(2.0, rel=1e-12)
