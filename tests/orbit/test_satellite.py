"""Unit tests for the satellite-orbit evolution module
(``proteus.orbit.satellite``), based on Korenaga (2023) Eqs. 58-59.

Exercises the right-hand sides ``dω_dt`` and ``da_dt``, the angular-
momentum bookkeeping ``Ltot``, the ``orbitals`` wrapper used by
``scipy.integrate.solve_ivp``, and the ``update_satellite``
orchestrator that mutates ``hf_row`` in place.

Anti-happy-path coverage:

- Zero tidal-power input must yield zero rotational and orbital
  evolution: a fixed-point check that holds regardless of
  integrator tolerance.
- Angular-momentum book-keeping (``Ltot``) must scale linearly in
  ``ω`` and as ``a**0.5`` in the orbital term; both exponents are
  pinned to discriminating numeric values.
- ``orbitals`` must return ``[da_dt, dω_dt]`` in that order
  (a swap would silently corrupt the integration).
- ``update_satellite`` first-call branch must convert the user-set
  ``axial_period`` from hours to seconds, and must fall back to
  spin-orbit-resonance when the config requests it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from proteus.orbit.satellite import (
    Ltot,
    da_dt,
    dω_dt,
    orbitals,
    update_satellite,
)
from proteus.utils.constants import const_G, secs_per_hour

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# Reusable parameter tuple (I, L, G, Mpl, Msa, dE_tidal).
# Use simple unit-scaled values so algebra is easy to verify.
def _params(I=1.0, L=10.0, G=1.0, Mpl=1.0, Msa=0.1, dE=1.0):
    return (I, L, G, Mpl, Msa, dE)


# ---------------------------------------------------------------------------
# Ltot: angular momentum bookkeeping
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_ltot_returns_zero_when_omega_and_a_are_zero():
    """``L = I ω + Mpl (G(Mpl+Msa) a)**0.5``: both terms vanish at
    ``ω = 0`` and ``a = 0``."""
    assert Ltot(ω=0.0, a=0.0, params=_params()) == 0.0


@pytest.mark.physics_invariant
def test_ltot_is_linear_in_spin():
    """The spin term is ``I ω``; doubling ``ω`` adds exactly ``I dω`` to L."""
    base = Ltot(ω=1.0, a=1.0, params=_params())
    boosted = Ltot(ω=3.0, a=1.0, params=_params())
    # The orbital term cancels in the difference, leaving ``I (ω' - ω) = 2``.
    assert boosted - base == pytest.approx(2.0, rel=1e-12)


@pytest.mark.physics_invariant
def test_ltot_orbital_term_scales_as_sqrt_a():
    """Orbital term is ``Mpl * sqrt(G * (Mpl+Msa) * a)``, scaling as
    ``a**0.5``. Multiplying ``a`` by 4 must multiply the orbital
    contribution by 2.0, not 4.0 or 16.0.
    """
    # Strip out the spin term by setting ω = 0.
    L_small = Ltot(ω=0.0, a=1.0, params=_params())
    L_big = Ltot(ω=0.0, a=4.0, params=_params())
    assert L_big / L_small == pytest.approx(2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# dω_dt: spin derivative
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_d_omega_dt_vanishes_when_dE_tidal_is_zero():
    """No tidal dissipation → no spin-down (numerator is zero)."""
    assert dω_dt(a=1.0, ω=1.0, params=_params(dE=0.0)) == 0.0


@pytest.mark.physics_invariant
def test_d_omega_dt_sign_follows_negative_dE_tidal():
    """The formula prepends a minus sign on ``dE_tidal``: with
    ``dE_tidal > 0`` the spin derivative must be negative.
    """
    val = dω_dt(a=1.0, ω=1.0, params=_params(dE=1.0))
    assert val < 0.0


# ---------------------------------------------------------------------------
# da_dt: semi-major axis derivative
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_da_dt_zero_when_dE_tidal_zero():
    """``da_dt = -2 I a / (L - I ω) * dω_dt``: zero whenever ``dω_dt`` is zero."""
    assert da_dt(a=1.0, ω=1.0, params=_params(dE=0.0)) == 0.0


@pytest.mark.physics_invariant
def test_da_dt_kinematic_relation_holds():
    """``da_dt`` must equal ``-2 I a / (L - I ω) * dω_dt`` exactly
    (mathematical identity)."""
    I, L = 1.5, 8.0
    a, ω = 2.0, 0.5
    params = _params(I=I, L=L)
    expected = -2.0 * I * a / (L - I * ω) * dω_dt(a=a, ω=ω, params=params)
    assert da_dt(a=a, ω=ω, params=params) == pytest.approx(expected, rel=1e-14)


# ---------------------------------------------------------------------------
# orbitals: ODE wrapper
# ---------------------------------------------------------------------------


def test_orbitals_returns_da_then_d_omega_in_that_order():
    """``orbitals`` returns ``[da_dt, dω_dt]``: order must match the
    state-vector convention used by ``update_satellite``."""
    z = [1.5, 0.5]
    out = orbitals(t=0.0, z=z, params=_params())
    assert out[0] == da_dt(a=z[0], ω=z[1], params=_params())
    assert out[1] == dω_dt(a=z[0], ω=z[1], params=_params())
    assert len(out) == 2


def test_orbitals_is_autonomous():
    """The system has no explicit time dependence, so identical state gives identical RHS."""
    z = [1.5, 0.5]
    early = orbitals(t=0.0, z=z, params=_params())
    late = orbitals(t=1e9, z=z, params=_params())
    assert early == late


# ---------------------------------------------------------------------------
# update_satellite: top-level orchestrator
# ---------------------------------------------------------------------------


def _make_config(
    *,
    semimajoraxis_sat: float = 3.844e8,  # m, Earth-Moon distance
    mass_sat: float = 7.342e22,  # kg, Moon mass
    axial_period_h: float | None = 24.0,
) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            orbit=SimpleNamespace(
                semimajoraxis_sat=semimajoraxis_sat,
                mass_sat=mass_sat,
                axial_period=axial_period_h,
            )
        ),
    )


def _make_hf_row(
    *,
    time: float,
    R_int: float = 6.371e6,
    M_int: float = 5.972e24,
    F_tidal: float = 1e-3,
    orbital_period_s: float = 86400.0,
) -> dict:
    return {
        'Time': time,
        'R_int': R_int,
        'M_int': M_int,
        'F_tidal': F_tidal,
        'orbital_period': orbital_period_s,
    }


def test_update_satellite_first_call_converts_axial_period_hours_to_seconds():
    """A user-supplied ``axial_period`` in hours must be multiplied by
    ``secs_per_hour`` on the first call. A regression that stored the
    raw value (24.0) would be off by a factor of 3600.
    """
    cfg = _make_config(axial_period_h=24.0)
    hf_row = _make_hf_row(time=0.0)
    update_satellite(hf_row, cfg, dt=1.0)
    assert hf_row['axial_period'] == pytest.approx(24.0 * secs_per_hour)


def test_update_satellite_first_call_falls_back_to_sor_when_axial_period_is_none():
    """When ``config.orbit.axial_period`` is ``None``, the planet locks
    into a 1:1 spin-orbit resonance with the satellite's orbital period."""
    cfg = _make_config(axial_period_h=None)
    hf_row = _make_hf_row(time=0.5, orbital_period_s=4.32e4)
    update_satellite(hf_row, cfg, dt=1.0)
    assert hf_row['axial_period'] == pytest.approx(4.32e4)


def test_update_satellite_first_call_seeds_satellite_mass_and_sma():
    """The first-call branch must copy ``mass_sat`` and ``semimajoraxis_sat``
    from the config to ``hf_row`` verbatim."""
    cfg = _make_config(semimajoraxis_sat=1.5e9, mass_sat=2e22)
    hf_row = _make_hf_row(time=0.0)
    update_satellite(hf_row, cfg, dt=1.0)
    assert hf_row['semimajorax_sat'] == pytest.approx(1.5e9)
    assert hf_row['M_sat'] == pytest.approx(2e22)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_update_satellite_angular_momentum_matches_korenaga_2023_eq60():
    """Pin the planet-satellite angular-momentum bookkeeping against
    Korenaga (2023) Icarus 400, 115564, Eq. 60:

        L = I_E * Omega + M_M * sqrt(G * (M_E + M_M) * a)

    where M_M is the SATELLITE mass (Moon), not M_E (Earth). This is
    the M_M << M_E limit of the textbook reduced-mass orbital angular
    momentum L_orb = mu * sqrt(G (M_E + M_M) a) with reduced mass
    mu = M_E * M_M / (M_E + M_M); the limit's relative error is M_M / M_E
    ~ 1/81 ~ 1.2% for the Earth-Moon system. Cross-check value:
    Touma and Wisdom (1994) report the present-day Earth-Moon orbital
    angular momentum as ~2.85e34 kg m^2 / s.
    """
    cfg = _make_config(semimajoraxis_sat=3.844e8, mass_sat=7.342e22, axial_period_h=24.0)
    hf_row = _make_hf_row(time=0.0, R_int=6.371e6, M_int=5.972e24)
    update_satellite(hf_row, cfg, dt=1.0)
    I = 2 / 5 * 5.972e24 * 6.371e6**2
    omega = 2 * np.pi / (24.0 * secs_per_hour)
    # Eq. 60: orbital prefactor is the satellite mass M_M.
    expected = I * omega + 7.342e22 * (const_G * (5.972e24 + 7.342e22) * 3.844e8) ** 0.5
    assert hf_row['plan_sat_am'] == pytest.approx(expected, rel=1e-6)
    # Sign guard: total system AM is positive for a prograde Moon.
    assert hf_row['plan_sat_am'] > 0.0
    # Scale guard: Korenaga Eq. 60 evaluated on the Earth-Moon system
    # lands at ~2.89e34 kg m^2 / s, matching Touma and Wisdom (1994).
    # The [1e34, 1e35] bracket catches any SI-vs-CGS or kg-vs-g unit
    # slip; it would also catch a regression to the pre-fix M_planet
    # prefactor, which inflated the result to ~2.4e36.
    assert 1e34 < hf_row['plan_sat_am'] < 1e35


@pytest.mark.physics_invariant
def test_update_satellite_finite_for_long_integration_step():
    """Adversarial-but-physical: a long-baseline integration must
    produce finite ``semimajorax_sat`` and ``axial_period``.
    """
    cfg = _make_config()
    # Bootstrap by running first-call once, then advance.
    hf_row = _make_hf_row(time=0.0, F_tidal=1e-3)
    update_satellite(hf_row, cfg, dt=1.0)
    hf_row['Time'] = 1e3
    update_satellite(hf_row, cfg, dt=10.0)
    assert np.isfinite(hf_row['semimajorax_sat'])
    assert np.isfinite(hf_row['axial_period'])
    assert hf_row['axial_period'] > 0.0


def test_update_satellite_mutates_hf_row_in_place():
    """The orchestrator must mutate ``hf_row`` and return ``None``."""
    cfg = _make_config()
    hf_row = _make_hf_row(time=0.0)
    result = update_satellite(hf_row, cfg, dt=1.0)
    assert result is None
    # All four first-call outputs must be set.
    for key in ('semimajorax_sat', 'M_sat', 'axial_period', 'plan_sat_am'):
        assert key in hf_row
