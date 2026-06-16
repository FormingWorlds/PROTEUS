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
    """``L = I omega + Mpl (G(Mpl+Msa) a)**0.5``: both terms vanish at
    ``omega = 0`` and ``a = 0``."""
    assert Ltot(ω=0.0, a=0.0, params=_params()) == pytest.approx(0.0, abs=1e-12)
    # Limit-input invariant: when only omega is zero but a is positive,
    # the orbital term must remain. A regression that masked Ltot to
    # zero on either-zero input would fail here.
    assert Ltot(ω=0.0, a=1.0, params=_params()) > 0.0


@pytest.mark.physics_invariant
def test_ltot_is_linear_in_spin():
    """The spin term is ``I omega``; doubling ``omega`` adds exactly
    ``I d_omega`` to L."""
    base = Ltot(ω=1.0, a=1.0, params=_params())
    boosted = Ltot(ω=3.0, a=1.0, params=_params())
    # The orbital term cancels in the difference, leaving ``I * d_omega = 2``.
    assert boosted - base == pytest.approx(2.0, rel=1e-12)
    # Linearity guard at a third spin value: L(omega=5) - L(omega=1)
    # must give I*(5-1) = 4. A regression that introduced quadratic
    # omega dependence would fail one of the two equalities.
    extra = Ltot(ω=5.0, a=1.0, params=_params())
    assert extra - base == pytest.approx(4.0, rel=1e-12)


@pytest.mark.physics_invariant
def test_ltot_orbital_term_scales_as_sqrt_a():
    """Orbital term is ``Msa * sqrt(G * (Mpl+Msa) * a)``, scaling as
    ``a**0.5``. Multiplying ``a`` by 4 must multiply the orbital
    contribution by 2.0, not 4.0 or 16.0.
    """
    # Strip out the spin term by setting omega = 0.
    L_small = Ltot(ω=0.0, a=1.0, params=_params())
    L_big = Ltot(ω=0.0, a=4.0, params=_params())
    ratio = L_big / L_small
    assert ratio == pytest.approx(2.0, rel=1e-12)
    # Exponent guards: sqrt(4) = 2; linear-in-a gives 4, quadratic gives
    # 16. Both wrong-exponent landings are well separated from 2.
    assert abs(ratio - 4.0) > 1.0
    assert abs(ratio - 16.0) > 10.0


# ---------------------------------------------------------------------------
# dω_dt: spin derivative
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_d_omega_dt_vanishes_when_dE_tidal_is_zero():
    """No tidal dissipation gives no spin-down (numerator is zero)."""
    assert dω_dt(a=1.0, ω=1.0, params=_params(dE=0.0)) == pytest.approx(0.0, abs=1e-12)
    # Limit-input invariant: the zero result must be independent of
    # the (a, omega) operating point. Exercise a second state to
    # rule out an accidental cancellation at (1, 1).
    assert dω_dt(a=2.5, ω=0.7, params=_params(dE=0.0)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.physics_invariant
def test_d_omega_dt_sign_follows_negative_dE_tidal():
    """The formula prepends a minus sign on ``dE_tidal``: with
    ``dE_tidal > 0`` the spin derivative must be negative.
    """
    val = dω_dt(a=1.0, ω=1.0, params=_params(dE=1.0))
    assert val < 0.0
    # Sign symmetry: flipping the sign of dE_tidal must flip the sign
    # of the spin derivative. A regression that lost the linear-in-dE
    # dependence would give the same sign for both.
    val_neg = dω_dt(a=1.0, ω=1.0, params=_params(dE=-1.0))
    assert val_neg > 0.0


# ---------------------------------------------------------------------------
# da_dt: semi-major axis derivative
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_da_dt_zero_when_dE_tidal_zero():
    """``da_dt = -2 I a / (L - I omega) * d_omega_dt``: zero whenever
    ``d_omega_dt`` is zero."""
    assert da_dt(a=1.0, ω=1.0, params=_params(dE=0.0)) == pytest.approx(0.0, abs=1e-12)
    # Limit-input invariant: zero-tidal-power must zero da/dt for any
    # valid (a, omega). A regression that dropped the dE factor from
    # the chain through dw/dt would emit a non-zero value at a
    # different operating point.
    assert da_dt(a=3.0, ω=0.4, params=_params(dE=0.0)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.physics_invariant
def test_da_dt_matches_korenaga_eq59_closed_form_value():
    """Pin ``da_dt`` against an independently hand-computed value of
    Korenaga (2023) Eq. 59 at ``(I, L, G, Mpl, Msa, dE) = (1.5, 8.0, 1.0,
    1.0, 0.1, 1.0)`` and ``(a, ω) = (2.0, 0.5)``.

    Hand derivation:
        dω/dt = -1 / (1.5*0.5 + 0.15 / (2*(8 - 0.75)))
              = -1 / 0.76034482758...  =  -1.31519274376...
        da/dt = -2*1.5*2 / 7.25 * dω/dt
              =  -0.82758620690... * -1.31519274376...
              =   1.08843537414966...

    The pin is independent of the source (no call to ``dω_dt`` in the
    test) and so discriminates an Eq. 59 rearrangement bug; a regression
    that replaces the ``-2 I a`` prefactor by ``-3 I a`` would land at
    ~1.63 instead of ~1.088, well outside the 1e-12 tolerance.
    """
    params = _params(I=1.5, L=8.0)
    expected = 1.08843537414966
    actual = da_dt(a=2.0, ω=0.5, params=params)
    assert actual == pytest.approx(expected, rel=1e-12)
    # Sign guard: dE > 0 and L > I*ω make dω/dt < 0 (spin slows), which
    # combined with -2 I a / (L - I ω) < 0 must yield da/dt > 0 (orbit
    # expands). The factor of `-3` rearrangement would still be positive
    # so this guard alone does not catch it, but it pins the qualitative
    # outward-migration prediction Korenaga Eq. 59 makes for the prograde
    # Earth-Moon configuration.
    assert actual > 0.0
    # Scale guard: a prefactor of `-3` would land at ~1.63, outside the
    # [1.05, 1.15] window; a prefactor of `-1` would land at ~0.544.
    assert 1.05 < actual < 1.15


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
    # Autonomy invariant at a third t: a regression that picked up a
    # linear-in-t term would generally fail one of the three equalities
    # even if two coincided at the chosen pair.
    mid = orbitals(t=2.5, z=z, params=_params())
    assert mid == early


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
    # Scale guard: the converted value lands in seconds (~86400 s for
    # one rotation in hours = 24 h), well above the 100 s lower bound
    # and below 1e6 s. A regression that forgot the conversion would
    # leave the value at 24.0, below the lower bound.
    assert 1e4 < hf_row['axial_period'] < 1e6


def test_update_satellite_first_call_falls_back_to_sor_when_axial_period_is_none():
    """When ``config.orbit.axial_period`` is ``None``, the planet locks
    into a 1:1 spin-orbit resonance with the satellite's orbital period."""
    cfg = _make_config(axial_period_h=None)
    hf_row = _make_hf_row(time=0.5, orbital_period_s=4.32e4)
    update_satellite(hf_row, cfg, dt=1.0)
    assert hf_row['axial_period'] == pytest.approx(4.32e4)
    # 1:1 SOR pass-through: changing the orbital period must drive an
    # identical change in axial_period under SOR. A regression that
    # stamped a constant default would fail this second case.
    cfg2 = _make_config(axial_period_h=None)
    hf_row2 = _make_hf_row(time=0.5, orbital_period_s=2.16e4)
    update_satellite(hf_row2, cfg2, dt=1.0)
    assert hf_row2['axial_period'] == pytest.approx(2.16e4)


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
    ~ 1/81 ~ 1.2% for the Earth-Moon system.

    For Earth parameters the two components of Eq. 60 evaluate to:
      - spin term I_E * Omega ~ 7.05e33 kg m^2 / s
      - orbital term M_M sqrt(G (M_E + M_M) a) ~ 2.89e34 kg m^2 / s
      - total L_total ~ 3.60e34 kg m^2 / s

    The orbital component matches the Touma and Wisdom (1994) value
    of ~2.85e34 kg m^2 / s for the present-day Earth-Moon orbit.

    See ``docs/Validation/orbit/satellite.md`` for the validation
    registry entry and the re-derivation note.
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
    # lands at ~3.60e34 kg m^2 / s for the total (spin ~7.05e33 +
    # orbital ~2.89e34). The orbital component matches Touma and
    # Wisdom (1994) (~2.85e34). The [1e34, 1e35] bracket catches any
    # SI-vs-CGS or kg-vs-g unit slip and discriminates the M_sat form
    # in Eq. 60 from a substitution of M_planet, which would inflate
    # the orbital term to ~2.4e36 (well above the upper bound).
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
