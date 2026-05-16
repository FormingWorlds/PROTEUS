"""Unit tests for the Driscoll & Barnes (2015) tidal orbit evolution module
(``proteus.orbit.orbit``).

Exercises the ODE right-hand sides ``de_dt`` and ``da_dt``, the
``orbitals`` wrapper used by ``scipy.integrate.solve_ivp``, and the
``evolve_orbital`` orchestrator that mutates ``hf_row`` in place.

Anti-happy-path coverage:

- The eccentricity derivative is linear in ``e`` and vanishes at
  ``e=0`` so the zero-eccentricity orbit is a fixed point.
- The semi-major axis derivative satisfies the kinematic relation
  ``da_dt = 2 a e * de_dt`` and is identically zero on circular
  orbits.
- Discriminating numeric values pin the exponents (``a**6.5``,
  ``Rpl**5``) so that a bugged ``a**5`` or ``Rpl**4`` is caught.
- Adversarial ``hf_row`` inputs (zero ``Imk2``, near-unity ``e``)
  are exercised to confirm the orchestrator does not crash or
  return non-finite results.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from proteus.orbit.orbit import da_dt, de_dt, evolve_orbital, orbitals
from proteus.utils.constants import AU

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Reusable parameter tuple (Imk2, Mst, G, Rpl, Mpl) with unit scales so
# that algebra is easy to verify by hand.
_UNIT_PARAMS = (1.0, 1.0, 1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# de_dt: eccentricity derivative
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_de_dt_vanishes_at_zero_eccentricity():
    """A circular orbit (e=0) is a fixed point of the tidal evolution.
    Holds for any semi-major axis: e=0 zeros the prefactor regardless
    of a, so we exercise two values of a to confirm the fixed point is
    a property of the eccentricity factor, not an accidental zero at
    one a value."""
    assert de_dt(a=1.0, e=0.0, params=_UNIT_PARAMS) == 0.0
    # Limit-input invariant: a must drop out of the e=0 result; a
    # regression that introduced a stray a-dependent additive term
    # would only show at a != 1.
    assert de_dt(a=5.0, e=0.0, params=_UNIT_PARAMS) == 0.0


@pytest.mark.physics_invariant
def test_de_dt_is_linear_in_eccentricity():
    """``de_dt`` scales linearly in ``e`` per Driscoll and Barnes (2015) Eq. 16."""
    base = de_dt(a=1.0, e=0.01, params=_UNIT_PARAMS)
    scaled = de_dt(a=1.0, e=0.05, params=_UNIT_PARAMS)
    # 5x e -> 5x de/dt within float precision
    assert scaled == pytest.approx(5.0 * base, rel=1e-12)
    # Linearity guard: a quadratic-in-e regression would give a ratio
    # of 25, not 5. The absolute gap discriminates.
    assert abs(scaled / base - 25.0) > 10.0


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_de_dt_matches_driscoll_barnes_2015_eq16():
    """Pin de_dt against Driscoll and Barnes (2015) Astrobiology 15, 739,
    DOI 10.1089/ast.2015.1325, Eq. 16. (arXiv:1509.07452.)

    The paper writes the formula as

        de/dt = (21/2) Im(k2) M_*^(3/2) G^(1/2) R_p^5 / M_p * e / a^(13/2)

    with the paper convention Im(k2) < 0 for tidal dissipation (the
    paper's Eq. 4 makes -Im(k2) the positive dissipation efficiency).
    The PROTEUS source uses positive Imk2 in its calling convention, so
    the formula evaluated with Imk2 = +1 here returns a positive de/dt;
    documented as a known-sign-convention item in the source docstring.

    Discriminating evaluation point: Imk2 = Mst = G = Rpl = Mpl = 1,
    a = 2, e = 0.5 gives

        de/dt = (21/2) * 0.5 / 2**6.5 = 5.7996e-2
    """
    val = de_dt(a=2.0, e=0.5, params=_UNIT_PARAMS)
    expected = (21.0 / 2.0) * 0.5 / (2.0**6.5)
    assert val == pytest.approx(expected, rel=1e-12)
    # Exponent-error guard: an off-by-one in the semi-major-axis
    # exponent puts the result far outside the rel=1e-12 main assertion.
    # Check a**5 (way too big at 0.164) AND a**7 (way too small at 0.041
    # but only 0.017 away from 0.058, hence a tighter threshold). The
    # main pytest.approx(expected, rel=1e-12) above is the primary
    # guard; these are explicit demonstrations that the chosen test
    # point discriminates against the two closest neighbouring
    # exponents.
    wrong_a5 = (21.0 / 2.0) * 0.5 / (2.0**5)
    wrong_a7 = (21.0 / 2.0) * 0.5 / (2.0**7)
    assert abs(val - wrong_a5) > 0.05
    assert abs(val - wrong_a7) > 0.01
    # Sign guard: under the PROTEUS calling convention (positive Imk2)
    # the RHS is positive. A flip would fail this. Note: this does NOT
    # certify the convention matches Driscoll and Barnes; see source
    # docstring.
    assert val > 0.0
    # Scale guard: order of magnitude is ~6e-2. A unit-conversion bug
    # (kg vs g, AU vs m) would land outside the [1e-3, 1.0] bracket.
    assert 1e-3 < val < 1.0


@pytest.mark.physics_invariant
def test_de_dt_scales_as_radius_to_the_fifth_power():
    """Doubling ``Rpl`` should multiply ``de_dt`` by 32, not 16 or 64."""
    p_small = (1.0, 1.0, 1.0, 1.0, 1.0)
    p_big = (1.0, 1.0, 1.0, 2.0, 1.0)
    ratio = de_dt(a=1.0, e=0.1, params=p_big) / de_dt(a=1.0, e=0.1, params=p_small)
    assert ratio == pytest.approx(32.0, rel=1e-12)
    # Exponent guards: 2**5 = 32; reject the neighbours 2**4 = 16 and
    # 2**6 = 64. The base 2 choice makes adjacent-exponent regressions
    # land at well-separated values.
    assert abs(ratio - 16.0) > 10.0
    assert abs(ratio - 64.0) > 20.0


@pytest.mark.physics_invariant
def test_de_dt_inverse_planet_mass_dependence():
    """``de_dt`` is inversely proportional to planet mass ``Mpl``."""
    p_light = (1.0, 1.0, 1.0, 1.0, 1.0)
    p_heavy = (1.0, 1.0, 1.0, 1.0, 3.0)
    val_light = de_dt(a=1.0, e=0.1, params=p_light)
    val_heavy = de_dt(a=1.0, e=0.1, params=p_heavy)
    ratio = val_light / val_heavy
    assert ratio == pytest.approx(3.0, rel=1e-12)
    # Monotonicity guard: heavier planet always damps slower under
    # inverse-Mpl. A regression that put Mpl in the numerator would
    # flip the inequality.
    assert val_heavy < val_light


# ---------------------------------------------------------------------------
# da_dt: semi-major axis derivative
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_da_dt_is_zero_for_circular_orbit():
    """At ``e=0``, ``da_dt = 2 a e de_dt = 0`` regardless of ``a`` or params.

    A bug that dropped the ``e`` factor (``da_dt = 2 a de_dt``) would
    return a nonzero value here.
    """
    assert da_dt(a=10.0, e=0.0, params=_UNIT_PARAMS) == 0.0
    # Limit-input invariant: the fixed point is independent of a. A
    # regression that recovered an a-dependent constant term at e=0
    # would only show at a different a value.
    assert da_dt(a=0.5, e=0.0, params=_UNIT_PARAMS) == 0.0


@pytest.mark.physics_invariant
def test_da_dt_obeys_kinematic_identity():
    """``da_dt = 2 a e * de_dt`` must hold exactly (mathematical identity)."""
    a, e = 1.5, 0.3
    lhs = da_dt(a=a, e=e, params=_UNIT_PARAMS)
    rhs = 2.0 * a * e * de_dt(a=a, e=e, params=_UNIT_PARAMS)
    assert lhs == pytest.approx(rhs, rel=1e-14)
    # Identity guard at a different (a, e): the relation must hold
    # everywhere, not at one accidentally-coincidental point. Exercise
    # at a second point well away from the first.
    a2, e2 = 3.0, 0.7
    lhs2 = da_dt(a=a2, e=e2, params=_UNIT_PARAMS)
    rhs2 = 2.0 * a2 * e2 * de_dt(a=a2, e=e2, params=_UNIT_PARAMS)
    assert lhs2 == pytest.approx(rhs2, rel=1e-14)


@pytest.mark.physics_invariant
def test_da_dt_quadratic_in_eccentricity():
    """Together with the de_dt linearity, ``da_dt`` is quadratic in ``e``."""
    base = da_dt(a=1.0, e=0.01, params=_UNIT_PARAMS)
    scaled = da_dt(a=1.0, e=0.04, params=_UNIT_PARAMS)
    # (0.04 / 0.01)**2 = 16
    assert scaled == pytest.approx(16.0 * base, rel=1e-12)
    # Exponent guard: linear-in-e (ratio 4) and cubic (ratio 64) are
    # both rejected by the absolute gap from 16. The base 4x in e
    # makes adjacent-exponent landings well-separated.
    assert abs(scaled / base - 4.0) > 5.0
    assert abs(scaled / base - 64.0) > 20.0


# ---------------------------------------------------------------------------
# orbitals: ODE wrapper
# ---------------------------------------------------------------------------


def test_orbitals_returns_da_then_de_in_that_order():
    """``orbitals`` must return ``[da_dt, de_dt]``: order matters for solve_ivp.

    A swapped order would silently corrupt the integration trajectory.
    """
    z = [1.5, 0.3]
    out = orbitals(t=0.0, z=z, params=_UNIT_PARAMS)
    assert out[0] == da_dt(a=z[0], e=z[1], params=_UNIT_PARAMS)
    assert out[1] == de_dt(a=z[0], e=z[1], params=_UNIT_PARAMS)
    assert len(out) == 2


def test_orbitals_ignores_explicit_time_argument():
    """The system is autonomous: the same state at different ``t`` gives the same RHS."""
    z = [1.5, 0.3]
    a_early = orbitals(t=0.0, z=z, params=_UNIT_PARAMS)
    a_late = orbitals(t=1e9, z=z, params=_UNIT_PARAMS)
    assert a_early == a_late
    # Autonomy invariant: hold at a third time too. A regression that
    # picked up a time-dependent term (e.g. a stray dt in the
    # right-hand side) would generally fail one of the three
    # equalities even if two happened to coincide.
    a_mid = orbitals(t=1e3, z=z, params=_UNIT_PARAMS)
    assert a_mid == a_early


# ---------------------------------------------------------------------------
# evolve_orbital: top-level orchestrator that mutates hf_row in place
# ---------------------------------------------------------------------------


def _make_config(semimajoraxis_au: float = 1.0, eccentricity: float = 0.05) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            orbit=SimpleNamespace(
                semimajoraxis=semimajoraxis_au,
                eccentricity=eccentricity,
            )
        ),
    )


def _make_hf_row(
    *,
    time: float,
    sma_m: float = AU,
    ecc: float = 0.1,
    Imk2: float = 1e-3,
    M_star: float = 1.989e30,
    R_int: float = 6.371e6,
    M_int: float = 5.972e24,
) -> dict:
    return {
        'Time': time,
        'semimajorax': sma_m,
        'eccentricity': ecc,
        'Imk2': Imk2,
        'M_star': M_star,
        'R_int': R_int,
        'M_int': M_int,
    }


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_evolve_orbital_first_call_seeds_from_config_with_au_conversion():
    """On the first call (``Time <= 1``) the orchestrator must seed
    ``hf_row`` from ``config``, applying the AU → m conversion to the
    semi-major axis. The pin against ``0.5 * AU`` (~7.48e10 m) catches
    a regression that forgot the AU factor (which would leave
    ``semimajorax`` at 0.5 instead of ~7.48e10).
    """
    cfg = _make_config(semimajoraxis_au=0.5, eccentricity=0.2)
    hf_row = _make_hf_row(time=0.0, sma_m=999.0, ecc=999.0)  # garbage that must be overwritten
    evolve_orbital(hf_row, cfg, dt=1.0)
    assert hf_row['semimajorax'] == pytest.approx(0.5 * AU, rel=1e-12)
    assert hf_row['eccentricity'] == pytest.approx(0.2, rel=1e-12)
    # Explicit scale guard: a missing AU factor would leave the value
    # at 0.5; anything below 1e9 m would be sub-stellar-radius for any
    # real system. The lower bound discriminates AU-vs-meter slip.
    assert hf_row['semimajorax'] > 1e10


@pytest.mark.parametrize('time', [0.0, 0.5, 1.0])
def test_evolve_orbital_first_call_boundary_inclusive(time):
    """The ``Time <= 1`` boundary is inclusive: at exactly 1 yr the
    config-seed branch must still fire.
    """
    cfg = _make_config(semimajoraxis_au=0.3, eccentricity=0.01)
    hf_row = _make_hf_row(time=time, sma_m=42.0)
    evolve_orbital(hf_row, cfg, dt=0.1)
    assert hf_row['semimajorax'] == pytest.approx(0.3 * AU)
    # Config-seed branch must also overwrite the garbage eccentricity
    # in hf_row with the config value. A regression that gated the
    # eccentricity seed on a different boundary would leave
    # hf_row['eccentricity'] at the _make_hf_row default of 0.1.
    assert hf_row['eccentricity'] == pytest.approx(0.01, rel=1e-12)


@pytest.mark.physics_invariant
def test_evolve_orbital_zero_imk2_preserves_sma_and_eccentricity():
    """With ``Imk2 = 0`` the right-hand sides vanish identically; the
    integrator must leave ``sma`` and ``ecc`` unchanged regardless of
    the step length.
    """
    cfg = _make_config()
    hf_row = _make_hf_row(time=1e6, sma_m=2.0 * AU, ecc=0.4, Imk2=0.0)
    evolve_orbital(hf_row, cfg, dt=1e5)
    assert hf_row['semimajorax'] == pytest.approx(2.0 * AU)
    assert hf_row['eccentricity'] == pytest.approx(0.4)


@pytest.mark.physics_invariant
def test_evolve_orbital_zero_eccentricity_is_a_fixed_point():
    """``e = 0`` is a fixed point of the system; with positive ``Imk2``
    the integrator must keep the orbit circular and ``sma`` unchanged.
    """
    cfg = _make_config()
    hf_row = _make_hf_row(time=1e6, sma_m=AU, ecc=0.0, Imk2=1e-2)
    evolve_orbital(hf_row, cfg, dt=1e3)
    assert hf_row['eccentricity'] == pytest.approx(0.0, abs=1e-30)
    assert hf_row['semimajorax'] == pytest.approx(AU)


@pytest.mark.physics_invariant
def test_evolve_orbital_returns_finite_for_high_eccentricity():
    """Adversarial near-unity eccentricity: the orchestrator must not
    emit NaN or inf for an aggressive but physically valid input.
    """
    cfg = _make_config()
    hf_row = _make_hf_row(time=10.0, sma_m=AU, ecc=0.95, Imk2=1e-6)
    evolve_orbital(hf_row, cfg, dt=1.0)
    assert np.isfinite(hf_row['semimajorax'])
    assert np.isfinite(hf_row['eccentricity'])


def test_evolve_orbital_mutates_hf_row_in_place():
    """The orchestrator mutates ``hf_row`` and returns ``None``; it
    must not silently return a new dict.
    """
    cfg = _make_config(semimajoraxis_au=0.7, eccentricity=0.03)
    hf_row = _make_hf_row(time=0.0)
    result = evolve_orbital(hf_row, cfg, dt=1.0)
    assert result is None
    assert hf_row['semimajorax'] == pytest.approx(0.7 * AU)
