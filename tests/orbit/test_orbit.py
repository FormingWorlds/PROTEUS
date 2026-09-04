"""Unit tests for the star-planet tidal orbit evolution module
(``proteus.orbit.orbit``).

``sp0d``'s ODE right-hand sides (the Driscoll & Barnes 2015 ``de/dt``
and ``da/dt``) are private closures nested inside ``sp0d`` -- kept
that way deliberately (each orbital model has its own, differently
shaped right-hand side, so hoisting them to module scope would
require per-model name prefixes purely to satisfy testing, at the
cost of the module's readability). They are therefore tested here
as a black box, through the public ``sp0d(hf_row, dt)`` entry point:
calling ``sp0d`` with a very small ``dt`` and finite-differencing the
resulting change in ``semimajorax``/``eccentricity`` recovers the
instantaneous right-hand side to high precision (``solve_ivp``'s
default adaptive RK45 tracks the true derivative closely over a
sub-second time span; empirically, relative error is ~1e-4 at
``dt_yr=1e-6``), which is tight enough to reproduce the same
exponent- and sign-discrimination guards a direct closure call would
give.

Also exercises the public ``sp0d`` integrator over realistic step
sizes, and ``evolve_orbit_star``'s dispatch by
``config.orbit.star_planet_model``.

Anti-happy-path coverage (sp0d):

- The eccentricity derivative is linear in ``e`` and vanishes at
  ``e=0`` so the zero-eccentricity orbit is a fixed point.
- The semi-major axis derivative satisfies the kinematic relation
  ``da_dt = 2 a e * de_dt`` and is identically zero on circular
  orbits.
- Discriminating numeric values pin the exponents (``a**6.5``,
  ``Rpl**5``) so that a bugged ``a**5`` or ``Rpl**4`` is caught.
- Adversarial ``hf_row`` inputs (zero ``Imk2``, near-unity ``e``)
  are exercised to confirm ``sp0d`` does not crash or return
  non-finite results.
- ``evolve_orbit_star`` dispatch is exercised both for the
  recognized ``'sp0d'`` model and for an unrecognized model (no-op,
  the current source has no ``else`` branch).

``sp1d`` (planet spin + orbit, Hansen-coefficient-based) is tested the
same way, black-box through the public ``sp1d(hf_row, tides_o, dt)``
entry point, since its ODE closures are also private and per-model.
Unlike ``sp0d``, ``sp1d`` DOES track planetary spin as a state
variable, and its ``domega_dt``/``da_dt``/``de_dt`` are constructed so
that total angular momentum (``C_planet*Omega_p + L_orbital``) is
actually conserved -- verified here as a genuine physics invariant,
not assumed.

Anti-happy-path coverage (sp1d):

- Zero dissipation (all Love numbers 0) is an exact fixed point of
  ``(Omega_p, a, e)``.
- Total angular momentum is conserved to solver tolerance across a
  real, non-trivial evolution step.
- Adversarial near-unity eccentricity does not produce NaN/inf, and
  eccentricity is clamped at 0 rather than going negative.
- ``evolve_orbit_star`` dispatch to the ``'sp1d'`` branch is exercised
  through the public entry point, including its ``get_C_planet`` call.

Design note, not a bug: ``orbitals`` (the actual ODE right-hand side)
computes ``K_p0``/``K_p2`` from
``abs(Love_number.imag) * smooth_sign(forcing_frequency)`` rather than
using the Love number's raw signed imaginary part. This is necessary,
not incidental: the Love number is frozen for the whole
``solve_ivp`` call, but the actual forcing frequency (a function of
the solver's current trial ``Omega_p``/``a``/``e``) varies
continuously across the integration and can cross zero -- re-deriving
the dissipation sign from the CURRENT forcing frequency at every
evaluation is what keeps the response physically consistent (always
opposing relative motion) rather than risking an unphysical
sign/pumping artifact if the forcing frequency's sign flips mid-step.
``dE_dt`` (used only for a one-off debug-log power estimate at the
initial state, not for the dynamics) instead uses the raw signed
value directly (``-Love_number.imag``); that shortcut is reasonable
for a single-point snapshot at a known state, but would not be
correct if reused across a varying trajectory the way ``orbitals`` is.
A Love number carrying an unexpected raw sign therefore evolves the
orbit exactly the same way (by design), but can log a debug-only
tidal power estimate with the wrong sign.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_building.md
- docs/How-to/test_categorization.md
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import proteus.orbit.common as common_mod
from proteus.orbit.common import Tides_t
from proteus.orbit.orbit import evolve_orbit_star, sp0d, sp1d
from proteus.utils.constants import const_G, secs_per_year

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Finite-difference step for probing sp0d's instantaneous ODE right-hand
# side. Must be small enough that even a ~32x-faster-evolving case (e.g.
# the Rpl**5 scaling test, Rpl doubled) still has negligible bias from the
# rate itself changing across the step: empirically, the naive
# ratio-of-differences estimate for that 32x case is only accurate to
# ~3% at dt_yr=1e-6, but ~1e-5 relative at dt_yr=1e-9. Still far above
# float cancellation noise at the SI-unit magnitudes used here.
_FD_DT_YR = 1e-9


def _sp0d_instantaneous_rates(sma, ecc, Imk2, Mst, Rpl, Mpl, dt_yr=_FD_DT_YR):
    """Probe sp0d's private ODE right-hand side via finite difference:
    run one very short integration and divide the resulting change by
    the elapsed time. Returns ``(da_dt, de_dt)`` in SI units (m/s,
    1/s).
    """
    hf_row = {
        'semimajorax': sma,
        'eccentricity': ecc,
        'Imk2': Imk2,
        'M_star': Mst,
        'R_int': Rpl,
        'M_int': Mpl,
    }
    sp0d(hf_row, dt=dt_yr)
    dt_s = dt_yr * secs_per_year
    da_dt = (hf_row['semimajorax'] - sma) / dt_s
    de_dt = (hf_row['eccentricity'] - ecc) / dt_s
    return da_dt, de_dt


# Reusable "unit" system: Mst = Rpl = Mpl = Imk2 = 1 (SI units), so
# algebra is easy to verify by hand. G is always the real physical
# const_G (sp0d has no way to receive a substitute), so every
# "expected" value below uses const_G explicitly rather than 1.0.
_UNIT_SYS = dict(Imk2=1.0, Mst=1.0, Rpl=1.0, Mpl=1.0)


# ---------------------------------------------------------------------------
# sp0d's de/dt: eccentricity derivative (probed via finite difference)
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_sp0d_de_dt_vanishes_at_zero_eccentricity():
    """A circular orbit (e=0) is a fixed point of the tidal evolution.
    Holds for any semi-major axis: e=0 zeros the prefactor regardless
    of a, so we exercise two values of a to confirm the fixed point is
    a property of the eccentricity factor, not an accidental zero at
    one a value."""
    _, de1 = _sp0d_instantaneous_rates(sma=1.0, ecc=0.0, **_UNIT_SYS)
    assert de1 == pytest.approx(0.0, abs=1e-20)
    # Limit-input invariant: a must drop out of the e=0 result; a
    # regression that introduced a stray a-dependent additive term
    # would only show at a != 1.
    _, de5 = _sp0d_instantaneous_rates(sma=5.0, ecc=0.0, **_UNIT_SYS)
    assert de5 == pytest.approx(0.0, abs=1e-20)


@pytest.mark.physics_invariant
def test_sp0d_de_dt_is_linear_in_eccentricity():
    """sp0d's de/dt scales linearly in ``e`` per Driscoll and Barnes
    (2015) Eq. 16."""
    _, base = _sp0d_instantaneous_rates(sma=1.0, ecc=0.01, **_UNIT_SYS)
    _, scaled = _sp0d_instantaneous_rates(sma=1.0, ecc=0.05, **_UNIT_SYS)
    # 5x e -> 5x de/dt within finite-difference precision.
    assert scaled == pytest.approx(5.0 * base, rel=1e-3)
    # Linearity guard: a quadratic-in-e regression would give a ratio
    # of 25, not 5. The absolute gap discriminates.
    assert abs(scaled / base - 25.0) > 10.0


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_sp0d_de_dt_matches_driscoll_barnes_2015_eq16():
    """Pin sp0d's de/dt against Driscoll and Barnes (2015)
    Astrobiology 15, 739, DOI 10.1089/ast.2015.1325, Eq. 16.
    (arXiv:1509.07452.)

    The paper writes the formula as

        de/dt = (21/2) Im(k2) M_*^(3/2) G^(1/2) R_p^5 / M_p * e / a^(13/2)

    with the paper convention Im(k2) < 0 for tidal dissipation (the
    paper's Eq. 4 makes -Im(k2) the positive dissipation efficiency).
    The PROTEUS source uses positive Imk2 in its calling convention, so
    the formula evaluated with Imk2 = +1 here returns a positive de/dt;
    documented as a known-sign-convention item in the source docstring.

    Discriminating evaluation point: Imk2 = Mst = Rpl = Mpl = 1 (SI
    units), a = 2, e = 0.5, with the real ``const_G`` (sp0d always
    uses the physical constant, not a caller-supplied value):

        de/dt = (21/2) * const_G**0.5 * 0.5 / 2**6.5

    See ``docs/Validation/orbit/orbit.md`` for the validation registry
    entry.
    """
    _, val = _sp0d_instantaneous_rates(sma=2.0, ecc=0.5, **_UNIT_SYS)
    expected = (21.0 / 2.0) * const_G**0.5 * 0.5 / (2.0**6.5)
    assert val == pytest.approx(expected, rel=1e-3)
    # Exponent-error guard: an off-by-one in the semi-major-axis
    # exponent puts the result far outside the finite-difference noise
    # floor. Check a**5 (way too big) AND a**7 (too small), the two
    # closest neighbouring exponents.
    wrong_a5 = (21.0 / 2.0) * const_G**0.5 * 0.5 / (2.0**5)
    wrong_a7 = (21.0 / 2.0) * const_G**0.5 * 0.5 / (2.0**7)
    assert abs(val - wrong_a5) / expected > 1.0
    assert abs(val - wrong_a7) / expected > 0.1
    # Sign guard: under the PROTEUS calling convention (positive Imk2)
    # the RHS is positive. A flip would fail this. Note: this does NOT
    # certify the convention matches Driscoll and Barnes; see source
    # docstring.
    assert val > 0.0


@pytest.mark.physics_invariant
def test_sp0d_de_dt_scales_as_radius_to_the_fifth_power():
    """Doubling ``Rpl`` should multiply sp0d's de/dt by 32, not 16 or
    64."""
    _, small = _sp0d_instantaneous_rates(
        sma=1.0, ecc=0.1, Rpl=1.0, **{k: v for k, v in _UNIT_SYS.items() if k != 'Rpl'}
    )
    _, big = _sp0d_instantaneous_rates(
        sma=1.0, ecc=0.1, Rpl=2.0, **{k: v for k, v in _UNIT_SYS.items() if k != 'Rpl'}
    )
    ratio = big / small
    assert ratio == pytest.approx(32.0, rel=1e-3)
    # Exponent guards: 2**5 = 32; reject the neighbours 2**4 = 16 and
    # 2**6 = 64. The base 2 choice makes adjacent-exponent regressions
    # land at well-separated values.
    assert abs(ratio - 16.0) > 10.0
    assert abs(ratio - 64.0) > 20.0


@pytest.mark.physics_invariant
def test_sp0d_de_dt_inverse_planet_mass_dependence():
    """sp0d's de/dt is inversely proportional to planet mass
    ``Mpl``."""
    _, val_light = _sp0d_instantaneous_rates(
        sma=1.0, ecc=0.1, Mpl=1.0, **{k: v for k, v in _UNIT_SYS.items() if k != 'Mpl'}
    )
    _, val_heavy = _sp0d_instantaneous_rates(
        sma=1.0, ecc=0.1, Mpl=3.0, **{k: v for k, v in _UNIT_SYS.items() if k != 'Mpl'}
    )
    ratio = val_light / val_heavy
    assert ratio == pytest.approx(3.0, rel=1e-3)
    # Monotonicity guard: heavier planet always damps slower under
    # inverse-Mpl. A regression that put Mpl in the numerator would
    # flip the inequality.
    assert val_heavy < val_light


# ---------------------------------------------------------------------------
# sp0d's da/dt: semi-major axis derivative (probed via finite difference)
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_sp0d_da_dt_is_zero_for_circular_orbit():
    """At ``e=0``, ``da_dt = 2 a e de_dt = 0`` regardless of ``a`` or
    params.

    A bug that dropped the ``e`` factor (``da_dt = 2 a de_dt``) would
    return a nonzero value here.
    """
    da10, _ = _sp0d_instantaneous_rates(sma=10.0, ecc=0.0, **_UNIT_SYS)
    assert da10 == pytest.approx(0.0, abs=1e-20)
    # Limit-input invariant: the fixed point is independent of a. A
    # regression that recovered an a-dependent constant term at e=0
    # would only show at a different a value.
    da_half, _ = _sp0d_instantaneous_rates(sma=0.5, ecc=0.0, **_UNIT_SYS)
    assert da_half == pytest.approx(0.0, abs=1e-20)


@pytest.mark.physics_invariant
def test_sp0d_da_dt_obeys_kinematic_identity():
    """``da_dt = 2 a e * de_dt`` must hold (mathematical identity;
    finite-difference precision, not exact machine precision, since
    both sides are independently probed from the same short
    integration)."""
    a, e = 1.5, 0.3
    da, de = _sp0d_instantaneous_rates(sma=a, ecc=e, **_UNIT_SYS)
    rhs = 2.0 * a * e * de
    assert da == pytest.approx(rhs, rel=1e-2)
    # Identity guard at a different (a, e): the relation must hold
    # everywhere, not at one accidentally-coincidental point.
    a2, e2 = 3.0, 0.7
    da2, de2 = _sp0d_instantaneous_rates(sma=a2, ecc=e2, **_UNIT_SYS)
    rhs2 = 2.0 * a2 * e2 * de2
    assert da2 == pytest.approx(rhs2, rel=1e-2)


@pytest.mark.physics_invariant
def test_sp0d_da_dt_quadratic_in_eccentricity():
    """Together with the de/dt linearity, sp0d's da/dt is quadratic in
    ``e``."""
    da_base, _ = _sp0d_instantaneous_rates(sma=1.0, ecc=0.01, **_UNIT_SYS)
    da_scaled, _ = _sp0d_instantaneous_rates(sma=1.0, ecc=0.04, **_UNIT_SYS)
    # (0.04 / 0.01)**2 = 16
    assert da_scaled == pytest.approx(16.0 * da_base, rel=1e-2)
    # Exponent guard: linear-in-e (ratio 4) and cubic (ratio 64) are
    # both rejected by the absolute gap from 16.
    assert abs(da_scaled / da_base - 4.0) > 5.0
    assert abs(da_scaled / da_base - 64.0) > 20.0


# ---------------------------------------------------------------------------
# sp0d: public integrator that mutates hf_row in place
# ---------------------------------------------------------------------------


def _make_hf_row(
    *,
    sma_m: float = 1.5e11,
    ecc: float = 0.1,
    Imk2: float = 1e-3,
    M_star: float = 1.989e30,
    R_int: float = 6.371e6,
    M_int: float = 5.972e24,
) -> dict:
    return {
        'semimajorax': sma_m,
        'eccentricity': ecc,
        'Imk2': Imk2,
        'M_star': M_star,
        'R_int': R_int,
        'M_int': M_int,
    }


@pytest.mark.physics_invariant
def test_sp0d_zero_imk2_preserves_sma_and_eccentricity():
    """With ``Imk2 = 0`` the right-hand sides vanish identically; the
    integrator must leave ``sma`` and ``ecc`` unchanged regardless of
    the step length.
    """
    hf_row = _make_hf_row(sma_m=2.0 * 1.5e11, ecc=0.4, Imk2=0.0)
    sp0d(hf_row, dt=1e5)
    assert hf_row['semimajorax'] == pytest.approx(2.0 * 1.5e11)
    assert hf_row['eccentricity'] == pytest.approx(0.4)


@pytest.mark.physics_invariant
def test_sp0d_zero_eccentricity_is_a_fixed_point():
    """``e = 0`` is a fixed point of the system; with positive
    ``Imk2`` the integrator must keep the orbit circular and ``sma``
    unchanged.
    """
    hf_row = _make_hf_row(sma_m=1.5e11, ecc=0.0, Imk2=1e-2)
    sp0d(hf_row, dt=1e3)
    assert hf_row['eccentricity'] == pytest.approx(0.0, abs=1e-30)
    assert hf_row['semimajorax'] == pytest.approx(1.5e11)


@pytest.mark.physics_invariant
def test_sp0d_returns_finite_for_high_eccentricity():
    """Adversarial near-unity eccentricity: the integrator must not
    emit NaN or inf for an aggressive but physically valid input.
    """
    hf_row = _make_hf_row(ecc=0.95, Imk2=1e-6)
    sp0d(hf_row, dt=1.0)
    assert np.isfinite(hf_row['semimajorax'])
    assert np.isfinite(hf_row['eccentricity'])


def test_sp0d_mutates_hf_row_in_place():
    """``sp0d`` mutates ``hf_row`` and returns ``None``; it must not
    silently return a new dict.
    """
    hf_row = _make_hf_row(sma_m=0.7 * 1.5e11)
    result = sp0d(hf_row, dt=1.0)
    assert result is None
    assert hf_row['semimajorax'] == pytest.approx(0.7 * 1.5e11)


# ---------------------------------------------------------------------------
# evolve_orbit_star: dispatch by config.orbit.star_planet_model
# ---------------------------------------------------------------------------


def _make_star_planet_config(model) -> Any:
    return cast(Any, SimpleNamespace(orbit=SimpleNamespace(star_planet_model=model)))


def test_evolve_orbit_star_sp0d_model_evolves_hf_row():
    """``config.orbit.star_planet_model == 'sp0d'`` dispatches to
    ``sp0d``, which must actually change ``hf_row`` (nonzero Imk2, so
    the orbit is not a fixed point).

    ``tides_o`` is unused on this branch (only ``sp1d`` reads it), so
    a bare object stands in for it without needing a real ``Tides_t``.
    """
    hf_row = _make_hf_row(ecc=0.2, Imk2=1e-2)
    sma_before = hf_row['semimajorax']
    config = _make_star_planet_config('sp0d')
    interior_o = SimpleNamespace(dt=1e7)

    evolve_orbit_star(hf_row, config, tides_o=object(), interior_o=interior_o)

    # Discrimination: the orbit actually evolved (not a silent no-op).
    # An absolute (not relative-tolerance) gap avoids a false negative
    # from a real-but-tiny change at a short step: pytest.approx's
    # default rel window can otherwise swallow a genuine, if small,
    # displacement.
    assert abs(hf_row['semimajorax'] - sma_before) > 1e-3


def test_evolve_orbit_star_unrecognized_model_is_a_no_op():
    """An unrecognized (or ``None``) ``star_planet_model`` falls
    through both branches: the current source has no ``else``, so
    ``hf_row`` must be left completely untouched and no exception
    raised.
    """
    hf_row = _make_hf_row(ecc=0.2, Imk2=1e-2)
    hf_row_before = dict(hf_row)
    config = _make_star_planet_config(None)
    interior_o = SimpleNamespace(dt=1e4)

    evolve_orbit_star(hf_row, config, tides_o=object(), interior_o=interior_o)

    assert hf_row == hf_row_before


# ---------------------------------------------------------------------------
# sp1d: planet spin + orbit (Hansen-coefficient-based), black-box tested
# through the public sp1d(hf_row, tides_o, dt) entry point.
# ---------------------------------------------------------------------------

# sp1d calls proteus.orbit.common.get_all_m_hansen, which lazily builds a
# module-level cache (_hansen_table) via a full FFT sweep over a ~100-point
# eccentricity grid on first use -- on the order of a minute of wall time
# (the same trap fixed in tests/orbit/test_common.py). Every sp1d test
# force-builds a tiny, fast table instead, and monkeypatch restores the
# prior (possibly None) module state afterward so this can never leak into
# other tests sharing the same pytest process.
_FAST_E_GRID = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
_FAST_KMIN, _FAST_KMAX = -6, 6

_SP1D_MST = 1.989e30  # 1 M_sun
_SP1D_MPL = 5.972e24  # 1 M_earth
_SP1D_RST = 6.957e8  # 1 R_sun
_SP1D_RPL = 6.371e6  # 1 R_earth
_SP1D_CPL = 0.33 * _SP1D_MPL * _SP1D_RPL**2  # Earth-like moment-of-inertia factor


@pytest.fixture
def _fast_hansen_table(monkeypatch):
    """Force-build a small, fast Hansen table for the duration of one
    test, restoring whatever module-level table (if any) existed
    before, so this cannot bleed into other tests.
    """
    monkeypatch.setattr(common_mod, '_hansen_table', None)
    common_mod.init_hansen_table(
        e_grid=_FAST_E_GRID, kmin=_FAST_KMIN, kmax=_FAST_KMAX, n_deg=2, force=True
    )


def _make_planet_star_tides(lnk_value: complex) -> Tides_t:
    """Build a ``Tides_t`` with a ``('planet', 'star')`` mode table
    covering ``m in {0, 2}`` over the fast Hansen table's k-range, all
    modes carrying the same Love number (sp1d only reads the m=0 and
    m=2, degree-2 rows).
    """
    nmk = [(2, 0, k) for k in range(_FAST_KMIN, _FAST_KMAX + 1)] + [
        (2, 2, k) for k in range(_FAST_KMIN, _FAST_KMAX + 1)
    ]
    tides_o = Tides_t()
    entry = tides_o.add(primary='planet', perturber='star')
    entry.nmk = np.array(nmk, dtype=int)
    entry.LNk = np.full(len(nmk), lnk_value, dtype=complex)
    return tides_o


def _make_sp1d_hf_row(*, axial_period=86400.0, sma=0.02 * 1.496e11, ecc=0.3):
    return {
        'axial_period': axial_period,
        'semimajorax': sma,
        'eccentricity': ecc,
        'M_int': _SP1D_MPL,
        'M_star': _SP1D_MST,
        'R_int': _SP1D_RPL,
        'R_star': _SP1D_RST,
        'C_planet': _SP1D_CPL,
    }


def _sp1d_spin_and_orbital_am(hf_row: dict) -> tuple[float, float]:
    """Angular momentum components under sp1d's own bookkeeping:
    ``(planet spin AM, orbital AM)`` -- planet spin is
    ``C_planet * Omega_p``, orbital AM is the two-body reduced-mass
    form.
    """
    a, e, axial_period = hf_row['semimajorax'], hf_row['eccentricity'], hf_row['axial_period']
    omega_p = 2 * np.pi / axial_period
    mu = _SP1D_MST * _SP1D_MPL / (_SP1D_MST + _SP1D_MPL)
    l_orb = mu * np.sqrt(const_G * (_SP1D_MST + _SP1D_MPL) * a * (1 - e**2))
    return hf_row['C_planet'] * omega_p, l_orb


def _sp1d_total_am(hf_row: dict) -> float:
    """Total angular momentum: sum of the two components above."""
    spin, orb = _sp1d_spin_and_orbital_am(hf_row)
    return spin + orb


@pytest.mark.physics_invariant
def test_sp1d_zero_dissipation_is_an_exact_fixed_point(_fast_hansen_table):
    """With every Love number exactly 0 (no tidal response), spin,
    semimajor axis, and eccentricity must all be left exactly
    unchanged: every term in ``orbitals`` is proportional to
    ``K_p0``/``K_p2``, both zero.
    """
    hf_row = _make_sp1d_hf_row()
    axial_before, sma_before, ecc_before = (
        hf_row['axial_period'],
        hf_row['semimajorax'],
        hf_row['eccentricity'],
    )
    tides_o = _make_planet_star_tides(0.0 + 0.0j)

    sp1d(hf_row, tides_o, dt=1e5)

    assert hf_row['axial_period'] == pytest.approx(axial_before, rel=1e-12)
    assert hf_row['semimajorax'] == pytest.approx(sma_before, rel=1e-12)
    assert hf_row['eccentricity'] == pytest.approx(ecc_before, rel=1e-12)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_sp1d_conserves_total_angular_momentum(_fast_hansen_table):
    """Physical law (not specific to Driscoll & Barnes or any single
    paper): for an isolated two-body system with only internal tidal
    torques, total angular momentum (planet spin + orbital) is
    conserved. sp1d's ``domega_dt``/``da_dt``/``de_dt`` are
    constructed to respect this -- checked here across a real,
    non-trivial step (eccentricity drops from 0.3 to well below 0.1,
    semimajor axis shrinks by several percent), not a near-zero
    finite-difference probe.

    Tolerance rel=1e-6 matches ``solve_ivp``'s configured ``rtol``.
    """
    hf_row = _make_sp1d_hf_row(ecc=0.3)
    am_before = _sp1d_total_am(hf_row)
    tides_o = _make_planet_star_tides(-0.01 - 0.02j)

    sp1d(hf_row, tides_o, dt=1e5)

    # Discrimination: the step must have actually done something
    # substantial, or a bug that silently no-oped would trivially
    # "conserve" AM too.
    assert hf_row['eccentricity'] < 0.2
    am_after = _sp1d_total_am(hf_row)
    assert am_after == pytest.approx(am_before, rel=1e-6)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_sp1d_spin_am_gain_matches_orbital_am_loss(_fast_hansen_table):
    """A more targeted phrasing of the same conservation law above,
    needed because the planet's spin AM is ~1e-6 of the total system
    AM at these masses (Earth-like planet, Sun-like star): a bug
    confined entirely to ``domega_dt`` changes the spin term by a
    large relative amount while barely moving the SUM (total AM),
    since the sum is completely dominated by the orbital term.
    Comparing the two individually-sized deltas directly
    (``Delta(spin AM) == -Delta(orbital AM)``) is sensitive to
    exactly this class of bug, which the raw total-AM check above is
    not (verified by injecting a 17% error into ``domega_dt``'s
    coupling coefficient: the total-AM check above still passed, but
    this delta comparison failed by the same ~17%).

    Tolerance rel=5e-2: looser than the raw total-AM check because
    differencing two close orbital-AM values before comparing to the
    much smaller spin delta amplifies relative solver-tolerance and
    Hansen-table interpolation-grid noise; still tight enough to
    reject a double-digit-percent coupling error with a wide margin.
    """
    hf_row = _make_sp1d_hf_row(ecc=0.3)
    spin_before, orb_before = _sp1d_spin_and_orbital_am(hf_row)
    tides_o = _make_planet_star_tides(-0.01 - 0.02j)

    sp1d(hf_row, tides_o, dt=1e5)

    spin_after, orb_after = _sp1d_spin_and_orbital_am(hf_row)
    d_spin = spin_after - spin_before
    d_orb = orb_after - orb_before
    # Discrimination: both deltas must be substantial, not near-zero
    # (a near-zero step would make the ratio numerically meaningless).
    assert abs(d_spin) > 1e30
    assert abs(d_orb) > 1e30
    assert d_spin == pytest.approx(-d_orb, rel=5e-2)
    # Sign guard: spin gains AM (planet spins up) exactly as the
    # orbit loses it during circularization.
    assert d_spin > 0.0
    assert d_orb < 0.0


@pytest.mark.physics_invariant
def test_sp1d_dissipation_circularizes_regardless_of_input_love_number_sign(
    _fast_hansen_table,
):
    """``orbitals`` derives the dissipation direction from the sign of
    the CURRENT forcing frequency alone (``smooth_sign``), using only
    the magnitude of the Love number's imaginary part
    (``abs(LNk.imag)``) -- necessary because the Love number is fixed
    for the whole integration while the forcing frequency varies as
    the solver explores trial states (see module docstring). Pins
    that a positive- and a negative-imaginary-part Love number of the
    same magnitude produce IDENTICAL evolution: the raw sign carried
    by the input is not part of the dynamics' contract, unlike
    ``dE_dt``'s debug-only power estimate.
    """
    hf_row_neg = _make_sp1d_hf_row(ecc=0.3)
    hf_row_pos = _make_sp1d_hf_row(ecc=0.3)
    tides_neg = _make_planet_star_tides(-0.01 - 0.02j)
    tides_pos = _make_planet_star_tides(-0.01 + 0.02j)

    sp1d(hf_row_neg, tides_neg, dt=1e5)
    sp1d(hf_row_pos, tides_pos, dt=1e5)

    assert hf_row_neg['eccentricity'] == pytest.approx(hf_row_pos['eccentricity'], rel=1e-10)
    assert hf_row_neg['semimajorax'] == pytest.approx(hf_row_pos['semimajorax'], rel=1e-10)
    # Discrimination: this is circularization, not a coincidental
    # no-op -- eccentricity actually dropped from the 0.3 IC.
    assert hf_row_neg['eccentricity'] < 0.3


@pytest.mark.physics_invariant
def test_sp1d_eccentricity_clamped_at_zero_not_negative(_fast_hansen_table):
    """Aggressive dissipation at high initial eccentricity can drive
    the raw ODE solution for ``e`` slightly negative within a step;
    the source clamps this to exactly 0.0 rather than reporting an
    unphysical negative eccentricity, and the output must stay
    finite.
    """
    hf_row = _make_sp1d_hf_row(ecc=0.9)
    tides_o = _make_planet_star_tides(-0.01 - 0.02j)

    sp1d(hf_row, tides_o, dt=1e4)

    assert np.isfinite(hf_row['eccentricity'])
    assert np.isfinite(hf_row['semimajorax'])
    assert hf_row['eccentricity'] >= 0.0


def test_sp1d_sma_dot_matches_bookkeeping_identity(_fast_hansen_table):
    """``hf_row['sma_dot_planet']`` must equal the actual change in
    ``semimajorax`` over the step divided by the elapsed time in
    seconds -- pins the bookkeeping identity so a regression that
    used the wrong ``sol.y`` row/index would be caught even though
    ``semimajorax`` itself might still look plausible.
    """
    hf_row = _make_sp1d_hf_row(ecc=0.3)
    sma_before = hf_row['semimajorax']
    tides_o = _make_planet_star_tides(-0.01 - 0.02j)
    dt_yr = 1e5

    sp1d(hf_row, tides_o, dt=dt_yr)

    expected_sma_dot = (hf_row['semimajorax'] - sma_before) / (dt_yr * secs_per_year)
    assert hf_row['sma_dot_planet'] == pytest.approx(expected_sma_dot, rel=1e-9)


def _uniform_sphere_interior_for_c_planet(nlev_b: int = 20):
    """Minimal Interior_t-like stand-in for ``get_C_planet``: a
    uniform-density sphere so its own moment of inertia has a known
    closed form, mirroring the fixture in tests/orbit/test_common.py.
    """
    radius = np.linspace(0.0, _SP1D_RPL, nlev_b)
    density = np.full(nlev_b - 1, 5500.0)
    interior_o = SimpleNamespace(radius=radius, density=density)
    return interior_o


def test_evolve_orbit_star_sp1d_model_calls_get_c_planet_and_evolves_hf_row(
    _fast_hansen_table,
):
    """``config.orbit.star_planet_model == 'sp1d'`` dispatches through
    ``get_C_planet`` (populating ``hf_row['C_planet']`` from the
    interior profile) and then ``sp1d``, via the public
    ``evolve_orbit_star`` entry point.
    """
    hf_row = _make_sp1d_hf_row(ecc=0.3)
    del hf_row['C_planet']  # get_C_planet must populate this itself
    hf_row['M_int'] = _SP1D_MPL
    hf_row['R_int'] = _SP1D_RPL
    tides_o = _make_planet_star_tides(-0.01 - 0.02j)
    interior_o = _uniform_sphere_interior_for_c_planet()
    interior_o.dt = 1e5
    config = cast(
        Any,
        SimpleNamespace(
            orbit=SimpleNamespace(star_planet_model='sp1d'),
            interior_energetics=SimpleNamespace(module='aragog'),
        ),
    )

    evolve_orbit_star(hf_row, config, tides_o=tides_o, interior_o=interior_o)

    assert 'C_planet' in hf_row
    assert hf_row['C_planet'] > 0.0
    # Discrimination: the orbit actually evolved under sp1d, not a
    # silent no-op.
    assert hf_row['eccentricity'] < 0.3
