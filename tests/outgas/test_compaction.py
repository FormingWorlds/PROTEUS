"""Unit tests for ``proteus.outgas.compaction``.

The module under test computes how much interstitial melt a parcel retains
while it crosses a crystallising freezing front. Melt escapes only if it can
both percolate through the pores and have the solid matrix deform to close
them, so the slower of the two processes sets the drainage rate and a parcel
entering at porosity ``phi_top`` retains ``phi(t_res)`` from

    dphi/dt = -phi / max( L / w_D(phi), tau_s ).

This replaces the two free parameters of the published linear law: the
compaction time becomes the two timescales above, and the solidus-to-front
temperature interval disappears because the residence time is geometric rather
than thermal. These tests exercise:

* the mobility, the solver's ``aragog.eos.mobility_function``, against the
  separation velocity the solver computes with it,
* the Darcy velocity, including the matrix fraction the solver omits,
* the matrix deformation time and its independence of porosity,
* that the slower process controls, which is the sign of the combination and
  the one place an inverted comparison changes every answer,
* the timescales and trapped fractions of five martian fronts, against
  closed forms and a quadrature independent of the ODE solver,
* boundedness in ``[0, phi_top]`` without a clamp,
* the front-location guards: a doubled front, one too thin to resolve, one
  spanning too much of the mantle, one reaching the surface, and one holding
  melt denser than the solid, each with its cause recorded,
* a front still porous at the lowest node, which rests on the impermeable
  core-mantle boundary and is integrated with that boundary as its base,
* the lever-rule porosity and the volume-to-mass conversion.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from proteus.outgas.compaction import (
    BRANCH_DARCY,
    BRANCH_GUARD,
    BRANCH_MATRIX,
    BRANCH_NONE,
    darcy_velocity,
    drainage_integral,
    locate_front,
    matrix_time,
    mobility_function,
    porosity_from_densities,
    volume_to_mass_fraction,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Martian front of the five-case test: a 60 km front at a density
# contrast of 330 kg/m3 and martian gravity, entered at the disaggregation
# melt fraction. The fast front crosses it in 86 kyr, the slow one in 1.2 Myr.
_L = 60.0e3
_DRHO = 330.0
_G = 3.711
_PHI_C = 0.3
_SECS_PER_YEAR = 3.15576e7
_T_RES_FAST = _L / (0.70 / _SECS_PER_YEAR)
_T_RES_SLOW = _L / (0.05 / _SECS_PER_YEAR)


class _EndMemberEOS:
    """Phase-boundary densities the solver's separation velocity reads."""

    def __init__(self, rho_solid, rho_melt):
        self.rho = {'solid': rho_solid, 'melt': rho_melt}

    def _lookup_at_phase_boundary(self, prop, pressure, phase):
        return np.full(np.shape(pressure), self.rho[phase])


def _solver_velocity(porosity, grain_size, rho_s=4000.0, rho_l=3600.0, gravity=9.8, eta=100.0):
    """Separation velocity from the interior solver's own ``relative_velocity``.

    ``EntropyPhaseEvaluator.relative_velocity`` is called on a stand-in that
    carries the attributes it reads: a mush at the given porosity, fixed
    phase-boundary densities, gravity, grain size and the melt viscosity as
    drag. Nothing of the solver's law is transcribed here.
    """
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    phi = np.asarray(porosity, dtype=float)
    # An uninitialised evaluator, so the solver's own porosity and phase-boundary
    # lookups run on the attributes set here.
    stand_in = object.__new__(EntropyPhaseEvaluator)
    stand_in.__dict__.update(
        _const_properties=False,
        _eos=_EndMemberEOS(rho_s, rho_l),
        pressure=np.full(phi.shape, 1.0e10),
        _density=rho_s - phi * (rho_s - rho_l),
        _g=gravity,
        _grain_size=grain_size,
        _separation_viscosity='melt',
        _visc_liquid=eta,
        _viscosity_val=np.full(phi.shape, 1.0e12),
    )
    return EntropyPhaseEvaluator.relative_velocity(stand_in)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_mobility_reproduces_the_interior_solver_in_every_regime():
    """Cross-implementation check against the permeability the interior solver
    uses for gravitational separation (Bower et al. 2018 section 2.1), by
    calling the solver's own ``relative_velocity`` and backing the mobility out
    of it, so the mobility the drainage uses is the one the solver's separation
    velocity uses. The drainage velocity is the solver's melt velocity times
    (1 - phi), the buoyancy taken against the mixture rather than the solid,
    the one deliberate difference between the two."""
    pytest.importorskip('aragog')
    rho_s, rho_l, gravity, eta = 4000.0, 3600.0, 9.8, 100.0
    # Both blends and all three regimes, from the porosity floor to Stokes.
    phi = np.array([0.02, 0.05, 0.0769452, 0.1, 0.3, 0.5, 0.771462, 0.9, 0.99])
    for grain in (1.0e-4, 1.0e-3, 1.0e-2):
        v = _solver_velocity(phi, grain, rho_s, rho_l, gravity, eta)
        theirs = v * eta / ((rho_s - rho_l) * gravity)
        mine = mobility_function(phi, grain)
        # The solver soft-clips its porosity (width 1e-3), which raises it by
        # about 1e-6 / (4 phi); on the phi^2 branch that is 1.25e-3 of the
        # mobility at phi = 0.02 and under 2e-4 from 0.05 up.
        np.testing.assert_allclose(mine[phi >= 0.05], theirs[phi >= 0.05], rtol=2.0e-4)
        np.testing.assert_allclose(mine[phi < 0.05], theirs[phi < 0.05], rtol=1.5e-3)
        # The drainage velocity is the solver's velocity times (1 - phi).
        w = darcy_velocity(phi, grain, rho_s - rho_l, gravity, eta)
        np.testing.assert_allclose(w[phi >= 0.05], ((1.0 - phi) * v)[phi >= 0.05], rtol=2.0e-4)
    # Discrimination: a Rumpf-Gupte exponent of 4 instead of 4.5 would be off
    # by 0.3**-0.5 = 1.8 at phi = 0.3, far outside the tolerance.
    v_rg = _solver_velocity(np.array([0.3]), 1.0e-3)[0] * eta / ((rho_s - rho_l) * gravity)
    wrong = 1.0e-6 * 0.3**4.0 * (5.0 / 7.0)
    assert abs(wrong / v_rg - 1.0) > 0.5

    # The three regimes must actually differ, or the comparison above would
    # pass on a single-branch stub. Probe one porosity well inside each.
    f_bkc = mobility_function(0.03, 1.0e-3)
    f_rg = mobility_function(0.30, 1.0e-3)
    f_stokes = mobility_function(0.85, 1.0e-3)
    assert f_bkc < f_rg < f_stokes
    # Scale guard: deep in the Stokes regime the mobility is a^2 * 2/9 = 2.2e-7 m2,
    # five orders above the low-porosity branch; probed at 0.95 because the tanh
    # blend is still 2 percent short of the limit at 0.85.
    assert mobility_function(0.95, 1.0e-3) == pytest.approx(1.0e-6 * 2.0 / 9.0, rel=2e-3)
    assert f_bkc < 1.0e-4 * f_stokes

    # Edge case: at zero porosity the blend leaves a Stokes residue of order 1e-20
    # m2, as the interior solver does, twelve orders below the value at phi = 0.3,
    # so no drainage can come of it.
    solid = mobility_function(0.0, 1.0e-3)
    assert solid < 1.0e-18
    assert solid < 1.0e-9 * mobility_function(0.3, 1.0e-3)
    # Error contract: a non-physical grain size is refused rather than
    # silently producing a zero permeability.
    with pytest.raises(ValueError, match='grain_size'):
        mobility_function(0.3, 0.0)


@pytest.mark.physics_invariant
def test_darcy_velocity_carries_the_matrix_fraction_and_the_grain_scaling():
    """The percolation speed is (1-phi)|drho| g F(phi)/eta_melt. The (1-phi)
    factor is the melt flux relative to the mixture rather than to the solid;
    the interior solver omits it, and it is 30 percent at phi = 0.3."""
    grain, eta = 1.0e-3, 100.0
    w = darcy_velocity(_PHI_C, grain, _DRHO, _G, eta)
    expected = (1.0 - _PHI_C) * _DRHO * _G * mobility_function(_PHI_C, grain) / eta
    assert w == pytest.approx(expected, rel=1e-12)
    # Matrix-fraction guard: dropping (1-phi) inflates the speed by 1/0.7.
    without = _DRHO * _G * mobility_function(_PHI_C, grain) / eta
    assert without / w == pytest.approx(1.0 / 0.7, rel=1e-12)
    # Sign and scale: melt rises, and at millimetre grains this is metres per
    # year, not metres per second (1e-6 m/s scale).
    assert w > 0.0
    assert 1.0e-8 < w < 1.0e-4

    # Permeability scales as the grain size squared, so a decade in grain is
    # two decades in speed. This is the sensitivity that dominates the answer.
    w_coarse = darcy_velocity(_PHI_C, 1.0e-2, _DRHO, _G, eta)
    assert w_coarse / w == pytest.approx(100.0, rel=1e-9)
    # Inverse in the melt viscosity.
    assert darcy_velocity(_PHI_C, grain, _DRHO, _G, 10.0) / w == pytest.approx(10.0, rel=1e-9)

    # Edge case: a fully molten cell has no matrix to percolate through.
    assert darcy_velocity(1.0, grain, _DRHO, _G, eta) == pytest.approx(0.0, abs=1e-30)
    with pytest.raises(ValueError, match='melt viscosity'):
        darcy_velocity(_PHI_C, grain, _DRHO, _G, 0.0)


@pytest.mark.physics_invariant
def test_matrix_time_is_independent_of_porosity_and_linear_in_viscosity():
    """tau_s = mu_s/(drho g L). The compaction pressure drives a volumetric
    strain rate against the bulk viscosity zeta = mu_s/phi, and closing a
    porosity phi takes phi*zeta/(drho g L), so the porosity cancels."""
    tau = matrix_time(1.0e18, _DRHO, _G, _L)
    expected = 1.0e18 / (_DRHO * _G * _L)
    assert tau == pytest.approx(expected, rel=1e-12)
    # 431 yr for a soft mush under a 60 km martian front.
    assert tau / _SECS_PER_YEAR == pytest.approx(431.2, rel=1.0e-3)
    # Scale guard: ~1e10 s, not ~1e13 (a years-for-seconds slip).
    assert 1.0e10 < tau < 1.0e11

    # Linear in the mush viscosity, which is the parameter that decides the
    # regime; four decades of viscosity is four decades of time.
    assert matrix_time(1.0e22, _DRHO, _G, _L) / tau == pytest.approx(1.0e4, rel=1e-12)
    # Inverse in the front thickness.
    assert matrix_time(1.0e18, _DRHO, _G, 2.0 * _L) / tau == pytest.approx(0.5, rel=1e-12)

    # Edge case: a front of no thickness cannot compact at all.
    assert not np.isfinite(matrix_time(1.0e18, _DRHO, _G, 0.0))
    assert not np.isfinite(matrix_time(1.0e18, 0.0, _G, _L))


@pytest.mark.physics_invariant
def test_drainage_is_limited_by_the_slower_of_the_two_processes():
    """Melt leaves only if it both percolates and the matrix deforms, so the
    drainage time is max(tau_D, tau_s). Taking the faster process instead
    inverts the physics and drains a stiff mush that cannot actually compact."""
    grain, eta_m = 1.0e-3, 100.0
    # A stiff matrix with a fast percolation branch: tau_s dominates, so the
    # parcel retains nearly all its melt despite the pores being permeable.
    stiff, tau_d, tau_s = drainage_integral(
        _PHI_C, _T_RES_FAST, _L, grain, _DRHO, _G, eta_m, 1.0e22
    )
    assert tau_s > tau_d
    # Matrix-limited throughout, the integral has the closed form
    # phi_c exp(-t_res / tau_s), which the stiff solver reproduces to 1e-6.
    assert stiff == pytest.approx(_PHI_C * np.exp(-_T_RES_FAST / tau_s), rel=1.0e-6)
    assert stiff == pytest.approx(0.294096, rel=1.0e-5)
    # Discrimination: had the faster process been taken, the stiff case would
    # drain to well under 0.1 instead of sitting just below phi_c.
    assert stiff > 0.25

    # A soft matrix at the same permeability drains further, because now the
    # percolation branch is the slow one and it is slower than tau_s was small.
    soft, _, tau_s_soft = drainage_integral(
        _PHI_C, _T_RES_FAST, _L, grain, _DRHO, _G, eta_m, 1.0e18
    )
    assert tau_s_soft < tau_s
    assert soft < stiff
    assert soft == pytest.approx(0.194, rel=0.02)

    # Monotonicity: a softer matrix can never trap more than a stiffer one at
    # the same permeability, which holds across the whole viscosity range.
    retained = [
        drainage_integral(_PHI_C, _T_RES_FAST, _L, grain, _DRHO, _G, eta_m, mu)[0]
        for mu in (1.0e17, 1.0e18, 1.0e20, 1.0e22)
    ]
    assert np.all(np.diff(retained) >= -1e-12)


def _residence_time_grid(grain, melt_visc, mush_visc, matrix_fraction=True):
    """Time a parcel takes to drain from phi_c down to each porosity on a grid.

    The drainage equation is separable, so t(phi) = int_phi^phi_c tau(p) dp / p
    with tau = max(L / w_D, tau_s), a quadrature rather than an ODE solve.
    Trapezoid rule in ln(phi) on 20001 nodes down to phi = 1e-4, accurate to
    about 2e-7 in the porosity it returns. ``matrix_fraction=False`` drops the
    (1 - phi) of w_D, the most plausible slip in the percolation speed.
    """
    u = np.linspace(np.log(_PHI_C), np.log(1.0e-4), 20001)
    phi = np.exp(u)
    w = darcy_velocity(phi, grain, _DRHO, _G, melt_visc)
    if not matrix_fraction:
        w = w / (1.0 - phi)
    tau = np.maximum(_L / w, matrix_time(mush_visc, _DRHO, _G, _L))
    t = np.concatenate([[0.0], np.cumsum(0.5 * (tau[1:] + tau[:-1]) * -np.diff(u))])
    return phi, t


def _retained_by_quadrature(t_res, grain, melt_visc, mush_visc, matrix_fraction=True):
    """Porosity left after ``t_res``, by inverting :func:`_residence_time_grid`."""
    phi, t = _residence_time_grid(grain, melt_visc, mush_visc, matrix_fraction)
    return float(np.exp(np.interp(t_res, t, np.log(phi))))


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_drainage_matches_an_independent_quadrature_over_five_martian_fronts():
    """Trapped melt fractions and timescales for a 60 km martian front, entered
    at phi_c = 0.3 under drho = 330 kg/m3 and g = 3.711 m/s2, crossed at 70 and
    5 cm/yr (t_res = L / v_f = 85.7 kyr and 1.2 Myr), for five sets of grain
    size a, melt viscosity eta_m and mush viscosity mu_s.

    The model is the one shipped: w_D = (1 - phi) drho g F(phi) / eta_m with the
    three-regime F of Bower et al. (2018), tau_s = mu_s / (drho g L), and
    dphi/dt = -phi / max(L / w_D, tau_s). Its two timescales at the entry
    porosity have closed forms, and since the equation is separable the
    retained porosity F_vol solves t_res = int_F^phi_c max(L / w_D, tau_s) dphi
    / phi, a quadrature independent of the ODE solver. In the Rumpf-Gupte
    regime, F = (5/7) a^2 phi^4.5, that integral also has a series closed form.

    Expected, tau_D / t_res and tau_s / t_res at 70 cm/yr, then F_vol at 70
    and at 5 cm/yr:

    a [m]   eta_m [Pa s]  mu_s [Pa s]  tau_D/t    tau_s/t    F_70      F_5
    1e-4    10            1e20         8.1653     0.50314    0.27165   0.18134
    1e-3    100           1e22         0.81653    50.314     0.29410   0.22713
    1e-3    100           1e18         0.81653    0.0050314  0.19422   0.10897
    1e-3    1             1e18         0.0081653  0.0050314  0.069535  0.024485
    1e-2    1             1e18         8.1653e-5  0.0050314  0.0094342 0.0025379
    """
    cases = (
        (100.0e-6, 10.0, 1.0e20, 8.1653, 0.50314, 0.27165, 0.18134),
        (1.0e-3, 100.0, 1.0e22, 0.81653, 50.314, 0.29410, 0.22713),
        (1.0e-3, 100.0, 1.0e18, 0.81653, 0.0050314, 0.19422, 0.10897),
        (1.0e-3, 1.0, 1.0e18, 0.0081653, 0.0050314, 0.069535, 0.024485),
        (1.0e-2, 1.0, 1.0e18, 8.1653e-5, 0.0050314, 0.0094342, 0.0025379),
    )
    retained = {}
    for a, eta_m, eta_s, tau_d_pin, tau_s_pin, fast_pin, slow_pin in cases:
        fast, tau_d, tau_s = drainage_integral(
            _PHI_C, _T_RES_FAST, _L, a, _DRHO, _G, eta_m, eta_s
        )
        slow, _, _ = drainage_integral(_PHI_C, _T_RES_SLOW, _L, a, _DRHO, _G, eta_m, eta_s)
        retained[a, eta_m, eta_s] = (fast, slow)

        # Timescales in closed form. At phi = 0.3 the Stokes tanh weight adds
        # 4.5e-7 to the Rumpf-Gupte mobility, hence rel = 1e-6 on tau_D.
        w_top = (1.0 - _PHI_C) * _DRHO * _G * (5.0 / 7.0) * a**2 * _PHI_C**4.5 / eta_m
        assert tau_d == pytest.approx(_L / w_top, rel=1.0e-6)
        assert tau_s == pytest.approx(eta_s / (_DRHO * _G * _L), rel=1.0e-12)
        assert tau_d / _T_RES_FAST == pytest.approx(tau_d_pin, rel=1.0e-4)
        assert tau_s / _T_RES_FAST == pytest.approx(tau_s_pin, rel=1.0e-4)

        # Retained porosity against the quadrature, which is accurate to 2e-7;
        # the solver's rtol of 1e-8 leaves it within 1e-6.
        for t_res, value, pin in ((_T_RES_FAST, fast, fast_pin), (_T_RES_SLOW, slow, slow_pin)):
            reference = _retained_by_quadrature(t_res, a, eta_m, eta_s)
            assert value == pytest.approx(reference, rel=1.0e-6)
            assert value == pytest.approx(pin, rel=1.0e-4)
        # A slower front always drains further.
        assert slow < fast

    # Series closed form in the Rumpf-Gupte regime, dphi/dt = -(C/L)(1-phi) phi^5.5:
    # H(phi) = sum_k phi^(k-4.5) / (k-4.5) falls by C t_res / L. Held to 1e-5, as
    # near phi = 0.18 the lower tanh blend still adds about 1e-6.
    k = np.arange(200)

    def series(phi):
        return float(np.sum(phi ** (k - 4.5) / (k - 4.5)))

    for (a, eta_m, eta_s), t_res, index in (
        ((100.0e-6, 10.0, 1.0e20), _T_RES_FAST, 0),
        ((100.0e-6, 10.0, 1.0e20), _T_RES_SLOW, 1),
        ((1.0e-3, 100.0, 1.0e18), _T_RES_FAST, 0),
    ):
        c = _DRHO * _G * (5.0 / 7.0) * a**2 / eta_m
        drop = series(_PHI_C) - c * t_res / _L
        closed = brentq(lambda phi, drop=drop: series(phi) - drop, 0.1, _PHI_C, xtol=1e-14)
        assert retained[a, eta_m, eta_s][index] == pytest.approx(closed, rel=1.0e-5)

    # Discrimination: without the (1 - phi) of w_D the percolation-limited
    # cases drain further, by 0.008 to 0.0095 at 70 cm/yr, some 50 times the
    # pinning tolerance; the matrix-limited case is unaffected.
    for a, eta_m, eta_s in ((100.0e-6, 10.0, 1.0e20), (1.0e-3, 100.0, 1.0e18)):
        slip = _retained_by_quadrature(_T_RES_FAST, a, eta_m, eta_s, matrix_fraction=False)
        assert retained[a, eta_m, eta_s][0] - slip > 0.007
    stiff = retained[1.0e-3, 100.0, 1.0e22][0]
    assert _retained_by_quadrature(_T_RES_FAST, 1.0e-3, 100.0, 1.0e22, False) == pytest.approx(
        stiff, rel=1.0e-6
    )
    # The retained fraction spans a factor of thirty across the cases, so no
    # constant could pass the loop above.
    assert retained[100.0e-6, 10.0, 1.0e20][0] / retained[1.0e-2, 1.0, 1.0e18][0] > 20.0


@pytest.mark.physics_invariant
def test_drainage_is_bounded_without_a_clamp_and_decays_with_residence():
    """The integral is bounded in [0, phi_top] by construction, because the
    porosity decays but never reaches zero: the percolation speed falls as
    phi^2 or steeper, so the last melt leaves ever more slowly. That is what
    lets the front scheme run without the clamp the linear law needs."""
    grain, eta_m, eta_s = 1.0e-3, 1.0, 1.0e18
    previous = _PHI_C
    for t_res in (0.0, 1.0e10, 1.0e12, 1.0e14, 1.0e16):
        retained, _, _ = drainage_integral(_PHI_C, t_res, _L, grain, _DRHO, _G, eta_m, eta_s)
        assert 0.0 <= retained <= _PHI_C
        assert retained <= previous + 1e-12
        previous = retained
    # Even after an absurdly long residence the melt is not fully expelled.
    assert previous > 0.0

    # A zero-length residence retains exactly what entered, with no drainage.
    unchanged, _, _ = drainage_integral(_PHI_C, 0.0, _L, grain, _DRHO, _G, eta_m, eta_s)
    assert unchanged == pytest.approx(_PHI_C, rel=1e-12)

    # Edge case: an empty pore space stays empty rather than going negative.
    empty, _, _ = drainage_integral(0.0, _T_RES_FAST, _L, grain, _DRHO, _G, eta_m, eta_s)
    assert empty == pytest.approx(0.0, abs=1e-30)
    # Error contract: a non-finite residence returns the entry porosity as an
    # upper bound rather than integrating to a NaN.
    bad, _, _ = drainage_integral(_PHI_C, float('nan'), _L, grain, _DRHO, _G, eta_m, eta_s)
    assert bad == pytest.approx(_PHI_C, rel=1e-12)


@pytest.mark.physics_invariant
def test_front_location_bounds_the_mush_and_refuses_what_it_cannot_resolve():
    """The front runs from the solver's own rheological transition at the top
    to the porosity floor at the base. A front the mesh cannot resolve, one
    split in two, one spanning too much of the mantle, or one holding melt
    denser than the solid all take the guard branch, where the caller uses the
    entry porosity as an upper bound instead of integrating."""
    n = 40
    r = np.linspace(3.0e6, 6.0e6, n)
    phi = np.linspace(0.0, 1.0, n)  # melt fraction rises outward
    por = np.clip(phi * 0.95, 0.0, 1.0)
    rho_s = np.full(n, 3300.0)
    rho_l = np.full(n, 2970.0)

    geom, branch = locate_front(r, phi, por, rho_s, rho_l, rfront_loc=0.5, phi_min=0.01)
    assert branch == BRANCH_DARCY
    assert geom is not None
    assert geom.guard_reason == ''
    # The front is bracketed by the two thresholds and has positive thickness.
    assert geom.r_base < geom.r_top
    assert geom.thickness == pytest.approx(geom.r_top - geom.r_base, rel=1e-12)
    assert 0.0 < geom.porosity_top <= 0.5
    assert geom.delta_rho == pytest.approx(330.0, rel=1e-12)

    # Raising the transition moves the top of the front outward on a fixed
    # profile, which is what ties the trapping front to the solver's rheology.
    higher, higher_branch = locate_front(
        r, phi, por, rho_s, rho_l, rfront_loc=0.8, phi_min=0.01
    )
    assert higher.r_top > geom.r_top
    # That front spans about 79% of the mantle. By default it is integrated like
    # any other, because a thicker front only lengthens the residence time; the
    # thin-front limit returns only when max_front_fraction is set below it.
    assert higher_branch == BRANCH_DARCY
    assert higher.thickness > 0.7 * (r[-1] - r[0])
    _, limited = locate_front(
        r, phi, por, rho_s, rho_l, rfront_loc=0.8, phi_min=0.01, max_front_fraction=0.5
    )
    assert limited == BRANCH_GUARD

    # A fully molten column has no front at all.
    none_geom, none_branch = locate_front(
        r, np.ones(n), np.ones(n), rho_s, rho_l, rfront_loc=0.5, phi_min=0.01
    )
    assert none_geom is None
    assert none_branch == BRANCH_NONE

    # Too few nodes to resolve: the guard fires and still reports a geometry so
    # the caller can take the upper bound.
    thin_por = np.where((phi < 0.5) & (phi > 0.45), 0.4, 0.0)
    thin_geom, thin_branch = locate_front(
        r, phi, thin_por, rho_s, rho_l, rfront_loc=0.5, phi_min=0.01, n_front_min=5
    )
    assert thin_branch == BRANCH_GUARD
    assert thin_geom is not None
    assert 'fewer than the 5 needed' in thin_geom.guard_reason

    # Melt denser than the solid drains downward, so the upward-drainage
    # picture does not apply and the guard fires.
    dense_geom, dense_branch = locate_front(
        r, phi, por, np.full(n, 2900.0), rho_l, rfront_loc=0.5, phi_min=0.01
    )
    assert dense_branch == BRANCH_GUARD
    assert dense_geom.dense_melt
    assert 'denser than the solid' in dense_geom.guard_reason

    # A front spanning more of the mantle than the thin-front picture allows.
    thick_geom, thick_branch = locate_front(
        r, phi, por, rho_s, rho_l, rfront_loc=0.5, phi_min=0.01, max_front_fraction=0.01
    )
    assert thick_branch == BRANCH_GUARD
    assert 'above the 1% a thin front allows' in thick_geom.guard_reason

    # Two separate porous layers: the single-front picture cannot say which
    # one the crystallising parcel crosses. Nodes 8 and 9 are closed, leaving
    # porous runs at nodes 1 to 7 and 10 to 19.
    split_por = por.copy()
    split_por[8:10] = 0.0
    split_geom, split_branch = locate_front(
        r, phi, split_por, rho_s, rho_l, rfront_loc=0.5, phi_min=0.01
    )
    assert split_branch == BRANCH_GUARD
    assert 'split into 2 separate layers' in split_geom.guard_reason


@pytest.mark.physics_invariant
def test_a_front_still_porous_at_the_lowest_node_rests_on_the_core_mantle_boundary():
    """A mantle crystallising from the bottom up is still porous at its lowest
    node while the rheological front rises through it. The front then rests on
    the impermeable core-mantle boundary and is integrated with that boundary
    as its base. It is refused only if it also reaches the surface node, and a
    boundary placed inside the mesh is rejected as an inconsistent geometry."""
    n = 40
    r_floor = 3.0e6  # core-mantle boundary [m]
    dr = 75.0e3  # uniform cell size [m]
    r = r_floor + dr * (np.arange(n) + 0.5)  # staggered nodes, half a cell above it
    # Melt fraction rises outward from 0.3 at the base, so the lowest node is
    # already mush while the porosity there is far above the floor of 0.01.
    phi = np.linspace(0.3, 1.0, n)
    por = phi.copy()
    rho_s = np.full(n, 3300.0)
    rho_l = np.full(n, 2970.0)

    geom, branch = locate_front(
        r, phi, por, rho_s, rho_l, rfront_loc=0.5, phi_min=0.01, r_floor=r_floor
    )
    assert branch == BRANCH_DARCY
    assert geom.guard_reason == ''
    assert int(geom.index[0]) == 0
    # The melt fraction crosses 0.5 at k* = 0.2 * 39 / 0.7 = 11.143 node
    # spacings, so the top sits at r_floor + dr * (k* + 0.5).
    k_star = 0.2 * 39.0 / 0.7
    r_top = r_floor + dr * (k_star + 0.5)
    assert geom.r_top == pytest.approx(r_top, rel=1e-12)
    # The base is the boundary itself. Taking the lowest node instead would
    # shorten the front by half a cell, 37.5 km of an 873 km front.
    assert geom.r_base == pytest.approx(r_floor, rel=1e-12)
    assert geom.thickness == pytest.approx(r_top - r_floor, rel=1e-12)
    assert 8.0e5 < geom.thickness < 9.0e5

    # With no boundary given, the lowest node stands in for it.
    near, near_branch = locate_front(r, phi, por, rho_s, rho_l, rfront_loc=0.5, phi_min=0.01)
    assert near_branch == BRANCH_DARCY
    assert near.r_base == pytest.approx(r[0], rel=1e-12)
    assert geom.thickness - near.thickness == pytest.approx(0.5 * dr, rel=1e-9)

    # Error contract: a boundary above the lowest node would put mantle nodes
    # inside the core.
    with pytest.raises(ValueError, match='lies above the lowest node'):
        locate_front(
            r, phi, por, rho_s, rho_l, rfront_loc=0.5, phi_min=0.01, r_floor=r[0] + 1.0
        )

    # Edge case: mush from the boundary to the surface node has no top to
    # integrate from, so it keeps the guard even though its base is allowed.
    mush = np.full(n, 0.4)
    top_geom, top_branch = locate_front(
        r, mush, mush, rho_s, rho_l, rfront_loc=0.5, phi_min=0.01, r_floor=r_floor
    )
    assert top_branch == BRANCH_GUARD
    assert top_geom.guard_reason == 'the front reaches the surface node'


def _two_phase_density(phi_s, rho_s, rho_l):
    """Mush density the interior solver builds: the two phase volumes add up."""
    return 1.0 / (phi_s / rho_l + (1.0 - phi_s) / rho_s)


@pytest.mark.physics_invariant
def test_porosity_and_mass_conversion_follow_the_density_lever_rule():
    """Inside the mush the porosity is the lever rule on density between the
    two end members, which for the solver's two-phase mixture is the melt
    volume fraction exactly, whichever phase is denser and down to a mush with
    one percent melt. Fully solid and fully molten nodes take their melt
    fraction. The trapped fraction then converts from the volume fraction the
    compaction physics uses to the mass fraction the volatile budget needs."""
    phi_s = np.array([0.01, 0.2, 0.37])
    # Melt lighter than the solid, as through most of the mantle, and denser
    # than it, as the shallow phase-boundary tables have it at 1.9 GPa.
    for rho_s, rho_l in ((3300.0, 2970.0), (3084.0, 3456.0)):
        s, l = np.full(3, rho_s), np.full(3, rho_l)
        rho = _two_phase_density(phi_s, s, l)
        por = porosity_from_densities(rho, s, l, phi_s)
        np.testing.assert_allclose(por, phi_s * rho / l, rtol=1e-12)
        # A lighter melt fills more volume than its mass share, a denser less.
        assert np.all((por > phi_s) == (rho_l < rho_s))
    # Discrimination: the floored denominator read the dense-melt mush as solid
    # rock, even at 37% melt.
    floored = np.clip((3084.0 - rho) / 1.0e-6, 0.0, 1.0)
    np.testing.assert_allclose(floored, 0.0, atol=0.0)
    # One percent melt by mass is 0.89 percent by volume here, not zero.
    assert 0.008 < por[0] < 0.01

    # Fully molten node at 5.05 GPa from the Earth-analogue run: a melt hotter
    # than the phase boundary is lighter than the boundary solid there, so the
    # lever rule would read it as solid. Its porosity is 1 by definition.
    molten = porosity_from_densities(
        np.array([3147.3]), np.array([3152.0]), np.array([3181.0]), np.array([1.0])
    )
    assert molten[0] == pytest.approx(1.0, rel=1e-12)
    assert (3152.0 - 3147.3) / (3152.0 - 3181.0) < 0.0
    # Fully solid node where the melt is denser: a solid colder than the phase
    # boundary is denser than the boundary solid, and the lever rule would give
    # it a spurious 26/372 = 0.07 of porosity. Its porosity is 0.
    solid = porosity_from_densities(
        np.array([3110.0]), np.array([3084.0]), np.array([3456.0]), np.array([0.0])
    )
    assert solid[0] == pytest.approx(0.0, abs=0.0)
    assert (3084.0 - 3110.0) / (3084.0 - 3456.0) > 0.05

    # Edge case: equal phase-boundary densities make the lever rule 0/0; mass
    # and volume fractions coincide there, so the melt fraction is returned.
    same = porosity_from_densities(
        np.array([3300.0]), np.array([3300.0]), np.array([3300.0]), np.array([0.4])
    )
    assert same[0] == pytest.approx(0.4, rel=1e-12)
    # Boundedness: a mush density outside the end members cannot give a
    # porosity outside [0, 1].
    out = porosity_from_densities(
        np.array([1000.0, 5000.0]),
        np.full(2, 3300.0),
        np.full(2, 2970.0),
        np.full(2, 0.5),
    )
    assert np.all((out >= 0.0) & (out <= 1.0))

    # The melt is lighter, so its mass fraction is below its volume fraction.
    mass = volume_to_mass_fraction(0.3, 2970.0, 3300.0)
    assert mass == pytest.approx(0.3 * 2970.0 / (0.3 * 2970.0 + 0.7 * 3300.0), rel=1e-12)
    assert mass < 0.3
    # About 0.93 of the volume fraction at a ten percent contrast.
    assert mass / 0.3 == pytest.approx(0.928, rel=0.01)
    # Limits: all melt or no melt convert to themselves whatever the densities.
    assert volume_to_mass_fraction(1.0, 2970.0, 3300.0) == pytest.approx(1.0, rel=1e-12)
    assert volume_to_mass_fraction(0.0, 2970.0, 3300.0) == pytest.approx(0.0, abs=1e-30)


@pytest.mark.physics_invariant
def test_a_mush_holding_dense_melt_is_porous_and_takes_the_dense_melt_guard():
    """Near the surface the phase-boundary tables make the melt denser than the
    solid. A mush there is still porous. Read as solid rock, it cut the front
    short at the last node with lighter melt, and the dense melt never reached
    the guard. With the porosity it has, the front runs on to the surface and
    takes the guard branch, naming both causes."""
    n = 40
    r = np.linspace(3.0e6, 6.0e6, n)
    # The whole column is mush below the transition, as in the Earth-analogue
    # run once no node exceeds rfront_loc; the top four nodes carry the
    # dense-melt tables of the shallow mantle.
    phi_s = np.linspace(0.0, 0.45, n)
    rho_s, rho_l = np.full(n, 4000.0), np.full(n, 3600.0)
    rho_s[-4:], rho_l[-4:] = 3084.0, 3456.0
    rho = _two_phase_density(phi_s, rho_s, rho_l)
    por = porosity_from_densities(rho, rho_s, rho_l, phi_s)
    np.testing.assert_allclose(por, phi_s * rho / rho_l, rtol=1e-12)
    assert np.all(por[-4:] > 0.35)

    geom, branch = locate_front(r, phi_s, por, rho_s, rho_l, rfront_loc=0.5, phi_min=0.01)
    assert branch == BRANCH_GUARD
    assert geom.dense_melt
    assert 'denser than the solid' in geom.guard_reason
    assert 'reaches the surface node' in geom.guard_reason

    # Discrimination: the floored denominator read the dense nodes as solid,
    # which ended the front four nodes below the surface and integrated it as
    # an ordinary front.
    drho = rho_s - rho_l
    floored = np.clip((rho_s - rho) / np.where(drho > 1.0e-6, drho, 1.0e-6), 0.0, 1.0)
    old_geom, old_branch = locate_front(
        r, phi_s, floored, rho_s, rho_l, rfront_loc=0.5, phi_min=0.01
    )
    assert old_branch == BRANCH_DARCY
    assert int(old_geom.index[-1]) == n - 5

    # Edge case: dense melt above the transition is too molten to be part of
    # the front, so the front still ends at the transition and is integrated.
    hot = np.linspace(0.0, 0.9, n)
    hot_por = porosity_from_densities(_two_phase_density(hot, rho_s, rho_l), rho_s, rho_l, hot)
    hot_geom, hot_branch = locate_front(
        r, hot, hot_por, rho_s, rho_l, rfront_loc=0.5, phi_min=0.01
    )
    assert hot_branch == BRANCH_DARCY
    assert not hot_geom.dense_melt
    assert int(hot_geom.index[-1]) < n - 4


def test_branch_codes_are_distinct_and_cover_the_reported_regimes():
    """Every drainage outcome a run can report has its own code, so the branch
    column distinguishes a percolation-limited step from a matrix-limited one
    and both from the guard the caller falls back to."""
    codes = (BRANCH_NONE, BRANCH_DARCY, BRANCH_MATRIX, BRANCH_GUARD)
    assert len(set(codes)) == len(codes)
    assert BRANCH_NONE == 0
    # The two drainage branches are adjacent and distinct from the guard, which
    # is what lets a run be summarised by the fraction of mass on each.
    assert BRANCH_MATRIX != BRANCH_DARCY
    assert BRANCH_GUARD not in (BRANCH_DARCY, BRANCH_MATRIX)
