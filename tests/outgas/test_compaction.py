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

* the permeability transcription against the interior solver's own, which is
  the only thing keeping the two from drifting apart,
* the Darcy velocity, including the matrix fraction the solver omits,
* the matrix deformation time and its independence of porosity,
* that the slower process controls, which is the sign of the combination and
  the one place an inverted comparison changes every answer,
* the trapped fractions of the published scaling table,
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

# Martian front of the published scaling table: a 60 km front at a density
# contrast of 330 kg/m3 and martian gravity, entered at the disaggregation
# melt fraction. The fast front crosses it in 86 kyr, the slow one in 1.2 Myr.
_L = 60.0e3
_DRHO = 330.0
_G = 3.711
_PHI_C = 0.3
_SECS_PER_YEAR = 3.15576e7
_T_RES_FAST = _L / (0.70 / _SECS_PER_YEAR)
_T_RES_SLOW = _L / (0.05 / _SECS_PER_YEAR)


def _solver_mobility(porosity, grain_size):
    """The interior solver's own permeability, transcribed from its source.

    ``aragog.eos.entropy_phase.EntropyPhaseEvaluator.relative_velocity`` builds
    this inline, so it cannot be imported; reproducing it here from that source
    is what makes the comparison below a real cross-check rather than the module
    agreeing with itself.
    """
    from aragog.utilities import tanh_weight

    por = np.maximum(porosity, 1.0e-20)
    one_m_por = np.maximum(1.0 - porosity, 1.0e-20)
    f_bkc = grain_size**2 * por**2 / (one_m_por**2 * 1000.0)
    f_rg = grain_size**2 * por**4.5 * (5.0 / 7.0)
    f_stokes = grain_size**2 * 2.0 / 9.0
    w_rg = tanh_weight(porosity, 0.0769452, 0.02)
    w_stokes = tanh_weight(porosity, 0.771462, 0.05)
    return np.maximum(
        (1.0 - w_rg) * f_bkc + (w_rg - w_stokes) * f_rg + w_stokes * f_stokes, 0.0
    )


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_mobility_reproduces_the_interior_solver_in_every_regime():
    """Cross-implementation check against the permeability the interior solver
    uses for gravitational separation (Bower et al. 2018 section 2.1, as
    implemented in aragog ``eos/entropy_phase.py``). The law is transcribed
    rather than imported because the solver does not expose it, so this is the
    only thing stopping the two from drifting apart."""
    pytest.importorskip('aragog')
    for grain in (1.0e-4, 1.0e-3, 1.0e-2):
        phi = np.linspace(1.0e-4, 0.99, 300)
        mine = mobility_function(phi, grain)
        theirs = _solver_mobility(phi, grain)
        np.testing.assert_allclose(mine, theirs, rtol=1e-13, atol=0.0)

    # The three regimes must actually differ, or the comparison above would
    # pass on a single-branch stub. Probe one porosity well inside each.
    f_bkc = mobility_function(0.03, 1.0e-3)
    f_rg = mobility_function(0.30, 1.0e-3)
    f_stokes = mobility_function(0.85, 1.0e-3)
    assert f_bkc < f_rg < f_stokes
    # Scale guard: deep in the Stokes regime, where the blend has converged,
    # the mobility is the grain-size limit a^2 * 2/9 = 2.2e-7 m2, five orders
    # above the low-porosity branch. A collapsed blend would not span that.
    # Probed at 0.95 rather than 0.85 because the tanh blend is only 2 percent
    # short of the limit at the lower value.
    assert mobility_function(0.95, 1.0e-3) == pytest.approx(1.0e-6 * 2.0 / 9.0, rel=2e-3)
    assert f_bkc < 1.0e-4 * f_stokes

    # Edge case: at zero porosity the mobility is negligible rather than
    # exactly zero. The smooth regime blend leaves a Stokes residue of order
    # 1e-20 m2, which the interior solver carries too (the comparison above is
    # exact there); it is twelve orders below the value at phi = 0.3, so no
    # drainage can come of it.
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
    # 430 yr for a soft mush under a 60 km martian front, the published value.
    assert tau / _SECS_PER_YEAR == pytest.approx(430.0, rel=0.02)
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
    assert stiff == pytest.approx(0.294, rel=0.02)
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


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_drainage_reproduces_the_published_scaling_table():
    """Trapped melt fractions for a 60 km martian front, against the five cases
    of Table 2 of the PROTEUS compaction note (Lichtenberg, 21 September 2026),
    which evaluates the same drainage equation with the same permeability model.

    Agreement is to a few percent rather than exactly: the table omits the
    (1-phi) matrix fraction in the percolation speed, which this module applies,
    so these values sit slightly above the published ones.
    """
    cases = (
        # grain [m], melt visc [Pa s], mush visc [Pa s], fast, slow
        (100.0e-6, 10.0, 1.0e20, 0.26, 0.17),
        (1.0e-3, 100.0, 1.0e22, 0.29, 0.23),
        (1.0e-3, 100.0, 1.0e18, 0.19, 0.11),
        (1.0e-3, 1.0, 1.0e18, 0.07, 0.02),
        (1.0e-2, 1.0, 1.0e18, 0.009, 0.003),
    )
    for grain, eta_m, eta_s, published_fast, published_slow in cases:
        fast, _, _ = drainage_integral(_PHI_C, _T_RES_FAST, _L, grain, _DRHO, _G, eta_m, eta_s)
        slow, _, _ = drainage_integral(_PHI_C, _T_RES_SLOW, _L, grain, _DRHO, _G, eta_m, eta_s)
        assert fast == pytest.approx(published_fast, abs=0.015)
        assert slow == pytest.approx(published_slow, abs=0.015)
        # A slower front always drains further, for every parameter set.
        assert slow <= fast + 1e-12

    # The table spans a factor of thirty across its rows, so a stub returning a
    # constant could not pass the loop above.
    stiff, _, _ = drainage_integral(_PHI_C, _T_RES_FAST, _L, 1e-3, _DRHO, _G, 100.0, 1e22)
    coarse, _, _ = drainage_integral(_PHI_C, _T_RES_FAST, _L, 1e-2, _DRHO, _G, 1.0, 1e18)
    assert stiff / coarse > 20.0


@pytest.mark.physics_invariant
def test_drainage_is_bounded_without_a_clamp_and_decays_with_residence():
    """The integral is bounded in [0, phi_top] by construction, because the
    porosity decays but never reaches zero: the percolation speed falls as
    phi^2 or steeper, so the last melt leaves ever more slowly. That is what
    lets the dynamic mode run without the clamp the linear law needs."""
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


@pytest.mark.physics_invariant
def test_porosity_and_mass_conversion_follow_the_density_lever_rule():
    """Porosity is the lever rule on density between the two end members, and
    the trapped fraction converts from the volume fraction the compaction
    physics uses to the mass fraction the volatile budget needs."""
    rho_s, rho_l = np.array([3300.0]), np.array([2970.0])
    # A mixture halfway between the end members is half melt by volume.
    assert porosity_from_densities(np.array([3135.0]), rho_s, rho_l)[0] == pytest.approx(
        0.5, rel=1e-12
    )
    assert porosity_from_densities(rho_s, rho_s, rho_l)[0] == pytest.approx(0.0, abs=1e-12)
    assert porosity_from_densities(rho_l, rho_s, rho_l)[0] == pytest.approx(1.0, rel=1e-12)
    # Boundedness: a density outside the end members cannot give a porosity
    # outside [0, 1].
    out = porosity_from_densities(np.array([1000.0, 5000.0]), rho_s, rho_l)
    assert np.all((out >= 0.0) & (out <= 1.0))

    # Edge case: equal end-member densities floor the denominator instead of
    # dividing by zero.
    same = porosity_from_densities(np.array([3300.0]), rho_s, rho_s)
    assert np.all(np.isfinite(same))

    # The melt is lighter, so its mass fraction is below its volume fraction.
    mass = volume_to_mass_fraction(0.3, 2970.0, 3300.0)
    assert mass == pytest.approx(0.3 * 2970.0 / (0.3 * 2970.0 + 0.7 * 3300.0), rel=1e-12)
    assert mass < 0.3
    # About 0.93 of the volume fraction at a ten percent contrast.
    assert mass / 0.3 == pytest.approx(0.928, rel=0.01)
    # Limits: all melt or no melt convert to themselves whatever the densities.
    assert volume_to_mass_fraction(1.0, 2970.0, 3300.0) == pytest.approx(1.0, rel=1e-12)
    assert volume_to_mass_fraction(0.0, 2970.0, 3300.0) == pytest.approx(0.0, abs=1e-30)


def test_branch_codes_are_distinct_and_cover_the_reported_regimes():
    """Every drainage outcome a run can report has its own code, so the branch
    column distinguishes a percolation-limited step from a matrix-limited one
    and both from the guard the caller falls back to."""
    codes = (BRANCH_NONE, BRANCH_DARCY, BRANCH_MATRIX, BRANCH_GUARD)
    assert len(set(codes)) == len(codes)
    assert BRANCH_NONE == 0
    # The two dynamic branches are adjacent and distinct from the guard, which
    # is what lets a run be summarised by the fraction of mass on each.
    assert BRANCH_MATRIX != BRANCH_DARCY
    assert BRANCH_GUARD not in (BRANCH_DARCY, BRANCH_MATRIX)
