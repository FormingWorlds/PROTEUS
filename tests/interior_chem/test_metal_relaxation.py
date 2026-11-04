"""Tests for src/proteus/interior_chem/metal_relaxation.py, the convective
delivery time of the Fe-metal saturation step.

Invariants checked: the applied fraction 1 - exp(-dt/tau) lies in [0, 1] and
rises with dt; the mixing length matches Aragog's two profiles and vanishes
at the boundaries for 'nearest'; kappa_h / l returns the velocity that built
kappa_h; the transit time is the harmonic sum of dr / v, independent of the
order of the two end cells and of the radial ordering of the mesh, and
infinite through a cell that does not convect.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from proteus.interior_chem.metal_relaxation import (
    CONSTANT_MIXING_LENGTH_FRACTION,
    cell_velocity,
    mixing_length,
    relaxed_fraction,
    transit_time,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Unequal shell thicknesses (0.3, 0.5, 0.2, 0.4 Mm), so a sum that drops dr or
# uses a mean thickness gives a different time.
_R = np.array([3.5e6, 3.8e6, 4.3e6, 4.5e6, 4.9e6])


@pytest.mark.physics_invariant
def test_relaxed_fraction_is_one_minus_exp_and_not_linear():
    """At dt = tau the applied fraction is 1 - 1/e, not the linear dt/tau = 1
    nor one half; at dt = 1e-3 tau it approaches dt/tau."""
    f = relaxed_fraction(3.0e7, 3.0e7)
    assert f == pytest.approx(1.0 - math.exp(-1.0), rel=1e-14)
    assert abs(f - 1.0) > 0.3  # linear dt/tau would apply the full extent
    assert abs(f - 0.5) > 0.1
    small = relaxed_fraction(3.0e4, 3.0e7)
    assert small == pytest.approx(1.0e-3, rel=1e-3)


@pytest.mark.physics_invariant
def test_relaxed_fraction_limits_and_bounds():
    """tau = 0 is Schaefer's instantaneous reaction (fraction 1), dt <= 0 or an
    infinite tau applies nothing, and the fraction stays in [0, 1] and rises
    with dt for a finite tau."""
    assert relaxed_fraction(10.0, 0.0) == pytest.approx(1.0, abs=0.0)
    assert relaxed_fraction(0.0, 5.0) == pytest.approx(0.0, abs=0.0)
    assert relaxed_fraction(-1.0, 5.0) == pytest.approx(0.0, abs=0.0)
    assert relaxed_fraction(10.0, float('inf')) == pytest.approx(0.0, abs=0.0)
    fr = [relaxed_fraction(dt, 7.0) for dt in np.logspace(-6, 3, 40)]
    assert all(0.0 <= x <= 1.0 for x in fr)
    assert np.all(np.diff(fr) >= 0.0)
    assert fr[-1] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.physics_invariant
def test_nearest_mixing_length_is_the_distance_to_the_nearer_boundary():
    """'nearest' is zero at both boundaries and min(r - r_in, r_out - r)
    inside; 'constant' is a quarter of the shell thickness everywhere."""
    ell = mixing_length(_R, 'nearest')
    np.testing.assert_allclose(ell, [0.0, 0.3e6, 0.6e6, 0.4e6, 0.0], rtol=0, atol=1e-6)
    const = mixing_length(_R, 'constant')
    np.testing.assert_allclose(const, CONSTANT_MIXING_LENGTH_FRACTION * 1.4e6, rtol=1e-14)
    # Same lengths when the mesh is listed from the surface down.
    np.testing.assert_allclose(mixing_length(_R[::-1], 'nearest'), ell[::-1], atol=1e-6)


def test_unknown_mixing_length_profile_raises():
    """A profile Aragog does not build is rejected rather than guessed."""
    with pytest.raises(ValueError, match='mixing-length profile'):
        mixing_length(_R, 'quarter')
    with pytest.raises(ValueError, match='mixing-length profile'):
        cell_velocity(_R, np.ones_like(_R), 'quarter')


@pytest.mark.physics_invariant
def test_cell_velocity_recovers_the_velocity_that_built_kappa_h():
    """Aragog builds kappa_h = v * l; dividing by the same l returns v. A
    velocity linear in radius averages exactly onto the cell centres, and
    the boundary nodes (l = 0) take their neighbour's value."""
    v_nodes = 0.1 + 1.0e-7 * (_R - _R[0])  # m/s, 0.1 to 0.24
    kappa = v_nodes * mixing_length(_R, 'nearest')
    v_cell = cell_velocity(_R, kappa, 'nearest')
    expected = 0.5 * (v_nodes[:-1] + v_nodes[1:])
    expected[0] = v_nodes[1]  # CMB node copies its inner neighbour
    expected[-1] = v_nodes[-2]
    np.testing.assert_allclose(v_cell, expected, rtol=1e-12)
    # kappa itself (l missing) would be ~1e5 times larger
    assert np.all(v_cell < 1.0)


def test_cell_velocity_rejects_a_mismatched_eddy_diffusivity():
    """kappa_h on the staggered cells instead of the basic nodes is an error,
    not a silent shift by half a cell."""
    with pytest.raises(ValueError, match='basic nodes'):
        cell_velocity(_R, np.ones(_R.size - 1), 'constant')


@pytest.mark.physics_invariant
def test_transit_time_is_the_harmonic_sum_dominated_by_the_slowest_layer():
    """Cells 1-3 at 1, 1e-3 and 1 m/s: tau = 0.5e6/1 + 0.2e6/1e-3 + 0.4e6/1 s,
    which the path length over the mean velocity underestimates ~ 120 times."""
    v = np.array([5.0, 1.0, 1.0e-3, 1.0])
    tau = transit_time(_R, v, i_react=1, i_top=3)
    assert tau == pytest.approx(0.5e6 + 0.2e9 + 0.4e6, rel=1e-14)
    assert tau / (1.1e6 / np.mean(v[1:])) > 100.0
    assert transit_time(_R, v, i_react=3, i_top=1) == pytest.approx(tau, rel=1e-14)
    # Surface-down mesh: same cells, same time
    assert transit_time(_R[::-1], v[::-1], 2, 0) == pytest.approx(tau, rel=1e-14)


@pytest.mark.physics_invariant
def test_transit_time_is_infinite_through_a_non_convecting_cell():
    """A stratified cell (kappa_h = 0) on the path stops the delivery; the same
    cell outside the path does not."""
    v = np.array([0.0, 1.0, 2.0, 1.0])
    assert transit_time(_R, v, 0, 2) == float('inf')
    assert transit_time(_R, v, 1, 2) == pytest.approx(0.5e6 + 0.1e6, rel=1e-14)
    assert transit_time(_R, np.array([1.0, np.nan, 1.0, 1.0]), 0, 3) == float('inf')
