"""Unit tests for the partial giant-impact re-melt rule.

Source: ``proteus.interior_energetics.remelt``. The rule raises the outermost
mantle shells that hold the melted mass fraction f to the initial condition.

Invariants checked:
  - melted mass closure: sum(w m) = f M, with w in [0, 1]
  - the melt fills from the surface down, whatever order the cells come in
  - f = 1 gives the full re-melt bit for bit, f = 0 leaves the state unchanged
  - the re-melt never cools a cell or the scalar mantle
  - f outside [0, 1], a non-finite f and invalid cell masses are refused
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from proteus.interior_energetics.remelt import (
    REMELT_CLOSURE_RTOL,
    remelt_entropy,
    remelt_temperature,
    remelt_weights,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Base first, as the Aragog staggered mesh stores them; unequal masses so a
# base-first fill and a surface-first fill give different weights.
MASS_BASE_FIRST = np.array([5.0, 3.0, 2.0])
RADII_BASE_FIRST = np.array([1.0e6, 2.0e6, 3.0e6])


@pytest.mark.physics_invariant
def test_weights_fill_the_outer_shells_first():
    """f = 0.3 of 10 kg melts the 2 kg surface cell and a third of the 3 kg cell above the base."""
    w = remelt_weights(0.3, MASS_BASE_FIRST, RADII_BASE_FIRST)
    np.testing.assert_allclose(w, [0.0, 1.0 / 3.0, 1.0], rtol=0, atol=1e-15)
    assert float(np.sum(w * MASS_BASE_FIRST)) == pytest.approx(3.0, rel=1e-14)
    # A base-first fill would melt 3 kg of the 5 kg base cell instead.
    assert abs(w[0] - 0.6) > 0.5


@pytest.mark.physics_invariant
def test_weights_follow_the_radii_not_the_array_order():
    """Reversing the cell order with its radii gives the reversed weights."""
    w = remelt_weights(0.3, MASS_BASE_FIRST, RADII_BASE_FIRST)
    w_rev = remelt_weights(0.3, MASS_BASE_FIRST[::-1], RADII_BASE_FIRST[::-1])
    np.testing.assert_array_equal(w_rev, w[::-1])
    np.testing.assert_equal(w_rev[0], 1.0)  # the surface cell is now first


def test_weights_at_the_ends_are_exact():
    """f = 1 melts every cell in full and f = 0 none, with no round-off."""
    ones = remelt_weights(1.0, MASS_BASE_FIRST, RADII_BASE_FIRST)
    zeros = remelt_weights(0.0, MASS_BASE_FIRST, RADII_BASE_FIRST)
    assert ones.tolist() == [1.0, 1.0, 1.0]
    assert zeros.tolist() == [0.0, 0.0, 0.0]


@pytest.mark.physics_invariant
def test_weights_close_the_melted_mass_over_the_input_domain():
    """Closure, bounds and surface-first order hold for any masses, radii order and f."""
    hypothesis = pytest.importorskip('hypothesis')
    st = hypothesis.strategies

    @hypothesis.settings(derandomize=True, deadline=None, max_examples=300)
    @hypothesis.given(
        masses=st.lists(st.floats(1e15, 1e24), min_size=1, max_size=200),
        f=st.one_of(
            st.floats(0.0, 1.0),
            st.sampled_from([0.0, 1.0, 5e-324, 1e-300, math.nextafter(1.0, 0.0)]),
        ),
        seed=st.integers(0, 2**32 - 1),
    )
    def check(masses, f, seed):
        m = np.asarray(masses)
        radii = np.random.default_rng(seed).permutation(m.size).astype(float) + 1.0
        w = remelt_weights(f, m, radii)
        assert w.shape == m.shape and np.all((w >= 0.0) & (w <= 1.0))
        # A subnormal weight carries few bits, hence the 1-ulp-of-zero term.
        tol = REMELT_CLOSURE_RTOL * f * m.sum() + 5e-324 * m.max()
        assert abs(np.sum(w * m) - f * m.sum()) <= tol
        # From the surface down the weights never rise.
        assert np.all(np.diff(w[np.argsort(-radii, kind='stable')]) <= 0.0)

    check()


@pytest.mark.parametrize(
    'f', [math.nan, math.inf, -1e-12, 1.0 + 1e-12, True, None, '0.5'], ids=repr
)
def test_weights_refuse_a_fraction_outside_the_unit_interval(f):
    """A melt fraction that is not a finite real number in [0, 1] is refused."""
    with pytest.raises(ValueError, match='melt fraction'):
        remelt_weights(f, MASS_BASE_FIRST, RADII_BASE_FIRST)
    with pytest.raises(ValueError, match='melt fraction'):
        remelt_temperature(2000.0, 4000.0, f)


@pytest.mark.parametrize(
    ('masses', 'radii'),
    [
        ([5.0, 0.0, 2.0], [1.0, 2.0, 3.0]),
        ([5.0, -3.0, 2.0], [1.0, 2.0, 3.0]),
        ([5.0, math.nan, 2.0], [1.0, 2.0, 3.0]),
        ([5.0, math.inf, 2.0], [1.0, 2.0, 3.0]),
        ([], []),
        ([5.0, 3.0], [1.0, 2.0, 3.0]),
        ([5.0, 3.0, 2.0], [1.0, math.nan, 3.0]),
    ],
    ids=['zero', 'negative', 'nan', 'inf', 'empty', 'shape', 'nan_radius'],
)
def test_weights_refuse_invalid_cells(masses, radii):
    """Cell masses must be finite and positive, and the radii finite and of the same shape."""
    with pytest.raises(ValueError, match='cell'):
        remelt_weights(0.5, np.asarray(masses, float), np.asarray(radii, float))
    with pytest.raises(ValueError, match='cell'):
        remelt_weights(1.0, np.asarray(masses, float), np.asarray(radii, float))


@pytest.mark.physics_invariant
def test_full_weight_is_todays_maximum_bit_for_bit():
    """w = 1 takes np.maximum(S_end, S_ic), not the blend, which can differ by 1 ulp."""
    # The last pair: S_end + 1.0 * (S_ic - S_end) lands 1 ulp above S_ic.
    S_end = np.array([2400.1, 3950.0, 2999.9, 968.4110283258282])
    S_ic = np.array([3100.3, 3100.3, 3333.3, 2001.5269857441442])
    assert S_end[3] + 1.0 * (S_ic[3] - S_end[3]) != S_ic[3]
    S_new = remelt_entropy(S_end, S_ic, np.ones(4))
    np.testing.assert_array_equal(S_new, np.maximum(S_end, S_ic))
    np.testing.assert_equal(S_new[1], 3950.0)  # the hotter cell keeps its entropy


@pytest.mark.physics_invariant
def test_partial_weight_blends_by_mass_share_and_never_cools():
    """A cell melted by a third moves a third of the way to the initial condition; w = 0 keeps S_end."""
    S_end = np.array([2400.0, 2400.0, 3950.0])
    S_ic = np.array([3000.0, 3000.0, 3000.0])
    S_new = remelt_entropy(S_end, S_ic, np.array([0.0, 1.0 / 3.0, 0.5]))
    np.testing.assert_allclose(S_new, [2400.0, 2600.0, 3950.0], rtol=1e-15)
    assert S_new[0] == S_end[0]
    assert np.all(S_new >= S_end)


@pytest.mark.physics_invariant
def test_partial_entropy_stays_between_the_end_state_and_the_full_remelt():
    """For any profiles and weights: S_end <= S_new <= max(S_end, S_ic)."""
    hypothesis = pytest.importorskip('hypothesis')
    st = hypothesis.strategies
    s = st.floats(100.0, 5000.0)

    @hypothesis.settings(derandomize=True, deadline=None, max_examples=300)
    @hypothesis.given(
        rows=st.lists(st.tuples(s, s, st.floats(0.0, 1.0)), min_size=1, max_size=100)
    )
    def check(rows):
        S_end, S_ic, w = (np.asarray(c) for c in zip(*rows))
        S_new = remelt_entropy(S_end, S_ic, w)
        assert np.all(S_new >= S_end)
        assert np.all(S_new <= np.maximum(S_end, S_ic))

    check()


@pytest.mark.parametrize('bad', [np.array([0.5, 1.5, 0.0]), np.array([0.5, np.nan, 0.0])])
def test_partial_entropy_refuses_weights_outside_the_unit_interval(bad):
    """A weight outside [0, 1] or a non-finite weight is refused before any cell changes."""
    S_end = np.full(3, 2400.0)
    with pytest.raises(ValueError, match='weight'):
        remelt_entropy(S_end, np.full(3, 3000.0), bad)
    assert S_end.tolist() == [2400.0, 2400.0, 2400.0]


@pytest.mark.physics_invariant
def test_scalar_mantle_moves_by_the_melted_share():
    """The 0D projection: T_new = T_end + f (T_init - T_end), today's max at f = 1."""
    assert remelt_temperature(2000.0, 4000.0, 0.3) == pytest.approx(2600.0, rel=1e-15)
    np.testing.assert_equal(remelt_temperature(2000.0, 4000.0, 1.0), 4000.0)
    np.testing.assert_equal(remelt_temperature(2000.0, 4000.0, 0.0), 2000.0)
    # A mantle hotter than the initial condition keeps its temperature at every f.
    np.testing.assert_equal(remelt_temperature(4500.0, 4000.0, 0.3), 4500.0)
    np.testing.assert_equal(remelt_temperature(4500.0, 4000.0, 1.0), 4500.0)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_melt_front_of_a_uniform_mantle_matches_the_closed_form(validation_plot):
    """Analytical limit: in a mantle shell of uniform density between r_c and R, melting the
    outer mass fraction f puts the melt front at r_f = (R^3 - f (R^3 - r_c^3))^(1/3).

    The front of the weights is the radius inside the partly melted cell where the mass
    above it equals w m. A fill linear in radius, r = R - f (R - r_c), misses by 280 km at
    f = 0.3, and a fill from the base up puts the front at (r_c^3 + f (R^3 - r_c^3))^(1/3).
    See ``docs/Validation/interior_energetics/remelt.md``.
    """
    r_c, R = 3.5e6, 6.4e6
    edges = np.linspace(r_c, R, 401)
    r_in, r_out = edges[:-1], edges[1:]
    mass = r_out**3 - r_in**3  # uniform density: mass is proportional to the shell volume
    f_values = np.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    front = []
    for f in f_values:
        w = remelt_weights(f, mass, 0.5 * (r_in + r_out))
        k = int(np.flatnonzero((w > 0.0) & (w < 1.0))[0])
        front.append(np.cbrt(r_out[k] ** 3 - w[k] * (r_out[k] ** 3 - r_in[k] ** 3)))
    front = np.array(front)
    closed = np.cbrt(R**3 - f_values * (R**3 - r_c**3))
    np.testing.assert_allclose(front, closed, rtol=1e-12)
    linear = R - f_values * (R - r_c)
    from_base = np.cbrt(r_c**3 + f_values * (R**3 - r_c**3))
    assert abs(front[2] - linear[2]) > 2.0e5
    # Base-up and top-down fills meet at f = 0.5 only.
    assert np.all(np.abs(front - from_base)[f_values != 0.5] > 1.0e5)

    def figure():
        import matplotlib.pyplot as plt

        f_line = np.linspace(0.0, 1.0, 51)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(f_line, np.cbrt(R**3 - f_line * (R**3 - r_c**3)) / 1e3, label='Closed form')
        ax.plot(f_line, (R - f_line * (R - r_c)) / 1e3, '--', label='Linear in r')
        ax.plot(f_values, front / 1e3, 'o', label='Weights')
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([3500, 5000, 6500])
        ax.set_xlabel('Melted mass fraction $f$')
        ax.set_ylabel('Melt front radius [km]')
        ax.legend()
        fig.tight_layout()
        return fig

    validation_plot(figure, 'interior_energetics_remelt_front')
