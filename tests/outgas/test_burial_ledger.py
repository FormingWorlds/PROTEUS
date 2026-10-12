"""Unit tests for ``proteus.outgas.burial_ledger``.

The ledger records the mass trapping buried against the interval of melt
fraction the mantle froze through, and a remelt returns the mass recorded
below the new melt fraction. These tests exercise:

* that a remelt returns the most recently buried mass first, not the mean of
  the solid,
* mass conservation: what a remelt returns plus what the ledger still holds
  equals what was buried,
* a remelt that ends inside a layer, which returns the remelted part of it,
* freezing and remelting the same interval, which cancel exactly,
* a cross-implementation check against a fine melt-fraction grid on a seeded
  random history of freezing and remelting.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.outgas.burial_ledger import Layer, build_ledger, ledger_total, remelt_to

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.physics_invariant
def test_a_remelt_returns_the_most_recently_buried_mass_first():
    """The mantle freezes from Phi = 1 to 0.7 burying 3e19 kg, then to 0.5
    burying 1e19 kg. Remelting to 0.6 returns half of the second layer, 5e18
    kg, which is the solid that froze last; returning the mean of the solid
    would give a quarter of everything, 1e19 kg."""
    phi = np.array([1.0, 0.7, 0.5])
    trapped = np.array([[0.0, 0.0], [3.0e19, 6.0e17], [4.0e19, 6.0e17]])
    layers = build_ledger(phi, trapped)
    assert [(layer.lo, layer.hi) for layer in layers] == [(0.7, 1.0), (0.5, 0.7)]
    np.testing.assert_allclose(ledger_total(layers, 2), [4.0e19, 6.0e17], rtol=1e-15)

    released = remelt_to(layers, 0.6)
    np.testing.assert_allclose(released, [5.0e18, 0.0], rtol=1e-12)
    # Discrimination: the solid-share rule returns (0.6 - 0.5) / 0.5 of it all.
    assert abs(released[0] - 0.2 * 4.0e19) > 1.0e18
    # Conservation: returned plus still held is what was buried.
    np.testing.assert_allclose(released + ledger_total(layers, 2), [4.0e19, 6.0e17], rtol=1e-15)
    assert layers[-1].lo == pytest.approx(0.6, abs=1e-15)

    # Remelting through the rest of the second layer and into the first.
    more = remelt_to(layers, 0.85)
    np.testing.assert_allclose(more, [5.0e18 + 1.5e19, 3.0e17], rtol=1e-12)
    # Edge cases: a complete remelt empties the ledger, and remelting an empty
    # ledger returns nothing.
    rest = remelt_to(layers, 1.0)
    np.testing.assert_allclose(rest, [1.5e19, 3.0e17], rtol=1e-12)
    assert layers == []
    assert remelt_to(layers, 1.0).size == 0


@pytest.mark.physics_invariant
def test_freezing_and_remelting_the_same_interval_cancel():
    """A history that cycles between Phi = 0.41 and 0.40 a hundred times,
    burying a different mass on each freeze, leaves the ledger holding exactly
    what the freeze down to 0.40 left it, since each remelt returns the layer
    the freeze before it laid down. Non-finite and unchanged melt fractions
    are steps with nothing to record."""
    phi = [1.0, 0.7, 0.41, 0.40]
    trapped = [0.0, 2.0e19, 3.0e19, 3.1e19]
    for k in range(100):
        # Remelt to 0.41 returns the top layer; the next freeze buries anew.
        trapped.append(trapped[-1] - (1.0e18 if k == 0 else 1.0e18 + 1.0e15 * k))
        phi.append(0.41)
        trapped.append(trapped[-1] + 1.0e18 + 1.0e15 * (k + 1))
        phi.append(0.40)
    phi += [float('nan'), 0.40]
    trapped += [trapped[-1], trapped[-1]]
    history = np.array(trapped)[:, None]
    layers = build_ledger(np.array(phi), history)
    assert len(layers) == 3
    np.testing.assert_allclose(ledger_total(layers, 1), [trapped[-1]], rtol=1e-12)
    np.testing.assert_allclose(layers[-1].mass, [1.0e18 + 1.0e15 * 100], rtol=1e-9)
    # Error contract: a history of one row has nothing to replay.
    assert build_ledger(np.array([0.4]), np.array([[1.0e19]])) == []


def _grid_reference(phi: np.ndarray, buried: np.ndarray, n_bins: int = 200_000):
    """Independent ledger on a fine melt-fraction grid: deposit each freeze
    uniformly over its bins, empty every bin below the melt fraction after
    each remelt. Returns the mass held per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    held = np.zeros(n_bins)
    for i in range(1, phi.size):
        lo, hi = min(phi[i - 1], phi[i]), max(phi[i - 1], phi[i])
        if phi[i] < phi[i - 1]:
            overlap = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0.0, None)
            held += buried[i] * overlap / (hi - lo)
        elif phi[i] > phi[i - 1]:
            below = np.clip(np.minimum(edges[1:], phi[i]) - edges[:-1], 0.0, None)
            held *= 1.0 - below / np.diff(edges)
    return edges, held


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_ledger_matches_a_fine_grid_over_a_random_history():
    """Cross-implementation check: a seeded history of 400 steps that mostly
    freeze and sometimes remelt, replayed by the layer ledger and by an
    independent fine-grid ledger with 2e5 bins. Both must hold the same mass
    below every melt fraction, which is what a later remelt would return."""
    rng = np.random.default_rng(42)
    steps = rng.normal(-0.002, 0.003, 400)
    phi = np.clip(1.0 + np.concatenate([[0.0], np.cumsum(steps)]), 0.05, 1.0)
    buried = np.where(
        np.diff(phi, prepend=phi[0]) < 0.0, rng.uniform(1e17, 1e18, phi.size), 0.0
    )
    # The trapped history the run would write: burials add, remelts subtract
    # what the ledger returned, which the replay ignores in favour of phi.
    trapped = np.cumsum(buried)
    layers = build_ledger(phi, trapped[:, None])
    edges, held = _grid_reference(phi, buried)
    assert np.diff(phi).max() > 0.0  # the history does remelt

    # Discretisation error of the grid: at most one bin of the densest layer.
    for cut in (phi[-1] + 0.01, 0.3, 0.6, 1.0):
        copy = [Layer(layer.lo, layer.hi, layer.mass.copy()) for layer in layers]
        from_ledger = remelt_to(copy, cut)[0] if copy else 0.0
        below = np.clip(np.minimum(edges[1:], cut) - edges[:-1], 0.0, None) / np.diff(edges)
        assert from_ledger == pytest.approx(float(np.sum(held * below)), rel=1e-3)
    assert ledger_total(layers, 1)[0] == pytest.approx(held.sum(), rel=1e-9)
    # Discrimination: the history buried more than the ledger still holds,
    # so the remelts did return mass.
    assert ledger_total(layers, 1)[0] < 0.9 * buried.sum()
