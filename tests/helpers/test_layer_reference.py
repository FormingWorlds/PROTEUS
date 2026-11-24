"""Unit tests for ``tests/helpers/_layer_reference.py``.

The continuous drainage of one freezing mush node is the reference the layer
scheme's freezing nodes are checked against, so it is checked here against
the closed forms its two regimes have at a constant drainage rate:

* with crystals that take none of the volatiles, the pore melt's volatiles
  fall as ``exp(-r t)`` in either regime and across the switch between them,
* with crystals that keep the melt's concentration, the volatiles follow the
  melt volume, and the drained share is ``r`` times the time integral of the
  falling porosity,
* without drainage, the node keeps all it buried.

Testing standards: docs/How-to/testing.md,
docs/Explanations/test_framework.md
"""

from __future__ import annotations

import numpy as np
import pytest
from _layer_reference import kept_share

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_T = 1.0e10  # time over which the porosity falls to zero [s]
_PHI0 = 0.3
_FLOOR = 0.01
_ENTRY = 0.5
_T_STOP = _T * (1.0 - _FLOOR / _PHI0)


def test_a_constant_rate_drains_the_volatiles_exponentially_in_either_regime():
    """At a constant relative rate r the pore volatiles fall as exp(-r t),
    whether the porosity caps the melt throughout (r T = 0.5), or the melt
    first drains below the porosity and is caught by it later (r T = 5), up
    to the time the porosity reaches the floor. A switch that lost or doubled
    a stretch of time would miss by more than the tolerance, and so would a
    node locked at the end of the fall instead of at the floor."""
    for r_t in (0.5, 5.0):
        rate = r_t / _T
        kept = kept_share(lambda f: rate, _PHI0, _T, _FLOOR, _ENTRY)
        assert kept == pytest.approx(np.exp(-rate * _T_STOP), rel=1e-8), r_t
        assert abs(kept / np.exp(-r_t) - 1.0) > 1e-2


def test_crystals_that_keep_the_melt_concentration_leave_the_volatiles_with_the_melt():
    """With D = 1 the crystallised melt keeps its volatiles: the pore volatiles
    follow the melt volume, which the porosity caps at phi0 (1 - t / T), and
    the drained share is r (phi0 / entry) (t - t^2 / 2T) at the floor. With
    D = 0 the same node keeps far less, since the pore melt that is left
    carries the crystallised melt's volatiles out with it."""
    rate = 0.5 / _T
    kept = kept_share(lambda f: rate, _PHI0, _T, _FLOOR, _ENTRY, partition=1.0)
    drained = rate * (_PHI0 / _ENTRY) * (_T_STOP - _T_STOP**2 / (2.0 * _T))
    assert kept == pytest.approx(1.0 - drained, rel=1e-8)
    passed_on = kept_share(lambda f: rate, _PHI0, _T, _FLOOR, _ENTRY, partition=0.0)
    assert passed_on < kept - 0.15


def test_a_node_that_does_not_drain_keeps_everything_it_buried():
    """Without drainage the volatiles stay at the node, in the pore melt or in
    its crystals, whatever share the crystals take."""
    for d in (0.0, 0.0017, 1.0):
        kept = kept_share(lambda f: 0.0, _PHI0, _T, _FLOOR, _ENTRY, partition=d)
        assert kept == pytest.approx(1.0, rel=1e-12), d
