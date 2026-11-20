"""Unit tests for remelting in ``proteus.outgas.node_record``.

Remelting is followed node by node in the per-node record of the layer
scheme. These tests exercise:

* a mush or solid node that becomes liquid returning everything trapped at it,
  species by species,
* a solid node re-entering the mush from below with its own pore melt, which
  buries nothing new from the ocean,
* a liquid node returning the melted share of its crystals to the ocean, and a
  mush node moving it into its pore melt.

The helpers come from ``test_node_record.py``. See ``docs/How-to/testing.md``
and ``docs/Explanations/test_framework.md`` for the test framework.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.outgas.layer_drainage import (
    PHASE_LIQUID,
    melt_volume_fraction,
)
from proteus.outgas.node_record import (
    RECORD_SPECIES,
    advance_record,
)

from .test_node_record import (
    _mesh_kwargs,
    _stiff_kwargs,
    _volatile_kwargs,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.physics_invariant
def test_a_remelted_mush_node_returns_its_buried_volatiles_to_the_ocean():
    """A mush node that warms back above the transition gives its melt back.

    Nothing reaches the floor, and every kilogram the node locked on entry
    returns to the melt: the net move of the two steps is zero, and the return
    is counted as a remelt.
    """
    n = 3
    kw = _mesh_kwargs(n)
    liquid = {'CO2': 3.6e21}
    record, first = advance_record(
        None,
        time_prev=0.0,
        time_now=1.0,
        melt_fraction=np.array([0.3, 1.0, 1.0]),
        **_volatile_kwargs(liquid, 1.8e24),
        **kw,
    )
    locked = first.moved_kg['CO2']
    assert locked == pytest.approx(0.3 * 1.0e23 * 3.6e21 / 1.8e24, rel=1e-12)
    liquid['CO2'] -= locked
    record, back = advance_record(
        record,
        time_prev=1.0,
        time_now=2.0,
        melt_fraction=np.ones(n),
        **_volatile_kwargs(liquid, 1.8e24),
        **kw,
    )
    assert back.exited.size == 0
    assert back.n_remelted == 1
    assert back.moved_kg['CO2'] == pytest.approx(-locked, rel=1e-12)
    # Over the year before it remelts, the node drains 5e-6 of its melt out of
    # its top; that share returns as drained melt, the rest by the remelt.
    assert back.released_kg['CO2'] == pytest.approx(locked, rel=1e-12)
    assert back.remelt_kg['CO2'] == pytest.approx(locked, rel=1e-4)
    assert back.remelt_kg['CO2'] < back.released_kg['CO2']
    assert record.phase[0] == PHASE_LIQUID
    assert np.isnan(record.psi[0])
    assert record.trapped('CO2') == pytest.approx(0.0, abs=0.0)


@pytest.mark.physics_invariant
def test_a_solid_node_remelting_to_liquid_returns_its_locked_mass():
    """Remelting is followed node by node: a solid node that melts gives back what it buried.

    Node 0 freezes with its entry pore melt of water and argon; node 1 freezes
    the same way. When only node 0 becomes liquid again, exactly its own buried
    mass returns, species by species, and node 1 keeps everything. A scheme
    that kept buried mass through a full remelt would return nothing; one that
    released the whole record would return twice as much.
    """
    n = 3
    kw = _stiff_kwargs(n)
    liquid = {'H2O': 1.8e21, 'Ar': 3.6e17}
    record, first = advance_record(
        None,
        time_prev=0.0,
        time_now=1.0,
        melt_fraction=np.array([0.3, 0.3, 1.0]),
        **_volatile_kwargs(liquid, 1.8e24, partition={}),
        **kw,
    )
    record, frozen = advance_record(
        record,
        time_prev=1.0,
        time_now=2.0,
        melt_fraction=np.array([0.0, 0.0, 1.0]),
        **_volatile_kwargs(liquid, 1.8e24, partition={}),
        **kw,
    )
    assert frozen.exited.tolist() == [0, 1]
    buried = {sp: record.crystal[sp].copy() for sp in liquid}
    # Each node buried its pore melt, 0.3 of 1e23 kg at the melt concentration.
    assert buried['Ar'][0] == pytest.approx(0.3 * 1.0e23 * 3.6e17 / 1.8e24, rel=1e-9)
    record, back = advance_record(
        record,
        time_prev=2.0,
        time_now=3.0,
        melt_fraction=np.array([1.0, 0.0, 1.0]),
        **_volatile_kwargs(liquid, 1.8e24, partition={}),
        **kw,
    )
    assert back.n_remelted == 1
    for sp in liquid:
        assert back.remelt_kg[sp] == pytest.approx(buried[sp][0], rel=1e-12)
        assert back.moved_kg[sp] == pytest.approx(-buried[sp][0], rel=1e-12)
        assert record.crystal[sp][0] == pytest.approx(0.0, abs=0.0)
        assert record.crystal[sp][1] == pytest.approx(buried[sp][1], rel=1e-15)
    # Remelting the rest returns the whole record and leaves nothing locked.
    record, last = advance_record(
        record,
        time_prev=3.0,
        time_now=4.0,
        melt_fraction=np.ones(n),
        **_volatile_kwargs(liquid, 1.8e24, partition={}),
        **kw,
    )
    assert last.remelt_kg['H2O'] == pytest.approx(buried['H2O'][1], rel=1e-12)
    assert all(record.trapped(sp) == pytest.approx(0.0, abs=0.0) for sp in RECORD_SPECIES)


@pytest.mark.physics_invariant
def test_melt_formed_from_a_solid_node_buries_nothing_new_from_the_ocean():
    """A solid node flickering across the floor keeps its record and locks nothing.

    Node 0 freezes with its entry pore melt, then flickers across the floor ten
    times, never reaching the transition: nothing is locked, buried or returned.
    A later remelt past the transition returns its buried mass, and entering
    the mush again from above locks new ocean melt.
    """
    n = 3
    kw = _stiff_kwargs(n)
    liquid = {'CO2': 3.6e21}
    record, first = advance_record(
        None,
        time_prev=0.0,
        time_now=1.0,
        melt_fraction=np.array([0.3, 1.0, 1.0]),
        **_volatile_kwargs(liquid, 1.8e24),
        **kw,
    )
    frozen = np.array([0.0, 1.0, 1.0])
    record, floor = advance_record(
        record,
        time_prev=1.0,
        time_now=2.0,
        melt_fraction=frozen,
        **_volatile_kwargs(liquid, 1.8e24),
        **kw,
    )
    assert floor.trapped[0] == pytest.approx(0.3, rel=1e-9)
    buried = record.trapped('CO2')
    assert buried == pytest.approx(first.moved_kg['CO2'], rel=1e-12)
    t = 2.0
    for _ in range(10):
        for melt in (np.array([0.02, 1.0, 1.0]), frozen):
            record, again = advance_record(
                record,
                time_prev=t,
                time_now=t + 1.0,
                melt_fraction=melt,
                **_volatile_kwargs(liquid, 1.8e24),
                **kw,
            )
            assert again.moved_kg['CO2'] == pytest.approx(0.0, abs=1.0)
            if melt[0] > 0.0:
                # Re-entering from below, the node's pore melt is its own melt,
                # not the melt of the transition, about 0.53 by volume.
                own = melt_volume_fraction(melt[:1], kw['rho_solid'][:1], kw['rho_melt'][:1])
                assert record.psi[0] == pytest.approx(float(own[0]), rel=1e-12)
                assert record.psi[0] < 0.05
            t += 1.0
    assert record.trapped('CO2') == pytest.approx(buried, rel=1e-12)
    record, molten = advance_record(
        record,
        time_prev=t,
        time_now=t + 1.0,
        melt_fraction=np.ones(n),
        **_volatile_kwargs(liquid, 1.8e24),
        **kw,
    )
    assert molten.moved_kg['CO2'] == pytest.approx(-buried, rel=1e-12)
    record, entry = advance_record(
        record,
        time_prev=t + 1.0,
        time_now=t + 2.0,
        melt_fraction=np.array([0.4, 1.0, 1.0]),
        **_volatile_kwargs(liquid, 1.8e24),
        **kw,
    )
    assert entry.moved_kg['CO2'] == pytest.approx(0.5 * 1.0e23 * 2.0e-3, rel=1e-12)


@pytest.mark.physics_invariant
def test_a_liquid_node_returns_the_share_of_its_crystals_that_melts():
    """Liquid nodes cooled from 1 to 0.6 and heated back to 1 return all their
    crystals' share, as the front scheme does on a full remelt. A node cycling
    between 0.7 and 0.6 returns a quarter of its store on each warming, which
    brings it back to its value at 0.7, so the cycles do not pump water into
    the record; keeping it all would add D C m 0.1 per cycle."""
    n, d, c, m = 3, 0.0017, 1.0e-3, 1.0e23
    kw = _stiff_kwargs(n)
    vol = _volatile_kwargs({'H2O': c * 1.0e24}, 1.0e24)

    def step_to(record, x, t):
        return advance_record(
            record, time_prev=t - 1.0, time_now=t, melt_fraction=np.full(n, x), **vol, **kw
        )

    record, _ = step_to(None, 1.0, 1.0)
    record, cool = step_to(record, 0.6, 2.0)
    assert record.trapped('H2O') == pytest.approx(n * d * c * m * 0.4, rel=1e-12)
    record, warm = step_to(record, 1.0, 3.0)
    assert record.trapped('H2O') == pytest.approx(0.0, abs=1.0e-9 * n * d * c * m)
    assert warm.released_kg['H2O'] == pytest.approx(cool.buried_kg['H2O'], rel=1e-12)

    record, _ = step_to(record, 0.7, 4.0)
    start = record.trapped('H2O')
    assert start == pytest.approx(n * d * c * m * 0.3, rel=1e-12)
    for k in range(5):
        record, _ = step_to(record, 0.6, 5.0 + 2 * k)
        record, back = step_to(record, 0.7, 6.0 + 2 * k)
        assert back.released_kg['H2O'] == pytest.approx(n * d * c * m * 0.1, rel=1e-12)
    assert record.trapped('H2O') == pytest.approx(start, rel=1e-12)
    assert abs(record.trapped('H2O') - (start + 5 * n * d * c * m * 0.1)) > 0.5 * start


@pytest.mark.physics_invariant
def test_melting_crystals_in_the_mush_give_their_share_to_the_pore_melt():
    """A solid node holding CO2 re-enters the mush from below at x = 0.3 and
    moves 0.3 of its crystal store into its new pore melt, which then drains
    to the ocean above and returns CO2; a mush node warming from 0.2 to 0.4
    moves a quarter of its crystal store into its pore melt. Nothing leaves
    the record on the step the crystals melt."""
    n = 3
    kw = _mesh_kwargs(n, melt_visc=0.1)
    record, _ = advance_record(
        None, time_prev=0.0, time_now=1.0, melt_fraction=np.array([0.0, 0.2, 1.0]), **kw
    )
    record.crystal['CO2'] = np.array([4.0e17, 8.0e17, 0.0])
    before = record.trapped('CO2')
    x = np.array([0.3, 0.4, 1.0])
    melted, first = advance_record(
        record, time_prev=1.0, time_now=1.0 + 1.0e-9, melt_fraction=x, **kw
    )
    assert melted.crystal['CO2'][0] == pytest.approx(0.7 * 4.0e17, rel=1e-12)
    assert melted.pore['CO2'][0] == pytest.approx(0.3 * 4.0e17, rel=1e-12)
    assert melted.crystal['CO2'][1] == pytest.approx(0.75 * 8.0e17, rel=1e-12)
    assert melted.pore['CO2'][1] == pytest.approx(0.25 * 8.0e17, rel=1e-6)
    assert melted.trapped('CO2') == pytest.approx(before, rel=1e-6)
    assert first.released_kg['CO2'] < 1.0e-6 * before
    drained, later = advance_record(
        melted, time_prev=1.0 + 1.0e-9, time_now=1.0e4, melt_fraction=x, **kw
    )
    assert later.released_kg['CO2'] > 0.5 * (0.3 * 4.0e17 + 0.25 * 8.0e17)
    assert drained.trapped('CO2') == pytest.approx(before - later.released_kg['CO2'], rel=1e-12)
