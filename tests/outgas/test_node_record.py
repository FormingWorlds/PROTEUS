"""Unit tests for ``proteus.outgas.node_record``.

The layer record carries the pore melt of every mush node and the
volatile mass locked at every node from one step to the next. These tests
exercise:

* the life of a node: entry at the transition, drainage, trapping at the floor,
  the no-drainage bound for a node that crosses the whole mush within one step,
  sealed mush layers, and melt the tables put above the solid density,
* the volatiles: locked at entry at that step's concentration, carried and mixed
  with the drained melt, returned when it leaves a run, capped at the melt
  inventory, and conserved between the melt, the pore melt and the buried record,
* remelting node by node: a node that becomes liquid returns its pore melt and
  its buried mass, species by species, noble gases included, while a solid node
  flickering across the floor locks nothing new,
* the assignment of trapped mass the record does not hold.

The snapshot round trip is tested in ``test_node_record_io.py``.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.outgas.compaction import BRANCH_DARCY, BRANCH_GUARD, BRANCH_MATRIX
from proteus.outgas.layer_drainage import (
    PHASE_LIQUID,
    PHASE_MUSH,
    PHASE_SOLID,
    compaction_times,
    drain_mush_layers,
    mass_fraction_of_volume,
    melt_volume_fraction,
    node_geometry,
)
from proteus.outgas.node_record import (
    RECORD_SPECIES,
    NodeRecord,
    adopt_unrecorded,
    advance_record,
)
from proteus.utils.constants import secs_per_year

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# End-member densities with the ten percent contrast of a silicate mush.
_RHO_S = 4000.0
_RHO_L = 3600.0
_GRAIN = 1.0e-3
_DRHO = 400.0
_G = 9.8


def _uniform(n: int, value: float) -> np.ndarray:
    return np.full(n, float(value))


def _mesh_kwargs(n: int, **overrides) -> dict:
    """Profiles of a uniform mesh between a core at 3500 km and a surface at 6300 km."""
    kwargs = dict(
        rho_solid=_uniform(n, _RHO_S),
        rho_melt=_uniform(n, _RHO_L),
        gravity=_uniform(n, _G),
        r_basic=np.linspace(3.5e6, 6.3e6, n + 1),
        mass=_uniform(n, 1.0e23),
        rfront_loc=0.5,
        phi_min=0.01,
        grain_size=_GRAIN,
        melt_visc=100.0,
        mush_visc=1.0e20,
        buoyancy='mixture',
    )
    kwargs.update(overrides)
    return kwargs


def _stiff_kwargs(n: int) -> dict:
    """A mesh whose melt cannot drain: a glass-like melt and a rigid matrix."""
    return _mesh_kwargs(n, melt_visc=1.0e30, mush_visc=1.0e40)


def _volatile_kwargs(liquid: dict[str, float], melt_mass: float, **extra) -> dict:
    """Concentrations, partition and inventory of a step, from the melt reservoir."""
    kwargs = dict(
        concentration={sp: kg / melt_mass for sp, kg in liquid.items()},
        partition={'H2O': 0.0017},
        liquid=dict(liquid),
    )
    kwargs.update(extra)
    return kwargs


@pytest.mark.physics_invariant
def test_node_enters_drains_and_traps_less_than_the_transition():
    """A node crossing the transition drains until it reaches the floor.

    The node enters with the pore melt of the transition, whose mass fraction
    is exactly ``rfront_loc``; it drains over the steps it spends in the mush
    and is trapped with less. The record passed in is left untouched.
    """
    n = 6
    kw = _mesh_kwargs(n, melt_visc=1.0)
    x = [np.ones(n) for _ in range(4)]
    x[1][0] = 0.4  # node 0 enters the mush
    x[2][0] = 0.2  # and stays in it
    x[3][0] = 0.0  # then reaches the floor
    record, _ = advance_record(None, time_prev=0.0, time_now=1.0, melt_fraction=x[0], **kw)
    record, entry = advance_record(
        record, time_prev=1.0, time_now=1.0e4, melt_fraction=x[1], **kw
    )
    assert entry.n_entered == 1
    assert record.phase[0] == PHASE_MUSH
    at_transition = mass_fraction_of_volume(record.psi[:1], _RHO_L, _RHO_S)[0]
    assert at_transition == pytest.approx(0.5, rel=1e-12)
    kept = record.copy()
    drained, _ = advance_record(
        record, time_prev=1.0e4, time_now=2.0e4, melt_fraction=x[2], **kw
    )
    # The step returns a new record and leaves the one it was given as it was.
    np.testing.assert_array_equal(record.psi, kept.psi)
    assert drained.psi_entry[0] == pytest.approx(kept.psi[0], rel=1e-12)
    assert drained.psi[0] < kept.psi[0]
    record, floor = advance_record(
        drained, time_prev=2.0e4, time_now=3.0e4, melt_fraction=x[3], **kw
    )
    assert floor.exited.tolist() == [0]
    assert 0.0 < floor.trapped[0] < 0.5
    assert floor.branch.tolist() == [BRANCH_DARCY]
    # Entered 5/6 of the way through the first step: (1 - 0.5) / (1 - 0.4).
    entry_time = 1.0 + (5.0 / 6.0) * (1.0e4 - 1.0)
    assert floor.residence[0] == pytest.approx(3.0e4 - entry_time, rel=1e-9)
    assert record.phase[0] == PHASE_SOLID
    assert np.isnan(record.psi[0])


@pytest.mark.physics_invariant
def test_node_crossing_the_whole_mush_in_one_step_keeps_the_bound():
    """A node that goes from liquid to solid within one step is never drained.

    It keeps the pore melt of the transition, so it traps exactly
    ``rfront_loc`` by mass, flagged as the no-drainage bound with no resolved
    residence time.
    """
    n = 4
    kw = _mesh_kwargs(n, melt_visc=1.0)
    record, _ = advance_record(
        None, time_prev=0.0, time_now=1.0, melt_fraction=np.ones(n), **kw
    )
    x = np.ones(n)
    x[0] = 0.0
    record, step = advance_record(record, time_prev=1.0, time_now=1.0e6, melt_fraction=x, **kw)
    assert step.exited.tolist() == [0]
    assert step.trapped[0] == pytest.approx(0.5, rel=1e-12)
    assert step.branch.tolist() == [BRANCH_GUARD]
    assert np.isnan(step.residence[0])
    assert any('within one step' in reason for reason in step.held)


@pytest.mark.physics_invariant
def test_sealed_and_dense_melt_mush_hold_their_melt_and_buoyant_mush_above_drains():
    """Mush under a solid node, or at or below a dense-melt node, keeps its melt.

    Node 1 is mush under the solid node 2, so nothing leaves it. Nodes 3 to 5
    form one mush layer below the open ocean; node 4's tabulated melt is denser
    than its solid. Melt cannot rise through node 4, so it and node 3 below it
    take the no-drainage bound, while node 5 above it drains. Node 3 is
    buoyant, so a scheme that sealed only the dense node would drain it.
    """
    n = 7
    rho_l = _uniform(n, _RHO_L)
    rho_l[4] = 4200.0
    kw = _mesh_kwargs(n, rho_melt=rho_l, melt_visc=1.0)
    x = np.array([0.0, 0.3, 0.0, 0.3, 0.3, 0.3, 1.0])
    record, _ = advance_record(None, time_prev=0.0, time_now=1.0, melt_fraction=x, **kw)
    psi_before = record.psi.copy()
    record, step = advance_record(record, time_prev=1.0, time_now=1.0e5, melt_fraction=x, **kw)
    for held in (1, 3, 4):
        assert record.psi[held] == pytest.approx(psi_before[held], rel=1e-15)
        assert record.t_held[held] == pytest.approx(1.0e5 - 1.0, rel=1e-12)
        assert step.node_branch[held] == BRANCH_GUARD
    assert record.psi[5] < 0.5 * psi_before[5]
    assert step.node_branch[5] == BRANCH_DARCY
    assert any('sealed' in reason for reason in step.held)
    assert any('2 at or below dense melt' in reason for reason in step.held)
    assert step.n_dense == 1


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
def test_first_sight_buries_the_mush_it_finds_and_nothing_else():
    """With no record, the mush found locks its pore melt at the current concentration.

    Solid nodes have no pore melt on record and liquid nodes are ocean. Without
    concentrations nothing is locked. Invalid mesh inputs are rejected.
    """
    n = 4
    kw = _mesh_kwargs(n)
    x = np.array([0.0, 0.2, 0.7, 1.0])
    record, step = advance_record(None, time_prev=0.0, time_now=5.0, melt_fraction=x, **kw)
    assert step.exited.size == 0
    assert record.phase.tolist() == [PHASE_SOLID, PHASE_MUSH, PHASE_LIQUID, PHASE_LIQUID]
    phi = melt_volume_fraction(np.array([0.2]), _RHO_S, _RHO_L)[0]
    assert record.psi[1] == pytest.approx(phi, rel=1e-12)
    assert all(record.trapped(sp) == pytest.approx(0.0, abs=0.0) for sp in RECORD_SPECIES)
    locking, first = advance_record(
        None,
        time_prev=0.0,
        time_now=5.0,
        melt_fraction=x,
        **_volatile_kwargs({'CO2': 2.0e21}, 1.0e24),
        **kw,
    )
    # The mush node's pore melt is 0.2 of its mass, at 2e-3.
    assert first.moved_kg['CO2'] == pytest.approx(0.2 * 1.0e23 * 2.0e-3, rel=1e-12)
    assert locking.pore['CO2'][0] == pytest.approx(0.0, abs=0.0)
    with pytest.raises(ValueError, match='node boundaries'):
        advance_record(
            None,
            time_prev=0.0,
            time_now=1.0,
            melt_fraction=x,
            **_mesh_kwargs(n, r_basic=np.arange(3.0)),
        )
    with pytest.raises(ValueError, match='rfront_loc'):
        advance_record(
            None,
            time_prev=0.0,
            time_now=1.0,
            melt_fraction=x,
            **_mesh_kwargs(n, rfront_loc=0.0),
        )


@pytest.mark.physics_invariant
def test_unrecorded_trapped_mass_is_adopted_by_the_nodes_that_are_not_liquid():
    """Trapped mass missing from the record goes to the mush and solid nodes, by mass.

    The record holds 1e18 kg of water but the helpfile 4e18 kg: the 3e18 kg
    difference is buried on nodes 0 and 1 in proportion 1:2 of their mass, none
    on the liquid node 2. A record that already holds the trapped mass adopts
    nothing, and nothing is adopted while every node is liquid.
    """
    status = np.array([PHASE_SOLID, PHASE_MUSH, PHASE_LIQUID])
    record = NodeRecord.initial(status, np.array([0.0, 0.2, 1.0]), np.zeros(3), 0.0)
    record.crystal['H2O'] = np.array([1.0e18, 0.0, 0.0])
    mass = np.array([1.0e23, 2.0e23, 5.0e23])
    adopted = adopt_unrecorded(record, mass, {'H2O': 4.0e18, 'CO2': 0.0})
    assert adopted == pytest.approx({'H2O': 3.0e18}, rel=1e-12)
    np.testing.assert_allclose(record.crystal['H2O'], [2.0e18, 2.0e18, 0.0], rtol=1e-12)
    assert record.trapped('H2O') == pytest.approx(4.0e18, rel=1e-12)
    assert adopt_unrecorded(record, mass, {'H2O': 4.0e18}) == {}
    molten = NodeRecord.initial(np.full(3, PHASE_LIQUID), np.ones(3), np.ones(3), 0.0)
    assert adopt_unrecorded(molten, mass, {'H2O': 4.0e18}) == {}
    assert molten.trapped('H2O') == pytest.approx(0.0, abs=0.0)


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
def test_buried_volatiles_move_with_the_melt_and_keep_their_entry_concentration():
    """Pore melt carries the concentration it entered with, and mixes as it drains.

    Node 0 enters at 1e-3 and node 1 a step later at 3e-3. Drainage moves melt
    from node 0 into node 1 only, so node 0 keeps 1e-3 exactly and node 1 falls
    toward it; what leaves the top of the mush layer returns to the melt. A node
    crossing the whole mush in one step buries the ocean melt of that step.
    """
    n = 3
    kw = _mesh_kwargs(n, melt_visc=1.0, mush_visc=1.0e14)

    def conc_of(record, k):
        f_entry = mass_fraction_of_volume(record.psi_entry[k], _RHO_L, _RHO_S)
        melt = f_entry * record.psi[k] / record.psi_entry[k] * 1.0e23
        return record.pore['H2O'][k] / melt

    step_kw = dict(partition={}, liquid={'H2O': 1.0e30})
    record, _ = advance_record(
        None,
        time_prev=0.0,
        time_now=1.0,
        melt_fraction=np.ones(n),
        concentration={'H2O': 1e-3},
        **step_kw,
        **kw,
    )
    record, _ = advance_record(
        record,
        time_prev=1.0,
        time_now=2.0,
        melt_fraction=np.array([0.4, 1.0, 1.0]),
        concentration={'H2O': 1e-3},
        **step_kw,
        **kw,
    )
    record, _ = advance_record(
        record,
        time_prev=2.0,
        time_now=3.0,
        melt_fraction=np.array([0.3, 0.4, 1.0]),
        concentration={'H2O': 3e-3},
        **step_kw,
        **kw,
    )
    assert conc_of(record, 0) == pytest.approx(1e-3, rel=1e-9)
    assert conc_of(record, 1) == pytest.approx(3e-3, rel=1e-9)
    before = float(np.sum(record.pore['H2O']))
    record, drained = advance_record(
        record,
        time_prev=3.0,
        time_now=1.0e3,
        melt_fraction=np.array([0.2, 0.3, 1.0]),
        concentration={'H2O': 5e-3},
        **step_kw,
        **kw,
    )
    assert conc_of(record, 0) == pytest.approx(1e-3, rel=1e-9)
    assert 1e-3 < conc_of(record, 1) < 3e-3
    # What left the mush layer is exactly what the pore melt lost.
    assert drained.released_kg['H2O'] == pytest.approx(
        before - float(np.sum(record.pore['H2O'])), rel=1e-9
    )
    assert drained.released_kg['H2O'] > 0.0
    assert drained.remelt_kg['H2O'] == pytest.approx(0.0, abs=0.0)
    record, floor = advance_record(
        record,
        time_prev=1.0e3,
        time_now=2.0e3,
        melt_fraction=np.array([0.2, 0.3, 0.0]),
        concentration={'H2O': 5e-3},
        **step_kw,
        **kw,
    )
    assert floor.exited.tolist() == [2]
    assert floor.buried_kg['H2O'] == pytest.approx(0.5 * 1.0e23 * 5e-3, rel=1e-12)


@pytest.mark.physics_invariant
def test_volatiles_are_conserved_between_melt_pore_melt_and_record():
    """Over a rising front the melt, the pore melt and the buried record close.

    Every step's net move equals its locks minus its returns, the melt never
    goes negative, and the locks are capped at the melt inventory when the
    front would lock more than the melt holds.
    """
    n = 20
    kw = _mesh_kwargs(n)

    def profile(k):
        x = np.ones(n)
        for i in range(n):
            if i < k - 3:
                x[i] = 0.0
            elif i < k:
                x[i] = 0.4 - 0.1 * (i - (k - 3))
        return x

    liquid = {'H2O': 1.8e21, 'CO2': 3.6e21}
    total = dict(liquid)
    record, t = None, 0.0
    for k in range(1, 12):
        record, step = advance_record(
            record,
            time_prev=t,
            time_now=t + 5.0e4,
            melt_fraction=profile(k),
            **_volatile_kwargs(liquid, 1.0e24),
            **kw,
        )
        for sp in liquid:
            assert step.moved_kg[sp] == pytest.approx(
                step.buried_kg[sp] - step.released_kg[sp], rel=1e-12, abs=1.0
            )
            liquid[sp] -= step.moved_kg[sp]
            assert liquid[sp] >= -1.0
        t += 5.0e4
    for sp in total:
        assert record.trapped(sp) + liquid[sp] == pytest.approx(total[sp], rel=1e-12)
    # Edge case: a melt too small for the front caps every lock at what it holds.
    capped_state, capped = advance_record(
        None,
        time_prev=0.0,
        time_now=1.0,
        melt_fraction=profile(10),
        concentration={'CO2': 1.0},
        liquid={'CO2': 1.0e18},
        **kw,
    )
    assert 'CO2' in capped.capped
    assert capped.moved_kg['CO2'] == pytest.approx(1.0e18, rel=1e-12)
    assert capped_state.trapped('CO2') == pytest.approx(1.0e18, rel=1e-12)


def test_a_step_leaves_every_array_of_its_input_record_untouched():
    """The record passed in is not modified, in any of its arrays or records."""
    n = 6
    kw = _mesh_kwargs(n, melt_visc=1.0)
    x = np.array([0.3, 0.2, 0.0, 0.3, 0.45, 1.0])
    record, _ = advance_record(
        None,
        time_prev=0.0,
        time_now=1.0,
        melt_fraction=x,
        **_volatile_kwargs({'CO2': 2e21}, 1e24),
        **kw,
    )
    record.crystal['H2O'][:] = 1.0e17
    kept = record.copy()
    x2 = np.array([0.0, 0.1, 0.0, 0.0, 0.3, 1.0])
    advance_record(
        record,
        time_prev=1.0,
        time_now=1.0e4,
        melt_fraction=x2,
        **_volatile_kwargs({'CO2': 1e21}, 1e24),
        **kw,
    )
    names = ('phase', 'psi', 'psi_entry', 't_entry', 'melt_fraction', 't_darcy', 't_matrix')
    for name in names:
        np.testing.assert_array_equal(getattr(record, name), getattr(kept, name))
    np.testing.assert_array_equal(record.t_held, kept.t_held)
    for sp in RECORD_SPECIES:
        np.testing.assert_array_equal(record.crystal[sp], kept.crystal[sp])
        np.testing.assert_array_equal(record.pore[sp], kept.pore[sp])


@pytest.mark.physics_invariant
def test_a_sealed_node_reaching_the_floor_keeps_all_its_melt():
    """A node under a solid layer cannot drain, so it freezes with its entry melt.

    It is reported on the no-drainage bound, and a sub-step cap that stops the
    integration early leaves more melt behind than the full integration.
    """
    n = 4
    kw = _mesh_kwargs(n, melt_visc=1.0)
    x = np.array([0.3, 0.0, 1.0, 1.0])
    record, _ = advance_record(None, time_prev=0.0, time_now=1.0, melt_fraction=x, **kw)
    record, floor = advance_record(
        record,
        time_prev=1.0,
        time_now=1.0e4,
        melt_fraction=np.array([0.0, 0.0, 1.0, 1.0]),
        **kw,
    )
    assert floor.exited.tolist() == [0]
    assert floor.trapped[0] == pytest.approx(0.3, rel=1e-9)
    assert floor.branch.tolist() == [BRANCH_GUARD]
    # Cap the sub-steps of an open column: it drains less than when integrated.
    from proteus.outgas.compaction import darcy_velocity

    r = np.linspace(4.0e6, 4.5e6, 3)
    volume, area, centre = node_geometry(r)
    tau = compaction_times([np.arange(2)], r, centre, _uniform(2, _DRHO), _uniform(2, _G), 1e19)

    def speed_of(p, idx):
        return darcy_velocity(p, _GRAIN, _DRHO, _G, 1.0, buoyancy='mixture')

    args = (np.array([0.4, 0.4]), [np.arange(2)], volume, area, speed_of, tau)
    full = drain_mush_layers(*args, 1.0e4 * secs_per_year)
    capped = drain_mush_layers(*args, 1.0e4 * secs_per_year, max_substeps=1)
    assert capped.n_substeps == 1
    assert capped.elapsed < full.elapsed
    assert np.sum(capped.psi * volume) > np.sum(full.psi * volume)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_a_matrix_limited_node_traps_the_closed_form_share_of_its_melt(validation_plot):
    """A node held in the mush, drained only by compaction, keeps psi_entry exp(-t / tau_s).

    The top node of a 6-node mantle enters at the transition (F = 0.5), stays
    at that melt fraction and spends three matrix times
    ``tau_s = mu_s / (drho g d)`` in the mush, ``d`` the depth of its centre
    below its top, before it freezes. A 1 cm grain and a melt of 0.1 Pa s keep
    the percolation about 70 times faster than the matrix to the end, so it
    reaches the floor with ``0.5 exp(-3)`` of its mass as melt and locks that
    share of the water it buried, the closed form of the front scheme's matrix
    limit. The run uses the production sub-step, ``SUBSTEP_FRACTION = 0.1``.
    The tolerance of 5e-3 covers the 1.7e-3 the second-order sub-steps of
    0.058 ``tau_s`` add over 52 steps.
    """
    n, mush_visc, d_h2o = 6, 1.0e22, 0.0017
    kw = _mesh_kwargs(n, grain_size=1.0e-2, melt_visc=0.1, mush_visc=mush_visc)
    vol = dict(concentration={'H2O': 1.0e-3}, partition={'H2O': d_h2o}, liquid={'H2O': 1.0e21})
    r = kw['r_basic']
    tau = mush_visc / (_DRHO * _G * 0.5 * (r[-1] - r[-2])) / secs_per_year
    x = np.ones(n)
    record, _ = advance_record(None, time_prev=0.0, time_now=0.0, melt_fraction=x, **kw)
    # Just below the transition, the solver porosity never caps the pore melt.
    x[-1] = 0.5 - 1.0e-6
    record, entry = advance_record(
        record, time_prev=0.0, time_now=100.0, melt_fraction=x, **vol, **kw
    )
    # The pore melt buries C F m; crystals grown while the node was liquid take
    # D C m (1 - F) from the ocean.
    locked = float(record.pore['H2O'][-1])
    crystals = d_h2o * 1.0e-3 * 0.5 * 1.0e23
    assert locked == pytest.approx(1.0e-3 * 0.5 * 1.0e23, rel=1e-12)
    assert entry.buried_kg['H2O'] == pytest.approx(locked + crystals, rel=1e-12)
    times, kept, moved = [0.0], [1.0], entry.buried_kg['H2O']
    t_prev, t_end = 100.0, 100.0 + 3.0 * tau
    while x[-1] > 0.0:
        t_now = min(t_prev + 2.0e4, t_end)
        if t_now >= t_end:
            x[-1] = 0.0
        record, step = advance_record(
            record, time_prev=t_prev, time_now=t_now, melt_fraction=x, **vol, **kw
        )
        moved += step.moved_kg['H2O']
        assert np.all(step.node_branch[step.node_branch > 0] == BRANCH_MATRIX)
        frac = (
            step.trapped[0] / 0.5 if step.exited.size else record.psi[-1] / record.psi_entry[-1]
        )
        times.append((t_now - 100.0) / tau)
        kept.append(float(frac))
        t_prev = t_now

    exact = 0.5 * np.exp(-3.0)
    assert step.exited.tolist() == [n - 1]
    assert step.trapped[0] == pytest.approx(exact, rel=5e-3)
    assert record.crystal['H2O'][-1] - crystals == pytest.approx(
        locked * np.exp(-3.0), rel=5e-3
    )
    # What the record holds is what the steps moved, to rounding.
    assert moved == pytest.approx(record.crystal['H2O'][-1], rel=1e-10)
    # Discrimination: the depth of the node's base halves tau, and a forward-Euler
    # update at the same steps ends 8 percent low.
    assert abs(0.5 * np.exp(-6.0) - step.trapped[0]) > 0.5 * exact
    h = 2.0e4 / tau
    assert abs(0.5 * (1.0 - h) ** (3.0 / h) - step.trapped[0]) > 0.05 * exact

    def figure():
        import matplotlib.pyplot as plt

        t = np.asarray(times)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
        grid = np.linspace(0.0, 3.0, 40)
        every = slice(None, None, 3)
        ax1.semilogy(grid, np.exp(-grid), '-', label='exp(-t / tau_s)')
        ax1.semilogy(
            t[every], np.asarray(kept)[every], 'o', mfc='none', ms=4, label='layer record'
        )
        ax1.set_xlabel('Time in the mush / tau_s')
        ax1.set_ylabel('Retained melt / entry melt')
        ax1.legend(fontsize=8)
        dev = np.asarray(kept) * np.exp(t) - 1.0
        ax2.plot(t[every], dev[every], 'o', mfc='none', ms=4)
        ax2.plot(grid, grid * 0.058**2 / 6.0, '--', label='RK2 bias n h^3 / 6')
        ax2.axhline(5e-3, color='C2', lw=1, label='tolerance')
        ax2.set_xlabel('Time in the mush / tau_s')
        ax2.set_ylabel('Relative deviation')
        ax2.legend(fontsize=8)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'node_record_matrix_limited_node')


@pytest.mark.physics_invariant
def test_each_store_takes_the_settled_burial_of_its_transition():
    """One step from a molten mesh: node 0 cools within the liquid from 1 to
    0.7, node 1 enters the mush from 0.8 to 0.3, node 2 crosses the whole mush
    from 0.9 to the floor. Node 0's crystals take D C m 0.3 from the ocean;
    node 1 buries C F m in its pore melt plus D C m 0.3 for the crystals it
    grew while liquid; node 2 locks C F m plus D C m 0.4. A melt holding too
    little scales every new burial by the same factor."""
    n, d, c, m, f = 4, 0.0017, 1.0e-3, 1.0e23, 0.5
    kw = _stiff_kwargs(n)
    record, _ = advance_record(
        None, time_prev=0.0, time_now=1.0, melt_fraction=np.array([1.0, 0.8, 0.9, 1.0]), **kw
    )
    x = np.array([0.7, 0.3, 0.0, 1.0])
    vol = _volatile_kwargs({'H2O': c * 1.0e24}, 1.0e24)
    new, step = advance_record(
        record, time_prev=1.0, time_now=1.0e3, melt_fraction=x, **vol, **kw
    )
    assert new.crystal['H2O'][0] == pytest.approx(d * c * m * 0.3, rel=1e-12)
    assert new.pore['H2O'][1] == pytest.approx(c * f * m, rel=1e-12)
    assert new.crystal['H2O'][1] == pytest.approx(d * c * m * 0.3, rel=1e-12)
    assert new.crystal['H2O'][2] == pytest.approx(c * f * m + d * c * m * 0.4, rel=1e-12)
    assert new.trapped('H2O') == pytest.approx(step.buried_kg['H2O'], rel=1e-12)
    # Discrimination: a pore burial of (1 - D) C F m is 8.5e16 kg short.
    assert abs(new.pore['H2O'][1] - (1.0 - d) * c * f * m) > 1.0e-4 * c * f * m
    assert new.crystal['H2O'][3] == pytest.approx(0.0, abs=0.0)

    short = _volatile_kwargs({'H2O': 0.25 * step.buried_kg['H2O']}, 1.0e24)
    short['concentration'] = {'H2O': c}
    capped, cut = advance_record(
        record, time_prev=1.0, time_now=1.0e3, melt_fraction=x, **short, **kw
    )
    assert cut.capped == ['H2O']
    np.testing.assert_allclose(capped.pore['H2O'], 0.25 * new.pore['H2O'], rtol=1e-12)
    np.testing.assert_allclose(capped.crystal['H2O'], 0.25 * new.crystal['H2O'], rtol=1e-12)


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
