"""Unit tests for ``proteus.outgas.layer_state``.

The layer state carries the retained pore melt of every mush node and the
volatile mass locked at every node from one step to the next. These tests
exercise:

* the life of a node: entry at the transition, drainage, trapping at the floor,
  the no-drainage bound for a node that crosses the whole mush within one step,
  sealed runs, and melt the tables put above the solid density,
* the volatiles: locked at entry at that step's concentration, carried and mixed
  with the drained melt, returned when it leaves a run, capped at the melt
  inventory, and conserved between the melt, the pore melt and the buried record,
* remelting node by node: a node that becomes liquid returns its pore melt and
  its buried mass, species by species, noble gases included, while a solid node
  flickering across the floor locks nothing new,
* the assignment of trapped mass the record does not hold,
* the per-element view of the record, and the round trip of the state through
  an interior snapshot.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import netCDF4 as nc
import numpy as np
import pytest

from proteus.outgas.common import element_masses_from_species
from proteus.outgas.compaction import BRANCH_DARCY, BRANCH_GUARD, BRANCH_MATRIX
from proteus.outgas.layer_drainage import (
    STATUS_LIQUID,
    STATUS_MUSH,
    STATUS_SOLID,
    compaction_times,
    drain_runs,
    mass_fraction_of_volume,
    melt_volume_fraction,
    shell_geometry,
)
from proteus.outgas.layer_state import (
    LAYER_SPECIES,
    LayerState,
    adopt_unrecorded,
    advance_layers,
    read_state,
    species_elements,
    write_state,
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
    and is trapped with less. The state passed in is left untouched.
    """
    n = 6
    kw = _mesh_kwargs(n, melt_visc=1.0)
    x = [np.ones(n) for _ in range(4)]
    x[1][0] = 0.4  # node 0 enters the mush
    x[2][0] = 0.2  # and stays in it
    x[3][0] = 0.0  # then reaches the floor
    state, _ = advance_layers(None, time_prev=0.0, time_now=1.0, melt_fraction=x[0], **kw)
    state, entry = advance_layers(
        state, time_prev=1.0, time_now=1.0e4, melt_fraction=x[1], **kw
    )
    assert entry.n_entered == 1
    assert state.status[0] == STATUS_MUSH
    at_transition = mass_fraction_of_volume(state.psi[:1], _RHO_L, _RHO_S)[0]
    assert at_transition == pytest.approx(0.5, rel=1e-12)
    kept = state.copy()
    drained, _ = advance_layers(
        state, time_prev=1.0e4, time_now=2.0e4, melt_fraction=x[2], **kw
    )
    # The step returns a new state and leaves the one it was given as it was.
    np.testing.assert_array_equal(state.psi, kept.psi)
    assert drained.psi_entry[0] == pytest.approx(kept.psi[0], rel=1e-12)
    assert drained.psi[0] < kept.psi[0]
    state, floor = advance_layers(
        drained, time_prev=2.0e4, time_now=3.0e4, melt_fraction=x[3], **kw
    )
    assert floor.exited.tolist() == [0]
    assert 0.0 < floor.trapped[0] < 0.5
    assert floor.branch.tolist() == [BRANCH_DARCY]
    # Entered 5/6 of the way through the first step: (1 - 0.5) / (1 - 0.4).
    entry_time = 1.0 + (5.0 / 6.0) * (1.0e4 - 1.0)
    assert floor.residence[0] == pytest.approx(3.0e4 - entry_time, rel=1e-9)
    assert state.status[0] == STATUS_SOLID
    assert np.isnan(state.psi[0])


@pytest.mark.physics_invariant
def test_node_crossing_the_whole_mush_in_one_step_keeps_the_bound():
    """A node that goes from liquid to solid within one step is never drained.

    It keeps the pore melt of the transition, so it traps exactly
    ``rfront_loc`` by mass, flagged as the no-drainage bound with no resolved
    residence time.
    """
    n = 4
    kw = _mesh_kwargs(n, melt_visc=1.0)
    state, _ = advance_layers(None, time_prev=0.0, time_now=1.0, melt_fraction=np.ones(n), **kw)
    x = np.ones(n)
    x[0] = 0.0
    state, step = advance_layers(state, time_prev=1.0, time_now=1.0e6, melt_fraction=x, **kw)
    assert step.exited.tolist() == [0]
    assert step.trapped[0] == pytest.approx(0.5, rel=1e-12)
    assert step.branch.tolist() == [BRANCH_GUARD]
    assert np.isnan(step.residence[0])
    assert any('within one step' in reason for reason in step.held)


@pytest.mark.physics_invariant
def test_a_sealed_run_retains_its_melt_and_dense_melt_drains():
    """Mush capped by a solid node keeps everything; dense melt drains and is counted.

    Node 1 is mush under a solid node 2, so nothing leaves it. Node 4 is mush
    whose tabulated melt is denser than its solid, below the open ocean; it
    drains on the magnitude of the contrast, as the interior solver's own
    separation velocity does, and is reported.
    """
    n = 6
    rho_l = _uniform(n, _RHO_L)
    rho_l[4] = 4200.0
    kw = _mesh_kwargs(n, rho_melt=rho_l, melt_visc=1.0)
    x = np.array([0.0, 0.3, 0.0, 1.0, 0.3, 1.0])
    state, _ = advance_layers(None, time_prev=0.0, time_now=1.0, melt_fraction=x, **kw)
    psi_before = state.psi.copy()
    state, step = advance_layers(state, time_prev=1.0, time_now=1.0e5, melt_fraction=x, **kw)
    assert state.psi[1] == pytest.approx(psi_before[1], rel=1e-15)
    assert state.t_held[1] == pytest.approx(1.0e5 - 1.0, rel=1e-12)
    assert step.node_branch[1] == BRANCH_GUARD
    assert any('sealed' in reason for reason in step.held)
    # The dense node drained on |drho| = 200 kg/m3 like any other.
    assert state.psi[4] < 0.5 * psi_before[4]
    assert step.node_branch[4] == BRANCH_DARCY
    assert step.n_dense == 1


@pytest.mark.physics_invariant
def test_a_remelted_mush_node_returns_its_locked_volatiles_to_the_ocean():
    """A mush node that warms back above the transition gives its melt back.

    Nothing reaches the floor, and every kilogram the node locked on entry
    returns to the melt: the net move of the two steps is zero, and the return
    is counted as a remelt.
    """
    n = 3
    kw = _mesh_kwargs(n)
    liquid = {'CO2': 3.6e21}
    state, first = advance_layers(
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
    state, back = advance_layers(
        state,
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
    assert state.status[0] == STATUS_LIQUID
    assert np.isnan(state.psi[0])
    assert state.locked('CO2') == pytest.approx(0.0, abs=0.0)


@pytest.mark.physics_invariant
def test_a_solid_node_remelting_to_liquid_returns_its_buried_mass():
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
    state, first = advance_layers(
        None,
        time_prev=0.0,
        time_now=1.0,
        melt_fraction=np.array([0.3, 0.3, 1.0]),
        **_volatile_kwargs(liquid, 1.8e24, partition={}),
        **kw,
    )
    state, frozen = advance_layers(
        state,
        time_prev=1.0,
        time_now=2.0,
        melt_fraction=np.array([0.0, 0.0, 1.0]),
        **_volatile_kwargs(liquid, 1.8e24, partition={}),
        **kw,
    )
    assert frozen.exited.tolist() == [0, 1]
    buried = {sp: state.buried[sp].copy() for sp in liquid}
    # Each node buried its pore melt, 0.3 of 1e23 kg at the melt concentration.
    assert buried['Ar'][0] == pytest.approx(0.3 * 1.0e23 * 3.6e17 / 1.8e24, rel=1e-9)
    state, back = advance_layers(
        state,
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
        assert state.buried[sp][0] == pytest.approx(0.0, abs=0.0)
        assert state.buried[sp][1] == pytest.approx(buried[sp][1], rel=1e-15)
    # Remelting the rest returns the whole record and leaves nothing locked.
    state, last = advance_layers(
        state,
        time_prev=3.0,
        time_now=4.0,
        melt_fraction=np.ones(n),
        **_volatile_kwargs(liquid, 1.8e24, partition={}),
        **kw,
    )
    assert last.remelt_kg['H2O'] == pytest.approx(buried['H2O'][1], rel=1e-12)
    assert all(state.locked(sp) == pytest.approx(0.0, abs=0.0) for sp in LAYER_SPECIES)


@pytest.mark.physics_invariant
def test_first_sight_locks_the_mush_it_finds_and_nothing_else():
    """With no state, the mush found locks its pore melt at the current concentration.

    Solid nodes have no pore melt on record and liquid nodes are ocean. Without
    concentrations nothing is locked. Invalid mesh inputs are rejected.
    """
    n = 4
    kw = _mesh_kwargs(n)
    x = np.array([0.0, 0.2, 0.7, 1.0])
    state, step = advance_layers(None, time_prev=0.0, time_now=5.0, melt_fraction=x, **kw)
    assert step.exited.size == 0
    assert state.status.tolist() == [STATUS_SOLID, STATUS_MUSH, STATUS_LIQUID, STATUS_LIQUID]
    phi = melt_volume_fraction(np.array([0.2]), _RHO_S, _RHO_L)[0]
    assert state.psi[1] == pytest.approx(phi, rel=1e-12)
    assert all(state.locked(sp) == pytest.approx(0.0, abs=0.0) for sp in LAYER_SPECIES)
    locking, first = advance_layers(
        None,
        time_prev=0.0,
        time_now=5.0,
        melt_fraction=x,
        **_volatile_kwargs({'CO2': 2.0e21}, 1.0e24),
        **kw,
    )
    # The mush node's pore melt is 0.2 of its mass, at 2e-3.
    assert first.moved_kg['CO2'] == pytest.approx(0.2 * 1.0e23 * 2.0e-3, rel=1e-12)
    assert locking.retained['CO2'][0] == pytest.approx(0.0, abs=0.0)
    with pytest.raises(ValueError, match='shell boundaries'):
        advance_layers(
            None,
            time_prev=0.0,
            time_now=1.0,
            melt_fraction=x,
            **_mesh_kwargs(n, r_basic=np.arange(3.0)),
        )
    with pytest.raises(ValueError, match='rfront_loc'):
        advance_layers(
            None,
            time_prev=0.0,
            time_now=1.0,
            melt_fraction=x,
            **_mesh_kwargs(n, rfront_loc=0.0),
        )


@pytest.mark.physics_invariant
def test_element_profile_splits_species_by_stoichiometry():
    """Water locked at a node adds its hydrogen and oxygen to that node's elements.

    The per-element sums equal the whole-planet split of the same species mass,
    with 7.94 kg of oxygen for every kilogram of hydrogen in water; water still
    in pore melt counts toward its node, and a noble gas is its own element.
    """
    state = LayerState.initial(np.full(3, STATUS_SOLID), np.zeros(3), np.zeros(3), 0.0)
    state.buried['H2O'] = np.array([0.0, 2.0e18, 1.0e18])
    state.buried['CO2'] = np.array([5.0e17, 0.0, 0.0])
    whole = element_masses_from_species({'H2O': 3.0e18, 'CO2': 5.0e17})
    for element, mass in whole.items():
        assert float(np.sum(state.element_profile(element))) == pytest.approx(mass, rel=1e-12)
    profile_o, profile_h = state.element_profile('O'), state.element_profile('H')
    assert profile_o[1] / profile_h[1] == pytest.approx(7.94, rel=2e-3)
    assert profile_h[0] == pytest.approx(0.0, abs=0.0)
    state.retained['H2O'] = np.array([1.0e18, 0.0, 0.0])
    extra = element_masses_from_species({'H2O': 1.0e18})['H']
    assert state.element_profile('H')[0] == pytest.approx(extra, rel=1e-12)
    assert species_elements('Ar') == {'Ar': 1.0}
    state.buried['Ar'] = np.array([0.0, 0.0, 4.0e15])
    assert state.element_profile('Ar')[2] == pytest.approx(4.0e15, rel=1e-15)


@pytest.mark.physics_invariant
def test_unrecorded_trapped_mass_is_adopted_by_the_nodes_that_are_not_liquid():
    """Trapped mass missing from the record goes to the mush and solid nodes, by mass.

    The record holds 1e18 kg of water but the helpfile 4e18 kg: the 3e18 kg
    difference is buried on nodes 0 and 1 in proportion 1:2 of their mass, none
    on the liquid node 2. A record that already holds the trapped mass adopts
    nothing, and nothing is adopted while every node is liquid.
    """
    status = np.array([STATUS_SOLID, STATUS_MUSH, STATUS_LIQUID])
    state = LayerState.initial(status, np.array([0.0, 0.2, 1.0]), np.zeros(3), 0.0)
    state.buried['H2O'] = np.array([1.0e18, 0.0, 0.0])
    mass = np.array([1.0e23, 2.0e23, 5.0e23])
    adopted = adopt_unrecorded(state, mass, {'H2O': 4.0e18, 'CO2': 0.0})
    assert adopted == pytest.approx({'H2O': 3.0e18}, rel=1e-12)
    np.testing.assert_allclose(state.buried['H2O'], [2.0e18, 2.0e18, 0.0], rtol=1e-12)
    assert state.locked('H2O') == pytest.approx(4.0e18, rel=1e-12)
    assert adopt_unrecorded(state, mass, {'H2O': 4.0e18}) == {}
    molten = LayerState.initial(np.full(3, STATUS_LIQUID), np.ones(3), np.ones(3), 0.0)
    assert adopt_unrecorded(molten, mass, {'H2O': 4.0e18}) == {}
    assert molten.locked('H2O') == pytest.approx(0.0, abs=0.0)


@pytest.mark.physics_invariant
def test_state_round_trips_through_an_interior_snapshot(tmp_path):
    """The per-node state written beside the solver fields reads back unchanged.

    The locked mass on record, buried and in pore melt, is conserved through
    the round trip. A snapshot without the state, or sized for another mesh,
    reads as ``None``; a file without the solver's staggered dimension is
    refused.
    """
    n = 5
    path = str(tmp_path / '100p000_int.nc')
    with nc.Dataset(path, mode='w') as ds:
        ds.createDimension('staggered', n)
        ds.createVariable('time', np.float64)
        ds['time'][0] = 100.0
    assert read_state(path, n) is None
    state = LayerState.initial(
        np.array([2, 2, 1, 0, 0]),
        np.array([0.0, 0.0, 0.2, 1.0, 1.0]),
        np.zeros(n),
        50.0,
    )
    state.buried['H2O'] = np.array([1.0e18, 3.0e18, 0.0, 0.0, 0.0])
    state.retained['CO2'][2] = 4.0e17
    write_state(path, state)
    write_state(path, state)  # a second write of the same step overwrites
    back = read_state(path, n)
    assert back.status.tolist() == state.status.tolist()
    np.testing.assert_array_equal(np.isnan(back.psi), np.isnan(state.psi))
    np.testing.assert_allclose(back.psi[2], state.psi[2], rtol=0.0)
    for sp in LAYER_SPECIES:
        np.testing.assert_allclose(back.buried[sp], state.buried[sp], rtol=0.0)
        np.testing.assert_allclose(back.retained[sp], state.retained[sp], rtol=0.0)
    assert back.locked('CO2') == pytest.approx(4.0e17, rel=1e-15)
    assert back.locked('H2O') == pytest.approx(4.0e18, rel=1e-15)
    assert read_state(path, n + 1) is None
    other = str(tmp_path / 'bare.nc')
    with nc.Dataset(other, mode='w') as ds:
        ds.createDimension('basic', n + 1)
    with pytest.raises(ValueError, match='staggered'):
        write_state(other, state)


@pytest.mark.physics_invariant
def test_melt_formed_from_a_solid_node_locks_nothing_new():
    """A solid node flickering across the floor keeps its record and locks nothing.

    Node 0 freezes with its entry pore melt, then flickers across the floor ten
    times, never reaching the transition: nothing is locked, buried or returned.
    A later remelt past the transition returns its buried mass, and entering
    the mush again from above locks new ocean melt.
    """
    n = 3
    kw = _stiff_kwargs(n)
    liquid = {'CO2': 3.6e21}
    state, first = advance_layers(
        None,
        time_prev=0.0,
        time_now=1.0,
        melt_fraction=np.array([0.3, 1.0, 1.0]),
        **_volatile_kwargs(liquid, 1.8e24),
        **kw,
    )
    frozen = np.array([0.0, 1.0, 1.0])
    state, floor = advance_layers(
        state,
        time_prev=1.0,
        time_now=2.0,
        melt_fraction=frozen,
        **_volatile_kwargs(liquid, 1.8e24),
        **kw,
    )
    assert floor.trapped[0] == pytest.approx(0.3, rel=1e-9)
    buried = state.locked('CO2')
    assert buried == pytest.approx(first.moved_kg['CO2'], rel=1e-12)
    t = 2.0
    for _ in range(10):
        for melt in (np.array([0.02, 1.0, 1.0]), frozen):
            state, again = advance_layers(
                state,
                time_prev=t,
                time_now=t + 1.0,
                melt_fraction=melt,
                **_volatile_kwargs(liquid, 1.8e24),
                **kw,
            )
            assert again.moved_kg['CO2'] == pytest.approx(0.0, abs=1.0)
            t += 1.0
    assert state.locked('CO2') == pytest.approx(buried, rel=1e-12)
    state, molten = advance_layers(
        state,
        time_prev=t,
        time_now=t + 1.0,
        melt_fraction=np.ones(n),
        **_volatile_kwargs(liquid, 1.8e24),
        **kw,
    )
    assert molten.moved_kg['CO2'] == pytest.approx(-buried, rel=1e-12)
    state, entry = advance_layers(
        state,
        time_prev=t + 1.0,
        time_now=t + 2.0,
        melt_fraction=np.array([0.4, 1.0, 1.0]),
        **_volatile_kwargs(liquid, 1.8e24),
        **kw,
    )
    assert entry.moved_kg['CO2'] == pytest.approx(0.5 * 1.0e23 * 2.0e-3, rel=1e-12)


@pytest.mark.physics_invariant
def test_locked_volatiles_move_with_the_melt_and_keep_their_entry_concentration():
    """Pore melt carries the concentration it entered with, and mixes as it drains.

    Node 0 enters at 1e-3 and node 1 a step later at 3e-3. Drainage moves melt
    from node 0 into node 1 only, so node 0 keeps 1e-3 exactly and node 1 falls
    toward it; what leaves the top of the run returns to the melt. A node
    crossing the whole mush in one step buries the ocean melt of that step.
    """
    n = 3
    kw = _mesh_kwargs(n, melt_visc=1.0, mush_visc=1.0e14)

    def conc_of(state, k):
        f_entry = mass_fraction_of_volume(state.psi_entry[k], _RHO_L, _RHO_S)
        melt = f_entry * state.psi[k] / state.psi_entry[k] * 1.0e23
        return state.retained['H2O'][k] / melt

    step_kw = dict(partition={}, liquid={'H2O': 1.0e30})
    state, _ = advance_layers(
        None,
        time_prev=0.0,
        time_now=1.0,
        melt_fraction=np.ones(n),
        concentration={'H2O': 1e-3},
        **step_kw,
        **kw,
    )
    state, _ = advance_layers(
        state,
        time_prev=1.0,
        time_now=2.0,
        melt_fraction=np.array([0.4, 1.0, 1.0]),
        concentration={'H2O': 1e-3},
        **step_kw,
        **kw,
    )
    state, _ = advance_layers(
        state,
        time_prev=2.0,
        time_now=3.0,
        melt_fraction=np.array([0.3, 0.4, 1.0]),
        concentration={'H2O': 3e-3},
        **step_kw,
        **kw,
    )
    assert conc_of(state, 0) == pytest.approx(1e-3, rel=1e-9)
    assert conc_of(state, 1) == pytest.approx(3e-3, rel=1e-9)
    before = float(np.sum(state.retained['H2O']))
    state, drained = advance_layers(
        state,
        time_prev=3.0,
        time_now=1.0e3,
        melt_fraction=np.array([0.2, 0.3, 1.0]),
        concentration={'H2O': 5e-3},
        **step_kw,
        **kw,
    )
    assert conc_of(state, 0) == pytest.approx(1e-3, rel=1e-9)
    assert 1e-3 < conc_of(state, 1) < 3e-3
    # What left the run is exactly what the pore melt lost.
    assert drained.released_kg['H2O'] == pytest.approx(
        before - float(np.sum(state.retained['H2O'])), rel=1e-9
    )
    assert drained.released_kg['H2O'] > 0.0
    assert drained.remelt_kg['H2O'] == pytest.approx(0.0, abs=0.0)
    state, floor = advance_layers(
        state,
        time_prev=1.0e3,
        time_now=2.0e3,
        melt_fraction=np.array([0.2, 0.3, 0.0]),
        concentration={'H2O': 5e-3},
        **step_kw,
        **kw,
    )
    assert floor.exited.tolist() == [2]
    assert floor.locked_kg['H2O'] == pytest.approx(0.5 * 1.0e23 * 5e-3, rel=1e-12)


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
    state, t = None, 0.0
    for k in range(1, 12):
        state, step = advance_layers(
            state,
            time_prev=t,
            time_now=t + 5.0e4,
            melt_fraction=profile(k),
            **_volatile_kwargs(liquid, 1.0e24, crystallised=1.0e23),
            **kw,
        )
        for sp in liquid:
            assert step.moved_kg[sp] == pytest.approx(
                step.locked_kg[sp] - step.released_kg[sp], rel=1e-12, abs=1.0
            )
            liquid[sp] -= step.moved_kg[sp]
            assert liquid[sp] >= -1.0
        t += 5.0e4
    for sp in total:
        assert state.locked(sp) + liquid[sp] == pytest.approx(total[sp], rel=1e-12)
    # Edge case: a melt too small for the front caps every lock at what it holds.
    capped_state, capped = advance_layers(
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
    assert capped_state.locked('CO2') == pytest.approx(1.0e18, rel=1e-12)


def test_a_step_leaves_every_array_of_its_input_state_untouched():
    """The state passed in is not modified, in any of its arrays or records."""
    n = 6
    kw = _mesh_kwargs(n, melt_visc=1.0)
    x = np.array([0.3, 0.2, 0.0, 0.3, 0.45, 1.0])
    state, _ = advance_layers(
        None,
        time_prev=0.0,
        time_now=1.0,
        melt_fraction=x,
        **_volatile_kwargs({'CO2': 2e21}, 1e24),
        **kw,
    )
    state.buried['H2O'][:] = 1.0e17
    kept = state.copy()
    x2 = np.array([0.0, 0.1, 0.0, 0.0, 0.3, 1.0])
    advance_layers(
        state,
        time_prev=1.0,
        time_now=1.0e4,
        melt_fraction=x2,
        **_volatile_kwargs({'CO2': 1e21}, 1e24),
        **kw,
    )
    names = ('status', 'psi', 'psi_entry', 't_entry', 'melt_fraction', 't_darcy', 't_matrix')
    for name in names:
        np.testing.assert_array_equal(getattr(state, name), getattr(kept, name))
    np.testing.assert_array_equal(state.t_held, kept.t_held)
    for sp in LAYER_SPECIES:
        np.testing.assert_array_equal(state.buried[sp], kept.buried[sp])
        np.testing.assert_array_equal(state.retained[sp], kept.retained[sp])


@pytest.mark.physics_invariant
def test_a_sealed_node_reaching_the_floor_keeps_all_its_melt():
    """A node under a solid layer cannot drain, so it freezes with its entry melt.

    It is reported on the no-drainage bound, and a sub-step cap that stops the
    integration early leaves more melt behind than the full integration.
    """
    n = 4
    kw = _mesh_kwargs(n, melt_visc=1.0)
    x = np.array([0.3, 0.0, 1.0, 1.0])
    state, _ = advance_layers(None, time_prev=0.0, time_now=1.0, melt_fraction=x, **kw)
    state, floor = advance_layers(
        state, time_prev=1.0, time_now=1.0e4, melt_fraction=np.array([0.0, 0.0, 1.0, 1.0]), **kw
    )
    assert floor.exited.tolist() == [0]
    assert floor.trapped[0] == pytest.approx(0.3, rel=1e-9)
    assert floor.branch.tolist() == [BRANCH_GUARD]
    # Cap the sub-steps of an open column: it drains less than when integrated.
    from proteus.outgas.compaction import darcy_velocity

    r = np.linspace(4.0e6, 4.5e6, 3)
    volume, area, centre = shell_geometry(r)
    tau = compaction_times([np.arange(2)], r, centre, _uniform(2, _DRHO), _uniform(2, _G), 1e19)

    def speed_of(p, idx):
        return darcy_velocity(p, _GRAIN, _DRHO, _G, 1.0, buoyancy='mixture')

    args = (np.array([0.4, 0.4]), [np.arange(2)], volume, area, speed_of, tau)
    full = drain_runs(*args, 1.0e4 * secs_per_year)
    capped = drain_runs(*args, 1.0e4 * secs_per_year, max_substeps=1)
    assert capped.n_substeps == 1
    assert capped.elapsed < full.elapsed
    assert np.sum(capped.psi * volume) > np.sum(full.psi * volume)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_a_matrix_limited_node_traps_the_closed_form_share_of_its_melt(validation_plot):
    """A node held in the mush, drained only by compaction, keeps psi_entry exp(-t / tau_s).

    The top shell of a 6-shell mantle enters at the transition (F = 0.5) and
    spends three matrix times ``tau_s = mu_s / (drho g d)`` in the mush, ``d``
    the depth of its centre below its top, before it freezes. A 1 cm grain and
    a melt of 0.1 Pa s keep the percolation 7 times faster than the matrix to
    the end, so it reaches the floor with ``0.5 exp(-3)`` of its mass as melt
    and buries that share of the water it locked, the closed form of the
    front scheme's matrix limit. The tolerance of 5e-3 covers the 1.7e-3 the
    second-order sub-steps of 0.058 ``tau_s`` add over 52 steps.
    """
    n, mush_visc, d_h2o = 6, 1.0e22, 0.0017
    kw = _mesh_kwargs(n, grain_size=1.0e-2, melt_visc=0.1, mush_visc=mush_visc)
    vol = dict(concentration={'H2O': 1.0e-3}, partition={'H2O': d_h2o}, liquid={'H2O': 1.0e21})
    r = kw['r_basic']
    tau = mush_visc / (_DRHO * _G * 0.5 * (r[-1] - r[-2])) / secs_per_year
    x = np.ones(n)
    state, _ = advance_layers(None, time_prev=0.0, time_now=0.0, melt_fraction=x, **kw)
    x[-1] = 0.3
    state, entry = advance_layers(
        state, time_prev=0.0, time_now=100.0, melt_fraction=x, **vol, **kw
    )
    locked = entry.locked_kg['H2O']
    assert locked == pytest.approx((1.0 - d_h2o) * 1.0e-3 * 0.5 * 1.0e23, rel=1e-12)
    times, kept, moved = [0.0], [1.0], locked
    t_prev, t_end = 100.0, 100.0 + 3.0 * tau
    while x[-1] > 0.0:
        t_now = min(t_prev + 2.0e4, t_end)
        if t_now >= t_end:
            x[-1] = 0.0
        state, step = advance_layers(
            state, time_prev=t_prev, time_now=t_now, melt_fraction=x, **vol, **kw
        )
        moved += step.moved_kg['H2O']
        assert np.all(step.node_branch[step.node_branch > 0] == BRANCH_MATRIX)
        frac = (
            step.trapped[0] / 0.5 if step.exited.size else state.psi[-1] / state.psi_entry[-1]
        )
        times.append((t_now - 100.0) / tau)
        kept.append(float(frac))
        t_prev = t_now

    exact = 0.5 * np.exp(-3.0)
    assert step.exited.tolist() == [n - 1]
    assert step.trapped[0] == pytest.approx(exact, rel=5e-3)
    assert state.buried['H2O'][-1] == pytest.approx(locked * np.exp(-3.0), rel=5e-3)
    # What the record holds is what the steps moved, to rounding.
    assert moved == pytest.approx(state.buried['H2O'][-1], rel=1e-10)
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
            t[every], np.asarray(kept)[every], 'o', mfc='none', ms=4, label='layer state'
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

    validation_plot(figure, 'layer_state_matrix_limited_node')
