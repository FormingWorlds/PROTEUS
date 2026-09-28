"""Unit tests for ``proteus.outgas.layer_drainage``.

The layer drainage keeps the retained pore melt of every mush node of the
interior mesh as state, drains it node by node into the node above, and traps
what a node still holds when it reaches the porosity floor. These tests
exercise:

* the melt volume fraction of a node and its inverse, the trapped mass fraction,
* the liquid, mush and solid classification at its two thresholds,
* the shell geometry of the spherical mesh,
* the face flux as the lesser of the permeability capacity and the melt that
  compaction supplies from below, against a direct evaluation of the recurrence,
* a single node against the drainage integral of the front scheme, and against
  the exponential decay of the compaction-limited branch,
* a uniform column against the analytic rarefaction of the kinematic wave, and
  its convergence under mesh refinement,
* conservation of the melt of a column plus what leaves it,
* the life of a node: entry at the transition, drainage, trapping at the floor,
  the no-drainage bound for a node that crosses the whole mush within one step,
  remelting, sealed runs, and melt the tables put above the solid density,
* the volatiles: locked at entry at that step's concentration, carried and mixed
  with the drained melt, returned when it leaves a run or its node remelts,
  buried at the floor, never locked again by melt a solid node forms, capped at
  the melt inventory, and conserved between the melt, the pore melt and the
  buried record,
* the per-element record of the buried mass, and the round trip of the state
  through an interior snapshot.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import netCDF4 as nc
import numpy as np
import pytest

from proteus.outgas.common import element_masses_from_species
from proteus.outgas.compaction import (
    BRANCH_DARCY,
    BRANCH_GUARD,
    SECS_PER_YEAR,
    darcy_velocity,
    drainage_integral,
)
from proteus.outgas.layer_drainage import (
    STATUS_LIQUID,
    STATUS_MUSH,
    STATUS_SOLID,
    LayerState,
    add_locked_mass,
    advance_layers,
    classify_nodes,
    compaction_times,
    drain_runs,
    mass_fraction_of_volume,
    melt_volume_fraction,
    read_state,
    run_fluxes,
    shell_geometry,
    write_state,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# End-member densities with the ten percent contrast of a silicate mush.
_RHO_S = 4000.0
_RHO_L = 3600.0

# Drainage inputs of the default PROTEUS mush: millimetre grains, a density
# contrast of 400 kg/m3 under 9.8 m/s2.
_GRAIN = 1.0e-3
_DRHO = 400.0
_G = 9.8


def _uniform(n: int, value: float) -> np.ndarray:
    return np.full(n, float(value))


def _speed(melt_visc: float):
    """Darcy speed of a node at uniform contrast and gravity [m s-1]."""

    def speed_of(psi, index):
        k = np.size(index)
        return darcy_velocity(psi, _GRAIN, _uniform(k, _DRHO), _uniform(k, _G), melt_visc)

    return speed_of


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
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.physics_invariant
def test_volume_fraction_follows_the_lever_rule_and_inverts_exactly():
    """The melt volume fraction weights the mass fraction by the density ratio.

    At half melt by mass and a lighter melt, the melt takes more than half the
    volume: 0.5 * 4000 / (0.5 * 4000 + 0.5 * 3600) = 0.526. Converting back
    recovers the mass fraction, which is what makes a node that entered at the
    transition and never drained trap exactly ``rfront_loc``.
    """
    x = np.array([0.0, 0.3, 0.5, 1.0])
    phi = melt_volume_fraction(x, _RHO_S, _RHO_L)
    assert phi[2] == pytest.approx(2000.0 / 3800.0, rel=1e-12)
    # Densities swapped, the most plausible error, give 0.474 instead.
    assert abs(phi[2] - 1800.0 / 3800.0) > 0.05
    # A lighter melt always takes the larger share of the volume.
    assert np.all(phi[1:3] > x[1:3])
    # The end members are fixed points.
    assert phi[0] == pytest.approx(0.0, abs=1e-15)
    assert phi[3] == pytest.approx(1.0, rel=1e-15)
    np.testing.assert_allclose(mass_fraction_of_volume(phi, _RHO_L, _RHO_S), x, rtol=1e-12)
    # Melt denser than its solid stays well defined and takes the smaller share.
    dense = melt_volume_fraction(np.array([0.5]), _RHO_L, _RHO_S)
    assert dense[0] == pytest.approx(1800.0 / 3800.0, rel=1e-12)


@pytest.mark.physics_invariant
def test_classification_puts_each_threshold_on_the_documented_side():
    """A node at the transition is liquid, and a node at the floor is solid.

    Both thresholds are inclusive on the side the solver's rheology puts them:
    ``x >= rfront_loc`` is still the magma ocean, and ``phi <= phi_min`` has no
    connected melt left to drain.
    """
    x = np.array([0.6, 0.5, 0.4999, 0.3, 0.2, 0.0])
    phi = np.array([0.62, 0.52, 0.52, 0.02, 0.01, 0.0])
    status = classify_nodes(x, phi, rfront_loc=0.5, phi_min=0.01)
    expected = [
        STATUS_LIQUID,
        STATUS_LIQUID,
        STATUS_MUSH,
        STATUS_MUSH,
        STATUS_SOLID,
        STATUS_SOLID,
    ]
    assert status.tolist() == expected
    # Raising the transition moves the boundary node into the mush.
    assert classify_nodes(x, phi, rfront_loc=0.55, phi_min=0.01)[1] == STATUS_MUSH


@pytest.mark.physics_invariant
def test_shell_geometry_sums_to_the_spherical_shell():
    """Shell volumes add up to the whole mantle, and faces grow as r squared."""
    r = np.array([3.5e6, 4.0e6, 5.5e6, 6.3e6])
    volume, area_top, centre = shell_geometry(r)
    whole = 4.0 / 3.0 * np.pi * (6.3e6**3 - 3.5e6**3)
    assert np.sum(volume) == pytest.approx(whole, rel=1e-12)
    assert area_top[-1] == pytest.approx(4.0 * np.pi * 6.3e6**2, rel=1e-12)
    # A plane-parallel estimate (area times thickness at the top face) would
    # overstate the thick middle shell by about 20 percent.
    assert abs(volume[1] - area_top[1] * 1.5e6) / volume[1] > 0.1
    assert centre[0] == pytest.approx(3.75e6, rel=1e-12)
    with pytest.raises(ValueError, match='ascending'):
        shell_geometry(np.array([4.0e6, 3.5e6]))


@pytest.mark.physics_invariant
def test_face_flux_is_the_lesser_of_capacity_and_supply():
    """The closed form agrees with the recurrence it replaces, node by node.

    Five nodes chosen so the limit switches along the run: the second node's
    capacity is below the supply reaching it, the others pass what they are
    supplied. Without compaction nothing can leave, and instant compaction
    leaves only the permeability as a limit.
    """
    psi = np.array([0.1, 0.02, 0.3, 0.2, 0.4])
    volume = np.array([2.0, 1.0, 1.0, 3.0, 1.0])
    area = np.ones(5)
    speed = np.array([5.0, 0.5, 2.0, 3.0, 4.0])
    tau = np.array([1.0, 2.0, 1.0, 4.0, 1.0])
    # Capacities 0.5, 0.01, 0.6, 0.6, 1.6 and supplies 0.2, 0.01, 0.3, 0.15,
    # 0.4 give fluxes 0.2, 0.01, 0.31, 0.46, 0.86.
    flux, perc = run_fluxes(psi, volume, area, speed, tau)
    np.testing.assert_allclose(flux, [0.2, 0.01, 0.31, 0.46, 0.86], rtol=1e-12)
    expected, q = [], 0.0
    for k in range(5):
        q = min(area[k] * psi[k] * speed[k], q + psi[k] * volume[k] / tau[k])
        expected.append(q)
    np.testing.assert_allclose(flux, expected, rtol=1e-12)
    assert perc.tolist() == [False, True, False, False, False]
    # Fluxes are never negative, whatever the mix of limits.
    assert np.all(flux >= 0.0)
    # An infinitely stiff matrix releases no melt at all.
    stiff, _ = run_fluxes(psi, volume, area, speed, np.full(5, np.inf))
    np.testing.assert_allclose(stiff, 0.0, atol=0.0)
    # Instant compaction leaves each face at its own capacity.
    soft, soft_perc = run_fluxes(psi, volume, area, speed, np.zeros(5))
    np.testing.assert_allclose(soft, area * psi * speed, rtol=1e-12)
    assert np.all(soft_perc)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_single_node_reproduces_the_front_drainage_integral():
    """One node with L = V/A drains exactly as the front scheme's parcel.

    Cross-check against the front scheme's LSODA integral at rtol 1e-8, over
    five percolation times of a fluid melt so the drainage is far from both
    limits. The second-order sub-stepping at a fortieth of the node timescale
    stays within 1e-3 of it.
    """
    r = np.array([5.0e6, 5.06e6])
    volume, area, _ = shell_geometry(r)
    length = volume[0] / area[0]
    psi0, melt_visc = 0.3, 1.0
    w0 = darcy_velocity(psi0, _GRAIN, _DRHO, _G, melt_visc)
    duration = 5.0 * length / w0
    layer = drain_runs(
        np.array([psi0]),
        [np.array([0])],
        volume,
        area,
        _speed(melt_visc),
        np.array([0.0]),
        duration,
        substep_fraction=0.025,
    )
    front, _, _ = drainage_integral(psi0, duration, length, _GRAIN, _DRHO, _G, melt_visc, 1e-30)
    assert layer.psi[0] == pytest.approx(front, rel=1e-3)
    # Drainage happened, and it is self-limiting: far from both no drainage and
    # an emptied node.
    assert 0.3 * psi0 < layer.psi[0] < 0.6 * psi0
    # Melt that left the node is what the node lost.
    assert layer.returned == pytest.approx((psi0 - layer.psi[0]) * volume[0], rel=1e-12)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_compaction_limited_node_decays_exponentially():
    """Where the matrix controls, a node loses melt at 1/tau_s: psi0 exp(-t/tau_s).

    Analytic limit. The permeability is made so large that it never binds, and
    three e-folds are integrated. The tolerance of 2e-3 is the Runge-Kutta
    error at a twentieth of tau_s per sub-step.
    """
    volume, area = np.array([1.0e15]), np.array([1.0e10])
    psi0, tau = 0.3, 1.0e10
    decay = drain_runs(
        np.array([psi0]),
        [np.array([0])],
        volume,
        area,
        lambda p, idx: np.ones(np.size(idx)),
        np.array([tau]),
        3.0 * tau,
        substep_fraction=0.05,
    )
    assert decay.psi[0] == pytest.approx(psi0 * np.exp(-3.0), rel=2e-3)
    # A forward-Euler scheme at the same sub-step would land 7 percent low.
    assert abs(decay.psi[0] - psi0 * 0.95**60) / decay.psi[0] > 0.05
    assert decay.t_matrix[0] == pytest.approx(3.0 * tau, rel=1e-12)
    assert decay.t_darcy[0] == pytest.approx(0.0, abs=0.0)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_uniform_column_follows_the_kinematic_wave_rarefaction():
    """A uniform column on an impermeable base drains as the analytic rarefaction.

    For a flux ``q = K psi^3`` the fan from the base reaches the top at
    ``t_a = H / (3 K psi0^2)``. Before that the top still passes
    ``K psi0^3``, so ``M / M0 = 1 - K psi0^2 t / H``; after it the whole column
    is in the fan, ``psi = sqrt(z / (3 K t))`` and ``M / M0 = (2/3) sqrt(t_a / t)``.
    Upwind differencing converges at first order and overstates what is
    retained, so the coarse mesh sits above the finer one and both above the
    exact value.
    """
    height, k, psi0 = 1.0e5, 1.0e-6, 0.3
    t_a = height / (3.0 * k * psi0**2)

    def retained(n: int, t: float) -> float:
        dz = height / n
        out = drain_runs(
            _uniform(n, psi0),
            [np.arange(n)],
            _uniform(n, dz),
            np.ones(n),
            lambda p, idx: k * p**2,
            np.zeros(n),
            t,
        )
        return float(np.sum(out.psi) * dz / (psi0 * height))

    assert retained(100, 0.5 * t_a) == pytest.approx(1.0 - k * psi0**2 * 0.5 * t_a / height)
    fine, coarse = retained(100, 4.0 * t_a), retained(50, 4.0 * t_a)
    exact = (2.0 / 3.0) * np.sqrt(0.25)
    assert fine == pytest.approx(exact, rel=0.03)
    # A regression to draining over the whole column thickness per step, the
    # front picture, would leave exp(-4)-like values, far below 1/3.
    assert fine > 0.3
    assert exact < fine < coarse


@pytest.mark.physics_invariant
def test_column_conserves_melt_and_keeps_it_bounded():
    """Melt of a draining column plus what left through its top is conserved.

    A ten-node column with porosity rising upward, partly compaction- and partly
    percolation-limited, over 1e5 years. No node goes negative, none exceeds the
    largest starting porosity, and a longer drainage returns more.
    """
    r = np.linspace(4.0e6, 4.5e6, 11)
    volume, area, centre = shell_geometry(r)
    run = np.arange(10)
    psi0 = np.linspace(0.05, 0.45, 10)
    tau = compaction_times([run], r, centre, _uniform(10, _DRHO), _uniform(10, _G), 1.0e19)
    short = drain_runs(psi0, [run], volume, area, _speed(10.0), tau, 1.0e4 * SECS_PER_YEAR)
    long = drain_runs(psi0, [run], volume, area, _speed(10.0), tau, 1.0e5 * SECS_PER_YEAR)
    before = float(np.sum(psi0 * volume))
    assert float(np.sum(long.psi * volume)) + long.returned == pytest.approx(before, rel=1e-12)
    assert np.all(long.psi >= 0.0)
    assert np.all(long.psi <= psi0.max())
    assert long.returned > short.returned > 0.0
    # Nothing drains over a zero-length step.
    still = drain_runs(psi0, [run], volume, area, _speed(10.0), tau, 0.0)
    np.testing.assert_allclose(still.psi, psi0, rtol=0.0)
    with pytest.raises(ValueError, match='substep_fraction'):
        drain_runs(psi0, [run], volume, area, _speed(10.0), tau, 1.0, substep_fraction=0.0)


@pytest.mark.physics_invariant
def test_compaction_time_grows_toward_the_top_of_a_run():
    """The compaction pressure is the matrix load above a node, zero at the top.

    ``tau_s = mu_s / (drho g d)`` with ``d`` the depth below the top face of the
    run, so the base of a run compacts fastest and nodes outside any run never.
    """
    r = np.linspace(0.0, 4.0e5, 5) + 4.0e6
    _, _, centre = shell_geometry(r)
    tau = compaction_times(
        [np.array([0, 1, 2])], r, centre, _uniform(4, _DRHO), _uniform(4, _G), 1e20
    )
    depth = r[3] - centre[:3]
    np.testing.assert_allclose(tau[:3], 1e20 / (_DRHO * _G * depth), rtol=1e-12)
    assert tau[0] < tau[1] < tau[2]
    assert np.isinf(tau[3])


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
def test_a_remelted_node_returns_its_locked_volatiles_to_the_ocean():
    """A mush node that warms back above the transition gives its melt back.

    Nothing reaches the floor, and every kilogram the node locked on entry
    returns to the melt: the net move of the two steps is zero.
    """
    n = 3
    kw = _mesh_kwargs(n)
    liquid = {'CO2': 3.6e21}
    x0 = np.array([0.3, 1.0, 1.0])
    state, first = advance_layers(
        None,
        time_prev=0.0,
        time_now=1.0,
        melt_fraction=x0,
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
    assert state.status[0] == STATUS_LIQUID
    assert np.isnan(state.psi[0])
    assert state.retained['CO2'][0] == pytest.approx(0.0, abs=0.0)


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
    assert all(state.locked(e) == pytest.approx(0.0, abs=0.0) for e in state.kg)
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
    assert locking.retained['CO2'].tolist()[0] == pytest.approx(0.0, abs=0.0)
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
def test_buried_species_split_into_elements_by_stoichiometry():
    """Water buried at a node adds its hydrogen and oxygen to that node's record.

    The per-element sums equal the whole-planet split of the same species mass,
    with 7.94 kg of oxygen for every kilogram of hydrogen in water, and water
    still in pore melt counts toward the element totals of its node.
    """
    state = LayerState.initial(np.full(3, STATUS_SOLID), np.zeros(3), np.zeros(3), 0.0)
    water = np.array([0.0, 2.0e18, 1.0e18])
    add_locked_mass(state, {'H2O': water, 'CO2': np.array([5.0e17, 0.0, 0.0])})
    whole = element_masses_from_species({'H2O': 3.0e18, 'CO2': 5.0e17})
    for element, mass in whole.items():
        assert state.locked(element) == pytest.approx(mass, rel=1e-12)
    assert state.kg['O'][1] / state.kg['H'][1] == pytest.approx(7.94, rel=2e-3)
    assert state.kg['H'][0] == pytest.approx(0.0, abs=0.0)
    state.retained['H2O'] = np.array([1.0e18, 0.0, 0.0])
    extra = element_masses_from_species({'H2O': 1.0e18})['H']
    assert state.element_profile('H')[0] == pytest.approx(extra, rel=1e-12)


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
        ['CO2'],
    )
    add_locked_mass(state, {'H2O': np.array([1.0e18, 3.0e18, 0.0, 0.0, 0.0])})
    state.retained['CO2'][2] = 4.0e17
    write_state(path, state)
    write_state(path, state)  # a second write of the same step overwrites
    back = read_state(path, n)
    assert back.status.tolist() == state.status.tolist()
    np.testing.assert_array_equal(np.isnan(back.psi), np.isnan(state.psi))
    np.testing.assert_allclose(back.psi[2], state.psi[2], rtol=0.0)
    for element in state.kg:
        np.testing.assert_allclose(back.kg[element], state.kg[element], rtol=0.0)
    np.testing.assert_allclose(back.retained['CO2'], state.retained['CO2'], rtol=0.0)
    assert back.locked('C') == pytest.approx(state.locked('C'), rel=1e-15)
    assert back.locked('H') > 4.0e17
    assert read_state(path, n + 1) is None
    other = str(tmp_path / 'bare.nc')
    with nc.Dataset(other, mode='w') as ds:
        ds.createDimension('basic', n + 1)
    with pytest.raises(ValueError, match='staggered'):
        write_state(other, state)


@pytest.mark.physics_invariant
def test_melt_formed_from_a_solid_node_locks_nothing_new():
    """Buried volatiles are not released, so melt a solid node forms carries none.

    Node 0 freezes with its entry pore melt, then flickers across the floor ten
    times: nothing is locked or buried again. A later remelt past the
    transition makes it ocean, and entering the mush again from above locks new
    ocean melt.
    """
    n = 3
    kw = _stiff_kwargs(n)
    liquid = {'CO2': 3.6e21}
    x = np.array([0.3, 1.0, 1.0])
    state, first = advance_layers(
        None,
        time_prev=0.0,
        time_now=1.0,
        melt_fraction=x,
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
    buried = state.locked('C')
    carbon_per_co2 = element_masses_from_species({'CO2': 1.0})['C']
    assert buried == pytest.approx(first.moved_kg['CO2'] * carbon_per_co2, rel=1e-12)
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
    assert state.locked('C') == pytest.approx(buried, rel=1e-12)
    state, _ = advance_layers(
        state,
        time_prev=t,
        time_now=t + 1.0,
        melt_fraction=np.ones(n),
        **_volatile_kwargs(liquid, 1.8e24),
        **kw,
    )
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
    rho_l, rho_s = _RHO_L, _RHO_S

    def conc_of(state, k):
        f_entry = mass_fraction_of_volume(state.psi_entry[k], rho_l, rho_s)
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
    total = {sp: kg for sp, kg in liquid.items()}
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
    carbon_per_co2 = element_masses_from_species({'CO2': 1.0})['C']
    on_record = (
        float(np.sum(state.retained['CO2'])) + float(np.sum(state.kg['C'])) / carbon_per_co2
    )
    assert on_record + liquid['CO2'] == pytest.approx(total['CO2'], rel=1e-12)
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
    assert float(np.sum(capped_state.retained['CO2'])) == pytest.approx(1.0e18, rel=1e-12)


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
    state.kg['H'][:] = 1.0e17
    kept = state.copy()
    x2 = np.array([0.0, 0.1, 0.0, 0.0, 0.3, 0.4])
    advance_layers(
        state,
        time_prev=1.0,
        time_now=1.0e4,
        melt_fraction=x2,
        **_volatile_kwargs({'CO2': 1e21}, 1e24),
        **kw,
    )
    names = (
        'status',
        'psi',
        'psi_entry',
        't_entry',
        'melt_fraction',
        't_darcy',
        't_matrix',
        't_held',
    )
    for name in names:
        np.testing.assert_array_equal(getattr(state, name), getattr(kept, name))
    np.testing.assert_array_equal(state.kg['H'], kept.kg['H'])
    np.testing.assert_array_equal(state.retained['CO2'], kept.retained['CO2'])
    assert set(state.retained) == {'CO2'}


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
    r = np.linspace(4.0e6, 4.5e6, 3)
    volume, area, centre = shell_geometry(r)
    tau = compaction_times([np.arange(2)], r, centre, _uniform(2, _DRHO), _uniform(2, _G), 1e19)
    args = (
        np.array([0.4, 0.4]),
        [np.arange(2)],
        volume,
        area,
        _speed(1.0),
        tau,
        1.0e4 * SECS_PER_YEAR,
    )
    full = drain_runs(*args)
    capped = drain_runs(*args, max_substeps=1)
    assert capped.n_substeps == 1
    assert capped.elapsed < full.elapsed
    assert np.sum(capped.psi * volume) > np.sum(full.psi * volume)
