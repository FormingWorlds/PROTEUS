"""Unit tests for ``proteus.outgas.layer_drainage``.

The layer drainage drains the retained pore melt of every mush node of the
interior mesh into the node above. These tests exercise:

* the melt volume fraction of a node and its inverse, the trapped mass fraction,
* the liquid, mush and solid classification at its two thresholds,
* the shell geometry of the spherical mesh,
* the face flux as the lesser of the permeability capacity and the melt that
  compaction supplies from below, against a direct evaluation of the recurrence,
* a single node against the drainage integral of the front scheme, under both
  separation buoyancies, and against the exponential decay of the
  compaction-limited branch,
* a uniform column against the analytic rarefaction of the kinematic wave,
* conservation of the melt of a column plus what leaves it,
* the matrix deformation time against the load above a node.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.outgas.compaction import darcy_velocity, drainage_integral
from proteus.outgas.layer_drainage import (
    PHASE_LIQUID,
    PHASE_MUSH,
    PHASE_SOLID,
    classify_nodes,
    compaction_times,
    drain_mush_layers,
    mass_fraction_of_volume,
    melt_volume_fraction,
    mush_layer_fluxes,
    node_array,
    node_geometry,
    thickest_mush_layer,
)
from proteus.utils.constants import secs_per_year

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# End-member densities with the ten percent contrast of a silicate mush.
_RHO_S = 4000.0
_RHO_L = 3600.0

# Drainage inputs of the default PROTEUS mush: millimetre grains, a density
# contrast of 400 kg/m3 under 9.8 m/s2, and the solver's default buoyancy.
_GRAIN = 1.0e-3
_DRHO = 400.0
_G = 9.8
_BUOYANCY = 'mixture'


def _uniform(n: int, value: float) -> np.ndarray:
    return np.full(n, float(value))


def _speed(melt_visc: float, buoyancy: str = _BUOYANCY):
    """Percolation speed of a node at uniform contrast and gravity [m s-1]."""

    def speed_of(psi, index):
        k = np.size(index)
        return darcy_velocity(
            psi, _GRAIN, _uniform(k, _DRHO), _uniform(k, _G), melt_visc, buoyancy=buoyancy
        )

    return speed_of


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
        PHASE_LIQUID,
        PHASE_LIQUID,
        PHASE_MUSH,
        PHASE_MUSH,
        PHASE_SOLID,
        PHASE_SOLID,
    ]
    assert status.tolist() == expected
    # Raising the transition moves the boundary node into the mush.
    assert classify_nodes(x, phi, rfront_loc=0.55, phi_min=0.01)[1] == PHASE_MUSH


@pytest.mark.physics_invariant
def test_shell_geometry_sums_to_the_spherical_shell():
    """Shell volumes add up to the whole mantle, and faces grow as r squared."""
    r = np.array([3.5e6, 4.0e6, 5.5e6, 6.3e6])
    volume, area_top, centre = node_geometry(r)
    whole = 4.0 / 3.0 * np.pi * (6.3e6**3 - 3.5e6**3)
    assert np.sum(volume) == pytest.approx(whole, rel=1e-12)
    assert area_top[-1] == pytest.approx(4.0 * np.pi * 6.3e6**2, rel=1e-12)
    # A plane-parallel estimate (area times thickness at the top face) would
    # overstate the thick middle shell by about 20 percent.
    assert abs(volume[1] - area_top[1] * 1.5e6) / volume[1] > 0.1
    assert centre[0] == pytest.approx(3.75e6, rel=1e-12)
    with pytest.raises(ValueError, match='ascending'):
        node_geometry(np.array([4.0e6, 3.5e6]))


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
    flux, perc = mush_layer_fluxes(psi, volume, area, speed, tau)
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
    stiff, _ = mush_layer_fluxes(psi, volume, area, speed, np.full(5, np.inf))
    np.testing.assert_allclose(stiff, 0.0, atol=0.0)
    # Instant compaction leaves each face at its own capacity.
    soft, soft_perc = mush_layer_fluxes(psi, volume, area, speed, np.zeros(5))
    np.testing.assert_allclose(soft, area * psi * speed, rtol=1e-12)
    assert np.all(soft_perc)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_single_node_reproduces_the_front_drainage_integral(validation_plot):
    """One node with L = V/A drains exactly as the front scheme's parcel.

    Cross-check against the front scheme's LSODA integral at rtol 1e-8, at ten
    times over five percolation times of a fluid melt so the drainage is far
    from both limits, under either separation buoyancy. The second-order
    sub-stepping at a fortieth of the node timescale stays within 1e-3 of it.
    """
    r = np.array([5.0e6, 5.06e6])
    volume, area, _ = node_geometry(r)
    length = volume[0] / area[0]
    psi0, melt_visc = 0.3, 1.0
    history = {}
    for buoyancy in ('mixture', 'solid'):
        w0 = darcy_velocity(psi0, _GRAIN, _DRHO, _G, melt_visc, buoyancy=buoyancy)
        times = np.linspace(0.5, 5.0, 10) * length / w0
        psi, returned, layer, front = psi0, 0.0, [], []
        for t_prev, t_now in zip(np.concatenate(([0.0], times[:-1])), times):
            out = drain_mush_layers(
                np.array([psi]),
                [np.array([0])],
                volume,
                area,
                _speed(melt_visc, buoyancy),
                np.array([0.0]),
                t_now - t_prev,
                substep_fraction=0.025,
            )
            psi, returned = float(out.psi[0]), returned + out.returned
            ref, _, _ = drainage_integral(
                psi0, t_now, length, _GRAIN, _DRHO, _G, melt_visc, 1e-30, buoyancy=buoyancy
            )
            layer.append(psi)
            front.append(ref)
        np.testing.assert_allclose(layer, front, rtol=1e-3)
        # Drainage happened, and it is self-limiting: far from both no drainage
        # and an emptied node.
        assert 0.3 * psi0 < psi < 0.6 * psi0
        # Melt that left the node is what the node lost.
        assert returned == pytest.approx((psi0 - psi) * volume[0], rel=1e-12)
        history[buoyancy] = (times * w0 / length, np.array(layer), np.array(front))

    def figure():
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
        for k, (buoyancy, (t, layer, front)) in enumerate(history.items()):
            ax1.plot(t, front / psi0, '-', color=f'C{k}', label=buoyancy)
            ax1.plot(t, layer / psi0, 'o', mfc='none', color=f'C{k}')
            ax2.plot(t, layer / front - 1.0, 'o-', mfc='none', color=f'C{k}')
        ax1.set_xlabel('t w_D(psi0) / L')
        ax1.set_ylabel('psi / psi0')
        ax1.legend(fontsize=8)
        ax2.axhspan(-1e-3, 1e-3, color='C2', alpha=0.25, zorder=0, label='tolerance')
        ax2.set_xlabel('t w_D(psi0) / L')
        ax2.set_ylabel('layer / front - 1')
        fig.tight_layout()
        return fig

    validation_plot(figure, 'layer_drainage_single_node')


@pytest.mark.physics_invariant
def test_mixture_buoyancy_retains_more_than_the_solid_one():
    """Under ``'mixture'`` the buoyancy carries 1 - phi, so a node drains slower.

    Over the same time a node at psi0 = 0.3 keeps more melt under ``'mixture'``
    than under ``'solid'``. The two differ by about 5 percent, twenty times the
    1e-3 integration error, so a scheme that ignored the solver's buoyancy
    setting would show.
    """
    r = np.array([5.0e6, 5.06e6])
    volume, area, _ = node_geometry(r)
    length = volume[0] / area[0]
    psi0 = 0.3
    duration = 3.0 * length / darcy_velocity(psi0, _GRAIN, _DRHO, _G, 1.0, buoyancy='solid')
    kept = {
        b: drain_mush_layers(
            np.array([psi0]),
            [np.array([0])],
            volume,
            area,
            _speed(1.0, b),
            np.array([0.0]),
            duration,
            substep_fraction=0.025,
        ).psi[0]
        for b in ('mixture', 'solid')
    }
    assert kept['mixture'] > kept['solid'] * 1.02
    assert 0.0 < kept['solid'] < psi0


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
    decay = drain_mush_layers(
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
def test_uniform_column_follows_the_kinematic_wave_rarefaction(validation_plot):
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
    checkpoints = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])

    def retained(n: int) -> np.ndarray:
        dz = height / n
        psi, out = _uniform(n, psi0), []
        for t_prev, t_now in zip(np.concatenate(([0.0], checkpoints[:-1])), checkpoints):
            psi = drain_mush_layers(
                psi,
                [np.arange(n)],
                _uniform(n, dz),
                np.ones(n),
                lambda p, idx: k * p**2,
                np.zeros(n),
                (t_now - t_prev) * t_a,
            ).psi
            out.append(float(np.sum(psi) * dz / (psi0 * height)))
        return np.array(out)

    fine, coarse = retained(100), retained(50)
    exact = np.where(
        checkpoints <= 1.0, 1.0 - checkpoints / 3.0, (2.0 / 3.0) * np.sqrt(1.0 / checkpoints)
    )
    # Upwind smearing brings the fan to the top early, 1.3e-5 off at 0.75 t_a.
    early = checkpoints <= 0.5
    np.testing.assert_allclose(fine[early], exact[early], rtol=1e-6)
    assert fine[-1] == pytest.approx(exact[-1], rel=0.03)
    # A regression to draining over the whole column thickness per step, the
    # front picture, would leave exp(-4)-like values, far below 1/3.
    assert fine[-1] > 0.3
    assert exact[-1] < fine[-1] < coarse[-1]

    def figure():
        import matplotlib.pyplot as plt

        grid = np.linspace(0.05, 4.0, 60)
        curve = np.where(grid <= 1.0, 1.0 - grid / 3.0, (2.0 / 3.0) * np.sqrt(1.0 / grid))
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(grid, curve, '-', label='kinematic wave, exact')
        ax.plot(checkpoints, fine, 'o', mfc='none', label='layer drainage, 100 nodes')
        ax.plot(checkpoints, coarse, 's', mfc='none', label='layer drainage, 50 nodes')
        ax.axvline(1.0, color='0.5', lw=0.8, ls=':')
        ax.set_xlabel('t / t_a')
        ax.set_ylabel('M / M0')
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'layer_drainage_kinematic_wave')


@pytest.mark.physics_invariant
def test_column_conserves_melt_and_keeps_it_bounded():
    """Melt of a draining column plus what left through its top is conserved.

    A ten-node column with porosity rising upward, partly compaction- and partly
    percolation-limited, over 1e5 years. No node goes negative, none exceeds the
    largest starting porosity, and a longer drainage returns more.
    """
    r = np.linspace(4.0e6, 4.5e6, 11)
    volume, area, centre = node_geometry(r)
    run = np.arange(10)
    psi0 = np.linspace(0.05, 0.45, 10)
    tau = compaction_times([run], r, centre, _uniform(10, _DRHO), _uniform(10, _G), 1.0e19)
    short = drain_mush_layers(
        psi0, [run], volume, area, _speed(10.0), tau, 1.0e4 * secs_per_year
    )
    long = drain_mush_layers(
        psi0, [run], volume, area, _speed(10.0), tau, 1.0e5 * secs_per_year
    )
    before = float(np.sum(psi0 * volume))
    assert float(np.sum(long.psi * volume)) + long.returned == pytest.approx(before, rel=1e-12)
    assert np.all(long.psi >= 0.0)
    assert np.all(long.psi <= psi0.max())
    assert long.returned > short.returned > 0.0
    # Nothing drains over a zero-length step.
    still = drain_mush_layers(psi0, [run], volume, area, _speed(10.0), tau, 0.0)
    np.testing.assert_allclose(still.psi, psi0, rtol=0.0)
    with pytest.raises(ValueError, match='substep_fraction'):
        drain_mush_layers(
            psi0, [run], volume, area, _speed(10.0), tau, 1.0, substep_fraction=0.0
        )


@pytest.mark.physics_invariant
def test_tracer_moves_with_the_melt_and_is_conserved():
    """A carried quantity leaves each node at that node's concentration.

    Two stacked nodes, the lower one tagged and the upper one not. After a
    short drainage the upper node holds tagged melt from below, and the tagged
    mass of the column plus what left through the top equals what it started
    with. The concentration of what left is never above the top node's.
    """
    volume, area = np.array([1.0, 1.0]), np.array([1.0, 1.0])
    psi0 = np.array([0.3, 0.3])
    tracer0 = np.array([[0.3, 0.0]])  # one unit per unit melt volume, lower node only
    out = drain_mush_layers(
        psi0,
        [np.arange(2)],
        volume,
        area,
        lambda p, idx: 0.1 * p**2,
        np.zeros(2),
        5.0,
        tracer=tracer0,
    )
    assert out.tracer[0, 1] > 0.0
    left = float(np.sum(tracer0 * volume) - np.sum(out.tracer * volume))
    melt_left = float(np.sum(psi0 * volume) - np.sum(out.psi * volume))
    assert left >= 0.0
    # The top node started untagged, so the tagged share of what left is below one.
    assert left < melt_left
    assert np.all(out.tracer >= 0.0)


@pytest.mark.physics_invariant
def test_compaction_time_grows_toward_the_top_of_a_run():
    """The compaction pressure is the matrix load above a node, zero at the top.

    ``tau_s = mu_s / (drho g d)`` with ``d`` the depth below the top face of the
    run, so the base of a run compacts fastest and nodes outside any run never.
    """
    r = np.linspace(0.0, 4.0e5, 5) + 4.0e6
    _, _, centre = node_geometry(r)
    tau = compaction_times(
        [np.array([0, 1, 2])], r, centre, _uniform(4, _DRHO), _uniform(4, _G), 1e20
    )
    depth = r[3] - centre[:3]
    np.testing.assert_allclose(tau[:3], 1e20 / (_DRHO * _G * depth), rtol=1e-12)
    assert tau[0] < tau[1] < tau[2]
    assert np.isinf(tau[3])


@pytest.mark.physics_invariant
def test_thickest_run_and_node_arrays():
    """The thickest mush run is chosen by radial extent, not by node count.

    Two runs: three thin nodes and one thick node; the thick single node wins.
    A scalar broadcasts to every node, and a mis-sized or non-finite profile
    is refused before any drainage uses it.
    """
    r = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 14.0])
    status = np.array([PHASE_MUSH, PHASE_MUSH, PHASE_MUSH, PHASE_SOLID, PHASE_MUSH])
    run, extent = thickest_mush_layer(status, r)
    assert run.tolist() == [4]
    assert extent == pytest.approx(10.0)
    none, nan = thickest_mush_layer(np.full(5, PHASE_SOLID), r)
    assert none is None and np.isnan(nan)
    np.testing.assert_allclose(node_array(9.8, 3, 'gravity'), [9.8, 9.8, 9.8])
    with pytest.raises(ValueError, match='gravity must hold 3'):
        node_array([9.8, 9.8], 3, 'gravity')
    with pytest.raises(ValueError, match='finite'):
        node_array([9.8, np.nan, 9.8], 3, 'gravity')
