"""Explicit drainage of pore melt through the mush, node by node.

The front scheme of :mod:`proteus.outgas.compaction` follows one parcel across a
single freezing front and applies the melt it retains to the mass crystallised
in the step. The layer scheme drops that picture. Every node of the interior
mesh that lies in the mush keeps its own pore melt in its record from one
coupling step to the next. It drains that melt into the node above at the rate
its own porosity, density contrast and gravity allow, and keeps what it still
holds when it reaches the porosity floor. This is the per-node drainage of
Miyazaki and Korenaga (2019) on the interior solver's mesh. It needs no single
front, no front speed and no residence time; melt expelled from one node passes
through the nodes above it. This module holds the drainage physics;
:mod:`proteus.outgas.node_record` carries the per-node record and the volatiles.

Phase states
------------
Every staggered node, one cell of the mesh, is classified at every step from
the solver melt fraction ``x`` and the melt volume fraction ``phi`` it implies:

* liquid, ``x >= rfront_loc``: above the rheological transition, part of the
  convecting magma ocean;
* mush, ``x < rfront_loc`` and ``phi > phi_min``: the crystal framework is
  connected, so the pore melt leaves only by drainage;
* solid, ``phi <= phi_min``: the pore melt left in the node is locked.

A contiguous set of nodes in one phase state is a liquid, mush or solid layer.

``phi = x rho_s / (x rho_s + (1 - x) rho_l)`` uses the phase-boundary densities.
It is taken from the melt fraction rather than from the mixture density,
because the phase-boundary densities belong to the solidus and liquidus
temperatures and not to the node temperature, and is well defined whichever
phase is denser. A node entering the mush from above carries the pore melt it
had at the transition, ``psi = phi(x = rfront_loc)``, whose mass fraction is
exactly ``rfront_loc``, the bound the front scheme caps its guard at.

Drainage
--------
The pore melt ``psi`` of a node changes only by drainage, as in the front
integral; melt that crystallises in the pores keeps its volatiles. The melt
volume flux out of the top of node ``k`` of a mush layer is

    q_k = min( A_k psi_k w_D(psi_k),  q_(k-1) + psi_k V_k / tau_s,k ),   q_(base) = 0

with ``A_k`` the area of the node's upper face, ``V_k`` its volume, ``w_D`` the
percolation speed of :func:`~proteus.outgas.compaction.darcy_velocity` (the
solver's separation velocity under its ``separation_buoyancy``) at the node's
own density contrast and gravity, and ``tau_s,k = mu_s / (|drho| g d_k)`` the
matrix deformation time under the compaction pressure at depth ``d_k`` below
the top of the mush layer. The first term is the most the node's permeability
passes; the second is the melt that compaction of the node and of the nodes
beneath it supplies; the smaller controls, as in the front integral. A single
node with ``L = V / A`` and the same matrix time obeys the front integral's
equation exactly. The contrast enters as its magnitude, as in the solver's
separation velocity, so a node whose tables put the melt above the solid
density drains like any other and is counted.

Melt that leaves the top of a mush layer rejoins the magma ocean, or leaves
through the surface when the layer reaches the top of the mesh. A mush layer
capped by a solid node is sealed: nothing leaves it. The fluxes are integrated
with a second-order Runge-Kutta scheme, on sub-steps short enough that no node
passes on more than :data:`SUBSTEP_FRACTION` of what it holds in one of them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from proteus.outgas.compaction import _contiguous_runs

log = logging.getLogger('fwl.' + __name__)

PHASE_LIQUID = 0  # above the rheological transition, part of the magma ocean
PHASE_MUSH = 1  # connected crystal framework, pore melt drains
PHASE_SOLID = 2  # at or below the porosity floor, pore melt locked

# Largest part of a node's turnover time one sub-step may span. The flux of a
# percolating node rises as up to the 5.5th power of its porosity, so a tenth
# keeps the upwind scheme stable and positive in every regime.
SUBSTEP_FRACTION = 0.1

# Sub-steps allowed in one coupling step before the rest of the step is left
# undrained; fast drainage slows itself, so this is a safety net.
MAX_SUBSTEPS = 20000


@dataclass(frozen=True)
class DrainageParams:
    """Material and numerical parameters of one drainage call."""

    grain_size: float  # crystal grain size [m]
    melt_visc: float  # melt viscosity [Pa s]
    mush_visc: float  # mush matrix viscosity [Pa s]
    buoyancy: str  # the solver's separation_buoyancy, 'mixture' or 'solid'
    substep_fraction: float = SUBSTEP_FRACTION
    max_substeps: int = MAX_SUBSTEPS


@dataclass
class Drainage:
    """Outcome of draining the open layers of the mush over one coupling step."""

    psi: np.ndarray  # pore-melt volume fraction afterwards [1]
    t_darcy: np.ndarray  # time each node spent percolation-limited [s]
    t_matrix: np.ndarray  # time each node spent compaction-limited [s]
    returned: float  # melt volume that left the layers [m3]
    n_substeps: int
    elapsed: float  # time integrated; short of the step only if truncated [s]
    matrix_speed: float  # largest melt flux per area at the end [m s-1]
    tracer: np.ndarray | None = None  # carried quantity per node volume, afterwards


def melt_volume_fraction(
    melt_fraction: np.ndarray, rho_solid: np.ndarray, rho_melt: np.ndarray
) -> np.ndarray:
    """Volume fraction of melt from the melt mass fraction [1].

    ``phi = x rho_s / (x rho_s + (1 - x) rho_l)``, the melt volume ``x / rho_l``
    over the total ``x / rho_l + (1 - x) / rho_s``. Well defined whichever phase
    is denser, unlike the lever rule on the mixture density.

    Parameters
    ----------
    melt_fraction : ndarray
        Melt mass fraction [1]; clipped to [0, 1].
    rho_solid, rho_melt : ndarray
        Solid and melt densities [kg m-3].
    """
    x = np.clip(np.asarray(melt_fraction, dtype=float), 0.0, 1.0)
    melt = x * np.asarray(rho_solid, dtype=float)
    total = melt + (1.0 - x) * np.asarray(rho_melt, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(total > 0.0, melt / total, x)


def mass_fraction_of_volume(
    volume_fraction: np.ndarray, rho_melt: np.ndarray, rho_solid: np.ndarray
) -> np.ndarray:
    """Melt mass fraction of a mixture from its melt volume fraction [1].

    The inverse of :func:`melt_volume_fraction`, and the array form of
    :func:`~proteus.outgas.compaction.volume_to_mass_fraction`.

    Parameters
    ----------
    volume_fraction : ndarray
        Melt volume fraction [1]; clipped to [0, 1].
    rho_melt, rho_solid : ndarray
        Melt and solid densities [kg m-3].
    """
    f = np.clip(np.asarray(volume_fraction, dtype=float), 0.0, 1.0)
    melt = f * np.asarray(rho_melt, dtype=float)
    total = melt + (1.0 - f) * np.asarray(rho_solid, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(total > 0.0, melt / total, f)


def classify_nodes(
    melt_fraction: np.ndarray,
    volume_fraction: np.ndarray,
    rfront_loc: float,
    phi_min: float,
) -> np.ndarray:
    """Liquid, mush or solid code of every node; see the module docstring.

    Parameters
    ----------
    melt_fraction : ndarray
        Solver melt mass fraction [1].
    volume_fraction : ndarray
        Melt volume fraction it implies [1].
    rfront_loc : float
        Rheological transition, the melt fraction at the top of the mush [1].
    phi_min : float
        Porosity floor below which a node is solid [1].
    """
    x = np.asarray(melt_fraction, dtype=float)
    status = np.full(x.size, PHASE_MUSH, dtype=int)
    status[x >= rfront_loc] = PHASE_LIQUID
    status[(x < rfront_loc) & (np.asarray(volume_fraction, dtype=float) <= phi_min)] = (
        PHASE_SOLID
    )
    return status


def node_geometry(r_basic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Volume [m3], upper-face area [m2] and centre radius [m] of every node.

    ``r_basic`` holds the node boundaries [m], ascending from the core-mantle
    boundary, one more than there are nodes; each node is a spherical cell.

    Raises
    ------
    ValueError
        If there are fewer than two boundaries or they are not strictly ascending.
    """
    r = np.asarray(r_basic, dtype=float).ravel()
    if r.size < 2 or np.any(np.diff(r) <= 0.0):
        raise ValueError('node boundaries must be at least two strictly ascending radii')
    volume = (4.0 * np.pi / 3.0) * (r[1:] ** 3 - r[:-1] ** 3)
    area_top = 4.0 * np.pi * r[1:] ** 2
    centre = 0.5 * (r[:-1] + r[1:])
    return volume, area_top, centre


def compaction_times(
    layers: list[np.ndarray],
    r_basic: np.ndarray,
    centre: np.ndarray,
    delta_rho: np.ndarray,
    gravity: np.ndarray,
    mush_visc: float,
) -> np.ndarray:
    """Matrix deformation time of every node in ``layers`` [s]; infinite elsewhere.

    ``tau_s = mu_s / (|drho| g d)`` with ``d`` the depth [m] of the node centre
    below the upper face of the layer's top node. With the melt connected to the
    ocean above, the matrix carries its own buoyancy-reduced weight, so the
    pressure that compacts it grows from zero at the top of the layer to
    ``|drho| g d``.
    """
    tau = np.full(np.size(centre), np.inf)
    for layer in layers:
        depth = float(r_basic[layer[-1] + 1]) - centre[layer]
        load = np.abs(delta_rho[layer]) * gravity[layer] * depth
        tau[layer] = np.where(load > 0.0, mush_visc / np.where(load > 0.0, load, 1.0), np.inf)
    return tau


def mush_layer_fluxes(
    psi: np.ndarray,
    volume: np.ndarray,
    area_top: np.ndarray,
    speed: np.ndarray,
    tau_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Melt volume flux out of the top of every node of one layer [m3 s-1].

    ``q_k = min(A_k psi_k w_k, q_(k-1) + psi_k V_k / tau_s,k)`` with nothing
    entering the base, evaluated in closed form as
    ``q_k = S_k + min(0, min_(j<=k) (c_j - S_j))`` with ``c`` the capacity and
    ``S`` the cumulative supply. Also returns which nodes their own
    permeability limits, rather than the melt supplied to them.
    """
    capacity = area_top * psi * speed
    positive = tau_s > 0.0
    supply = np.where(positive, psi * volume / np.where(positive, tau_s, 1.0), np.inf)
    # A node supplied with more than it can pass passes its capacity whatever
    # arrives from below, so this clip changes no flux and avoids cancellation.
    supply = np.minimum(supply, capacity)
    cumulative = np.cumsum(supply)
    slack = np.minimum.accumulate(np.minimum(capacity - cumulative, 0.0))
    flux = np.maximum(cumulative + slack, 0.0)
    inflow = np.concatenate(([0.0], flux[:-1]))
    return flux, capacity <= inflow + supply


def _rates(psi, tracer, layers, volume, area_top, speed_of, tau_s):
    """Rates of change of the pore melt and what it carries, and the step limit.

    The limit is the shortest time any node takes to pass on what it holds,
    ``psi V / q``, which keeps every node's melt and carried mass positive and
    the upwind transport monotone.
    """
    rate = np.zeros_like(psi)
    t_rate = None if tracer is None else np.zeros_like(tracer)
    limited = np.zeros(psi.size, dtype=bool)
    returned = 0.0
    span = np.inf
    for layer in layers:
        p = psi[layer]
        q, perc = mush_layer_fluxes(
            p, volume[layer], area_top[layer], speed_of(p, layer), tau_s[layer]
        )
        inflow = np.concatenate(([0.0], q[:-1]))
        rate[layer] = (inflow - q) / volume[layer]
        limited[layer] = perc
        returned += float(q[-1])
        if tracer is not None:
            held = tracer[:, layer]
            conc = np.divide(held, p, out=np.zeros_like(held), where=p > 0.0)
            out = q * conc
            into = np.concatenate((np.zeros((held.shape[0], 1)), out[:, :-1]), axis=1)
            t_rate[:, layer] = (into - out) / volume[layer]
        with np.errstate(divide='ignore', invalid='ignore'):
            turnover = np.where(q > 0.0, p * volume[layer] / q, np.inf)
        span = min(span, float(np.min(turnover)))
    return rate, t_rate, returned, limited, span


def drain_mush_layers(
    psi: np.ndarray,
    layers: list[np.ndarray],
    volume: np.ndarray,
    area_top: np.ndarray,
    speed_of: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tau_s: np.ndarray,
    duration: float,
    substep_fraction: float = SUBSTEP_FRACTION,
    max_substeps: int = MAX_SUBSTEPS,
    tracer: np.ndarray | None = None,
) -> Drainage:
    """Drain the pore melt of every open mush layer over ``duration`` seconds.

    ``speed_of(psi, index)`` returns the drainage speed of the nodes ``index``
    at pore-melt volume fractions ``psi`` [m s-1]. Every flux is counted once
    leaving one node and once entering the next, so the melt of the layers plus
    what left through their tops is conserved to rounding. ``tracer``, of shape
    ``(quantities, nodes)``, is anything the melt carries per unit node volume,
    here the buried mass of each species [kg m-3]; it moves with the melt at the
    concentration of the node it leaves and is conserved the same way.

    Raises
    ------
    ValueError
        If ``substep_fraction`` lies outside (0, 1].
    """
    if not 0.0 < substep_fraction <= 1.0:
        raise ValueError(f'substep_fraction must lie in (0, 1], got {substep_fraction!r}')
    psi = np.array(psi, dtype=float, copy=True)
    if tracer is not None:
        tracer = np.array(tracer, dtype=float, copy=True)
    t_darcy = np.zeros_like(psi)
    t_matrix = np.zeros_like(psi)
    members = np.zeros(psi.size, dtype=bool)
    for layer in layers:
        members[layer] = True
    elapsed, returned, steps = 0.0, 0.0, 0
    while layers and elapsed < duration:
        if steps >= max_substeps:
            log.warning(
                'Layer drainage stopped after %d sub-steps with %.3g of %.3g s '
                'integrated; the rest of the step is left undrained, an upper '
                'bound on what the mush retains.',
                steps,
                elapsed,
                duration,
            )
            break
        rate1, t_rate1, out1, limit1, span = _rates(
            psi, tracer, layers, volume, area_top, speed_of, tau_s
        )
        dt = min(duration - elapsed, substep_fraction * span)
        if not dt > 0.0:
            break
        trial = np.clip(psi + dt * rate1, 0.0, 1.0)
        t_trial = None if tracer is None else np.maximum(tracer + dt * t_rate1, 0.0)
        rate2, t_rate2, out2, _, _ = _rates(
            trial, t_trial, layers, volume, area_top, speed_of, tau_s
        )
        psi = np.clip(psi + 0.5 * dt * (rate1 + rate2), 0.0, 1.0)
        if tracer is not None:
            tracer = np.maximum(tracer + 0.5 * dt * (t_rate1 + t_rate2), 0.0)
        returned += 0.5 * dt * (out1 + out2)
        t_darcy[members & limit1] += dt
        t_matrix[members & ~limit1] += dt
        elapsed += dt
        steps += 1

    matrix_speed = 0.0
    for layer in layers:
        q, _ = mush_layer_fluxes(
            psi[layer],
            volume[layer],
            area_top[layer],
            speed_of(psi[layer], layer),
            tau_s[layer],
        )
        matrix_speed = max(matrix_speed, float(np.max(q / area_top[layer])))
    return Drainage(
        psi=psi,
        t_darcy=t_darcy,
        t_matrix=t_matrix,
        returned=returned,
        n_substeps=steps,
        elapsed=elapsed,
        matrix_speed=matrix_speed,
        tracer=tracer,
    )


def node_array(values, n: int, name: str) -> np.ndarray:
    """``values`` as ``n`` finite floats, a scalar broadcast to every node.

    Raises
    ------
    ValueError
        If the values cannot fill ``n`` nodes or one is not finite.
    """
    a = np.asarray(values, dtype=float).ravel()
    if a.size == 1:
        a = np.full(n, a[0])
    if a.size != n or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must hold {n} finite values, got {a.size}')
    return a


def mush_layers(mask: np.ndarray) -> list[np.ndarray]:
    """Index arrays of the contiguous runs of ``True`` in ``mask``, bottom up."""
    return _contiguous_runs(np.asarray(mask, dtype=bool))


def thickest_mush_layer(
    phase: np.ndarray, r_basic: np.ndarray
) -> tuple[np.ndarray | None, float]:
    """The mush layer of largest radial extent, and that extent [m]."""
    layers = mush_layers(phase == PHASE_MUSH)
    if not layers:
        return None, float('nan')
    extents = [float(r_basic[layer[-1] + 1] - r_basic[layer[0]]) for layer in layers]
    k = int(np.argmax(extents))
    return layers[k], extents[k]
