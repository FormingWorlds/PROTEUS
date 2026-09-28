"""Explicit drainage of pore melt through the mush, node by node.

The front scheme of :mod:`proteus.outgas.compaction` follows one parcel across a
single freezing front and applies the melt it retains to the mass crystallised
in the step. The layer scheme here drops that picture. Every node of the
interior mesh that lies in the mush keeps its own retained pore melt, and the
volatiles in it, as state from one step to the next. It drains that melt into
the node above at the rate its own porosity, density contrast and gravity
allow, and what it still holds when it reaches the porosity floor stays there.
This is the per-layer drainage of Miyazaki and Korenaga (2019) on the interior
solver's mesh. It needs no single front, no front speed and no residence time;
melt expelled from one layer passes through the layers above it; and the
trapped volatiles keep the depth at which they were trapped.

Node states
-----------
Every staggered node is classified at every step from the solver melt
fraction ``x`` and the melt volume fraction ``phi`` it implies:

* liquid, ``x >= rfront_loc``: above the rheological transition, part of the
  convecting magma ocean;
* mush, ``x < rfront_loc`` and ``phi > phi_min``: the crystal framework is
  connected, so the pore melt leaves only by drainage;
* solid, ``phi <= phi_min``: the pore melt left in the node is trapped.

``phi = x rho_s / (x rho_s + (1 - x) rho_l)`` uses the phase-boundary densities.
It is taken from the melt fraction rather than recomputed from the mixture
density, because the phase-boundary densities belong to the solidus and
liquidus temperatures and not to the node temperature. That biases the density
lever rule wherever the mushy zone is wide, and near the surface it can report
a node as solid while the solver still holds a third of it molten.

A node entering the mush from above carries the pore melt it had at the
transition, ``psi = phi(x = rfront_loc)``, whose mass fraction is exactly
``rfront_loc``, the bound the front scheme caps its guard at. A node entering
it from below, after remelting, carries its current volume fraction.

Drainage
--------
The retained pore melt ``psi`` of a node changes only by drainage, as in the
front integral. Melt that crystallises in the pores keeps its volatiles, so
crystallisation in place does not reduce what the node retains. The melt volume
flux out of the top of node ``k`` of a contiguous run of mush nodes is

    q_k = min( A_k psi_k w_D(psi_k),  q_(k-1) + psi_k V_k / tau_s,k ),   q_(base) = 0

with ``A_k`` the area of the node's upper face, ``V_k`` its volume, ``w_D`` the
Darcy velocity of :func:`~proteus.outgas.compaction.darcy_velocity` at the
node's own density contrast and gravity, and ``tau_s,k = mu_s / (drho g d_k)``
the matrix deformation time under the compaction pressure at depth ``d_k``
below the top of the run. The first term is the most the node's permeability
passes. The second is the melt that compaction of the node and of the nodes
beneath it supplies. The smaller controls, as in the front integral. A single
node with ``L = V / A`` and the same matrix time obeys the front integral's
equation exactly.

The contrast enters as its magnitude, as in the interior solver's own
separation velocity. The phase-boundary tables can put the melt above the
solid density, and in the current ones they do so over the top few nodes of an
Earth-sized mantle, where the tabulated melt density rises as the pressure
falls below about 5 GPa. Holding the melt of such nodes, or treating them as
barriers, lets a few nodes seal or dominate the whole mantle; they are counted
and reported instead, and drain like any other node.

Melt that leaves the top of a run rejoins the magma ocean, or leaves through
the surface when the run reaches the top of the mesh. A run capped by a solid
node is sealed: nothing leaves it, so it retains all of its melt. The fluxes
are integrated with a second-order Runge-Kutta scheme, on sub-steps short
enough that no node passes on more than a tenth of what it holds in one of
them.

The nodes that drain over a step are the mush of the previous step. A node
therefore starts draining on the step after it enters the mush, and drains
through the step in which it reaches the floor. The two offsets average half a
step each, so the time a node drains equals its residence on average, as long
as the step length changes little over a residence; steps that lengthen with
time tip it toward draining slightly too long. A node that passes from liquid
to solid within one step is never drained. It is trapped with the melt of the
transition, the no-drainage bound, and counted as such.

Volatile budget
---------------
A node's pore melt leaves contact with the magma ocean when its crystal
framework locks: melt drains upward only, so nothing flows back from the ocean
to renew it. Its volatiles are therefore locked from that moment, at the
concentration the chemistry solver held then, and carried with the melt as it
drains; they leave the chemistry solver's reach, as buried mass does. A node
entering the mush from above locks

    R_Z = (1 - D_Z) C_Z F m

of each species, with ``C_Z`` that step's melt concentration, ``m`` the node
mass and ``F`` the melt mass fraction of its pore melt, exactly ``rfront_loc``
on entry from above. Drainage removes melt, and the volatiles in it, in
proportion to the retained volume, so afterwards ``F = F_entry psi / psi_entry``
and the locked concentration of a node that receives nothing from below stays
its entry value. Melt passing from one node into the next takes its share of the locked
mass with it. Melt that drains out of the top of a run rejoins the ocean, and
the volatiles it carries return to the melt reservoir. A node that remelts back
into the ocean returns what it still holds. What a node holds when it reaches
the floor stays buried for good. Melt formed by remelting a solid node is made
from that node's own buried material, so it locks nothing new, and a node that
flickers across the floor buries nothing again. Every step's crystals take
``D_Z C_Z dM_RM`` wherever they form. Over the life of a node the two add up to
``[(1 - F) D_Z + F] C m``, Sim et al. (2024) Eq. 6 applied to the node's own
mass. For ``D_Z >= 1`` the locked melt term is zero: the crystals have then
already taken at least the melt concentration.

The new locks of a step, crystals included, are capped together at what the
melt reservoir holds after this step's returns. The net mass a step moves into
the solid reservoir can therefore be negative, when drained melt returns more
than entering nodes lock.

The mass locked at each node is kept per species while the node is mush, and
per element once it is solid, so the depth distribution of the trapped
volatiles is part of the state. It is written to the interior snapshot next to
the solver's own fields, and a resumed run reads it back from there.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from proteus.outgas.common import element_masses_from_species
from proteus.outgas.compaction import (
    BRANCH_DARCY,
    BRANCH_GUARD,
    BRANCH_MATRIX,
    BRANCH_NONE,
    SECS_PER_YEAR,
    _contiguous_runs,
    darcy_velocity,
)
from proteus.utils.constants import vol_list

log = logging.getLogger('fwl.' + __name__)

STATUS_LIQUID = 0  # above the rheological transition, part of the magma ocean
STATUS_MUSH = 1  # connected crystal framework, pore melt drains
STATUS_SOLID = 2  # at or below the porosity floor, pore melt trapped

# Largest part of a node's turnover time one sub-step may span. The flux of a
# percolating node rises as a power of its porosity of up to 5.5 in the
# Rumpf-Gupte regime, so its characteristic speed is up to 5.5 times its
# drainage speed; a tenth keeps the upwind scheme stable and positive for every
# regime, with a Runge-Kutta error of about half a percent over three e-folds of
# drainage.
SUBSTEP_FRACTION = 0.1

# Sub-steps allowed in one coupling step before the rest of the step is left
# undrained. Fast drainage slows itself, because the permeability falls with
# the porosity, so the count grows only logarithmically with the step length;
# the cap is a safety net, not a working limit.
MAX_SUBSTEPS = 20000

# Mass of each element carried by one kilogram of each volatile species.
_ELEMENT_SPLIT = {s: element_masses_from_species({s: 1.0}) for s in vol_list}

# Elements the per-node record tracks: every element a trapped species carries.
RECORD_ELEMENTS = tuple(sorted({e for split in _ELEMENT_SPLIT.values() for e in split}))

# Per-node arrays written to the interior snapshot, with their units.
_STATE_FIELDS = {
    'status': '0 liquid, 1 mush, 2 solid',
    'psi': '',
    'psi_entry': '',
    't_entry': 'yr',
    'melt_fraction': '',
    't_darcy': 'yr',
    't_matrix': 'yr',
    't_held': 'yr',
}


def _snapshot_name(name: str) -> str:
    """Variable name of a per-node state array in the interior snapshot."""
    return f'trap_{name}_s'


def _split(species: str) -> dict[str, float]:
    """Mass of each element in one kilogram of ``species``."""
    return _ELEMENT_SPLIT.get(species, {species: 1.0})


@dataclass
class LayerState:
    """Per-node trapping state on the staggered nodes of the interior mesh."""

    status: np.ndarray  # STATUS_* code at the last classification
    psi: np.ndarray  # retained pore-melt volume fraction [1]; NaN outside the mush
    psi_entry: np.ndarray  # pore-melt volume fraction on entering the mush [1]
    t_entry: np.ndarray  # time the node entered the mush [yr]
    melt_fraction: np.ndarray  # solver melt fraction at the last classification [1]
    t_darcy: np.ndarray  # time spent percolation-limited in this residence [yr]
    t_matrix: np.ndarray  # time spent compaction-limited in this residence [yr]
    t_held: np.ndarray  # time spent unable to drain, sealed [yr]
    kg: dict[str, np.ndarray] = field(default_factory=dict)  # buried, per element [kg]
    retained: dict[str, np.ndarray] = field(default_factory=dict)  # in pore melt [kg]

    @property
    def size(self) -> int:
        """Number of staggered nodes the state covers."""
        return int(self.status.size)

    def copy(self) -> LayerState:
        """Independent copy, so a step never mutates the state it started from."""
        return LayerState(
            status=self.status.copy(),
            psi=self.psi.copy(),
            psi_entry=self.psi_entry.copy(),
            t_entry=self.t_entry.copy(),
            melt_fraction=self.melt_fraction.copy(),
            t_darcy=self.t_darcy.copy(),
            t_matrix=self.t_matrix.copy(),
            t_held=self.t_held.copy(),
            kg={e: v.copy() for e, v in self.kg.items()},
            retained={s: v.copy() for s, v in self.retained.items()},
        )

    def element_profile(self, element: str) -> np.ndarray:
        """Mass of one element locked at every node, buried or in pore melt [kg]."""
        total = np.array(self.kg.get(element, np.zeros(self.size)), dtype=float)
        for species, kg in self.retained.items():
            share = _split(species).get(element, 0.0)
            if share:
                total = total + share * kg
        return total

    def locked(self, element: str) -> float:
        """Mass of one element locked in the mesh, summed over nodes [kg]."""
        return float(np.sum(self.element_profile(element)))

    @classmethod
    def initial(
        cls,
        status: np.ndarray,
        volume_fraction: np.ndarray,
        melt_fraction: np.ndarray,
        time: float,
        species: tuple[str, ...] | list[str] = (),
    ) -> LayerState:
        """State of a mesh first seen at ``time``, with nothing locked yet.

        Mush nodes start with their current volume fraction: how much they
        drained before they were first seen is unknown, and this is the most
        they can still hold. The caller locks their volatiles.
        """
        n = int(np.size(status))
        mush = np.asarray(status) == STATUS_MUSH
        psi = np.where(mush, volume_fraction, np.nan)
        return cls(
            status=np.asarray(status, dtype=int).copy(),
            psi=psi,
            psi_entry=psi.copy(),
            t_entry=np.where(mush, float(time), np.nan),
            melt_fraction=np.asarray(melt_fraction, dtype=float).copy(),
            t_darcy=np.zeros(n),
            t_matrix=np.zeros(n),
            t_held=np.zeros(n),
            kg={e: np.zeros(n) for e in RECORD_ELEMENTS},
            retained={s: np.zeros(n) for s in species},
        )


@dataclass
class Drainage:
    """Outcome of draining the open runs of the mush over one coupling step."""

    psi: np.ndarray  # retained pore-melt volume fraction afterwards [1]
    t_darcy: np.ndarray  # time each node spent percolation-limited [s]
    t_matrix: np.ndarray  # time each node spent compaction-limited [s]
    returned: float  # melt volume that left the runs [m3]
    n_substeps: int
    elapsed: float  # time integrated; short of the step only if truncated [s]
    matrix_speed: float  # largest melt flux per area at the end [m s-1]
    tracer: np.ndarray | None = None  # carried quantity per node volume, afterwards


@dataclass
class LayerStep:
    """What one coupling step of the layer drainage did."""

    exited: np.ndarray  # nodes that reached the porosity floor this step
    trapped: np.ndarray  # trapped melt mass fraction, F_entry psi / psi_entry [1]
    branch: np.ndarray  # drainage branch code of each exited node
    residence: np.ndarray  # time each exited node spent in the mush [yr]
    node_branch: np.ndarray  # branch code of every node that drained this step
    n_mush: int = 0
    n_entered: int = 0
    n_remelted: int = 0
    n_dense: int = 0  # mush nodes whose tables put the melt above the solid density
    n_substeps: int = 0
    truncated: bool = False
    held: list[str] = field(default_factory=list)  # why nodes could not drain
    thickness: float = float('nan')  # radial extent of the thickest mush run [m]
    r_base: float = float('nan')  # base radius of that run [m]
    rho_solid_base: float = float('nan')  # solid density at that base [kg m-3]
    node_thickness: float = float('nan')  # mean node thickness of that run [m]
    tau_d: float = float('nan')  # percolation time of that run at its top [s]
    tau_s: float = float('nan')  # matrix time of that run [s]
    matrix_speed: float = float('nan')  # largest melt flux per area [m s-1]
    returned: float = 0.0  # melt volume returned to the ocean or surface [m3]
    moved_kg: dict[str, float] = field(default_factory=dict)  # net into the solid [kg]
    locked_kg: dict[str, float] = field(default_factory=dict)  # newly locked [kg]
    released_kg: dict[str, float] = field(default_factory=dict)  # returned to melt [kg]
    capped: list[str] = field(default_factory=list)  # species limited by the melt


def melt_volume_fraction(
    melt_fraction: np.ndarray, rho_solid: np.ndarray, rho_melt: np.ndarray
) -> np.ndarray:
    """Volume fraction of melt from the melt mass fraction [1].

    ``phi = x rho_s / (x rho_s + (1 - x) rho_l)``, the melt volume ``x / rho_l``
    over the total ``x / rho_l + (1 - x) / rho_s``. Well defined whichever phase
    is denser, unlike the lever rule on the mixture density.
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
    """Liquid, mush or solid code of every node; see the module docstring."""
    x = np.asarray(melt_fraction, dtype=float)
    status = np.full(x.size, STATUS_MUSH, dtype=int)
    status[x >= rfront_loc] = STATUS_LIQUID
    status[(x < rfront_loc) & (np.asarray(volume_fraction, dtype=float) <= phi_min)] = (
        STATUS_SOLID
    )
    return status


def shell_geometry(r_basic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Volume, upper-face area and centre radius of every spherical shell.

    ``r_basic`` holds the shell boundaries, ascending from the core-mantle
    boundary, one more than there are shells.
    """
    r = np.asarray(r_basic, dtype=float).ravel()
    if r.size < 2 or np.any(np.diff(r) <= 0.0):
        raise ValueError('shell boundaries must be at least two strictly ascending radii')
    volume = (4.0 * np.pi / 3.0) * (r[1:] ** 3 - r[:-1] ** 3)
    area_top = 4.0 * np.pi * r[1:] ** 2
    centre = 0.5 * (r[:-1] + r[1:])
    return volume, area_top, centre


def compaction_times(
    runs: list[np.ndarray],
    r_basic: np.ndarray,
    centre: np.ndarray,
    delta_rho: np.ndarray,
    gravity: np.ndarray,
    mush_visc: float,
) -> np.ndarray:
    """Matrix deformation time of every node in ``runs`` [s]; infinite elsewhere.

    ``tau_s = mu_s / (drho g d)`` with ``d`` the depth of the node centre below
    the upper face of the run's top node. With the melt connected to the ocean
    above, the matrix carries its own buoyancy-reduced weight, so the pressure
    that compacts it grows from zero at the top of the run to ``drho g d``.
    """
    tau = np.full(np.size(centre), np.inf)
    for run in runs:
        depth = float(r_basic[run[-1] + 1]) - centre[run]
        load = np.abs(delta_rho[run]) * gravity[run] * depth
        tau[run] = np.where(load > 0.0, mush_visc / np.where(load > 0.0, load, 1.0), np.inf)
    return tau


def run_fluxes(
    psi: np.ndarray,
    volume: np.ndarray,
    area_top: np.ndarray,
    speed: np.ndarray,
    tau_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Melt volume flux out of the top of every node of one run [m3 s-1].

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
    # arrives from below, so clipping the supply there changes no flux and
    # keeps the cumulative sum free of cancellation.
    supply = np.minimum(supply, capacity)
    cumulative = np.cumsum(supply)
    slack = np.minimum.accumulate(np.minimum(capacity - cumulative, 0.0))
    flux = np.maximum(cumulative + slack, 0.0)
    inflow = np.concatenate(([0.0], flux[:-1]))
    return flux, capacity <= inflow + supply


def _rates(psi, tracer, runs, volume, area_top, speed_of, tau_s):
    """Rates of change of the retained melt and what it carries, and the step limit.

    The limit is the shortest time any node takes to pass on what it holds,
    ``psi V / q``. That keeps every node's melt and carried mass positive and
    the upwind transport monotone, and it equals the node's own drainage time
    where the node drains itself: ``V / (A w)`` when its permeability limits,
    ``tau_s`` when its compaction does.
    """
    rate = np.zeros_like(psi)
    t_rate = None if tracer is None else np.zeros_like(tracer)
    limited = np.zeros(psi.size, dtype=bool)
    returned = 0.0
    span = np.inf
    for run in runs:
        p = psi[run]
        q, perc = run_fluxes(p, volume[run], area_top[run], speed_of(p, run), tau_s[run])
        inflow = np.concatenate(([0.0], q[:-1]))
        rate[run] = (inflow - q) / volume[run]
        limited[run] = perc
        returned += float(q[-1])
        if tracer is not None:
            held = tracer[:, run]
            conc = np.divide(held, p, out=np.zeros_like(held), where=p > 0.0)
            out = q * conc
            into = np.concatenate((np.zeros((held.shape[0], 1)), out[:, :-1]), axis=1)
            t_rate[:, run] = (into - out) / volume[run]
        with np.errstate(divide='ignore', invalid='ignore'):
            turnover = np.where(q > 0.0, p * volume[run] / q, np.inf)
        span = min(span, float(np.min(turnover)))
    return rate, t_rate, returned, limited, span


def drain_runs(
    psi: np.ndarray,
    runs: list[np.ndarray],
    volume: np.ndarray,
    area_top: np.ndarray,
    speed_of: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tau_s: np.ndarray,
    duration: float,
    substep_fraction: float = SUBSTEP_FRACTION,
    max_substeps: int = MAX_SUBSTEPS,
    tracer: np.ndarray | None = None,
) -> Drainage:
    """Drain the retained melt of every open run over ``duration`` seconds.

    ``speed_of(psi, index)`` returns the drainage speed of the nodes ``index``
    at retained volume fractions ``psi`` [m s-1]. Every flux is counted once
    leaving one node and once entering the next, so the melt of the runs plus
    what left through their tops is conserved to rounding. ``tracer``, of shape
    ``(quantities, nodes)``, is anything the melt carries per unit node volume,
    here the locked mass of each species; it moves with the melt at the
    concentration of the node it leaves and is conserved the same way.
    """
    if not 0.0 < substep_fraction <= 1.0:
        raise ValueError(f'substep_fraction must lie in (0, 1], got {substep_fraction!r}')
    psi = np.array(psi, dtype=float, copy=True)
    if tracer is not None:
        tracer = np.array(tracer, dtype=float, copy=True)
    t_darcy = np.zeros_like(psi)
    t_matrix = np.zeros_like(psi)
    members = np.zeros(psi.size, dtype=bool)
    for run in runs:
        members[run] = True
    elapsed, returned, steps = 0.0, 0.0, 0
    while runs and elapsed < duration:
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
            psi, tracer, runs, volume, area_top, speed_of, tau_s
        )
        dt = min(duration - elapsed, substep_fraction * span)
        if not dt > 0.0:
            break
        trial = np.clip(psi + dt * rate1, 0.0, 1.0)
        t_trial = None if tracer is None else np.maximum(tracer + dt * t_rate1, 0.0)
        rate2, t_rate2, out2, _, _ = _rates(
            trial, t_trial, runs, volume, area_top, speed_of, tau_s
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
    for run in runs:
        q, _ = run_fluxes(
            psi[run], volume[run], area_top[run], speed_of(psi[run], run), tau_s[run]
        )
        matrix_speed = max(matrix_speed, float(np.max(q / area_top[run])))
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


def _node_array(values, n: int, name: str) -> np.ndarray:
    """``values`` as ``n`` finite floats, a scalar broadcast to every node."""
    a = np.asarray(values, dtype=float).ravel()
    if a.size == 1:
        a = np.full(n, a[0])
    if a.size != n or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must hold {n} finite values, got {a.size}')
    return a


def _thickest_run(status: np.ndarray, r_basic: np.ndarray) -> tuple[np.ndarray | None, float]:
    """The mush run of largest radial extent, and that extent [m]."""
    runs = _contiguous_runs(status == STATUS_MUSH)
    if not runs:
        return None, float('nan')
    extents = [float(r_basic[run[-1] + 1] - r_basic[run[0]]) for run in runs]
    k = int(np.argmax(extents))
    return runs[k], extents[k]


@dataclass
class _Mesh:
    """Profiles of one step on the staggered nodes, shared by the stage functions."""

    x: np.ndarray  # solver melt fraction [1]
    phi: np.ndarray  # melt volume fraction it implies [1]
    status: np.ndarray  # class of every node now
    rho_s: np.ndarray  # solid phase-boundary density [kg m-3]
    rho_l: np.ndarray  # melt phase-boundary density [kg m-3]
    gravity: np.ndarray  # [m s-2]
    mass: np.ndarray  # node mass [kg]
    r_basic: np.ndarray  # shell boundaries [m]
    volume: np.ndarray  # shell volumes [m3]
    area_top: np.ndarray  # upper-face areas [m2]
    centre: np.ndarray  # shell centres [m]
    psi_transition: np.ndarray  # volume fraction at the transition [1]
    rfront_loc: float

    @property
    def n(self) -> int:
        """Number of staggered nodes."""
        return int(self.x.size)


def _mesh(
    melt_fraction, rho_solid, rho_melt, gravity, r_basic, mass, rfront_loc, phi_min
) -> _Mesh:
    """Validated profiles and node classes of one step."""
    if not 0.0 < rfront_loc <= 1.0:
        raise ValueError(f'rfront_loc must lie in (0, 1], got {rfront_loc!r}')
    x = np.clip(np.asarray(melt_fraction, dtype=float).ravel(), 0.0, 1.0)
    n = x.size
    rho_s = _node_array(rho_solid, n, 'rho_solid')
    rho_l = _node_array(rho_melt, n, 'rho_melt')
    r_b = np.asarray(r_basic, dtype=float).ravel()
    if r_b.size != n + 1:
        raise ValueError(f'r_basic must hold {n + 1} shell boundaries, got {r_b.size}')
    volume, area_top, centre = shell_geometry(r_b)
    phi = melt_volume_fraction(x, rho_s, rho_l)
    return _Mesh(
        x=x,
        phi=phi,
        status=classify_nodes(x, phi, rfront_loc, phi_min),
        rho_s=rho_s,
        rho_l=rho_l,
        gravity=_node_array(gravity, n, 'gravity'),
        mass=_node_array(mass, n, 'mass'),
        r_basic=r_b,
        volume=volume,
        area_top=area_top,
        centre=centre,
        psi_transition=melt_volume_fraction(np.full(n, rfront_loc), rho_s, rho_l),
        rfront_loc=float(rfront_loc),
    )


@dataclass
class _Transitions:
    """Which nodes a step moves between classes, and the pore-melt mass they free."""

    locking: np.ndarray  # nodes whose pore melt locks now [bool]
    bound: np.ndarray  # nodes that crossed the whole mush in the step [bool]
    freeze: dict[str, np.ndarray]  # pore-melt mass reaching the floor [kg]
    release: dict[str, float]  # mass returned to the melt reservoir [kg]
    x_prev: np.ndarray  # melt fraction at the previous classification [1]


def _drain(new: LayerState, mesh: _Mesh, species, dt_yr, params) -> tuple:
    """Drain the open runs of the previous mush and carry their locked volatiles.

    Returns the drainage outcome, the open and sealed runs, and the mass of
    each species that left through the tops of the open runs.
    """
    n = mesh.n
    active = (new.status == STATUS_MUSH) & np.isfinite(new.psi)
    open_runs, sealed = [], []
    for run in _contiguous_runs(active):
        above = int(run[-1]) + 1
        if above >= n or new.status[above] == STATUS_LIQUID:
            open_runs.append(run)
        else:
            sealed.append(run)
    drho = mesh.rho_s - mesh.rho_l
    grain, melt_visc, mush_visc, substep_fraction, max_substeps = params

    def speed_of(p, index):
        return darcy_velocity(p, grain, drho[index], mesh.gravity[index], melt_visc)

    tau_s = compaction_times(
        open_runs, mesh.r_basic, mesh.centre, drho, mesh.gravity, mush_visc
    )
    tracer = np.array([new.retained[sp] / mesh.volume for sp in species]).reshape(
        len(species), n
    )
    drained = drain_runs(
        new.psi,
        open_runs,
        mesh.volume,
        mesh.area_top,
        speed_of,
        tau_s,
        dt_yr * SECS_PER_YEAR,
        substep_fraction=substep_fraction,
        max_substeps=max_substeps,
        tracer=tracer if species else None,
    )
    moved = np.zeros(n, dtype=bool)
    for run in open_runs:
        moved[run] = True
    release = {}
    for k, sp in enumerate(species):
        after = np.where(moved, drained.tracer[k] * mesh.volume, new.retained[sp])
        # Everything that left the open runs went out through their tops.
        release[sp] = max(float(np.sum(new.retained[sp][moved] - after[moved])), 0.0)
        new.retained[sp] = after
    return drained, open_runs, sealed, release


def _evolve(state: LayerState, mesh: _Mesh, species, time_prev, time_now, params):
    """Drain the previous mush, then move every node to its class of this step.

    Returns the new state, whose carried and frozen pore melt is settled but
    whose new locks are not yet applied, the step record, and the transitions.
    """
    n = mesh.n
    new = state.copy()
    for sp in species:
        new.retained.setdefault(sp, np.zeros(n))
    dt_yr = max(time_now - time_prev, 0.0)
    active = (state.status == STATUS_MUSH) & np.isfinite(state.psi)
    drained, open_runs, sealed, release = _drain(new, mesh, species, dt_yr, params)
    new.psi = drained.psi
    new.t_darcy = new.t_darcy + drained.t_darcy / SECS_PER_YEAR
    new.t_matrix = new.t_matrix + drained.t_matrix / SECS_PER_YEAR
    node_branch = np.full(n, BRANCH_NONE)
    for run in open_runs:
        node_branch[run] = np.where(
            drained.t_darcy[run] >= drained.t_matrix[run], BRANCH_DARCY, BRANCH_MATRIX
        )
    for run in sealed:
        new.t_held[run] += dt_yr
        node_branch[run] = BRANCH_GUARD
    reasons = []
    if sealed:
        reasons.append(f'{sum(r.size for r in sealed)} sealed below a solid layer')

    # Nodes that reached the porosity floor. Those that were mush drained over
    # the step and keep what they hold; those that were still liquid crossed
    # the whole mush within it and bury the melt of the transition. A mush node
    # with no retained melt on record, as a damaged snapshot could leave, has
    # no drainage history either and is bounded like one never seen.
    floor = mesh.status == STATUS_SOLID
    stray = (state.status == STATUS_MUSH) & ~np.isfinite(state.psi)
    exit_drained = active & floor
    exit_bound = ((state.status == STATUS_LIQUID) | stray) & floor
    exited = np.flatnonzero(exit_drained | exit_bound)
    drained_exit = exit_drained[exited]
    # Drainage removes melt, and the volatiles in it, in proportion to the
    # retained volume, so the retained melt mass is the entry melt mass scaled
    # by psi / psi_entry.
    entry = new.psi_entry[exited]
    f_entry = mass_fraction_of_volume(entry, mesh.rho_l[exited], mesh.rho_s[exited])
    scaled = np.divide(
        f_entry * new.psi[exited], entry, out=np.zeros(exited.size), where=entry > 0.0
    )
    trapped = np.where(drained_exit, scaled, mesh.rfront_loc)
    spent = new.t_darcy[exited] + new.t_matrix[exited]
    bounded = ~drained_exit | (new.t_held[exited] > spent)
    branch = np.where(
        bounded,
        BRANCH_GUARD,
        np.where(new.t_darcy[exited] >= new.t_matrix[exited], BRANCH_DARCY, BRANCH_MATRIX),
    )
    residence = np.where(drained_exit, time_now - new.t_entry[exited], np.nan)
    if np.any(exit_bound):
        reasons.append(f'{int(exit_bound.sum())} crossed the whole mush within one step')

    # Pore melt reaching the floor stays; a node that remelted back into the
    # ocean returns what its pore melt holds.
    remelt = active & (mesh.status == STATUS_LIQUID)
    freeze = {}
    for sp in species:
        freeze[sp] = np.where(exit_drained, new.retained[sp], 0.0)
        release[sp] = release.get(sp, 0.0) + float(np.sum(new.retained[sp][remelt]))
        new.retained[sp] = np.where(exit_drained | remelt, 0.0, new.retained[sp])

    # Every node that leaves the mush gives up its drainage state.
    leave = active & (mesh.status != STATUS_MUSH)
    new.psi[leave] = np.nan
    new.psi_entry[leave] = np.nan
    new.t_entry[leave] = np.nan
    for arr in (new.t_darcy, new.t_matrix, new.t_held):
        arr[leave | exit_bound] = 0.0

    from_above = (state.status == STATUS_LIQUID) & (mesh.status == STATUS_MUSH)
    from_below = ((state.status == STATUS_SOLID) | stray) & (mesh.status == STATUS_MUSH)
    entering = from_above | from_below
    new.psi[from_above] = mesh.psi_transition[from_above]
    new.psi[from_below] = mesh.phi[from_below]
    new.psi_entry[entering] = new.psi[entering]
    # Time of the crossing of the transition, interpolated within the step on
    # the melt fraction, so the residence time of a node is resolved finer than
    # one step.
    x_prev = state.melt_fraction
    drop = x_prev - mesh.x
    frac = np.divide(x_prev - mesh.rfront_loc, drop, out=np.ones(n), where=drop > 0.0)
    crossing = time_prev + np.clip(frac, 0.0, 1.0) * dt_yr
    new.t_entry[from_above] = crossing[from_above]
    new.t_entry[from_below] = time_now
    for arr in (new.t_darcy, new.t_matrix, new.t_held):
        arr[entering] = 0.0
    new.status = mesh.status.copy()
    new.melt_fraction = mesh.x.copy()

    step = LayerStep(
        exited=exited,
        trapped=trapped,
        branch=branch,
        residence=residence,
        node_branch=node_branch,
        n_entered=int(entering.sum()),
        n_remelted=int(remelt.sum()),
        n_substeps=drained.n_substeps,
        truncated=drained.elapsed < dt_yr * SECS_PER_YEAR * (1.0 - 1.0e-12) and bool(open_runs),
        held=reasons,
        matrix_speed=drained.matrix_speed if open_runs else float('nan'),
        returned=drained.returned,
    )
    moves = _Transitions(
        locking=from_above, bound=exit_bound, freeze=freeze, release=release, x_prev=x_prev
    )
    return new, step, moves


def _settle(
    new: LayerState,
    mesh: _Mesh,
    moves: _Transitions,
    step: LayerStep,
    species,
    conc: dict[str, float],
    part: dict[str, float],
    liquid: dict[str, float],
    crystallised: float,
) -> None:
    """Lock, bury and cap the volatiles of one step, and write them into the state.

    Entering nodes lock ``(1 - D) C F m`` in their pore melt, nodes that crossed
    the whole mush bury the same for the melt of the transition, and the
    crystals bury ``D C dM_RM`` spread by the mass each node crystallised. These
    new locks are scaled together per species so they never exceed what the
    melt holds after this step's returns. Frozen pore melt was locked already
    and moves to the buried record unscaled.
    """
    held_melt = mass_fraction_of_volume(np.nan_to_num(new.psi), mesh.rho_l, mesh.rho_s)
    held_melt = held_melt * mesh.mass
    transition_melt = mesh.rfront_loc * mesh.mass
    weights = mesh.mass * np.maximum(moves.x_prev - mesh.x, 0.0)
    if not np.sum(weights) > 0.0:
        # The global melt fraction fell while no node's did, as on the step a
        # held melt fraction is released; the crystals are spread by solid mass.
        weights = mesh.mass * (1.0 - mesh.x)
    total_weight = float(np.sum(weights))
    for sp in species:
        c = conc.get(sp, 0.0)
        d = part.get(sp, 0.0)
        melt_share = max(1.0 - d, 0.0) * c
        lock = np.where(moves.locking, melt_share * held_melt, 0.0)
        bury = np.where(moves.bound, melt_share * transition_melt, 0.0)
        if crystallised > 0.0 and total_weight > 0.0:
            bury = bury + d * c * crystallised * weights / total_weight
        release = float(moves.release.get(sp, 0.0))
        wanted = float(np.sum(lock) + np.sum(bury))
        available = max(float(liquid.get(sp, 0.0)), 0.0) + release
        scale = 1.0
        if wanted > available:
            scale = available / wanted
            step.capped.append(sp)
        new.retained[sp] = new.retained.get(sp, np.zeros(mesh.n)) + scale * lock
        buried = scale * bury + moves.freeze.get(sp, np.zeros(mesh.n))
        for element, share in _split(sp).items():
            new.kg[element] = new.kg.get(element, np.zeros(mesh.n)) + share * buried
        step.locked_kg[sp] = scale * wanted
        step.released_kg[sp] = release
        step.moved_kg[sp] = scale * wanted - release


def advance_layers(
    state: LayerState | None,
    *,
    time_prev: float,
    time_now: float,
    melt_fraction: np.ndarray,
    rho_solid: np.ndarray,
    rho_melt: np.ndarray,
    gravity: np.ndarray,
    r_basic: np.ndarray,
    mass: np.ndarray,
    rfront_loc: float,
    phi_min: float,
    grain_size: float,
    melt_visc: float,
    mush_visc: float,
    concentration: dict[str, float] | None = None,
    partition: dict[str, float] | None = None,
    liquid: dict[str, float] | None = None,
    crystallised: float = 0.0,
    substep_fraction: float = SUBSTEP_FRACTION,
    max_substeps: int = MAX_SUBSTEPS,
) -> tuple[LayerState, LayerStep]:
    """Advance the per-node state and its volatiles over one coupling step.

    Drains the mush of the previous step, then classifies every node on the
    current profiles and moves each node between the liquid, the mush and the
    solid, locking, carrying, returning and burying volatiles as the module
    docstring states. Returns a new state, leaving ``state`` untouched, and
    what the step did; ``LayerStep.moved_kg`` is the net mass of each species
    the caller moves from the melt into the solid reservoir.

    Parameters
    ----------
    concentration : dict
        Melt concentration of each species the chemistry solver last computed
        [kg kg-1].
    partition : dict
        Crystal/melt partition coefficient of each species [1].
    liquid : dict
        Mass of each species in the melt reservoir [kg], the cap on new locks.
    crystallised : float
        Mantle mass crystallised over the step [kg], ``dM_RM``.

    A ``state`` of ``None``, or one sized for another mesh, starts the record on
    the current profiles without draining: the mush it finds locks its pore
    melt at the current concentrations.
    """
    mesh = _mesh(
        melt_fraction, rho_solid, rho_melt, gravity, r_basic, mass, rfront_loc, phi_min
    )
    n = mesh.n
    conc = {sp: max(float(c), 0.0) for sp, c in (concentration or {}).items()}
    part = {sp: max(float(d), 0.0) for sp, d in (partition or {}).items()}
    params = (grain_size, melt_visc, mush_visc, substep_fraction, max_substeps)

    if state is None or state.size != n:
        if state is not None:
            log.warning(
                'Trapping: the per-node state covers %d nodes but the mesh has %d; '
                'the layer drainage restarts from the current mush.',
                state.size,
                n,
            )
        species = sorted(conc)
        new = LayerState.initial(mesh.status, mesh.phi, mesh.x, time_now, species)
        empty = np.array([], dtype=int)
        step = LayerStep(
            exited=empty,
            trapped=np.array([]),
            branch=empty,
            residence=np.array([]),
            node_branch=np.full(n, BRANCH_NONE),
        )
        moves = _Transitions(
            locking=mesh.status == STATUS_MUSH,
            bound=np.zeros(n, dtype=bool),
            freeze={},
            release={},
            x_prev=mesh.x,
        )
    else:
        species = sorted(set(conc) | set(state.retained))
        new, step, moves = _evolve(
            state, mesh, species, float(time_prev), float(time_now), params
        )

    _settle(new, mesh, moves, step, species, conc, part, liquid or {}, crystallised)
    mush = new.status == STATUS_MUSH
    step.n_dense = int(np.sum(mush & (mesh.rho_s - mesh.rho_l <= 0.0)))
    _describe(step, new, mesh, grain_size, melt_visc, mush_visc)
    return new, step


def _describe(step, state, mesh, grain, melt_visc, mush_visc):
    """Front-like diagnostics of the thickest mush run, for comparison runs."""
    step.n_mush = int(np.sum(state.status == STATUS_MUSH))
    run, extent = _thickest_run(state.status, mesh.r_basic)
    if run is None:
        return
    top = int(run[-1])
    drho = mesh.rho_s - mesh.rho_l
    step.thickness = extent
    step.r_base = float(mesh.r_basic[run[0]])
    step.rho_solid_base = float(mesh.rho_s[run[0]])
    step.node_thickness = extent / run.size
    psi_top = float(state.psi[top]) if np.isfinite(state.psi[top]) else 0.0
    w_top = darcy_velocity(
        psi_top, grain, float(drho[top]), float(mesh.gravity[top]), melt_visc
    )
    step.tau_d = extent / w_top if w_top > 0.0 else float('inf')
    load = abs(float(np.mean(drho[run]))) * float(np.mean(mesh.gravity[run])) * extent
    step.tau_s = mush_visc / load if load > 0.0 else float('inf')


def add_locked_mass(state: LayerState, species_kg: dict[str, np.ndarray]) -> None:
    """Add per-node buried species masses to the per-element record [kg]."""
    for species, kg in species_kg.items():
        for element, share in _split(species).items():
            if element not in state.kg:
                state.kg[element] = np.zeros(state.size)
            state.kg[element] = state.kg[element] + share * np.asarray(kg, dtype=float)


def write_state(path: str, state: LayerState) -> None:
    """Add the per-node state to an interior snapshot, beside the solver fields.

    The snapshot must already hold the ``staggered`` dimension the solver
    writes, sized for this state. Variables present from an earlier write of
    the same file are overwritten.
    """
    import netCDF4 as nc

    with nc.Dataset(path, mode='a') as ds:
        dim = ds.dimensions.get('staggered')
        if dim is None or len(dim) != state.size:
            raise ValueError(
                f'{path} has no staggered dimension of {state.size} nodes to hold '
                'the per-node trapping state'
            )
        columns = {name: getattr(state, name) for name in _STATE_FIELDS}
        units = dict(_STATE_FIELDS)
        for element, kg in state.kg.items():
            columns[f'kg_{element}'] = kg
            units[f'kg_{element}'] = 'kg'
        for species, kg in state.retained.items():
            columns[f'retained_{species}'] = kg
            units[f'retained_{species}'] = 'kg'
        for name, values in columns.items():
            key = _snapshot_name(name)
            var = ds.variables.get(key)
            if var is None:
                var = ds.createVariable(key, np.float64, ('staggered',))
            var[:] = np.asarray(values, dtype=float)
            var.units = units[name]


def read_state(path: str, n: int) -> LayerState | None:
    """The per-node state stored in an interior snapshot, or ``None``.

    ``None`` when the snapshot carries no state, as one written before the
    layer drainage ran does, or a state sized for another mesh.
    """
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        names = [_snapshot_name(f) for f in _STATE_FIELDS]
        if any(name not in ds.variables for name in names):
            return None

        def column(key, fill):
            return np.ma.filled(ds.variables[key][:].astype(float), fill)

        arrays = {f: column(_snapshot_name(f), np.nan) for f in _STATE_FIELDS}
        kg = {}
        for element in RECORD_ELEMENTS:
            key = _snapshot_name(f'kg_{element}')
            if key in ds.variables:
                kg[element] = column(key, 0.0)
        retained = {}
        for species in vol_list:
            key = _snapshot_name(f'retained_{species}')
            if key in ds.variables:
                retained[species] = column(key, 0.0)
    stored = list(arrays.values()) + list(kg.values()) + list(retained.values())
    if any(np.size(a) != n for a in stored):
        return None
    for element in RECORD_ELEMENTS:
        kg.setdefault(element, np.zeros(n))
    return LayerState(
        status=np.rint(arrays['status']).astype(int),
        psi=arrays['psi'],
        psi_entry=arrays['psi_entry'],
        t_entry=arrays['t_entry'],
        melt_fraction=arrays['melt_fraction'],
        t_darcy=arrays['t_darcy'],
        t_matrix=arrays['t_matrix'],
        t_held=arrays['t_held'],
        kg=kg,
        retained=retained,
    )
