"""Per-node record of the layer scheme, and the volatiles it traps.

:mod:`proteus.outgas.layer_drainage` drains the pore melt of the mush layers
node by node. This module carries what that drainage needs from one coupling
step to the next: the pore melt every node still holds and the volatile mass
trapped at it. It moves every node between the liquid, mush and solid phase
states and keeps the record consistent with them;
:mod:`proteus.outgas.node_record_io` saves it in the interior snapshots so a
resumed run continues it.

Vocabulary
----------
A node is one cell of the 1-D interior mesh. A layer is a contiguous set of
nodes in one phase state: a liquid, mush or solid layer. Volatiles held at a
mush node are buried; volatiles held at a solid node are locked. Trapped is
both together, the mass the ``_kg_trapped`` columns hold.

Volatile budget
---------------
A node's pore melt leaves contact with the magma ocean when its crystal
framework connects: melt drains upward only, so nothing flows back from the
ocean to renew it. Its volatiles are therefore buried from that moment, at the
concentration the chemistry solver held then, and carried with the melt as it
drains. A node entering the mush from above buries

    R_Z = (1 - D_Z) C_Z F m

of each species, with ``C_Z`` that step's melt concentration, ``m`` the node
mass and ``F`` the melt mass fraction of its pore melt, exactly ``rfront_loc``
on entry from above. Drainage removes melt, and the volatiles in it, in
proportion to the pore melt volume. Melt passing from one node into the next
takes its share of the buried mass with it; melt that drains out of the top of
a mush layer rejoins the ocean and its volatiles return to the melt reservoir.
What a node holds when it reaches the porosity floor is locked. Every step's
crystals take ``D_Z C_Z dM_RM``, spread over the nodes by the mass each
crystallised. Over the life of a node the two add up to
``[(1 - F) D_Z + F] C m``, Sim et al. (2024) Eq. 6 applied to the node's own
mass. The noble gases follow the same rules at ``D_Z = 0``.

The record keeps two stores per species: ``pore``, the volatiles dissolved in
the pore melt, which drain with it, and ``crystal``, the volatiles in the
crystals and in pore melt that froze at the floor, which stay at their node.

The new burials of a step, crystals included, are capped together at what the
melt reservoir holds after this step's returns, so the net mass a step moves
into the solid reservoir can be negative, when returns exceed new burials.

Remelting
---------
Remelting is followed node by node. A node that becomes liquid again, from the
mush or from the solid, returns everything trapped at it to the melt
reservoir: its solid has rejoined the magma ocean. A solid node that re-enters
the mush from below makes its new pore melt from its own locked material, so
it buries nothing new and keeps its crystal store until it becomes liquid. The
record is kept per species, so a release returns exactly the species that went
in.

Trapped mass that the record does not hold, as after resuming a run whose
snapshot carries no record, is assigned to the nodes that are not liquid in
proportion to their mass (:func:`adopt_unrecorded`), so it remelts with them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from proteus.outgas.compaction import (
    BRANCH_DARCY,
    BRANCH_GUARD,
    BRANCH_MATRIX,
    BRANCH_NONE,
    darcy_velocity,
)
from proteus.outgas.layer_drainage import (
    MAX_SUBSTEPS,
    PHASE_LIQUID,
    PHASE_MUSH,
    PHASE_SOLID,
    SUBSTEP_FRACTION,
    DrainageParams,
    classify_nodes,
    compaction_times,
    drain_mush_layers,
    mass_fraction_of_volume,
    melt_volume_fraction,
    mush_layers,
    node_array,
    node_geometry,
    thickest_mush_layer,
)
from proteus.utils.constants import noble_gases, secs_per_year, vol_list

log = logging.getLogger('fwl.' + __name__)

# Species the per-node record carries: the volatiles and the noble gases.
RECORD_SPECIES = tuple(vol_list) + tuple(noble_gases)


@dataclass
class NodeRecord:
    """Per-node trapping record on the staggered nodes of the interior mesh."""

    phase: np.ndarray  # PHASE_* code at the last classification
    psi: np.ndarray  # pore-melt volume fraction [1]; NaN outside the mush
    psi_entry: np.ndarray  # pore-melt volume fraction on entering the mush [1]
    t_entry: np.ndarray  # time the node entered the mush [yr]
    melt_fraction: np.ndarray  # solver melt fraction at the last classification [1]
    t_darcy: np.ndarray  # time spent percolation-limited in this residence [yr]
    t_matrix: np.ndarray  # time spent compaction-limited in this residence [yr]
    t_held: np.ndarray  # time spent unable to drain, sealed [yr]
    crystal: dict[str, np.ndarray] = field(default_factory=dict)  # per species [kg]
    pore: dict[str, np.ndarray] = field(default_factory=dict)  # in pore melt [kg]

    @property
    def size(self) -> int:
        """Number of staggered nodes the record covers."""
        return int(self.phase.size)

    def copy(self) -> NodeRecord:
        """Independent copy, so a step never mutates the record it started from."""
        return NodeRecord(
            phase=self.phase.copy(),
            psi=self.psi.copy(),
            psi_entry=self.psi_entry.copy(),
            t_entry=self.t_entry.copy(),
            melt_fraction=self.melt_fraction.copy(),
            t_darcy=self.t_darcy.copy(),
            t_matrix=self.t_matrix.copy(),
            t_held=self.t_held.copy(),
            crystal={s: v.copy() for s, v in self.crystal.items()},
            pore={s: v.copy() for s, v in self.pore.items()},
        )

    def species_profile(self, species: str) -> np.ndarray:
        """Mass of one species trapped at every node, in crystals or pore melt [kg]."""
        zero = np.zeros(self.size)
        return np.asarray(self.crystal.get(species, zero), dtype=float) + np.asarray(
            self.pore.get(species, zero), dtype=float
        )

    def trapped(self, species: str) -> float:
        """Mass of one species trapped in the mesh, summed over nodes [kg]."""
        return float(np.sum(self.species_profile(species)))

    @classmethod
    def initial(
        cls,
        phase: np.ndarray,
        volume_fraction: np.ndarray,
        melt_fraction: np.ndarray,
        time: float,
    ) -> NodeRecord:
        """Record of a mesh first seen at ``time`` [yr], with nothing trapped yet.

        Mush nodes start with their current volume fraction: how much they
        drained before they were first seen is unknown, and this is the most
        they can still hold. The caller buries their volatiles.
        """
        n = int(np.size(phase))
        mush = np.asarray(phase) == PHASE_MUSH
        psi = np.where(mush, volume_fraction, np.nan)
        return cls(
            phase=np.asarray(phase, dtype=int).copy(),
            psi=psi,
            psi_entry=psi.copy(),
            t_entry=np.where(mush, float(time), np.nan),
            melt_fraction=np.asarray(melt_fraction, dtype=float).copy(),
            t_darcy=np.zeros(n),
            t_matrix=np.zeros(n),
            t_held=np.zeros(n),
            crystal={s: np.zeros(n) for s in RECORD_SPECIES},
            pore={s: np.zeros(n) for s in RECORD_SPECIES},
        )


@dataclass
class LayerStep:
    """What one coupling step of the layer scheme did."""

    exited: np.ndarray  # nodes that reached the porosity floor this step
    trapped: np.ndarray  # trapped melt mass fraction, F_entry psi / psi_entry [1]
    branch: np.ndarray  # drainage branch code of each exited node
    residence: np.ndarray  # time each exited node spent in the mush [yr]
    node_branch: np.ndarray  # branch code of every node that drained this step
    n_mush: int = 0
    n_entered: int = 0
    n_remelted: int = 0  # nodes that became liquid again this step
    n_dense: int = 0  # mush nodes whose tables put the melt above the solid density
    n_substeps: int = 0
    held: list[str] = field(default_factory=list)  # why nodes could not drain
    thickness: float = float('nan')  # radial extent of the thickest mush layer [m]
    r_base: float = float('nan')  # base radius of that layer [m]
    rho_solid_base: float = float('nan')  # solid density at that base [kg m-3]
    node_thickness: float = float('nan')  # mean node thickness of that layer [m]
    tau_d: float = float('nan')  # percolation time of that layer at its top [s]
    tau_s: float = float('nan')  # matrix time of that layer [s]
    matrix_speed: float = float('nan')  # largest melt flux per area [m s-1]
    moved_kg: dict[str, float] = field(default_factory=dict)  # net into the solid [kg]
    buried_kg: dict[str, float] = field(default_factory=dict)  # newly buried [kg]
    released_kg: dict[str, float] = field(default_factory=dict)  # returned to melt [kg]
    remelt_kg: dict[str, float] = field(default_factory=dict)  # of it, by remelting [kg]
    capped: list[str] = field(default_factory=list)  # species limited by the melt


@dataclass
class _Mesh:
    """Profiles of one step on the staggered nodes, shared by the stage functions."""

    x: np.ndarray  # solver melt fraction [1]
    phi: np.ndarray  # melt volume fraction it implies [1]
    phase: np.ndarray  # phase state of every node now
    rho_s: np.ndarray  # solid phase-boundary density [kg m-3]
    rho_l: np.ndarray  # melt phase-boundary density [kg m-3]
    gravity: np.ndarray  # [m s-2]
    mass: np.ndarray  # node mass [kg]
    r_basic: np.ndarray  # node boundaries [m]
    volume: np.ndarray  # node volumes [m3]
    area_top: np.ndarray  # upper-face areas [m2]
    centre: np.ndarray  # node centres [m]
    psi_transition: np.ndarray  # volume fraction at the transition [1]
    rfront_loc: float

    @property
    def n(self) -> int:
        """Number of staggered nodes."""
        return int(self.x.size)


def _mesh(melt_fraction, rho_solid, rho_melt, gravity, r_basic, mass, rfront_loc, phi_min):
    """Validated profiles and phase states of one step."""
    if not 0.0 < rfront_loc <= 1.0:
        raise ValueError(f'rfront_loc must lie in (0, 1], got {rfront_loc!r}')
    x = np.clip(np.asarray(melt_fraction, dtype=float).ravel(), 0.0, 1.0)
    n = x.size
    rho_s = node_array(rho_solid, n, 'rho_solid')
    rho_l = node_array(rho_melt, n, 'rho_melt')
    r_b = np.asarray(r_basic, dtype=float).ravel()
    if r_b.size != n + 1:
        raise ValueError(f'r_basic must hold {n + 1} node boundaries, got {r_b.size}')
    volume, area_top, centre = node_geometry(r_b)
    phi = melt_volume_fraction(x, rho_s, rho_l)
    return _Mesh(
        x=x,
        phi=phi,
        phase=classify_nodes(x, phi, rfront_loc, phi_min),
        rho_s=rho_s,
        rho_l=rho_l,
        gravity=node_array(gravity, n, 'gravity'),
        mass=node_array(mass, n, 'mass'),
        r_basic=r_b,
        volume=volume,
        area_top=area_top,
        centre=centre,
        psi_transition=melt_volume_fraction(np.full(n, rfront_loc), rho_s, rho_l),
        rfront_loc=float(rfront_loc),
    )


@dataclass
class _Transitions:
    """Which nodes a step moves between phase states, and the mass they free or bury."""

    burying: np.ndarray  # nodes whose pore melt is buried now [bool]
    bound: np.ndarray  # nodes that crossed the whole mush in the step [bool]
    freeze: dict[str, np.ndarray]  # pore-melt volatiles locked at the floor [kg]
    release: dict[str, float]  # mass returned to the melt reservoir [kg]
    remelt: dict[str, float]  # of it, by nodes becoming liquid [kg]
    x_prev: np.ndarray  # melt fraction at the previous classification [1]


def _drain(new: NodeRecord, mesh: _Mesh, species, dt_yr, params: DrainageParams) -> tuple:
    """Drain the open mush layers of the previous step and carry their volatiles.

    Returns the drainage outcome, the open and sealed mush layers, and the mass
    of each species that left through the tops of the open layers.
    """
    n = mesh.n
    active = (new.phase == PHASE_MUSH) & np.isfinite(new.psi)
    open_layers, sealed = [], []
    for layer in mush_layers(active):
        above = int(layer[-1]) + 1
        if above >= n or new.phase[above] == PHASE_LIQUID:
            open_layers.append(layer)
        else:
            sealed.append(layer)
    drho = mesh.rho_s - mesh.rho_l

    def speed_of(p, index):
        return darcy_velocity(
            p,
            params.grain_size,
            drho[index],
            mesh.gravity[index],
            params.melt_visc,
            buoyancy=params.buoyancy,
        )

    tau_s = compaction_times(
        open_layers, mesh.r_basic, mesh.centre, drho, mesh.gravity, params.mush_visc
    )
    tracer = np.array([new.pore[sp] / mesh.volume for sp in species]).reshape(len(species), n)
    drained = drain_mush_layers(
        new.psi,
        open_layers,
        mesh.volume,
        mesh.area_top,
        speed_of,
        tau_s,
        dt_yr * secs_per_year,
        substep_fraction=params.substep_fraction,
        max_substeps=params.max_substeps,
        tracer=tracer if species else None,
    )
    moved = np.zeros(n, dtype=bool)
    for layer in open_layers:
        moved[layer] = True
    release = {}
    for k, sp in enumerate(species):
        after = np.where(moved, drained.tracer[k] * mesh.volume, new.pore[sp])
        # Everything that left the open layers went out through their tops.
        release[sp] = max(float(np.sum(new.pore[sp][moved] - after[moved])), 0.0)
        new.pore[sp] = after
    return drained, open_layers, sealed, release


def _exits(record, new, mesh, active, time_now):
    """Nodes that reached the porosity floor, their trapped fraction and branch."""
    floor = mesh.phase == PHASE_SOLID
    # A mush node with no pore melt on record, as a damaged snapshot could
    # leave, has no drainage history either and is bounded like one never seen.
    stray = (record.phase == PHASE_MUSH) & ~np.isfinite(record.psi)
    exit_drained = active & floor
    exit_bound = ((record.phase == PHASE_LIQUID) | stray) & floor
    exited = np.flatnonzero(exit_drained | exit_bound)
    drained_exit = exit_drained[exited]
    # Drainage removes melt in proportion to the pore volume, so the pore melt
    # mass is the entry melt mass scaled by psi / psi_entry.
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
    return exited, trapped, branch, residence, exit_drained, exit_bound, stray


def _branches(new, drained, open_layers, sealed, dt_yr):
    """Branch code of every node that drained or was held, and why nodes were held."""
    node_branch = np.full(new.size, BRANCH_NONE)
    for layer in open_layers:
        node_branch[layer] = np.where(
            drained.t_darcy[layer] >= drained.t_matrix[layer], BRANCH_DARCY, BRANCH_MATRIX
        )
    for layer in sealed:
        new.t_held[layer] += dt_yr
        node_branch[layer] = BRANCH_GUARD
    reasons = []
    if sealed:
        reasons.append(f'{sum(lay.size for lay in sealed)} sealed below a solid layer')
    return node_branch, reasons


def _freeze_and_remelt(record, new, mesh, species, exit_drained, release):
    """Lock the pore melt reaching the floor and free the nodes turning liquid.

    Returns the frozen pore-melt volatiles per node, the mass returned by
    remelting per species, and which nodes remelted.
    """
    remelt = (record.phase != PHASE_LIQUID) & (mesh.phase == PHASE_LIQUID)
    freeze, remelted = {}, {}
    for sp in species:
        freeze[sp] = np.where(exit_drained, new.pore[sp], 0.0)
        remelted[sp] = float(np.sum(new.pore[sp][remelt] + new.crystal[sp][remelt]))
        release[sp] = release.get(sp, 0.0) + remelted[sp]
        new.pore[sp] = np.where(exit_drained | remelt, 0.0, new.pore[sp])
        new.crystal[sp] = np.where(remelt, 0.0, new.crystal[sp])
    return freeze, remelted, remelt


def _reclassify(record, new, mesh, active, exit_bound, stray, time_prev, time_now):
    """Move every node to its phase state of this step; return the entering nodes."""
    n = mesh.n
    dt_yr = max(time_now - time_prev, 0.0)
    leave = active & (mesh.phase != PHASE_MUSH)
    new.psi[leave] = np.nan
    new.psi_entry[leave] = np.nan
    new.t_entry[leave] = np.nan
    for arr in (new.t_darcy, new.t_matrix, new.t_held):
        arr[leave | exit_bound] = 0.0

    from_above = (record.phase == PHASE_LIQUID) & (mesh.phase == PHASE_MUSH)
    from_below = ((record.phase == PHASE_SOLID) | stray) & (mesh.phase == PHASE_MUSH)
    entering = from_above | from_below
    new.psi[from_above] = mesh.psi_transition[from_above]
    new.psi[from_below] = mesh.phi[from_below]
    new.psi_entry[entering] = new.psi[entering]
    # Crossing time of the transition, interpolated within the step on the melt
    # fraction, so a node's residence is resolved finer than one step.
    x_prev = record.melt_fraction
    drop = x_prev - mesh.x
    frac = np.divide(x_prev - mesh.rfront_loc, drop, out=np.ones(n), where=drop > 0.0)
    crossing = time_prev + np.clip(frac, 0.0, 1.0) * dt_yr
    new.t_entry[from_above] = crossing[from_above]
    new.t_entry[from_below] = time_now
    for arr in (new.t_darcy, new.t_matrix, new.t_held):
        arr[entering] = 0.0
    new.phase = mesh.phase.copy()
    new.melt_fraction = mesh.x.copy()
    return from_above, entering


def _evolve(record: NodeRecord, mesh: _Mesh, species, time_prev, time_now, params):
    """Drain the previous mush, then move every node to its phase state of this step.

    Returns the new record, whose carried, frozen and remelted mass is settled
    but whose new burials are not yet applied, the step record, and the transitions.
    """
    new = record.copy()
    for sp in species:
        new.pore.setdefault(sp, np.zeros(mesh.n))
        new.crystal.setdefault(sp, np.zeros(mesh.n))
    dt_yr = max(time_now - time_prev, 0.0)
    active = (record.phase == PHASE_MUSH) & np.isfinite(record.psi)
    drained, open_layers, sealed, release = _drain(new, mesh, species, dt_yr, params)
    new.psi = drained.psi
    new.t_darcy = new.t_darcy + drained.t_darcy / secs_per_year
    new.t_matrix = new.t_matrix + drained.t_matrix / secs_per_year
    node_branch, reasons = _branches(new, drained, open_layers, sealed, dt_yr)

    exited, trapped, branch, residence, exit_drained, exit_bound, stray = _exits(
        record, new, mesh, active, time_now
    )
    if np.any(exit_bound):
        reasons.append(f'{int(exit_bound.sum())} crossed the whole mush within one step')
    freeze, remelted, remelt = _freeze_and_remelt(
        record, new, mesh, species, exit_drained, release
    )
    from_above, entering = _reclassify(
        record, new, mesh, active, exit_bound, stray, time_prev, time_now
    )

    step = LayerStep(
        exited=exited,
        trapped=trapped,
        branch=branch,
        residence=residence,
        node_branch=node_branch,
        n_entered=int(entering.sum()),
        n_remelted=int(remelt.sum()),
        n_substeps=drained.n_substeps,
        held=reasons,
        matrix_speed=drained.matrix_speed if open_layers else float('nan'),
    )
    moves = _Transitions(
        burying=from_above,
        bound=exit_bound,
        freeze=freeze,
        release=release,
        remelt=remelted,
        x_prev=record.melt_fraction,
    )
    return new, step, moves


def _settle(new, mesh, moves, step, species, conc, part, liquid, crystallised) -> None:
    """Bury and cap the volatiles of one step, and write them into the record.

    Entering nodes bury ``(1 - D) C F m`` in their pore melt, nodes that crossed
    the whole mush lock the same for the melt of the transition, and the
    crystals take ``D C dM_RM`` spread by the mass each node crystallised. These
    new burials are scaled together per species so they never exceed what the
    melt holds after this step's returns. Frozen pore melt was buried already
    and moves to the crystal store unscaled.
    """
    held_melt = mass_fraction_of_volume(np.nan_to_num(new.psi), mesh.rho_l, mesh.rho_s)
    held_melt = held_melt * mesh.mass
    transition_melt = mesh.rfront_loc * mesh.mass
    weights = mesh.mass * np.maximum(moves.x_prev - mesh.x, 0.0)
    if not np.sum(weights) > 0.0:
        # The global melt fraction fell while no node's did; spread the crystals
        # by solid mass instead.
        weights = mesh.mass * (1.0 - mesh.x)
    total_weight = float(np.sum(weights))
    for sp in species:
        c = conc.get(sp, 0.0)
        d = part.get(sp, 0.0)
        melt_share = max(1.0 - d, 0.0) * c
        bury = np.where(moves.burying, melt_share * held_melt, 0.0)
        into_crystal = np.where(moves.bound, melt_share * transition_melt, 0.0)
        if crystallised > 0.0 and total_weight > 0.0:
            into_crystal = into_crystal + d * c * crystallised * weights / total_weight
        release = float(moves.release.get(sp, 0.0))
        wanted = float(np.sum(bury) + np.sum(into_crystal))
        available = max(float(liquid.get(sp, 0.0)), 0.0) + release
        scale = 1.0
        if wanted > available:
            scale = available / wanted
            step.capped.append(sp)
        new.pore[sp] = new.pore.get(sp, np.zeros(mesh.n)) + scale * bury
        new.crystal[sp] = (
            new.crystal.get(sp, np.zeros(mesh.n))
            + scale * into_crystal
            + moves.freeze.get(sp, np.zeros(mesh.n))
        )
        step.buried_kg[sp] = scale * wanted
        step.released_kg[sp] = release
        step.remelt_kg[sp] = float(moves.remelt.get(sp, 0.0))
        step.moved_kg[sp] = scale * wanted - release


def _first_sight(mesh: _Mesh, time_now: float) -> tuple:
    """A new record on the current profiles, and transitions that bury its mush."""
    n = mesh.n
    new = NodeRecord.initial(mesh.phase, mesh.phi, mesh.x, time_now)
    empty = np.array([], dtype=int)
    step = LayerStep(
        exited=empty,
        trapped=np.array([]),
        branch=empty,
        residence=np.array([]),
        node_branch=np.full(n, BRANCH_NONE),
    )
    moves = _Transitions(
        burying=mesh.phase == PHASE_MUSH,
        bound=np.zeros(n, dtype=bool),
        freeze={},
        release={},
        remelt={},
        x_prev=mesh.x,
    )
    return new, step, moves


def advance_record(
    record: NodeRecord | None,
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
    buoyancy: str,
    concentration: dict[str, float] | None = None,
    partition: dict[str, float] | None = None,
    liquid: dict[str, float] | None = None,
    crystallised: float = 0.0,
    substep_fraction: float = SUBSTEP_FRACTION,
    max_substeps: int = MAX_SUBSTEPS,
) -> tuple[NodeRecord, LayerStep]:
    """Advance the per-node record and its volatiles over one coupling step.

    Drains the mush of the previous step, then classifies every node on the
    current profiles and moves each node between the liquid, mush and solid
    phase states, burying, carrying, returning and locking volatiles as the
    module docstring states. Returns a new record, leaving ``record``
    untouched, and what the step did; ``LayerStep.moved_kg`` is the net mass of
    each species the caller moves from the melt into the solid reservoir.

    Parameters
    ----------
    time_prev, time_now : float
        Model time at the previous and the current step [yr].
    melt_fraction : ndarray
        Solver melt mass fraction at every staggered node [1].
    rho_solid, rho_melt : ndarray
        Solid and melt phase-boundary densities at every node [kg m-3].
    gravity : ndarray
        Gravity at every node [m s-2].
    r_basic : ndarray
        Node boundaries [m], one more than the nodes, ascending.
    mass : ndarray
        Node masses [kg].
    rfront_loc, phi_min : float
        Rheological transition and porosity floor [1].
    grain_size : float
        Crystal grain size [m].
    melt_visc, mush_visc : float
        Melt and mush viscosity [Pa s].
    buoyancy : str
        The solver's ``separation_buoyancy``, ``'mixture'`` or ``'solid'``.
    concentration : dict
        Melt concentration of each species the chemistry solver last computed
        [kg kg-1].
    partition : dict
        Crystal/melt partition coefficient of each species [1].
    liquid : dict
        Mass of each species in the melt reservoir [kg], the cap on new burials.
    crystallised : float
        Mantle mass crystallised over the step [kg], ``dM_RM``.

    A ``record`` of ``None``, or one sized for another mesh, starts the record
    on the current profiles without draining: the mush it finds buries its pore
    melt at the current concentrations.
    """
    mesh = _mesh(
        melt_fraction, rho_solid, rho_melt, gravity, r_basic, mass, rfront_loc, phi_min
    )
    conc = {sp: max(float(c), 0.0) for sp, c in (concentration or {}).items()}
    part = {sp: max(float(d), 0.0) for sp, d in (partition or {}).items()}
    params = DrainageParams(
        grain_size=grain_size,
        melt_visc=melt_visc,
        mush_visc=mush_visc,
        buoyancy=buoyancy,
        substep_fraction=substep_fraction,
        max_substeps=max_substeps,
    )

    if record is None or record.size != mesh.n:
        if record is not None:
            log.warning(
                'Trapping: the per-node record covers %d nodes but the mesh has %d; '
                'the layer drainage restarts from the current mush.',
                record.size,
                mesh.n,
            )
        species = list(RECORD_SPECIES)
        new, step, moves = _first_sight(mesh, time_now)
    else:
        species = sorted(set(RECORD_SPECIES) | set(record.pore) | set(record.crystal))
        new, step, moves = _evolve(
            record, mesh, species, float(time_prev), float(time_now), params
        )

    _settle(new, mesh, moves, step, species, conc, part, liquid or {}, crystallised)
    mush = new.phase == PHASE_MUSH
    step.n_dense = int(np.sum(mush & (mesh.rho_s - mesh.rho_l <= 0.0)))
    _describe(step, new, mesh, params)
    return new, step


def _describe(step, record, mesh, params: DrainageParams):
    """Front-like diagnostics of the thickest mush layer, for comparison runs."""
    step.n_mush = int(np.sum(record.phase == PHASE_MUSH))
    layer, extent = thickest_mush_layer(record.phase, mesh.r_basic)
    if layer is None:
        return
    top = int(layer[-1])
    drho = mesh.rho_s - mesh.rho_l
    step.thickness = extent
    step.r_base = float(mesh.r_basic[layer[0]])
    step.rho_solid_base = float(mesh.rho_s[layer[0]])
    step.node_thickness = extent / layer.size
    psi_top = float(record.psi[top]) if np.isfinite(record.psi[top]) else 0.0
    w_top = darcy_velocity(
        psi_top,
        params.grain_size,
        float(drho[top]),
        float(mesh.gravity[top]),
        params.melt_visc,
        buoyancy=params.buoyancy,
    )
    step.tau_d = extent / w_top if w_top > 0.0 else float('inf')
    load = abs(float(np.mean(drho[layer]))) * float(np.mean(mesh.gravity[layer])) * extent
    step.tau_s = params.mush_visc / load if load > 0.0 else float('inf')


def adopt_unrecorded(
    record: NodeRecord, mass: np.ndarray, trapped: dict[str, float], rtol: float = 1.0e-9
) -> dict[str, float]:
    """Assign trapped mass the per-node record does not hold to the nodes that are not liquid.

    ``trapped`` holds the whole-planet trapped mass of each species [kg], the
    ``_kg_trapped`` columns. Where it exceeds what the record holds by more
    than ``rtol`` of itself, the difference is added to the crystal store of
    the mush and solid nodes in proportion to their ``mass`` [kg], so it
    remelts with them. Returns the mass adopted per species [kg]; nothing is
    adopted while every node is liquid.
    """
    m = np.asarray(mass, dtype=float).ravel()
    holders = record.phase != PHASE_LIQUID
    weight = np.where(holders, np.maximum(m, 0.0), 0.0)
    total = float(np.sum(weight))
    adopted = {}
    if not total > 0.0:
        return adopted
    for sp, kg in trapped.items():
        missing = float(kg) - record.trapped(sp)
        if not np.isfinite(missing) or missing <= rtol * max(float(kg), 0.0):
            continue
        store = record.crystal.setdefault(sp, np.zeros(record.size))
        record.crystal[sp] = store + missing * weight / total
        adopted[sp] = missing
    return adopted
