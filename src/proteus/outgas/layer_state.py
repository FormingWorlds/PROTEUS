"""Per-node state of the layer drainage, and the volatiles it locks and buries.

:mod:`proteus.outgas.layer_drainage` drains the pore melt of the mush node by
node. This module carries what that drainage needs from one step to the next,
the retained pore melt of every node and the volatile mass held at it, moves
every node between the liquid, the mush and the solid, and saves the state in
the interior snapshots so a resumed run continues it.

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
proportion to the retained volume. Melt passing from one node into the next
takes its share of the locked mass with it; melt that drains out of the top of
a run rejoins the ocean and its volatiles return to the melt reservoir. What a
node holds when it reaches the porosity floor stays buried. Every step's
crystals take ``D_Z C_Z dM_RM``, spread over the nodes by the mass each
crystallised. Over the life of a node the two add up to
``[(1 - F) D_Z + F] C m``, Sim et al. (2024) Eq. 6 applied to the node's own
mass. The noble gases follow the same rules at ``D_Z = 0``.

The new locks of a step, crystals included, are capped together at what the
melt reservoir holds after this step's returns, so the net mass a step moves
into the solid reservoir can be negative, when returns exceed new locks.

Remelting
---------
Remelting is followed node by node. A node that becomes liquid again, from the
mush or from the solid, returns everything locked at it, pore melt and buried
mass alike, to the melt reservoir: its solid has rejoined the magma ocean. A
solid node that re-enters the mush from below makes its new pore melt from its
own buried material, so it locks nothing new and keeps its buried record until
it becomes liquid. The record is kept per species, so a release returns
exactly the species that went in.

Trapped mass that the record does not hold, as after resuming a run whose
snapshot carries no state, is assigned to the nodes that are not liquid in
proportion to their mass (:func:`adopt_unrecorded`), so it remelts with them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from proteus.outgas.common import element_masses_from_species
from proteus.outgas.compaction import (
    BRANCH_DARCY,
    BRANCH_GUARD,
    BRANCH_MATRIX,
    BRANCH_NONE,
    _contiguous_runs,
    darcy_velocity,
)
from proteus.outgas.layer_drainage import (
    MAX_SUBSTEPS,
    STATUS_LIQUID,
    STATUS_MUSH,
    STATUS_SOLID,
    SUBSTEP_FRACTION,
    classify_nodes,
    compaction_times,
    drain_runs,
    mass_fraction_of_volume,
    melt_volume_fraction,
    node_array,
    shell_geometry,
    thickest_run,
)
from proteus.utils.constants import noble_gases, secs_per_year, vol_list

log = logging.getLogger('fwl.' + __name__)

# Species the per-node record carries: the volatiles and the noble gases.
LAYER_SPECIES = tuple(vol_list) + tuple(noble_gases)

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


def species_elements(species: str) -> dict[str, float]:
    """Mass of each element in one kilogram of ``species`` [kg kg-1].

    A noble gas is its own element.
    """
    if species in vol_list:
        return element_masses_from_species({species: 1.0})
    return {species: 1.0}


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
    buried: dict[str, np.ndarray] = field(default_factory=dict)  # per species [kg]
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
            buried={s: v.copy() for s, v in self.buried.items()},
            retained={s: v.copy() for s, v in self.retained.items()},
        )

    def species_profile(self, species: str) -> np.ndarray:
        """Mass of one species locked at every node, buried or in pore melt [kg]."""
        zero = np.zeros(self.size)
        return np.asarray(self.buried.get(species, zero), dtype=float) + np.asarray(
            self.retained.get(species, zero), dtype=float
        )

    def element_profile(self, element: str) -> np.ndarray:
        """Mass of one element locked at every node, buried or in pore melt [kg]."""
        total = np.zeros(self.size)
        for species in set(self.buried) | set(self.retained):
            share = species_elements(species).get(element, 0.0)
            if share:
                total = total + share * self.species_profile(species)
        return total

    def locked(self, species: str) -> float:
        """Mass of one species locked in the mesh, summed over nodes [kg]."""
        return float(np.sum(self.species_profile(species)))

    @classmethod
    def initial(
        cls,
        status: np.ndarray,
        volume_fraction: np.ndarray,
        melt_fraction: np.ndarray,
        time: float,
    ) -> LayerState:
        """State of a mesh first seen at ``time`` [yr], with nothing locked yet.

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
            buried={s: np.zeros(n) for s in LAYER_SPECIES},
            retained={s: np.zeros(n) for s in LAYER_SPECIES},
        )


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
    n_remelted: int = 0  # nodes that became liquid again this step
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
    remelt_kg: dict[str, float] = field(default_factory=dict)  # of it, by remelting [kg]
    capped: list[str] = field(default_factory=list)  # species limited by the melt


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


def _mesh(melt_fraction, rho_solid, rho_melt, gravity, r_basic, mass, rfront_loc, phi_min):
    """Validated profiles and node classes of one step."""
    if not 0.0 < rfront_loc <= 1.0:
        raise ValueError(f'rfront_loc must lie in (0, 1], got {rfront_loc!r}')
    x = np.clip(np.asarray(melt_fraction, dtype=float).ravel(), 0.0, 1.0)
    n = x.size
    rho_s = node_array(rho_solid, n, 'rho_solid')
    rho_l = node_array(rho_melt, n, 'rho_melt')
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
    """Which nodes a step moves between classes, and the mass they free or lock."""

    locking: np.ndarray  # nodes whose pore melt locks now [bool]
    bound: np.ndarray  # nodes that crossed the whole mush in the step [bool]
    freeze: dict[str, np.ndarray]  # pore-melt mass reaching the floor [kg]
    release: dict[str, float]  # mass returned to the melt reservoir [kg]
    remelt: dict[str, float]  # of it, by nodes becoming liquid [kg]
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
    grain, melt_visc, mush_visc, buoyancy, substep_fraction, max_substeps = params

    def speed_of(p, index):
        return darcy_velocity(
            p, grain, drho[index], mesh.gravity[index], melt_visc, buoyancy=buoyancy
        )

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
        dt_yr * secs_per_year,
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


def _exits(state, new, mesh, active, time_now):
    """Nodes that reached the porosity floor, their trapped fraction and branch."""
    floor = mesh.status == STATUS_SOLID
    # A mush node with no retained melt on record, as a damaged snapshot could
    # leave, has no drainage history either and is bounded like one never seen.
    stray = (state.status == STATUS_MUSH) & ~np.isfinite(state.psi)
    exit_drained = active & floor
    exit_bound = ((state.status == STATUS_LIQUID) | stray) & floor
    exited = np.flatnonzero(exit_drained | exit_bound)
    drained_exit = exit_drained[exited]
    # Drainage removes melt in proportion to the retained volume, so the retained
    # melt mass is the entry melt mass scaled by psi / psi_entry.
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


def _evolve(state: LayerState, mesh: _Mesh, species, time_prev, time_now, params):
    """Drain the previous mush, then move every node to its class of this step.

    Returns the new state, whose carried, frozen and remelted mass is settled
    but whose new locks are not yet applied, the step record, and the transitions.
    """
    n = mesh.n
    new = state.copy()
    for sp in species:
        new.retained.setdefault(sp, np.zeros(n))
        new.buried.setdefault(sp, np.zeros(n))
    dt_yr = max(time_now - time_prev, 0.0)
    active = (state.status == STATUS_MUSH) & np.isfinite(state.psi)
    drained, open_runs, sealed, release = _drain(new, mesh, species, dt_yr, params)
    new.psi = drained.psi
    new.t_darcy = new.t_darcy + drained.t_darcy / secs_per_year
    new.t_matrix = new.t_matrix + drained.t_matrix / secs_per_year
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

    exited, trapped, branch, residence, exit_drained, exit_bound, stray = _exits(
        state, new, mesh, active, time_now
    )
    if np.any(exit_bound):
        reasons.append(f'{int(exit_bound.sum())} crossed the whole mush within one step')

    # Pore melt reaching the floor is buried. A node that becomes liquid again
    # returns everything locked at it, its pore melt and its buried mass alike.
    remelt = (state.status != STATUS_LIQUID) & (mesh.status == STATUS_LIQUID)
    freeze, remelted = {}, {}
    for sp in species:
        freeze[sp] = np.where(exit_drained, new.retained[sp], 0.0)
        remelted[sp] = float(np.sum(new.retained[sp][remelt] + new.buried[sp][remelt]))
        release[sp] = release.get(sp, 0.0) + remelted[sp]
        new.retained[sp] = np.where(exit_drained | remelt, 0.0, new.retained[sp])
        new.buried[sp] = np.where(remelt, 0.0, new.buried[sp])

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
    # the melt fraction, so a node's residence is resolved finer than one step.
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
        truncated=drained.elapsed < dt_yr * secs_per_year * (1.0 - 1.0e-12) and bool(open_runs),
        held=reasons,
        matrix_speed=drained.matrix_speed if open_runs else float('nan'),
        returned=drained.returned,
    )
    moves = _Transitions(
        locking=from_above,
        bound=exit_bound,
        freeze=freeze,
        release=release,
        remelt=remelted,
        x_prev=x_prev,
    )
    return new, step, moves


def _settle(new, mesh, moves, step, species, conc, part, liquid, crystallised) -> None:
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
        # The global melt fraction fell while no node's did; spread the crystals
        # by solid mass instead.
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
        new.buried[sp] = (
            new.buried.get(sp, np.zeros(mesh.n))
            + scale * bury
            + moves.freeze.get(sp, np.zeros(mesh.n))
        )
        step.locked_kg[sp] = scale * wanted
        step.released_kg[sp] = release
        step.remelt_kg[sp] = float(moves.remelt.get(sp, 0.0))
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
    buoyancy: str,
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
    time_prev, time_now : float
        Model time at the previous and the current step [yr].
    melt_fraction : ndarray
        Solver melt mass fraction at every staggered node [1].
    rho_solid, rho_melt : ndarray
        Solid and melt phase-boundary densities at every node [kg m-3].
    gravity : ndarray
        Gravity at every node [m s-2].
    r_basic : ndarray
        Shell boundaries [m], one more than the nodes, ascending.
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
    params = (grain_size, melt_visc, mush_visc, buoyancy, substep_fraction, max_substeps)

    if state is None or state.size != n:
        if state is not None:
            log.warning(
                'Trapping: the per-node state covers %d nodes but the mesh has %d; '
                'the layer drainage restarts from the current mush.',
                state.size,
                n,
            )
        species = list(LAYER_SPECIES)
        new = LayerState.initial(mesh.status, mesh.phi, mesh.x, time_now)
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
            remelt={},
            x_prev=mesh.x,
        )
    else:
        species = sorted(set(LAYER_SPECIES) | set(state.retained) | set(state.buried))
        new, step, moves = _evolve(
            state, mesh, species, float(time_prev), float(time_now), params
        )

    _settle(new, mesh, moves, step, species, conc, part, liquid or {}, crystallised)
    mush = new.status == STATUS_MUSH
    step.n_dense = int(np.sum(mush & (mesh.rho_s - mesh.rho_l <= 0.0)))
    _describe(step, new, mesh, grain_size, melt_visc, mush_visc, buoyancy)
    return new, step


def _describe(step, state, mesh, grain, melt_visc, mush_visc, buoyancy):
    """Front-like diagnostics of the thickest mush run, for comparison runs."""
    step.n_mush = int(np.sum(state.status == STATUS_MUSH))
    run, extent = thickest_run(state.status, mesh.r_basic)
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
        psi_top, grain, float(drho[top]), float(mesh.gravity[top]), melt_visc, buoyancy=buoyancy
    )
    step.tau_d = extent / w_top if w_top > 0.0 else float('inf')
    load = abs(float(np.mean(drho[run]))) * float(np.mean(mesh.gravity[run])) * extent
    step.tau_s = mush_visc / load if load > 0.0 else float('inf')


def adopt_unrecorded(
    state: LayerState, mass: np.ndarray, trapped: dict[str, float], rtol: float = 1.0e-9
) -> dict[str, float]:
    """Assign trapped mass the per-node record does not hold to the nodes that are not liquid.

    ``trapped`` holds the whole-planet trapped mass of each species [kg], the
    ``_kg_trapped`` columns. Where it exceeds what the record holds by more
    than ``rtol`` of itself, the difference is added to the buried record of
    the mush and solid nodes in proportion to their ``mass`` [kg], so it
    remelts with them. Returns the mass adopted per species [kg]; nothing is
    adopted while every node is liquid.
    """
    m = np.asarray(mass, dtype=float).ravel()
    holders = state.status != STATUS_LIQUID
    weight = np.where(holders, np.maximum(m, 0.0), 0.0)
    total = float(np.sum(weight))
    adopted = {}
    if not total > 0.0:
        return adopted
    for sp, kg in trapped.items():
        missing = float(kg) - state.locked(sp)
        if not np.isfinite(missing) or missing <= rtol * max(float(kg), 0.0):
            continue
        record = state.buried.setdefault(sp, np.zeros(state.size))
        state.buried[sp] = record + missing * weight / total
        adopted[sp] = missing
    return adopted


def write_state(path: str, state: LayerState) -> None:
    """Add the per-node state to an interior snapshot, beside the solver fields.

    The snapshot must already hold the ``staggered`` dimension the solver
    writes, sized for this state. Variables present from an earlier write of
    the same file are overwritten.

    Raises
    ------
    ValueError
        If the snapshot has no ``staggered`` dimension of the state's size.
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
        for species, kg in state.buried.items():
            columns[f'buried_{species}'] = kg
            units[f'buried_{species}'] = 'kg'
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
        buried, retained = {}, {}
        for species in LAYER_SPECIES:
            for record, prefix in ((buried, 'buried'), (retained, 'retained')):
                key = _snapshot_name(f'{prefix}_{species}')
                if key in ds.variables:
                    record[species] = column(key, 0.0)
    stored = list(arrays.values()) + list(buried.values()) + list(retained.values())
    if any(np.size(a) != n for a in stored):
        return None
    for species in LAYER_SPECIES:
        buried.setdefault(species, np.zeros(n))
        retained.setdefault(species, np.zeros(n))
    return LayerState(
        status=np.rint(arrays['status']).astype(int),
        psi=arrays['psi'],
        psi_entry=arrays['psi_entry'],
        t_entry=arrays['t_entry'],
        melt_fraction=arrays['melt_fraction'],
        t_darcy=arrays['t_darcy'],
        t_matrix=arrays['t_matrix'],
        t_held=arrays['t_held'],
        buried=buried,
        retained=retained,
    )
