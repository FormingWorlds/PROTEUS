"""The trapping step of ``outgas.trap_mode = 'layers'``.

Connects the per-node drainage of :mod:`proteus.outgas.layer_drainage` and the
record of :mod:`proteus.outgas.node_record` to the helpfile row and the
interior. The scheme runs on the staggered nodes of the Aragog mesh, the cells
between its basic nodes: the melt fraction and the node masses are the
solver's own staggered fields, and the phase-boundary densities and the
gravity, which the solver writes at the basic nodes, are taken as the mean of
each node's two boundaries.

Each step moves the net mass of every species the record trapped minus what it
returned from the melt into the solid reservoir. The move goes through the
``_kg_trapped`` bookkeeping of :mod:`proteus.outgas.trap_bookkeeping`, shared
with the front scheme, so the chemistry, escape, desiccation and the closure
check treat the two schemes alike. A return is capped at what the trapped
reservoir holds. The per-node record lives on the interior object between
steps and in the interior snapshots on disk, from which a resumed run reads it
back.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import numpy as np

from proteus.outgas.compaction import (
    BRANCH_FALLBACK,
    BRANCH_GUARD,
    BRANCH_NONE,
    BRANCH_REMELT,
    DEFAULT_MUSH_LOG10VISC,
)
from proteus.outgas.layer_drainage import node_geometry
from proteus.outgas.node_record import RECORD_SPECIES, adopt_unrecorded, advance_record
from proteus.outgas.node_record_io import read_record, write_record
from proteus.outgas.trap_bookkeeping import (
    TrappingStep,
    apply_to_reservoirs,
    critical_melt_fraction,
    front_speed,
    melt_concentration,
    partition_coefficients,
    record_step,
    species_carried,
    trapped_kg,
)
from proteus.utils.constants import secs_per_year

log = logging.getLogger('fwl.' + __name__)


def _node_mean(values, n: int) -> np.ndarray | None:
    """Mean of each node's two boundary values, from ``n + 1`` basic-node values."""
    if values is None:
        return None
    a = np.asarray(values, dtype=float).ravel()
    if a.size != n + 1 or not np.all(np.isfinite(a)):
        return None
    return 0.5 * (a[:-1] + a[1:])


def layer_profiles(hf_row: dict, interior_o) -> dict | None:
    """Interior profiles the layer drainage reads on the staggered nodes, or ``None``.

    Returns the melt fraction [1], the solid and melt phase-boundary densities
    [kg m-3], the gravity [m s-2], the node boundaries [m] and the node masses
    [kg]. ``None`` when a profile is absent, mis-sized or not finite, which no
    interior but Aragog avoids; the gravity falls back to the surface value.
    """
    x = np.asarray(getattr(interior_o, 'phi', None), dtype=float).ravel()
    r_basic = np.asarray(getattr(interior_o, 'radius', None), dtype=float).ravel()
    n = x.size
    usable = n >= 2 and r_basic.size == n + 1 and np.all(np.diff(r_basic) > 0.0)
    rho_s = _node_mean(getattr(interior_o, 'rho_solid_b', None), n) if usable else None
    rho_l = _node_mean(getattr(interior_o, 'rho_melt_b', None), n) if usable else None
    if not usable or rho_s is None or rho_l is None or not np.all(np.isfinite(x)):
        log.warning(
            'Trapping: the interior profiles the layer drainage needs are unavailable '
            '(melt fraction %d nodes, node boundaries %d, phase-boundary densities %s); '
            'the step traps at the crystal partition coefficients alone. The layer '
            'drainage needs an aragog interior.',
            n,
            r_basic.size,
            'present' if rho_s is not None and rho_l is not None else 'absent',
        )
        return None
    gravity = _node_mean(getattr(interior_o, 'g_b', None), n)
    if gravity is None or np.any(gravity <= 0.0):
        g_surf = float(hf_row.get('gravity', 0.0))
        log.warning(
            'Trapping: the interior solver carries no usable basic-node gravity, so the '
            'mush drains under the surface gravity %.3f m/s2.',
            g_surf,
        )
        gravity = np.full(n, g_surf)
    mass = _node_masses(hf_row, interior_o, r_basic, n)
    if mass is None:
        return None
    return {
        'n': n,
        'melt_fraction': x,
        'rho_solid': rho_s,
        'rho_melt': rho_l,
        'gravity': gravity,
        'r_basic': r_basic,
        'mass': mass,
    }


def _node_masses(hf_row: dict, interior_o, r_basic, n: int) -> np.ndarray | None:
    """Node masses [kg] that sum to the mantle mass, or ``None``.

    The solver's staggered masses are its density times a fixed node volume,
    which grows as the mantle cools and no longer sums to ``M_mantle``. Their
    shares are kept and rescaled to ``M_mantle``, the mass the melt
    concentrations of the chemistry refer to.
    """
    mass = np.asarray(getattr(interior_o, 'mass', None), dtype=float).ravel()
    if mass.size != n or not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
        # Node volume times density is the mass the solver integrates.
        volume, _, _ = node_geometry(r_basic)
        density = np.asarray(getattr(interior_o, 'density', None), dtype=float).ravel()
        mass = volume * density if density.size == n else np.full(n, np.nan)
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            log.warning(
                'Trapping: the interior carries no usable node masses, so the step '
                'traps at the crystal partition coefficients alone.'
            )
            return None
    m_mantle = float(hf_row.get('M_mantle', 0.0))
    if np.isfinite(m_mantle) and m_mantle > 0.0:
        mass = mass * (m_mantle / float(np.sum(mass)))
    return mass


def _snapshot_for(dirs: dict | None, time: float) -> str | None:
    """The interior snapshot written for ``time`` [yr], or ``None`` if there is none."""
    output = (dirs or {}).get('output')
    if not output or not time > 0.0:
        return None
    from proteus.utils.coupler import _snapshot_belongs_to
    from proteus.utils.helper import snapshot_path_for_time

    path = snapshot_path_for_time(os.path.join(output, 'data'), time, '_int.nc')
    if not os.path.isfile(path) or not _snapshot_belongs_to(path, time):
        return None
    return path


def _restore_record(dirs: dict | None, row: dict, n: int | None = None):
    """The per-node record a resumed run left in the snapshot of ``row``, or ``None``."""
    time = float(row.get('Time', 0.0))
    path = _snapshot_for(dirs, time)
    record = None
    if path is not None:
        try:
            record = read_record(path, n)
        except (OSError, RuntimeError, ValueError) as exc:
            log.warning('Trapping: could not read the per-node record from %s (%s).', path, exc)
    if record is None:
        log.warning(
            'Trapping: no per-node record for t = %.6g yr to resume from; the layer '
            'drainage restarts from the current mush, and the trapped mass on the '
            'helpfile is assigned to the nodes that are not liquid.',
            time,
        )
        return None
    log.info('Trapping: resumed the per-node record of %d nodes from %s', record.size, path)
    return record


def _persist_record(dirs: dict | None, hf_row: dict, record) -> None:
    """Write the per-node record into this step's interior snapshot, if there is one.

    Steps without a snapshot are skipped: a resume only lands on a step that
    wrote one, and that snapshot carries the record of its own step.
    """
    path = _snapshot_for(dirs, float(hf_row.get('Time', 0.0)))
    if path is None:
        return
    try:
        write_record(path, record)
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning(
            'Trapping: could not write the per-node record to %s (%s); a resume from '
            'this step restarts the layer drainage.',
            path,
            exc,
        )


def restore_node_record(config, dirs: dict | None, hf_row: dict, interior_o) -> None:
    """On a resume with ``trap_mode = 'layers'``, read the record back onto the interior.

    Called once by the resume path for the row the run resumes from. Without
    a record in that snapshot, the interior keeps none, and the first layer
    step assigns the trapped mass on the helpfile to its nodes.
    """
    if getattr(config.outgas, 'trap_mode', 'none') != 'layers':
        return
    interior_o.trap_record = _restore_record(dirs, hf_row)


def persist_node_record(dirs: dict | None, hf_row: dict, interior_o) -> None:
    """Write the per-node record into the snapshot for ``hf_row['Time']``.

    For callers outside the trapping step that rewrite that snapshot from the
    interior solver alone, as the end-of-run write does, which would otherwise
    drop the record a resume from the last step needs. No-op without a record.
    """
    record = getattr(interior_o, 'trap_record', None)
    if record is not None:
        _persist_record(dirs, hf_row, record)


def _majority(branches: np.ndarray, weights: np.ndarray) -> int:
    """The branch code carrying the largest total weight."""
    codes = np.unique(branches)
    totals = [float(np.sum(weights[branches == c])) for c in codes]
    return int(codes[int(np.argmax(totals))])


def _describe(step: TrappingStep, layers, hf_row: dict, prev: dict, mass) -> None:
    """Fill the helpfile diagnostics of one layer step.

    ``F_tl``, the residence time and the branch describe the nodes that reached
    the floor, weighted by their mass; with none, the branch is that of the
    nodes that drained. The node count, thickness, timescales and speed
    describe the thickest mush layer, which stands for the front of the front
    scheme. The matrix speed over the front speed has no single front top here
    and stays NaN.
    """
    exited = layers.exited
    m_exit = mass[exited]
    step.n_exited = int(exited.size)
    step.n_mush = float(layers.n_mush)
    step.n_front = layers.layer_nodes
    step.n_substeps = layers.n_substeps
    step.l_front = layers.thickness
    step.tau_d = layers.tau_d / secs_per_year
    step.tau_s = layers.tau_s / secs_per_year
    if exited.size:
        total = float(np.sum(m_exit))
        step.f_tl = float(np.sum(layers.trapped * m_exit) / total)
        step.frac_bound = float(np.sum(m_exit[layers.branch == BRANCH_GUARD]) / total)
        resolved = np.isfinite(layers.residence)
        if np.any(resolved):
            step.t_res = float(
                np.sum(layers.residence[resolved] * m_exit[resolved]) / np.sum(m_exit[resolved])
            )
        step.branch = _majority(layers.branch, m_exit)
    else:
        drained = layers.node_branch != BRANCH_NONE
        if np.any(drained):
            step.branch = _majority(layers.node_branch[drained], mass[drained])

    dt_s = (float(hf_row['Time']) - float(prev['Time'])) * secs_per_year
    if not (np.isfinite(layers.r_base) and np.isfinite(layers.rho_solid_base)):
        return
    step.v_front = front_speed(step.dm_rm, layers.r_base, layers.rho_solid_base, dt_s)
    if layers.thickness > 0.0 and dt_s > 0.0:
        step.front_courant = step.v_front * dt_s / layers.thickness


def _log(step: TrappingStep, layers) -> None:
    """Report one layer step, and every node it could not drain."""
    log.info(
        '    layers     = %d mush nodes, %d entered, %d at the floor, %d remelted, '
        '%d sub-steps; buried %.3e kg, returned %.3e kg, net %.3e kg',
        layers.n_mush,
        layers.n_entered,
        step.n_exited,
        layers.n_remelted,
        step.n_substeps,
        sum(layers.buried_kg.values()),
        sum(layers.released_kg.values()),
        step.total_trapped,
    )
    if layers.n_dense:
        log.info(
            '    layers     = %d mush nodes have tabulated melt denser than their '
            'solid; they and the mush nodes below them hold their pore melt',
            layers.n_dense,
        )
    if layers.held and np.isfinite(step.frac_bound) and step.frac_bound > 0.0:
        # Held nodes keep the pore melt they had when drainage stopped, which can
        # trap far more than draining would, so each such step reports its causes.
        log.warning(
            'Trapping: mush nodes could not drain (%s); %.0f%% of the mass that '
            'reached the porosity floor this step did so on the no-drainage bound, '
            'keeping the pore melt it held when drainage stopped, which for a node '
            'that drained before it was sealed is less than its entry pore melt.',
            '; '.join(layers.held),
            100.0 * step.frac_bound,
        )
    elif layers.held:
        log.info('    held nodes = %s', '; '.join(layers.held))


def _capped_moves(hf_row: dict, moved: dict[str, float]) -> dict[str, float]:
    """Net moves per species, a return limited to what the trapped reservoir holds."""
    out = {}
    for species, mass in moved.items():
        if mass < 0.0:
            mass = -min(-mass, trapped_kg(hf_row, species))
        if mass != 0.0:
            out[species] = mass
    return out


def ocean_melt_fraction(phi_global: float, melt_fraction, mass, rfront_loc: float) -> float:
    """Melt of the liquid nodes as a fraction of the mantle mass [1].

    ``Phi_global`` times the liquid nodes' share of the melt, ``sum_liquid x m
    / sum x m`` over nodes of melt fraction ``x`` and mass ``m`` [kg], a node
    being liquid at ``x >= rfront_loc``. The share keeps the result at or below
    ``Phi_global`` whatever the mass scale. Zero when no node holds melt.
    """
    x = np.asarray(melt_fraction, dtype=float).ravel()
    m = np.asarray(mass, dtype=float).ravel()
    melt = float(np.sum(x * m))
    if not melt > 0.0:
        return 0.0
    ocean = float(np.sum((x * m)[x >= rfront_loc]))
    return float(phi_global) * ocean / melt


@contextmanager
def ocean_melt_for_chemistry(config, hf_row: dict):
    """Hand the chemistry only the ocean melt for one solve, in ``layers``.

    The pore melt of the mush and solid nodes is cut off from the ocean, so in
    ``layers`` the solve dissolves volatiles into the liquid nodes' melt alone:
    ``Phi_global`` reads ``trap_Phi_ocean`` while the context is open and is
    restored after it. Outside ``layers``, or before a layer step has written
    the column, it changes nothing.
    """
    ocean = float(hf_row.get('trap_Phi_ocean', float('nan')))
    if getattr(config.outgas, 'trap_mode', 'none') != 'layers' or not np.isfinite(ocean):
        yield
        return
    saved = hf_row['Phi_global']
    hf_row['Phi_global'] = min(max(ocean, 0.0), float(saved))
    try:
        yield
    finally:
        hf_row['Phi_global'] = saved


def _fallback(config, hf_row: dict, prev: dict, interior_o, dm_rm: float, melt_mass: float):
    """A layer step without interior profiles, which leaves the burial ledger alone.

    Freezing traps the crystals' share ``D C dM_RM`` alone; the record does not
    take it, and adopts it on the next step with profiles. Remelting returns
    the remelted share ``(Phi - Phi_prev) / (1 - Phi_prev)`` of every trapped
    species and removes the same share from the record, so no mass is returned
    twice.
    """
    step = TrappingStep(mode='layers', dm_rm=dm_rm, f_tl=0.0, melt_mass=melt_mass)
    step.branch = BRANCH_FALLBACK
    if dm_rm > 0.0 and melt_mass > 0.0:
        part = partition_coefficients(config)
        for sp in RECORD_SPECIES:
            kg = float(hf_row.get(f'{sp}_kg_liquid', 0.0))
            if kg <= 0.0 or not species_carried(hf_row, sp):
                continue
            wanted = part.get(sp, 0.0) * melt_concentration(kg, melt_mass) * dm_rm
            if wanted > kg:
                step.supply_capped.append(sp)
            if min(wanted, kg) > 0.0:
                step.trapped_kg[sp] = min(wanted, kg)
    elif dm_rm < 0.0:
        phi_prev = float(prev.get('Phi_global', 0.0))
        solid_prev = 1.0 - phi_prev
        rise = float(hf_row.get('Phi_global', phi_prev)) - phi_prev
        share = float(np.clip(rise / solid_prev, 0.0, 1.0)) if solid_prev > 0.0 else 1.0
        for sp in RECORD_SPECIES:
            back = share * trapped_kg(hf_row, sp)
            if back > 0.0:
                step.trapped_kg[sp] = -back
        record = getattr(interior_o, 'trap_record', None)
        if record is not None:
            for store in (record.crystal, record.pore):
                for sp in store:
                    store[sp] = store[sp] * (1.0 - share)
        step.remelted = True
        step.branch = BRANCH_REMELT
    apply_to_reservoirs(hf_row, step.trapped_kg)
    record_step(hf_row, step)
    return step


def run_layer_trapping(
    config,
    hf_row: dict,
    prev: dict,
    interior_o,
    dirs: dict | None,
    dm_rm: float,
    melt_mass: float,
) -> TrappingStep:
    """One step of the layer scheme.

    Without usable interior profiles the step takes :func:`_fallback`.

    Parameters
    ----------
    config : Config
        Model configuration.
    hf_row : dict
        Current helpfile row, modified in place.
    prev : dict
        Previous helpfile row.
    interior_o : Interior_t
        Interior state; carries the per-node record as ``trap_record``.
    dirs : dict or None
        Run directories; the record is written into the interior snapshot of each
        step that has one, and a resumed run reads it back from there.
    dm_rm : float
        Mantle mass crystallised over the step [kg], negative when it remelted.
    melt_mass : float
        Melt mass the dissolved volatiles were last partitioned into [kg], when
        the previous step wrote no ocean melt fraction.
    """
    profiles = layer_profiles(hf_row, interior_o)
    if profiles is None:
        return _fallback(config, hf_row, prev, interior_o, dm_rm, melt_mass)
    tr = config.outgas
    rfront_loc = critical_melt_fraction(config)
    # The previous solve dissolved into the previous step's ocean melt alone. The
    # helpfile stores an unset value as 0, which then means no layer step wrote it.
    ocean_prev = float(prev.get('trap_Phi_ocean', float('nan')))
    if np.isfinite(ocean_prev) and ocean_prev > 0.0:
        melt_mass = float(prev.get('M_mantle', hf_row.get('M_mantle', 0.0))) * ocean_prev
    record = _usable_record(interior_o, profiles['n'])
    trapped = {sp: trapped_kg(hf_row, sp) for sp in RECORD_SPECIES}
    if record is not None:
        adopt_unrecorded(record, profiles['mass'], trapped)
    # The melt concentrations of the previous chemistry solve, on the melt it
    # dissolved into; a species whose elements the run does not carry is not buried.
    liquid = {sp: float(hf_row.get(f'{sp}_kg_liquid', 0.0)) for sp in RECORD_SPECIES}
    concentration = {
        sp: melt_concentration(kg, melt_mass)
        for sp, kg in liquid.items()
        if kg > 0.0 and species_carried(hf_row, sp)
    }
    mush_log10 = float(tr.trap_mush_log10visc)
    new_record, layers = advance_record(
        record,
        time_prev=float(prev['Time']),
        time_now=float(hf_row['Time']),
        melt_fraction=profiles['melt_fraction'],
        rho_solid=profiles['rho_solid'],
        rho_melt=profiles['rho_melt'],
        gravity=profiles['gravity'],
        r_basic=profiles['r_basic'],
        mass=profiles['mass'],
        rfront_loc=rfront_loc,
        phi_min=float(tr.trap_phi_min),
        grain_size=float(config.interior_energetics.grain_size),
        melt_visc=10.0 ** float(config.interior_energetics.melt_log10visc),
        mush_visc=10.0 ** (mush_log10 if mush_log10 > 0.0 else DEFAULT_MUSH_LOG10VISC),
        buoyancy=config.interior_energetics.aragog.separation_buoyancy,
        concentration=concentration,
        partition=partition_coefficients(config),
        liquid=liquid,
        # Mush found by a new record on a run that has trapped mass holds that mass.
        bury_found_mush=not any(kg > 0.0 for kg in trapped.values()),
    )
    step = TrappingStep(mode='layers', dm_rm=dm_rm, f_tl=0.0, melt_mass=melt_mass)
    _describe(step, layers, hf_row, prev, profiles['mass'])
    step.phi_ocean = ocean_melt_fraction(
        float(hf_row.get('Phi_global', 0.0)),
        profiles['melt_fraction'],
        profiles['mass'],
        rfront_loc,
    )
    step.trapped_kg = _capped_moves(hf_row, layers.moved_kg)
    step.supply_capped = list(layers.capped)
    if sum(layers.remelt_kg.values()) > 0.0 and step.total_trapped < 0.0:
        # Diagnostic only: escape and desiccation read the returned mass itself.
        step.remelted = True
        step.branch = BRANCH_REMELT
    apply_to_reservoirs(hf_row, step.trapped_kg)
    if record is None:
        # A record started this step holds only this step's burials; trapped mass
        # from before it is assigned now, so it remelts with its nodes.
        adopt_unrecorded(
            new_record, profiles['mass'], {sp: trapped_kg(hf_row, sp) for sp in RECORD_SPECIES}
        )
    interior_o.trap_record = new_record
    record_step(hf_row, step)
    _log(step, layers)
    _persist_record(dirs, hf_row, new_record)
    return step


def _usable_record(interior_o, n: int):
    """The record to continue, or ``None`` to start one.

    A record sized for another mesh is dropped, and its trapped mass is then
    adopted by the new record like any unrecorded mass.
    """
    record = getattr(interior_o, 'trap_record', None)
    if record is not None and record.size != n:
        log.warning(
            'Trapping: the per-node record covers %d nodes but the mesh has %d; '
            'the record restarts and adopts the trapped mass.',
            record.size,
            n,
        )
        return None
    return record
