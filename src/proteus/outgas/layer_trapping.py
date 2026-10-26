"""The trapping step of ``outgas.trap_mode = 'layers'``.

Connects the per-node layer drainage of :mod:`proteus.outgas.layer_drainage`
and :mod:`proteus.outgas.layer_state` to the helpfile row and the interior.
The scheme runs on the staggered nodes of the Aragog mesh, the shells between
its basic nodes: the melt fraction and the node masses are the solver's own
staggered fields, and the phase-boundary densities and the gravity, which the
solver writes at the basic nodes, are taken as the mean of each shell's two
boundaries.

Each step moves the net mass of every species the layer state locked minus what
it returned from the melt into the solid reservoir, through the same
``_kg_trapped`` bookkeeping as the front scheme, so the chemistry, escape,
desiccation and the closure check treat the two schemes alike. A return is
capped at what the trapped reservoir holds. The per-node state lives on the
interior object between steps and in the interior snapshots on disk, from
which a resumed run reads it back.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from proteus.outgas.compaction import (
    BRANCH_GUARD,
    BRANCH_NONE,
    BRANCH_REMELT,
    DEFAULT_MUSH_LOG10VISC,
)
from proteus.outgas.layer_drainage import shell_geometry
from proteus.outgas.layer_state import (
    LAYER_SPECIES,
    adopt_unrecorded,
    advance_layers,
    read_state,
    write_state,
)
from proteus.outgas.trapping import (
    TrappingStep,
    _apply_to_reservoirs,
    _carried,
    _record,
    _trapped,
    critical_melt_fraction,
    melt_concentration,
    partition_coefficients,
)
from proteus.utils.constants import secs_per_year

log = logging.getLogger('fwl.' + __name__)


def _shell_mean(values, n: int) -> np.ndarray | None:
    """Mean of each shell's two boundary values, from ``n + 1`` basic-node values."""
    if values is None:
        return None
    a = np.asarray(values, dtype=float).ravel()
    if a.size != n + 1 or not np.all(np.isfinite(a)):
        return None
    return 0.5 * (a[:-1] + a[1:])


def layer_profiles(hf_row: dict, interior_o) -> dict | None:
    """Interior profiles the layer drainage reads on the staggered nodes, or ``None``.

    Returns the melt fraction [1], the solid and melt phase-boundary densities
    [kg m-3], the gravity [m s-2], the shell boundaries [m] and the node masses
    [kg]. ``None`` when a profile is absent, mis-sized or not finite, which no
    interior but Aragog avoids; the gravity falls back to the surface value.
    """
    x = np.asarray(getattr(interior_o, 'phi', None), dtype=float).ravel()
    r_basic = np.asarray(getattr(interior_o, 'radius', None), dtype=float).ravel()
    n = x.size
    usable = n >= 2 and r_basic.size == n + 1 and np.all(np.diff(r_basic) > 0.0)
    rho_s = _shell_mean(getattr(interior_o, 'rho_solid_b', None), n) if usable else None
    rho_l = _shell_mean(getattr(interior_o, 'rho_melt_b', None), n) if usable else None
    if not usable or rho_s is None or rho_l is None or not np.all(np.isfinite(x)):
        log.warning(
            'Trapping: the interior profiles the layer drainage needs are unavailable '
            '(melt fraction %d nodes, shell boundaries %d, phase-boundary densities %s); '
            'the step buries at the crystal partition coefficients alone. The layer '
            'drainage needs an aragog interior.',
            n,
            r_basic.size,
            'present' if rho_s is not None and rho_l is not None else 'absent',
        )
        return None
    gravity = _shell_mean(getattr(interior_o, 'g_b', None), n)
    if gravity is None or np.any(gravity <= 0.0):
        g_surf = float(hf_row.get('gravity', 0.0))
        log.warning(
            'Trapping: the interior solver carries no usable basic-node gravity, so the '
            'mush drains under the surface gravity %.3f m/s2.',
            g_surf,
        )
        gravity = np.full(n, g_surf)
    mass = np.asarray(getattr(interior_o, 'mass', None), dtype=float).ravel()
    if mass.size != n or not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
        # Shell volume times density is the mass the solver integrates.
        volume, _, _ = shell_geometry(r_basic)
        density = np.asarray(getattr(interior_o, 'density', None), dtype=float).ravel()
        mass = volume * density if density.size == n else np.full(n, np.nan)
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            log.warning(
                'Trapping: the interior carries no usable node masses, so the step '
                'buries at the crystal partition coefficients alone.'
            )
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


def _restore_state(dirs: dict | None, prev: dict, n: int):
    """The per-node state a resumed run left in its last snapshot, or ``None``."""
    time = float(prev.get('Time', 0.0))
    path = _snapshot_for(dirs, time)
    state = None
    if path is not None:
        try:
            state = read_state(path, n)
        except (OSError, RuntimeError, ValueError) as exc:
            log.warning('Trapping: could not read the per-node state from %s (%s).', path, exc)
    if state is None:
        log.warning(
            'Trapping: no per-node state for t = %.6g yr to resume from; the layer '
            'drainage restarts from the current mush, and the trapped mass on the '
            'helpfile is assigned to the nodes that are not liquid.',
            time,
        )
        return None
    log.info('Trapping: resumed the per-node state of %d nodes from %s', state.size, path)
    return state


def _persist_state(dirs: dict | None, hf_row: dict, state) -> None:
    """Write the per-node state into this step's interior snapshot, if there is one.

    Steps without a snapshot are skipped: a resume only lands on a step that
    wrote one, and that snapshot carries the state of its own step.
    """
    path = _snapshot_for(dirs, float(hf_row.get('Time', 0.0)))
    if path is None:
        return
    try:
        write_state(path, state)
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning(
            'Trapping: could not write the per-node state to %s (%s); a resume from '
            'this step restarts the layer drainage.',
            path,
            exc,
        )


def persist_layer_state(dirs: dict | None, hf_row: dict, interior_o) -> None:
    """Write the layer drainage state into the snapshot for ``hf_row['Time']``.

    For callers outside the trapping step that rewrite that snapshot from the
    interior solver alone, as the end-of-run write does, which would otherwise
    drop the state a resume from the last step needs. No-op without a state.
    """
    state = getattr(interior_o, 'trap_state', None)
    if state is not None:
        _persist_state(dirs, hf_row, state)


def _majority(branches: np.ndarray, weights: np.ndarray) -> int:
    """The branch code carrying the largest total weight."""
    codes = np.unique(branches)
    totals = [float(np.sum(weights[branches == c])) for c in codes]
    return int(codes[int(np.argmax(totals))])


def _describe(step: TrappingStep, layers, hf_row: dict, prev: dict, mass) -> None:
    """Fill the helpfile diagnostics of one layer step.

    ``F_tl``, the residence time and the branch describe the nodes that reached
    the floor, weighted by their mass; with none, the branch is that of the
    nodes that drained. The timescales, thickness and speed describe the thickest mush run, for
    comparison with the front scheme.
    """
    exited = layers.exited
    m_exit = mass[exited]
    step.n_exited = int(exited.size)
    step.n_front = layers.n_mush
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
    area = 4.0 * np.pi * layers.r_base**2 * layers.rho_solid_base
    if not (dt_s > 0.0 and np.isfinite(area) and area > 0.0):
        return
    step.v_front = max(float(step.dm_rm), 0.0) / (area * dt_s)
    if layers.node_thickness > 0.0:
        step.front_courant = step.v_front * dt_s / layers.node_thickness
    if step.v_front > 0.0 and np.isfinite(layers.matrix_speed):
        step.w_matrix_over_vf = layers.matrix_speed / step.v_front


def _log(step: TrappingStep, layers) -> None:
    """Report one layer step, and every node it could not drain."""
    log.info(
        '    layers     = %d mush nodes, %d entered, %d at the floor, %d remelted, '
        '%d sub-steps; locked %.3e kg, returned %.3e kg, net %.3e kg',
        layers.n_mush,
        layers.n_entered,
        step.n_exited,
        layers.n_remelted,
        step.n_substeps,
        sum(layers.locked_kg.values()),
        sum(layers.released_kg.values()),
        step.total_trapped,
    )
    if layers.n_dense:
        log.info(
            '    layers     = %d mush nodes have tabulated melt denser than their '
            'solid; they drain on the magnitude of the contrast',
            layers.n_dense,
        )
    if layers.held and np.isfinite(step.frac_bound) and step.frac_bound > 0.0:
        # Nodes on the bound keep their entry pore melt, which can lock far more
        # than draining would, so each such step is reported with its causes.
        log.warning(
            'Trapping: mush nodes could not drain (%s); %.0f%% of the mass that '
            'reached the porosity floor this step kept its entry pore melt, the '
            'no-drainage upper bound.',
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
            mass = -min(-mass, _trapped(hf_row, species))
        if mass != 0.0:
            out[species] = mass
    return out


def run_layer_trapping(
    config,
    hf_row: dict,
    prev: dict,
    interior_o,
    dirs: dict | None,
    dm_rm: float,
    melt_mass: float,
) -> TrappingStep | None:
    """One step of the layer drainage, or ``None`` when the interior cannot support it.

    Parameters
    ----------
    config : Config
        Model configuration.
    hf_row : dict
        Current helpfile row, modified in place.
    prev : dict
        Previous helpfile row.
    interior_o : Interior_t
        Interior state; carries the per-node state as ``trap_state``.
    dirs : dict or None
        Run directories; the state is written into the interior snapshot of each
        step that has one, and a resumed run reads it back from there.
    dm_rm : float
        Mantle mass crystallised over the step [kg], negative when it remelted.
    melt_mass : float
        Melt mass the dissolved volatiles were last partitioned into [kg].
    """
    profiles = layer_profiles(hf_row, interior_o)
    if profiles is None:
        return None
    tr = config.outgas
    state = getattr(interior_o, 'trap_state', None)
    if state is None and bool(getattr(getattr(config, 'params', None), 'resume', False)):
        state = _restore_state(dirs, prev, profiles['n'])
    trapped = {sp: _trapped(hf_row, sp) for sp in LAYER_SPECIES}
    if state is not None and state.size == profiles['n']:
        adopt_unrecorded(state, profiles['mass'], trapped)
    # The melt concentrations of the previous chemistry solve, on the melt it
    # dissolved into; a species whose elements the run does not carry locks nothing.
    liquid = {sp: float(hf_row.get(f'{sp}_kg_liquid', 0.0)) for sp in LAYER_SPECIES}
    concentration = {
        sp: melt_concentration(kg, melt_mass)
        for sp, kg in liquid.items()
        if kg > 0.0 and _carried(hf_row, sp)
    }
    mush_log10 = float(tr.trap_mush_log10visc)
    new_state, layers = advance_layers(
        state,
        time_prev=float(prev['Time']),
        time_now=float(hf_row['Time']),
        melt_fraction=profiles['melt_fraction'],
        rho_solid=profiles['rho_solid'],
        rho_melt=profiles['rho_melt'],
        gravity=profiles['gravity'],
        r_basic=profiles['r_basic'],
        mass=profiles['mass'],
        rfront_loc=critical_melt_fraction(config),
        phi_min=float(tr.trap_phi_min),
        grain_size=float(config.interior_energetics.grain_size),
        melt_visc=10.0 ** float(config.interior_energetics.melt_log10visc),
        mush_visc=10.0 ** (mush_log10 if mush_log10 > 0.0 else DEFAULT_MUSH_LOG10VISC),
        buoyancy=config.interior_energetics.aragog.separation_buoyancy,
        concentration=concentration,
        partition=partition_coefficients(config),
        liquid=liquid,
        crystallised=max(float(dm_rm), 0.0),
    )
    step = TrappingStep(mode='layers', dm_rm=dm_rm, f_tl=0.0, melt_mass=melt_mass)
    _describe(step, layers, hf_row, prev, profiles['mass'])
    step.trapped_kg = _capped_moves(hf_row, layers.moved_kg)
    step.supply_capped = list(layers.capped)
    if sum(layers.remelt_kg.values()) > 0.0 and step.total_trapped < 0.0:
        # A net return by remelting nodes is a remelting step for escape and desiccation.
        step.remelted = True
        step.branch = BRANCH_REMELT
    _apply_to_reservoirs(hf_row, step.trapped_kg)
    if state is None:
        # A record started this step holds only this step's locks; trapped mass
        # from before it is assigned now, so it remelts with its nodes.
        adopt_unrecorded(
            new_state, profiles['mass'], {sp: _trapped(hf_row, sp) for sp in LAYER_SPECIES}
        )
    interior_o.trap_state = new_state
    _record(hf_row, step)
    _log(step, layers)
    _persist_state(dirs, hf_row, new_state)
    return step
