"""Solid-phase volatile trapping during mantle crystallisation.

As the mantle crystallises, each increment of solidified mass buries volatiles
by two routes: lattice incorporation into the crystals at the crystal/melt
partition coefficient ``D_Z``, and burial of the interstitial melt that the
crystal pile fails to expel, at the trapped melt fraction ``F_tl``. The two
combine into one effective partition coefficient

    D_eff = (1 - F_tl) * D_Z + F_tl

so the mass a species loses from the melt over one step is

    dm_trap = D_eff * C_Z * dM_RM

with ``C_Z`` the species mass fraction in the melt and ``dM_RM`` the mass
crystallised over the step. A species at ``D_Z = 0`` still traps
``F_tl * C_Z * dM_RM``: interstitial melt is buried whatever the crystal
chemistry does, and it is how the noble gases dissolved in the melt are
buried. This is Sim, Hirschmann and Hier-Majumder (2024), JGR Planets 129,
e2024JE008346, Eq. 6.

``C_Z`` is the concentration the chemistry solver last computed, which is one
value for the whole mantle: the dissolved mass it left in the helpfile row over
the melt ``M_mantle * Phi_global`` it dissolved that mass into. Trapping runs
before this step's solve, so dividing by the current, smaller melt instead
would overstate the concentration by ``Phi(t-1) / Phi(t)``.

Critical melt fraction
----------------------
The disaggregation melt fraction ``phi_c`` is the interior solver's
rheological transition, ``interior_energetics.rfront_loc``: the melt fraction at
which the solver switches the mantle from melt-like to solid-like rheology is
the model's own disaggregation point, and a separate value would let trapping
and the rheology disagree about where the crystal framework locks. It marks the
top of the freezing front and caps the no-drainage bound.

Trapped-melt fraction
---------------------
``outgas.trap_mode`` selects whether trapping runs.

``none``
    No trapping. The bracket is not evaluated and no mass moves. This is the
    default, so enabling trapping is an explicit choice.
``front``
    ``F_tl`` is the melt a parcel still holds after crossing the freezing front
    the interior solver resolves, from the drainage integral of
    :mod:`proteus.outgas.compaction`, converted from a volume to a mass
    fraction. A front the mesh cannot resolve takes the
    no-drainage bound, the entry porosity capped at ``phi_c``. A step on which
    the interior state is unavailable buries at ``D_Z`` alone (``F_tl = 0``)
    and says so.
``layers``
    The mush drains node by node on the Aragog mesh instead of as one front
    (:mod:`proteus.outgas.node_record`, :mod:`proteus.outgas.layer_trapping`).
    Each node buries the volatiles of its pore melt when it enters the mush,
    drains that melt through the nodes above it, and locks what is left when
    it reaches the porosity floor; the chemistry dissolves into the melt of
    the liquid nodes alone. ``F_tl`` is the mass-weighted trapped fraction of
    the nodes that reached the floor over the step. Remelting is decided per
    node as well, so the burial ledger below is not used, also on a step
    without interior profiles, which traps at ``D_Z`` alone when the mantle
    freezes and returns the remelted share when it melts.

Remelting
---------
In the ``front`` mode, a step on which the global melt fraction rises remelts part of the solid
mantle and releases what trapping buried in it. The solid that remelts first is
the solid that froze last, so the step returns the mass buried while the melt
fraction fell through the interval it now rises back through, read from the
burial ledger of :mod:`proteus.outgas.burial_ledger`. Freezing and remelting
the same interval therefore cancel, and a mantle that remelts completely
returns everything trapped, whatever the melt has done in between. Trapped mass
the history does not account for, as in a run resumed from a helpfile written
before trapping, returns in proportion to the solid that remelted,
``(Phi(t) - Phi(t-1)) / (1 - Phi(t-1))``.

Reservoir bookkeeping
---------------------
Trapped mass moves from ``{sp}_kg_liquid`` to ``{sp}_kg_solid``, leaving
``{sp}_kg_total`` unchanged, and is recorded in ``{sp}_kg_trapped``. The same
mass is moved between the per-element reservoirs by stoichiometry, so the
per-element closure ``total == atm + liquid + solid + condensed`` holds after
the step. Trapping never touches ``{e}_kg_condensed``, the element mass in
outgassing condensates such as graphite, which only atmodeller with
``outgas.atmodeller.include_condensates`` fills; it is zero otherwise.

``_kg_trapped`` is the part of ``_kg_solid`` that trapping owns. The chemistry
backend writes zero to ``_kg_solid`` on every solve, CALLIOPE, atmodeller and the
dummy backend alike, and atmodeller writes its condensates to ``_kg_condensed``
instead. ``_kg_solid`` therefore holds only trapped mass, and condensed carbon is
never mistaken for trapped melt.

Trapped mass must be excluded anywhere the code assumes the whole-planet
inventory is reachable. :func:`locked_solid_mass` is the single definition of
that exclusion. :func:`unreachable_mass` adds the condensed mass held by
``outgas.prevent_desaturation`` (:func:`locked_condensed_mass`), and the escape
step and the desiccation gate subtract that sum. :func:`trapped_mass_withheld`
hides the trapped mass from the chemistry for the duration of a solve;
desiccation keeps it with the rest of the solid mantle.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from proteus.outgas.burial_ledger import build_ledger, ledger_total, remelt_to
from proteus.outgas.compaction import (
    BRANCH_DARCY,
    BRANCH_FALLBACK,
    BRANCH_GUARD,
    BRANCH_MATRIX,
    BRANCH_NONE,
    BRANCH_REMELT,
    DEFAULT_MUSH_LOG10VISC,
    drainage_integral,
    locate_front,
    porosity_from_densities,
    volume_to_mass_fraction,
)
from proteus.outgas.layer_trapping import run_layer_trapping
from proteus.outgas.trap_bookkeeping import (
    TRAPPED_COLUMNS,
    TRAPPED_SPECIES,
    TrappingStep,
    apply_to_reservoirs,
    critical_melt_fraction,
    front_speed,
    melt_concentration,
    move_mass,
    partition_coefficients,
    record_step,
    reservoir_sum,
    shift_reservoir,
    species_carried,
    trapped_kg,
)
from proteus.utils.constants import noble_gases, secs_per_year, vol_element_list, vol_list
from proteus.utils.helper import eval_gas_mmw

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Whether trapping runs. Kept in the order the config validator reports them.
TRAPPING_MODES = ('none', 'front', 'layers')


def effective_partition(f_tl: float | np.ndarray, d_z: float) -> float | np.ndarray:
    """Effective crystal/melt partition coefficient including trapped melt.

    Reduces to ``f_tl`` for a perfectly incompatible species (``d_z = 0``) and
    to 1 for a species that does not fractionate (``d_z = 1``). ``f_tl`` may be
    a scalar or a per-step array; the algebra is the same either way.

    Parameters
    ----------
    f_tl : float or ndarray
        Trapped melt fraction, a mass fraction in [0, 1].
    d_z : float
        Crystal/melt partition coefficient [1], non-negative.

    Returns
    -------
    float or ndarray
        ``(1 - f_tl) * d_z + f_tl`` [1].

    Raises
    ------
    ValueError
        If ``f_tl`` is not finite or lies outside [0, 1], or ``d_z`` is negative.
    """
    fraction = np.asarray(f_tl, dtype=float)
    if not np.all(np.isfinite(fraction)):
        raise ValueError('F_tl must be finite at every step')
    if np.any((fraction < 0.0) | (fraction > 1.0)):
        outside = fraction[(fraction < 0.0) | (fraction > 1.0)]
        raise ValueError(f'F_tl must lie in [0, 1], got {outside.ravel()[0]!r}')
    if d_z < 0.0:
        raise ValueError(f'D_Z must be non-negative, got {d_z!r}')
    return (1.0 - f_tl) * d_z + f_tl


def trapped_mass(
    d_z: float,
    f_tl: float,
    c_z: float,
    dm_rm: float,
    available_kg: float,
) -> tuple[float, bool]:
    """Mass of one species buried over a crystallisation increment [kg].

    ``[(1 - F_tl) D_Z + F_tl] C_Z dM_RM``, Sim et al. (2024) Eq. 6, capped at
    ``available_kg``, the mass the melt holds. A non-positive increment buries
    nothing. Returns the mass and whether the cap bound.

    Parameters
    ----------
    d_z : float
        Crystal/melt partition coefficient [1], non-negative.
    f_tl : float
        Trapped melt fraction, a mass fraction in [0, 1].
    c_z : float
        Mass fraction of the species in the melt [1].
    dm_rm : float
        Mantle mass crystallised over the increment [kg].
    available_kg : float
        Mass of the species the melt holds [kg].

    Returns
    -------
    tuple
        Buried mass [kg], at least 0, and whether the cap at ``available_kg`` bound.
    """
    raw = float(effective_partition(f_tl, d_z)) * float(c_z) * max(float(dm_rm), 0.0)
    available = max(float(available_kg), 0.0)
    return max(min(raw, available), 0.0), raw > available


def crystallised_mass_from_phi(m_mantle: float, phi_prev: float, phi_now: float) -> float:
    """Mass crystallised over one step [kg], from the melt fraction.

    ``dM_RM = M_mantle * [Phi(t-1) - Phi(t)]``. Phi_global is a mass fraction
    and therefore intensive, so a Zalmoxis re-solve that changes ``M_mantle``
    does not inject a spurious increment the way differencing
    ``M_mantle_solid`` would. It also matches the melt reservoir CALLIOPE
    dissolves into, which is sized from Phi_global rather than from
    ``M_mantle_liquid``; the prevent-warming clamp moves Phi_global without
    resynchronising those masses, so the two disagree on clamped steps.

    A negative increment means the mantle remelted that much mass, which
    releases trapped volatiles back to the melt.

    Parameters
    ----------
    m_mantle : float
        Mantle mass [kg]. A non-positive or non-finite value gives 0.
    phi_prev, phi_now : float
        Global melt fraction at the previous and the current step [1]. A
        non-finite value gives 0.
    """
    if not np.isfinite(m_mantle) or m_mantle <= 0.0:
        return 0.0
    if not (np.isfinite(phi_prev) and np.isfinite(phi_now)):
        return 0.0
    return float(m_mantle) * (float(phi_prev) - float(phi_now))


def remelted_fraction(phi_prev: float, phi_now: float) -> float:
    """Share of the solid mantle that remelted over one step [1].

    ``(Phi(t) - Phi(t-1)) / (1 - Phi(t-1))``, clipped to ``[0, 1]``: 1 when the
    mantle is fully molten again, 0 when it froze further or did not change.

    Parameters
    ----------
    phi_prev, phi_now : float
        Global melt fraction at the previous and the current step [1]. A
        non-finite value gives 0.
    """
    if not (np.isfinite(phi_prev) and np.isfinite(phi_now)):
        return 0.0
    solid_prev = 1.0 - float(phi_prev)
    if solid_prev <= 0.0:
        return 1.0 if phi_now > phi_prev else 0.0
    return float(np.clip((float(phi_now) - float(phi_prev)) / solid_prev, 0.0, 1.0))


def locked_solid_mass(hf_row: dict, element: str) -> float:
    """Whole-planet mass of one element locked in the solid mantle [kg].

    The single definition of what trapping has put beyond reach, read from
    ``{element}_kg_trapped``. Outgassing condensates such as atmodeller's graphite
    are held in ``{element}_kg_condensed``, not in ``_kg_solid``; see
    :func:`locked_condensed_mass` for when they are locked too.
    """
    return trapped_kg(hf_row, element)


def locked_condensed_mass(hf_row: dict, element: str, config: Config | None = None) -> float:
    """Mass of one element held in condensates that no longer re-vapourise [kg].

    The ``{element}_kg_condensed`` reservoir with ``outgas.prevent_desaturation``
    on, and 0 otherwise (also without a config), since then each solve may
    re-vapourise the condensate.
    """
    if config is None or getattr(config.outgas, 'prevent_desaturation', False) is not True:
        return 0.0
    mass = float(hf_row.get(f'{element}_kg_condensed', 0.0))
    return max(0.0, mass) if np.isfinite(mass) else 0.0


def unreachable_mass(hf_row: dict, element: str, config: Config | None = None) -> float:
    """Whole-planet mass of one element beyond the reach of escape and outgassing [kg].

    The mass trapped in the solid mantle, plus the condensed reservoir when
    condensates are kept from re-vapourising. The escape step subtracts this mass
    from the mass available to escape, and the desiccation gate subtracts it from
    the inventory it tests, so a planet whose atmosphere and melt have both emptied
    still registers as desiccated.
    """
    return locked_solid_mass(hf_row, element) + locked_condensed_mass(hf_row, element, config)


def escapable_inventory(hf_row: dict, element: str, config: Config | None = None) -> float:
    """Whole-planet inventory of one element that escape can still reach [kg]."""
    total = float(hf_row.get(f'{element}_kg_total', 0.0))
    if not np.isfinite(total):
        return total
    return max(0.0, total - unreachable_mass(hf_row, element, config))


def trapping_returned_mass(hf_row: dict) -> bool:
    """Whether this step's trapping returned trapped mass to the melt.

    Reads ``trap_kg_returned``, the mass the step's net moves returned, which a
    remelt in either scheme and melt draining back to the ocean in ``layers``
    both produce. ``trap_branch`` is a diagnostic and is not read here.
    """
    returned = float(hf_row.get('trap_kg_returned', 0.0))
    return bool(np.isfinite(returned) and returned > 0.0)


def trapping_active(config: Config) -> bool:
    """Whether the run traps volatiles; with ``trap_mode = 'none'`` nothing
    this module owns may change what the run does."""
    mode = getattr(config.outgas, 'trap_mode', 'none')
    return isinstance(mode, str) and mode != 'none'


def derived_total_elements(config: Config) -> tuple[str, ...]:
    """Elements whose whole-planet total the chemistry recomputes each step.

    Oxygen under ``planet.fO2_source = 'user_constant'`` is the only case.
    Holding the oxygen fugacity at a fixed buffer offset requires the chemistry
    to move oxygen, so ``O_kg_total`` is an output of the solve rather than a
    conserved budget the escape chain debits; ``outgas/common.py`` says as much
    where it lists the ``_kg_total`` slot as escape-owned for every element
    except O. Under ``'from_O_budget'`` the wrapper restores the authoritative
    budget after the copy, so oxygen is conserved state there and is absent
    from this tuple.

    Trapping still treats oxygen like every other element around a solve: the
    trapped oxygen is taken out before it and added back to the total the
    chemistry writes, so ``O_kg_total`` is the whole planet's oxygen under
    either source. The per-element closure is not asserted for it, because the
    chemistry total closes only to the solver's own convention.
    """
    if getattr(getattr(config, 'planet', None), 'fO2_source', '') == 'user_constant':
        return ('O',)
    return ()


def withhold_trapped_mass(hf_row: dict) -> None:
    """Take the trapped mass out of the reservoirs a chemistry solve owns.

    Every outgassing backend partitions a whole-planet inventory between melt and
    atmosphere, and rewrites the ``_kg_solid`` columns, with zero for every backend;
    atmodeller writes its condensates to ``_kg_condensed`` instead. Trapped mass
    belongs to neither. Left in an element total, it would be re-dissolved by the
    solve; left in a solid column, it would be overwritten by the backend. It is
    therefore subtracted from each element's
    ``_kg_total`` and from each species' and element's ``_kg_solid``, which leaves
    the backend exactly the share it owns.

    Pair with :func:`restore_trapped_mass`, or use :func:`trapped_mass_withheld`,
    which pairs the two. An element whose total the chemistry recomputes, such
    as oxygen under a buffered fO2, is treated the same way: the solve writes its
    own total over the withheld one and the restore adds the trapped share to
    it, and a solve that writes nothing leaves the total where it started.
    """
    for species in vol_list:
        mass = trapped_kg(hf_row, species)
        if mass > 0.0:
            shift_reservoir(hf_row, f'{species}_kg_solid', -mass)
    for element in (*vol_element_list, *noble_gases):
        mass = trapped_kg(hf_row, element)
        if mass <= 0.0:
            continue
        shift_reservoir(hf_row, f'{element}_kg_solid', -mass)
        shift_reservoir(hf_row, f'{element}_kg_total', -mass)


def restore_trapped_mass(hf_row: dict) -> None:
    """Put the trapped mass back after a chemistry solve.

    Adds it to whatever the backend wrote into ``_kg_solid``, and restores each
    element's whole-planet ``_kg_total``. A species total is an output of the
    solve, the backend's atmosphere plus melt, so for every trapped species it is
    rebuilt as ``atm + liquid + solid``, and the solid and total moles follow the
    masses. Putting back the species total saved before the solve would instead
    overwrite the partition the solve has just produced.
    """
    for species in vol_list:
        mass = trapped_kg(hf_row, species)
        if mass <= 0.0:
            continue
        shift_reservoir(hf_row, f'{species}_kg_solid', mass)
        hf_row[f'{species}_kg_total'] = reservoir_sum(hf_row, species, 'kg')
        mmw = eval_gas_mmw(species)
        hf_row[f'{species}_mol_solid'] = float(hf_row[f'{species}_kg_solid']) / mmw
        hf_row[f'{species}_mol_total'] = reservoir_sum(hf_row, species, 'mol')
    for element in (*vol_element_list, *noble_gases):
        mass = trapped_kg(hf_row, element)
        if mass <= 0.0:
            continue
        shift_reservoir(hf_row, f'{element}_kg_solid', mass)
        shift_reservoir(hf_row, f'{element}_kg_total', mass)
        if element in noble_gases:
            # A noble gas is its own element; its moles follow the masses.
            mmw = eval_gas_mmw(element)
            hf_row[f'{element}_mol_solid'] = float(hf_row[f'{element}_kg_solid']) / mmw
            hf_row[f'{element}_mol_total'] = float(hf_row[f'{element}_kg_total']) / mmw


@contextmanager
def trapped_mass_withheld(hf_row: dict):
    """Hide the trapped mass from the chemistry for the duration of a solve.

    Withholds on entry and restores on exit, also when the solve raises, so the
    row never carries the reachable inventory as the whole planet's.
    """
    withhold_trapped_mass(hf_row)
    try:
        yield
    finally:
        restore_trapped_mass(hf_row)


def _usable_gravity(values, n: int) -> np.ndarray | None:
    """``values`` as ``n`` finite, positive accelerations [m s-2], else ``None``."""
    if values is None:
        return None
    g = np.asarray(values, dtype=float).ravel()
    if g.size == 1:
        g = np.full(n, g[0])
    if g.size != n or not np.all(np.isfinite(g)) or np.any(g <= 0.0):
        return None
    return g


def _basic_profiles(interior_o) -> tuple[np.ndarray, ...] | None:
    """The interior solver's basic-node profiles the front drains with, or ``None``.

    Radius [m], melt fraction [1], mixture density, solid and melt densities at
    the phase boundary [kg m-3], and gravity [m s-2], each at the N+1 basic
    nodes from the core-mantle boundary up. Aragog writes them on every step
    (``SolverOutput.phi_basic``, ``rho_basic``, ``rho_solid_b``, ``rho_melt_b``,
    ``g_b``); the gravity is ``None`` when unusable, and the rest returns
    ``None`` when absent, mis-sized or non-finite, which no backend but aragog
    supplies.
    """
    radius = np.asarray(getattr(interior_o, 'radius', []), dtype=float).ravel()
    fields = [getattr(interior_o, name, None) for name in _BASIC_FIELDS]
    arrays = [np.asarray(f, dtype=float).ravel() if f is not None else None for f in fields]
    usable = radius.size > 1 and all(
        a is not None and a.size == radius.size and np.all(np.isfinite(a)) for a in arrays
    )
    if not usable:
        log.warning(
            'Trapping: the interior basic-node profiles needed to locate a freezing '
            'front are unavailable (radius %d nodes; %s), so the step buries at the '
            'crystal partition coefficients alone. The drainage integral needs an '
            'aragog interior.',
            radius.size,
            ', '.join(
                f'{name} {"absent" if a is None else a.size}'
                for name, a in zip(_BASIC_FIELDS, arrays)
            ),
        )
        return None
    gravity = _usable_gravity(getattr(interior_o, 'g_b', None), radius.size)
    return (radius, *arrays, gravity)


# Basic-node fields of the interior state, besides the radius and the gravity.
_BASIC_FIELDS = ('phi_b', 'rho_b', 'rho_solid_b', 'rho_melt_b')


def _drainage_fraction(
    config, hf_row: dict, prev: dict, interior_o, dm_rm: float
) -> TrappingStep | None:
    """Trapped melt fraction from the drainage integral over the front.

    Returns ``None`` when the interior state needed to locate a front is not
    available, so the caller can bury at the crystal partition coefficients alone.
    """
    tr = config.outgas
    profiles = _basic_profiles(interior_o)
    if profiles is None:
        return None
    radius, phi_b, rho_b, rho_s, rho_l, g_b = profiles

    porosity = porosity_from_densities(rho_b, rho_s, rho_l, phi_b)
    # The lowest basic node is the core-mantle boundary, the base of a front that
    # is still porous there.
    rfront_loc = critical_melt_fraction(config)
    geom, branch = locate_front(
        radius,
        phi_b,
        porosity,
        rho_s,
        rho_l,
        rfront_loc=rfront_loc,
        phi_min=float(tr.trap_phi_min),
        n_front_min=int(tr.trap_n_front_min),
        max_front_fraction=float(tr.trap_max_front_fraction),
    )
    if geom is None:
        return TrappingStep(
            mode='front', dm_rm=0.0, f_tl=0.0, melt_mass=0.0, branch=BRANCH_NONE
        )

    # Front speed from the crystallised mass, the increment the flux uses, not from
    # the node-snapped front depth, whose difference is zero on most steps and a
    # spike on the rest.
    dt_s = (float(hf_row['Time']) - float(prev['Time'])) * secs_per_year
    v_f = front_speed(dm_rm, geom.r_base, geom.rho_solid_base, dt_s)

    grain = float(config.interior_energetics.grain_size)
    melt_visc = 10.0 ** float(config.interior_energetics.melt_log10visc)
    mush_log10 = float(tr.trap_mush_log10visc)
    mush_visc = 10.0 ** (mush_log10 if mush_log10 > 0.0 else DEFAULT_MUSH_LOG10VISC)
    # Gravity averaged over the front nodes, like the density contrast: a front
    # near the core-mantle boundary sits well below the surface value.
    if g_b is None:
        gravity = float(hf_row.get('gravity', 0.0))
        log.warning(
            'Trapping: the interior solver carries no usable basic-node gravity, so '
            'the front drains under the surface gravity %.3f m/s2.',
            gravity,
        )
    else:
        gravity = float(np.mean(g_b[geom.index]))

    step = TrappingStep(
        mode='front',
        dm_rm=0.0,
        f_tl=0.0,
        melt_mass=0.0,
        branch=branch,
        n_front=int(geom.index.size),
        l_front=geom.thickness,
        v_front=v_f,
    )

    # Guard paths take the entry porosity as a conservative upper bound rather
    # than integrating a front the mesh cannot resolve. The advance over one step
    # is reported, not guarded: t_res = L / v_f does not depend on the step length.
    courant = (v_f * dt_s / geom.thickness) if geom.thickness > 0.0 else float('inf')
    step.front_courant = courant
    reasons = [geom.guard_reason or 'the front is unresolved'] if branch == BRANCH_GUARD else []
    if v_f <= 0.0:
        reasons.append('the front speed is undefined on a step with no length')
    if reasons:
        step.branch = BRANCH_GUARD
        step.guard_reason = '; '.join(reasons)
        # No melt drains, but no more can be held than the crystal framework
        # locks at the critical melt fraction, whatever the density-derived
        # porosity says.
        entry = volume_to_mass_fraction(geom.porosity_top, geom.rho_melt, geom.rho_solid_base)
        step.f_tl = min(entry, critical_melt_fraction(config))
        return step

    t_res = geom.thickness / v_f
    f_vol, tau_d, tau_s = drainage_integral(
        geom.porosity_top,
        t_res,
        geom.thickness,
        grain,
        geom.delta_rho,
        gravity,
        melt_visc,
        mush_visc,
        buoyancy=config.interior_energetics.aragog.separation_buoyancy,
    )
    step.f_tl = volume_to_mass_fraction(f_vol, geom.rho_melt, geom.rho_solid_base)
    step.tau_d = tau_d / secs_per_year
    step.tau_s = tau_s / secs_per_year
    step.t_res = t_res / secs_per_year
    step.branch = BRANCH_MATRIX if tau_s > tau_d else BRANCH_DARCY
    w_top = geom.thickness / tau_d if np.isfinite(tau_d) and tau_d > 0.0 else 0.0
    step.w_matrix_over_vf = (geom.porosity_top * w_top / v_f) if v_f > 0.0 else float('nan')
    if np.isfinite(step.w_matrix_over_vf) and step.w_matrix_over_vf > 0.1:
        log.warning(
            'Trapping: matrix speed is %.2f of the front speed, so the '
            'parcel is not fixed in the matrix frame.',
            step.w_matrix_over_vf,
        )
    return step


def run_trapping(
    config: Config,
    hf_row: dict,
    hf_all,
    interior_o=None,
    *,
    init_stage: bool = False,
    dirs: dict | None = None,
) -> TrappingStep | None:
    """Bury volatiles into the solid mantle over this crystallisation step.

    Called once per iteration, after the interior has advanced and any structure
    re-solve has finished, and before escape and outgassing read the
    inventories. Returns ``None`` when trapping is inactive: during the
    initialisation stage, whose interior steps do not advance the mantle,
    when ``hf_all`` holds no earlier row or the time is not above 0, where
    there is no previous state to difference, and whenever the mode is
    ``none``. The first step after the initialisation stage differences
    against the last row of that stage.

    Parameters
    ----------
    config : Config
        Model configuration.
    hf_row : dict
        Current helpfile row, modified in place.
    hf_all : pandas.DataFrame or None
        Completed rows. The last one supplies the previous melt fraction.
    interior_o : Interior_t or None
        Interior state. The front scheme reads its melt-fraction, density and
        pressure profiles to locate the freezing front.
    init_stage : bool
        True during the initialisation stage, which traps nothing.
    dirs : dict or None
        Run directories. The ``layers`` mode keeps its per-node state in the
        interior snapshots under them, for a resumed run to read back.
    """
    mode = getattr(config.outgas, 'trap_mode', 'none')
    # A config that never declared a mode traps nothing: the schema restricts the
    # field at load, so a non-string means a config built in code without trapping,
    # whose other attributes nobody chose as trapping parameters.
    if not isinstance(mode, str):
        return None
    if mode not in TRAPPING_MODES:
        raise ValueError(f'Unknown trapping mode {mode!r}; expected one of {TRAPPING_MODES}')
    if mode == 'none':
        return None
    if mode == 'layers':
        # Only a layer step of this iteration may hand the chemistry an ocean melt.
        hf_row['trap_Phi_ocean'] = float('nan')
    if init_stage:
        # The initialisation iterations advance the clock but not the mantle,
        # and the time is reset to zero after each, so there is nothing to bury.
        return None
    if hf_all is None or len(hf_all) < 1:
        return None
    if float(hf_row.get('Time', 0.0)) <= 0.0:
        return None

    prev = hf_all.iloc[-1].to_dict()
    m_mantle = float(hf_row.get('M_mantle', 0.0))
    phi_now = float(hf_row.get('Phi_global', float('nan')))
    phi_prev = float(prev.get('Phi_global', float('nan')))
    dm_rm = crystallised_mass_from_phi(m_mantle, phi_prev, phi_now)
    # The dissolved masses in hf_row are still those of the previous chemistry
    # solve, so the melt they were dissolved into is the previous one. Dividing
    # by it recovers the concentration the solver computed.
    m_mantle_prev = float(prev.get('M_mantle', m_mantle))
    melt_mass = m_mantle_prev * max(0.0, min(1.0, phi_prev)) if np.isfinite(phi_prev) else 0.0
    if mode == 'layers':
        return run_layer_trapping(config, hf_row, prev, interior_o, dirs, dm_rm, melt_mass)
    if dm_rm < 0.0:
        released = released_mass(hf_row, hf_all, phi_prev, phi_now)
        return _release(hf_row, mode, dm_rm, melt_mass, released)
    if dm_rm == 0.0 or melt_mass <= 0.0:
        step = TrappingStep(mode=mode, dm_rm=dm_rm, f_tl=0.0, melt_mass=melt_mass)
        record_step(hf_row, step)
        return step

    step = _drainage_fraction(config, hf_row, prev, interior_o, dm_rm)
    if step is None:
        # Without the interior profiles there is no front; the crystals still
        # take up D_Z, so the step buries at that alone, on a branch of its own.
        step = TrappingStep(mode=mode, dm_rm=0.0, f_tl=0.0, melt_mass=0.0)
        step.branch = BRANCH_FALLBACK
    step.mode = mode
    step.dm_rm = dm_rm
    step.melt_mass = melt_mass

    d_z = partition_coefficients(config)
    for species in TRAPPED_SPECIES:
        kg_liquid = float(hf_row.get(f'{species}_kg_liquid', 0.0))
        if kg_liquid <= 0.0 or not species_carried(hf_row, species):
            continue
        c_z = melt_concentration(kg_liquid, melt_mass)
        mass, capped = trapped_mass(d_z[species], step.f_tl, c_z, dm_rm, kg_liquid)
        if mass <= 0.0:
            continue
        if capped:
            step.supply_capped.append(species)
        step.trapped_kg[species] = mass

    apply_to_reservoirs(hf_row, step.trapped_kg)
    record_step(hf_row, step)
    return step


def released_mass(hf_row: dict, hf_all, phi_prev: float, phi_now: float) -> dict[str, float]:
    """Mass each trapped species and element returns to the melt on a remelt [kg].

    The mass the burial ledger holds below ``phi_now``, rebuilt from the
    completed rows in ``hf_all``. Trapped mass the ledger does not hold returns
    in proportion to the solid that remelted; a complete remelt returns all of
    it. Each release is capped at what is trapped.
    """
    current = np.array([trapped_kg(hf_row, name) for name in TRAPPED_COLUMNS])
    share = remelted_fraction(phi_prev, phi_now)
    if share >= 1.0:
        return dict(zip(TRAPPED_COLUMNS, current))
    history = np.column_stack(
        [
            hf_all[f'{name}_kg_trapped'].to_numpy(dtype=float)
            if f'{name}_kg_trapped' in hf_all
            else np.zeros(len(hf_all))
            for name in TRAPPED_COLUMNS
        ]
    )
    layers = build_ledger(hf_all['Phi_global'].to_numpy(dtype=float), history)
    held = ledger_total(layers, len(TRAPPED_COLUMNS))
    from_ledger = remelt_to(layers, phi_now) if layers else np.zeros(len(TRAPPED_COLUMNS))
    unledgered = np.maximum(current - held, 0.0)
    release = np.minimum(from_ledger + share * unledgered, current)
    return dict(zip(TRAPPED_COLUMNS, release))


def _release(
    hf_row: dict, mode: str, dm_rm: float, melt_mass: float, released: dict[str, float]
) -> TrappingStep:
    """Return the ``released`` mass of each trapped species and element to the melt."""
    step = TrappingStep(mode=mode, dm_rm=dm_rm, f_tl=0.0, melt_mass=melt_mass, remelted=True)
    step.branch = BRANCH_REMELT
    for name, mass in released.items():
        if mass <= 0.0:
            continue
        if name in TRAPPED_SPECIES:
            step.trapped_kg[name] = -mass
        move_mass(hf_row, name, -mass)
    record_step(hf_row, step)
    return step
