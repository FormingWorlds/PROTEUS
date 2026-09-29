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
chemistry does. This is Sim, Hirschmann and Hier-Majumder (2024), JGR Planets
129, e2024JE008346, Eq. 6.

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
    fraction. A front the mesh or the step cannot resolve takes the
    no-drainage bound, the entry porosity capped at ``phi_c``. A step on which
    the interior state is unavailable buries at ``D_Z`` alone (``F_tl = 0``)
    and says so.

Remelting
---------
A step on which the global melt fraction rises remelts part of the solid
mantle. The flux is computed exactly as for a crystallising step, the same
bracket at the same ``C_Z`` and ``F_tl`` over the remelted mass, and applied
in reverse: the mass moves from the trapped reservoir back into the melt,
capped at what trapping holds of each species. The remelted material is the
most recently crystallised, so it is released at the rate it was buried.

Reservoir bookkeeping
---------------------
Trapped mass moves from ``{sp}_kg_liquid`` to ``{sp}_kg_solid``, leaving
``{sp}_kg_total`` unchanged, and is recorded in ``{sp}_kg_trapped``. The same
mass is moved between the per-element reservoirs by stoichiometry, so the
per-element closure ``total == atm + liquid + solid`` holds after the step.

``_kg_trapped`` is the part of ``_kg_solid`` that trapping owns. The rest of
``_kg_solid`` belongs to the chemistry backend, which writes it on every solve:
zero for CALLIOPE and the dummy backend, and the condensed graphite for
atmodeller. Keeping the two apart lets the chemistry own its share while the
trapped share stays buried, so condensed carbon is never mistaken for trapped
melt.

Trapped mass must be excluded anywhere the code assumes the whole-planet
inventory is reachable. :func:`locked_solid_mass` is the single definition of
that exclusion, used by the escape step and the desiccation gate.
:func:`trapped_mass_withheld` hides it from the chemistry for the duration of a
solve, and :func:`keep_only_trapped_mass` keeps it through desiccation.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from proteus.outgas.common import element_masses_from_species
from proteus.outgas.compaction import (
    BRANCH_DARCY,
    BRANCH_FALLBACK,
    BRANCH_GUARD,
    BRANCH_MATRIX,
    BRANCH_NONE,
    DEFAULT_MUSH_LOG10VISC,
    drainage_integral,
    locate_front,
    porosity_from_densities,
    volume_to_mass_fraction,
)
from proteus.utils.constants import noble_gases, vol_element_list, vol_list
from proteus.utils.helper import eval_gas_mmw

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Whether trapping runs. Kept in the order the config validator reports them.
TRAPPING_MODES = ('none', 'front')


@dataclass
class TrappingStep:
    """What one trapping step moved, and the guards it tripped."""

    mode: str
    dm_rm: float
    f_tl: float
    melt_mass: float
    trapped_kg: dict[str, float] = field(default_factory=dict)
    remelted: bool = False
    supply_capped: list[str] = field(default_factory=list)
    branch: int = BRANCH_NONE
    tau_d: float = float('nan')
    tau_s: float = float('nan')
    t_res: float = float('nan')
    n_front: int = 0
    l_front: float = float('nan')
    v_front: float = float('nan')
    front_courant: float = float('nan')
    w_matrix_over_vf: float = float('nan')
    guard_reason: str = ''
    n_exited: int = 0  # mush nodes that reached the porosity floor this step
    frac_bound: float = float('nan')  # share of their mass on the no-drainage bound
    n_substeps: int = 0  # drainage sub-steps the step took

    @property
    def total_trapped(self) -> float:
        """Net mass buried this step, summed over species [kg].

        Negative on a remelting step, which releases trapped mass to the melt.
        """
        return float(sum(self.trapped_kg.values()))


def effective_partition(f_tl: float | np.ndarray, d_z: float) -> float | np.ndarray:
    """Effective crystal/melt partition coefficient including trapped melt.

    Reduces to ``f_tl`` for a perfectly incompatible species (``d_z = 0``) and
    to 1 for a species that does not fractionate (``d_z = 1``). ``f_tl`` may be
    a scalar or a per-step array; the algebra is the same either way.
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


def melt_concentration(
    kg_liquid: float | np.ndarray, melt_mass: float | np.ndarray
) -> float | np.ndarray:
    """Species mass fraction in the melt [1]; zero wherever no melt remains.

    A fully crystallised mantle carries no melt to draw from, so the quotient is
    replaced by zero instead of dividing by zero.
    """
    kg = np.maximum(np.asarray(kg_liquid, dtype=float), 0.0)
    melt = np.asarray(melt_mass, dtype=float)
    c_z = np.zeros(np.shape(kg), dtype=float)
    np.divide(kg, melt, out=c_z, where=melt > 0.0)
    if np.ndim(kg_liquid) == 0:
        return float(c_z)
    return c_z


def trapped_mass(
    d_z: float,
    f_tl: float,
    c_z: float,
    dm_rm: float,
    available_kg: float,
) -> tuple[float, bool]:
    """Mass of one species moved over a crystallisation increment [kg].

    ``[(1 - F_tl) D_Z + F_tl] C_Z |dM_RM|``, Sim et al. (2024) Eq. 6. The same
    flux is buried on a crystallising step and released on a remelting one, so
    only the magnitude of ``dm_rm`` enters. The result is capped at
    ``available_kg``: the mass the melt holds when the species is buried, the
    mass trapping holds when it is released. Returns the mass and whether the
    cap bound.
    """
    raw = float(effective_partition(f_tl, d_z)) * float(c_z) * abs(float(dm_rm))
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
    """
    if not np.isfinite(m_mantle) or m_mantle <= 0.0:
        return 0.0
    if not (np.isfinite(phi_prev) and np.isfinite(phi_now)):
        return 0.0
    return float(m_mantle) * (float(phi_prev) - float(phi_now))


def critical_melt_fraction(config: Config) -> float:
    """Disaggregation melt fraction ``phi_c`` [1], the solver's rheological transition.

    The single definition of ``phi_c`` for the trapping step, so the top of the
    drainage front and the no-drainage bound cannot drift apart from each other
    or from the interior rheology.
    """
    return float(config.interior_energetics.rfront_loc)


def partition_coefficients(config: Config) -> dict[str, float]:
    """Crystal/melt partition coefficient per species, from config.

    Water is the only species with appreciable lattice incorporation; the rest
    default to zero and are buried through the interstitial-melt term alone.
    """
    return {s: float(getattr(config.outgas, f'D_const_{s}', 0.0)) for s in vol_list}


def _trapped(hf_row: dict, name: str) -> float:
    """Mass of one species or element that trapping has buried [kg], else zero."""
    mass = float(hf_row.get(f'{name}_kg_trapped', 0.0))
    if not np.isfinite(mass) or mass <= 0.0:
        return 0.0
    return mass


def _shift(hf_row: dict, key: str, delta: float) -> None:
    """Add ``delta`` to one reservoir, floored at zero; a non-finite value is left alone."""
    value = float(hf_row.get(key, 0.0))
    if np.isfinite(value):
        hf_row[key] = max(0.0, value + delta)


def _reservoir_sum(hf_row: dict, name: str, unit: str) -> float:
    """Atmosphere plus melt plus solid for one species, in ``kg`` or ``mol``."""
    return sum(float(hf_row.get(f'{name}_{unit}_{r}', 0.0)) for r in ('atm', 'liquid', 'solid'))


def locked_solid_mass(hf_row: dict, element: str) -> float:
    """Whole-planet mass of one element locked in the solid mantle [kg].

    The single definition of what trapping has put beyond reach, read from
    ``{element}_kg_trapped``. The rest of ``{element}_kg_solid`` is the chemistry
    backend's own condensate, such as atmodeller's graphite, which is not locked.
    The escape step subtracts this mass from the mass available to escape, and
    the desiccation gate subtracts it from the inventory it tests, so a planet
    whose atmosphere and melt have both emptied still registers as desiccated.
    """
    return _trapped(hf_row, element)


def escapable_inventory(hf_row: dict, element: str) -> float:
    """Whole-planet inventory of one element that escape can still reach [kg]."""
    total = float(hf_row.get(f'{element}_kg_total', 0.0))
    if not np.isfinite(total):
        return total
    return max(0.0, total - locked_solid_mass(hf_row, element))


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

    Trapping must therefore neither withhold nor debit the oxygen total in the
    ``user_constant`` case: the trapped oxygen is recorded in ``O_kg_trapped``
    and ``O_kg_solid`` as a diagnostic, and the per-element closure is not
    asserted for it.
    """
    if getattr(getattr(config, 'planet', None), 'fO2_source', '') == 'user_constant':
        return ('O',)
    return ()


def withhold_trapped_mass(hf_row: dict, derived: tuple[str, ...] = ()) -> None:
    """Take the trapped mass out of the reservoirs a chemistry solve owns.

    Every outgassing backend partitions a whole-planet inventory between melt and
    atmosphere, and rewrites the ``_kg_solid`` columns with its own condensate:
    zero for CALLIOPE and the dummy backend, graphite for atmodeller. Trapped mass
    belongs to neither. Left in an element total, it would be re-dissolved by the
    solve; left in a solid column, it would be added to by a backend that sums its
    condensate into that column. It is therefore subtracted from each element's
    ``_kg_total`` and from each species' and element's ``_kg_solid``, which leaves
    the backend exactly the share it owns.

    Pair with :func:`restore_trapped_mass`, or use :func:`trapped_mass_withheld`,
    which pairs the two. ``derived`` names elements whose total the chemistry
    recomputes (see :func:`derived_total_elements`); their total is left alone.
    """
    for species in vol_list:
        mass = _trapped(hf_row, species)
        if mass > 0.0:
            _shift(hf_row, f'{species}_kg_solid', -mass)
    for element in vol_element_list:
        mass = _trapped(hf_row, element)
        if mass <= 0.0:
            continue
        _shift(hf_row, f'{element}_kg_solid', -mass)
        if element not in derived:
            _shift(hf_row, f'{element}_kg_total', -mass)


def restore_trapped_mass(hf_row: dict, derived: tuple[str, ...] = ()) -> None:
    """Put the trapped mass back after a chemistry solve.

    Adds it to whatever the backend wrote into ``_kg_solid``, and restores each
    element's whole-planet ``_kg_total``. A species total is an output of the
    solve, the backend's atmosphere plus melt, so for every trapped species it is
    rebuilt as ``atm + liquid + solid``, and the solid and total moles follow the
    masses. Putting back the species total saved before the solve would instead
    overwrite the partition the solve has just produced.
    """
    for species in vol_list:
        mass = _trapped(hf_row, species)
        if mass <= 0.0:
            continue
        _shift(hf_row, f'{species}_kg_solid', mass)
        hf_row[f'{species}_kg_total'] = _reservoir_sum(hf_row, species, 'kg')
        mmw = eval_gas_mmw(species)
        hf_row[f'{species}_mol_solid'] = float(hf_row[f'{species}_kg_solid']) / mmw
        hf_row[f'{species}_mol_total'] = _reservoir_sum(hf_row, species, 'mol')
    for element in vol_element_list:
        mass = _trapped(hf_row, element)
        if mass <= 0.0:
            continue
        _shift(hf_row, f'{element}_kg_solid', mass)
        if element not in derived:
            _shift(hf_row, f'{element}_kg_total', mass)


@contextmanager
def trapped_mass_withheld(hf_row: dict, derived: tuple[str, ...] = ()):
    """Hide the trapped mass from the chemistry for the duration of a solve.

    Withholds on entry and restores on exit, also when the solve raises, so the
    row never carries the reachable inventory as the whole planet's.
    """
    withhold_trapped_mass(hf_row, derived)
    try:
        yield
    finally:
        restore_trapped_mass(hf_row, derived)


def keep_only_trapped_mass(hf_row: dict, derived: tuple[str, ...] = ()) -> None:
    """Leave only the trapped mass in a row whose atmosphere and melt were emptied.

    Desiccation empties the atmosphere and the melt, while what trapping buried
    stays in the solid mantle. Once the outgassing reservoirs have been zeroed,
    each trapped species and element gets its trapped mass back as its solid
    reservoir, and the total of every volatile and noble element is set to what
    it still holds, its trapped mass or zero, so that each total equals the sum of
    its reservoirs. The total of a ``derived`` element follows the chemistry's
    convention, which excludes the solid, and is left as it is.
    """
    for species in vol_list:
        mass = _trapped(hf_row, species)
        if mass <= 0.0:
            continue
        mol = mass / eval_gas_mmw(species)
        hf_row[f'{species}_kg_solid'] = mass
        hf_row[f'{species}_kg_total'] = mass
        hf_row[f'{species}_mol_solid'] = mol
        hf_row[f'{species}_mol_total'] = mol
    for element in list(vol_element_list) + list(noble_gases):
        mass = _trapped(hf_row, element)
        hf_row[f'{element}_kg_solid'] = mass
        if element not in derived:
            hf_row[f'{element}_kg_total'] = mass


def _apply_to_reservoirs(hf_row: dict, moved: dict[str, float]) -> None:
    """Move mass between the melt and the solid, per species and per element.

    A positive mass is buried: it leaves ``_kg_liquid`` for ``_kg_solid``, and
    is added to ``_kg_trapped``, the share of ``_kg_solid`` that trapping owns
    and that the chemistry solve must leave alone. A negative mass is released
    by remelting and moves back the other way.
    """
    for species, mass in moved.items():
        if mass != 0.0:
            _move(hf_row, species, mass)
    for element, mass in element_masses_from_species(moved).items():
        if element in vol_element_list and mass != 0.0:
            _move(hf_row, element, mass)


def _move(hf_row: dict, name: str, mass: float) -> None:
    """Take ``mass`` of one species or element out of the melt into the solid."""
    liq = float(hf_row.get(f'{name}_kg_liquid', 0.0))
    sol = float(hf_row.get(f'{name}_kg_solid', 0.0))
    hf_row[f'{name}_kg_liquid'] = max(0.0, liq - mass)
    hf_row[f'{name}_kg_solid'] = max(0.0, sol + mass)
    hf_row[f'{name}_kg_trapped'] = max(0.0, _trapped(hf_row, name) + mass)


SECS_PER_YEAR = 3.15576e7  # Julian year, matching the interior solver's step length


def _phase_densities(interior_o, pressure: np.ndarray):
    """End-member densities at the node pressures [kg m-3], or ``None``.

    Read from the interior solver's own equation of state so the porosity and
    the density contrast are the ones the solver itself would compute. Uses a
    private lookup because the solver does not yet expose a public accessor;
    returns ``None`` when the solver or its EOS is absent, which is every
    backend other than aragog.
    """
    solver = getattr(interior_o, 'aragog_solver', None)
    # The entropy solver holds its equation of state as ``entropy_eos``. The
    # bare ``eos`` name belongs to an unrelated root-finding helper in the same
    # module, so both are tried rather than assuming either.
    eos = getattr(solver, 'entropy_eos', None) or getattr(solver, 'eos', None)
    lookup = getattr(eos, '_lookup_at_phase_boundary', None)
    if lookup is None:
        log.warning(
            'Trapping: no phase-boundary lookup on the interior solver '
            '(solver=%s, eos=%s), so the drainage integral cannot run and the '
            'step buries at the crystal partition coefficients alone.',
            type(solver).__name__ if solver is not None else None,
            type(eos).__name__ if eos is not None else None,
        )
        return None
    try:
        rho_s = np.asarray(lookup('density', pressure, 'solid'), dtype=float).ravel()
        rho_l = np.asarray(lookup('density', pressure, 'melt'), dtype=float).ravel()
    except Exception as exc:
        log.warning(
            'Phase-boundary densities unavailable (%s); trapping buries at the '
            'crystal partition coefficients alone this step.',
            exc,
        )
        return None
    if rho_s.shape != rho_l.shape or not np.all(np.isfinite(rho_s + rho_l)):
        log.warning(
            'Trapping: phase-boundary densities are inconsistent or non-finite '
            '(shapes %s and %s), so the step buries at the crystal partition '
            'coefficients alone.',
            rho_s.shape,
            rho_l.shape,
        )
        return None
    return rho_s, rho_l


def _staggered_radii(radius: np.ndarray, n_stag: int) -> np.ndarray:
    """Midpoints of the basic mesh, which is where the staggered fields live."""
    r = np.asarray(radius, dtype=float).ravel()
    if r.size == n_stag:
        return r
    if r.size == n_stag + 1:
        return 0.5 * (r[:-1] + r[1:])
    return r[:n_stag]


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


def _node_gravity(interior_o, r_stag: np.ndarray) -> np.ndarray | None:
    """Gravitational acceleration on the staggered nodes [m s-2], or ``None``.

    The interior solver's own per-node values come first, so the front drains
    under the gravity the thermal evolution uses. With a Zalmoxis structure
    they are its radial profile interpolated onto the solver's nodes, or its
    surface value everywhere when ``scalar_gravity_override`` flattens that
    profile on purpose. The structure profile the solver was built from is the
    fallback, interpolated here. Both are read from solver internals that have
    no public accessor yet.
    """
    solver = getattr(interior_o, 'aragog_solver', None)
    phase = getattr(getattr(solver, 'state', None), 'phase_staggered', None)
    g = _usable_gravity(getattr(phase, '_g', None), r_stag.size)
    if g is not None:
        return g
    mesh = getattr(getattr(solver, 'parameters', None), 'mesh', None)
    r_eos = np.asarray(getattr(mesh, 'eos_radius', []), dtype=float).ravel()
    g_eos = np.asarray(getattr(mesh, 'eos_gravity', []), dtype=float).ravel()
    if r_eos.size > 1 and r_eos.size == g_eos.size and np.all(np.diff(r_eos) > 0.0):
        return _usable_gravity(np.interp(r_stag, r_eos, g_eos), r_stag.size)
    return None


def _drainage_fraction(
    config, hf_row: dict, prev: dict, interior_o, dm_rm: float
) -> TrappingStep | None:
    """Trapped melt fraction from the drainage integral over the front.

    Returns ``None`` when the interior state needed to locate a front is not
    available, so the caller can bury at the crystal partition coefficients alone.
    """
    tr = config.outgas
    phi_solver = np.asarray(getattr(interior_o, 'phi', None), dtype=float).ravel()
    rho = np.asarray(getattr(interior_o, 'density', None), dtype=float).ravel()
    pres = np.asarray(getattr(interior_o, 'pres', None), dtype=float).ravel()
    radius = getattr(interior_o, 'radius', None)
    if phi_solver.size == 0 or rho.size != phi_solver.size or radius is None:
        log.warning(
            'Trapping: the interior profiles needed to locate a freezing front '
            'are unavailable (phi %d, density %d, radius %s), so the step buries '
            'at the crystal partition coefficients alone. The drainage integral '
            'needs an aragog interior; no other backend supplies these.',
            phi_solver.size,
            rho.size,
            'absent' if radius is None else str(np.asarray(radius).size),
        )
        return None
    densities = _phase_densities(interior_o, pres)
    if densities is None:
        return None
    rho_s, rho_l = densities

    porosity = porosity_from_densities(rho, rho_s, rho_l, phi_solver)
    r_stag = _staggered_radii(radius, phi_solver.size)
    # The basic mesh starts at the core-mantle boundary, the base of a front
    # that is still porous at the lowest staggered node.
    r_basic = np.asarray(radius, dtype=float).ravel()
    r_floor = float(r_basic.min()) if r_basic.size == phi_solver.size + 1 else None
    rfront_loc = critical_melt_fraction(config)
    geom, branch = locate_front(
        r_stag,
        phi_solver,
        porosity,
        rho_s,
        rho_l,
        rfront_loc=rfront_loc,
        phi_min=float(tr.trap_phi_min),
        n_front_min=int(tr.trap_n_front_min),
        max_front_fraction=float(tr.trap_max_front_fraction),
        r_floor=r_floor,
    )
    if geom is None:
        return TrappingStep(
            mode='front', dm_rm=0.0, f_tl=0.0, melt_mass=0.0, branch=BRANCH_NONE
        )

    # Front speed from the mass crystallised over the step rather than the
    # reported front depth, which is snapped to the nearest node and whose
    # difference is zero on most steps and a spike on the rest. It is the same
    # increment the flux uses, so a step cannot freeze by one measure and melt
    # by another; a remelting step moves the front back at the same speed.
    dt_s = (float(hf_row['Time']) - float(prev['Time'])) * SECS_PER_YEAR
    area = 4.0 * np.pi * geom.r_base**2 * geom.rho_solid_base
    v_f = abs(float(dm_rm)) / (area * dt_s) if (area > 0.0 and dt_s > 0.0) else 0.0

    grain = float(config.interior_energetics.grain_size)
    melt_visc = 10.0 ** float(config.interior_energetics.melt_log10visc)
    mush_log10 = float(tr.trap_mush_log10visc)
    mush_visc = 10.0 ** (mush_log10 if mush_log10 > 0.0 else DEFAULT_MUSH_LOG10VISC)
    # Gravity averaged over the front nodes, like the density contrast: a front
    # near the core-mantle boundary sits well below the surface value.
    g_nodes = _node_gravity(interior_o, r_stag)
    if g_nodes is None:
        gravity = float(hf_row.get('gravity', 0.0))
        log.warning(
            'Trapping: the interior solver carries no usable per-node gravity, so '
            'the front drains under the surface gravity %.3f m/s2.',
            gravity,
        )
    else:
        gravity = float(np.mean(g_nodes[geom.index]))

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
    # than integrating a front the mesh or the step cannot resolve.
    courant = (v_f * dt_s / geom.thickness) if geom.thickness > 0.0 else float('inf')
    step.front_courant = courant
    reasons = [geom.guard_reason or 'the front is unresolved'] if branch == BRANCH_GUARD else []
    if v_f <= 0.0:
        reasons.append('the front speed is undefined on a step with no length')
    elif courant > 1.0:
        reasons.append(f'the front advanced {courant:.2f} of its thickness in one step')
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
    )
    step.f_tl = volume_to_mass_fraction(f_vol, geom.rho_melt, geom.rho_solid_base)
    step.tau_d = tau_d / SECS_PER_YEAR
    step.tau_s = tau_s / SECS_PER_YEAR
    step.t_res = t_res / SECS_PER_YEAR
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


def _record(hf_row: dict, step: TrappingStep) -> None:
    """Publish the step's diagnostics so a trajectory can be read back."""
    hf_row['trap_dM_RM'] = step.dm_rm
    hf_row['trap_F_tl'] = step.f_tl
    hf_row['trap_kg_step'] = step.total_trapped
    hf_row['trap_branch'] = float(step.branch)
    hf_row['trap_tau_D'] = step.tau_d
    hf_row['trap_tau_s'] = step.tau_s
    hf_row['trap_t_res'] = step.t_res
    hf_row['trap_n_front'] = float(step.n_front)
    hf_row['trap_L_front'] = step.l_front
    hf_row['trap_v_front'] = step.v_front
    hf_row['trap_front_courant'] = step.front_courant
    hf_row['trap_w_matrix_over_vf'] = step.w_matrix_over_vf
    hf_row['trap_n_exited'] = float(step.n_exited)
    hf_row['trap_frac_bound'] = step.frac_bound
    hf_row['trap_n_substeps'] = float(step.n_substeps)
    hf_row['trap_kg_cumulative'] = (
        float(hf_row.get('trap_kg_cumulative', 0.0)) + step.total_trapped
    )
    if step.total_trapped != 0.0:
        log.info(
            '    %s = %.3e kg  (F_tl = %.4g, dM_RM = %.3e kg)',
            'trapped ' if step.total_trapped > 0.0 else 'released',
            abs(step.total_trapped),
            step.f_tl,
            step.dm_rm,
        )
    if step.branch == BRANCH_GUARD:
        # The guard replaces the drainage integral by its no-drainage limit,
        # which can bury far more than the integral would, so every step that
        # takes it is reported with its cause rather than only counted.
        log.warning(
            'Trapping: the freezing front could not be integrated (%s). F_tl = %.3f '
            'is the no-drainage upper bound, the entry porosity capped at the '
            'critical melt fraction; %.3e kg was %s under it this step.',
            step.guard_reason or 'cause not recorded',
            step.f_tl,
            abs(step.total_trapped),
            'released' if step.total_trapped < 0.0 else 'buried',
        )


def run_trapping(
    config: Config,
    hf_row: dict,
    hf_all,
    interior_o=None,
    *,
    init_stage: bool = False,
) -> TrappingStep | None:
    """Bury volatiles into the solid mantle over this crystallisation step.

    Called once per iteration, after the interior has advanced and any structure
    re-solve has finished, and before escape and outgassing read the
    inventories. Returns ``None`` when trapping is inactive: during the
    initialisation stage, whose interior steps do not advance the mantle, on
    the first step, where there is no previous state to difference, and
    whenever the mode is ``none``.

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
    """
    mode = getattr(config.outgas, 'trap_mode', 'none')
    # A config that never declared a mode leaves trapping inactive. The schema
    # restricts the field to TRAPPING_MODES at load, so a non-string here means
    # a programmatically built config that does not configure trapping at all,
    # and running the bracket on whatever its other attributes return would
    # bury mass from parameters nobody chose.
    if not isinstance(mode, str):
        return None
    if mode not in TRAPPING_MODES:
        raise ValueError(f'Unknown trapping mode {mode!r}; expected one of {TRAPPING_MODES}')
    if mode == 'none':
        return None
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
    if dm_rm == 0.0 or melt_mass <= 0.0:
        step = TrappingStep(mode=mode, dm_rm=dm_rm, f_tl=0.0, melt_mass=melt_mass)
        _record(hf_row, step)
        return step

    step = _drainage_fraction(config, hf_row, prev, interior_o, dm_rm)
    if step is None:
        # The front cannot be located without the interior profiles. The
        # crystals still take up D_Z of each species, so the step moves mass at
        # the crystal partition coefficients alone, on a branch of its own so a
        # run cannot report it as a front that drained completely.
        step = TrappingStep(mode=mode, dm_rm=0.0, f_tl=0.0, melt_mass=0.0)
        step.branch = BRANCH_FALLBACK
    step.mode = mode
    step.dm_rm = dm_rm
    step.melt_mass = melt_mass
    step.remelted = dm_rm < 0.0

    # Crystallisation buries from the melt; remelting releases, by the same
    # flux, from what trapping holds.
    d_z = partition_coefficients(config)
    for species in vol_list:
        kg_liquid = float(hf_row.get(f'{species}_kg_liquid', 0.0))
        if kg_liquid <= 0.0:
            continue
        c_z = melt_concentration(kg_liquid, melt_mass)
        supply = kg_liquid if dm_rm > 0.0 else _trapped(hf_row, species)
        mass, capped = trapped_mass(d_z[species], step.f_tl, c_z, dm_rm, supply)
        if mass <= 0.0:
            continue
        if capped:
            step.supply_capped.append(species)
        step.trapped_kg[species] = mass if dm_rm > 0.0 else -mass

    _apply_to_reservoirs(hf_row, step.trapped_kg)
    _record(hf_row, step)
    return step
