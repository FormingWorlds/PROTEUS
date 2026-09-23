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
Sim et al. take the disaggregation melt fraction ``phi_c`` as a free parameter
(0.3). Here it is the interior solver's rheological transition,
``interior_energetics.rfront_loc``: the melt fraction at which the solver
switches the mantle from melt-like to solid-like rheology is the model's own
disaggregation point, and a separate value would let trapping and the
rheology disagree about where the crystal framework locks. The same value
clamps the published law, sets the derived ``DeltaT``, marks the top of the
drainage front, and caps the no-drainage bound.

Trapped-melt fraction
---------------------
``outgas.trap_mode`` selects how ``F_tl`` is obtained. The mass balance above is
identical in all three cases; only the source of ``F_tl`` changes.

``none``
    No trapping. The bracket is not evaluated and no mass moves. This is the
    default, so enabling trapping is an explicit choice.
``constant``
    A fixed scalar, ``outgas.trap_F_tl``. Sim et al. report a constant
    comparison case at 0.01.
``dynamic``
    Recomputed every step from the secular cooling rate, Sim et al. Eq. 7:

        F_tl = -(phi_c * tau / DeltaT) * dT/dt

    clamped to ``[0, phi_c]``. The upper bound is the disaggregation melt
    fraction, beyond which the paper states the relation breaks down; here it is
    ``interior_energetics.rfront_loc`` (see Critical melt fraction). The lower
    bound appears in neither the paper nor its published source: their magma
    ocean cools monotonically, so warming steps never arise. A PROTEUS run can
    warm, and a warming step here yields ``F_tl = 0``, which still buries the
    species at ``D_Z`` because crystal partitioning continues without any melt
    being retained.

    The prefactor is checked against the model the paper was run with,
    https://github.com/joycesim/MOE at commit e7edd1c, whose
    ``mars_module.py`` line 1241 reads
    ``self.Ftl[ii] = -self.phic * self.tau * self.dTdt[ii] / self.deltaT``.

Departure from Sim et al.: DeltaT is derived, not fixed
-------------------------------------------------------
Sim et al. fix ``DeltaT = 100 C`` for every simulation. Here it is derived from
the active melting curves instead. ``DeltaT`` is defined as the temperature
difference between the solidus and the temperature at which the melt fraction
reaches ``phi_c``. The melt fraction follows the lever rule between the melting
curves (``zalmoxis.mixing.compute_melt_fraction``), so

    DeltaT = phi_c * (T_liquidus(P) - T_solidus(P))

evaluated at a reference pressure. With the PALEOS liquidus, the default
``mushy_zone_factor = 0.8`` and ``phi_c = rfront_loc = 0.5`` this gives about
183 K at the surface, rising to roughly 550 K at 100 GPa (110 K and 330 K at the
paper's ``phi_c = 0.3``). The reference pressure is therefore a real physical
knob and is logged with the derived value. A derived ``DeltaT`` scales with
``phi_c``, so the ratio ``phi_c / DeltaT`` in the published law does not, and
only the clamp moves with ``rfront_loc``; a fixed ``trap_delta_T`` override
brings the full ``phi_c`` dependence back.

``interior_struct.zalmoxis.mantle_eos`` and ``mushy_zone_factor`` carry defaults
whatever ``interior_struct.module`` is set to, so the derivation runs for every
backend. If the curves cannot be loaded the paper's 100 K is used and the
substitution is logged. Setting ``outgas.trap_delta_T`` to a positive value
overrides the derivation.

The reference pressure is lithostatic and defaults to 1 bar. It is deliberately
not taken from ``hf_row['P_surf']``, which is the atmospheric surface pressure
and can reach hundreds of bar without the rock beneath it being any deeper.

Reservoir bookkeeping
---------------------
Trapped mass moves from ``{sp}_kg_liquid`` to ``{sp}_kg_solid``, leaving
``{sp}_kg_total`` unchanged. The same mass is moved between the per-element
reservoirs by stoichiometry, so the per-element closure
``total == atm + liquid + solid`` holds after the step.

Once ``{sp}_kg_solid`` is nonzero it must be excluded anywhere the code assumes
the whole-planet inventory is reachable. :func:`locked_solid_mass` is the single
definition of that exclusion, used by the escape step and the desiccation gate.

TODO: ``outgas.module = 'atmodeller'`` already writes condensate mass (graphite)
into the same ``_kg_solid`` columns. Trapped melt and condensed carbon are
different physics sharing one column, and this pass does not separate them.
"""

from __future__ import annotations

import logging
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
    BRANCH_PUBLISHED,
    DEFAULT_MUSH_LOG10VISC,
    drainage_integral,
    locate_front,
    porosity_from_densities,
    volume_to_mass_fraction,
)
from proteus.utils.constants import element_list, vol_list

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# How F_tl is obtained. Kept in the order the config validator reports them.
TRAPPING_MODES = ('none', 'constant', 'dynamic')

# Helpfile column differenced for dT/dt. Sim et al. define the secular cooling
# rate on the magma-ocean potential temperature; T_pot is its analogue here and
# every interior module writes it.
TEMPERATURE_COLUMN = 'T_pot'

# Lithostatic reference pressure for the DeltaT derivation [Pa]. The melting
# curve is undefined at exactly zero pressure, so the surface is taken as 1 bar.
DELTA_T_REFERENCE_PRESSURE = 1.0e5

# Fallback when the melting curves cannot be loaded [K]: the value Sim et al.
# fix for every simulation.
SIM_DELTA_T_K = 100.0

# Derived-DeltaT cache, keyed by the inputs that change it. Loading the melting
# curves reads tables, so it must not happen once per timestep.
_DELTA_T_CACHE: dict[tuple, float] = {}


@dataclass
class TrappingStep:
    """What one trapping step moved, and the guards it tripped."""

    mode: str
    dm_rm: float
    f_tl: float
    dt_dt: float
    delta_t: float
    melt_mass: float
    trapped_kg: dict[str, float] = field(default_factory=dict)
    remelted: bool = False
    clamped_high: bool = False
    clamped_low: bool = False
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

    @property
    def total_trapped(self) -> float:
        """Mass buried this step, summed over species [kg]."""
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


def raw_trapped_fraction(
    dt_dt: float | np.ndarray,
    phi_c: float,
    tau: float,
    delta_t: float,
) -> np.ndarray:
    """Trapped melt fraction before clamping, after Sim et al. Eq. 7.

    The prefactor is the one piece the offline calculator and the live coupling
    must never disagree on, so both take it from here. The clamp is applied by
    the caller: the offline tool reports the unclamped series alongside the
    clamped one so a run can be read for how far outside the valid window it
    sat, while the live step needs only the clamped value.
    """
    if not 0.0 < phi_c <= 1.0:
        raise ValueError(f'phi_c must lie in (0, 1], got {phi_c!r}')
    if tau <= 0.0:
        raise ValueError(f'tau must be positive, got {tau!r}')
    if delta_t <= 0.0:
        raise ValueError(f'DeltaT must be positive, got {delta_t!r}')
    raw = -(phi_c * tau / delta_t) * np.asarray(dt_dt, dtype=float)
    return np.where(np.isfinite(raw), raw, 0.0)


def trapped_fraction(
    dt_dt: float | np.ndarray,
    phi_c: float,
    tau: float,
    delta_t: float,
) -> tuple[float | np.ndarray, bool | np.ndarray, bool | np.ndarray]:
    """Trapped melt fraction from the cooling rate, after Sim et al. Eq. 7.

    Parameters
    ----------
    dt_dt : float or ndarray
        Secular cooling rate [K yr-1]. Negative while the mantle cools.
    phi_c : float
        Disaggregation melt fraction [1]. Also the upper clamp.
    tau : float
        Compaction time scale [yr].
    delta_t : float
        Solidus to freezing-front temperature difference [K].

    Returns
    -------
    tuple
        ``F_tl`` clamped to ``[0, phi_c]``, and whether each clamp bound.
    """
    raw = raw_trapped_fraction(dt_dt, phi_c, tau, delta_t)
    clipped = np.clip(raw, 0.0, phi_c)
    high = raw > phi_c
    low = raw < 0.0
    if np.ndim(dt_dt) == 0:
        return float(clipped), bool(high), bool(low)
    return clipped, high, low


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
    f_tl: float | np.ndarray | None,
    c_z: float | np.ndarray,
    dm_rm: float | np.ndarray,
    available_kg: float | np.ndarray,
) -> float | np.ndarray:
    """Mass of one species buried over a crystallisation increment [kg].

    ``f_tl`` of ``None`` is the no-trapping mode: the bracket is not evaluated
    and nothing moves. That is the absence of the process, not ``F_tl = 0``,
    which would still bury the species at ``D_Z``.

    The result is capped at ``available_kg``, the mass the melt actually holds,
    which the unbounded product can exceed when ``dM_RM`` is large.
    """
    if f_tl is None:
        zero = np.zeros(np.shape(np.asarray(c_z, dtype=float)), dtype=float)
        return 0.0 if np.ndim(c_z) == 0 else zero
    raw = effective_partition(f_tl, d_z) * np.asarray(c_z, dtype=float) * dm_rm
    capped = np.minimum(raw, np.maximum(np.asarray(available_kg, dtype=float), 0.0))
    capped = np.maximum(capped, 0.0)
    if np.ndim(c_z) == 0 and np.ndim(f_tl) == 0 and np.ndim(dm_rm) == 0:
        return float(capped)
    return capped


def crystallised_mass_from_phi(
    m_mantle: float, phi_prev: float, phi_now: float
) -> tuple[float, bool]:
    """Mass crystallised over one step [kg], from the melt fraction.

    ``dM_RM = M_mantle * [Phi(t-1) - Phi(t)]``. Phi_global is a mass fraction
    and therefore intensive, so a Zalmoxis re-solve that changes ``M_mantle``
    does not inject a spurious increment the way differencing
    ``M_mantle_solid`` would. It also matches the melt reservoir CALLIOPE
    dissolves into, which is sized from Phi_global rather than from
    ``M_mantle_liquid``; the prevent-warming clamp moves Phi_global without
    resynchronising those masses, so the two disagree on clamped steps.

    A negative increment means the mantle remelted, which releases trapped
    volatiles rather than burying them. The release branch is out of scope, so
    those steps are clamped to zero and reported.
    """
    if not np.isfinite(m_mantle) or m_mantle <= 0.0:
        return 0.0, False
    if not (np.isfinite(phi_prev) and np.isfinite(phi_now)):
        return 0.0, False
    dm = float(m_mantle) * (float(phi_prev) - float(phi_now))
    if dm < 0.0:
        return 0.0, True
    return dm, False


def critical_melt_fraction(config: Config) -> float:
    """Disaggregation melt fraction ``phi_c`` [1], the solver's rheological transition.

    The single definition of ``phi_c`` for the trapping step, so the published
    law's clamp, the derived ``DeltaT``, the drainage front and the no-drainage
    bound cannot drift apart from each other or from the interior rheology.
    """
    return float(config.interior_energetics.rfront_loc)


def derive_delta_T(
    config: Config,
    phi_c: float,
    pressure: float = DELTA_T_REFERENCE_PRESSURE,
) -> float:
    """Solidus to freezing-front temperature difference [K].

    ``DeltaT = phi_c * (T_liquidus - T_solidus)`` at ``pressure``, which follows
    from the lever-rule melt fraction the structure solver uses. Falls back to
    the 100 K Sim et al. fix if the melting curves cannot be loaded.
    """
    eos = config.interior_struct.zalmoxis.mantle_eos
    mzf = config.interior_struct.zalmoxis.mushy_zone_factor
    key = (eos, float(mzf), float(phi_c), float(pressure))
    if key in _DELTA_T_CACHE:
        return _DELTA_T_CACHE[key]

    delta_t = SIM_DELTA_T_K
    try:
        from proteus.interior_struct.zalmoxis import (
            load_zalmoxis_solidus_liquidus_functions,
        )

        curves = load_zalmoxis_solidus_liquidus_functions(eos, config)
        if curves is None:
            raise ValueError(f'no melting curves for mantle_eos {eos!r}')
        solidus_func, liquidus_func = curves
        t_sol = float(solidus_func(pressure))
        t_liq = float(liquidus_func(pressure))
        if not (np.isfinite(t_sol) and np.isfinite(t_liq)) or t_liq <= t_sol:
            raise ValueError(f'melting curves give T_liq={t_liq!r}, T_sol={t_sol!r}')
        delta_t = float(phi_c) * (t_liq - t_sol)
        log.info(
            'Trapping DeltaT = %.1f K, derived at %.3g Pa from %s '
            '(T_liq %.1f K, T_sol %.1f K). Sim et al. fix 100 K.',
            delta_t,
            pressure,
            eos,
            t_liq,
            t_sol,
        )
    except Exception as exc:
        log.warning(
            'Trapping DeltaT could not be derived from the melting curves (%s); '
            'using the %.1f K of Sim et al. instead.',
            exc,
            SIM_DELTA_T_K,
        )

    _DELTA_T_CACHE[key] = delta_t
    return delta_t


def resolve_delta_T(config: Config, phi_c: float) -> float:
    """DeltaT [K]: the configured override when positive, else the derivation."""
    override = float(getattr(config.outgas, 'trap_delta_T', -1.0))
    if override > 0.0:
        return override
    return derive_delta_T(config, phi_c)


def partition_coefficients(config: Config) -> dict[str, float]:
    """Crystal/melt partition coefficient per species, from config.

    Water is the only species with appreciable lattice incorporation; the rest
    default to zero and are buried through the interstitial-melt term alone.
    """
    return {s: float(getattr(config.outgas, f'D_const_{s}', 0.0)) for s in vol_list}


def locked_solid_mass(hf_row: dict, element: str) -> float:
    """Whole-planet mass of one element locked in the solid mantle [kg].

    The single definition of what trapping has put beyond reach. The escape step
    subtracts it from the mass available to escape, and the desiccation gate
    subtracts it from the inventory it tests, so a planet whose atmosphere and
    melt have both emptied still registers as desiccated.
    """
    mass = float(hf_row.get(f'{element}_kg_solid', 0.0))
    if not np.isfinite(mass) or mass <= 0.0:
        return 0.0
    return mass


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
    ``user_constant`` case: the trapped oxygen is recorded in ``O_kg_solid``
    as a diagnostic, and the per-element closure is not asserted for it.
    """
    if getattr(getattr(config, 'planet', None), 'fO2_source', '') == 'user_constant':
        return ('O',)
    return ()


def snapshot_solid_reservoirs(hf_row: dict) -> dict[str, float]:
    """Record every solid-mantle reservoir before the chemistry solve runs.

    Pair with :func:`restore_solid_reservoirs`. The solid reservoirs are owned
    by trapping, not by the chemistry: CALLIOPE has no solid phase and writes
    every ``_kg_solid`` field as a hard zero, so a trapped inventory would be
    erased on the first iteration it survived to.
    """
    from proteus.utils.constants import gas_list

    return {
        name: float(hf_row.get(f'{name}_kg_solid', 0.0))
        for name in list(gas_list) + list(element_list)
    }


def restore_solid_reservoirs(hf_row: dict, snapshot: dict[str, float]) -> None:
    """Put back the solid reservoirs the chemistry solve flattened.

    Takes the larger of the snapshot and whatever the backend produced, rather
    than overwriting. atmodeller writes real condensate mass (graphite) into
    these same columns, so the larger value keeps whichever source actually
    filled the reservoir without counting both. Trapped melt and condensed
    carbon share one column and are not yet separable; see the module TODO.
    """
    for name, before in snapshot.items():
        if before > 0.0:
            hf_row[f'{name}_kg_solid'] = max(before, float(hf_row.get(f'{name}_kg_solid', 0.0)))


def withhold_locked_totals(hf_row: dict, derived: tuple[str, ...] = ()) -> dict[str, float]:
    """Hide the solid-trapped mass from the chemistry, returning the full totals.

    Every outgassing backend partitions a whole-planet inventory between melt
    and atmosphere and has no solid reservoir of its own, so a total that still
    counted the trapped mass would have the chemistry re-dissolve volatiles the
    crystallising mantle has already buried. The per-element closure would then
    over-count by exactly the solid reservoir.

    Pair with :func:`restore_locked_totals`, which puts the full totals back
    once the solve has finished. Applying this once around the dispatch covers
    every backend rather than each one separately.

    ``derived`` names elements whose total the chemistry owns; see
    :func:`derived_total_elements`. Those are left untouched.
    """
    from proteus.utils.constants import gas_list

    saved: dict[str, float] = {}
    for name in list(gas_list) + list(element_list):
        if name in derived:
            # The chemistry overwrites this total from its own constraint, so
            # withholding and restoring it would put back a stale value that
            # cannot match what the solve just produced.
            continue
        if f'{name}_kg_total' not in hf_row:
            continue
        locked = locked_solid_mass(hf_row, name)
        if locked <= 0.0:
            continue
        total = float(hf_row[f'{name}_kg_total'])
        if not np.isfinite(total):
            continue
        saved[name] = total
        hf_row[f'{name}_kg_total'] = max(0.0, total - locked)
    return saved


def restore_locked_totals(hf_row: dict, saved: dict[str, float]) -> None:
    """Put back the whole-planet totals :func:`withhold_locked_totals` hid."""
    for name, total in saved.items():
        hf_row[f'{name}_kg_total'] = total


def _apply_to_reservoirs(hf_row: dict, trapped: dict[str, float]) -> None:
    """Move trapped mass from liquid to solid, per species and per element."""
    for species, mass in trapped.items():
        if mass <= 0.0:
            continue
        liq = float(hf_row.get(f'{species}_kg_liquid', 0.0))
        sol = float(hf_row.get(f'{species}_kg_solid', 0.0))
        hf_row[f'{species}_kg_liquid'] = max(0.0, liq - mass)
        hf_row[f'{species}_kg_solid'] = sol + mass

    for element, mass in element_masses_from_species(trapped).items():
        if element not in element_list or mass <= 0.0:
            continue
        liq = float(hf_row.get(f'{element}_kg_liquid', 0.0))
        sol = float(hf_row.get(f'{element}_kg_solid', 0.0))
        hf_row[f'{element}_kg_liquid'] = max(0.0, liq - mass)
        hf_row[f'{element}_kg_solid'] = sol + mass


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
            'step falls back to the published law.',
            type(solver).__name__ if solver is not None else None,
            type(eos).__name__ if eos is not None else None,
        )
        return None
    try:
        rho_s = np.asarray(lookup('density', pressure, 'solid'), dtype=float).ravel()
        rho_l = np.asarray(lookup('density', pressure, 'melt'), dtype=float).ravel()
    except Exception as exc:
        log.warning(
            'Phase-boundary densities unavailable (%s); trapping falls back '
            'to the published law this step.',
            exc,
        )
        return None
    if rho_s.shape != rho_l.shape or not np.all(np.isfinite(rho_s + rho_l)):
        log.warning(
            'Trapping: phase-boundary densities are inconsistent or non-finite '
            '(shapes %s and %s), so the step falls back to the published law.',
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


def _drainage_fraction(config, hf_row: dict, prev: dict, interior_o) -> TrappingStep | None:
    """Trapped melt fraction from the drainage integral over the front.

    Returns ``None`` when the interior state needed to locate a front is not
    available, so the caller can fall back to the published law.
    """
    tr = config.outgas
    phi_solver = np.asarray(getattr(interior_o, 'phi', None), dtype=float).ravel()
    rho = np.asarray(getattr(interior_o, 'density', None), dtype=float).ravel()
    pres = np.asarray(getattr(interior_o, 'pres', None), dtype=float).ravel()
    radius = getattr(interior_o, 'radius', None)
    if phi_solver.size == 0 or rho.size != phi_solver.size or radius is None:
        log.warning(
            'Trapping: the interior profiles needed to locate a freezing front '
            'are unavailable (phi %d, density %d, radius %s), so the step falls '
            'back to the published law. The drainage integral needs an aragog '
            'interior; no other backend supplies these.',
            phi_solver.size,
            rho.size,
            'absent' if radius is None else str(np.asarray(radius).size),
        )
        return None
    densities = _phase_densities(interior_o, pres)
    if densities is None:
        return None
    rho_s, rho_l = densities

    porosity = porosity_from_densities(rho, rho_s, rho_l)
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
            mode='dynamic',
            dm_rm=0.0,
            f_tl=0.0,
            dt_dt=0.0,
            delta_t=0.0,
            melt_mass=0.0,
            branch=BRANCH_NONE,
        )

    # Front speed from the crystallised mass rather than the reported front
    # depth: the solid mantle mass is an integrated, closed quantity that moves
    # smoothly, whereas the front depth is snapped to the nearest node and its
    # difference is zero on most steps and a spike on the rest.
    dt_s = (float(hf_row['Time']) - float(prev['Time'])) * SECS_PER_YEAR
    dm_solid = float(hf_row['M_mantle_solid']) - float(prev['M_mantle_solid'])
    area = 4.0 * np.pi * geom.r_base**2 * geom.rho_solid_base
    v_f = dm_solid / (area * dt_s) if (area > 0.0 and dt_s > 0.0) else 0.0

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
        mode='dynamic',
        dm_rm=0.0,
        f_tl=0.0,
        dt_dt=0.0,
        delta_t=0.0,
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
        reasons.append(f'the solid mantle did not grow (front speed {v_f:.3g} m/s)')
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


def _resolve_f_tl(
    config: Config, hf_row: dict, prev: dict, phi_c: float
) -> tuple[float, float, float, bool, bool]:
    """F_tl for this step, plus the cooling rate, DeltaT, and the clamp flags."""
    if config.outgas.trap_mode == 'constant':
        # The cooling rate and DeltaT are not consulted in this mode. Reported
        # as zero rather than NaN so the helpfile carries a clean column the
        # writer does not have to coerce.
        return float(config.outgas.trap_F_tl), 0.0, 0.0, False, False

    dt = float(hf_row.get('Time', 0.0)) - float(prev.get('Time', 0.0))
    t_now = float(hf_row.get(TEMPERATURE_COLUMN, float('nan')))
    t_prev = float(prev.get(TEMPERATURE_COLUMN, float('nan')))
    if dt > 0.0 and np.isfinite(t_now) and np.isfinite(t_prev):
        dt_dt = (t_now - t_prev) / dt
    else:
        dt_dt = 0.0

    delta_t = resolve_delta_T(config, phi_c)
    tau = float(config.outgas.trap_tau)
    f_tl, high, low = trapped_fraction(dt_dt, phi_c, tau, delta_t)
    return f_tl, dt_dt, delta_t, bool(high), bool(low)


def _record(hf_row: dict, step: TrappingStep) -> None:
    """Publish the step's diagnostics so a trajectory can be read back."""
    hf_row['trap_dM_RM'] = step.dm_rm
    hf_row['trap_F_tl'] = step.f_tl
    hf_row['trap_dT_dt'] = step.dt_dt
    hf_row['trap_delta_T'] = step.delta_t
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
    hf_row['trap_kg_cumulative'] = (
        float(hf_row.get('trap_kg_cumulative', 0.0)) + step.total_trapped
    )
    if step.total_trapped > 0.0:
        log.info(
            '    trapped    = %.3e kg  (F_tl = %.4g, dM_RM = %.3e kg)',
            step.total_trapped,
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
            'critical melt fraction; %.3e kg was buried under it this step.',
            step.guard_reason or 'cause not recorded',
            step.f_tl,
            step.total_trapped,
        )


def run_trapping(config: Config, hf_row: dict, hf_all, interior_o=None) -> TrappingStep | None:
    """Bury volatiles into the solid mantle over this crystallisation step.

    Called once per iteration, after the interior has advanced and any structure
    re-solve has finished, and before escape and outgassing read the
    inventories. Returns ``None`` when trapping is inactive: on the first step,
    where there is no previous state to difference, and whenever the mode is
    ``none``.

    Parameters
    ----------
    config : Config
        Model configuration.
    hf_row : dict
        Current helpfile row, modified in place.
    hf_all : pandas.DataFrame or None
        Completed rows. The last one supplies the previous melt fraction and
        temperature.
    interior_o : Interior_t or None
        Interior state. Required by ``trap_tau_source = 'aragog'``, which reads
        the melt-fraction, density and pressure profiles to locate the freezing
        front; the published-law paths do not use it.
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
    if hf_all is None or len(hf_all) < 1:
        return None
    if float(hf_row.get('Time', 0.0)) <= 0.0:
        return None

    prev = hf_all.iloc[-1].to_dict()
    phi_c = critical_melt_fraction(config)
    m_mantle = float(hf_row.get('M_mantle', 0.0))
    phi_now = float(hf_row.get('Phi_global', float('nan')))
    phi_prev = float(prev.get('Phi_global', float('nan')))
    dm_rm, remelted = crystallised_mass_from_phi(m_mantle, phi_prev, phi_now)

    # Dynamic mode with tau_source = 'aragog' replaces the published linear law
    # by the drainage integral over the resolved front, which needs neither tau
    # nor DeltaT. It falls back to the published law when the interior state it
    # needs is unavailable, which is every backend other than aragog.
    drained = None
    if mode == 'dynamic' and getattr(config.outgas, 'trap_tau_source', 'fixed') == 'aragog':
        drained = _drainage_fraction(config, hf_row, prev, interior_o)

    if drained is not None:
        f_tl, dt_dt, delta_t, high, low = drained.f_tl, 0.0, 0.0, False, False
    else:
        f_tl, dt_dt, delta_t, high, low = _resolve_f_tl(config, hf_row, prev, phi_c)
    # The dissolved masses in hf_row are still those of the previous chemistry
    # solve, so the melt they were dissolved into is the previous one. Dividing
    # by it recovers the concentration the solver computed.
    m_mantle_prev = float(prev.get('M_mantle', m_mantle))
    melt_mass = m_mantle_prev * max(0.0, min(1.0, phi_prev)) if np.isfinite(phi_prev) else 0.0

    step = TrappingStep(
        mode=mode,
        dm_rm=dm_rm,
        f_tl=f_tl,
        dt_dt=dt_dt,
        delta_t=delta_t,
        melt_mass=melt_mass,
        remelted=remelted,
        clamped_high=high,
        clamped_low=low,
    )
    if drained is not None:
        step.branch = drained.branch
        step.tau_d = drained.tau_d
        step.tau_s = drained.tau_s
        step.t_res = drained.t_res
        step.n_front = drained.n_front
        step.l_front = drained.l_front
        step.v_front = drained.v_front
        step.front_courant = drained.front_courant
        step.w_matrix_over_vf = drained.w_matrix_over_vf
        step.guard_reason = drained.guard_reason
    elif mode == 'dynamic' and getattr(config.outgas, 'trap_tau_source', 'fixed') == 'aragog':
        # The drainage integral was asked for and could not run. Recorded
        # distinctly from a deliberate published-law step so a run cannot
        # report a silent fallback as normal operation.
        step.branch = BRANCH_FALLBACK
    elif mode in ('constant', 'dynamic'):
        step.branch = BRANCH_PUBLISHED
    if dm_rm <= 0.0 or melt_mass <= 0.0:
        _record(hf_row, step)
        return step

    d_z = partition_coefficients(config)
    for species in vol_list:
        kg_liquid = float(hf_row.get(f'{species}_kg_liquid', 0.0))
        if kg_liquid <= 0.0:
            continue
        c_z = melt_concentration(kg_liquid, melt_mass)
        raw = float(effective_partition(f_tl, d_z[species])) * c_z * dm_rm
        mass = min(raw, kg_liquid)
        if mass <= 0.0:
            continue
        if raw > kg_liquid:
            step.supply_capped.append(species)
        step.trapped_kg[species] = mass

    _apply_to_reservoirs(hf_row, step.trapped_kg)
    _record(hf_row, step)
    return step
