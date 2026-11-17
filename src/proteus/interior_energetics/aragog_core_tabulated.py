"""Aragog ``core_module`` core on the PALEOS tabulated profile across a structure update.

The core is an isentrope of the PALEOS liquid iron table through its CMB temperature. A
structure update keeps the core mass and the total core entropy, ``M s_c - ds_fusion M_ic``
with ``s_c`` the table entropy at the CMB, and books the change of the core heat content
``T C_s(T) - R(T)``: the secular capacity of the state times its temperature, counted from 0 K
as on the Gaussian profile, less the latent and gravitational energy ``R`` released between the
onset of the inner core and ``T``. Aragog's own heat content of a tabulated budget counts from
the lowest table temperature, a constant per structure, so its difference between 2 structures
is not used here.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger('fwl.' + __name__)

MASS_RTOL = 1e-9  # core mass of the 2 structures of an update
ENTROPY_RTOL = 1e-9  # total entropy kept by the carried temperature


def core_content(budget, t_core: float) -> float:
    """Core heat content [J] at CMB temperature ``t_core`` [K], counted from 0 K.

    The Gaussian profile gives Aragog's ``heat_content``. The tabulated profile gives
    ``T C_s(T)`` less the latent and gravitational energy released from the onset to ``T``
    (0 above the onset), the same form without the reference temperature of the table.
    """
    if getattr(budget, 'tabulated', False) is not True:
        return float(budget.heat_content(t_core))
    p, t = budget.profiles, float(t_core)
    released = 0.0
    if t < budget.t_onset:
        d_t = t - budget.t_onset
        released = p.secular_heat_change(budget.t_onset, d_t) - budget.heat_change(
            budget.t_onset, d_t
        )
    return t * float(p.secular_capacity(t)) - released


def _same_table(a, b) -> bool:
    """True when 2 liquid iron tables hold the same grid and entropy."""
    if a is b:
        return True
    return all(
        np.array_equal(getattr(a, k), getattr(b, k)) for k in ('pressure', 'temperature')
    ) and np.array_equal(
        a.value('s', a.pressure[:, None], a.temperature),
        b.value('s', b.pressure[:, None], b.temperature),
    )


def _inner_mass(budget, t_core: float) -> float:
    from proteus.interior_energetics.aragog_core import _inner_core_mass

    return _inner_core_mass(budget, t_core) if t_core < budget.t_onset else 0.0


def carried_temperature(old, new, t_old: float) -> tuple[float, bool]:
    """CMB temperature [K] of ``new`` that keeps the mass and the total entropy of ``old`` at
    ``t_old`` [K], and whether an inner core takes part (``t_old`` or the result below onset).

    A liquid core keeps ``s(P_cmb, T)`` of the table; with an inner core the balance is
    ``M [s(P_cmb', T') - s(P_cmb, T)] = ds_fusion [M_ic'(T') - M_ic(T)]``.

    Raises
    ------
    ValueError
        When the 2 budgets differ in core mass or table, when no tabulated temperature keeps
        the entropy, or when the result misses the entropy by more than ``ENTROPY_RTOL``.
    """
    from scipy.optimize import brentq

    po, pn = old.profiles, new.profiles
    m = float(po.m_core)
    if not abs(float(pn.m_core) / m - 1.0) <= MASS_RTOL:
        raise ValueError(
            f'core mass changes across the update: {m:.10e} -> {pn.m_core:.10e} kg'
        )
    if not _same_table(po.table, pn.table):
        raise ValueError('the 2 structures of the update use different liquid iron tables')
    tab, t_old = pn.table, float(t_old)
    s_old, ic_old = float(tab.value('s', po.p_cmb, t_old)), _inner_mass(old, t_old)
    lo, hi = float(pn.t_grid[0]), float(pn.t_grid[-1])

    def liquid(t):
        return float(tab.value('s', pn.p_cmb, t)) - s_old

    def total(t):
        return m * liquid(t) - new.ds_fusion * (_inner_mass(new, t) - ic_old)

    if not liquid(lo) < 0.0 < liquid(hi):
        raise ValueError(
            f'no tabulated CMB temperature ({lo:.1f} to {hi:.1f} K) keeps the core entropy'
        )
    t_new = brentq(liquid, lo, hi, xtol=1e-10, rtol=1e-15)
    frozen = bool(t_old < old.t_onset or t_new < new.t_onset)
    if frozen:
        if not total(lo) < 0.0 < total(hi):
            raise ValueError('no tabulated CMB temperature keeps the core entropy with fusion')
        t_new = brentq(total, lo, hi, xtol=1e-10, rtol=1e-15)
    miss = abs(total(t_new)) / (m * abs(s_old))
    if not miss <= ENTROPY_RTOL:
        raise ValueError(f'carried core temperature misses the entropy by {miss:.2e} relative')
    return float(t_new), frozen


def carry_tabulated(interior_o, solver, old, t_old: float) -> None:
    """Carry a tabulated core to the budget the solver rebuilt for the new structure.

    The temperature keeps the mass and the total entropy (``carried_temperature``) and the
    content change is added to ``interior_o._core_struct_booked`` for
    ``step_dE_struct_core_J``.

    Raises
    ------
    ValueError
        From ``carried_temperature``, when the solver budget is not tabulated, or when the
        content change is not finite.
    """
    new, t_old = solver._core_module_budget, float(t_old)
    if getattr(new, 'tabulated', False) is not True:
        raise ValueError('a tabulated core is carried only to a tabulated budget')
    t_new, frozen = carried_temperature(old, new, t_old)
    d_e = core_content(new, t_new) - core_content(old, t_old)
    if not np.isfinite(d_e):
        raise ValueError(f'core heat change across a structure update is not finite: {d_e}')
    solver.set_initial_core_temperature(t_new)
    if getattr(interior_o, '_core_t_end', None) is None:
        interior_o._core_t_end, interior_o._core_budget_end = t_old, old
    interior_o._core_struct_booked = getattr(interior_o, '_core_struct_booked', 0.0) + d_e
    log.info(
        '    tabulated core across the structure update: T_core %.9f K -> %.9f K, heat content '
        '%+.9e J; P_cmb %.6e -> %.6e Pa; inner core %s',
        t_old,
        t_new,
        d_e,
        old.profiles.p_cmb,
        new.profiles.p_cmb,
        'in the balance' if frozen else 'absent',
    )
