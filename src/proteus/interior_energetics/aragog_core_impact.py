"""Aragog ``core_module`` core state across a giant-impact re-melt."""

from __future__ import annotations

import logging

import numpy as np

from proteus.interior_energetics.aragog_core import (
    fitted_core_budget,
    install_core_profile,
    profile_key,
    start_core_temperature,
)
from proteus.utils.constants import secs_per_year

log = logging.getLogger('fwl.' + __name__)


def remelt_core_module(
    hf_row: dict, interior_o, solver, t_core_pre: float, s_bottom: float
) -> float:
    """Lift T_core on an impact re-melt and leave the core refit for the next reset.

    The core temperature becomes ``max(t_core_pre, T_basal)``, with ``T_basal`` the
    re-melted bottom cell's entropy ``s_bottom`` evaluated at ``P_cmb``. The refit
    of the core profile and the heat booking wait for ``refit_core_at_reset``,
    which runs on the grown planet's final mesh. Several impacts before that reset
    keep the first pre-impact temperature and budget and the last lifted
    temperature, so the refit books the whole chain once.

    Parameters
    ----------
    hf_row : dict
        Helpfile row after the impact's structure solve; read only.
    interior_o : Interior_t
        Interior state; it carries the pending refit.
    solver : EntropySolver
        The Aragog solver, holding the pre-impact budget.
    t_core_pre : float
        Core temperature at the end of the landing step [K].
    s_bottom : float
        Re-melted entropy of the bottom mantle cell [J kg-1 K-1].

    Returns
    -------
    float
        Core temperature after the impact [K].

    Raises
    ------
    ValueError
        When ``t_core_pre`` or ``T_basal`` is not finite.
    """
    p_cmb = float(hf_row['P_cmb'])
    t_basal = float(np.asarray(solver.entropy_eos.temperature(p_cmb, s_bottom)).flat[0])
    if not (np.isfinite(t_basal) and np.isfinite(t_core_pre)):
        raise ValueError(f'core or basal temperature is not finite: {t_core_pre}, {t_basal}')
    t_core = max(float(t_core_pre), t_basal)
    pending = getattr(interior_o, '_core_refit_pending', None)
    if pending is None:
        pending = {'t_pre': float(t_core_pre), 'budget': solver._core_module_budget}
        interior_o._core_refit_pending = pending
    pending['t_new'] = t_core
    log.info(
        '    core temperature %.1f K -> %.1f K (basal mantle %.1f K); refit at the next solve',
        t_core_pre,
        t_core,
        t_basal,
    )
    return t_core


def refit_core_at_reset(hf_row: dict, interior_o, solver) -> None:
    """Refit the core profile to the grown core and book the impact's core heat.

    Runs after the solver reset that follows an impact, so the Gaussian profile is
    fit to the row's ``M_core`` and ``P_center`` with the solver's own CMB radius and
    pressure, the geometry the next solve integrates on. The heat of the lift under
    the refitted profile, ``E_new(T_new) - E_new(T_pre)``, is booked in
    ``step_dE_impact_core_J``, and the refit's content difference at ``T_pre``,
    ``E_new - E_old`` (the added iron's heat content from the profile's
    reference), in ``step_dE_impact_core_refit_J``. Both land on the first kept row after
    the impact, through ``interior_o._core_impact_booked``. A failing fit, budget build
    or heat check raises before anything changes; the final profile check raises
    after the solver budget is rebuilt, and the run then stops.

    Parameters
    ----------
    hf_row : dict
        Helpfile row of the step being solved; read only.
    interior_o : Interior_t
        Interior state; its pending refit is consumed and its frozen profile replaced.
    solver : EntropySolver
        The Aragog solver, already reset onto the grown planet's mesh.

    Raises
    ------
    ValueError
        From the profile fit or the budget factory, when a core heat change is not
        finite, or when the rebuilt solver budget does not carry the refitted profile.
    """
    pending = getattr(interior_o, '_core_refit_pending', None)
    if pending is None:
        return
    m_core, p_cen = float(hf_row['M_core']), float(hf_row['P_center'])
    r_cmb, p_cmb = float(solver._r_basic_flat[0]), float(solver._P_basic_flat[0])
    new, refit = fitted_core_budget(solver, m_core, p_cen)
    t_pre, t_new = pending['t_pre'], pending['t_new']
    e_pre = new.heat_content(t_pre)
    dE = new.heat_content(t_new) - e_pre
    dE_refit = e_pre - pending['budget'].heat_content(t_pre)
    if not (np.isfinite(dE) and np.isfinite(dE_refit)):
        raise ValueError(f'core heat change is not finite: lift {dE}, refit {dE_refit}')

    install_core_profile(interior_o, solver, refit, m_core, p_cen)
    interior_o._core_refit_pending = None
    # Added, not set: a skipped call keeps the booking of an impact before this one.
    lift, refit_heat = getattr(interior_o, '_core_impact_booked', None) or (0.0, 0.0)
    interior_o._core_impact_booked = (lift + dE, refit_heat + dE_refit)

    log.info(
        '    core refit: rho_cen %.2f kg/m^3, length_scale %.1f km at R_core %.6e m, '
        'P_cmb %.6e Pa; profile M_core %.6e kg (structure %.6e), P_center %.6e Pa '
        '(structure %.6e)',
        refit['rho_cen'],
        refit['length_scale'] / 1e3,
        r_cmb,
        p_cmb,
        float(new.profiles.enclosed_mass(r_cmb)),
        m_core,
        float(new.profiles.pressure(0.0)),
        p_cen,
    )
    log.info(
        '    core impact heat: lift %.6e J booked (%.2f K -> %.2f K), refit content '
        'change %.6e J booked',
        dE,
        t_pre,
        t_new,
        dE_refit,
    )


def core_call_heat(out, interior_o, lift: float = 0.0) -> float:
    """Heat change of the core over one solver call, for the core ledger [J].

    ``out.step_dE_core_J`` (the effective capacity integrated over the call's T_core
    trajectory) net of the core's internal source over the call, plus the jump of the
    heat content between the end of the last kept call and the start of this one,
    each with the budget of its own time. An impact or a structure update appears
    here as that jump, so the ledger closes only when the same heat is booked;
    ``lift``, the heat booked on this call, is logged against the jump at DEBUG. The
    end of the last kept call is ``interior_o._core_t_end`` and ``_core_budget_end``,
    which ``aragog_core.commit_core_call`` moves.
    """
    solver = interior_o.aragog_solver
    t_start = start_core_temperature(solver, 'core_module')
    t_prev, budget = getattr(interior_o, '_core_t_end', None), solver._core_module_budget
    budget_prev = getattr(interior_o, '_core_budget_end', budget)
    moved = budget_prev is not budget and profile_key(budget_prev) != profile_key(budget)
    jump = 0.0
    if t_prev is not None and (moved or abs(t_start - t_prev) > 1e-9):
        jump = budget.heat_content(t_start) - budget_prev.heat_content(t_prev)
    if jump or lift:
        log.debug(
            '    core jump %.17g J, booked lift %.17g J, difference %.17g J',
            jump,
            lift,
            jump - lift,
        )
    source = float(solver._core_module_q_radio) * float(out.dt_actual) * secs_per_year
    return float(out.step_dE_core_J) - source + jump
