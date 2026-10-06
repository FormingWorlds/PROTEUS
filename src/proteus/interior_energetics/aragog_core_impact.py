"""Aragog ``core_module`` core state across a giant-impact re-melt."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger('fwl.' + __name__)

# core_module params the solver reads itself; the budget factory rejects them.
_SOLVER_ONLY_KEYS = ('q_radio', 'ra_crit_cmb')


def remelt_core_module(
    hf_row: dict, interior_o, solver, t_core_pre: float, s_bottom: float
) -> float:
    """Refit the core profile to the grown core, clamp T_core, and book the heat.

    The Gaussian profile is refit to the live ``M_core``, ``P_center``, ``R_core``
    and ``P_cmb`` of the structure solve that follows the impact. The core
    temperature becomes ``max(t_core_pre, T_basal)``, with ``T_basal`` the
    re-melted bottom cell's entropy ``s_bottom`` evaluated at ``P_cmb``. The
    booked heat is ``E_new(T_core_new) - E_old(t_core_pre)`` from
    ``CoreEnergyBudget.heat_content``: the content change of the refit at fixed
    ``t_core_pre`` (the impactor iron's own heat is not booked, as for the
    mantle) plus the clamp heat under the refitted profile. It is added to
    ``step_dE_impact_core_J`` and to ``step_dE_impact_J``.

    Parameters
    ----------
    hf_row : dict
        Helpfile row after the impact's structure solve; mutated in place.
    interior_o : Interior_t
        Interior state; its frozen core-profile attributes are replaced.
    solver : EntropySolver
        The Aragog solver, holding the pre-impact budget and the core params.
    t_core_pre : float
        Core temperature at the end of the landing step [K].
    s_bottom : float
        Re-melted entropy of the bottom mantle cell [J kg-1 K-1].

    Returns
    -------
    float
        Core temperature after the impact [K].
    """
    from aragog.core import build_core_module_budget, fit_gaussian_core_profiles

    params = solver.parameters.boundary_conditions.core_module_params
    m_core, p_cen = float(hf_row['M_core']), float(hf_row['P_center'])
    r_cmb, p_cmb = float(hf_row['R_core']), float(hf_row['P_cmb'])
    fit = fit_gaussian_core_profiles(
        m_core=m_core,
        p_cen=p_cen,
        r_cmb=r_cmb,
        p_cmb=p_cmb,
        alpha=float(params['alpha']),
        c_p=float(params['c_p']),
    )
    params['rho_cen'] = float(fit.rho_cen)
    params['length_scale'] = float(fit.length_scale)
    interior_o._frozen_core_rho_cen = params['rho_cen']
    interior_o._frozen_core_length_scale = params['length_scale']
    interior_o._frozen_core_m_core = m_core
    interior_o._frozen_core_p_cen = p_cen

    budget_params = {k: v for k, v in params.items() if k not in _SOLVER_ONLY_KEYS}
    new = build_core_module_budget(budget_params, r_cmb=r_cmb, p_cmb_fallback=p_cmb)
    old = solver._core_module_budget

    t_basal = float(np.asarray(solver.entropy_eos.temperature(p_cmb, s_bottom)).flat[0])
    t_core = max(float(t_core_pre), t_basal)
    dE = new.heat_content(t_core) - old.heat_content(float(t_core_pre))
    hf_row['step_dE_impact_core_J'] = float(hf_row.get('step_dE_impact_core_J') or 0.0) + dE
    hf_row['step_dE_impact_J'] = float(hf_row.get('step_dE_impact_J') or 0.0) + dE

    log.info(
        '    core refit: rho_cen %.2f kg/m^3, length_scale %.1f km; profile M_core %.6e kg '
        '(structure %.6e), P_center %.6e Pa (structure %.6e)',
        params['rho_cen'],
        params['length_scale'] / 1e3,
        float(new.profiles.enclosed_mass(r_cmb)),
        m_core,
        float(new.profiles.pressure(0.0)),
        p_cen,
    )
    log.info(
        '    core temperature %.1f K -> %.1f K (basal mantle %.1f K); %.3e J booked',
        t_core_pre,
        t_core,
        t_basal,
        dE,
    )
    return t_core
