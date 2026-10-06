"""Unit tests for the core_module core state across a giant-impact re-melt."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from proteus.interior_energetics.aragog_core_impact import remelt_core_module

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

PRE = dict(m_core=1.90e24, p_cen=3.60e11, r_cmb=3.40e6, p_cmb=1.30e11)
POST = dict(M_core=2.06e24, P_center=3.81e11, R_core=3.47e6, P_cmb=1.38e11)


class _Budget:
    """Stub budget: heat content linear in T with a profile-dependent slope and offset."""

    def __init__(self, slope, offset, m_core, p_cen):
        self.slope, self.offset = slope, offset
        self.profiles = SimpleNamespace(
            enclosed_mass=lambda r: m_core, pressure=lambda r: p_cen
        )

    def heat_content(self, t):
        return self.slope * t + self.offset


def _setup(t_basal):
    params = {
        'alpha': 1.35e-5,
        'c_p': 840.0,
        'rho_cen': 12000.0,
        'length_scale': 7.0e6,
        'fit_profile': False,
        'q_radio': 0.0,
        'ra_crit_cmb': 450.0,
    }
    old = _Budget(1.6e27, 2.0e29, PRE['m_core'], PRE['p_cen'])
    eos = SimpleNamespace(temperature=lambda p, s: np.array([t_basal]))
    solver = SimpleNamespace(
        parameters=SimpleNamespace(
            boundary_conditions=SimpleNamespace(core_module_params=params)
        ),
        _core_module_budget=old,
        entropy_eos=eos,
    )
    hf_row = dict(POST, step_dE_impact_J=5.0e28, step_dE_impact_core_J=0.0)
    return params, old, solver, hf_row, SimpleNamespace()


def _run(t_core_pre, t_basal):
    params, old, solver, hf_row, interior_o = _setup(t_basal)
    new = _Budget(1.8e27, -3.0e29, POST['M_core'], POST['P_center'])
    fit = SimpleNamespace(rho_cen=12345.0, length_scale=7.1e6)
    with (
        patch('aragog.core.fit_gaussian_core_profiles', return_value=fit) as fit_mock,
        patch('aragog.core.build_core_module_budget', return_value=new) as build_mock,
    ):
        t_new = remelt_core_module(hf_row, interior_o, solver, t_core_pre, 3100.0)
    return t_new, hf_row, params, interior_o, old, new, fit_mock, build_mock


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('t_pre', 't_basal', 't_expected'),
    [(5091.0, 6089.0, 6089.0), (5600.0, 5200.0, 5600.0)],
    ids=['basal_hotter', 'core_hotter'],
)
def test_core_temperature_is_lifted_never_lowered_and_heat_booked(t_pre, t_basal, t_expected):
    """T_core after the impact is max(kept, basal); the booked heat is
    E_new(T_new) - E_old(T_pre). Booking the old profile alone, or (when the core is
    lifted) the refit alone, misses by more than 1e28 J for these slopes."""
    t_new, hf_row, _, _, old, new, _, _ = _run(t_pre, t_basal)
    assert t_new == pytest.approx(t_expected, rel=1e-15)
    expected = new.heat_content(t_expected) - old.heat_content(t_pre)
    assert hf_row['step_dE_impact_core_J'] == pytest.approx(expected, rel=1e-12)
    assert hf_row['step_dE_impact_J'] == pytest.approx(5.0e28 + expected, rel=1e-12)
    clamp_only = old.heat_content(t_expected) - old.heat_content(t_pre)
    refit_only = new.heat_content(t_pre) - old.heat_content(t_pre)
    wrong = (clamp_only, refit_only) if t_expected > t_pre else (clamp_only,)
    assert min(abs(w - expected) for w in wrong) > 1.0e28


@pytest.mark.physics_invariant
def test_refit_uses_the_live_structure_and_replaces_the_frozen_profile():
    """The refit takes M_core, P_center, R_core and P_cmb from the post-impact row, the
    budget factory gets the refitted profile without the solver-only keys, and the
    frozen profile and its structure constraints move to the grown core."""
    _, _, params, interior_o, _, _, fit_mock, build_mock = _run(5091.0, 6089.0)
    fit_kwargs = fit_mock.call_args.kwargs
    assert fit_kwargs == dict(
        m_core=POST['M_core'],
        p_cen=POST['P_center'],
        r_cmb=POST['R_core'],
        p_cmb=POST['P_cmb'],
        alpha=1.35e-5,
        c_p=840.0,
    )
    budget_params = build_mock.call_args.args[0]
    assert 'q_radio' not in budget_params and 'ra_crit_cmb' not in budget_params
    assert budget_params['rho_cen'] == pytest.approx(12345.0)
    assert build_mock.call_args.kwargs == dict(
        r_cmb=POST['R_core'], p_cmb_fallback=POST['P_cmb']
    )
    assert params['length_scale'] == pytest.approx(7.1e6)
    assert interior_o._frozen_core_rho_cen == pytest.approx(12345.0)
    assert interior_o._frozen_core_m_core == pytest.approx(POST['M_core'])
    assert interior_o._frozen_core_p_cen == pytest.approx(POST['P_center'])
    assert params['ra_crit_cmb'] == pytest.approx(450.0)
