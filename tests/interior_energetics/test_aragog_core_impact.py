"""Unit tests for the core_module core state across a giant-impact re-melt."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from proteus.interior_energetics.aragog_core_impact import (
    core_call_heat,
    refit_core_at_reset,
    remelt_core_module,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

STRUCT = dict(M_core=2.06e24, P_center=3.81e11, P_cmb=1.38e11)
MESH = dict(r_cmb=3.508e6, p_cmb=1.057e11)


class _Budget:
    """Stub budget: heat content linear in T with a profile-dependent slope and offset."""

    def __init__(self, slope, offset, m_core=STRUCT['M_core'], p_cen=STRUCT['P_center']):
        self.slope, self.offset = slope, offset
        self.profiles = SimpleNamespace(
            enclosed_mass=lambda r: m_core, pressure=lambda r: p_cen, rho_cen=0.0
        )

    def heat_content(self, t):
        return self.slope * t + self.offset


class _Solver(SimpleNamespace):
    """Solver stand-in: _cache_bc_constants rebuilds the budget from the params."""

    def _cache_bc_constants(self):
        self.rebuilds += 1
        params = self.parameters.boundary_conditions.core_module_params
        self._core_module_budget = SimpleNamespace(
            profiles=SimpleNamespace(rho_cen=params['rho_cen'])
        )


def _solver(t_basal=6089.0):
    params = {
        'alpha': 1.35e-5,
        'c_p': 840.0,
        'rho_cen': 12000.0,
        'length_scale': 7.0e6,
        'fit_profile': False,
        'q_radio': 0.0,
        'ra_crit_cmb': 450.0,
    }
    seen = []
    eos = SimpleNamespace(temperature=lambda p, s: seen.append((p, s)) or np.array([t_basal]))
    solver = _Solver(
        parameters=SimpleNamespace(
            boundary_conditions=SimpleNamespace(core_module_params=params)
        ),
        _core_module_budget=_Budget(1.6e27, 2.0e29),
        _r_basic_flat=np.array([MESH['r_cmb'], 3.6e6]),
        _P_basic_flat=np.array([MESH['p_cmb'], 1.0e11]),
        entropy_eos=eos,
        rebuilds=0,
    )
    return solver, params, seen


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('t_pre', 't_basal', 't_expected'),
    [(5091.0, 6089.0, 6089.0), (5600.0, 5200.0, 5600.0)],
    ids=['basal_hotter', 'core_hotter'],
)
def test_the_remelt_lifts_the_core_and_defers_the_refit(t_pre, t_basal, t_expected):
    """T_core after the impact is max(kept, basal), with T_basal the bottom entropy at
    P_cmb; the refit and the booking wait for the reset, so no column changes yet."""
    solver, _, seen = _solver(t_basal)
    hf_row = dict(STRUCT, step_dE_impact_J=5.0e28, step_dE_impact_core_J=0.0)
    interior_o = SimpleNamespace()
    t_new = remelt_core_module(hf_row, interior_o, solver, t_pre, 3123.4)
    assert t_new == pytest.approx(t_expected, rel=1e-15)
    assert seen == [(pytest.approx(STRUCT['P_cmb']), pytest.approx(3123.4))]
    pending = interior_o._core_refit_pending
    assert (pending['t_pre'], pending['t_new']) == (t_pre, pytest.approx(t_expected))
    assert pending['budget'] is solver._core_module_budget
    assert hf_row == dict(STRUCT, step_dE_impact_J=5.0e28, step_dE_impact_core_J=0.0)


def test_a_chain_of_remelts_keeps_the_first_start_and_the_last_lift():
    """Two impacts before a reset keep the first pre-impact temperature and budget and
    the last lifted temperature, so the reset books the chain once."""
    solver, _, _ = _solver(6089.0)
    interior_o = SimpleNamespace()
    t1 = remelt_core_module(dict(STRUCT), interior_o, solver, 5091.0, 3100.0)
    first_budget = solver._core_module_budget
    solver._core_module_budget = _Budget(9.9e27, 0.0)
    solver.entropy_eos = SimpleNamespace(temperature=lambda p, s: np.array([6300.0]))
    t2 = remelt_core_module(dict(STRUCT), interior_o, solver, t1, 3100.0)
    pending = interior_o._core_refit_pending
    assert (pending['t_pre'], pending['t_new']) == (5091.0, pytest.approx(t2))
    assert t2 == pytest.approx(6300.0)
    assert pending['budget'] is first_budget


@pytest.mark.parametrize('bad', ['core', 'basal'])
def test_a_non_finite_temperature_raises_before_anything_is_pending(bad):
    """A NaN core or basal temperature stops the re-melt and leaves no pending refit."""
    solver, _, _ = _solver(float('nan') if bad == 'basal' else 6089.0)
    interior_o = SimpleNamespace()
    t_pre = float('nan') if bad == 'core' else 5091.0
    with pytest.raises(ValueError, match='not finite'):
        remelt_core_module(dict(STRUCT), interior_o, solver, t_pre, 3100.0)
    assert getattr(interior_o, '_core_refit_pending', None) is None


def test_the_reset_refit_does_nothing_without_a_pending_impact():
    """A reset without a preceding impact leaves the profile, the budget and the row."""
    solver, params, _ = _solver()
    before, budget = dict(params), solver._core_module_budget
    with patch('aragog.core.fit_gaussian_core_profiles') as fit_mock:
        refit_core_at_reset(dict(STRUCT), SimpleNamespace(), solver)
    fit_mock.assert_not_called()
    assert params == before
    assert solver._core_module_budget is budget


def _reset_refit(new, fit_effect=None, build_effect=None):
    solver, params, _ = _solver()
    old = solver._core_module_budget
    pending = {'t_pre': 5091.0, 't_new': 6089.0, 'budget': old}
    interior_o = SimpleNamespace(_core_refit_pending=pending)
    fit = SimpleNamespace(rho_cen=12345.0, length_scale=7.1e6)
    fit_kw = {'side_effect': fit_effect} if fit_effect else {'return_value': fit}
    build_kw = {'side_effect': build_effect} if build_effect else {'return_value': new}
    with (
        patch('aragog.core.fit_gaussian_core_profiles', **fit_kw) as fit_mock,
        patch('aragog.core.build_core_module_budget', **build_kw) as build_mock,
    ):
        refit_core_at_reset(dict(STRUCT), interior_o, solver)
    return solver, params, interior_o, old, fit_mock, build_mock


@pytest.mark.physics_invariant
def test_the_reset_refit_uses_the_solver_mesh_and_books_only_the_lift():
    """The refit fits M_core and P_center at the solver's own CMB radius and pressure,
    books the lift under the refitted profile, E_new(T_new) - E_new(T_pre), and records
    the refit term E_new(T_pre) - E_old(T_pre) apart. Booking E_new(T_new) - E_old(T_pre)
    instead would add the refit term, 5.2e29 J here."""
    new = _Budget(1.8e27, -3.0e29)
    solver, params, interior_o, old, fit_mock, build_mock = _reset_refit(new)
    lift = new.heat_content(6089.0) - new.heat_content(5091.0)
    refit = new.heat_content(5091.0) - old.heat_content(5091.0)
    assert interior_o._core_impact_booked == (pytest.approx(lift), pytest.approx(refit))
    assert abs(refit) > 5.0e29
    assert fit_mock.call_args.kwargs == dict(
        m_core=STRUCT['M_core'],
        p_cen=STRUCT['P_center'],
        r_cmb=MESH['r_cmb'],
        p_cmb=MESH['p_cmb'],
        alpha=1.35e-5,
        c_p=840.0,
    )
    assert build_mock.call_args.kwargs == dict(
        r_cmb=MESH['r_cmb'], p_cmb_fallback=MESH['p_cmb']
    )
    assert 'q_radio' not in build_mock.call_args.args[0]
    assert (params['rho_cen'], params['length_scale']) == (12345.0, 7.1e6)
    assert solver.rebuilds == 1
    assert solver._core_module_budget.profiles.rho_cen == pytest.approx(12345.0)
    assert interior_o._frozen_core_m_core == pytest.approx(STRUCT['M_core'])
    assert interior_o._core_refit_pending is None


@pytest.mark.parametrize('failure', ['fit', 'budget', 'heat_nan'])
def test_a_failed_reset_refit_changes_nothing(failure):
    """A fit or budget error, or a non-finite heat change, raises with the params, the
    solver budget and the pending refit untouched and nothing booked."""
    new = _Budget(float('nan') if failure == 'heat_nan' else 1.8e27, -3.0e29)
    solver, params, _ = _solver()
    old = solver._core_module_budget
    pending = {'t_pre': 5091.0, 't_new': 6089.0, 'budget': old}
    interior_o = SimpleNamespace(_core_refit_pending=pending)
    before = dict(params)
    fit = SimpleNamespace(rho_cen=12345.0, length_scale=7.1e6)
    fit_kw = (
        {'side_effect': ValueError('p_cen too low')}
        if failure == 'fit'
        else {'return_value': fit}
    )
    build_kw = (
        {'side_effect': ValueError('r_peak')} if failure == 'budget' else {'return_value': new}
    )
    with (
        patch('aragog.core.fit_gaussian_core_profiles', **fit_kw),
        patch('aragog.core.build_core_module_budget', **build_kw),
        pytest.raises(ValueError),
    ):
        refit_core_at_reset(dict(STRUCT), interior_o, solver)
    assert params == before
    assert solver._core_module_budget is old
    assert interior_o._core_refit_pending is pending
    assert getattr(interior_o, '_core_impact_booked', None) is None


def test_a_rebuilt_budget_without_the_refit_profile_raises():
    """If the solver's rebuilt budget does not carry the refitted central density (the
    params the solver reads are not the ones updated), the refit stops loudly."""
    solver, params, _ = _solver()
    solver._cache_bc_constants = lambda: None
    pending = {'t_pre': 5091.0, 't_new': 6089.0, 'budget': solver._core_module_budget}
    interior_o = SimpleNamespace(_core_refit_pending=pending)
    fit = SimpleNamespace(rho_cen=12345.0, length_scale=7.1e6)
    with (
        patch('aragog.core.fit_gaussian_core_profiles', return_value=fit),
        patch('aragog.core.build_core_module_budget', return_value=_Budget(1.8e27, -3.0e29)),
        pytest.raises(ValueError, match='refitted core profile'),
    ):
        refit_core_at_reset(dict(STRUCT), interior_o, solver)
    assert params['rho_cen'] == pytest.approx(12345.0)


@pytest.mark.physics_invariant
def test_core_call_heat_adds_the_jump_since_the_last_call_and_removes_the_source(caplog):
    """The core's heat change over a call is the capacity integral net of its internal
    source, plus the heat of a T_core jump between calls measured with the solver's
    budget; a continuous T_core adds no jump, and the call's final T_core is kept."""
    budget = _Budget(2.0e27, 0.0)
    solver = SimpleNamespace(
        _S0=np.array([3000.0, -1e-5, 6124.0]),
        _core_module_budget=budget,
        _core_module_q_radio=1.0e12,
    )
    out = SimpleNamespace(step_dE_core_J=-4.0e29, T_core=6000.0, dt_actual=10.0)
    interior_o = SimpleNamespace(_core_t_end=5153.0)
    with caplog.at_level('DEBUG', logger='fwl.proteus.interior_energetics.aragog_core_impact'):
        heat = core_call_heat(out, interior_o, solver, 3.15576e7, lift=1.9e30)
    jump = 2.0e27 * (6124.0 - 5153.0)
    assert heat == pytest.approx(-4.0e29 - 1.0e12 * 10.0 * 3.15576e7 + jump, rel=1e-12)
    logged = float(re.search(r'difference (\S+) J', caplog.text).group(1))
    assert logged == pytest.approx(jump - 1.9e30, rel=1e-12)
    assert interior_o._core_t_end == pytest.approx(6000.0)
    solver._S0[-1] = 6000.0
    heat = core_call_heat(out, interior_o, solver, 3.15576e7)
    assert heat == pytest.approx(-4.0e29 - 3.15576e20, rel=1e-12)
