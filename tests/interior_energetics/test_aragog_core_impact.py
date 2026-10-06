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

    def dtcmb_dt(self, t_cmb, q_cmb, q_sources=0.0):
        return (q_sources - q_cmb) / self.slope


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


@pytest.mark.physics_invariant
def test_two_impacts_in_one_step_book_the_combined_change():
    """A second impact before the next solve books against the first refit, so the two
    bookings sum to E_2(T_2) - E_0(T_pre). Booking the second against the pre-step budget
    would add E_1(T_1) - E_0(T_1), about 7.2e29 J (21 %), a second time."""
    _, old, solver, hf_row, interior_o = _setup(6089.0)
    new1 = _Budget(1.8e27, -3.0e29, POST['M_core'], POST['P_center'])
    new2 = _Budget(2.0e27, -9.0e29, POST['M_core'], POST['P_center'])
    fit = SimpleNamespace(rho_cen=12345.0, length_scale=7.1e6)
    with (
        patch('aragog.core.fit_gaussian_core_profiles', return_value=fit),
        patch('aragog.core.build_core_module_budget', side_effect=[new1, new2]),
    ):
        t1 = remelt_core_module(hf_row, interior_o, solver, 5091.0, 3100.0)
        solver.entropy_eos = SimpleNamespace(temperature=lambda p, s: np.array([6300.0]))
        t2 = remelt_core_module(hf_row, interior_o, solver, t1, 3100.0)
    combined = new2.heat_content(t2) - old.heat_content(5091.0)
    assert (t1, t2) == (pytest.approx(6089.0), pytest.approx(6300.0))
    assert hf_row['step_dE_impact_core_J'] == pytest.approx(combined, rel=1e-12)
    assert hf_row['step_dE_impact_J'] == pytest.approx(5.0e28 + combined, rel=1e-12)
    assert solver._core_module_budget is new2
    assert solver._core_module_budget_dtcmb_dt(5000.0, 1.0e13) == pytest.approx(
        -1.0e13 / 2.0e27
    )
    assert abs((new1.heat_content(t1) - old.heat_content(t1)) / combined) > 0.05


@pytest.mark.physics_invariant
@pytest.mark.parametrize('prior', [7.0e27, float('nan')], ids=['finite', 'nan'])
def test_core_column_accumulates_and_a_nan_prior_counts_as_zero(prior):
    """The core column adds to its prior value; a NaN prior (a row the diagnostics write
    skipped) starts from zero instead of turning the column NaN."""
    params, old, solver, hf_row, interior_o = _setup(6089.0)
    hf_row['step_dE_impact_core_J'] = prior
    new = _Budget(1.8e27, -3.0e29, POST['M_core'], POST['P_center'])
    fit = SimpleNamespace(rho_cen=12345.0, length_scale=7.1e6)
    with (
        patch('aragog.core.fit_gaussian_core_profiles', return_value=fit),
        patch('aragog.core.build_core_module_budget', return_value=new),
    ):
        remelt_core_module(hf_row, interior_o, solver, 5091.0, 3100.0)
    dE = new.heat_content(6089.0) - old.heat_content(5091.0)
    base = prior if np.isfinite(prior) else 0.0
    assert hf_row['step_dE_impact_core_J'] == pytest.approx(base + dE, rel=1e-12)
    assert hf_row['step_dE_impact_J'] == pytest.approx(5.0e28 + dE, rel=1e-12)


@pytest.mark.physics_invariant
def test_basal_temperature_comes_from_the_bottom_entropy_at_the_cmb_pressure():
    """T_basal is the EOS temperature of the re-melted bottom entropy at P_cmb, not at
    P_center or another pressure."""
    _, _, solver, hf_row, interior_o = _setup(6089.0)
    seen = []
    solver.entropy_eos = SimpleNamespace(
        temperature=lambda p, s: seen.append((p, s)) or np.array([6089.0])
    )
    fit = SimpleNamespace(rho_cen=12345.0, length_scale=7.1e6)
    new = _Budget(1.8e27, -3.0e29, POST['M_core'], POST['P_center'])
    with (
        patch('aragog.core.fit_gaussian_core_profiles', return_value=fit),
        patch('aragog.core.build_core_module_budget', return_value=new),
    ):
        t_new = remelt_core_module(hf_row, interior_o, solver, 5091.0, 3123.4)
    assert seen == [(pytest.approx(POST['P_cmb']), pytest.approx(3123.4))]
    assert t_new == pytest.approx(6089.0)


@pytest.mark.parametrize('failure', ['fit', 'budget', 'basal_nan', 'core_nan', 'heat_nan'])
def test_a_failed_refit_or_basal_lookup_changes_nothing(failure):
    """A fit or budget error, or a non-finite core temperature, basal temperature or booked
    heat, raises before any state is touched: the profile params, the frozen attributes,
    the solver budget and the booked columns stay as they were."""
    params, old, solver, hf_row, interior_o = _setup(
        float('nan') if failure == 'basal_nan' else 6089.0
    )
    before = (dict(params), dict(hf_row), vars(interior_o).copy())
    fit = SimpleNamespace(rho_cen=12345.0, length_scale=7.1e6)
    slope = float('nan') if failure == 'heat_nan' else 1.8e27
    new = _Budget(slope, -3.0e29, POST['M_core'], POST['P_center'])
    fit_kw = (
        {'side_effect': ValueError('p_cen below the incompressible limit')}
        if failure == 'fit'
        else {'return_value': fit}
    )
    build_kw = (
        {'side_effect': ValueError('r_peak < r_cmb')}
        if failure == 'budget'
        else {'return_value': new}
    )
    with (
        patch('aragog.core.fit_gaussian_core_profiles', **fit_kw),
        patch('aragog.core.build_core_module_budget', **build_kw),
        pytest.raises(ValueError),
    ):
        t_pre = float('nan') if failure == 'core_nan' else 5091.0
        remelt_core_module(hf_row, interior_o, solver, t_pre, 3100.0)
    assert (dict(params), dict(hf_row), vars(interior_o)) == before
    assert solver._core_module_budget is old


@pytest.mark.physics_invariant
def test_a_nan_mantle_booking_stays_visible_in_the_total():
    """Only the core column treats a non-finite prior as zero; a NaN mantle booking in
    step_dE_impact_J stays NaN instead of being replaced by the core heat."""
    _, _, solver, hf_row, interior_o = _setup(6089.0)
    hf_row['step_dE_impact_J'] = float('nan')
    new = _Budget(1.8e27, -3.0e29, POST['M_core'], POST['P_center'])
    fit = SimpleNamespace(rho_cen=12345.0, length_scale=7.1e6)
    with (
        patch('aragog.core.fit_gaussian_core_profiles', return_value=fit),
        patch('aragog.core.build_core_module_budget', return_value=new),
    ):
        remelt_core_module(hf_row, interior_o, solver, 5091.0, 3100.0)
    assert np.isnan(hf_row['step_dE_impact_J'])
    assert np.isfinite(hf_row['step_dE_impact_core_J'])
