"""Unit tests for the core_module core in the PROTEUS Aragog wrapper (aragog_core.py)."""

from __future__ import annotations

import re
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_write_core_diagnostics_wiring():
    """The diagnostics writer converts the helpfile F_cmb into a CMB heat
    flow through the budget's own area (q = F * 4 pi r_cmb^2), forwards
    the configured q_radio, writes all seven core_* columns, and builds
    the wrapper-side entropy budget on the solver's own budget.

    The mocked budget returns physically plausible Earth-core values
    (r_icb ~ 1.2e6 m, C_eff ~ 1.8e27 J/K) so a units slip (missing area
    factor, W vs W/m2) shifts q_cmb by 14 orders of magnitude and fails
    the call-argument assertion rather than passing on round numbers.
    """
    from proteus.config._interior import AragogCoreModule
    from proteus.interior_energetics.aragog_core import write_core_diagnostics

    config = MagicMock()
    config.interior_energetics.aragog.core_module = AragogCoreModule(
        q_radio=2.0e12, k_core=90.0
    )
    runner = SimpleNamespace(_config=config)

    r_cmb = 3.48e6
    budget = MagicMock()
    budget.profiles.r_cmb = r_cmb
    budget.r_icb.return_value = 1.22e6
    budget.effective_capacity.return_value = 1.77e27
    solver = MagicMock()
    solver._core_module_budget = budget
    runner.aragog_solver = solver

    entropy = MagicMock()
    entropy.entropy_margin.return_value = 9.6e8
    entropy.b_rms_core.return_value = 1.1e-3
    output = {'T_cmb': 4864.0, 'F_cmb': 1.5e5}
    with (
        patch('aragog.core.CoreEntropyBudget', return_value=entropy) as mock_ent_cls,
        patch('aragog.core.crystallization_regime', return_value=1) as mock_regime,
    ):
        write_core_diagnostics(runner, output)

        # Entropy budget built once from the config's diagnostics fields.
        assert mock_ent_cls.call_args.kwargs['k_core'] == pytest.approx(90.0)

        # No elapsed time (the init call): the fallback carries the
        # area factor, q = F * 4 pi r_cmb^2.
        q_expected = 1.5e5 * 4.0 * np.pi * r_cmb**2
        args = entropy.entropy_margin.call_args
        assert args.args[0] == pytest.approx(4864.0)
        assert args.args[1] == pytest.approx(q_expected, rel=1e-12)
        assert args.kwargs['q_radio'] == pytest.approx(2.0e12)
        assert args.kwargs['t_shell'] is None
        assert budget.effective_capacity.call_args.args == (pytest.approx(4864.0),)
        assert budget.effective_capacity.call_args.kwargs == {}
        _ = mock_regime  # regime asserted through the output below

        # With elapsed time the diagnostics take the CMB power from the call's heat, not from
        # F_cmb times the area; the two inputs differ here, so a regression fails the pin.
        from proteus.utils.constants import secs_per_year

        output_dt = {'T_cmb': 4864.0, 'F_cmb': 1.5e5, 'step_dE_F_cmb_J': 3.0e28}
        write_core_diagnostics(runner, output_dt, dt_actual_yr=100.0)
        q_avg = 3.0e28 / (100.0 * secs_per_year)
        assert abs(q_avg - q_expected) > 0.1 * q_expected  # paths distinguishable
        assert entropy.entropy_margin.call_args.args[1] == pytest.approx(q_avg, rel=1e-12)

        assert output['core_r_icb'] == pytest.approx(1.22e6)
        assert output['core_C_eff'] == pytest.approx(1.77e27)
        assert output['core_dynamo_margin'] == pytest.approx(9.6e8)
        assert output['core_B_rms'] == pytest.approx(1.1e-3)
        assert output['core_regime'] == pytest.approx(1.0)
        assert output['core_strat_depth'] == pytest.approx(0.0)
        assert output['core_T_top'] == pytest.approx(4864.0)

        # A stratified core: the layer base from the solve bounds the capacity, the shell
        # reaches the entropy margin, and its top cell is the top of the core.
        t_shell = np.array([4870.0, 4910.0, 4955.0])
        strat = SimpleNamespace(core_T_shell=t_shell, core_layer_base=r_cmb - 250.0e3)
        write_core_diagnostics(runner, output, out=strat)
        assert output['core_strat_depth'] == pytest.approx(250.0e3)
        assert output['core_T_top'] == pytest.approx(4955.0)
        kwargs = budget.effective_capacity.call_args.kwargs
        assert kwargs['gravitational_upper'] == pytest.approx(r_cmb - 250.0e3)
        assert entropy.entropy_margin.call_args.kwargs['t_shell'] is t_shell

        assert mock_ent_cls.call_args.args[0] is budget

    # Guard return: a solver without the budget (mode mismatch after a
    # config edit mid-run) writes nothing rather than crashing.
    bare = MagicMock(spec=[])
    runner.aragog_solver = bare
    before = dict(output)
    write_core_diagnostics(runner, output)
    assert output == before


@pytest.mark.unit
def test_the_start_core_temperature_needs_the_slot_of_the_boundary():
    """The start state of a core_module solve holds the core temperature after the CMB
    gradient; a boundary without that slot, or a start state too short for it, gives None."""
    from proteus.interior_energetics.aragog_core import start_core_temperature

    solver = SimpleNamespace(_n_stag=3, _S0=np.array([2400.0, 2390.0, 2380.0, -1e-5, 5100.0]))
    assert start_core_temperature(solver, 'core_module') == pytest.approx(5100.0, rel=1e-15)
    assert start_core_temperature(solver, 'energy_balance') is None
    solver._S0 = solver._S0[:3]
    assert start_core_temperature(solver, 'core_module') is None


@pytest.mark.unit
def test_the_core_columns_change_the_output_only_until_the_call_is_committed():
    """The core columns of a call are computed before its snapshot is written, so the heat
    booked by an impact (1.79e30 J, refit 5.2e29 J) and by a structure update (3.1e29 J)
    and the end of the last kept call (5000 K) stay as they are until the call returned;
    the commit then uses the bookings up and moves the end to this call (4500 K, the
    budget of the solver)."""
    from proteus.interior_energetics import aragog_core

    budget = object()
    interior_o = SimpleNamespace(
        _core_impact_booked=(1.79e30, 5.2e29),
        _core_struct_booked=3.1e29,
        _core_t_end=5000.0,
        aragog_solver=SimpleNamespace(_core_module_budget=budget),
    )
    out = SimpleNamespace(dt_actual=75.0, T_core=4500.0)
    output = {}
    core = 'proteus.interior_energetics.aragog_core'
    with (
        patch(f'{core}.write_core_diagnostics'),
        patch(f'{core}_impact.core_call_heat', return_value=-2.0e28) as heat,
    ):
        aragog_core.write_core_columns(None, output, out, interior_o)
    assert heat.call_args.kwargs == {'lift': pytest.approx(1.79e30 + 5.2e29 + 3.1e29)}
    assert output == {
        'step_dE_impact_core_J': 1.79e30,
        'step_dE_impact_core_refit_J': 5.2e29,
        'step_dE_struct_core_J': 3.1e29,
        'step_dE_core_J': -2.0e28,
    }
    state = ('_core_impact_booked', '_core_struct_booked', '_core_t_end')
    assert [getattr(interior_o, k) for k in state] == [(1.79e30, 5.2e29), 3.1e29, 5000.0]
    aragog_core.commit_core_call(interior_o, out)
    assert [getattr(interior_o, k) for k in state] == [None, 0.0, 4500.0]
    assert interior_o._core_budget_end is budget


ALPHA, C_P = 1.35e-5, 840.0


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('dP', 'rho', 'ln_shape', 'weights', 'expected'),
    [
        ([1.0e10] * 4, [1.0e4] * 4, 0.0, None, 1.016201),
        ([-1.0e10] * 4, [1.0e4] * 4, 0.0, None, 1.0 / 1.016201),
        ([1.0e10, 2.0e10], [1.0e4, 1.2e4], 0.0, [1.0, 3.0], 1.024400),
        ([1.0e10] * 4, [1.0e4] * 4, np.log(1.01), None, 1.016201 / 1.01),
    ],
    ids=['uniform', 'uniform_decompression', 'two_parcels_by_hand', 'shape_change'],
)
def test_the_entropy_rule_by_hand(dP, rho, ln_shape, weights, expected):
    """The total entropy of the core is kept when the mass-weighted mean of
    c_p ln(T'/T) - alpha dP / rho is zero. A uniform density and pressure change gives the
    relation of one parcel, exp(alpha dP / (rho c_p)) = exp(0.016071) = 1.016201, and its
    inverse for a decompression. Two parcels of masses 1 and 3 with 10 and 20 GPa at 1.0e4
    and 1.2e4 kg m-3 give exp((0.016071 + 3 * 0.026786) / 4) = exp(0.024107) = 1.024400.
    A new adiabat 1 percent steeper in every parcel lowers the CMB ratio by that factor."""
    from proteus.interior_energetics.aragog_core import isentropic_ratio

    ratio = isentropic_ratio(ALPHA, C_P, dP, rho, ln_shape, weights)
    assert ratio == pytest.approx(expected, abs=1.0e-6)
    if weights is None and np.isscalar(ln_shape) and ln_shape == 0.0:
        assert ratio == pytest.approx(np.exp(ALPHA * dP[0] / (rho[0] * C_P)), rel=1e-15)


class _RealSolver(SimpleNamespace):
    """Solver stand-in that builds its budget as Aragog does, from the live params on its
    own CMB radius and pressure, and records the core temperature it is given."""

    def _cache_bc_constants(self):
        from aragog.core import build_core_module_budget

        params = dict(self.parameters.boundary_conditions.core_module_params)
        self._core_module_q_radio = float(params.pop('q_radio'))
        params.pop('ra_crit_cmb')
        self._core_module_budget = build_core_module_budget(
            params,
            r_cmb=float(self._r_basic_flat[0]),
            p_cmb_fallback=float(self._P_basic_flat[0]),
        )

    def set_initial_core_temperature(self, t_core):
        self._T_core_init = t_core


# A structure before and after 2 updates of a coupled run: CMB radius [m], CMB and central
# pressure [Pa]. The core mass stays the first one, 1.891517e24 kg.
M_CORE = 1.891517e24
GEOMETRY = [
    (3.409104e6, 1.029887e11, 3.413884e11),
    (3.411170e6, 1.114017e11, 3.607284e11),
    (3.385975e6, 1.265341e11, 3.878284e11),
]


def _core_solver():
    """A real core budget fitted to the first structure, frozen as the wrapper does."""
    import attrs

    from proteus.config._interior import AragogCoreModule

    params = attrs.asdict(AragogCoreModule())
    for key in ('f_ohm', 'flux_geometry'):
        params.pop(key)
    r_cmb, p_cmb, p_cen = GEOMETRY[0]
    params.update(m_core=M_CORE, p_cen=p_cen)
    solver = _RealSolver(
        parameters=SimpleNamespace(
            boundary_conditions=SimpleNamespace(core_module_params=params)
        ),
        _r_basic_flat=np.array([r_cmb, 3.5e6]),
        _P_basic_flat=np.array([p_cmb, 1.0e11]),
        _T_core_init=None,
    )
    solver._cache_bc_constants()
    profiles = solver._core_module_budget.profiles
    params.update(
        rho_cen=float(profiles.rho_cen),
        length_scale=float(profiles.length_scale),
        fit_profile=False,
    )
    del params['m_core'], params['p_cen']
    return solver, SimpleNamespace(_frozen_core_m_core=M_CORE, aragog_solver=solver)


def _linear_heat(budget, t_core):
    """Stand-in for the heat content [J]: linear in the temperature, one slope per budget."""
    return 1e20 * budget.profiles.r_cmb * t_core


def _update(solver, interior_o, step, t_old, heat=True, row=None):
    """Move the solver to structure ``step`` as a reset does and carry the core; without
    ``heat`` the heat contents, slow below the nucleation onset, are ``_linear_heat``."""
    from aragog.core.budget import CoreEnergyBudget

    from proteus.interior_energetics.aragog_core import carry_core_across_update

    old = solver._core_module_budget
    r_cmb, p_cmb, p_cen = GEOMETRY[step]
    solver._r_basic_flat, solver._P_basic_flat = (
        np.array([r_cmb, 3.5e6]),
        np.array([p_cmb, 1e11]),
    )
    solver._cache_bc_constants()
    row = {'P_center': p_cen, **({'M_core': 1.93e24} if row is None else row)}
    stub = patch.object(CoreEnergyBudget, 'heat_content', _linear_heat)
    with nullcontext() if heat else stub:
        carry_core_across_update(row, interior_o, solver, (old, t_old))
    return old


def _mass_mean(old, new, t_old, t_new, n=4000):
    """Mass-weighted mean of c_p ln(T'/T) - alpha dP / rho [J kg-1 K-1] between 2 profiles
    of one mass, on a radial grid of 20001 points and ``n`` equal-mass parcels."""

    def parcels(profiles):
        r = np.linspace(0.0, profiles.r_cmb, 20001)
        mass = np.asarray(profiles.enclosed_mass(r))
        return np.interp((np.arange(n) + 0.5) / n * mass[-1], mass, r)

    r_old, r_new = parcels(old), parcels(new)
    dP = np.asarray(new.pressure(r_new)) - np.asarray(old.pressure(r_old))
    rho = 0.5 * (np.asarray(old.density(r_old)) + np.asarray(new.density(r_new)))
    ratio = np.asarray(new.adiabat(r_new, t_new)) / np.asarray(old.adiabat(r_old, t_old))
    return float(np.mean(C_P * np.log(ratio) - ALPHA * dP / rho))


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_core_keeps_its_mass_and_its_entropy_across_a_structure_update(caplog):
    """A structure update that raises the CMB pressure by 8.4 GPa: the refitted profile
    holds the first core mass and the new central pressure, the core temperature rises by
    about 60 K (1.0 percent, the compression of an adiabatic core) and the total entropy
    is kept, checked by a separate quadrature on 4000 parcels. The booked heat is the
    change of the heat content between the two budgets, the solver gets the new
    temperature, the profile scalars follow, and the range of the temperature ratio over
    the mass parcels is below the limit, so no warning is logged."""
    from proteus.interior_energetics import aragog_core

    solver, interior_o = _core_solver()
    t_old = 6053.74
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.aragog_core'):
        old = _update(solver, interior_o, 1, t_old)
    new, t_new = solver._core_module_budget, solver._T_core_init

    r_cmb, _, p_cen = GEOMETRY[1]
    assert float(new.profiles.enclosed_mass(r_cmb)) == pytest.approx(M_CORE, rel=1e-9)
    assert float(new.profiles.pressure(0.0)) == pytest.approx(p_cen, rel=1e-9)
    assert t_new / t_old == pytest.approx(1.01003, abs=2e-5)
    assert abs(_mass_mean(old.profiles, new.profiles, t_old, t_new)) < 1e-6 * C_P
    # A core that kept its temperature would lose 8.4 J kg-1 K-1 of entropy.
    assert _mass_mean(old.profiles, new.profiles, t_old, t_old) < -8.0
    booked = new.heat_content(t_new) - old.heat_content(t_old)
    assert interior_o._core_struct_booked == pytest.approx(booked, rel=1e-12)
    assert 1.5e29 < booked < 1.9e29
    assert (interior_o._core_t_end, interior_o._core_budget_end) == (t_old, old)
    assert interior_o._frozen_core_rho_cen == float(new.profiles.rho_cen)
    assert interior_o._frozen_core_length_scale == float(new.profiles.length_scale)
    assert interior_o._frozen_core_p_cen == p_cen
    assert aragog_core.frozen_profile(interior_o)['r_cmb'] == r_cmb
    assert 'not the old one compressed' not in caplog.text


@pytest.mark.unit
def test_a_profile_that_is_not_the_old_one_compressed_is_reported(caplog, monkeypatch):
    """The mass parcels of the same update differ by 5.2e-3 in their own temperature
    ratio, because the refitted profile is not the old one compressed; with a limit below
    that range the update logs a warning with the number."""
    from proteus.interior_energetics import aragog_core

    monkeypatch.setattr(aragog_core, 'ENTROPY_SPREAD_MAX', 1.0e-3)
    solver, interior_o = _core_solver()
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.aragog_core'):
        _update(solver, interior_o, 1, 6053.74)
    assert re.search(r'parcels differ by 5\.[12]\de-03', caplog.text)
    assert 'limit 1.0e-03' in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize('case', ['same_cmb', 'no_budget', 'no_temperature'])
def test_a_reset_that_keeps_the_cmb_leaves_the_core_as_it_is(case):
    """A reset that left the CMB radius and pressure as they were, also when it built a new
    budget object, and a first set-up without a core state: the carry returns and changes
    nothing on the solver or on the interior state."""
    from proteus.interior_energetics.aragog_core import carry_core_across_update

    r_cmb, p_cmb, _ = GEOMETRY[0]
    old = SimpleNamespace(profiles=SimpleNamespace(r_cmb=r_cmb, p_cmb=p_cmb))
    solver = SimpleNamespace(
        _core_module_budget=object(),
        _r_basic_flat=np.array([r_cmb, 3.5e6]),
        _P_basic_flat=np.array([p_cmb, 1.0e11]),
    )
    state = dict(vars(solver))
    interior_o = SimpleNamespace()
    before = {'same_cmb': (old, 6000.0), 'no_budget': (None, 6000.0)}.get(case, (old, None))
    carry_core_across_update({}, interior_o, solver, before)
    assert vars(interior_o) == {}
    assert vars(solver) == state


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_ledger_measures_the_heat_two_structure_updates_booked():
    """Two structure updates before one kept call, as after a skipped step: the booking is
    the sum of both changes of the heat content, and the jump the next call measures from
    the end of the last kept call (its temperature and its budget) equals it, so the
    ledger closes. A core that is carried but not booked would show as that jump."""
    from proteus.interior_energetics.aragog_core import commit_core_call
    from proteus.interior_energetics.aragog_core_impact import core_call_heat

    solver, interior_o = _core_solver()
    first, t0 = solver._core_module_budget, 6053.74
    interior_o._core_t_end, interior_o._core_budget_end = t0, first
    _update(solver, interior_o, 1, t0)
    one = interior_o._core_struct_booked
    _update(solver, interior_o, 2, solver._T_core_init)
    booked, t2 = interior_o._core_struct_booked, solver._T_core_init
    last = solver._core_module_budget
    assert booked == pytest.approx(last.heat_content(t2) - first.heat_content(t0), rel=1e-12)
    assert booked > one > 0.0
    assert (interior_o._core_t_end, interior_o._core_budget_end) == (t0, first)

    solver._S0, solver._n_stag = np.array([3100.0, 0.0, t2]), 1
    out = SimpleNamespace(step_dE_core_J=0.0, T_core=t2 - 1.0, dt_actual=0.0)
    assert core_call_heat(out, interior_o) == pytest.approx(booked, rel=1e-12)
    commit_core_call(interior_o, out)
    assert interior_o._core_struct_booked == 0.0
    assert (interior_o._core_t_end, interior_o._core_budget_end) == (t2 - 1.0, last)
    solver._S0[2] = t2 - 1.0
    assert core_call_heat(out, interior_o) == 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_a_resume_carries_the_core_from_the_geometry_of_its_snapshot():
    """A run stopped after a structure update and a skipped step resumes from the snapshot
    of the last kept call: its core temperature and profile on the CMB radius and pressure
    it was written with. The solver of the resumed row is on the updated structure, so the
    core is carried from the snapshot's budget and the heat is booked again; a snapshot on
    the solver's own geometry, or one without the geometry, starts no carry."""
    from proteus.interior_energetics.aragog_core import (
        carry_core_across_update,
        snapshot_core_state,
    )

    solver, interior_o = _core_solver()
    snapshot_budget = solver._core_module_budget
    r_cmb, p_cmb, _ = GEOMETRY[0]
    assert snapshot_core_state(interior_o, solver) == (None, None)
    interior_o._frozen_core_r_cmb, interior_o._frozen_core_p_cmb = r_cmb, p_cmb
    assert snapshot_core_state(interior_o, solver) == (None, None)

    r_new, p_new, p_cen = GEOMETRY[1]
    solver._r_basic_flat, solver._P_basic_flat = (
        np.array([r_new, 3.5e6]),
        np.array([p_new, 1e11]),
    )
    solver._cache_bc_constants()
    solver._T_core_init = 6053.74
    old, t_old = snapshot_core_state(interior_o, solver)
    assert t_old == pytest.approx(6053.74, rel=1e-15)
    assert (old.profiles.r_cmb, old.profiles.p_cmb) == (r_cmb, p_cmb)
    assert old.heat_content(t_old) == pytest.approx(
        snapshot_budget.heat_content(t_old), rel=1e-12
    )
    carry_core_across_update(
        {'P_center': p_cen, 'M_core': 1.93e24}, interior_o, solver, (old, t_old)
    )
    new = solver._core_module_budget
    assert interior_o._core_struct_booked == pytest.approx(
        new.heat_content(solver._T_core_init) - snapshot_budget.heat_content(t_old), rel=1e-12
    )
    assert (interior_o._core_t_end, interior_o._core_budget_end) == (t_old, old)


def _fusion_budget(inner_core_mass):
    """Budget stand-in whose inner core mass [kg] is a function of the core temperature."""
    profiles = SimpleNamespace(enclosed_mass=lambda radius: radius, c_p=C_P)
    r_icb = lambda t_core: [inner_core_mass(t_core[0])]  # noqa: E731
    return SimpleNamespace(profiles=profiles, r_icb_batch=r_icb, ds_fusion=172.8)


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('m_ic_old', 'm_ic_new', 'expected'),
    [
        (1.0e23, 1.7e23, 1.0076077),
        (1.7e23, 1.0e23, 1.0 / 1.0076077),
        (1.0e23, 1.0e23, 1.0),
        (0.0, 1.9e24, 1.2284022),
        (1.9e24, 0.0, 1.0 / 1.2284022),
    ],
    ids=['freezing', 'melting', 'no_change', 'whole_core_freezes', 'whole_core_melts'],
)
def test_the_entropy_of_fusion_by_hand(m_ic_old, m_ic_new, expected):
    """A core of 1.9e24 kg whose inner core grows by 7e22 kg at the update gives up
    172.8 J kg-1 K-1 * 7e22 kg of entropy, which the core takes up:
    ln(T'/T_liquid) = 172.8 * 7e22 / (1.9e24 * 840) = 7.5789e-3, a ratio of 1.0076077.
    The same mass melting cools the core by the inverse ratio, and an inner core that
    does not change leaves the liquid value. A whole core that freezes or melts is the
    largest change, ln(T'/T_liquid) = 172.8 / 840 = 0.205714 (a ratio of 1.2284022), and
    the root is found inside the bracket."""
    from proteus.interior_energetics.aragog_core import _with_fusion_entropy

    old, new = _fusion_budget(lambda t: m_ic_old), _fusion_budget(lambda t: m_ic_new)
    t_new = _with_fusion_entropy(old, new, 4500.0, 4545.0, 1.9e24)
    assert t_new / 4545.0 == pytest.approx(expected, abs=1.0e-7)
    assert (t_new - 4545.0) * (m_ic_new - m_ic_old) >= 0.0  # freezing heats, melting cools


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_entropy_of_fusion_with_an_inner_core_that_follows_the_temperature():
    """An inner core whose mass falls by 2e21 kg per kelvin of the new core temperature:
    the root satisfies the balance to 1e-9 and lies between the liquid value and the value
    for an inner core fixed at the liquid temperature, because the heating melts a part of
    what the pressure froze."""
    from proteus.interior_energetics.aragog_core import _with_fusion_entropy

    grown = lambda t: 1.7e23 - 2.0e21 * (t - 4545.0)  # noqa: E731
    old, new = _fusion_budget(lambda t: 1.0e23), _fusion_budget(grown)
    t_new = _with_fusion_entropy(old, new, 4500.0, 4545.0, 1.9e24)
    scale = 172.8 / (1.9e24 * C_P)
    assert np.log(t_new / 4545.0) == pytest.approx(scale * (grown(t_new) - 1.0e23), rel=1e-9)
    assert 4545.0 < t_new < 4545.0 * 1.0076077


@pytest.mark.unit
def test_an_inner_core_that_is_not_a_number_stops_the_root_find_with_a_clear_error():
    """The bracket spans the largest change the entropy of fusion can give, the whole core
    frozen or melted; a budget that returns no number for its inner core has no root in it,
    and the error names the range."""
    from proteus.interior_energetics.aragog_core import _with_fusion_entropy

    old, new = _fusion_budget(lambda t: 1.0e23), _fusion_budget(lambda t: float('nan'))
    with pytest.raises(ValueError, match='no core temperature keeps the entropy') as error:
        _with_fusion_entropy(old, new, 4500.0, 4545.0, 1.9e24)
    # 4545 K * exp(-/+ 1.01 * 172.8 / 840): the range a whole core frozen or melted spans.
    assert 'between 3692.3 and 5594.6 K' in str(error.value)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_a_liquid_core_takes_the_rule_of_the_liquid_bit_for_bit(monkeypatch):
    """Above the nucleation onset before and after the update the new temperature is the
    old one times the ratio of the liquid rule, to the last bit, and neither the entropy
    of fusion nor the inner core mass is evaluated."""
    from proteus.interior_energetics import aragog_core

    def never(*args):
        raise AssertionError('the inner core was evaluated for a liquid core')

    monkeypatch.setattr(aragog_core, '_with_fusion_entropy', never)
    monkeypatch.setattr(aragog_core, '_inner_core_mass', never)
    solver, interior_o = _core_solver()
    t_old = 6053.74
    old = _update(solver, interior_o, 1, t_old)
    new = solver._core_module_budget
    assert t_old > old.t_onset and solver._T_core_init > new.t_onset
    q = (np.arange(256) + 0.5) / 256
    dP, rho, ln_shape = aragog_core._parcels(old.profiles, new.profiles, q)
    ratio = aragog_core.isentropic_ratio(ALPHA, C_P, dP, rho, ln_shape)
    assert solver._T_core_init == t_old * ratio


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_a_core_with_an_inner_core_keeps_its_entropy_with_the_entropy_of_fusion(caplog):
    """50 K below the nucleation onset the same update raises the melting curve more than
    the adiabat, so the inner core grows (9.8e22 to 1.3e23 kg). The entropy the frozen mass
    gives up heats the core by about 15 K more than the liquid rule, and the total balance,
    core mass times the mass-weighted mean of c_p ln(T'/T) - alpha dP / rho against
    ds_fusion times the change of the inner core mass, closes on a separate quadrature.
    The booked heat is the heat content of the new budget at that temperature less the
    one of the old budget before, and the log line holds the 2 inner core masses."""
    from proteus.interior_energetics import aragog_core

    solver, interior_o = _core_solver()
    old = solver._core_module_budget
    t_old = old.t_onset - 50.0
    with caplog.at_level('INFO', logger='fwl.proteus.interior_energetics.aragog_core'):
        _update(solver, interior_o, 1, t_old, heat=False)
    new, t_new = solver._core_module_budget, solver._T_core_init
    booked = _linear_heat(new, t_new) - _linear_heat(old, t_old)
    assert interior_o._core_struct_booked == pytest.approx(booked, rel=1e-12)

    t_liquid = t_old * 1.010031
    assert 10.0 < t_new - t_liquid < 20.0
    m_old = aragog_core._inner_core_mass(old, t_old)
    m_new = aragog_core._inner_core_mass(new, t_new)
    assert 9.0e22 < m_old < m_new < 1.5e23
    assert f'inner core mass {m_old:.4e} -> {m_new:.4e} kg' in caplog.text
    sensible = M_CORE * _mass_mean(old.profiles, new.profiles, t_old, t_new)
    assert sensible == pytest.approx(new.ds_fusion * (m_new - m_old), rel=2e-3)


@pytest.mark.unit
@pytest.mark.parametrize(
    'stratified, row, error, match',
    [(True, None, ValueError, 'stratified budget'), (False, {}, KeyError, 'M_core')],
    ids=['no_heat_content', 'row_without_the_core_mass'],
)
def test_a_carry_that_fails_leaves_the_solver_and_the_interior_as_they_were(
    stratified, row, error, match
):
    """The new state is computed on a candidate budget; when its heat content cannot be
    evaluated (a stratified budget gives none), or a value of the log line cannot be
    computed, the error leaves the profile of the solver, its core temperature and the
    bookings untouched."""
    solver, interior_o = _core_solver()
    params = solver.parameters.boundary_conditions.core_module_params
    params['stratification'] = stratified
    kept, state = dict(params), dict(vars(interior_o))
    with pytest.raises(error, match=match):
        _update(solver, interior_o, 1, 6053.74, row=row)
    assert params == kept and vars(interior_o) == state and solver._T_core_init is None


@pytest.mark.unit
def test_a_snapshot_without_the_core_mass_keeps_the_mass_of_the_profile_before():
    """A resume from a snapshot that holds no core mass carries the core with the mass of
    the profile it restored, 1.8915e24 kg, not with the core mass of the structure."""
    solver, interior_o = _core_solver()
    interior_o._frozen_core_m_core = None
    _update(solver, interior_o, 1, 6053.74)
    profiles = solver._core_module_budget.profiles
    assert float(profiles.enclosed_mass(profiles.r_cmb)) == pytest.approx(M_CORE, rel=1e-6)
    assert interior_o._frozen_core_m_core == pytest.approx(M_CORE, rel=1e-6)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_an_update_that_starts_the_inner_core_counts_its_entropy_of_fusion():
    """A liquid core 23 K above its nucleation onset: the update raises the onset by 77 K
    and the liquid rule raises the core by 46 K only, so the update starts an inner core.
    Its entropy of fusion is counted although the core was liquid before: the new
    temperature is above the liquid value and the core has an inner core."""
    from proteus.interior_energetics import aragog_core

    solver, interior_o = _core_solver()
    old, t_old = solver._core_module_budget, 4600.0
    assert t_old > old.t_onset
    _update(solver, interior_o, 1, t_old, heat=False)
    new, t_new = solver._core_module_budget, solver._T_core_init
    assert t_old * 1.010031 < new.t_onset
    assert 1.0 < t_new - t_old * 1.010031 < 4.0
    assert aragog_core._inner_core_mass(old, t_old) == 0.0
    assert aragog_core._inner_core_mass(new, t_new) > 1.0e21
    booked = _linear_heat(new, t_new) - _linear_heat(old, t_old)
    assert interior_o._core_struct_booked == pytest.approx(booked, rel=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('override', 'expected'), [(5100.0, 5100.0), (None, 4999.0)], ids=['set_start', 'solution']
)
def test_the_core_state_before_a_reset_is_what_the_next_solve_would_start_from(
    override, expected
):
    """The core temperature read before a reset is the set start when one is set (a resume,
    a re-melt, a carry of a skipped step) and else the end of the last solution, together
    with the budget of that state; a solver without a budget gives none."""
    from proteus.interior_energetics.aragog_core import core_state

    budget = object()
    solver = SimpleNamespace(
        _T_core_init=override,
        _core_module_budget=budget,
        get_current_core_temperature=lambda: 4999.0,
    )
    assert core_state(solver) == (budget, expected)
    del solver._core_module_budget
    assert core_state(solver) == (None, expected)


@pytest.mark.unit
def test_a_heat_change_that_is_not_a_number_stops_the_carry_before_the_solver_changes():
    """A heat content that is not a number raises with the value named, and the solver
    keeps its profile and its core temperature."""
    from aragog.core.budget import CoreEnergyBudget

    solver, interior_o = _core_solver()
    kept = dict(solver.parameters.boundary_conditions.core_module_params)
    nan_heat = patch.object(CoreEnergyBudget, 'heat_content', lambda self, t_core: np.nan)
    with nan_heat, pytest.raises(ValueError, match='is not finite: nan'):
        _update(solver, interior_o, 1, 6053.74)
    assert solver.parameters.boundary_conditions.core_module_params == kept
    assert solver._T_core_init is None and not hasattr(interior_o, '_core_struct_booked')


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_an_update_that_melts_the_inner_core_counts_its_entropy_of_fusion():
    """The reverse update lowers the CMB pressure by 8.4 GPa and the nucleation onset by
    77 K. A core 5 K below its onset has an inner core before it; the liquid rule alone
    would put it 26 K above the new onset, fully liquid. The entropy of fusion of the mass
    that melts is taken from the core: the new temperature is below the liquid value."""
    from proteus.interior_energetics import aragog_core

    solver, interior_o = _core_solver()
    _update(solver, interior_o, 1, 6053.74)
    old, t_old = solver._core_module_budget, solver._core_module_budget.t_onset - 5.0
    m_old, first = aragog_core._inner_core_mass(old, t_old), interior_o._core_struct_booked
    _update(solver, interior_o, 0, t_old, heat=False)
    new, t_new = solver._core_module_budget, solver._T_core_init
    booked = _linear_heat(new, t_new) - _linear_heat(old, t_old)
    assert interior_o._core_struct_booked - first == pytest.approx(booked, rel=1e-9)
    q = (np.arange(256) + 0.5) / 256
    dP, rho, ln_shape = aragog_core._parcels(old.profiles, new.profiles, q)
    t_liquid = t_old * aragog_core.isentropic_ratio(ALPHA, C_P, dP, rho, ln_shape)

    assert m_old > 1.0e21 and t_liquid > new.t_onset
    assert aragog_core._inner_core_mass(new, t_liquid) == 0.0
    assert 0.01 < t_liquid - t_new < 5.0
