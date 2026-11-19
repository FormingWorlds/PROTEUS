"""Aragog boundary of the open lid part and the T_magma clip guard of a lid step."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from proteus.atmos_clim.common import (
    LID_SLOPE_GUARD_SIGN,
    LID_SLOPE_GUARD_SIZE,
    lid_flux_slope,
    lid_open_conductance,
)
from proteus.interior_energetics.aragog import AragogRunner
from proteus.interior_energetics.wrapper import guard_t_magma_clip
from proteus.utils.constants import const_sigma
from tests.atmos_clim.test_common import _row

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]


def _runner_bc(lid=True):
    bc = SimpleNamespace(
        outer_boundary_condition=4,
        subgrid_lid=lid,
        lid_mush_base=lid,
        outer_boundary_value=None,
        lid_surface_flux_cap=-1.0,
        lid_surface_flux_slope=-1.0,
        open_surface_conductance=-1.0,
        equilibrium_temperature=None,
        surface_temperature=-1.0,
    )
    params = SimpleNamespace(
        solver=SimpleNamespace(start_time=None, end_time=None),
        boundary_conditions=bc,
        energy=SimpleNamespace(tidal_array=None),
    )
    interior_o = SimpleNamespace(
        aragog_solver=SimpleNamespace(parameters=params, solution=None), tides=np.zeros(3)
    )
    return bc, interior_o


@pytest.mark.parametrize(('w', 'lid'), [(0.0, True), (0.4, True), (0.4, False)])
def test_update_solver_passes_the_open_conductance_of_a_live_lid(w, lid):
    """A live lid gets ``G_eff`` of the row slope and the guard column; the held surface is the
    one ``surface_temperature`` field, and no lid surface field is written."""
    bc, interior_o = _runner_bc(lid)
    row = _row(w, G=21.9, T_b=1440.0, T_surf=727.3, F_atm=7000.0, lid_F_open=165000.0)
    row.update(Time=30000.0, T_eqm=255.0)
    hf_all = pd.DataFrame(
        {'Time': [29975.0, 30000.0], 'T_surf': [717.3, 727.3], 'F_atm': [6500.0, 7000.0]}
    )
    AragogRunner.update_solver(25.0, row, interior_o, hf_all=hf_all)
    assert not hasattr(bc, 'lid_surface_temperature')
    assert bc.surface_temperature == row['T_surf']
    if w > 0.0 and lid:
        G_eff, code = lid_open_conductance(row, lid_flux_slope(hf_all, row))
        assert bc.open_surface_conductance == G_eff
        assert G_eff > 0.0
        assert row['lid_slope_guard'] == code
        assert bc.lid_surface_flux_cap == pytest.approx(21.9 * (1440.0 - 727.3), rel=1e-14)
    else:
        assert bc.open_surface_conductance is None
        assert row['lid_slope_guard'] == 0


@pytest.mark.parametrize(('T_prev', 'slope'), [(527.3, 70.0), (927.3, 100.0)])
def test_update_solver_cuts_the_slope_at_the_grey_secant_of_the_last_2_rows(T_prev, slope):
    """The size cut of the open response uses the surface of the row before. After a warming
    step a slope between the grey secant over the 2 rows and the grey tangent at T_surf is cut
    to that secant; after a cooling step a slope between the tangent and the larger secant
    passes uncut."""
    bc, interior_o = _runner_bc()
    T_s = 727.3
    row = _row(0.4, G=21.9, T_b=1440.0, T_surf=T_s, F_atm=7000.0, lid_F_open=165000.0)
    row.update(Time=30000.0, T_eqm=255.0)
    F_prev = 7000.0 - slope * (T_s - T_prev)
    hf_all = pd.DataFrame(
        {
            'Time': [29975.0, 30000.0],
            'T_surf': [T_prev, T_s],
            'F_atm': [F_prev, 7000.0],
            'T_magma': [3000.0, 3000.0],
        }
    )
    assert lid_flux_slope(hf_all, row) == pytest.approx(slope, rel=1e-12)
    secant = const_sigma * (T_s**4 - T_prev**4) / (T_s - T_prev)
    tangent = 4.0 * const_sigma * T_s**3
    assert min(secant, tangent) < slope < max(secant, tangent)
    AragogRunner.update_solver(25.0, row, interior_o, hf_all=hf_all)
    cut = slope > secant
    assert row['lid_slope_guard'] == (LID_SLOPE_GUARD_SIZE if cut else 0)
    want = secant * (1.0 - 1e-12) if cut else slope
    assert bc.open_surface_conductance == pytest.approx(
        lid_open_conductance(row, want, T_prev)[0], rel=1e-9
    )


def test_update_solver_cuts_the_slope_with_the_end_of_its_own_secant():
    """The call site passes the slope and its other end as one pair from lid_flux_secant: a
    secant whose row before lies at 527.3 K is cut to the grey secant over 527.3 to 727.3 K,
    whatever the frame holds."""
    bc, interior_o = _runner_bc()
    T_s, T_o = 727.3, 527.3
    row = _row(0.4, G=21.9, T_b=1440.0, T_surf=T_s, F_atm=7000.0, lid_F_open=165000.0)
    row.update(Time=30000.0, T_eqm=255.0)
    secant = const_sigma * (T_s**4 - T_o**4) / (T_s - T_o)
    with patch('proteus.interior_energetics.aragog.lid_flux_secant', return_value=(70.0, T_o)):
        AragogRunner.update_solver(25.0, row, interior_o, hf_all=None)
    assert row['lid_slope_guard'] == LID_SLOPE_GUARD_SIZE
    assert bc.open_surface_conductance == pytest.approx(
        lid_open_conductance(row, secant * (1.0 - 1e-12), T_o)[0], rel=1e-9
    )


def test_a_live_lid_without_a_usable_slope_holds_the_start_value():
    """A live lid without a usable slope gets conductance 0 and guard code 1."""
    bc, interior_o = _runner_bc()
    row = _row(0.4, G=21.9, T_b=1440.0, T_surf=727.3, F_atm=7000.0, lid_F_open=165000.0)
    row.update(Time=30000.0, T_eqm=255.0)
    AragogRunner.update_solver(25.0, row, interior_o, hf_all=None)
    # the guard of a slope of 0, exact
    assert (bc.open_surface_conductance, row['lid_slope_guard']) == lid_open_conductance(
        row, 0.0
    )
    assert row['lid_slope_guard'] == LID_SLOPE_GUARD_SIGN


@pytest.mark.parametrize(
    ('prev_w', 'w', 'w_max', 'stops'),
    [
        (0.0, 0.0, 0.0, False),
        (0.0, 0.0, math.nan, False),
        (0.2, 0.0, 0.0, True),
        (0.0, 0.3, 0.3, True),
        (0.0, 0.3, 0.0, True),
        (0.0, 0.0, 0.05, True),
    ],
)
def test_a_t_magma_clip_stops_only_with_a_live_lid_in_the_step(prev_w, w, w_max, stops):
    """A clip in a step where a lid acts (weight above 0 at the start, the end or an accepted
    state) stops the run with the clipped value; elsewhere it is counted."""
    row = {'Time': 9.7e6, 'T_magma': 1600.0, 'lid_weight': w, 'lid_weight_max': w_max}
    with patch('proteus.interior_energetics.wrapper.UpdateStatusfile') as status:
        if stops:
            with pytest.raises(RuntimeError, match=r'T_magma clip from 4709\.61 K'):
                guard_t_magma_clip({}, row, prev_w, 4709.61)
            assert [c.args[1] for c in status.call_args_list] == [21]
        else:
            guard_t_magma_clip({}, row, prev_w, 4709.61)
            assert row['T_magma_clipped'] == 1
            status.assert_not_called()


def test_a_row_without_lid_columns_counts_the_clip():
    """A row without lid columns counts a clip and does not stop."""
    row = {'Time': 1.0, 'T_magma': 3000.0}
    guard_t_magma_clip({}, row, float('nan'), 3100.0)
    assert row['T_magma_clipped'] == 1


def test_an_accepted_state_out_of_the_tables_ends_the_run():
    """The Aragog error for an accepted state outside the EOS tables is passed up with
    status 21; it is not absorbed as a failed step that keeps the previous state."""
    from aragog.solver.entropy_solver import AcceptedStateRangeError

    from proteus.interior_energetics.wrapper import run_interior
    from tests.interior_energetics.test_wrapper import (
        _make_run_interior_config,
        _make_run_interior_state,
        _mock_interior_o,
    )

    config = _make_run_interior_config(prevent_warming=False, module='aragog')
    hf_all, hf_row = _make_run_interior_state()
    interior_o = _mock_interior_o()
    interior_o.ic = 2
    runner = MagicMock()
    runner.run_solver.side_effect = AcceptedStateRangeError('cell 48 entropy 1e5 at t=1 yr')
    with (
        patch('proteus.interior_energetics.aragog.AragogRunner', return_value=runner),
        patch('proteus.interior_energetics.wrapper.UpdateStatusfile') as status,
        patch('proteus.interior_energetics.wrapper.update_planet_mass'),
        patch('proteus.interior_energetics.timestep.next_step', return_value=10.0),
        pytest.raises(AcceptedStateRangeError, match='cell 48'),
    ):
        run_interior({}, config, hf_all, dict(hf_row), interior_o, MagicMock())
    assert interior_o.aragog_fail_count == 0
    assert 21 in [c.args[1] for c in status.call_args_list]


def _range_solver(fail_attempts, n_total):
    """A scripted solver whose solve() raises AcceptedStateRangeError on the given attempts."""
    from aragog.solver.entropy_solver import AcceptedStateRangeError

    from tests.interior_energetics.test_aragog import _retry_out, _retry_solver

    solver, calls = _retry_solver([_retry_out(0)] * n_total)
    record = solver.solve.side_effect

    def solve():
        record()
        if len(calls) in fail_attempts:
            raise AcceptedStateRangeError(f'cell 48 out of the tables at attempt {len(calls)}')

    solver.solve.side_effect = solve
    return solver, calls


def test_a_state_out_of_the_tables_retries_with_half_the_step(monkeypatch):
    """An attempt whose accepted state leaves the EOS tables is a failed attempt: the ladder
    halves the step, and a later attempt inside the tables is the result."""
    from tests.interior_energetics.test_aragog import _retry_runner

    solver, calls = _range_solver({1, 2}, 4)
    solver._S0, solver._last_compression_J = np.array([1.0, 2.0]), 3.5
    starts = []
    fail = solver.solve.side_effect

    def solve_from_a_state():
        # each attempt records its start, then moves the state as a failed solve does
        starts.append((solver._S0.copy(), solver._last_compression_J))
        solver._S0, solver._last_compression_J = np.array([9.0, 9.0]), -1.0
        fail()

    solver.solve.side_effect = solve_from_a_state
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch)
    out = runner._solve_with_retry(hf_row, interior_o)
    assert out.status == 0
    assert [round(c['dt'], 6) for c in calls] == [100.0, 50.0, 25.0]
    assert solver.reset.call_count == 2
    for S0, compression in starts:
        np.testing.assert_array_equal(S0, [1.0, 2.0])
        assert compression == starts[0][1]


@pytest.mark.parametrize(
    'first', [dict(cvode_flag=-1, cvode_flag_name='TOO_MUCH_WORK'), dict(cvode_flag=-4)]
)
def test_a_retry_after_a_state_out_of_the_tables_runs_at_the_base_controls(monkeypatch, first):
    """After a stiff or another failed attempt, the attempt that follows a state out of the
    tables runs at half the step with the base rtol, step budget and atol scale."""
    from tests.interior_energetics.test_aragog import (
        _RETRY_BASE_MAX_STEPS,
        _RETRY_BASE_RTOL,
        _retry_out,
        _retry_runner,
        _retry_solver,
    )

    solver, calls = _retry_solver([_retry_out(-1, **first), _retry_out(0)])
    from aragog.solver.entropy_solver import AcceptedStateRangeError

    record = solver.solve.side_effect

    def solve():
        record()
        if len(calls) == 2:
            raise AcceptedStateRangeError('cell 48 out of the tables at attempt 2')

    solver.solve.side_effect = solve
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch)
    out = runner._solve_with_retry(hf_row, interior_o)
    assert out.status == 0
    assert len(calls) == 3
    assert (calls[1]['rtol'], calls[1]['max_steps'], calls[1]['atol_sf']) != (
        _RETRY_BASE_RTOL,
        _RETRY_BASE_MAX_STEPS,
        1.0,
    )
    assert calls[2]['dt'] < calls[1]['dt']
    assert calls[2]['rtol'] == _RETRY_BASE_RTOL
    assert calls[2]['max_steps'] == _RETRY_BASE_MAX_STEPS
    assert calls[2]['atol_sf'] == calls[0]['atol_sf']


def test_a_state_out_of_the_tables_on_every_attempt_stops_the_run(monkeypatch):
    """When the last attempt still leaves the tables the error reaches the caller, which stops
    the run with status 21; no attempt outside the tables is accepted."""
    from aragog.solver.entropy_solver import AcceptedStateRangeError

    from tests.interior_energetics.test_aragog import _retry_runner

    solver, calls = _range_solver(set(range(1, 20)), 10)
    runner, interior_o, hf_row = _retry_runner(solver, monkeypatch)
    with pytest.raises(AcceptedStateRangeError, match='attempt 6'):
        runner._solve_with_retry(hf_row, interior_o)
    assert [round(c['dt'], 6) for c in calls] == [100.0, 50.0, 25.0, 12.5, 6.25, 3.125]
    solver.get_state.assert_not_called()


@pytest.mark.parametrize(
    ('prev_w', 'w_end', 'T_magma', 'stops', 'clipped'),
    [
        (0.3, 0.0, 3500.0, True, None),
        (0.0, 0.3, 3500.0, True, None),
        (0.0, 0.0, 3500.0, False, 1),
        (0.3, 0.0, 3005.0, False, 0),
    ],
)
def test_run_interior_guards_the_t_magma_clip_of_a_lid_step(
    prev_w, w_end, T_magma, stops, clipped
):
    """At its call site in run_interior: a clip where the previous row or the end of the step
    has a live lid stops the run with status 21; without a lid the clip is counted; a step
    without a clip counts 0."""
    from proteus.interior_energetics.common import Interior_t
    from proteus.interior_energetics.wrapper import run_interior
    from tests.interior_energetics.test_wrapper import (
        _make_run_interior_config,
        _make_run_interior_state,
    )

    config = _make_run_interior_config(prevent_warming=False)
    hf_all, hf_row = _make_run_interior_state(prev_f_int=1.0)
    hf_all['lid_weight'] = prev_w
    hf_row.update(lid_weight=w_end, lid_weight_max=0.0)
    out = {
        'T_magma': T_magma,
        'T_surf': 2805.0,
        'Phi_global': 0.7,
        'F_int': 2.0,
        'M_mantle': 4.0e24,
        'M_mantle_liquid': 1.0e24,
        'M_mantle_solid': 3.0e24,
        'M_core': 2.0e24,
    }
    interior_o = Interior_t(nlev_b=10)
    interior_o.ic = 2
    with (
        patch('proteus.interior_energetics.dummy.run_dummy_int', return_value=(110.0, out)),
        patch('proteus.interior_energetics.wrapper.update_planet_mass'),
        patch('proteus.interior_energetics.wrapper.UpdateStatusfile') as status,
    ):
        if stops:
            with pytest.raises(RuntimeError, match=r'T_magma clip from 3500\.00 K'):
                run_interior({}, config, hf_all, hf_row, interior_o, MagicMock(), verbose=False)
            assert [c.args[1] for c in status.call_args_list] == [21]
        else:
            run_interior({}, config, hf_all, hf_row, interior_o, MagicMock(), verbose=False)
            assert hf_row['T_magma_clipped'] == clipped
            status.assert_not_called()


@pytest.mark.parametrize(('obc', 'mush'), [(4, True), (4, False), (1, True), (1, False)])
def test_only_a_flux_lid_with_the_mush_base_gets_an_open_response(obc, mush):
    """The open response needs the flux boundary (outer condition 4) and the mush base; any
    other live lid gets no conductance and guard code 0."""
    bc, interior_o = _runner_bc(True)
    bc.outer_boundary_condition, bc.lid_mush_base = obc, mush
    row = _row(0.4, G=21.9, T_b=1440.0, T_surf=727.3, F_atm=7000.0, lid_F_open=165000.0)
    row.update(Time=30000.0, T_eqm=255.0)
    hf_all = pd.DataFrame(
        {'Time': [29975.0, 30000.0], 'T_surf': [717.3, 727.3], 'F_atm': [6500.0, 7000.0]}
    )
    AragogRunner.update_solver(25.0, row, interior_o, hf_all=hf_all)
    if obc == 4 and mush:
        assert bc.open_surface_conductance > 0.0
    else:
        assert bc.open_surface_conductance is None
        assert row['lid_slope_guard'] == 0
