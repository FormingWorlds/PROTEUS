"""Aragog boundary of the open lid part and the T_magma clip guard of a lid step."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from proteus.atmos_clim.common import lid_flux_slope, lid_open_conductance
from proteus.interior_energetics.aragog import AragogRunner
from proteus.interior_energetics.wrapper import guard_t_magma_clip
from tests.atmos_clim.test_common import _row


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
    assert bc.surface_temperature == 727.3
    if w > 0.0 and lid:
        G_eff, code = lid_open_conductance(row, lid_flux_slope(hf_all, row))
        assert bc.open_surface_conductance == G_eff
        assert G_eff > 0.0
        assert row['lid_slope_guard'] == code
        assert bc.lid_surface_flux_cap == pytest.approx(21.9 * (1440.0 - 727.3), rel=1e-14)
    else:
        assert bc.open_surface_conductance is None
        assert row['lid_slope_guard'] == 0


def test_a_live_lid_without_a_usable_slope_holds_the_start_value():
    bc, interior_o = _runner_bc()
    row = _row(0.4, G=21.9, T_b=1440.0, T_surf=727.3, F_atm=7000.0, lid_F_open=165000.0)
    row.update(Time=30000.0, T_eqm=255.0)
    AragogRunner.update_solver(25.0, row, interior_o, hf_all=None)
    assert bc.open_surface_conductance == 0.0
    assert row['lid_slope_guard'] == 1


@pytest.mark.parametrize(
    ('prev_w', 'w', 'w_max', 'stops'),
    [
        (0.0, 0.0, 0.0, False),
        (0.0, 0.0, math.nan, False),
        (0.2, 0.0, 0.0, True),
        (0.0, 0.3, 0.3, True),
        (0.0, 0.0, 0.05, True),
    ],
)
def test_a_t_magma_clip_stops_only_with_a_live_lid_in_the_step(prev_w, w, w_max, stops):
    """G2: a clip in a step where a lid acts (weight above 0 at the start, the end or an accepted
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
    row = {'Time': 1.0, 'T_magma': 3000.0}
    guard_t_magma_clip({}, row, float('nan'), 3100.0)
    assert row['T_magma_clipped'] == 1


def test_an_accepted_state_out_of_the_tables_ends_the_run():
    """G1: the Aragog error for an accepted state outside the EOS tables is passed up with
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
