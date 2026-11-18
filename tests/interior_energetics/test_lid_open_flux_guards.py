"""Aragog boundary of the open lid part and the T_magma clip guard of a lid step."""

from __future__ import annotations

import math
from types import SimpleNamespace

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
    if stops:
        with pytest.raises(RuntimeError, match=r'T_magma clip from 4709\.61 K'):
            guard_t_magma_clip(row, prev_w, 4709.61)
    else:
        guard_t_magma_clip(row, prev_w, 4709.61)
        assert row['T_magma_clipped'] == 1


def test_a_row_without_lid_columns_counts_the_clip():
    row = {'Time': 1.0, 'T_magma': 3000.0}
    guard_t_magma_clip(row, float('nan'), 3100.0)
    assert row['T_magma_clipped'] == 1
