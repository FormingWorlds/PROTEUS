"""Slope of the lid cap and the open response when the secant of the last 2 rows is unusable.

Under a live lid an unusable secant (a surface step below ``LID_SLOPE_DT_MIN``, a slope of 0
or below, or one that is not finite) is replaced by the newest usable secant of 2 consecutive
earlier rows, or by the grey tangent ``4 sigma T_surf^3`` where the run has none.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from proteus.atmos_clim.common import LID_SLOPE_DT_MIN
from proteus.interior_energetics.aragog import AragogRunner
from proteus.utils.constants import const_sigma
from tests.atmos_clim.test_common import _row

pytestmark = pytest.mark.unit


def _update(T, F, w=1.0):
    """Run update_solver on a history with surfaces ``T`` and fluxes ``F``, the last entry being
    the current row; return the boundary parameters and the row."""
    bc = SimpleNamespace(
        outer_boundary_condition=4,
        subgrid_lid=True,
        lid_mush_base=True,
        outer_boundary_value=None,
        lid_surface_flux_cap=None,
        equilibrium_temperature=None,
    )
    params = SimpleNamespace(
        solver=SimpleNamespace(start_time=None, end_time=None),
        boundary_conditions=bc,
        energy=SimpleNamespace(tidal_array=None),
    )
    interior_o = SimpleNamespace(
        aragog_solver=SimpleNamespace(parameters=params, solution=None), tides=np.zeros(3)
    )
    hf_all = pd.DataFrame(
        {'Time': [10.0 * (i + 1) for i in range(len(T))], 'T_surf': T, 'F_atm': F}
    )
    row = _row(w, G=21.9, T_b=1440.0, T_surf=float(T[-1]), F_atm=float(F[-1]))
    row.update(Time=float(hf_all['Time'].iloc[-1]), T_eqm=255.0, lid_F_open=float(F[-1]))
    AragogRunner.update_solver(25.0, row, interior_o, hf_all=hf_all)
    return bc, row


def _newest_usable(T, F):
    """Newest usable secant of 2 consecutive rows before the current one, or None."""
    for i in range(len(T) - 2, 0, -1):
        dT = T[i] - T[i - 1]
        if abs(dT) >= LID_SLOPE_DT_MIN:
            s = (F[i] - F[i - 1]) / dT
            if math.isfinite(s) and s > 0.0:
                return s, len(T) - 1 - i
    return None


@pytest.mark.parametrize(
    'T_last, F_last',
    [(1170.005, 14801.0), (1171.0, 14790.0)],
    ids=['short step', 'slope below 0'],
)
def test_a_live_lid_takes_the_newest_usable_secant(T_last, F_last, caplog):
    """A short step or a slope below 0 under a live lid takes the secant of the 2 rows before
    (40 W m-2 K-1, 1 row back) for the cap, with guard code 3 and the age in the log."""
    with caplog.at_level(logging.INFO):
        bc, row = _update([1150.0, 1160.0, 1170.0, T_last], [14000.0, 14400.0, 14800.0, F_last])
    assert bc.lid_surface_flux_slope == pytest.approx(40.0, rel=1e-12)
    assert row['lid_slope_guard'] == 3
    assert any('1 row' in r.getMessage() and 'secant' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    'T, F',
    [([1170.0], [14800.0]), ([1170.0, 1170.0, 1170.001], [14800.0, 14700.0, 14600.0])],
    ids=['first row', 'no usable secant'],
)
def test_a_live_lid_without_a_usable_secant_takes_the_grey_tangent(T, F):
    """Without any usable secant in the run the cap takes 4 sigma T_surf^3, guard code 4."""
    bc, row = _update(T, F)
    assert bc.lid_surface_flux_slope == pytest.approx(4.0 * const_sigma * T[-1] ** 3, rel=1e-12)
    assert row['lid_slope_guard'] == 4


def test_a_usable_secant_is_kept():
    """A usable secant of the last 2 rows is passed unchanged, with no fallback code."""
    bc, row = _update([1150.0, 1160.0, 1170.0], [14000.0, 14400.0, 14900.0])
    assert bc.lid_surface_flux_slope == pytest.approx(50.0, rel=1e-12)
    assert row['lid_slope_guard'] not in (3, 4)


def test_without_a_live_lid_no_fallback_acts():
    """Without a live lid the cap is off and the open part has no response, as before."""
    bc, row = _update(
        [1150.0, 1160.0, 1170.0, 1170.005], [1.4e4, 1.44e4, 1.48e4, 1.48e4], w=0.0
    )
    assert bc.lid_surface_flux_slope is None
    assert row['lid_slope_guard'] == 0


@settings(max_examples=120, deadline=None)
@given(
    T=st.lists(st.floats(1000.0, 1500.0), min_size=2, max_size=8),
    F=st.lists(st.floats(1.0e3, 3.0e4), min_size=9, max_size=9),
    tiny=st.floats(0.0, 0.009),
)
def test_the_fallback_is_the_newest_usable_secant_of_any_age(T, F, tiny):
    """For any history whose last step is short, the cap takes the newest usable secant of 2
    consecutive earlier rows, or the grey tangent of the current surface where none exists."""
    T = [*T, T[-1] + tiny]
    F = F[: len(T)]
    bc, row = _update(T, F)
    want = _newest_usable(T, F)
    if want is None:
        assert bc.lid_surface_flux_slope == pytest.approx(4.0 * const_sigma * T[-1] ** 3)
        assert row['lid_slope_guard'] == 4
    else:
        assert bc.lid_surface_flux_slope == pytest.approx(want[0], rel=1e-12)
        assert row['lid_slope_guard'] in (2, 3)
