"""Slope of the lid cap and the open response when the secant of the last 2 rows is unusable.

Under a live lid an unusable secant (a surface step below ``LID_SLOPE_DT_MIN``, a slope of 0
or below, or one that is not finite) is replaced by the newest usable secant of 2 consecutive
earlier rows, or by the grey tangent ``4 sigma T_surf^3`` where the run has none.
"""

from __future__ import annotations

import logging
import math

import pandas as pd
import pytest

pytest.importorskip('hypothesis')

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from proteus.atmos_clim.common import LID_SLOPE_DT_MIN  # noqa: E402
from proteus.interior_energetics.aragog import AragogRunner  # noqa: E402
from proteus.utils.constants import const_sigma  # noqa: E402
from tests.atmos_clim.test_common import _row  # noqa: E402
from tests.interior_energetics.test_lid_open_flux_guards import _runner_bc  # noqa: E402

pytestmark = pytest.mark.unit


def _update(T, F, w=1.0):
    """Run update_solver on a history with surfaces ``T`` and fluxes ``F``, the last entry being
    the current row; return the boundary parameters and the row."""
    bc, interior_o = _runner_bc()
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
    assert row['lid_slope_guard'] == 0


def test_without_a_live_lid_no_fallback_acts():
    """Without a live lid the cap is off and the open part has no response, as before."""
    bc, row = _update(
        [1150.0, 1160.0, 1170.0, 1170.005], [1.4e4, 1.44e4, 1.48e4, 1.48e4], w=0.0
    )
    assert bc.lid_surface_flux_slope is None
    assert row['lid_slope_guard'] == 0


@settings(max_examples=120, deadline=None)
@given(
    T=st.lists(st.floats(1000.0, 1430.0), min_size=2, max_size=8),  # below the lid base 1440 K
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
        # the open part cuts the fallback at the grey tangent of the current surface
        grey = 4.0 * const_sigma * T[-1] ** 3
        assert bc.lid_surface_flux_slope == pytest.approx(want[0], rel=1e-12)
        assert bc.open_surface_slope == pytest.approx(min(want[0], grey), rel=1e-12)
        assert row['lid_slope_guard'] == (5 if want[0] > grey else 3)


def test_the_fallback_passes_over_an_earlier_pair_with_a_short_step(caplog):
    """An earlier pair whose surface step is below LID_SLOPE_DT_MIN is not usable either: the
    fallback goes on to the pair before it (40 W m-2 K-1, 2 rows back)."""
    T = [1150.0, 1160.0, 1160.005, 1160.008]
    F = [14000.0, 14400.0, 15400.0, 15400.5]
    with caplog.at_level(logging.INFO):
        bc, row = _update(T, F)
    assert bc.lid_surface_flux_slope == pytest.approx(40.0, rel=1e-12)
    assert row['lid_slope_guard'] == 3
    assert any('2 rows back' in r.getMessage() for r in caplog.records)


def test_a_fallback_above_the_grey_tangent_is_cut_with_its_own_code():
    """A fallback secant above the grey tangent of the current surface keeps its value for the
    cap and is cut to the tangent for the open part; the column records code 5, not the code 2
    of a cut last secant."""
    T = [1150.0, 1160.0, 1170.0, 1170.004]
    F = [14000.0, 14400.0, 24400.0, 24400.5]  # earlier secant 1000 W m-2 K-1
    bc, row = _update(T, F)
    grey = 4.0 * const_sigma * T[-1] ** 3
    assert grey < 1000.0
    assert bc.lid_surface_flux_slope == pytest.approx(1000.0, rel=1e-12)
    assert bc.open_surface_slope == pytest.approx(grey, rel=1e-12)
    assert row['lid_slope_guard'] == 5


def test_the_fallback_is_cut_at_the_grey_tangent_not_at_the_last_secant():
    """After a fallback the open part cuts at the grey tangent of the current surface: a slope
    between the grey secant of the last 2 rows and that tangent stays as it is (code 3)."""
    T0, T1 = 1170.0, 1171.0
    g_sec = const_sigma * (T0**2 + T1**2) * (T0 + T1)
    g_tan = 4.0 * const_sigma * T1**3
    s = 0.5 * (g_sec + g_tan)
    F = [14000.0, 14400.0, 14400.0 + 10.0 * s, 14390.0 + 10.0 * s]  # last secant below 0
    bc, row = _update([1150.0, 1160.0, T0, T1], F)
    assert g_sec < s < g_tan
    assert bc.lid_surface_flux_slope == pytest.approx(s, rel=1e-12)
    assert bc.open_surface_slope == pytest.approx(s, rel=1e-12)
    assert row['lid_slope_guard'] == 3
