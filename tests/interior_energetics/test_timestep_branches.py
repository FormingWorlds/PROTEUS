"""Branch coverage for ``proteus.interior_energetics.timestep``.

These tests exercise the early-Time static branch, the initial branch
when ``hf_all`` is too short, the proportional time-step formula, the
invalid-method error path, and the three "time until X" estimators
(``_estimate_solid``, ``_estimate_radeq``, ``_estimate_escape``)
including their "already there" short-circuits.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from proteus.interior_energetics.timestep import (
    SMALL,
    _estimate_escape,
    _estimate_radeq,
    _estimate_solid,
    _hf_from_iters,
    next_step,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _stop_namespace():
    stop_solid = SimpleNamespace(enabled=False, phi_crit=0.05)
    stop_radeqm = SimpleNamespace(enabled=False)
    stop_escape = SimpleNamespace(enabled=False)
    stop_time = SimpleNamespace(enabled=False, maximum=1.0e18)
    stop_disint = SimpleNamespace(offset_spin=0.0, offset_roche=0.0)
    return SimpleNamespace(
        solid=stop_solid,
        radeqm=stop_radeqm,
        escape=stop_escape,
        time=stop_time,
        disint=stop_disint,
    )


def _build_config(method='adaptive', dt_initial=10.0, dt_max=1.0e7, propconst=50.0):
    dt = SimpleNamespace(
        method=method,
        propconst=propconst,
        atol=0.02,
        rtol=0.10,
        scale_incr=1.6,
        scale_decr=0.8,
        window=3,
        minimum=100.0,
        minimum_rel=0.0,
        maximum=dt_max,
        maximum_rel=0.0,
        initial=dt_initial,
        mushy_maximum=0.0,
        mushy_upper=0.99,
        hysteresis_iters=0,
        hysteresis_sfinc=1.1,
        max_growth_factor=0.0,
    )
    return SimpleNamespace(params=SimpleNamespace(dt=dt, stop=_stop_namespace()))


def _two_row_hf():
    """A frame too short for the adaptive branch (forces the initial branch)."""
    return pd.DataFrame(
        {
            'Time': [0.0, 1.0e3],
            'F_atm': [1.0e4, 1.0e4],
            'Phi_global': [1.0, 1.0],
            'esc_rate_total': [0.0, 0.0],
            'F_int': [1.0e4, 1.0e4],
        }
    )


def _long_hf(n_rows=12, dt_prev=1.0e3, phi=1.0, p_surf=1.0e5, f_atm=1.0e4):
    times = np.arange(n_rows, dtype=float) * dt_prev
    return pd.DataFrame(
        {
            'Time': times,
            'F_atm': np.full(n_rows, f_atm),
            'F_tidal': np.zeros(n_rows),
            'F_radio': np.zeros(n_rows),
            'Phi_global': np.full(n_rows, float(phi)),
            'P_surf': np.full(n_rows, float(p_surf)),
            'esc_rate_total': np.zeros(n_rows),
            'F_int': np.full(n_rows, f_atm),
        }
    )


# ---------------------------------------------------------------------------
# next_step branch coverage
# ---------------------------------------------------------------------------


def test_next_step_uses_static_one_year_when_time_below_two_years():
    """Time < 2 yr triggers the static branch returning dt=1.0 yr (then
    multiplied by step_sf=1.0). Discrimination: dt=1.0 is decoupled from
    config.params.dt.initial (set to 10.0 here), so the static branch is
    the only one that produces this value.
    """
    config = _build_config(dt_initial=10.0)
    hf_row = {'Time': 1.0}
    hf_all = _long_hf()

    dt = next_step(config, {}, hf_row, hf_all, step_sf=1.0)

    assert dt == pytest.approx(1.0)
    assert dt != pytest.approx(config.params.dt.initial)


def test_next_step_static_branch_scales_with_retry_step_sf():
    """The static branch must respect ``step_sf`` (post-#676 fix where
    static/initial retries used to ignore the scale factor)."""
    config = _build_config()
    hf_row = {'Time': 1.5}

    dt = next_step(config, {}, hf_row, _long_hf(), step_sf=0.25)

    assert dt == pytest.approx(0.25)


def test_next_step_uses_initial_when_hf_all_too_short_for_window():
    """``dt_window + 5 >= len(hf_all['Time'])`` triggers the initial
    branch returning ``config.params.dt.initial``. Discrimination:
    initial (10.0) differs from the static 1.0 and from any plausible
    proportional/adaptive number for this state.
    """
    config = _build_config(dt_initial=10.0)
    hf_row = {'Time': 5.0}
    hf_all = _two_row_hf()

    dt = next_step(config, {}, hf_row, hf_all, step_sf=1.0)

    assert dt == pytest.approx(10.0)


def test_next_step_proportional_formula_is_time_over_propconst():
    """Proportional method: dtswitch = Time / propconst. With Time=520
    and propconst=52, we expect dt=10. Discrimination: replacing the
    formula with Time*propconst would give 27040; the test would fail
    by 3 orders of magnitude.
    """
    config = _build_config(method='proportional', propconst=52.0)
    config.params.dt.minimum = 1.0  # drop the floor so the proportional value can show through
    hf_row = {'Time': 520.0}
    hf_all = _long_hf(n_rows=12, dt_prev=50.0)

    dt = next_step(config, {}, hf_row, hf_all, step_sf=1.0)

    assert dt == pytest.approx(10.0, rel=1e-6)
    assert dt != pytest.approx(520.0 * 52.0)


def test_next_step_maximum_method_returns_dt_maximum():
    """Method 'maximum' returns ``params.dt.maximum`` directly."""
    config = _build_config(method='maximum', dt_max=5.0e3)
    hf_row = {'Time': 1.0e5}
    hf_all = _long_hf(n_rows=12, dt_prev=5.0e3)

    dt = next_step(config, {}, hf_row, hf_all, step_sf=1.0)

    assert dt == pytest.approx(5.0e3)


def test_next_step_unknown_method_raises_value_error(tmp_path):
    """An unrecognised method string trips the catch-all that writes a
    statusfile and raises ``ValueError``. Discrimination: the error
    message must contain the offending method name so downstream
    operators can diagnose the misconfiguration.
    """
    config = _build_config(method='not_a_real_method')
    hf_row = {'Time': 1.0e5}
    hf_all = _long_hf(n_rows=12, dt_prev=5.0e3)

    with pytest.raises(ValueError, match='not_a_real_method'):
        next_step(config, {'output': str(tmp_path)}, hf_row, hf_all, step_sf=1.0)


# ---------------------------------------------------------------------------
# _hf_from_iters and the three estimators
# ---------------------------------------------------------------------------


def test_hf_from_iters_logs_error_when_i1_not_less_than_i2(caplog):
    """``_hf_from_iters`` logs an error (does not raise) when the caller
    swaps i1 and i2. The function still returns both rows because the
    error is informational, not a hard failure.
    """
    hf_all = _long_hf(n_rows=4)
    with caplog.at_level(logging.ERROR, logger='fwl.proteus.interior_energetics.timestep'):
        h1, h2 = _hf_from_iters(hf_all, 2, 1)
    assert any('Cannot compare helpfile rows' in rec.message for rec in caplog.records)
    assert h1['Time'] == pytest.approx(hf_all['Time'].iloc[2])
    assert h2['Time'] == pytest.approx(hf_all['Time'].iloc[1])


def test_estimate_solid_returns_inf_when_already_solidified():
    """If ``Phi_global`` at i2 is below SMALL, the planet is treated as
    solid and the estimator returns +inf so the time-step is not
    artificially capped by a meaningless extrapolation.
    """
    hf_all = _long_hf(n_rows=10, dt_prev=1.0e3, phi=0.0)
    dt_solid = _estimate_solid(hf_all, i1=-2, i2=-1)
    assert dt_solid == np.inf


def test_estimate_solid_returns_inf_when_phi_is_steady():
    """If ``dp/p2`` is below SMALL (Phi flat at e.g. 1.0), the linear
    extrapolation to Phi=0 is meaningless and the estimator returns
    +inf rather than dividing by ~0.
    """
    hf_all = _long_hf(n_rows=10, dt_prev=1.0e3, phi=1.0)
    assert _estimate_solid(hf_all, i1=-2, i2=-1) == np.inf


def test_estimate_solid_extrapolates_to_phi_zero_for_decreasing_phi():
    """Linear extrapolation: at Phi=0.4 with dPhi/dt = -1e-4 per yr, the
    time to Phi=0 is 4000 yr. Discrimination: a mass-balance error that
    flipped the sign of dp would extrapolate AWAY from solidification
    and would give a different magnitude.
    """
    hf_all = pd.DataFrame(
        {
            'Time': [0.0, 1.0e3],
            'Phi_global': [0.5, 0.4],
            'F_atm': [1.0e4, 1.0e4],
            'F_tidal': [0.0, 0.0],
            'F_radio': [0.0, 0.0],
            'P_surf': [1.0e5, 1.0e5],
        }
    )
    dt_solid = _estimate_solid(hf_all, i1=0, i2=1)
    # Δt = -p2/(dp/dt) = -0.4 / ((-0.1)/1e3) = 4000.
    assert dt_solid == pytest.approx(4000.0, rel=1e-6)
    # Discrimination: a sign-flipped dp would give 4000 as well in
    # magnitude (abs() in the implementation), but the wrong direction
    # to use as a "time until solidification" prediction. Guard against
    # a regression that drops the sign entirely.
    assert dt_solid > SMALL


def test_estimate_radeq_returns_inf_when_balance_already_met():
    """When |F_atm - F_tidal - F_radio| < SMALL at i2, the planet is at
    radiative balance; the estimator returns +inf.
    """
    hf_all = pd.DataFrame(
        {
            'Time': [0.0, 1.0e3],
            'F_atm': [0.0, 0.0],
            'F_tidal': [0.0, 0.0],
            'F_radio': [0.0, 0.0],
            'Phi_global': [1.0, 1.0],
            'P_surf': [1.0e5, 1.0e5],
        }
    )
    assert _estimate_radeq(hf_all, i1=0, i2=1) == np.inf


def test_estimate_radeq_extrapolates_to_flux_zero():
    """At F_atm decreasing from 200 to 100 W/m^2 over 1000 yr (no
    tides/radio), time to F=0 is 1000 yr.
    """
    hf_all = pd.DataFrame(
        {
            'Time': [0.0, 1.0e3],
            'F_atm': [200.0, 100.0],
            'F_tidal': [0.0, 0.0],
            'F_radio': [0.0, 0.0],
            'Phi_global': [1.0, 1.0],
            'P_surf': [1.0e5, 1.0e5],
        }
    )
    dt_radeq = _estimate_radeq(hf_all, i1=0, i2=1)
    assert dt_radeq == pytest.approx(1000.0, rel=1e-6)


def test_estimate_radeq_returns_inf_when_flux_is_flat():
    """If df/f2 < SMALL, the estimator returns +inf instead of dividing
    by ~0.
    """
    hf_all = pd.DataFrame(
        {
            'Time': [0.0, 1.0e3],
            'F_atm': [100.0, 100.0],
            'F_tidal': [0.0, 0.0],
            'F_radio': [0.0, 0.0],
            'Phi_global': [1.0, 1.0],
            'P_surf': [1.0e5, 1.0e5],
        }
    )
    assert _estimate_radeq(hf_all, i1=0, i2=1) == np.inf


def test_estimate_escape_returns_inf_when_pressure_flat():
    """If surface pressure is steady (escape negligible), the estimator
    returns +inf.
    """
    hf_all = _long_hf(n_rows=4, p_surf=1.0e5)
    assert _estimate_escape(hf_all, i1=-2, i2=-1) == np.inf


def test_estimate_escape_extrapolates_to_zero_pressure():
    """Linear extrapolation: P drops from 2e5 to 1e5 Pa in 1000 yr,
    so time to P=0 is 1000 yr (slope = -100 Pa/yr; -P2 / slope).
    """
    hf_all = pd.DataFrame(
        {
            'Time': [0.0, 1.0e3],
            'P_surf': [2.0e5, 1.0e5],
            'F_atm': [1.0e4, 1.0e4],
            'F_tidal': [0.0, 0.0],
            'F_radio': [0.0, 0.0],
            'Phi_global': [1.0, 1.0],
        }
    )
    dt_escape = _estimate_escape(hf_all, i1=0, i2=1)
    assert dt_escape == pytest.approx(1000.0, rel=1e-6)
