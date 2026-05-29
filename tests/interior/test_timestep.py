"""
Unit tests for ``proteus.interior.timestep``.

References:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from proteus.interior.timestep import next_step


def _make_config(
    *,
    method: str = 'maximum',
    dt_maximum: float = 20.0,
    stop_time_enabled: bool = True,
    stop_time_maximum: float = 55.0,
):
    return SimpleNamespace(
        params=SimpleNamespace(
            dt=SimpleNamespace(
                method=method,
                initial=1.0,
                maximum=dt_maximum,
                maximum_rel=0.0,
                minimum=0.1,
                minimum_rel=0.0,
                proportional=SimpleNamespace(propconst=10.0),
                adaptive=SimpleNamespace(
                    window=1,
                    rtol=0.1,
                    atol=1e-6,
                    scale_incr=2.0,
                    scale_decr=0.5,
                ),
            ),
            stop=SimpleNamespace(
                time=SimpleNamespace(enabled=stop_time_enabled, maximum=stop_time_maximum),
                solid=SimpleNamespace(enabled=False),
                radeqm=SimpleNamespace(enabled=False),
                escape=SimpleNamespace(enabled=False),
            ),
        )
    )


def _make_hf_all():
    return pd.DataFrame({'Time': [10.0, 20.0, 30.0, 40.0, 45.0, 47.0, 48.0, 49.0]})


@pytest.mark.unit
def test_next_step_caps_dt_to_remaining_final_time():
    """Timestep must be clipped so Time + dt does not exceed stop.time.maximum."""
    config = _make_config(dt_maximum=20.0, stop_time_enabled=True, stop_time_maximum=55.0)
    hf_row = {'Time': 50.0}

    dt = next_step(config=config, dirs={}, hf_row=hf_row, hf_all=_make_hf_all(), step_sf=1.0)

    assert dt == pytest.approx(5.0)
    assert hf_row['Time'] + dt <= config.params.stop.time.maximum


@pytest.mark.unit
def test_next_step_handles_subyear_remaining_time_without_overshoot():
    """When remaining integration time is <1 year, dt should still avoid overshoot."""
    config = _make_config(dt_maximum=10.0, stop_time_enabled=True, stop_time_maximum=52)
    hf_row = {'Time': 50.0}

    dt = next_step(config=config, dirs={}, hf_row=hf_row, hf_all=_make_hf_all(), step_sf=1.0)

    assert dt == pytest.approx(2)
    assert hf_row['Time'] + dt <= config.params.stop.time.maximum


@pytest.mark.unit
def test_next_step_not_capped_by_stop_time_when_disabled():
    """If stop.time is disabled, dt should follow normal configured dt.maximum limits."""
    config = _make_config(dt_maximum=7.5, stop_time_enabled=False, stop_time_maximum=50.2)
    hf_row = {'Time': 50.0}

    dt = next_step(config=config, dirs={}, hf_row=hf_row, hf_all=_make_hf_all(), step_sf=1.0)

    assert dt == pytest.approx(7.5)
