"""Unit tests for the bolometric-scaling timestep guard.

Covers ``proteus.interior_energetics.timestep._estimate_bolscale`` event
which checks the current stellar age against the configured bolometric-scaling
and window fields, and the ``proteus.interior_energetics.timestep.next_step``.


Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from proteus.interior_energetics.timestep import _estimate_bolscale, next_step

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _config_with_star(bol_scale=1.0, bol_scale_start=None, bol_scale_duration=0.0):
    star = SimpleNamespace(
        bol_scale=bol_scale,
        bol_scale_start=bol_scale_start,
        bol_scale_duration=bol_scale_duration,
    )
    return SimpleNamespace(star=star)


def _hf_all_with_age(age_star, n_rows=3):
    return pd.DataFrame(
        {
            'Time': np.arange(n_rows, dtype=float),
            'age_star': np.full(n_rows, float(age_star)),
        }
    )


# ---------------------------------------------------------------------------
# _estimate_bolscale: trivial-disable routes
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_estimate_bolscale_disabled_when_bol_scale_is_one():
    """bol_scale == 1.0 is a no-op scaling factor regardless of the window,
    so the estimator must not constrain the timestep even with a window
    defined."""
    config = _config_with_star(bol_scale=1.0, bol_scale_start=0.5, bol_scale_duration=0.5)
    hf_all = _hf_all_with_age(age_star=0.7e9)  # inside the window, if it mattered
    assert _estimate_bolscale(hf_all, config) == np.inf

    # Discrimination: the same window with a scaling that is not unity does
    # constrain the step, so the sentinel above follows from bol_scale rather
    # than from a window that reaches nothing at this age.
    scaling = _config_with_star(bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5)
    assert np.isfinite(_estimate_bolscale(hf_all, scaling))


@pytest.mark.physics_invariant
def test_estimate_bolscale_disabled_when_bol_scale_start_is_none():
    """bol_scale_start=None means no window was ever configured, so a
    non-unity bol_scale alone must not constrain the timestep."""
    config = _config_with_star(bol_scale=2.0, bol_scale_start=None, bol_scale_duration=0.0)
    hf_all = _hf_all_with_age(age_star=0.7e9)
    assert _estimate_bolscale(hf_all, config) == np.inf

    # Discrimination: the same scaling with a window does constrain the step,
    # so the sentinel above follows from the missing start rather than from the
    # scaling being ignored.
    windowed = _config_with_star(bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5)
    assert np.isfinite(_estimate_bolscale(hf_all, windowed))


# ---------------------------------------------------------------------------
# _estimate_bolscale: active window, before / during / after
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_estimate_bolscale_before_window_returns_time_until_start():
    """Before the window opens, dt_bolscale = age_ini - age_now (converted
    from Gyr to yr via the *1e9 factor in the source).

    Window: start=0.5 Gyr -> age_ini=5.0e8 yr. age_now=4.0e8 yr, so the
    expected remaining time is 1.0e8 yr.
    """
    config = _config_with_star(bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5)
    hf_all = _hf_all_with_age(age_star=4.0e8)
    dt_bolscale = _estimate_bolscale(hf_all, config)
    assert dt_bolscale == pytest.approx(1.0e8, rel=1e-9)
    assert abs(dt_bolscale - 4.0e8) > 1.0e7


@pytest.mark.physics_invariant
def test_estimate_bolscale_inside_window_returns_time_until_end():
    """Inside the window, dt_bolscale = age_end - age_now.

    Window: [5.0e8, 1.0e9] yr. age_now=7.0e8 yr -> 3.0e8 yr remaining.
    """
    config = _config_with_star(bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5)
    hf_all = _hf_all_with_age(age_star=7.0e8)
    dt_bolscale = _estimate_bolscale(hf_all, config)
    assert dt_bolscale == pytest.approx(3.0e8, rel=1e-9)
    # Discrimination: the window is 5.0e8 yr wide, so a branch returning the
    # whole duration rather than what is left of it lands there instead.
    assert abs(dt_bolscale - 5.0e8) > 1.0e7


@pytest.mark.physics_invariant
def test_estimate_bolscale_after_window_returns_inf():
    """After the window has closed, the estimator no longer constrains
    the timestep (falls through both branches to the +inf sentinel)."""
    config = _config_with_star(bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5)
    hf_all = _hf_all_with_age(age_star=1.1e9)
    assert _estimate_bolscale(hf_all, config) == np.inf

    # Discrimination: an age inside the same window is constrained, so the
    # sentinel above follows from the window having closed rather than from
    # this configuration never constraining anything.
    inside = _hf_all_with_age(age_star=7.0e8)
    assert np.isfinite(_estimate_bolscale(inside, config))


@pytest.mark.physics_invariant
def test_estimate_bolscale_boundary_at_exact_start_uses_during_branch():
    """age_now exactly equal to age_ini fails the strict `age_now <
    age_ini` "before" check and falls into the `elif age_now < age_end`
    "during" branch, so the returned value is age_end - age_now, NOT the
    "before" formula's age_ini - age_now (which would be 0 here).
    """
    config = _config_with_star(bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5)
    hf_all = _hf_all_with_age(age_star=5.0e8)  # == age_ini exactly
    dt_bolscale = _estimate_bolscale(hf_all, config)
    assert dt_bolscale == pytest.approx(5.0e8, rel=1e-9)  # age_end - age_now = 1e9 - 5e8
    # Discrimination: the before branch would return age_ini - age_now, which
    # is zero at this boundary and would stall the step rather than size it.
    assert dt_bolscale > 1.0e7


@pytest.mark.physics_invariant
def test_estimate_bolscale_defaults_missing_age_star_column_to_zero():
    """hf_all without an 'age_star' column must not raise: the source
    reads it via ``.get('age_star', 0.0)``. This is the error-contract
    path for malformed/partial helpfile frames (no validation exists
    upstream, so the limit-input behaviour is the contract to pin).
    """
    config = _config_with_star(bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5)
    hf_all = pd.DataFrame({'Time': [0.0, 1.0, 2.0]})  # no age_star column
    dt_bolscale = _estimate_bolscale(hf_all, config)
    assert dt_bolscale == pytest.approx(5.0e8, rel=1e-9)
    # Discrimination: an age inside the window takes the other branch and
    # returns the time to its end, so the value above is the default of zero
    # driving the before-the-window formula rather than a figure this
    # configuration returns for any frame it is handed.
    during = _estimate_bolscale(_hf_all_with_age(age_star=7.0e8), config)
    assert during == pytest.approx(3.0e8, rel=1e-9)


# ---------------------------------------------------------------------------
# next_step integration: the bolscale clip actually binds the returned dt
# ---------------------------------------------------------------------------


def _next_step_config(dt_maximum, bol_scale, bol_scale_start, bol_scale_duration):
    dt = SimpleNamespace(
        method='maximum',
        propconst=52.0,
        atol=0.02,
        rtol=0.10,
        scale_incr=1.6,
        scale_decr=0.8,
        window=3,
        minimum=0.1,
        minimum_rel=0.0,
        maximum=dt_maximum,
        maximum_rel=0.0,
        initial=1.0,
        mushy_maximum=0.0,
        mushy_upper=0.99,
        hysteresis_iters=0,
        hysteresis_sfinc=1.1,
        max_growth_factor=0.0,
    )
    stop = SimpleNamespace(
        solid=SimpleNamespace(enabled=False, phi_crit=0.05),
        radeqm=SimpleNamespace(enabled=False),
        escape=SimpleNamespace(enabled=False),
        time=SimpleNamespace(enabled=False, maximum=1.0e18),
    )
    star = SimpleNamespace(
        bol_scale=bol_scale,
        bol_scale_start=bol_scale_start,
        bol_scale_duration=bol_scale_duration,
    )
    return SimpleNamespace(params=SimpleNamespace(dt=dt, stop=stop), star=star)


def _next_step_hf_all(age_star, n_rows=12):
    """Long enough (>dt.window+3) that next_step reaches the 'maximum'
    dt-method branch instead of the 'initial' branch."""
    return pd.DataFrame(
        {
            'Time': np.arange(n_rows, dtype=float) * 10.0,
            'age_star': np.full(n_rows, float(age_star)),
        }
    )


@pytest.mark.physics_invariant
def test_next_step_clips_dt_when_bolscale_window_is_imminent():
    """The dt.method='maximum' branch would normally return dt.maximum
    (1e5 yr) unmodified; with the bolometric-scaling window opening in
    5e4 yr, next_step must clip to that smaller remaining time instead of
    silently stepping past the window.
    """
    # Window: start=0.5 Gyr -> age_ini=5.0e8 yr. age_now is 5e4 yr short
    # of that, so the window opens in exactly 5e4 yr.
    config = _next_step_config(
        dt_maximum=1.0e5, bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5
    )
    hf_row = {'Time': 100.0}
    hf_all = _next_step_hf_all(age_star=5.0e8 - 5.0e4)

    dt = next_step(config, {}, hf_row, hf_all, step_sf=1.0)
    assert dt == pytest.approx(5.0e4, rel=1e-9)

    # Monotonicity/positivity guard
    assert dt < 1.0e5
    assert dt > 0.0


@pytest.mark.physics_invariant
def test_next_step_does_not_clip_when_window_is_far_away():
    """When the bolometric-scaling window is far in the future (remaining
    time well above dt.maximum), the clip does not bind and next_step
    returns the un-modified controller value."""
    config = _next_step_config(
        dt_maximum=1.0e5, bol_scale=2.0, bol_scale_start=50.0, bol_scale_duration=0.5
    )
    hf_row = {'Time': 100.0}
    hf_all = _next_step_hf_all(age_star=1.0e8)  # window opens at 5e10 yr, far away

    dt = next_step(config, {}, hf_row, hf_all, step_sf=1.0)
    assert dt == pytest.approx(1.0e5, rel=1e-9)
    # Discrimination: the same controller with the window close at hand does
    # clip, so the value above is the ceiling passing through rather than a
    # step the clip never reaches on any configuration.
    near = _next_step_config(
        dt_maximum=1.0e5, bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5
    )
    # 1.0e3 yr short of the window, well inside the 1.0e5 yr ceiling.
    close = _next_step_hf_all(age_star=4.99999e8)
    assert next_step(near, {}, hf_row, close, step_sf=1.0) < 1.0e5


@pytest.mark.physics_invariant
def test_next_step_skips_bolscale_clip_when_hf_all_is_none():
    """On the very first call there is no history yet, so hf_all is None and
    ``if hf_all is not None`` must skip `_estimate_bolscale` rather than
    crashing on ``None.iloc[-1]``. Force the static branch (Time < 2 yr) so
    no other part of next_step needs hf_all either.
    """
    config = _next_step_config(
        dt_maximum=1.0e5, bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5
    )
    hf_row = {'Time': 1.0}  # static branch: Time < 2.0

    # Static time-step branch returns exactly 1.0 yr
    dt = next_step(config, {}, hf_row, None, step_sf=1.0)
    assert dt == pytest.approx(1.0)

    # Discrimination: the same call given a history whose window opens a
    # quarter of a year out clips the static step to that remaining time, so
    # the value above is the clamp being skipped rather than a step no window
    # could shorten. The window opens at 5.0e8 yr.
    near_edge = _next_step_hf_all(age_star=5.0e8 - 0.25)
    assert next_step(config, {}, hf_row, near_edge, step_sf=1.0) == pytest.approx(
        0.25, rel=1e-9
    )


# ---------------------------------------------------------------------------
# interior_o.timestep_clamped: flag consumed by the main loop to force a
# stellar update at the window edge (proteus.py STELLAR FLUX MANAGEMENT).
# ---------------------------------------------------------------------------


def test_next_step_flags_interior_o_when_bolscale_clamps_the_step():
    """When the bolometric-scaling window opens before the controller's own
    dt would, next_step must mark ``interior_o.timestep_clamped = True`` so
    the main loop can force a stellar update even if the resulting step is
    shorter than ``params.dt.starinst``."""
    config = _next_step_config(
        dt_maximum=1.0e5, bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.5
    )
    hf_row = {'Time': 100.0}
    hf_all = _next_step_hf_all(age_star=5.0e8 - 5.0e4)
    # t_next_impact is infinite whenever no giant impact is scheduled, which
    # is every run without accretion; next_step reads it to decide whether to
    # cap dt so the loop lands on the impact.
    interior_o = SimpleNamespace(timestep_clamped=False, t_next_impact=float('inf'))

    dt = next_step(config, {}, hf_row, hf_all, step_sf=1.0, interior_o=interior_o)

    assert dt == pytest.approx(5.0e4, rel=1e-9)
    assert interior_o.timestep_clamped is True


def test_next_step_does_not_flag_interior_o_when_window_is_far_away():
    """Discrimination case: when the controller's own dt is already smaller
    than the remaining time to the window, the bolscale estimate did not
    bind, so the flag must stay False (else every step would force a
    stellar update, defeating the dt.starinst cadence entirely)."""
    config = _next_step_config(
        dt_maximum=1.0e5, bol_scale=2.0, bol_scale_start=50.0, bol_scale_duration=0.5
    )
    hf_row = {'Time': 100.0}
    hf_all = _next_step_hf_all(age_star=1.0e8)  # window opens at 5e10 yr, far away
    interior_o = SimpleNamespace(
        timestep_clamped=True,  # start True to prove it flips
        t_next_impact=float('inf'),
    )

    dt = next_step(config, {}, hf_row, hf_all, step_sf=1.0, interior_o=interior_o)

    assert dt == pytest.approx(1.0e5, rel=1e-9)
    assert interior_o.timestep_clamped is False
