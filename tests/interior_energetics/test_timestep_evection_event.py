"""Unit tests for the evection-resonance timestep guard.

Covers ``proteus.interior_energetics.timestep._evection_zone_active``,
``_estimate_evection_dt_cap``, and the corresponding blocks inside
``next_step``: the rate-based shrink cap (bounding the fractional change
in ``eccentricity_sat`` per macro-step, not a fixed number of years), the
evection-scoped growth limiter, and the ``interior_o.evection_cooldown_remaining``
counter that keeps the limiter active for a configurable tail after the
band is left.

A synthetic eccentricity trajectory (rising sharply through capture,
peaking, then decaying slowly) drives ``next_step`` across the whole
approach -> capture -> peak -> exit sequence, to check the cap actually
tracks the changing rate rather than only ever returning the flat
ceiling.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from proteus.interior_energetics.timestep import (
    _estimate_evection_dt_cap,
    _evection_zone_active,
    next_step,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dt_config(
    evection_maximum=10.0,
    evection_target_rel_de=0.05,
    evection_de_floor=0.02,
    evection_rate_window=2,
    evection_growth_factor=0.0,
    evection_cooldown_iters=0,
):
    return SimpleNamespace(
        params=SimpleNamespace(
            dt=SimpleNamespace(
                evection_maximum=evection_maximum,
                evection_target_rel_de=evection_target_rel_de,
                evection_de_floor=evection_de_floor,
                evection_rate_window=evection_rate_window,
                evection_growth_factor=evection_growth_factor,
                evection_cooldown_iters=evection_cooldown_iters,
            )
        )
    )


def _hf_all_ecc(times, eccs):
    return pd.DataFrame(
        {
            'Time': np.asarray(times, dtype=float),
            'eccentricity_sat': np.asarray(eccs, dtype=float),
        }
    )


def _next_step_config(
    dt_maximum=1.0e7,
    evection_maximum=10.0,
    evection_target_rel_de=0.05,
    evection_de_floor=0.02,
    evection_rate_window=2,
    evection_growth_factor=0.0,
    evection_cooldown_iters=0,
):
    """Minimal config for next_step's 'maximum' dt-method branch, extended
    with the evection fields under test. Mirrors
    tests/interior_energetics/test_timestep_bolscale_event.py's
    _next_step_config, adding only what this file needs.
    """
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
        evection_maximum=evection_maximum,
        evection_target_rel_de=evection_target_rel_de,
        evection_de_floor=evection_de_floor,
        evection_rate_window=evection_rate_window,
        evection_growth_factor=evection_growth_factor,
        evection_cooldown_iters=evection_cooldown_iters,
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
    star = SimpleNamespace(bol_scale=1.0, bol_scale_start=None, bol_scale_duration=0.0)
    return SimpleNamespace(params=SimpleNamespace(dt=dt, stop=stop), star=star)


def _long_hf_all(times, eccs, n_pad_rows=12):
    """Pads a short (times, eccs) evection history with enough leading rows
    (constant, pre-approach values, spaced 10 yr apart and ending exactly
    one step before ``times[0]``) that next_step reaches the 'maximum'
    dt-method branch (needs > dt.window + 3 rows), while the LAST TWO rows
    of the returned frame are exactly ``(times[0], eccs[0])`` and
    ``(times[1], eccs[1])`` -- i.e. the (Time, eccentricity_sat) gap
    ``_estimate_evection_dt_cap`` computes its rate from is exactly
    ``times[1] - times[0]``, not contaminated by the padding spacing.
    """
    times = np.asarray(times, dtype=float)
    eccs = np.asarray(eccs, dtype=float)
    pad_times = np.arange(n_pad_rows, dtype=float) * 10.0 - 10.0 * n_pad_rows + times[0]
    pad_eccs = np.full(n_pad_rows, eccs[0])
    return pd.DataFrame(
        {
            'Time': np.concatenate([pad_times, times]),
            'eccentricity_sat': np.concatenate([pad_eccs, eccs]),
        }
    )


# ---------------------------------------------------------------------------
# _evection_zone_active
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_evection_zone_active_true_from_either_flag():
    """The zone is active if EITHER the tight (in_evection_band) or the
    wider, pre-emptive (near_evection_band) flag is set -- an OR, not an
    AND, since near_evection_band exists specifically to fire before
    in_evection_band does.
    """
    assert _evection_zone_active({'in_evection_band': 1.0, 'near_evection_band': 0.0}) is True
    assert _evection_zone_active({'in_evection_band': 0.0, 'near_evection_band': 1.0}) is True
    # Discrimination: both flags absent/false must NOT activate the zone.
    assert _evection_zone_active({'in_evection_band': 0.0, 'near_evection_band': 0.0}) is False
    assert _evection_zone_active({}) is False


# ---------------------------------------------------------------------------
# _estimate_evection_dt_cap: disable routes
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_disabled_when_ceiling_is_zero():
    """evection_maximum=0 disables the whole mechanism regardless of the
    band flags or the eccentricity history -- the source's opt-out sentinel.
    """
    config = _dt_config(evection_maximum=0.0)
    hf_row = {'in_evection_band': 1.0}
    hf_all = _hf_all_ecc([0.0, 10.0], [0.10, 0.15])
    assert _estimate_evection_dt_cap(hf_row, hf_all, config) == np.inf

    # Discrimination: the same history/flag with a positive ceiling is
    # constrained, so the sentinel above follows from the disable switch,
    # not from this history never constraining anything.
    enabled = _dt_config(evection_maximum=10.0)
    assert np.isfinite(_estimate_evection_dt_cap(hf_row, hf_all, enabled))


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_disabled_when_not_in_or_near_band():
    """A positive ceiling with rich history but neither band flag set must
    not constrain the timestep -- this cap only ever applies inside the
    evection zone.
    """
    config = _dt_config(evection_maximum=10.0)
    hf_row = {'in_evection_band': 0.0, 'near_evection_band': 0.0}
    hf_all = _hf_all_ecc([0.0, 10.0], [0.10, 0.20])
    assert _estimate_evection_dt_cap(hf_row, hf_all, config) == np.inf

    # Discrimination: near_evection_band alone (the wider, pre-emptive
    # flag) is enough to activate the cap.
    hf_row_near = {'in_evection_band': 0.0, 'near_evection_band': 1.0}
    assert np.isfinite(_estimate_evection_dt_cap(hf_row_near, hf_all, config))


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_falls_back_to_ceiling_without_history():
    """No usable history (None, <2 rows, or a single-row frame right at
    band entry) must return the ceiling, not crash and not silently
    disable the cap.
    """
    config = _dt_config(evection_maximum=10.0)
    hf_row = {'in_evection_band': 1.0}
    assert _estimate_evection_dt_cap(hf_row, None, config) == pytest.approx(10.0)

    one_row = _hf_all_ecc([0.0], [0.10])
    assert _estimate_evection_dt_cap(hf_row, one_row, config) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _estimate_evection_dt_cap: rate-based scaling (the core physics)
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_scales_inversely_with_observed_rate():
    """A faster observed |de/dt| must produce a SMALLER cap than a slower
    one at the same eccentricity level and configuration -- this is the
    entire point of replacing the flat ceiling with a rate-based bound:
    fast capture gets a tight cap, a slowly-decaying quasi-resonant tail
    does not.

    Pinned against the closed-form formula
    ``dt_cap = target_rel_de * max(e_now, de_floor) / de_dt`` with
    target_rel_de=0.05, de_floor=0.02: e goes 0.10 -> 0.12 over 10 yr
    (de_dt=0.002/yr) gives dt_cap = 0.05*0.12/0.002 = 3.0 yr.
    """
    config = _dt_config(
        evection_maximum=100.0, evection_target_rel_de=0.05, evection_de_floor=0.02
    )
    hf_row = {'in_evection_band': 1.0}

    fast = _hf_all_ecc([0.0, 10.0], [0.10, 0.12])  # de/dt = 2.0e-3 /yr
    slow = _hf_all_ecc([0.0, 10.0], [0.10, 0.101])  # de/dt = 1.0e-4 /yr

    cap_fast = _estimate_evection_dt_cap(hf_row, fast, config)
    cap_slow = _estimate_evection_dt_cap(hf_row, slow, config)

    assert cap_fast == pytest.approx(3.0, rel=1e-9)
    assert cap_fast < cap_slow

    # Discrimination: using e_prev (0.10) instead of e_now (0.12) in the
    # numerator would give 0.05*0.10/0.002 = 2.5 yr, off by 0.5 yr -- well
    # outside the tolerance below.
    wrong_uses_e_prev = 0.05 * 0.10 / 2.0e-3
    assert abs(cap_fast - wrong_uses_e_prev) > 0.1


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_uses_de_floor_for_tiny_eccentricity():
    """Right at capture onset, e_now can be far smaller than de_floor; the
    ratio's denominator must clamp to de_floor rather than blow the
    allowed step down toward zero for a physically unremarkable reason
    (a tiny starting e, not a fast rate).
    """
    config = _dt_config(
        evection_maximum=100.0, evection_target_rel_de=0.05, evection_de_floor=0.02
    )
    hf_row = {'in_evection_band': 1.0}
    # e: 0.001 -> 0.002 over 10 yr (de_dt = 1.0e-4 /yr), e_now << de_floor.
    hf_all = _hf_all_ecc([0.0, 10.0], [0.001, 0.002])

    cap = _estimate_evection_dt_cap(hf_row, hf_all, config)
    expected = 0.05 * 0.02 / 1.0e-4  # de_floor used, not e_now=0.002
    assert cap == pytest.approx(expected, rel=1e-9)

    # Discrimination: using e_now directly (0.002) instead of the floor
    # would give a cap 10x smaller (1.0 yr vs 10.0 yr) -- clearly distinct.
    wrong_uses_e_now = 0.05 * 0.002 / 1.0e-4
    assert abs(cap - wrong_uses_e_now) > 1.0


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_falls_back_to_ceiling_at_zero_rate():
    """Exactly at peak eccentricity, de/dt crosses zero by definition. The
    rate-based estimate is undefined there, NOT infinite/huge: the cap
    must fall back to the ceiling rather than let a naive e/de_dt blow up
    at the most dynamically sensitive point in the trajectory.
    """
    config = _dt_config(evection_maximum=7.0)
    hf_row = {'in_evection_band': 1.0}
    plateau = _hf_all_ecc([0.0, 10.0], [0.60, 0.60])  # de/dt == 0 exactly

    assert _estimate_evection_dt_cap(hf_row, plateau, config) == pytest.approx(7.0)

    # Discrimination: a nonzero rate at the same ceiling gives a value
    # strictly below it. de/dt=8.0e-3/yr here gives 0.05*0.68/8.0e-3=4.25 yr,
    # comfortably under the 7.0 yr ceiling (a slower rate, e.g. e: 0.60->0.62
    # over the same 10 yr, would compute 15.5 yr and get clipped BACK to the
    # ceiling by the min() -- still correct, but not a useful discriminator).
    rising = _hf_all_ecc([0.0, 10.0], [0.60, 0.68])
    assert _estimate_evection_dt_cap(hf_row, rising, config) < 7.0


# ---------------------------------------------------------------------------
# _estimate_evection_dt_cap: evection_rate_window smooths oscillation aliasing
# ---------------------------------------------------------------------------


def _oscillating_hf_all(n_points=11, period_yr=50.0, amplitude=0.05, e0=0.5, secular_rate=0.0):
    """A synthetic eccentricity_sat history driven by a pure sinusoidal
    oscillation -- standing in for the evection angle's own circulation/
    libration (``de_res`` in ``ps1d_evec``'s ``orbitals()``, whose
    ``dphi/dt`` is generically comparable to or faster than ``n_star``,
    not slow/adiabatic) -- optionally with a genuine secular trend added
    on top. Sampled every 10 yr; the default 11 points span exactly two
    full periods of the default 50 yr period.
    """
    t = np.arange(n_points, dtype=float) * 10.0
    e = e0 + amplitude * np.sin(2 * np.pi * t / period_yr) + secular_rate * t
    return _hf_all_ecc(t, e)


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_default_window_aliases_onto_pure_oscillation():
    """With the default ``evection_rate_window=2`` (a plain two-point
    diff), a PURE oscillation in ``eccentricity_sat`` with zero net
    secular trend still produces a small, clearly-below-ceiling cap --
    the two-point estimate aliases onto the oscillation's local slope
    rather than the (here exactly zero) secular trend. This is the
    failure mode reported for a real evection-band run: staying stuck at
    a tiny dt for the whole resonance episode, not just near genuine
    capture.
    """
    config = _dt_config(evection_maximum=50.0, evection_rate_window=2)
    hf_row = {'in_evection_band': 1.0}
    hf_all = _oscillating_hf_all()

    cap = _estimate_evection_dt_cap(hf_row, hf_all, config)
    assert cap < 10.0
    assert cap > 0.0


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_wide_window_averages_out_pure_oscillation():
    """The SAME pure-oscillation history as above, but with
    ``evection_rate_window`` covering the whole 2-period span: the
    least-squares secular slope is close to zero (a full-period average
    of a sinusoid vanishes), so the cap must relax back to the ceiling
    instead of staying pinned at the two-point aliased value -- this is
    the actual fix for eccentricity oscillating due to the evection angle
    keeping dt stuck small no matter where the system is along the
    resonance.
    """
    config = _dt_config(evection_maximum=50.0, evection_rate_window=11)
    hf_row = {'in_evection_band': 1.0}
    hf_all = _oscillating_hf_all()

    cap_wide = _estimate_evection_dt_cap(hf_row, hf_all, config)
    assert cap_wide == pytest.approx(50.0, rel=1e-9)

    # Discrimination: the SAME history/config except for the window size
    # gives a value more than 5x smaller -- the ceiling recovery above
    # follows from widening the window, not from this history/config
    # combination never constraining anything in the first place.
    narrow_config = _dt_config(evection_maximum=50.0, evection_rate_window=2)
    cap_narrow = _estimate_evection_dt_cap(hf_row, hf_all, narrow_config)
    assert cap_wide > 5.0 * cap_narrow


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_wide_window_still_tracks_a_genuine_secular_trend():
    """A genuine secular trend (0.002/yr) superimposed on the SAME
    oscillation must still produce a finite cap comfortably below the
    ceiling, of the right order of magnitude for that trend -- the
    windowing must not mask a real approach to capture just because it
    also rejects the oscillation's own contribution.
    """
    config = _dt_config(evection_maximum=200.0, evection_rate_window=11)
    hf_row = {'in_evection_band': 1.0}
    hf_all = _oscillating_hf_all(secular_rate=0.002)

    cap = _estimate_evection_dt_cap(hf_row, hf_all, config)
    # Pure-trend closed form: target_rel_de * e_now / secular_rate
    # = 0.05 * 0.7 / 0.002 = 17.5 yr; the fitted value should land within
    # roughly a factor of 2 of that.
    assert cap > 5.0
    # Discrimination: nowhere near the 200 yr ceiling, which is what a fit
    # swamped by the oscillation into reporting an essentially zero rate
    # would produce instead.
    assert cap < 40.0


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_uses_available_rows_when_shorter_than_window():
    """A configured window larger than the available history (e.g. early
    in a run, just after the zone flag first activates) must use
    whatever rows exist rather than raising or silently disabling the cap.
    """
    config = _dt_config(evection_maximum=10.0, evection_rate_window=50)
    hf_row = {'in_evection_band': 1.0}
    hf_all = _hf_all_ecc([0.0, 10.0, 20.0], [0.10, 0.11, 0.13])

    cap = _estimate_evection_dt_cap(hf_row, hf_all, config)
    assert np.isfinite(cap)
    assert cap > 0.0

    # Discrimination: matches the least-squares fit over exactly the 3
    # available rows (not, say, silently falling back to a two-point diff
    # over the last two only).
    slope, _ = np.polyfit([0.0, 10.0, 20.0], [0.10, 0.11, 0.13], 1)
    expected = min(10.0, 0.05 * max(0.13, 0.02) / abs(slope))
    assert cap == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# next_step integration: the rate cap actually binds the returned dt
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_next_step_evection_cap_tracks_rate_not_a_flat_ceiling():
    """Across a synthetic approach -> fast-capture -> near-peak sequence,
    next_step's evection cap must vary with the observed rate (small
    during fast capture, closer to the ceiling near the plateau) rather
    than sitting at the same flat value throughout -- the behavioural
    difference this change actually introduces over the old fixed cap.
    """
    config = _next_step_config(
        dt_maximum=1.0e5,
        evection_maximum=10.0,
        evection_target_rel_de=0.05,
        evection_de_floor=0.02,
    )

    fast_capture = _long_hf_all([0.0, 5.0], [0.10, 0.30])  # de/dt = 0.04 /yr
    near_peak = _long_hf_all([0.0, 5.0], [0.60, 0.605])  # de/dt = 1.0e-3 /yr

    hf_row_fast = {'Time': fast_capture['Time'].iloc[-1], 'in_evection_band': 1.0}
    hf_row_peak = {'Time': near_peak['Time'].iloc[-1], 'in_evection_band': 1.0}

    dt_fast = next_step(config, {}, hf_row_fast, fast_capture, step_sf=1.0)
    dt_peak = next_step(config, {}, hf_row_peak, near_peak, step_sf=1.0)

    # Fast capture: 0.05 * 0.30 / 0.04 = 0.375 yr, well under the ceiling.
    assert dt_fast == pytest.approx(0.375, rel=1e-6)
    # Near-plateau: 0.05 * 0.605 / 1.0e-3 = 30.25 yr, clipped to the 10 yr
    # ceiling -- i.e. the cap relaxes back to the flat value once the rate
    # is slow, rather than staying pinned at the tight fast-capture value.
    assert dt_peak == pytest.approx(10.0, rel=1e-9)
    assert dt_fast < dt_peak


@pytest.mark.physics_invariant
def test_next_step_evection_cap_engages_from_near_band_before_capture():
    """near_evection_band alone (without in_evection_band) must already
    engage the rate cap -- the pre-emptive lead margin this flag exists
    for, absorbing the one-iteration lag between evolve_orbit_satellite
    setting the flags and next_step's next read of them.
    """
    config = _next_step_config(dt_maximum=1.0e5, evection_maximum=10.0)
    hf_all = _long_hf_all([0.0, 5.0], [0.10, 0.30])

    hf_row_near = {
        'Time': hf_all['Time'].iloc[-1],
        'in_evection_band': 0.0,
        'near_evection_band': 1.0,
    }
    dt_near = next_step(config, {}, hf_row_near, hf_all, step_sf=1.0)
    assert dt_near < 1.0e5  # constrained, not the unmodified controller value

    # Discrimination: neither flag set on the identical history/config
    # leaves the cap disabled entirely.
    hf_row_neither = {
        'Time': hf_all['Time'].iloc[-1],
        'in_evection_band': 0.0,
        'near_evection_band': 0.0,
    }
    dt_neither = next_step(config, {}, hf_row_neither, hf_all, step_sf=1.0)
    assert dt_neither == pytest.approx(1.0e5, rel=1e-9)


# ---------------------------------------------------------------------------
# next_step integration: evection-scoped growth limiter + cooldown counter
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_next_step_evection_growth_limiter_caps_regrowth_after_exit():
    """After the band is left (in_evection_band=near_evection_band=0) but
    the cooldown counter is still armed, dt must be bounded to
    ``dt_prev_actual * evection_growth_factor`` rather than snapping
    straight back to the ordinary controller's (much larger) value --
    otherwise leaving the band would be indistinguishable, dt-wise, from
    never having entered it.
    """
    config = _next_step_config(
        dt_maximum=1.0e5,
        evection_maximum=10.0,
        evection_growth_factor=1.3,
        evection_cooldown_iters=5,
    )
    # Last accepted step was 8 yr (rows spaced 8 yr apart); the plain
    # 'maximum' controller would otherwise jump straight to dt.maximum.
    hf_all = _long_hf_all([0.0, 8.0], [0.30, 0.30])
    hf_row = {
        'Time': hf_all['Time'].iloc[-1],
        'in_evection_band': 0.0,
        'near_evection_band': 0.0,
    }
    interior_o = SimpleNamespace(evection_cooldown_remaining=3)

    dt = next_step(config, {}, hf_row, hf_all, step_sf=1.0, interior_o=interior_o)

    assert dt == pytest.approx(8.0 * 1.3, rel=1e-9)
    # The counter must count down by exactly one, not reset or freeze.
    assert interior_o.evection_cooldown_remaining == 2

    # Discrimination: with the counter already at zero and both zone flags
    # false, the limiter must be fully disengaged -- the unmodified
    # controller value passes through.
    interior_o_expired = SimpleNamespace(evection_cooldown_remaining=0)
    dt_expired = next_step(
        config, {}, dict(hf_row), hf_all, step_sf=1.0, interior_o=interior_o_expired
    )
    assert dt_expired == pytest.approx(1.0e5, rel=1e-9)


@pytest.mark.physics_invariant
def test_next_step_evection_cooldown_rearms_while_zone_is_active():
    """Every call where the zone is active must refresh
    evection_cooldown_remaining to evection_cooldown_iters, not merely
    leave a stale counter in place -- a long stay in the band must not
    exhaust the cooldown tail before the system actually exits.
    """
    config = _next_step_config(
        dt_maximum=1.0e5,
        evection_maximum=10.0,
        evection_growth_factor=1.3,
        evection_cooldown_iters=6,
    )
    hf_all = _long_hf_all([0.0, 8.0], [0.30, 0.31])
    hf_row = {'Time': hf_all['Time'].iloc[-1], 'in_evection_band': 1.0}
    interior_o = SimpleNamespace(evection_cooldown_remaining=1)

    next_step(config, {}, hf_row, hf_all, step_sf=1.0, interior_o=interior_o)

    assert interior_o.evection_cooldown_remaining == 6
    # Discrimination: had the zone been inactive, the same starting counter
    # would only decrement by one, not jump up to the refresh value.
    interior_o_inactive = SimpleNamespace(evection_cooldown_remaining=1)
    hf_row_inactive = {
        'Time': hf_all['Time'].iloc[-1],
        'in_evection_band': 0.0,
        'near_evection_band': 0.0,
    }
    next_step(config, {}, hf_row_inactive, hf_all, step_sf=1.0, interior_o=interior_o_inactive)
    assert interior_o_inactive.evection_cooldown_remaining == 0


@pytest.mark.physics_invariant
def test_next_step_evection_growth_limiter_disabled_by_default():
    """evection_growth_factor=0 (the schema default) must leave dt
    unconstrained by the growth limiter even with an active zone flag and
    a small previous step -- opt-in only, matching the global
    max_growth_factor's own disabled-by-default convention.
    """
    config = _next_step_config(
        dt_maximum=1.0e5, evection_maximum=0.0, evection_growth_factor=0.0
    )
    # A small STEP (0.5 yr) at a large absolute Time, so the static
    # (Time < 2 yr) branch does not confound this with the growth limiter.
    hf_all = _long_hf_all([100.0, 100.5], [0.30, 0.30])
    hf_row = {'Time': hf_all['Time'].iloc[-1], 'in_evection_band': 1.0}
    interior_o = SimpleNamespace(evection_cooldown_remaining=0)

    dt = next_step(config, {}, hf_row, hf_all, step_sf=1.0, interior_o=interior_o)
    assert dt == pytest.approx(1.0e5, rel=1e-9)

    # Discrimination: the identical scenario WITH a positive growth factor
    # does constrain dt to dt_prev*factor (0.5*1.3=0.65 yr), so the
    # unconstrained value above follows from the disable switch, not from
    # this scenario never triggering the limiter at all.
    config_on = _next_step_config(
        dt_maximum=1.0e5, evection_maximum=0.0, evection_growth_factor=1.3
    )
    dt_on = next_step(
        config_on,
        {},
        dict(hf_row),
        hf_all,
        step_sf=1.0,
        interior_o=SimpleNamespace(evection_cooldown_remaining=0),
    )
    assert dt_on == pytest.approx(0.5 * 1.3, rel=1e-9)
