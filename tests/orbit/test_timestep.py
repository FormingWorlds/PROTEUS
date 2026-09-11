"""Unit tests for the evection dt-cap logic (``proteus.orbit.timestep``).

Mirrors ``src/proteus/orbit/timestep.py``, which was split out of
``proteus.orbit.satellite`` (and, before that, out of
``proteus.interior_energetics.timestep.next_step`` -- see
``_estimate_evection_dt_cap_yr``'s own docstring for that lineage) so this
dt-cap machinery reads as one coherent unit.

Exercises:

- ``_evection_rate_cap_yr``: the secular-rate half of the cap (bounding
  the fractional change in ``eccentricity_sat`` per macro-step, not a
  fixed number of years) -- disable routes (ceiling zero, zone inactive,
  no history), the closed-form rate scaling with a discrimination guard
  against using ``e_prev`` instead of ``e_now``, the ``de_floor`` clamp
  at capture onset, the zero-rate fallback at eccentricity peak, and the
  ``evection_rate_window``'s oscillation-smoothing behaviour (a pure
  sinusoid's two-point aliasing vs. a full-period average recovering the
  ceiling, and a genuine secular trend still tracked through the window).
- ``_estimate_evection_dt_cap_yr``: the growth-limiter half (bounding dt
  to ``dt_prev_actual_yr * evection_growth_factor`` while the zone is
  active or during the ``evection_cooldown_iters`` tail after leaving
  it), its ``tides_o.evection_cooldown_remaining`` counter refresh/decay,
  the opt-in-by-default disable, and that the two halves are actually
  folded together via ``min()`` rather than one silently overriding the
  other.

``proteus.orbit.satellite.evolve_orbit_satellite``'s own use of these
functions (maintaining ``tides_o.evection_ecc_history``, exporting the
single ``hf_row['evection_dt_cap_yr']`` column) is covered in
``tests/orbit/test_satellite.py`` instead.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_building.md
- docs/How-to/test_categorization.md
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from proteus.orbit.common import Tides_t
from proteus.orbit.timestep import _estimate_evection_dt_cap_yr, _evection_rate_cap_yr

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# _evection_rate_cap_yr: the secular-rate half of the evection dt cap
# (the other half is the growth limiter, see the section below). Ported
# from tests/interior_energetics/test_timestep_evection_event.py (now
# deleted -- that logic lived through timestep.next_step before the whole
# mechanism moved to orbit; see _estimate_evection_dt_cap_yr's own
# docstring). The closed-form math is unchanged; only the history source
# changed, from a `hf_all` DataFrame to `tides_o.evection_ecc_history`,
# and zone-active detection is now the CALLER's job (a plain bool
# parameter), not recomputed internally from hf_row flags -- those flags
# no longer exist at all; see
# tests/orbit/test_satellite.py::test_evolve_orbit_satellite_exports_a_single_evection_dt_cap_yr_column
# for that removal.
# ---------------------------------------------------------------------------


def _dt_cap_config(
    evection_maximum=10.0,
    evection_target_rel_de=0.05,
    evection_de_floor=0.02,
    evection_rate_window=2,
    evection_growth_factor=0.0,
    evection_cooldown_iters=0,
) -> Any:
    return cast(
        Any,
        SimpleNamespace(
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
        ),
    )


def _tides_with_ecc_history(times, eccs) -> Tides_t:
    return Tides_t(evection_ecc_history=list(zip(times, eccs)))


def _oscillating_ecc_history(
    n_points=11, period_yr=50.0, amplitude=0.05, e0=0.5, secular_rate=0.0
):
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
    return list(t), list(e)


@pytest.mark.physics_invariant
def test_evection_rate_cap_yr_disabled_when_ceiling_is_zero():
    """``evection_maximum=0`` disables the whole mechanism regardless of
    zone-active or the eccentricity history -- the source's opt-out
    sentinel.
    """
    tides_o = _tides_with_ecc_history([0.0, 10.0], [0.10, 0.15])

    disabled = _dt_cap_config(evection_maximum=0.0)
    assert _evection_rate_cap_yr(tides_o, True, disabled) == np.inf

    # Discrimination: the same history/zone_active with a positive ceiling
    # is constrained, so the sentinel above follows from the disable
    # switch, not from this history never constraining anything.
    enabled = _dt_cap_config(evection_maximum=10.0)
    assert np.isfinite(_evection_rate_cap_yr(tides_o, True, enabled))


@pytest.mark.physics_invariant
def test_evection_rate_cap_yr_disabled_when_zone_inactive():
    """A positive ceiling with rich history but ``zone_active=False`` must
    not constrain the timestep -- this cap only ever applies inside (or
    approaching) the evection zone.
    """
    config = _dt_cap_config(evection_maximum=10.0)
    tides_o = _tides_with_ecc_history([0.0, 10.0], [0.10, 0.20])
    assert _evection_rate_cap_yr(tides_o, False, config) == np.inf
    assert np.isfinite(_evection_rate_cap_yr(tides_o, True, config))


@pytest.mark.physics_invariant
def test_evection_rate_cap_yr_falls_back_to_ceiling_without_history():
    """No usable history (empty, or a single sample right at band entry)
    must return the ceiling, not crash and not silently disable the cap.
    """
    config = _dt_cap_config(evection_maximum=10.0)
    assert _evection_rate_cap_yr(Tides_t(), True, config) == pytest.approx(10.0)

    one_sample = _tides_with_ecc_history([0.0], [0.10])
    assert _evection_rate_cap_yr(one_sample, True, config) == pytest.approx(10.0)


@pytest.mark.physics_invariant
def test_evection_rate_cap_yr_scales_inversely_with_observed_rate():
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
    config = _dt_cap_config(
        evection_maximum=100.0, evection_target_rel_de=0.05, evection_de_floor=0.02
    )
    fast = _tides_with_ecc_history([0.0, 10.0], [0.10, 0.12])  # de/dt = 2.0e-3 /yr
    slow = _tides_with_ecc_history([0.0, 10.0], [0.10, 0.101])  # de/dt = 1.0e-4 /yr

    cap_fast = _evection_rate_cap_yr(fast, True, config)
    cap_slow = _evection_rate_cap_yr(slow, True, config)

    assert cap_fast == pytest.approx(3.0, rel=1e-9)
    assert cap_fast < cap_slow

    # Discrimination: using e_prev (0.10) instead of e_now (0.12) in the
    # numerator would give 0.05*0.10/0.002 = 2.5 yr, off by 0.5 yr -- well
    # outside the tolerance below.
    wrong_uses_e_prev = 0.05 * 0.10 / 2.0e-3
    assert abs(cap_fast - wrong_uses_e_prev) > 0.1


@pytest.mark.physics_invariant
def test_evection_rate_cap_yr_uses_de_floor_for_tiny_eccentricity():
    """Right at capture onset, e_now can be far smaller than de_floor; the
    ratio's denominator must clamp to de_floor rather than blow the
    allowed step down toward zero for a physically unremarkable reason
    (a tiny starting e, not a fast rate).
    """
    config = _dt_cap_config(
        evection_maximum=100.0, evection_target_rel_de=0.05, evection_de_floor=0.02
    )
    # e: 0.001 -> 0.002 over 10 yr (de_dt = 1.0e-4 /yr), e_now << de_floor.
    tides_o = _tides_with_ecc_history([0.0, 10.0], [0.001, 0.002])

    cap = _evection_rate_cap_yr(tides_o, True, config)
    expected = 0.05 * 0.02 / 1.0e-4  # de_floor used, not e_now=0.002
    assert cap == pytest.approx(expected, rel=1e-9)

    # Discrimination: using e_now directly (0.002) instead of the floor
    # would give a cap 10x smaller (1.0 yr vs 10.0 yr) -- clearly distinct.
    wrong_uses_e_now = 0.05 * 0.002 / 1.0e-4
    assert abs(cap - wrong_uses_e_now) > 1.0


@pytest.mark.physics_invariant
def test_evection_rate_cap_yr_falls_back_to_ceiling_at_zero_rate():
    """Exactly at peak eccentricity, de/dt crosses zero by definition. The
    rate-based estimate is undefined there, NOT infinite/huge: the cap
    must fall back to the ceiling rather than let a naive e/de_dt blow up
    at the most dynamically sensitive point in the trajectory.
    """
    config = _dt_cap_config(evection_maximum=7.0)
    plateau = _tides_with_ecc_history([0.0, 10.0], [0.60, 0.60])  # de/dt == 0 exactly

    assert _evection_rate_cap_yr(plateau, True, config) == pytest.approx(7.0)

    # Discrimination: a nonzero rate at the same ceiling gives a value
    # strictly below it.
    rising = _tides_with_ecc_history([0.0, 10.0], [0.60, 0.68])
    assert _evection_rate_cap_yr(rising, True, config) < 7.0


@pytest.mark.physics_invariant
def test_evection_rate_cap_yr_falls_back_to_ceiling_at_zero_time_span():
    """Two (or more) history samples recorded at the same timestamp give a
    zero-width window, so the secular slope ``de/dt`` is undefined (a
    ``0/0`` division, not merely small). The cap must fall back to the
    ceiling here too, the same as the zero-history and zero-rate cases,
    rather than raise a ``ZeroDivisionError`` or propagate a NaN/inf cap.
    """
    config = _dt_cap_config(evection_maximum=9.0)
    same_instant = _tides_with_ecc_history([5.0, 5.0], [0.30, 0.34])

    assert _evection_rate_cap_yr(same_instant, True, config) == pytest.approx(9.0)

    # Discrimination: the same eccentricity change over a nonzero span is
    # constrained well below the ceiling, so the fallback above follows
    # from the degenerate time span, not from this de being too small to
    # ever constrain anything.
    spread_out = _tides_with_ecc_history([0.0, 10.0], [0.30, 0.34])
    assert _evection_rate_cap_yr(spread_out, True, config) < 9.0


@pytest.mark.physics_invariant
def test_evection_rate_cap_yr_default_window_aliases_onto_pure_oscillation():
    """With the default ``evection_rate_window=2`` (a plain two-point
    diff), a PURE oscillation in eccentricity with zero net secular trend
    still produces a small, clearly-below-ceiling cap -- the two-point
    estimate aliases onto the oscillation's local slope rather than the
    (here exactly zero) secular trend. This is the failure mode reported
    for a real evection-band run: staying stuck at a tiny dt for the
    whole resonance episode, not just near genuine capture.
    """
    config = _dt_cap_config(evection_maximum=50.0, evection_rate_window=2)
    tides_o = _tides_with_ecc_history(*_oscillating_ecc_history())

    cap = _evection_rate_cap_yr(tides_o, True, config)
    assert cap < 10.0
    assert cap > 0.0


@pytest.mark.physics_invariant
def test_evection_rate_cap_yr_wide_window_averages_out_pure_oscillation():
    """The SAME pure-oscillation history as above, but with
    ``evection_rate_window`` covering the whole 2-period span: the
    least-squares secular slope is close to zero (a full-period average
    of a sinusoid vanishes), so the cap must relax back to the ceiling
    instead of staying pinned at the two-point aliased value.
    """
    times, eccs = _oscillating_ecc_history()
    tides_wide = _tides_with_ecc_history(times, eccs)
    tides_narrow = _tides_with_ecc_history(times, eccs)

    wide_config = _dt_cap_config(evection_maximum=50.0, evection_rate_window=11)
    cap_wide = _evection_rate_cap_yr(tides_wide, True, wide_config)
    assert cap_wide == pytest.approx(50.0, rel=1e-9)

    # Discrimination: the SAME history/config except for the window size
    # gives a value more than 5x smaller -- the ceiling recovery above
    # follows from widening the window, not from this history/config
    # combination never constraining anything in the first place.
    narrow_config = _dt_cap_config(evection_maximum=50.0, evection_rate_window=2)
    cap_narrow = _evection_rate_cap_yr(tides_narrow, True, narrow_config)
    assert cap_wide > 5.0 * cap_narrow


@pytest.mark.physics_invariant
def test_evection_rate_cap_yr_wide_window_still_tracks_a_genuine_secular_trend():
    """A genuine secular trend (0.002/yr) superimposed on the SAME
    oscillation must still produce a finite cap comfortably below the
    ceiling, of the right order of magnitude for that trend -- the
    windowing must not mask a real approach to capture just because it
    also rejects the oscillation's own contribution.
    """
    config = _dt_cap_config(evection_maximum=200.0, evection_rate_window=11)
    tides_o = _tides_with_ecc_history(*_oscillating_ecc_history(secular_rate=0.002))

    cap = _evection_rate_cap_yr(tides_o, True, config)
    # Pure-trend closed form: target_rel_de * e_now / secular_rate
    # = 0.05 * 0.7 / 0.002 = 17.5 yr; the fitted value should land within
    # roughly a factor of 2 of that.
    assert cap > 5.0
    # Discrimination: nowhere near the 200 yr ceiling, which is what a fit
    # swamped by the oscillation into reporting an essentially zero rate
    # would produce instead.
    assert cap < 40.0


@pytest.mark.physics_invariant
def test_evection_rate_cap_yr_uses_available_samples_when_shorter_than_window():
    """A configured window larger than the available history (e.g. early
    in a run, just after the zone flag first activates) must use
    whatever samples exist rather than raising or silently disabling the
    cap.
    """
    config = _dt_cap_config(evection_maximum=10.0, evection_rate_window=50)
    tides_o = _tides_with_ecc_history([0.0, 10.0, 20.0], [0.10, 0.11, 0.13])

    cap = _evection_rate_cap_yr(tides_o, True, config)
    assert np.isfinite(cap)
    assert cap > 0.0

    # Discrimination: matches the least-squares fit over exactly the 3
    # available samples (not, say, silently falling back to a two-point
    # diff over the last two only).
    slope, _ = np.polyfit([0.0, 10.0, 20.0], [0.10, 0.11, 0.13], 1)
    expected = min(10.0, 0.05 * max(0.13, 0.02) / abs(slope))
    assert cap == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# _estimate_evection_dt_cap_yr: the growth-limiter half (folded with
# _evection_rate_cap_yr above via min()) plus the cooldown counter. Ported
# from tests/interior_energetics/test_timestep_evection_event.py (now
# deleted), which used to test this through timestep.next_step and
# Interior_t.evection_cooldown_remaining before the growth limiter moved
# into orbit alongside the rate cap. ``dt_prev_actual_yr`` (the actual
# elapsed Time of the call whose end this cap is computed at) stands in
# for the old ``hf_all['Time']`` two-row gap; the cooldown counter now
# lives on ``tides_o`` instead of ``interior_o``.
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_yr_growth_limiter_caps_regrowth_after_exit():
    """After the band is left (zone_active=False) but the cooldown counter
    is still armed, the cap must bound dt to
    ``dt_prev_actual_yr * evection_growth_factor`` rather than leaving it
    unconstrained -- otherwise leaving the band would be indistinguishable,
    dt-wise, from never having entered it.
    """
    config = _dt_cap_config(
        evection_maximum=10.0, evection_growth_factor=1.3, evection_cooldown_iters=5
    )
    tides_o = Tides_t(evection_cooldown_remaining=3)

    # Last accepted macro-step was 8 yr.
    cap = _estimate_evection_dt_cap_yr(tides_o, False, 8.0, config)
    assert cap == pytest.approx(8.0 * 1.3, rel=1e-9)
    # The counter must count down by exactly one, not reset or freeze.
    assert tides_o.evection_cooldown_remaining == 2

    # Discrimination: with the counter already at zero and the zone
    # inactive, the limiter must be fully disengaged -- np.inf (unbound).
    expired = Tides_t(evection_cooldown_remaining=0)
    cap_expired = _estimate_evection_dt_cap_yr(expired, False, 8.0, config)
    assert cap_expired == np.inf


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_yr_cooldown_rearms_while_zone_is_active():
    """Every call where the zone is active must refresh
    ``evection_cooldown_remaining`` to ``evection_cooldown_iters``, not
    merely leave a stale counter in place -- a long stay in the band must
    not exhaust the cooldown tail before the system actually exits.
    """
    config = _dt_cap_config(
        evection_maximum=10.0, evection_growth_factor=1.3, evection_cooldown_iters=6
    )
    tides_o = Tides_t(evection_cooldown_remaining=1)

    _estimate_evection_dt_cap_yr(tides_o, True, 8.0, config)
    assert tides_o.evection_cooldown_remaining == 6

    # Discrimination: had the zone been inactive, the same starting
    # counter would only decrement by one, not jump up to the refresh value.
    inactive = Tides_t(evection_cooldown_remaining=1)
    _estimate_evection_dt_cap_yr(inactive, False, 8.0, config)
    assert inactive.evection_cooldown_remaining == 0


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_yr_growth_limiter_disabled_by_default():
    """``evection_growth_factor=0`` (the schema default) must leave the
    growth-limiter half unconstrained even with an active zone and a small
    previous step -- opt-in only, matching the global
    ``max_growth_factor``'s own disabled-by-default convention.
    """
    config = _dt_cap_config(
        evection_maximum=0.0, evection_growth_factor=0.0, evection_cooldown_iters=0
    )
    cap = _estimate_evection_dt_cap_yr(Tides_t(), True, 0.5, config)
    assert cap == np.inf

    # Discrimination: the identical scenario WITH a positive growth factor
    # does constrain the cap to dt_prev*factor (0.5*1.3=0.65 yr), so the
    # unconstrained value above follows from the disable switch, not from
    # this scenario never triggering the limiter at all.
    config_on = _dt_cap_config(
        evection_maximum=0.0, evection_growth_factor=1.3, evection_cooldown_iters=0
    )
    cap_on = _estimate_evection_dt_cap_yr(Tides_t(), True, 0.5, config_on)
    assert cap_on == pytest.approx(0.5 * 1.3, rel=1e-9)


@pytest.mark.physics_invariant
def test_estimate_evection_dt_cap_yr_folds_rate_and_growth_caps_via_min():
    """When BOTH the rate cap and the growth-limiter cap would bind, the
    combined function returns the smaller of the two -- confirms the
    two halves are actually folded together, not one silently
    overriding the other.
    """
    config = _dt_cap_config(
        evection_maximum=100.0,
        evection_target_rel_de=0.05,
        evection_de_floor=0.02,
        evection_growth_factor=1.0,
        evection_cooldown_iters=1,
    )
    # Rate cap: e 0.10 -> 0.12 over 10 yr -> 0.05*0.12/0.002 = 3.0 yr.
    tides_o = _tides_with_ecc_history([0.0, 10.0], [0.10, 0.12])
    # Growth cap: dt_prev_actual_yr * 1.0 = 20.0 yr -- looser than the rate cap.
    cap_rate_binds = _estimate_evection_dt_cap_yr(tides_o, True, 20.0, config)
    assert cap_rate_binds == pytest.approx(3.0, rel=1e-9)

    # Growth cap: dt_prev_actual_yr * 1.0 = 1.0 yr -- tighter than the rate cap.
    tides_o2 = _tides_with_ecc_history([0.0, 10.0], [0.10, 0.12])
    cap_growth_binds = _estimate_evection_dt_cap_yr(tides_o2, True, 1.0, config)
    assert cap_growth_binds == pytest.approx(1.0, rel=1e-9)
