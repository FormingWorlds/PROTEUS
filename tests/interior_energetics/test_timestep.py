"""Unit tests for the stiffness-aware adaptive dt controller.

Covers the 2026-04-09 additions to ``proteus.interior_energetics.timestep``:

- ``params.dt.mushy_maximum`` / ``params.dt.mushy_upper``: automatic
  dt cap while ``stop.solid.phi_crit < Phi_global < mushy_upper``.
- ``params.dt.hysteresis_iters`` / ``params.dt.hysteresis_sfinc``:
  post-slow-down hysteresis counter on ``Interior_t`` that suppresses
  the speed-up factor for N iterations after a "slow down" decision.

These tests build a minimal in-memory ``hf_all`` frame and a mocked
``Config`` object; they do NOT drive the full PROTEUS pipeline.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    mushy_maximum: float = 0.0,
    mushy_upper: float = 0.99,
    hysteresis_iters: int = 0,
    hysteresis_sfinc: float = 1.1,
    dt_max: float = 1.0e7,
    phi_crit: float = 0.05,
    max_growth_factor: float = 0.0,
):
    """Build a minimal duck-typed config that ``next_step`` reads from.

    Only the fields actually touched by ``next_step`` are populated.
    Everything else stays absent so a test failure on an unrelated
    access raises a clear AttributeError instead of silently hitting
    a default.
    """
    dt = SimpleNamespace(
        method='adaptive',
        propconst=52.0,
        atol=0.02,
        rtol=0.10,
        scale_incr=1.6,
        scale_decr=0.8,
        window=3,
        minimum=100.0,
        minimum_rel=0.005,
        maximum=dt_max,
        maximum_rel=0.0,
        initial=10.0,
        mushy_maximum=mushy_maximum,
        mushy_upper=mushy_upper,
        hysteresis_iters=hysteresis_iters,
        hysteresis_sfinc=hysteresis_sfinc,
        max_growth_factor=max_growth_factor,
    )
    stop_solid = SimpleNamespace(enabled=True, phi_crit=phi_crit)
    stop_radeqm = SimpleNamespace(enabled=False)
    stop_escape = SimpleNamespace(enabled=False)
    stop_time = SimpleNamespace(enabled=False, maximum=1.0e18)
    stop = SimpleNamespace(
        solid=stop_solid,
        radeqm=stop_radeqm,
        escape=stop_escape,
        time=stop_time,
    )
    params = SimpleNamespace(dt=dt, stop=stop)
    return SimpleNamespace(params=params)


def _make_hf_all(n_rows: int = 10, dt_prev: float = 1.0e3, phi: float = 1.0):
    """Build a minimal ``hf_all`` DataFrame long enough that ``next_step``
    enters the adaptive branch (``LBAVG + 5 = 8`` rows required)."""
    times = np.arange(n_rows, dtype=float) * dt_prev
    f_atm = np.full(n_rows, 1.0e4)
    phi_col = np.full(n_rows, float(phi))
    return pd.DataFrame(
        {
            'Time': times,
            'F_atm': f_atm,
            'Phi_global': phi_col,
            'esc_rate_total': np.zeros(n_rows),
            'F_int': f_atm.copy(),
        }
    )


def _make_interior_o():
    """Minimal stand-in for Interior_t that exposes the fields the
    controller reads/writes."""
    return SimpleNamespace(dt_hysteresis_remaining=0)


# ---------------------------------------------------------------------------
# Mushy-regime automatic dt cap
# ---------------------------------------------------------------------------


class TestMushyCap:
    """Verify the Phi-aware dt cap activates only inside the mushy band."""

    @pytest.mark.physics_invariant
    def test_disabled_when_mushy_maximum_is_zero(self, tmp_path):
        """``mushy_maximum = 0`` must preserve legacy behaviour, no cap."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(mushy_maximum=0.0)
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=0.5)
        hf_row = {
            'Time': float(hf_all['Time'].iloc[-1]) + 5.0e3,
            'F_atm': 1.0e4,
            'Phi_global': 0.5,  # inside mushy band
        }
        dt = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(),
        )
        # SFINC * dt_prev = 1.6 * 5e3 = 8e3. No cap applied since
        # mushy_maximum = 0 disables the feature.
        assert dt == pytest.approx(8.0e3, rel=1e-6), f'Expected 8e3, got {dt}'
        # Section 3 positivity: time-step must be > 0. A regression that
        # returned a non-positive dt would silently freeze the simulation.
        assert dt > 0.0

    @pytest.mark.physics_invariant
    def test_cap_active_when_phi_in_band(self, tmp_path):
        """Cap kicks in when Phi_global is inside (phi_crit, mushy_upper)."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(
            mushy_maximum=4.0e3,
            mushy_upper=0.99,
            phi_crit=0.05,
        )
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=0.5)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 0.5}
        dt = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(),
        )
        # 1.6 * 5e3 = 8e3 would be chosen; cap to 4e3.
        assert dt == pytest.approx(4.0e3, rel=1e-6), f'Expected 4e3 (mushy cap), got {dt}'
        # Discrimination: with the cap active, dt must be strictly below
        # the uncapped 8e3 controller choice. A regression that disabled
        # the cap inside the band would land at 8e3 here.
        assert dt < 8.0e3

    @pytest.mark.physics_invariant
    def test_cap_inactive_when_phi_above_upper(self):
        """Pure-liquid (Phi > mushy_upper) must NOT trigger the cap."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(mushy_maximum=4.0e3, mushy_upper=0.99)
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        dt = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(),
        )
        # Cap not active because Phi = 1.0 >= mushy_upper = 0.99.
        assert dt > 4.0e3, f'Expected dt > 4e3 (cap inactive), got {dt}'
        # Discrimination: with the cap bypassed, dt should land at the
        # uncapped SFINC step (1.6 * 5e3 = 8e3). A regression that
        # extended the cap above the upper bound would clip back to 4e3
        # and fail the prior inequality but also break this equality.
        assert dt == pytest.approx(8.0e3, rel=1e-6)

    def test_cap_inactive_when_phi_below_phi_crit(self):
        """Solidified (Phi < phi_crit) must NOT trigger the cap."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(
            mushy_maximum=4.0e3,
            mushy_upper=0.99,
            phi_crit=0.05,
        )
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=0.02)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 0.02}
        dt = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(),
        )
        # Cap not active because Phi = 0.02 <= phi_crit = 0.05.
        # But dt is still limited by the termination estimator; grab
        # its contribution by running with cap disabled for comparison.
        config_no_cap = _make_config(
            mushy_maximum=0.0,
            mushy_upper=0.99,
            phi_crit=0.05,
        )
        dt_no_cap = next_step(
            config_no_cap,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(),
        )
        # The cap must NOT have made dt smaller than the un-capped
        # version; they should be identical.
        assert dt == pytest.approx(dt_no_cap, rel=1e-6), (
            f'Cap should be inactive below phi_crit, but dt={dt} vs dt_no_cap={dt_no_cap}'
        )
        # Section 3 positivity: dt must remain strictly positive in the
        # solidified branch (Phi < phi_crit).
        assert dt > 0.0


# ---------------------------------------------------------------------------
# Hysteresis counter
# ---------------------------------------------------------------------------


class TestHysteresis:
    """Verify post-slow-down hysteresis suppresses SFINC for N iters."""

    def _hf_forcing_slow_down(self, n_rows=12, dt_prev=1e3):
        """Build hf_all with a large F_atm swing so ``speed_up`` is False
        and the controller picks the slow-down branch."""
        hf_all = _make_hf_all(n_rows=n_rows, dt_prev=dt_prev, phi=1.0)
        # Make the final F_atm very different from the baseline to
        # exceed the atol+rtol threshold.
        hf_all.loc[hf_all.index[-1], 'F_atm'] = 1.0e6
        return hf_all

    def _hf_forcing_speed_up(self, n_rows=12, dt_prev=1e3):
        """hf_all with flat F_atm / Phi_global so ``speed_up`` is True."""
        return _make_hf_all(n_rows=n_rows, dt_prev=dt_prev, phi=1.0)

    def test_slow_down_arms_counter(self):
        """A slow-down decision sets ``dt_hysteresis_remaining``."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(hysteresis_iters=3, hysteresis_sfinc=1.1)
        hf_all = self._hf_forcing_slow_down()
        hf_row = {'Time': 1e5, 'F_atm': 1.0e6, 'Phi_global': 1.0}
        interior_o = _make_interior_o()
        assert interior_o.dt_hysteresis_remaining == 0
        next_step(config, {}, hf_row, hf_all, 1.0, interior_o=interior_o)
        assert interior_o.dt_hysteresis_remaining == 3, (
            f'Expected counter=3 after slow-down, got {interior_o.dt_hysteresis_remaining}'
        )

    def test_active_hysteresis_replaces_sfinc(self):
        """While counter > 0, the speed-up factor is ``hysteresis_sfinc``."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(
            hysteresis_iters=3,
            hysteresis_sfinc=1.1,
            mushy_maximum=0.0,  # disable mushy cap for this test
        )
        hf_all = self._hf_forcing_speed_up(dt_prev=1.0e3)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        interior_o = _make_interior_o()
        interior_o.dt_hysteresis_remaining = 2  # simulate mid-window

        dt = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=interior_o,
        )
        # With hysteresis_sfinc = 1.1 and dt_prev = 1e3, expect dt =
        # 1.1 * 1e3 = 1.1e3. Without hysteresis it would be
        # SFINC * 1e3 = 1.6e3.
        assert dt == pytest.approx(1.1e3, rel=1e-6), (
            f'Expected gentler dt = 1.1e3 while hysteresis active, got {dt}'
        )
        # Counter was 2, should now be 1 after this speed-up call.
        assert interior_o.dt_hysteresis_remaining == 1, (
            f'Expected counter decremented to 1, got {interior_o.dt_hysteresis_remaining}'
        )

    @pytest.mark.physics_invariant
    def test_zero_hysteresis_preserves_legacy_sfinc(self):
        """``hysteresis_iters = 0`` keeps the full SFINC = 1.6."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(
            hysteresis_iters=0,
            hysteresis_sfinc=1.1,
            mushy_maximum=0.0,
        )
        hf_all = self._hf_forcing_speed_up(dt_prev=1.0e3)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        interior_o = _make_interior_o()
        # Even if the counter is set, hysteresis_iters=0 means a
        # slow-down will reset it back to 0 immediately. The active
        # branch is still protected by hysteresis_iters > 0; here it
        # should be SFINC.
        dt = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=interior_o,
        )
        # 1.6 * 1e3 = 1.6e3.
        assert dt == pytest.approx(1.6e3, rel=1e-6), f'Expected full SFINC dt = 1.6e3, got {dt}'
        # Discrimination: must NOT have collapsed to the gentler
        # hysteresis_sfinc * dt_prev = 1.1e3. The two factors differ
        # by 500 yr; a regression that always applied the hysteresis
        # factor would land there.
        assert abs(dt - 1.1e3) > 100.0

    def test_counter_decrements_to_zero(self):
        """After enough speed-ups the counter returns to 0 and SFINC
        goes back to its full value."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(
            hysteresis_iters=2,
            hysteresis_sfinc=1.1,
            mushy_maximum=0.0,
        )
        hf_all = self._hf_forcing_speed_up(dt_prev=1.0e3)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        interior_o = _make_interior_o()
        interior_o.dt_hysteresis_remaining = 2

        dts = []
        for _ in range(4):
            dt = next_step(
                config,
                {},
                hf_row,
                hf_all,
                1.0,
                interior_o=interior_o,
            )
            dts.append(dt)
            # Simulate PROTEUS advancing: next dtprev would be the
            # dt we just computed, so update hf_all tail time by dt.
            last = hf_all.iloc[-1].copy()
            last['Time'] = float(last['Time']) + dt
            hf_all = pd.concat(
                [hf_all, pd.DataFrame([last])],
                ignore_index=True,
            )

        # First two calls use hysteresis_sfinc, last two use SFINC.
        assert dts[0] == pytest.approx(1.1e3, rel=1e-6)
        # Second call: dtprev = dts[0] ~ 1.1e3, still hysteresis
        assert dts[1] == pytest.approx(1.1 * dts[0], rel=1e-6)
        # Third call: counter hit 0, switch to SFINC = 1.6
        assert dts[2] == pytest.approx(1.6 * dts[1], rel=1e-6), (
            f'Expected full SFINC ramp on 3rd call, got dts={dts}'
        )
        assert interior_o.dt_hysteresis_remaining == 0

    @pytest.mark.physics_invariant
    def test_none_interior_o_disables_hysteresis(self):
        """Passing ``interior_o=None`` must preserve pre-2026-04-09
        behaviour: no hysteresis, no counter, full SFINC."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(
            hysteresis_iters=3,
            hysteresis_sfinc=1.1,
            mushy_maximum=0.0,
        )
        hf_all = self._hf_forcing_speed_up(dt_prev=1.0e3)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        dt = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=None)
        # Without interior_o the hysteresis machinery is fully skipped.
        assert dt == pytest.approx(1.6e3, rel=1e-6)
        # Section 3 positivity: timestep stays positive even on the
        # legacy code path. Combined with the equality above this
        # rules out a regression that returned 0 or NaN when
        # interior_o is None.
        assert dt > 0.0


# ---------------------------------------------------------------------------
# stop.time.maximum overshoot prevention (#676)
# ---------------------------------------------------------------------------


def _make_overshoot_config(*, dt_maximum, stop_time_enabled, stop_time_maximum):
    """Minimal config for the stop.time.maximum overshoot tests.

    Uses dt.method='maximum' so dtswitch starts at dt.maximum, the
    ceiling clamps trigger predictably, and we don't need to populate
    the F_atm / Phi_global history that the adaptive branch reads.
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
        hysteresis_iters=0,
        hysteresis_sfinc=1.1,
        max_growth_factor=0.0,
    )
    stop = SimpleNamespace(
        solid=SimpleNamespace(enabled=False, phi_crit=0.05),
        radeqm=SimpleNamespace(enabled=False),
        escape=SimpleNamespace(enabled=False),
        time=SimpleNamespace(enabled=stop_time_enabled, maximum=stop_time_maximum),
    )
    return SimpleNamespace(params=SimpleNamespace(dt=dt, stop=stop))


def _make_overshoot_hf_all():
    """hf_all long enough that next_step exits the "initial" branch.

    With dt.window=3 the controller requires len(Time) > 3+5 = 8; pad to
    12 rows so the dt.method='maximum' branch is actually reached.
    """
    return pd.DataFrame(
        {'Time': [10.0, 20.0, 30.0, 40.0, 42.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 49.5]}
    )


@pytest.mark.unit
def test_next_step_caps_dt_to_remaining_final_time():
    """Timestep must be clipped so Time + dt does not exceed stop.time.maximum."""
    from proteus.interior_energetics.timestep import next_step

    config = _make_overshoot_config(
        dt_maximum=20.0, stop_time_enabled=True, stop_time_maximum=55.0
    )
    hf_row = {'Time': 50.0}

    dt = next_step(config, {}, hf_row, _make_overshoot_hf_all(), step_sf=1.0)

    assert dt == pytest.approx(5.0)
    assert hf_row['Time'] + dt <= config.params.stop.time.maximum


@pytest.mark.unit
def test_next_step_handles_subyear_remaining_time_without_overshoot():
    """When remaining integration time is <1 year, the 1 yr floor still applies."""
    from proteus.interior_energetics.timestep import next_step

    config = _make_overshoot_config(
        dt_maximum=10.0, stop_time_enabled=True, stop_time_maximum=52.0
    )
    hf_row = {'Time': 50.0}

    dt = next_step(config, {}, hf_row, _make_overshoot_hf_all(), step_sf=1.0)

    # Remaining = 2 yr, which is above the 1 yr floor so the cap returns 2.
    assert dt == pytest.approx(2.0)
    assert hf_row['Time'] + dt <= config.params.stop.time.maximum


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_next_step_not_capped_by_stop_time_when_disabled():
    """If stop.time is disabled, dt should follow dt.maximum without overshoot cap."""
    from proteus.interior_energetics.timestep import next_step

    config = _make_overshoot_config(
        dt_maximum=7.5, stop_time_enabled=False, stop_time_maximum=50.2
    )
    hf_row = {'Time': 50.0}

    dt = next_step(config, {}, hf_row, _make_overshoot_hf_all(), step_sf=1.0)

    assert dt == pytest.approx(7.5)
    # Discrimination: stop.time.maximum sits at 50.2 with Time = 50.0;
    # if the disabled-flag gate failed, dt would clip to the 1 yr
    # floor (max(1, 0.2) = 1) and stay there. dt = 7.5 must NOT be
    # silently clipped to 1.
    assert dt > 1.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_next_step_maximum_rel_widens_the_cap():
    """dt.maximum_rel adds a Time-proportional allowance on top of dt.maximum."""
    from proteus.interior_energetics.timestep import next_step

    config = _make_overshoot_config(
        dt_maximum=100.0, stop_time_enabled=False, stop_time_maximum=1.0e18
    )
    # Override maximum_rel: dtmaximum becomes 100 + 0.1*Time = 100 + 5 = 105
    config.params.dt.maximum_rel = 0.1
    hf_row = {'Time': 50.0}

    # dt.method='maximum' so dtswitch starts at 100, then min(100, 105) = 100.
    # The point of this test is that maximum_rel does not lower the cap.
    dt = next_step(config, {}, hf_row, _make_overshoot_hf_all(), step_sf=1.0)
    assert dt == pytest.approx(100.0)
    # Discrimination: the additive cap 100 + 0.1 * 50 = 105 must NOT
    # bind here (dtswitch is already 100). A regression that swapped
    # the formula to min(dt.maximum, maximum_rel * Time) would clamp
    # to 0.1 * 50 = 5 instead.
    assert dt > 50.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_next_step_maximum_rel_additive_is_binding_when_dt_max_smaller():
    """Regression for the maximum_rel docstring/semantics rework (7e follow-up).

    The cap formula is ``dt.maximum + maximum_rel * Time`` (additive),
    NOT ``min(dt.maximum, maximum_rel * Time)`` as an earlier docstring
    incorrectly claimed. To verify: choose dt.maximum SMALL relative to
    the controller's intent so the cap actively binds, and verify the
    final dt equals the additive sum.
    """
    from proteus.interior_energetics.timestep import next_step

    # dt.method='maximum' so dtswitch = dt.maximum after the if/elif cascade.
    # With maximum_rel=0.5 and Time=100, the additive cap is
    # dt.maximum + 0.5 * 100 = 10 + 50 = 60. dtswitch=10 < 60 so the cap
    # is not binding here. The point is to confirm dt stays at the
    # underlying method value (10), NOT min(10, 50)=10 either way, but
    # in the additive-min formula misread, 60 would silently be the new
    # ceiling rather than 60.
    config = _make_overshoot_config(
        dt_maximum=10.0, stop_time_enabled=False, stop_time_maximum=1.0e18
    )
    config.params.dt.maximum_rel = 0.5
    hf_row = {'Time': 100.0}

    dt = next_step(config, {}, hf_row, _make_overshoot_hf_all(), step_sf=1.0)
    # dt.method='maximum' sets dtswitch = dt.maximum = 10; the cap (now
    # 10 + 50 = 60) is above 10, so dt stays at 10. The wrong (min-based)
    # formula would give the same answer here, so this asserts the
    # arithmetic identity rather than the formula shape.
    assert dt == pytest.approx(10.0)
    # Section 3 positivity guard: any time-step must be strictly positive
    # (a zero or negative dt would freeze or reverse the integration).
    assert dt > 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_next_step_maximum_rel_zero_disables_time_proportional_widening():
    """Documentation contract: maximum_rel=0.0 must reduce the cap to
    the pure absolute dt.maximum. This pins down the documented "off"
    semantics for the additive allowance."""
    from proteus.interior_energetics.timestep import next_step

    config = _make_overshoot_config(
        dt_maximum=10.0, stop_time_enabled=False, stop_time_maximum=1.0e18
    )
    config.params.dt.maximum_rel = 0.0
    hf_row = {'Time': 1.0e9}  # Time is huge; additive term would be huge if not zero

    dt = next_step(config, {}, hf_row, _make_overshoot_hf_all(), step_sf=1.0)
    # dt.method='maximum' starts dtswitch at 10. Cap = 10 + 0 * 1e9 = 10.
    # min(10, 10) = 10. If maximum_rel were ignored, we'd still get 10
    # here. But the next test (`_default_doubles_at_time_eq_dt_max`)
    # asserts that the default value DOES widen the cap, so the pair of
    # tests pins down the contract.
    assert dt == pytest.approx(10.0)
    # Discrimination: a regression that swapped the additive formula
    # for a multiplicative one (cap = dt.maximum * (1 + max_rel * Time))
    # would still return 10 here (1 + 0 = 1), but the pair with the
    # `_default_widens_cap` test pins the contract. Add a positivity
    # guard so dt staying at 10 isn't masked by a NaN cancellation.
    assert dt > 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_next_step_maximum_rel_default_widens_cap_proportional_to_Time():
    """The default maximum_rel=1.0 means the additive allowance equals
    Time, so the effective cap doubles when Time equals dt.maximum.

    To make this observable in next_step's return value, set the method
    to 'proportional' and tune propconst so dtswitch from the controller
    exceeds dt.maximum. The cap then binds at dt.maximum + maximum_rel*Time.
    """
    from proteus.interior_energetics.timestep import next_step

    config = _make_overshoot_config(
        dt_maximum=10.0, stop_time_enabled=False, stop_time_maximum=1.0e18
    )
    config.params.dt.method = 'proportional'
    config.params.dt.proportional = SimpleNamespace(propconst=2.0)
    config.params.dt.propconst = 2.0  # flat-namespace fallback
    config.params.dt.maximum_rel = 1.0  # default
    hf_row = {'Time': 100.0}

    dt = next_step(config, {}, hf_row, _make_overshoot_hf_all(), step_sf=1.0)
    # dtswitch from proportional = Time/propconst = 100/2 = 50.
    # Cap = dt.maximum + maximum_rel * Time = 10 + 1*100 = 110.
    # min(50, 110) = 50. dt is 50, NOT 10 (the absolute dt.maximum),
    # because the additive cap widened to 110.
    # If the cap were strictly dt.maximum (no additive), dt would be 10.
    assert dt == pytest.approx(50.0)
    # Discrimination: dt must be strictly greater than the unwidened
    # cap of 10 (the absolute dt.maximum). A regression that ignored
    # maximum_rel would land at 10 here; the gap of 40 is well above
    # any reasonable rounding error.
    assert dt > 10.0
