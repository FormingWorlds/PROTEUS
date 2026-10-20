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
    evection_maximum: float = 0.0,
    hysteresis_iters: int = 0,
    hysteresis_sfinc: float = 1.1,
    dt_max: float = 1.0e7,
    phi_crit: float = 0.05,
    max_growth_factor: float = 0.0,
    bol_scale: float = 1.0,
    bol_scale_start: float | None = None,
    bol_scale_duration: float = 0.0,
    impact_maximum: float = 0.0,
    escape_dt_floor_frac: float = 1.0e-3,
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
        evection_maximum=evection_maximum,
        hysteresis_iters=hysteresis_iters,
        hysteresis_sfinc=hysteresis_sfinc,
        max_growth_factor=max_growth_factor,
        impact_maximum=impact_maximum,
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
    star = SimpleNamespace(
        bol_scale=bol_scale,
        bol_scale_start=bol_scale_start,
        bol_scale_duration=bol_scale_duration,
    )
    escape = SimpleNamespace(step_dt_floor_frac=escape_dt_floor_frac)
    return SimpleNamespace(params=params, star=star, escape=escape)


def _make_hf_all(
    n_rows: int = 10, dt_prev: float = 1.0e3, phi: float = 1.0, age_star: float = 0.0
):
    """Build a minimal ``hf_all`` DataFrame long enough that ``next_step``
    enters the adaptive branch (``LBAVG + 5 = 8`` rows required).

    ``age_star`` is the stellar age the bolometric-scaling clamp measures its
    window against. It is constant down the column because the clamp reads only
    the last row.
    """
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
            'age_star': np.full(n_rows, float(age_star)),
        }
    )


def _make_interior_o(t_next_impact=float('inf')):
    """Minimal stand-in for Interior_t that exposes the fields the
    controller reads/writes. The default impact time is infinite, which
    is what a run with accretion switched off carries."""
    return SimpleNamespace(dt_hysteresis_remaining=0, t_next_impact=t_next_impact)


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
        # Positivity (tests/AGENTS.md, Physics modules): time-step must be > 0. A
        # regression that returned a non-positive dt would silently freeze the simulation.
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
        # Positivity (tests/AGENTS.md, Physics modules): dt must remain strictly positive in the
        # solidified branch (Phi < phi_crit).
        assert dt > 0.0


# ---------------------------------------------------------------------------
# Evection-resonance automatic dt cap
# ---------------------------------------------------------------------------


class TestEvectionCap:
    """Verify ``next_step`` folds a precomputed ``hf_row['evection_dt_cap_yr']``
    into ``dtswitch`` via ``min()``.

    The cap VALUE itself -- both the secular |de/dt| rate cap and the
    evection-scoped growth limiter (with its cooldown counter, now on
    ``tides_o`` rather than ``interior_o``) -- is computed by
    ``proteus.orbit.satellite._estimate_evection_dt_cap_yr`` and exported
    into ``hf_row`` by ``evolve_orbit_satellite`` -- see
    ``tests/orbit/test_satellite.py`` for those cases.
    ``next_step`` has no evection-specific logic left at all: no
    ``in_evection_band``/``near_evection_band`` (removed from the
    helpfile entirely), no ``config.params.dt.evection_maximum`` read,
    no growth-limiter block -- just this one column folded into
    ``min()`` like any other dt cap.
    """

    @pytest.mark.physics_invariant
    def test_cap_folds_precomputed_value_into_dtswitch(self):
        """A finite ``evection_dt_cap_yr`` below the controller's own
        choice wins via ``min()``."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        hf_row = {
            'Time': 1e5,
            'F_atm': 1.0e4,
            'Phi_global': 1.0,
            'evection_dt_cap_yr': 50.0,
        }
        dt = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=_make_interior_o())
        # 1.6 * 5e3 = 8e3 would be chosen; cap to 50.
        assert dt == pytest.approx(50.0, rel=1e-6), f'Expected 50 (evection cap), got {dt}'
        # Discrimination: with the cap active, dt must be strictly below
        # the uncapped 8e3 controller choice.
        assert dt < 8.0e3

    @pytest.mark.physics_invariant
    def test_cap_inactive_when_value_is_inf_absent_or_zero(self):
        """Three equivalent "no cap" states must all leave ``dtswitch``
        unmodified: an explicit ``np.inf`` (orbit computed "not
        applicable"), the key entirely absent (star-planet-only runs, or
        a satellite model other than ps1d_evec, never write it), and a
        bare ``0.0`` -- the value ``ZeroHelpfileRow()`` initialises every
        registered helpfile column to before orbit's first call, which
        must NOT be read as a genuine (and nonsensical) zero-length dt
        cap.
        """
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        base = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}

        for label, hf_row in (
            ('inf', {**base, 'evection_dt_cap_yr': np.inf}),
            ('absent', dict(base)),
            ('zero_helpfile_row_sentinel', {**base, 'evection_dt_cap_yr': 0.0}),
        ):
            dt = next_step(config, {}, dict(hf_row), hf_all, 1.0, interior_o=_make_interior_o())
            assert dt == pytest.approx(8.0e3, rel=1e-6), f'{label}: expected 8e3, got {dt}'
            # Discrimination: the zero-sentinel case in particular must not
            # collapse dt to (approximately) zero.
            assert dt > 1.0, label

    @pytest.mark.physics_invariant
    def test_cap_not_needed_when_already_below_cap_value(self):
        """A large ``evection_dt_cap_yr`` that the controller's own
        choice is already comfortably below: the cap's inner comparison
        must not fire (dt passes through unmodified), the counterpart to
        ``test_cap_folds_precomputed_value_into_dtswitch`` where it
        does."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        hf_row = {
            'Time': 1e5,
            'F_atm': 1.0e4,
            'Phi_global': 1.0,
            'evection_dt_cap_yr': 1.0e6,
        }
        dt = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=_make_interior_o())
        # 1.6 * 5e3 = 8e3, well below the 1e6 cap: passes through uncapped.
        assert dt == pytest.approx(8.0e3, rel=1e-6), f'Expected 8e3 (cap not needed), got {dt}'
        # Discrimination: strictly below the 1e6 cap value itself, i.e.
        # genuinely uncapped rather than coincidentally clamped to it.
        assert dt < 1.0e6


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
        # Positivity (tests/AGENTS.md, Physics modules) on the legacy path: with the equality
        # above this rules out 0 or NaN when interior_o is None.
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
        evection_maximum=0.0,
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
    star = SimpleNamespace(bol_scale=1.0, bol_scale_start=None, bol_scale_duration=0.0)
    escape = SimpleNamespace(step_dt_floor_frac=1.0e-3)
    return SimpleNamespace(params=SimpleNamespace(dt=dt, stop=stop), star=star, escape=escape)


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
    # Positivity guard (tests/AGENTS.md, Physics modules): any time-step must be strictly
    # positive (a zero or negative dt would freeze or reverse the integration).
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


# ---------------------------------------------------------------------------
# Landing on scheduled giant impacts
# ---------------------------------------------------------------------------


class TestImpactClamp:
    """Verify dt is shortened to land on the next scheduled impact.

    An impact grows the planet and re-melts its mantle, so it has to be
    applied at the state the timeline places it at. The clamp is one-way:
    it may only shorten the step, and it is floored at the minimum step so
    that an imminent impact cannot drive dt to zero.
    """

    @pytest.mark.physics_invariant
    def test_no_pending_impact_leaves_the_step_untouched(self):
        """A run with accretion off carries an infinite impact time.

        This is the path every existing run takes, so it must return the
        controller's own choice unchanged.
        """
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        hf_row = {'Time': 1.0e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}

        dt = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=_make_interior_o())

        # SFINC * dt_prev = 1.6 * 5e3 = 8e3, the uncapped controller step.
        assert dt == pytest.approx(8.0e3, rel=1e-6), f'Expected 8e3, got {dt}'
        assert dt > 0.0

    @pytest.mark.physics_invariant
    def test_step_lands_exactly_on_an_impact_inside_the_step(self):
        """An impact closer than the controller's step pulls dt back to it."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        time_now = 1.0e5
        hf_row = {'Time': time_now, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        t_impact = time_now + 3.0e3

        dt = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(t_next_impact=t_impact),
        )

        # The step ends on the impact, to the precision of the time axis.
        assert hf_row['Time'] + dt == pytest.approx(t_impact, rel=1e-12)
        # Discrimination: the uncapped controller would have chosen 8e3
        # and stepped 5e3 past the impact, so an inactive clamp cannot
        # pass the equality above.
        assert dt < 8.0e3

    def test_a_distant_impact_does_not_lengthen_the_step(self):
        """The clamp is one-way; it must never grow dt.

        A timeline whose next impact is far away has to leave the
        stiffness-aware controller in charge.
        """
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        hf_row = {'Time': 1.0e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}

        dt = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(t_next_impact=1.0e5 + 2.0e4),
        )

        assert dt == pytest.approx(8.0e3, rel=1e-6), f'Expected 8e3, got {dt}'

        # The controller's own step, with no impact scheduled at all. A
        # distant impact must return exactly this, which is what makes the
        # clamp one-way rather than a step the timeline gets to set.
        unclamped = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(t_next_impact=float('inf')),
        )
        assert dt == pytest.approx(unclamped, rel=1e-12), (
            f'a distant impact moved the step from {unclamped} to {dt}, so the '
            'timeline is steering the controller instead of only shortening it'
        )

        # It also stops short of the impact, the property the clamp exists for.
        assert dt <= 2.0e4

    @pytest.mark.physics_invariant
    def test_an_imminent_impact_is_landed_exactly_below_the_minimum_step(self):
        """An impact closer than the minimum step is landed on, not overshot:
        the landing step may go below dtfloor (one short step, then the run
        continues from the impact).
        """
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        time_now = 1.0e5
        hf_row = {'Time': time_now, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        t_impact = time_now + 10.0

        dt = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(t_next_impact=t_impact),
        )

        # dt.minimum + dt.minimum_rel * Time = 600 yr would overshoot by 590 yr.
        assert dt == pytest.approx(10.0, rel=1e-9), f'Expected the 10 yr landing, got {dt}'
        assert hf_row['Time'] + dt == pytest.approx(t_impact, rel=1e-15)

    @pytest.mark.physics_invariant
    def test_impact_maximum_bounds_the_landing_step_below_the_remaining_time(self):
        """A positive ceiling cuts the landing step further than the time
        remaining to the impact would on its own.

        This is the case the ceiling exists for: a coarse phase leaves a
        lot of time remaining when the impact clamp first engages, and
        without a ceiling that whole remaining time becomes the landing
        step.
        """
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(impact_maximum=3.0e3)
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        time_now = 1.0e5
        hf_row = {'Time': time_now, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        t_impact = time_now + 6.0e3

        dt = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(t_next_impact=t_impact),
        )

        assert dt == pytest.approx(3.0e3, rel=1e-9), f'Expected the 3e3 ceiling, got {dt}'
        # Discrimination: the remaining-time clamp alone would land the step
        # at 6e3 (the impact is nearer than the controller's 8e3 choice), so
        # only an active ceiling can produce 3e3 here.
        assert hf_row['Time'] + dt < t_impact

    @pytest.mark.physics_invariant
    def test_impact_maximum_leaves_a_step_that_does_not_reach_the_impact(self):
        """The ceiling bounds the step that reaches the impact, not every step
        while one is pending: a distant impact leaves the controller's step."""
        from proteus.interior_energetics.timestep import next_step

        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        hf_row = {'Time': 1.0e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        steps = [
            next_step(
                _make_config(impact_maximum=cap),
                {},
                dict(hf_row),
                hf_all,
                1.0,
                interior_o=_make_interior_o(t_next_impact=1.0e5 + 2.0e4),
            )
            for cap in (3.0e3, 0.0)
        ]
        assert steps[0] == pytest.approx(steps[1], rel=1e-12)
        assert steps[0] == pytest.approx(8.0e3, rel=1e-6)

    @pytest.mark.physics_invariant
    def test_impact_maximum_caps_a_step_that_ends_exactly_on_the_impact(self):
        """A controller step equal to the time left reaches the impact, so the
        ceiling bounds it (5e3 * the 1.6 growth factor = 8e3 yr to the impact)."""
        from proteus.interior_energetics.timestep import next_step

        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        hf_row = {'Time': 1.0e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        dt = next_step(
            _make_config(impact_maximum=3.0e3),
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(t_next_impact=1.0e5 + 5.0e3 * 1.6),
        )
        # Three equal steps of 8000/3 yr, the last of which lands.
        assert dt == pytest.approx(8.0e3 / 3.0, rel=1e-9)
        assert hf_row['Time'] + dt < 1.0e5 + 8.0e3

    @staticmethod
    def _steps_to(t_impact, cap, growth=0.0, after=0):
        """Iterate next_step from T0 = 1e5 yr (history of 5e3 yr steps) with the
        main loop's snap until the run reaches t_impact, then ``after`` more steps
        with the impact counted on the landing row; return the step ends."""
        from proteus.accretion.common import snap_to_impact
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(impact_maximum=cap, max_growth_factor=growth)
        times = list(1.0e5 + 5.0e3 * np.arange(-11, 1, dtype=float))
        ends, left = [], after
        while times[-1] < t_impact or left > 0:
            pending = t_impact if times[-1] < t_impact else np.inf
            left -= pending == np.inf
            hf_all = _make_hf_all(n_rows=len(times))
            hf_all['Time'] = times
            hf_all['n_impacts_applied'] = [float(t >= t_impact) for t in times]
            hf_row = {'Time': times[-1], 'F_atm': 1.0e4, 'Phi_global': 1.0}
            dt = next_step(
                config, {}, hf_row, hf_all, 1.0, interior_o=_make_interior_o(pending)
            )
            times.append(snap_to_impact(times[-1] + dt, pending))
            ends.append(times[-1])
        return ends

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize('offset', [8.3e3, 1.6e4, 1.62e4, 3.22e4])
    def test_an_impact_under_the_ceiling_is_landed_exactly(self, offset):
        """With impact_maximum = 3000 yr the run lands exactly on the impact, by a
        step no longer than the ceiling, from approaches that end with a remainder
        below the 600 yr floor."""
        t = 1.0e5 + offset
        ends = self._steps_to(t, cap=3.0e3)
        assert ends[-1] == t
        assert ends[-1] - (ends[-2] if len(ends) > 1 else 1.0e5) <= 3.0e3 * (1 + 1e-12)

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize(
        'offset, growth, ends',
        [(8.3e3, 0.0, [108000.0, 108300.0]), (5.6e3, 1.1, [105500.0, 105600.0])],
    )
    def test_a_step_ending_short_of_an_impact_leaves_the_remainder(self, offset, growth, ends):
        """A capped step (8000 yr, or 5500 yr under a 1.1 growth cap) that stops less
        than the 600 yr floor short of an impact is kept; the next step lands exactly
        on the impact, shorter than the floor."""
        got = self._steps_to(1.0e5 + offset, cap=0.0, growth=growth)
        assert got == pytest.approx(ends, rel=1e-15, abs=0.0)
        assert got[-1] - got[-2] < 600.0

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize(
        'rem, ends', [(5.0e-4, [108000.0005]), (2.0e-3, [108000.0, 108000.002])]
    )
    def test_a_remainder_below_the_snapshot_resolution_is_absorbed(self, rem, ends):
        """An 8000 yr step stopping 5e-4 yr short of an impact is extended onto it; a
        2e-3 yr remainder gets its own step. Every row keeps its own snapshot name."""
        from proteus.utils.helper import format_subyear_time

        got = self._steps_to(1.08e5 + rem, cap=0.0)
        assert got == pytest.approx(ends, rel=1e-15, abs=0.0)
        names = [format_subyear_time(t) for t in [1.0e5, *got]]
        assert len(set(names)) == len(names)

    @pytest.mark.physics_invariant
    def test_the_growth_cap_regrows_from_the_floor_after_a_short_landing(self):
        """After a 2e-3 yr landing step under a 1.1 growth cap the next step is the
        627.5 yr floor, and 22 steps of 1.1x growth reach 5000 yr again."""
        ends = self._steps_to(1.055e5 + 2.0e-3, cap=0.0, growth=1.1, after=30)
        steps = np.diff([1.0e5, *ends])
        assert steps[1] == pytest.approx(2.0e-3, rel=1e-6)
        assert steps[2] == pytest.approx(100.0 + 0.005 * 105500.002, rel=1e-9)
        assert int(np.argmax(steps[2:] >= 5.0e3)) == 22

    def test_the_impact_ceiling_bounds_a_step_extended_onto_the_impact(self):
        """With impact_maximum = 100 yr the 8000 yr step that the snap-forward extends
        onto an impact 5e-4 yr further is still cut by the 600 yr ceiling, into 14 steps."""
        ends = self._steps_to(1.08e5 + 5.0e-4, cap=100.0)
        assert ends[-1] == 1.08e5 + 5.0e-4
        assert ends[0] - 1.0e5 == pytest.approx(8000.0005 / 14, rel=1e-12)

    def test_the_growth_base_after_a_landing_is_the_floor_not_the_step_before(self):
        """With the proportional method (Time/52, about 1923 yr) after a 2e-3 yr landing
        that followed 5000 yr steps, the 1.1 growth cap starts from the 600 yr floor."""
        from proteus.interior_energetics.timestep import next_step

        times = [*(1.0e5 + 5.0e3 * np.arange(-10, 1)), 1.0e5 + 2.0e-3]
        hf_all = _make_hf_all(n_rows=len(times))
        hf_all['Time'] = times
        hf_all['n_impacts_applied'] = [0.0] * (len(times) - 1) + [1.0]
        config = _make_config(max_growth_factor=1.1)
        config.params.dt.method = 'proportional'
        hf_row = {'Time': times[-1], 'F_atm': 1.0e4, 'Phi_global': 1.0}
        dt = next_step(config, {}, hf_row, hf_all, 1.0)
        assert dt == pytest.approx(1.1 * (100.0 + 0.005 * times[-1]), rel=1e-9)
        assert dt < times[-1] / 52.0

    @pytest.mark.parametrize(
        'times, n_impacts, expected',
        [
            (
                [*(1.0e5 + 5.0e3 * np.arange(-10, 1)), 1.0e5 + 2e-3, 1.0e5 + 4e-3],
                [0.0] * 11 + [1.0, 2.0],
                1.1 * (100.0 + 0.005 * (1.0e5 + 4e-3)),
            ),
            ([1.0e5, 1.0e5 + 2e-3], [0.0, 1.0], 10.0),
            ([1.0e5, 1.0e5 + 2e-3, 1.0e5 + 4e-3], [0.0, 1.0, 2.0], 10.0),
            (
                [*(1.0e5 + 5.0e3 * np.arange(-10, 1)), 1.0e5 + 227.48, 1.0e5 + 227.482]
                + [1.0e5 + 227.484],
                [0.0] * 12 + [1.0, 2.0],
                1.1 * 227.48,
            ),
        ],
        ids=['two landings', 'second row', 'landings on rows 1 and 2', 'after a short step'],
    )
    def test_the_growth_base_skips_earlier_landings(self, times, n_impacts, expected):
        """After two 2e-3 yr landings in a row the 1.1 growth cap starts from the 600 yr
        floor (proportional method, about 1923 yr), or from a shorter 227.48 yr step
        before them; after landings from the second row on it leaves the 10 yr initial
        step."""
        from proteus.interior_energetics.timestep import next_step

        hf_all = _make_hf_all(n_rows=len(times))
        hf_all['Time'] = times
        hf_all['n_impacts_applied'] = n_impacts
        config = _make_config(max_growth_factor=1.1)
        config.params.dt.method = 'proportional'
        hf_row = {'Time': times[-1], 'F_atm': 1.0e4, 'Phi_global': 1.0}
        dt = next_step(config, {}, hf_row, hf_all, 1.0)
        assert dt == pytest.approx(expected, rel=1e-9)
        assert dt > 1.1 * 2e-3

    def test_the_snap_forward_extends_a_step_held_at_dt_maximum(self):
        """With dt.maximum = 8000 yr, a stop time at 1e9 yr and an impact 5e-4 yr past
        the step end, the step lands on the impact, so the two rows keep different
        snapshot names."""
        from proteus.interior_energetics.timestep import next_step
        from proteus.utils.helper import format_subyear_time

        hf_row = {'Time': 1.0e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        t_impact = 1.08e5 + 5.0e-4
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3)
        config = _make_config(dt_max=8.0e3)
        config.params.stop.time.enabled, config.params.stop.time.maximum = True, 1.0e9
        dt = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(t_impact),
        )
        assert hf_row['Time'] + dt == t_impact
        assert format_subyear_time(hf_row['Time']) != format_subyear_time(t_impact)

    def test_the_growth_base_after_a_landing_keeps_a_ramp_below_the_floor(self):
        """A growth ramp below the 604 yr floor (227.48 yr before a 100 yr landing)
        continues at 1.1 x 227.48 yr after the landing."""
        from proteus.interior_energetics.timestep import next_step

        times = [*(1.0e5 + 5.0e3 * np.arange(-8, 1)), 1.0e5 + 188.0]
        times += [times[-1] + 206.8, times[-1] + 206.8 + 227.48]
        times.append(times[-1] + 100.0)
        hf_all = _make_hf_all(n_rows=len(times))
        hf_all['Time'] = times
        hf_all['n_impacts_applied'] = [0.0] * (len(times) - 1) + [1.0]
        hf_row = {'Time': times[-1], 'F_atm': 1.0e4, 'Phi_global': 1.0}
        dt = next_step(_make_config(max_growth_factor=1.1), {}, hf_row, hf_all, 1.0)
        assert dt == pytest.approx(1.1 * 227.48, rel=1e-9)
        assert dt < 100.0 + 0.005 * hf_row['Time']

    def test_the_snap_forward_extends_a_step_ending_short_of_the_stop_time(self):
        """A step held at dt.maximum ending 2e-4 yr before the stop time and 4e-4 yr
        before an impact is extended onto the impact, not left for a 4e-4 yr step."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(dt_max=8.0e3)
        config.params.stop.time.enabled, config.params.stop.time.maximum = True, 1.08e5 + 2e-4
        hf_row = {'Time': 1.0e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3)
        t_impact = 1.08e5 + 4e-4
        dt = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=_make_interior_o(t_impact))
        assert hf_row['Time'] + dt == t_impact
        assert dt > 8.0e3

    def test_the_snap_forward_does_not_pass_the_stop_time(self):
        """A step ending on the stop time, 5e-4 yr before an impact, is not extended
        onto the impact."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        config.params.stop.time.enabled, config.params.stop.time.maximum = True, 1.03e5
        hf_row = {'Time': 1.0e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3)
        dt = next_step(
            config, {}, hf_row, hf_all, 1.0, interior_o=_make_interior_o(1.03e5 + 5e-4)
        )
        assert dt == pytest.approx(3.0e3, rel=1e-12)
        assert hf_row['Time'] + dt < 1.03e5 + 5e-4

    @pytest.mark.parametrize('n_impacts', [[0.0] * 12, [1.0] * 12])
    def test_a_short_step_without_a_landing_keeps_its_growth_base(self, n_impacts):
        """A 10 yr previous step that did not land on an impact caps the next step
        at 11 yr under a 1.1 growth cap."""
        from proteus.interior_energetics.timestep import next_step

        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3)
        hf_all['Time'] = [*(1.0e5 + 5.0e3 * np.arange(-10, 1)), 1.0e5 + 10.0]
        hf_all['n_impacts_applied'] = n_impacts
        hf_row = {'Time': 1.0e5 + 10.0, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        dt = next_step(_make_config(max_growth_factor=1.1), {}, hf_row, hf_all, 1.0)
        assert dt == pytest.approx(11.0, rel=1e-9)
        assert dt < 100.0 + 0.005 * hf_row['Time']

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize(
        'kwargs, phi, stop_time, step_sf, cap',
        [
            ({'dt_max': 5.0e3}, 1.0, None, 1.0, 5.0e3),
            ({'mushy_maximum': 1.0e3}, 0.5, None, 1.0, 1.0e3),
            ({}, 1.0, None, 0.09, 720.0),
            ({}, 1.0, 1.05e5, 1.0, 5.0e3),
        ],
        ids=['dt_max', 'mushy', 'step_sf', 'stop_time'],
    )
    def test_a_near_impact_keeps_every_earlier_cap(self, kwargs, phi, stop_time, step_sf, cap):
        """dt.maximum, the mushy cap, a retry's step_sf and the stop time each keep
        their step when the impact lies 300 yr, less than the floor, beyond it."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(**kwargs)
        if stop_time is not None:
            config.params.stop.time.enabled, config.params.stop.time.maximum = True, stop_time
        hf_row = {'Time': 1.0e5, 'F_atm': 1.0e4, 'Phi_global': phi}
        dt = next_step(
            config,
            {},
            hf_row,
            _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=phi),
            step_sf,
            interior_o=_make_interior_o(1.0e5 + cap + 300.0),
        )
        assert dt == pytest.approx(cap, rel=1e-12)
        assert hf_row['Time'] + dt < 1.0e5 + cap + 300.0

    def test_the_static_step_is_not_changed_by_a_near_impact(self):
        """In the init stage (Time < 2 yr) the 1 yr step stays 1 yr with an impact
        500 yr ahead, inside a floor of 1e4 yr, as with a distant one."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        config.params.dt.minimum = 1.0e4
        steps = [
            next_step(
                config,
                {},
                {'Time': 0.0, 'F_atm': 1.0e4, 'Phi_global': 1.0},
                _make_hf_all(n_rows=1),
                1.0,
                interior_o=_make_interior_o(t),
            )
            for t in (500.0, 1.0e9)
        ]
        assert steps[0] == pytest.approx(1.0, rel=1e-12)
        assert steps[0] == pytest.approx(steps[1], rel=1e-12)

    def test_impact_maximum_does_not_shorten_a_step_already_below_it(self):
        """The ceiling never lengthens the step and stays inert once the
        remaining-time clamp has already produced something smaller."""
        from proteus.interior_energetics.timestep import next_step

        time_now = 1.0e5
        hf_row = {'Time': time_now, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        t_impact = time_now + 2.0e3
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)

        dt_with_ceiling = next_step(
            _make_config(impact_maximum=5.0e3),
            {},
            dict(hf_row),
            hf_all,
            1.0,
            interior_o=_make_interior_o(t_next_impact=t_impact),
        )
        dt_without_ceiling = next_step(
            _make_config(impact_maximum=0.0),
            {},
            dict(hf_row),
            hf_all,
            1.0,
            interior_o=_make_interior_o(t_next_impact=t_impact),
        )

        assert dt_with_ceiling == pytest.approx(dt_without_ceiling, rel=1e-12)
        assert dt_with_ceiling == pytest.approx(2.0e3, rel=1e-9)

    @pytest.mark.physics_invariant
    def test_impact_maximum_never_beats_the_minimum_floor(self):
        """A ceiling set below the minimum-step floor must not win.

        A misconfigured ceiling smaller than the floor must not split the
        approach into steps that collapse towards zero: the split uses the
        600 yr floor, so a 2000 yr approach takes four 500 yr steps.
        """
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(impact_maximum=50.0)
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        time_now = 1.0e5
        hf_row = {'Time': time_now, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        t_impact = time_now + 2000.0

        dt = next_step(
            config,
            {},
            hf_row,
            hf_all,
            1.0,
            interior_o=_make_interior_o(t_next_impact=t_impact),
        )

        # ceil(2000 / 600) = 4 equal steps, not ceil(2000 / 50) = 40.
        assert dt == pytest.approx(500.0, rel=1e-9), f'Expected 500 yr, got {dt}'
        assert dt > 50.0


class TestImpactAndBolscaleClampsTogether:
    """Verify the step when a giant impact and a stellar-scaling edge compete.

    Two independent events can shorten the same step: a scheduled impact,
    and the moment the bolometric scaling of the stellar flux switches on or
    off. Both caps only ever shorten dt, so the nearer event decides where
    the step lands, and the run reaches the further one on a later step.

    The controller's own choice throughout is 1.6 * 5e3 = 8e3 yr, so every
    value asserted below is well clear of it and of the other event's time.
    """

    # Window start in Gyr; the clamp reads it as 5.0e8 yr of stellar age.
    BOL_START_GYR = 0.5
    AGE_INI_YR = 5.0e8
    TIME_NOW = 1.0e5
    CONTROLLER_DT = 8.0e3

    def _setup(self, dt_to_edge, t_next_impact):
        """Place the scaling edge ``dt_to_edge`` years ahead and schedule an
        impact, returning everything ``next_step`` needs."""
        config = _make_config(
            bol_scale=2.0,
            bol_scale_start=self.BOL_START_GYR,
            bol_scale_duration=0.5,
        )
        hf_all = _make_hf_all(
            n_rows=12, dt_prev=5.0e3, phi=1.0, age_star=self.AGE_INI_YR - dt_to_edge
        )
        hf_row = {'Time': self.TIME_NOW, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        interior_o = _make_interior_o(t_next_impact=t_next_impact)
        return config, hf_all, hf_row, interior_o

    @pytest.mark.physics_invariant
    def test_the_nearer_impact_wins_and_the_step_stops_on_it(self):
        """An impact closer than the scaling edge decides the step.

        The impact has to be applied at the state the timeline places it at,
        because it grows the planet and re-melts its mantle. The scaling edge
        carries no such requirement: it is a property of stellar age alone and
        is recovered on the following step.
        """
        from proteus.interior_energetics.timestep import next_step

        dt_to_edge = 6.0e3
        t_impact = self.TIME_NOW + 3.0e3
        config, hf_all, hf_row, interior_o = self._setup(dt_to_edge, t_impact)

        dt = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=interior_o)

        # The step ends on the impact, not on the scaling edge.
        assert hf_row['Time'] + dt == pytest.approx(t_impact, rel=1e-12)
        # Neither an inert impact cap (landing at 6e3) nor the unconstrained
        # controller (8e3) produces the 3e3 step target.
        assert dt < dt_to_edge
        assert dt < self.CONTROLLER_DT
        # The invariant the impact cap exists for: never step past the impact.
        assert hf_row['Time'] + dt <= t_impact

    @pytest.mark.physics_invariant
    def test_the_nearer_scaling_edge_wins_without_overshooting_the_impact(self):
        """A scaling edge closer than the impact decides the step instead.

        The step lands on the edge, and because the impact is further away it
        is still ahead of the run rather than stepped over, which is what the
        one-way nature of both caps guarantees.
        """
        from proteus.interior_energetics.timestep import next_step

        dt_to_edge = 4.0e3
        t_impact = self.TIME_NOW + 6.0e3
        config, hf_all, hf_row, interior_o = self._setup(dt_to_edge, t_impact)

        dt = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=interior_o)

        assert dt == pytest.approx(dt_to_edge, rel=1e-9)
        # The impact is still pending, which is the property that fails if the
        # controller's 8e3 step were to survive: it would overshoot by 2e3.
        assert hf_row['Time'] + dt < t_impact
        assert dt < self.CONTROLLER_DT

    def test_the_clamp_flag_tracks_the_scaling_edge_and_not_the_impact(self):
        """The flag the main loop reads reports the scaling edge alone.

        The main loop forces an off-cadence stellar refresh whenever the flag
        is raised. It must therefore follow the bolometric window and not any
        other cap, or every impact would defeat the refresh cadence.

        When the impact cap then pulls the step short of the edge the flag
        stays raised, and the extra refresh that causes is harmless: the
        scaling factor is a function of stellar age, so recomputing it early
        returns the same pre-edge value the run already had.
        """
        from proteus.interior_energetics.timestep import next_step

        t_impact = self.TIME_NOW + 3.0e3

        # Edge near, impact nearer: the edge bound the step before the impact
        # cap moved it, so the flag is raised.
        config, hf_all, hf_row, interior_o = self._setup(6.0e3, t_impact)
        next_step(config, {}, hf_row, hf_all, 1.0, interior_o=interior_o)
        assert interior_o.timestep_clamped is True

        # With a distant scaling edge, the impact sets dt while the
        # clamp flag remains False since the scaling edge was not reached.
        config, hf_all, hf_row, interior_o = self._setup(4.0e8, t_impact)
        dt = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=interior_o)
        assert interior_o.timestep_clamped is False
        assert hf_row['Time'] + dt == pytest.approx(t_impact, rel=1e-12)


class TestEscapeStepLimit:
    """The controller must honour a shorter step requested by a capped escape
    step, without letting it override the run's own floor."""

    @pytest.mark.physics_invariant
    def test_next_step_shortens_dt_to_the_escape_limit(self):
        """A capped escape step hands the controller a maximum next step, and the
        controller must take it rather than the value its own error estimate
        wanted.

        The limit is set an order of magnitude below the unconstrained step, so a
        controller that ignored it would return a value ten times larger, which no
        tolerance could confuse with the limited one.
        """
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        hf_all = _make_hf_all(n_rows=10, dt_prev=1.0e3)
        hf_row = {'Time': float(hf_all['Time'].iloc[-1]), 'Phi_global': 1.0}

        interior_o = _make_interior_o()
        interior_o.escape_dt_limit = np.inf
        unconstrained = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=interior_o)

        interior_o.escape_dt_limit = unconstrained / 10.0
        limited = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=interior_o)

        assert limited == pytest.approx(unconstrained / 10.0, rel=1e-9)
        # Positivity: a limited step is still a usable forward step.
        assert limited > 0.0
        # Discrimination guard: ignoring the limit returns the unconstrained
        # value, ten times larger than the assertion above admits.
        assert limited < 0.2 * unconstrained

    @pytest.mark.unit
    def test_escape_limit_does_not_breach_the_configured_minimum(self):
        """An escape limit may take the step below `dt.minimum`, but only as far
        as the escape floor, so a runaway overshoot slows the run instead of
        stopping it dead.

        The ordinary floor would otherwise override the shortened step on almost
        every step the cap binds on, leaving the reduction inert exactly where it
        is needed. Edge case: the limit is set far below both floors, the regime
        a runaway overshoot produces, and the escape floor has to win.
        """
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        hf_all = _make_hf_all(n_rows=10, dt_prev=1.0e3)
        hf_row = {'Time': float(hf_all['Time'].iloc[-1]), 'Phi_global': 1.0}
        dt_min = config.params.dt.minimum
        esc_floor = dt_min * config.escape.step_dt_floor_frac

        interior_o = _make_interior_o()
        interior_o.escape_dt_limit = 1.0e-9  # far below the escape floor

        limited = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=interior_o)

        assert limited == pytest.approx(esc_floor, rel=1e-9)
        # It went below the ordinary floor, which is the whole point.
        assert limited < dt_min
        # Positivity: a bounded reduction is still a usable forward step, and it
        # is 1000x the raw limit, so a pass-through would be plainly separated.
        assert limited > 1.0e-9
        assert limited == pytest.approx(dt_min / 1000.0, rel=1e-9)

    @pytest.mark.unit
    def test_the_schema_defaults_give_a_floor_of_100_yr_and_an_escape_floor_of_a_tenth(self):
        """With the schema defaults of `params.dt.minimum`, `params.dt.minimum_rel` and
        `escape.step_dt_floor_frac`, an adaptive request below the floor (30 yr steps,
        so at most 48 yr) gives `100 + 1e-5 * Time` yr, and an escape limit far below
        both floors gives 0.1 yr."""
        from proteus.config._escape import Escape
        from proteus.config._params import TimeStepParams
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(escape_dt_floor_frac=Escape().step_dt_floor_frac)
        config.params.dt.minimum = TimeStepParams().minimum
        config.params.dt.minimum_rel = TimeStepParams().minimum_rel
        hf_all = _make_hf_all(n_rows=10, dt_prev=30.0)
        hf_row = {'Time': float(hf_all['Time'].iloc[-1]), 'Phi_global': 1.0}

        floored = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=_make_interior_o())
        assert floored == pytest.approx(100.0 + 1.0e-5 * hf_row['Time'], rel=1e-12)

        interior_o = _make_interior_o()
        interior_o.escape_dt_limit = 1.0e-9
        limited = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=interior_o)
        assert limited == pytest.approx(0.1, rel=1e-12)

    @pytest.mark.unit
    def test_absent_escape_limit_leaves_the_step_untouched(self):
        """An interior object that never saw an escape step must step exactly as
        it did before, so the limit is inert on every run where escape is off.

        Exercises both shapes the controller can meet: the attribute missing
        entirely, and the attribute present but infinite.
        """
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        hf_all = _make_hf_all(n_rows=10, dt_prev=1.0e3)
        hf_row = {'Time': float(hf_all['Time'].iloc[-1]), 'Phi_global': 1.0}

        bare = _make_interior_o()  # no escape_dt_limit attribute at all
        without = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=bare)

        explicit = _make_interior_o()
        explicit.escape_dt_limit = np.inf
        with_inf = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=explicit)

        assert without == pytest.approx(with_inf, rel=1e-12)
        assert without > 0.0

    @pytest.mark.unit
    def test_escape_limit_applies_in_the_static_and_initial_regimes(self):
        """Escape can be running before the controller reaches its adaptive
        branch, so the limit must bind in the early regimes too.

        The static branch fixes dt at 1.0 yr and the initial branch at
        `dt.initial`; a limit below both must win in each. The value used is
        neither of those two fixed numbers, so a branch that ignores the limit
        returns its own constant and is separated by three orders of magnitude.
        """
        from proteus.interior_energetics.timestep import next_step

        config = _make_config()
        interior_o = _make_interior_o()
        interior_o.escape_dt_limit = 0.001  # differs from 1.0 and dt.initial

        static = next_step(
            config, {}, {'Time': 1.0, 'Phi_global': 1.0}, None, 1.0, interior_o=interior_o
        )
        short = _make_hf_all(n_rows=4, dt_prev=1.0e3)
        initial = next_step(
            config,
            {},
            {'Time': float(short['Time'].iloc[-1]), 'Phi_global': 1.0},
            short,
            1.0,
            interior_o=interior_o,
        )

        esc_floor = config.params.dt.minimum * config.escape.step_dt_floor_frac
        assert static == pytest.approx(esc_floor, rel=1e-9)
        assert initial == pytest.approx(esc_floor, rel=1e-9)
        # Discrimination guard: ignoring the limit returns 1.0 and dt.initial
        # (10.0), both far above the escape floor of 0.1.
        assert static < 1.0
        assert initial < config.params.dt.initial

    @pytest.mark.unit
    def test_boundary_backend_forwards_the_interior_to_the_controller(self):
        """The boundary interior module must pass `interior_o` into the
        controller, or an escape overshoot never shortens its next step.

        Checked by call inspection rather than by running the solver, because the
        defect is a dropped argument: the limit is honoured by `next_step` and the
        only question is whether this backend hands it over.
        """
        import inspect

        from proteus.interior_energetics.boundary import BoundaryRunner

        src = inspect.getsource(BoundaryRunner.compute_time_step)
        call = src.split('next_step(')[1]
        assert 'interior_o=interior_o' in call
        # The parameter exists to be forwarded, so its presence in the signature
        # without the forwarding is exactly the silent-failure shape.
        assert 'interior_o' in inspect.signature(BoundaryRunner.compute_time_step).parameters


@pytest.mark.unit
def test_init_stage_step_never_exceeds_the_accretion_resume_horizon():
    """Time stays at zero through the init stage, so every init step comes
    from the static branch. The accretion resume check assumes such a step
    never exceeds its init-stage horizon; this pins the two together.

    The static branch is not floored to dt.minimum, so a minimum far above
    1 yr must leave the init step unchanged, and a pending impact inside the
    step must not lengthen it.
    """
    from proteus.accretion.wrapper import _INIT_STAGE_HORIZON_YR
    from proteus.interior_energetics.timestep import next_step

    config = _make_config()
    config.params.dt.minimum = 50.0  # far above the static step
    step = next_step(config, {}, {'Time': 0.0, 'Phi_global': 1.0}, None, 1.0)
    assert step == pytest.approx(_INIT_STAGE_HORIZON_YR, rel=1e-12)

    with_impact = next_step(
        config,
        {},
        {'Time': 0.0, 'Phi_global': 1.0},
        None,
        1.0,
        interior_o=_make_interior_o(t_next_impact=0.5),
    )
    assert 0.0 < with_impact <= _INIT_STAGE_HORIZON_YR
    # A retry shrinks the step; it never grows beyond the horizon.
    retried = next_step(config, {}, {'Time': 0.0, 'Phi_global': 1.0}, None, 0.5)
    assert retried == pytest.approx(0.5 * _INIT_STAGE_HORIZON_YR, rel=1e-12)


def _make_tau_drain_fixture(
    *,
    s_top: float = 2400.0,
    phi_top: float = 0.20,
    P_top: float = 4.46095e8,
    rho_top: float = 4000.0,
    dm_top: float = 1.0e22,
    r_outer: float = 6.371e6,
    cp_live: float = 1800.0,
    cp_solid_at_1800: float = 1232.0,
    cp_at_smin_plus_300: float = 528.9,
    s_min: float = 25.99,
    rel_dT: float = 0.08,
    rel_dT_solid: float = 0.02,
    top_cell_dt_limit: bool = True,
    with_solution: bool = True,
):
    """Build a synthetic Aragog EntropySolver and config for top-cell drain timescale tests."""
    vol_top = dm_top / rho_top if rho_top > 0.0 else 2.5e18

    def _cp_fn(p_arr, s_arr):
        s_val = float(np.asarray(s_arr, dtype=float).ravel()[0])
        if abs(s_val - 1800.0) < 1e-6:
            return np.array([cp_solid_at_1800], dtype=float)
        if abs(s_val - (s_min + 300.0)) < 1e-6:
            return np.array([cp_at_smin_plus_300], dtype=float)
        return np.array([cp_live], dtype=float)

    eos = SimpleNamespace(
        S_min=s_min,
        density=lambda p_arr, s_arr: np.array([rho_top], dtype=float),
        heat_capacity=_cp_fn,
        melt_fraction=lambda p_arr, s_arr: np.array([phi_top], dtype=float),
        temperature=lambda p_arr, s_arr: np.array([1500.0], dtype=float),
    )
    sol = (
        SimpleNamespace(y=np.array([[2800.0], [s_top]], dtype=float)) if with_solution else None
    )
    solver = SimpleNamespace(
        _r_basic_flat=np.array([5.5e6, 6.0e6, r_outer], dtype=float),
        _volume_flat=np.array([5.0e18, vol_top], dtype=float),
        _P_stag_flat=np.array([1.0e10, P_top], dtype=float),
        _n_stag=2,
        _S0=np.array([2800.0, s_top, 0.0], dtype=float),
        _solution=sol,
        entropy_staggered=np.array([[2800.0], [s_top]], dtype=float),
        entropy_eos=eos,
    )
    config = _make_config()
    config.interior_energetics = SimpleNamespace(
        module='aragog',
        aragog=SimpleNamespace(
            top_cell_dt_limit=top_cell_dt_limit,
            top_cell_dt_rel_dT=rel_dT,
            top_cell_dt_rel_dT_solid=rel_dT_solid,
        ),
    )
    interior_o = _make_interior_o()
    interior_o.aragog_solver = solver
    return config, interior_o, solver


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_estimate_tau_drain_f_atm_near_zero_uses_f_min():
    """Near-zero, zero, or negative F_atm uses max(abs(F_atm), _TAU_DRAIN_F_MIN = 1.0)."""
    from proteus.interior_energetics.timestep import (
        _TAU_DRAIN_F_MIN,
        _estimate_tau_drain,
    )
    from proteus.utils.constants import secs_per_year

    config, interior_o, solver = _make_tau_drain_fixture(
        phi_top=0.0,
        cp_solid_at_1800=1200.0,
        rel_dT=0.08,
        rel_dT_solid=0.02,
    )
    area_s = 4.0 * np.pi * float(solver._r_basic_flat[-1]) ** 2
    dr_eff = float(solver._volume_flat[-1]) / area_s
    expected_yr = (4000.0 * 1200.0 * dr_eff * (0.02 * 1500.0)) / (1.0 * secs_per_year)

    tau_zero = _estimate_tau_drain(config, {'F_atm': 0.0, 'T_surf': 1500.0}, interior_o)
    tau_half = _estimate_tau_drain(config, {'F_atm': 0.5, 'T_surf': 1500.0}, interior_o)
    tau_neg_half = _estimate_tau_drain(config, {'F_atm': -0.5, 'T_surf': 1500.0}, interior_o)
    tau_two = _estimate_tau_drain(config, {'F_atm': 2.0, 'T_surf': 1500.0}, interior_o)
    tau_pos_500 = _estimate_tau_drain(config, {'F_atm': 500.0, 'T_surf': 1500.0}, interior_o)
    tau_neg_500 = _estimate_tau_drain(config, {'F_atm': -500.0, 'T_surf': 1500.0}, interior_o)

    assert _TAU_DRAIN_F_MIN == pytest.approx(1.0, rel=1e-12)
    assert tau_zero == pytest.approx(expected_yr, rel=1e-12)
    assert tau_half == pytest.approx(expected_yr, rel=1e-12)
    assert tau_neg_half == pytest.approx(expected_yr, rel=1e-12)
    assert tau_two == pytest.approx(0.5 * expected_yr, rel=1e-12)
    assert tau_pos_500 == pytest.approx(expected_yr / 500.0, rel=1e-12)
    assert tau_neg_500 == pytest.approx(tau_pos_500, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_estimate_tau_drain_phi_outside_unit_interval_clamped():
    """Negative melt fraction is clamped at 0.0 and melt fraction above 1.0 saturates the ramp at 1.0."""
    from proteus.interior_energetics.timestep import _estimate_tau_drain

    cfg_neg, io_neg, _ = _make_tau_drain_fixture(phi_top=-0.15, rel_dT=0.08, rel_dT_solid=0.02)
    cfg_zero, io_zero, _ = _make_tau_drain_fixture(phi_top=0.0, rel_dT=0.08, rel_dT_solid=0.02)
    cfg_hi, io_hi, _ = _make_tau_drain_fixture(phi_top=1.25, rel_dT=0.08, rel_dT_solid=0.02)
    cfg_one, io_one, _ = _make_tau_drain_fixture(
        phi_top=1.0, rel_dT=0.08, rel_dT_solid=0.02, with_solution=False
    )

    row = {'F_atm': 500.0, 'T_surf': 1500.0}
    tau_neg = _estimate_tau_drain(cfg_neg, row, io_neg)
    tau_zero = _estimate_tau_drain(cfg_zero, row, io_zero)
    tau_hi = _estimate_tau_drain(cfg_hi, row, io_hi)
    tau_one = _estimate_tau_drain(cfg_one, row, io_one)

    assert tau_neg == pytest.approx(tau_zero, rel=1e-12)
    assert tau_hi == pytest.approx(tau_one, rel=1e-12)
    assert tau_one > tau_zero > 0.0


@pytest.mark.unit
def test_estimate_tau_drain_nan_or_invalid_state_logs_once_per_run_and_returns_inf(caplog):
    """Non-finite or non-positive solver state returns inf and logs a warning once per run."""
    import logging

    from proteus.interior_energetics.timestep import _estimate_tau_drain

    config, interior_o, _ = _make_tau_drain_fixture(s_top=float('nan'))
    io_none = _make_interior_o()
    io_none.aragog_solver = None

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics.timestep'):
        assert _estimate_tau_drain(
            config, {'F_atm': 500.0, 'T_surf': 1500.0}, io_none
        ) == float('inf')
        assert [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()] == []
        first = _estimate_tau_drain(config, {'F_atm': 500.0, 'T_surf': 1500.0}, interior_o)
        second = _estimate_tau_drain(config, {'F_atm': 500.0, 'T_surf': 1500.0}, interior_o)

    assert first == float('inf')
    assert second == float('inf')
    warns = [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()]
    assert len(warns) == 1
    assert getattr(interior_o, '_tau_drain_invalid_warned', False) is True


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_estimate_tau_drain_cp_eff_choice_at_phi_0p60_boundary():
    """Above _TAU_DRAIN_PHI_GATE = 0.60, cp_eff uses the live cp_top; at or below 0.60,
    it uses min(cp_top, cp_solid) to cap the linear solid-liquid heat-capacity blend at
    the solid-phase value across the lower mushy zone."""
    from proteus.interior_energetics.timestep import (
        _TAU_DRAIN_PHI_GATE,
        _estimate_tau_drain,
    )

    cfg_low, io_low, _ = _make_tau_drain_fixture(
        phi_top=0.59, cp_live=2400.0, cp_solid_at_1800=1200.0
    )
    cfg_edge, io_edge, _ = _make_tau_drain_fixture(
        phi_top=0.60, cp_live=2400.0, cp_solid_at_1800=1200.0
    )
    cfg_high, io_high, _ = _make_tau_drain_fixture(
        phi_top=0.61, cp_live=2400.0, cp_solid_at_1800=1200.0
    )
    cfg_cp_lt_solid, io_cp_lt_solid, _ = _make_tau_drain_fixture(
        phi_top=0.59, cp_live=900.0, cp_solid_at_1800=1200.0
    )

    row = {'F_atm': 500.0, 'T_surf': 1500.0}
    tau_low = _estimate_tau_drain(cfg_low, row, io_low)
    tau_edge = _estimate_tau_drain(cfg_edge, row, io_edge)
    tau_high = _estimate_tau_drain(cfg_high, row, io_high)
    tau_cp_lt_solid = _estimate_tau_drain(cfg_cp_lt_solid, row, io_cp_lt_solid)

    assert _TAU_DRAIN_PHI_GATE == pytest.approx(0.60, rel=1e-12)
    assert tau_low == pytest.approx(tau_edge, rel=1e-12)
    assert tau_high == pytest.approx(2.0 * tau_low, rel=1e-12)
    assert tau_cp_lt_solid == pytest.approx(0.75 * tau_low, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_estimate_tau_drain_eta_t_linear_ramp_and_0p40_endpoint():
    """Linear ramp between rel_dT_solid at phi=0 and rel_dT at _TAU_DRAIN_PHI_RAMP = 0.40."""
    from proteus.interior_energetics.timestep import (
        _TAU_DRAIN_PHI_RAMP,
        _estimate_tau_drain,
    )

    cfg_0, io_0, _ = _make_tau_drain_fixture(
        phi_top=0.0, cp_live=1200.0, cp_solid_at_1800=1200.0, rel_dT=0.08, rel_dT_solid=0.02
    )
    cfg_mid, io_mid, _ = _make_tau_drain_fixture(
        phi_top=0.20, cp_live=1200.0, cp_solid_at_1800=1200.0, rel_dT=0.08, rel_dT_solid=0.02
    )
    cfg_end, io_end, _ = _make_tau_drain_fixture(
        phi_top=0.40, cp_live=1200.0, cp_solid_at_1800=1200.0, rel_dT=0.08, rel_dT_solid=0.02
    )
    cfg_above, io_above, _ = _make_tau_drain_fixture(
        phi_top=0.55, cp_live=1200.0, cp_solid_at_1800=1200.0, rel_dT=0.08, rel_dT_solid=0.02
    )

    row = {'F_atm': 500.0, 'T_surf': 1500.0}
    tau_0 = _estimate_tau_drain(cfg_0, row, io_0)
    tau_mid = _estimate_tau_drain(cfg_mid, row, io_mid)
    tau_end = _estimate_tau_drain(cfg_end, row, io_end)
    tau_above = _estimate_tau_drain(cfg_above, row, io_above)

    assert _TAU_DRAIN_PHI_RAMP == pytest.approx(0.40, rel=1e-12)
    assert tau_mid == pytest.approx(2.5 * tau_0, rel=1e-12)
    assert tau_end == pytest.approx(4.0 * tau_0, rel=1e-12)
    assert tau_above == pytest.approx(tau_end, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_estimate_tau_drain_solid_s_cap_1800_choice():
    """With S_top = 2400.0 > 1800.0, the solid probe caps at _TAU_DRAIN_SOLID_S_CAP = 1800.0."""
    from proteus.interior_energetics.timestep import (
        _TAU_DRAIN_SOLID_S_CAP,
        _estimate_tau_drain,
    )
    from proteus.utils.constants import secs_per_year

    seen_s: list[float] = []
    cfg, io, solver = _make_tau_drain_fixture(
        s_top=2400.0,
        phi_top=0.20,
        s_min=106.11618925017348,
        rel_dT=0.08,
        rel_dT_solid=0.02,
    )

    def _sloped_cp(p_arr, s_arr):
        s_val = float(np.asarray(s_arr, dtype=float).ravel()[0])
        seen_s.append(s_val)
        if s_val > 2200.0:
            return np.array([2600.0], dtype=float)
        return np.array([512.0546103723684 + 0.4 * s_val], dtype=float)

    io.aragog_solver.entropy_eos.heat_capacity = _sloped_cp
    tau = _estimate_tau_drain(cfg, {'F_atm': 500.0, 'T_surf': 1500.0}, io)

    area_s = 4.0 * np.pi * float(solver._r_basic_flat[-1]) ** 2
    dr_eff = float(solver._volume_flat[-1]) / area_s
    cp_expected_at_1800 = 512.0546103723684 + 0.4 * 1800.0
    eta_eff = 0.02 + (0.08 - 0.02) * (0.20 / 0.40)
    expected_tau = (4000.0 * cp_expected_at_1800 * dr_eff * (eta_eff * 1500.0)) / (
        500.0 * secs_per_year
    )

    assert _TAU_DRAIN_SOLID_S_CAP == pytest.approx(1800.0, rel=1e-12)
    assert seen_s == [pytest.approx(2400.0, rel=1e-12), pytest.approx(1800.0, rel=1e-12)]
    assert tau == pytest.approx(expected_tau, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_estimate_tau_drain_solid_s_offset_300_choice():
    """With S_top < S_min + 300.0, the solid probe floors at S_min + 300.0 on both PALEOS tables."""
    from proteus.interior_energetics.timestep import (
        _TAU_DRAIN_SOLID_S_OFFSET,
        _estimate_tau_drain,
    )
    from proteus.utils.constants import secs_per_year

    for s_min_table, cp_base in (
        (106.11618925017348, 823.7276607666909),
        (25.992306518523165, 528.8995498470755),
    ):
        seen_s: list[float] = []
        cfg, io, solver = _make_tau_drain_fixture(
            s_top=150.0,
            phi_top=0.0,
            s_min=s_min_table,
            rel_dT=0.08,
            rel_dT_solid=0.02,
        )
        s_probe_expected = s_min_table + 300.0

        def _sloped_cp(
            p_arr,
            s_arr,
            _s_ref=s_probe_expected,
            _cp_ref=cp_base,
            _seen=seen_s,
        ):
            s_val = float(np.asarray(s_arr, dtype=float).ravel()[0])
            _seen.append(s_val)
            if abs(s_val - 150.0) < 1e-9:
                return np.array([1500.0], dtype=float)
            return np.array([_cp_ref + 1.25 * (s_val - _s_ref)], dtype=float)

        io.aragog_solver.entropy_eos.heat_capacity = _sloped_cp
        tau = _estimate_tau_drain(cfg, {'F_atm': 500.0, 'T_surf': 1500.0}, io)

        area_s = 4.0 * np.pi * float(solver._r_basic_flat[-1]) ** 2
        dr_eff = float(solver._volume_flat[-1]) / area_s
        expected_tau = (4000.0 * cp_base * dr_eff * (0.02 * 1500.0)) / (500.0 * secs_per_year)

        assert _TAU_DRAIN_SOLID_S_OFFSET == pytest.approx(300.0, rel=1e-12)
        assert seen_s == [
            pytest.approx(150.0, rel=1e-12),
            pytest.approx(s_probe_expected, rel=1e-12),
        ]
        assert tau == pytest.approx(expected_tau, rel=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    'target,attr_name,with_solution',
    [
        ('solver', '_solution', True),
        ('solver', '_S0', False),
        ('solver', '_n_stag', False),
        ('solver', '_r_basic_flat', True),
        ('solver', '_volume_flat', True),
        ('solver', '_P_stag_flat', True),
        ('solver', 'entropy_staggered', True),
        ('solver', 'entropy_eos', True),
        ('eos', 'S_min', True),
        ('eos', 'density', True),
        ('eos', 'heat_capacity', True),
        ('eos', 'melt_fraction', True),
        ('eos', 'temperature', True),
    ],
)
def test_estimate_tau_drain_missing_aragog_attr_warns_once_and_returns_inf(
    caplog, target, attr_name, with_solution
):
    """Missing any required Aragog solver or EOS attribute returns inf and logs one warning."""
    import logging

    from proteus.interior_energetics.timestep import (
        _ARAGOG_EOS_ATTRS,
        _ARAGOG_SOLVER_ATTRS,
        _estimate_tau_drain,
    )

    assert set(_ARAGOG_SOLVER_ATTRS) == {
        '_solution',
        '_S0',
        '_n_stag',
        '_r_basic_flat',
        '_volume_flat',
        '_P_stag_flat',
    }
    assert set(_ARAGOG_EOS_ATTRS) == {
        'S_min',
        'density',
        'heat_capacity',
        'melt_fraction',
        'temperature',
    }

    cfg, io, solver = _make_tau_drain_fixture(with_solution=with_solution)
    if attr_name == '_solution':
        solver._S0 = None
    elif attr_name == '_S0':
        delattr(solver, '_solution')
    obj = solver if target == 'solver' else solver.entropy_eos
    delattr(obj, attr_name)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics.timestep'):
        first = _estimate_tau_drain(cfg, {'F_atm': 500.0, 'T_surf': 1500.0}, io)
        second = _estimate_tau_drain(
            config=cfg, hf_row={'F_atm': 500.0, 'T_surf': 1500.0}, interior_o=io
        )

    assert first == float('inf')
    assert second == float('inf')
    warns = [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()]
    assert len(warns) == 1
    assert getattr(io, '_tau_drain_invalid_warned', False) is True


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_estimate_tau_drain_presolve_without_solution_attr_and_gradient_core_bc(caplog):
    """Before first solve (_solution attribute absent), _S0 is used cleanly including gradient core_bc."""
    import logging

    from proteus.interior_energetics.timestep import _estimate_tau_drain
    from proteus.utils.constants import secs_per_year

    cfg, io, solver = _make_tau_drain_fixture(
        s_top=1000.0,
        phi_top=0.0,
        s_min=106.116,
        cp_live=1100.0,
        rel_dT=0.08,
        rel_dT_solid=0.02,
        with_solution=False,
    )
    delattr(solver, '_solution')

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics.timestep'):
        tau_presolve = _estimate_tau_drain(cfg, {'F_atm': 500.0, 'T_surf': 1500.0}, io)
        assert [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()] == []

    area_s = 4.0 * np.pi * float(solver._r_basic_flat[-1]) ** 2
    dr_eff = float(solver._volume_flat[-1]) / area_s
    expected_tau = (4000.0 * 1100.0 * dr_eff * (0.02 * 1500.0)) / (500.0 * secs_per_year)
    assert tau_presolve == pytest.approx(expected_tau, rel=1e-12)

    solver._core_bc = 'gradient'
    solver._S0 = np.array([-1.0e-4, -1.0e-4, -1.0e-4, 1000.0], dtype=float)
    seen_s: list[float] = []
    orig_cp = solver.entropy_eos.heat_capacity

    def _rec_cp(p_arr, s_arr):
        seen_s.append(float(np.asarray(s_arr, dtype=float).ravel()[0]))
        return orig_cp(p_arr, s_arr)

    solver.entropy_eos.heat_capacity = _rec_cp
    tau_grad = _estimate_tau_drain(cfg, {'F_atm': 500.0, 'T_surf': 1500.0}, io)
    assert seen_s == [pytest.approx(1000.0, rel=1e-12), pytest.approx(1000.0, rel=1e-12)]
    assert tau_grad == pytest.approx(expected_tau, rel=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    'bad_case',
    ['s_min_none', 't_surf_none', 'f_atm_none', 'p_top_nonpositive'],
)
def test_estimate_tau_drain_none_or_nonpositive_pressure_warns_and_returns_inf(
    caplog, bad_case
):
    """None values in S_min/hf_row or non-positive P_top return inf and log one warning."""
    import logging

    from proteus.interior_energetics.timestep import _estimate_tau_drain

    cfg, io, solver = _make_tau_drain_fixture()
    row = {'F_atm': 500.0, 'T_surf': 1500.0}
    if bad_case == 's_min_none':
        solver.entropy_eos.S_min = None
    elif bad_case == 't_surf_none':
        row['T_surf'] = None
    elif bad_case == 'f_atm_none':
        row['F_atm'] = None
    elif bad_case == 'p_top_nonpositive':
        solver._P_stag_flat = np.array([1.0e10, 0.0], dtype=float)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics.timestep'):
        res = _estimate_tau_drain(cfg, row, io)

    assert res == float('inf')
    warns = [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()]
    assert len(warns) == 1


@pytest.mark.unit
def test_next_step_top_cell_dt_limit_off_never_calls_estimate_tau_drain():
    """With top_cell_dt_limit = False or module != 'aragog', next_step never calls
    _estimate_tau_drain and writes no _tau_drain_cap_valid state on interior_o."""
    from unittest.mock import patch

    from proteus.interior_energetics.timestep import next_step

    cfg, io, _ = _make_tau_drain_fixture(top_cell_dt_limit=False)
    io.__dict__.pop('_tau_drain_cap_valid', None)
    hf_all = _make_hf_all(n_rows=10, dt_prev=100.0)
    hf_row = {
        'Time': float(hf_all['Time'].iloc[-1]),
        'Phi_global': 0.2,
        'F_atm': 800.0,
        'T_surf': 1500.0,
    }

    with patch(
        'proteus.interior_energetics.timestep._estimate_tau_drain',
        side_effect=AssertionError('must not be called when top_cell_dt_limit is False'),
    ) as mock_est:
        dt = next_step(cfg, {}, hf_row, hf_all, 1.0, interior_o=io)

    mock_est.assert_not_called()
    assert dt >= cfg.params.dt.minimum
    assert '_tau_drain_cap_valid' not in io.__dict__

    cfg_spider, io_spider, _ = _make_tau_drain_fixture(top_cell_dt_limit=True)
    cfg_spider.interior_energetics.module = 'spider'
    io_spider.__dict__.pop('_tau_drain_cap_valid', None)
    with patch(
        'proteus.interior_energetics.timestep._estimate_tau_drain',
        side_effect=AssertionError('must not be called when module is spider'),
    ) as mock_spider_est:
        dt_spider = next_step(cfg_spider, {}, hf_row, hf_all, 1.0, interior_o=io_spider)

    mock_spider_est.assert_not_called()
    assert dt_spider >= cfg_spider.params.dt.minimum
    assert '_tau_drain_cap_valid' not in io_spider.__dict__


@pytest.mark.unit
def test_estimate_tau_drain_ic_one_and_pre_atmosphere_return_inf_without_warning_and_preserve_warn_once(
    caplog,
):
    """ic == 1 and pre-atmosphere T_surf <= 0 at Time <= 0 return inf with no warning,
    _estimate_tau_drain is pure with respect to _tau_drain_cap_valid, and one-warning-per-run holds."""
    import logging

    from proteus.interior_energetics.timestep import _estimate_tau_drain

    cfg, io, _ = _make_tau_drain_fixture()
    io.ic = 1
    io.__dict__.pop('_tau_drain_cap_valid', None)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics.timestep'):
        io.ic = 2
        res_pre_atm_neg = _estimate_tau_drain(
            cfg, {'Time': 0.0, 'F_atm': 500.0, 'T_surf': -1.0}, io
        )
        assert res_pre_atm_neg == float('inf')
        assert getattr(io, '_tau_drain_invalid_warned', False) is False
        assert [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()] == []
        assert '_tau_drain_cap_valid' not in io.__dict__

        io.ic = 1
        res_ic1 = _estimate_tau_drain(cfg, {'Time': 10.0, 'F_atm': 500.0, 'T_surf': 1500.0}, io)
        assert res_ic1 == float('inf')
        assert '_tau_drain_cap_valid' not in io.__dict__
        assert getattr(io, '_tau_drain_invalid_warned', False) is False
        assert [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()] == []

        io.ic = 2
        res_pre_atm = _estimate_tau_drain(cfg, {'Time': 0.0, 'F_atm': 500.0, 'T_surf': 0.0}, io)
        assert res_pre_atm == float('inf')
        assert '_tau_drain_cap_valid' not in io.__dict__
        assert getattr(io, '_tau_drain_invalid_warned', False) is False
        assert [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()] == []

        res_t0_valid = _estimate_tau_drain(
            cfg, {'Time': 0.0, 'F_atm': 500.0, 'T_surf': 1500.0}, io
        )
        assert 0.0 < res_t0_valid < float('inf')
        assert '_tau_drain_cap_valid' not in io.__dict__

        res_valid = _estimate_tau_drain(
            cfg, {'Time': 10.0, 'F_atm': 500.0, 'T_surf': 1500.0}, io
        )
        assert 0.0 < res_valid < float('inf')
        assert '_tau_drain_cap_valid' not in io.__dict__

        res_bad_1 = _estimate_tau_drain(cfg, {'Time': 10.0, 'F_atm': 500.0, 'T_surf': 0.0}, io)
        assert res_bad_1 == float('inf')
        assert '_tau_drain_cap_valid' not in io.__dict__
        assert getattr(io, '_tau_drain_invalid_warned', False) is True
        warns_1 = [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()]
        assert len(warns_1) == 1

        res_bad_2 = _estimate_tau_drain(cfg, {'Time': 20.0, 'F_atm': 500.0, 'T_surf': -5.0}, io)
        assert res_bad_2 == float('inf')
        assert '_tau_drain_cap_valid' not in io.__dict__
        assert getattr(io, '_tau_drain_invalid_warned', False) is True
        warns_2 = [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()]
        assert len(warns_2) == 1

    cfg_neg, io_neg, _ = _make_tau_drain_fixture()
    io_neg.ic = 2
    io_neg.__dict__.pop('_tau_drain_cap_valid', None)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics.timestep'):
        caplog.clear()
        res_t0_neg = _estimate_tau_drain(
            cfg_neg, {'Time': 0.0, 'F_atm': 500.0, 'T_surf': -1.0}, io_neg
        )
        assert res_t0_neg == float('inf')
        assert '_tau_drain_cap_valid' not in io_neg.__dict__
        assert getattr(io_neg, '_tau_drain_invalid_warned', False) is False
        assert [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()] == []
        res_t10_neg = _estimate_tau_drain(
            cfg_neg, {'Time': 10.0, 'F_atm': 500.0, 'T_surf': -1.0}, io_neg
        )
        assert res_t10_neg == float('inf')
        assert getattr(io_neg, '_tau_drain_invalid_warned', False) is True
        warns_neg = [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()]
        assert len(warns_neg) == 1


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_estimate_tau_drain_eos_depends_on_p_and_s_at_top_staggered_cell_and_multi_column_entropy():
    """EOS lookups evaluate at P_stag[-1] and the last column of entropy_staggered[-1, -1]."""
    from proteus.interior_energetics.timestep import _estimate_tau_drain
    from proteus.utils.constants import secs_per_year

    cfg, io, solver = _make_tau_drain_fixture(rel_dT=0.08, rel_dT_solid=0.02)
    p0 = 1.0e10
    p_top = 4.46095e8
    s_col0_top = 3100.0
    s_top = 2400.0
    solver._P_stag_flat = np.array([p0, p_top], dtype=float)
    solver.entropy_staggered = np.array([[2900.0, 2800.0], [s_col0_top, s_top]], dtype=float)
    solver._solution = SimpleNamespace(y=solver.entropy_staggered.copy())

    def _rho_fn(p_arr, s_arr):
        p_val = float(np.asarray(p_arr, dtype=float).ravel()[0])
        s_val = float(np.asarray(s_arr, dtype=float).ravel()[0])
        return np.array([3500.0 + 1.0e-6 * p_val + 0.2 * s_val], dtype=float)

    def _cp_fn(p_arr, s_arr):
        p_val = float(np.asarray(p_arr, dtype=float).ravel()[0])
        s_val = float(np.asarray(s_arr, dtype=float).ravel()[0])
        return np.array([900.0 + 2.0e-7 * p_val + 0.35 * s_val], dtype=float)

    def _phi_fn(p_arr, s_arr):
        p_val = float(np.asarray(p_arr, dtype=float).ravel()[0])
        s_val = float(np.asarray(s_arr, dtype=float).ravel()[0])
        return np.array(
            [np.clip((s_val - 1800.0) / (3000.0 + 1.0e-7 * p_val - 1800.0), 0.0, 1.0)],
            dtype=float,
        )

    solver.entropy_eos.density = _rho_fn
    solver.entropy_eos.heat_capacity = _cp_fn
    solver.entropy_eos.melt_fraction = _phi_fn

    row = {'Time': 50.0, 'F_atm': 500.0, 'T_surf': 1500.0}
    tau = _estimate_tau_drain(cfg, row, io)

    area_s = 4.0 * np.pi * float(solver._r_basic_flat[-1]) ** 2
    dr_eff = float(solver._volume_flat[-1]) / area_s

    def _analytic_tau(p_eval: float, s_eval: float, swap_args: bool = False) -> float:
        if swap_args:
            rho = float(_rho_fn([s_eval], [p_eval])[0])
            cp_live = float(_cp_fn([s_eval], [p_eval])[0])
            cp_sol = float(_cp_fn([1800.0], [p_eval])[0])
            phi = float(_phi_fn([s_eval], [p_eval])[0])
        else:
            rho = float(_rho_fn([p_eval], [s_eval])[0])
            cp_live = float(_cp_fn([p_eval], [s_eval])[0])
            cp_sol = float(_cp_fn([p_eval], [1800.0])[0])
            phi = float(_phi_fn([p_eval], [s_eval])[0])
        phi_c = max(0.0, phi)
        cp_eff = min(cp_live, cp_sol) if phi_c <= 0.60 else cp_live
        eta_eff = 0.02 + (0.08 - 0.02) * min(phi_c / 0.40, 1.0)
        return (rho * cp_eff * dr_eff * (eta_eff * 1500.0)) / (500.0 * secs_per_year)

    expected = _analytic_tau(p_top, s_top)
    wrong_p0 = _analytic_tau(p0, s_top)
    wrong_col0 = _analytic_tau(p_top, s_col0_top)
    wrong_swapped = _analytic_tau(p_top, s_top, swap_args=True)

    assert tau == pytest.approx(expected, rel=1e-12)
    assert tau != pytest.approx(wrong_p0, rel=1e-2)
    assert tau != pytest.approx(wrong_col0, rel=1e-2)
    assert tau != pytest.approx(wrong_swapped, rel=1e-2)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_next_step_top_cell_dt_limit_on_caps_dtswitch_below_dtminimum():
    """With top_cell_dt_limit = True, tau_drain caps dtswitch below dt.minimum down to esc_floor,
    and _tau_drain_cap_valid resets to False when floored or lengthened by an impact snap-forward."""
    from proteus.interior_energetics.timestep import _estimate_tau_drain, next_step

    cfg, io, _ = _make_tau_drain_fixture(
        phi_top=0.05,
        dm_top=2.0e21,
        cp_live=1200.0,
        cp_solid_at_1800=1200.0,
        rel_dT=0.08,
        rel_dT_solid=0.02,
        top_cell_dt_limit=True,
    )
    cfg.params.dt.minimum = 100.0
    cfg.escape.step_dt_floor_frac = 1.0e-3
    hf_all = _make_hf_all(n_rows=10, dt_prev=100.0)
    hf_row = {
        'Time': float(hf_all['Time'].iloc[-1]),
        'Phi_global': 0.05,
        'F_atm': 800.0,
        'T_surf': 1400.0,
    }

    expected_tau = _estimate_tau_drain(cfg, hf_row, io)
    io._tau_drain_cap_valid = False
    dt_on = next_step(cfg, {}, hf_row, hf_all, 1.0, interior_o=io)
    assert getattr(io, '_tau_drain_cap_valid', None) is True

    esc_floor = cfg.params.dt.minimum * cfg.escape.step_dt_floor_frac
    assert dt_on == pytest.approx(expected_tau, rel=1e-12)
    assert esc_floor < dt_on < cfg.params.dt.minimum

    io._tau_drain_cap_valid = True
    hf_row_huge_flux = dict(hf_row, F_atm=1.0e8)
    dt_floored = next_step(cfg, {}, hf_row_huge_flux, hf_all, 1.0, interior_o=io)
    assert dt_floored == pytest.approx(esc_floor, rel=1e-12)
    assert getattr(io, '_tau_drain_cap_valid', None) is False

    cfg.interior_energetics.aragog.top_cell_dt_limit = False
    io.__dict__.pop('_tau_drain_cap_valid', None)
    dt_off = next_step(cfg, {}, hf_row, hf_all, 1.0, interior_o=io)
    assert '_tau_drain_cap_valid' not in io.__dict__
    assert dt_off >= cfg.params.dt.minimum
    assert dt_on < 0.2 * dt_off

    cfg.interior_energetics.aragog.top_cell_dt_limit = True
    io._tau_drain_cap_valid = True
    hf_row_invalid = dict(hf_row, T_surf=0.0)
    dt_invalid = next_step(cfg, {}, hf_row_invalid, hf_all, 1.0, interior_o=io)
    assert dt_invalid >= cfg.params.dt.minimum
    assert getattr(io, '_tau_drain_cap_valid', None) is False

    from proteus.utils.helper import SUBYEAR_TIME_RESOLUTION

    io_snap = _make_interior_o(
        t_next_impact=float(hf_row['Time']) + expected_tau + 0.5 * SUBYEAR_TIME_RESOLUTION
    )
    io_snap.aragog_solver = io.aragog_solver
    dt_snapped = next_step(cfg, {}, hf_row, hf_all, 1.0, interior_o=io_snap)
    assert dt_snapped == pytest.approx(expected_tau + 0.5 * SUBYEAR_TIME_RESOLUTION, rel=1e-12)
    assert dt_snapped > expected_tau
    assert getattr(io_snap, '_tau_drain_cap_valid', None) is False


def _load_paleos_ptop_531p546_slice_fixture() -> dict:
    """Load the committed 1D PALEOS EOS slice fixture at P_top = 4.455575e8 Pa."""
    import json
    from pathlib import Path

    fixture_path = (
        Path(__file__).resolve().parent / 'fixtures' / 'paleos_ptop_531p546_slice.json'
    )
    return json.loads(fixture_path.read_text(encoding='utf-8'))


class _PaleosSliceEOS:
    """1D PALEOS EOS slice evaluator at fixed P_top reproducing Aragog's EntropyEOS."""

    def __init__(self, data: dict):
        self.S_min = float(data['S_min'])
        self.S_max = float(data['S_max'])
        self._p_ref = float(data['P_top_Pa'])
        self._s_sol = float(data['S_sol_at_P_top'])
        self._s_liq = float(data['S_liq_at_P_top'])
        self._solid = {
            k: np.asarray(v, dtype=float) for k, v in data['phase_slices']['solid'].items()
        }
        self._melt = {
            k: np.asarray(v, dtype=float) for k, v in data['phase_slices']['melt'].items()
        }

    def melt_fraction(self, p_arr, s_arr):
        s = np.asarray(s_arr, dtype=float)
        return np.clip((s - self._s_sol) / (self._s_liq - self._s_sol), 0.0, 1.0)

    def _blend_linear(self, prop: str, s_arr):
        s = np.asarray(s_arr, dtype=float)
        phi = self.melt_fraction(self._p_ref, s)
        mushy = (phi > 0.0) & (phi < 1.0)
        s_sol_q = np.where(mushy, self._s_sol, s)
        s_liq_q = np.where(mushy, self._s_liq, s)
        v_sol = np.interp(
            np.clip(s_sol_q, self._solid['S'][0], self._solid['S'][-1]),
            self._solid['S'],
            self._solid[prop],
        )
        v_melt = np.interp(
            np.clip(s_liq_q, self._melt['S'][0], self._melt['S'][-1]),
            self._melt['S'],
            self._melt[prop],
        )
        res = np.where(phi > 0.0, phi * v_melt, 0.0) + np.where(
            phi < 1.0, (1.0 - phi) * v_sol, 0.0
        )
        return np.where(np.isnan(phi), np.nan, res)

    def temperature(self, p_arr, s_arr):
        p = np.asarray(p_arr, dtype=float)
        return self._blend_linear('temperature', s_arr) + 1.0e-7 * (p - self._p_ref)

    def heat_capacity(self, p_arr, s_arr):
        return self._blend_linear('heat_capacity', s_arr)

    def density(self, p_arr, s_arr):
        s = np.asarray(s_arr, dtype=float)
        phi = self.melt_fraction(self._p_ref, s)
        mushy = (phi > 0.0) & (phi < 1.0)
        rho_sol_b = np.interp(
            np.clip(self._s_sol, self._solid['S'][0], self._solid['S'][-1]),
            self._solid['S'],
            self._solid['density'],
        )
        rho_liq_b = np.interp(
            np.clip(self._s_liq, self._melt['S'][0], self._melt['S'][-1]),
            self._melt['S'],
            self._melt['density'],
        )
        rho_mushy = 1.0 / (phi / rho_liq_b + (1.0 - phi) / rho_sol_b)
        v_sol = np.interp(
            np.clip(s, self._solid['S'][0], self._solid['S'][-1]),
            self._solid['S'],
            self._solid['density'],
        )
        v_melt = np.interp(
            np.clip(s, self._melt['S'][0], self._melt['S'][-1]),
            self._melt['S'],
            self._melt['density'],
        )
        rho_single = np.where(phi >= 0.5, v_melt, v_sol)
        return np.where(mushy, rho_mushy, rho_single)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_estimate_tau_drain_paleos_slice_531p546_chord_lookahead_and_window():
    """On the committed 1D PALEOS slice at P_top = 4.455575e8 Pa (t = 531.546 yr, row 386),
    _estimate_tau_drain returns 11.241963 yr without chord collapse and 4.878514 yr
    (chord ratio 0.433956) with the 1-evaluation chord-cp lookahead."""
    import re

    from proteus.interior_energetics.timestep import _estimate_tau_drain

    data = _load_paleos_ptop_531p546_slice_fixture()
    assert len(data['source_files_md5']) >= 10
    assert isinstance(data.get('source_table'), str)
    assert data['source_table'] != 'none'
    assert data['source_table'].startswith('PALEOS')
    assert 'none EOS slice' not in data['description']
    assert 'aragog_commit' not in data
    assert (
        re.fullmatch(
            r'[A-Za-z0-9._-]+-[0-9]+-g[0-9a-f]{40}',
            str(data.get('aragog_commit_eos_eval', '')),
        )
        is not None
    )
    assert 0.0 <= float(data['max_2d_vs_slice_rel_diff']) < 1.0e-14
    rows = {int(r['row']): r for r in data['snap531_rows_380_to_392']}
    r386 = rows[386]

    cfg, io, solver = _make_tau_drain_fixture(
        s_top=float(r386['S_top_J_kg_K']),
        P_top=float(data['P_top_Pa']),
        r_outer=float(data['r_outer_m']),
        rel_dT=0.08,
        rel_dT_solid=0.08,
        top_cell_dt_limit=True,
    )
    solver._volume_flat = np.array([5.0e18, float(data['vol_top_m3'])], dtype=float)
    eos_slice = _PaleosSliceEOS(data)
    ref_pts = data['reference_2d_points']
    s_ref = np.asarray(ref_pts['S_J_kg_K'], dtype=float)
    p_ref = np.full_like(s_ref, float(data['P_top_Pa']), dtype=float)
    s_sol_ref = float(data['S_sol_at_P_top'])
    s_liq_ref = float(data['S_liq_at_P_top'])
    assert len(s_ref) >= 25
    assert int(np.sum(np.abs(s_ref - s_sol_ref) <= 5.0)) >= 3
    assert int(np.sum(np.abs(s_ref - s_liq_ref) <= 5.0)) >= 3
    assert int(np.sum((s_ref >= 2429.71) & (s_ref < s_sol_ref))) >= 2
    assert int(np.sum((s_ref > s_sol_ref) & (s_ref < s_liq_ref))) >= 6
    assert 1 <= int(np.sum(s_ref >= 11283.0)) <= 3
    for key, slice_fn in (
        ('temperature_K', eos_slice.temperature),
        ('heat_capacity_J_kg_K', eos_slice.heat_capacity),
        ('density_kg_m3', eos_slice.density),
        ('melt_fraction', eos_slice.melt_fraction),
    ):
        ref_arr = np.asarray(ref_pts[key], dtype=float)
        got_arr = np.asarray(slice_fn(p_ref, s_ref), dtype=float).ravel()
        denom = np.maximum(np.abs(ref_arr), 1.0e-30)
        assert float(np.max(np.abs(got_arr - ref_arr) / denom)) < 1.0e-14
    solver.entropy_eos = eos_slice

    hf_386 = {
        'Time': float(r386['Time_yr']),
        'T_surf': float(r386['T_surf_K']),
        'F_atm': float(r386['F_atm_W_m2']),
    }
    tau_chord_386 = _estimate_tau_drain(cfg, hf_386, io)

    eos_no_collapse = _PaleosSliceEOS(data)
    eos_no_collapse.temperature = lambda p_arr, s_arr: np.array(
        [1918.1830315644013], dtype=float
    )
    solver.entropy_eos = eos_no_collapse
    tau_start_386 = _estimate_tau_drain(cfg, hf_386, io)

    assert 0.0 < tau_chord_386 < tau_start_386 < 100.0
    assert tau_start_386 == pytest.approx(11.241963252904497, rel=1e-6)
    assert tau_chord_386 == pytest.approx(4.878514423642098, rel=1e-6)
    assert (tau_chord_386 / tau_start_386) == pytest.approx(0.43395573638580115, rel=1e-5)
    t_top_386 = 1918.1830315644013
    t_trial_386 = 1567.854994037665
    tau_tmean_wrong = tau_chord_386 * (0.5 * (t_top_386 + t_trial_386) / t_top_386)
    assert tau_chord_386 != pytest.approx(tau_tmean_wrong, rel=5e-2)

    ratios_by_row: dict[int, float] = {}
    for row_idx in range(380, 393):
        rec = rows[row_idx]
        s_val = float(rec['S_top_J_kg_K'])
        solver.entropy_staggered = np.array([[2800.0], [s_val]], dtype=float)
        solver._solution = SimpleNamespace(y=solver.entropy_staggered.copy())
        hf_r = {
            'Time': float(rec['Time_yr']),
            'T_surf': float(rec['T_surf_K']),
            'F_atm': float(rec['F_atm_W_m2']),
        }
        solver.entropy_eos = eos_slice
        tau_c = _estimate_tau_drain(cfg, hf_r, io)
        solver.entropy_eos = eos_no_collapse
        tau_0 = _estimate_tau_drain(cfg, hf_r, io)
        ratio = tau_c / tau_0
        ratios_by_row[row_idx] = ratio
        if row_idx in range(380, 384):
            assert ratio == pytest.approx(1.0, rel=1e-12)
        else:
            assert 0.433 < ratio < 0.851
    assert min(ratios_by_row, key=ratios_by_row.get) == 386
    assert ratios_by_row[384] > ratios_by_row[385] > ratios_by_row[386] < ratios_by_row[387]


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_estimate_tau_drain_lookahead_floor_clip_and_edge_guards(caplog):
    """Verify S_trial floor clipping at S_min + 300.0 uses T_top*(S_top - S_trial)/dT_trial
    without loosening, handles T_trial >= T_top without dividing by zero, and warns once on
    non-finite or non-positive EOS temperatures."""
    import logging

    from proteus.interior_energetics.timestep import (
        _TAU_DRAIN_SOLID_S_OFFSET,
        _estimate_tau_drain,
    )
    from proteus.utils.constants import secs_per_year

    s_min = 106.11618925017348
    s_floor = s_min + _TAU_DRAIN_SOLID_S_OFFSET
    s_top = 600.0
    cp_val = 1200.0
    t_surf = 1500.0
    t_top = 600.0
    t_floor = 300.0
    queried_s: list[float] = []

    cfg, io, solver = _make_tau_drain_fixture(
        s_top=s_top,
        phi_top=0.0,
        s_min=s_min,
        cp_live=cp_val,
        cp_solid_at_1800=cp_val,
        cp_at_smin_plus_300=cp_val,
        rel_dT=0.08,
        rel_dT_solid=0.08,
    )

    def _temp_clipped(p_arr, s_arr):
        s_q = float(np.asarray(s_arr, dtype=float).ravel()[0])
        queried_s.append(s_q)
        if abs(s_q - s_top) < 1e-9:
            return np.array([t_top], dtype=float)
        return np.array([t_floor], dtype=float)

    solver.entropy_eos.temperature = _temp_clipped
    row = {'Time': 20.0, 'F_atm': 500.0, 'T_surf': t_surf}
    tau_clipped = _estimate_tau_drain(cfg, row, io)

    area_s = 4.0 * np.pi * float(solver._r_basic_flat[-1]) ** 2
    dr_eff = float(solver._volume_flat[-1]) / area_s
    dt_allow = 0.08 * t_surf
    tau_start = (4000.0 * cp_val * dr_eff * dt_allow) / (500.0 * secs_per_year)
    cp_chord_expected = min(cp_val, t_top * (s_top - s_floor) / (t_top - t_floor))
    cp_short_wrong = cp_val * dt_allow / (t_top - t_floor)
    tau_expected = (4000.0 * cp_chord_expected * dr_eff * dt_allow) / (500.0 * secs_per_year)
    tau_short_wrong = (4000.0 * cp_short_wrong * dr_eff * dt_allow) / (500.0 * secs_per_year)

    assert queried_s == [pytest.approx(s_top, rel=1e-12), pytest.approx(s_floor, rel=1e-12)]
    assert tau_clipped == pytest.approx(tau_expected, rel=1e-12)
    assert tau_clipped != pytest.approx(tau_short_wrong, rel=1e-2)
    assert 0.0 < tau_clipped < tau_start

    # A clipped trial within dT_allow (0 < T_top - T_trial <= dT_allow) keeps cp_eff
    t_floor_within_allow = 500.0
    solver.entropy_eos.temperature = lambda p_arr, s_arr: np.array(
        [
            t_top
            if abs(float(np.asarray(s_arr).ravel()[0]) - s_top) < 1e-9
            else t_floor_within_allow
        ],
        dtype=float,
    )
    tau_clipped_within_allow = _estimate_tau_drain(cfg, row, io)
    cp_dt_gt_zero_wrong = min(
        cp_val, t_top * (s_top - s_floor) / (t_top - t_floor_within_allow)
    )
    tau_dt_gt_zero_wrong = (4000.0 * cp_dt_gt_zero_wrong * dr_eff * dt_allow) / (
        500.0 * secs_per_year
    )
    assert tau_clipped_within_allow == pytest.approx(tau_start, rel=1e-12)
    assert tau_clipped_within_allow != pytest.approx(tau_dt_gt_zero_wrong, rel=1e-2)

    # At phi_top <= 0.60 with cp_top > cp_solid, cp_eff (not cp_top) sets S_trial
    cfg_low_phi, io_low_phi, solver_low_phi = _make_tau_drain_fixture(
        s_top=2400.0,
        phi_top=0.30,
        s_min=s_min,
        cp_live=3000.0,
        cp_solid_at_1800=1200.0,
        rel_dT=0.08,
        rel_dT_solid=0.08,
    )
    queried_low_phi_s: list[float] = []

    def _temp_low_phi(p_arr, s_arr):
        s_q = float(np.asarray(s_arr, dtype=float).ravel()[0])
        queried_low_phi_s.append(s_q)
        if abs(s_q - 2400.0) < 1e-9:
            return np.array([1500.0], dtype=float)
        if s_q >= 2250.0:
            return np.array([1300.0], dtype=float)
        return np.array([900.0], dtype=float)

    solver_low_phi.entropy_eos.temperature = _temp_low_phi
    tau_low_phi = _estimate_tau_drain(cfg_low_phi, row, io_low_phi)
    cp_chord_low_phi = 1500.0 * (2400.0 - 2304.0) / (1500.0 - 1300.0)
    expected_low_phi_tau = (4000.0 * cp_chord_low_phi * dr_eff * dt_allow) / (
        500.0 * secs_per_year
    )
    cp_chord_cp_top_wrong = 1500.0 * (2400.0 - 2160.0) / (1500.0 - 900.0)
    wrong_cp_top_tau = (4000.0 * cp_chord_cp_top_wrong * dr_eff * dt_allow) / (
        500.0 * secs_per_year
    )
    assert queried_low_phi_s == [
        pytest.approx(2400.0, rel=1e-12),
        pytest.approx(2304.0, rel=1e-12),
    ]
    assert tau_low_phi == pytest.approx(expected_low_phi_tau, rel=1e-12)
    assert tau_low_phi != pytest.approx(wrong_cp_top_tau, rel=1e-2)

    # When S_top < S_min + 300.0, S_trial clips at S_min + 300.0 > S_top and never loosens
    solver.entropy_staggered = np.array([[2800.0], [200.0]], dtype=float)
    solver._solution = SimpleNamespace(y=solver.entropy_staggered.copy())
    solver.entropy_eos.temperature = lambda p_arr, s_arr: np.array(
        [250.0 if float(np.asarray(s_arr).ravel()[0]) < 300.0 else 100.0], dtype=float
    )
    tau_below_floor = _estimate_tau_drain(cfg, row, io)
    assert tau_below_floor == pytest.approx(tau_start, rel=1e-12)

    # T_trial == T_top (dt_trial == 0) and T_trial > T_top (dt_trial < 0): no zero-division
    solver.entropy_staggered = np.array([[2800.0], [s_top]], dtype=float)
    solver._solution = SimpleNamespace(y=solver.entropy_staggered.copy())
    solver.entropy_eos.temperature = lambda p_arr, s_arr: np.array([1500.0], dtype=float)
    assert _estimate_tau_drain(cfg, row, io) == pytest.approx(tau_start, rel=1e-12)
    solver.entropy_eos.temperature = lambda p_arr, s_arr: np.array(
        [1500.0 if abs(float(np.asarray(s_arr).ravel()[0]) - s_top) < 1e-9 else 1650.0],
        dtype=float,
    )
    assert _estimate_tau_drain(cfg, row, io) == pytest.approx(tau_start, rel=1e-12)

    # Non-finite or non-positive T_top or T_trial returns inf and warns once per interior_o
    for bad_t_top, bad_t_trial in (
        (float('nan'), 1400.0),
        (0.0, 1400.0),
        (-50.0, 1400.0),
        (1500.0, float('nan')),
        (1500.0, float('inf')),
        (1500.0, 0.0),
        (1500.0, -10.0),
    ):
        cfg_b, io_b, solver_b = _make_tau_drain_fixture(
            s_top=2400.0, phi_top=0.80, rel_dT=0.08, rel_dT_solid=0.08
        )
        solver_b.entropy_eos.temperature = lambda p_arr, s_arr, _tt=bad_t_top, _tr=bad_t_trial: (
            np.array(
                [_tt if abs(float(np.asarray(s_arr).ravel()[0]) - 2400.0) < 1e-9 else _tr],
                dtype=float,
            )
        )
        caplog.clear()
        with caplog.at_level(
            logging.WARNING, logger='fwl.proteus.interior_energetics.timestep'
        ):
            res_b = _estimate_tau_drain(cfg_b, row, io_b)
        assert res_b == float('inf')
        assert getattr(io_b, '_tau_drain_invalid_warned', False) is True
        warns_b = [r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()]
        assert len(warns_b) == 1

    # Exception raised on the second (s_trial) temperature lookup warns once and returns inf
    for exc_cls in (ValueError, IndexError):
        cfg_e, io_e, solver_e = _make_tau_drain_fixture(
            s_top=2400.0, phi_top=0.80, rel_dT=0.08, rel_dT_solid=0.08
        )

        def _raise_on_trial(p_arr, s_arr, _exc=exc_cls):
            if abs(float(np.asarray(s_arr).ravel()[0]) - 2400.0) < 1e-9:
                return np.array([1500.0], dtype=float)
            raise _exc('trial temperature lookup failure')

        solver_e.entropy_eos.temperature = _raise_on_trial
        caplog.clear()
        with caplog.at_level(
            logging.WARNING, logger='fwl.proteus.interior_energetics.timestep'
        ):
            res_e = _estimate_tau_drain(cfg_e, row, io_e)
        assert res_e == float('inf')
        assert getattr(io_e, '_tau_drain_invalid_warned', False) is True
        assert (
            len([r.getMessage() for r in caplog.records if 'tau_drain' in r.getMessage()]) == 1
        )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_estimate_tau_drain_chord_lookahead_never_loosens_property():
    """Property check: chord-cp lookahead step is never longer than the start-of-step estimate."""
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from proteus.interior_energetics.timestep import _estimate_tau_drain

    checked: list[tuple[float, float]] = []

    @settings(derandomize=True, max_examples=50)
    @given(
        s_top=st.floats(
            min_value=150.0, max_value=5000.0, allow_nan=False, allow_infinity=False
        ),
        phi_top=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        t_surf=st.floats(
            min_value=400.0, max_value=3500.0, allow_nan=False, allow_infinity=False
        ),
        f_atm=st.floats(min_value=10.0, max_value=1.0e6, allow_nan=False, allow_infinity=False),
        t_top=st.floats(
            min_value=400.0, max_value=3500.0, allow_nan=False, allow_infinity=False
        ),
        trial_frac=st.floats(
            min_value=0.1, max_value=1.2, allow_nan=False, allow_infinity=False
        ),
    )
    def _check(s_top, phi_top, t_surf, f_atm, t_top, trial_frac):
        cfg, io, solver = _make_tau_drain_fixture(
            s_top=s_top, phi_top=phi_top, rel_dT=0.08, rel_dT_solid=0.08
        )
        row = {'Time': 25.0, 'F_atm': f_atm, 'T_surf': t_surf}
        solver.entropy_eos.temperature = lambda p_arr, s_arr: np.array([t_top], dtype=float)
        tau_start = _estimate_tau_drain(cfg, row, io)

        def _t_drop(p_arr, s_arr, _s0=s_top, _t0=t_top, _tf=trial_frac):
            s_q = float(np.asarray(s_arr, dtype=float).ravel()[0])
            if abs(s_q - _s0) < 1e-9:
                return np.array([_t0], dtype=float)
            return np.array([_t0 * _tf], dtype=float)

        solver.entropy_eos.temperature = _t_drop
        tau_lk = _estimate_tau_drain(cfg, row, io)
        assert 0.0 < tau_lk <= tau_start * (1.0 + 1e-12)
        checked.append((tau_start, tau_lk))

    _check()
    assert len(checked) >= 50
    assert any(lk < st_val * 0.99 for st_val, lk in checked)

    data = _load_paleos_ptop_531p546_slice_fixture()
    s_min_slice = float(data['S_min'])
    s_max_slice = float(data['S_max'])
    p_top_slice = float(data['P_top_Pa'])
    r_outer_slice = float(data['r_outer_m'])
    vol_top_slice = float(data['vol_top_m3'])
    slice_checked: list[tuple[float, float]] = []

    @settings(derandomize=True, max_examples=50)
    @given(
        s_top=st.floats(
            min_value=s_min_slice,
            max_value=s_max_slice,
            allow_nan=False,
            allow_infinity=False,
        ),
        t_surf=st.floats(
            min_value=300.0, max_value=3800.0, allow_nan=False, allow_infinity=False
        ),
        f_atm=st.floats(min_value=10.0, max_value=1.0e6, allow_nan=False, allow_infinity=False),
    )
    def _check_paleos_slice(s_top, t_surf, f_atm):
        cfg, io, solver = _make_tau_drain_fixture(
            s_top=s_top,
            P_top=p_top_slice,
            r_outer=r_outer_slice,
            rel_dT=0.08,
            rel_dT_solid=0.08,
        )
        solver._volume_flat = np.array([5.0e18, vol_top_slice], dtype=float)
        eos_slice = _PaleosSliceEOS(data)
        t_top_val = float(
            np.asarray(
                eos_slice.temperature(np.array([p_top_slice]), np.array([s_top])),
                dtype=float,
            ).ravel()[0]
        )
        solver.entropy_eos = eos_slice
        row = {'Time': 25.0, 'F_atm': f_atm, 'T_surf': t_surf}
        tau_lk = _estimate_tau_drain(cfg, row, io)

        eos_no_lk = _PaleosSliceEOS(data)
        eos_no_lk.temperature = lambda p_arr, s_arr, _t0=t_top_val: np.array([_t0], dtype=float)
        solver.entropy_eos = eos_no_lk
        tau_start = _estimate_tau_drain(cfg, row, io)
        assert 0.0 < tau_lk <= tau_start * (1.0 + 1e-12)
        slice_checked.append((tau_start, tau_lk))

    _check_paleos_slice()
    assert len(slice_checked) >= 50
    assert any(lk < st_val * 0.99 for st_val, lk in slice_checked)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_estimate_tau_drain_positivity_and_monotonic_flux_property():
    """Property check: tau_drain is positive, finite, and decreases monotonically in F_atm."""
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from proteus.interior_energetics.timestep import _estimate_tau_drain

    checked: list[tuple[float, float]] = []

    @settings(derandomize=True)
    @given(
        t_surf=st.floats(
            min_value=300.0, max_value=4000.0, allow_nan=False, allow_infinity=False
        ),
        phi_top=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        f_low=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    def _check(t_surf, phi_top, f_low):
        cfg, io, _ = _make_tau_drain_fixture(phi_top=phi_top)
        tau_lo = _estimate_tau_drain(cfg, {'F_atm': f_low, 'T_surf': t_surf}, io)
        tau_hi = _estimate_tau_drain(cfg, {'F_atm': 2.0 * f_low, 'T_surf': t_surf}, io)
        assert 0.0 < tau_hi < tau_lo < float('inf')
        checked.append((tau_lo, tau_hi))

    _check()
    assert len(checked) >= 10
    assert all(lo > hi > 0.0 for lo, hi in checked)


@pytest.mark.unit
def test_paleos_slice_fixture_builder_helpers(tmp_path, monkeypatch):
    """Unit test _resolve_source_table and _resolve_aragog_commit in build_paleos_ptop_slice_fixture."""
    import subprocess
    from types import SimpleNamespace

    from tests.interior_energetics.fixtures import build_paleos_ptop_slice_fixture as builder

    cfg_valid = tmp_path / 'valid_zalmoxis.toml'
    cfg_valid.write_text(
        '[interior_struct]\nmodule = "zalmoxis"\neos_dir = "none"\n'
        '[interior_struct.zalmoxis]\nmantle_eos = "PALEOS:MgSiO3"\n',
        encoding='utf-8',
    )
    assert builder._resolve_source_table(cfg_valid) == 'PALEOS:MgSiO3'

    # mantle_eos wins over eos_dir and over the structure module, as in generate_spider_tables.
    cfg_dummy_struct = tmp_path / 'dummy_struct.toml'
    cfg_dummy_struct.write_text(
        '[interior_struct]\nmodule = "dummy"\neos_dir = "PALEOS:MgSiO3"\n'
        '[interior_struct.zalmoxis]\nmantle_eos = "PALEOS-2phase:MgSiO3"\n',
        encoding='utf-8',
    )
    assert builder._resolve_source_table(cfg_dummy_struct) == 'PALEOS-2phase:MgSiO3'

    # A PALEOS H2O mantle uses the MgSiO3 2-phase tables, as energetics_eos_key maps it.
    cfg_h2o = tmp_path / 'h2o_mantle.toml'
    cfg_h2o.write_text(
        '[interior_struct]\nmodule = "zalmoxis"\n'
        '[interior_struct.zalmoxis]\nmantle_eos = "PALEOS:H2O"\n',
        encoding='utf-8',
    )
    assert builder._resolve_source_table(cfg_h2o) == 'PALEOS-2phase:MgSiO3'

    cfg_spider = tmp_path / 'spider_struct.toml'
    cfg_spider.write_text(
        '[interior_struct]\nmodule = "spider"\n'
        '[interior_struct.zalmoxis]\nmantle_eos = "PALEOS:MgSiO3"\n',
        encoding='utf-8',
    )
    with pytest.raises(RuntimeError, match='takes its P-S tables from FWL_DATA'):
        builder._resolve_source_table(cfg_spider)

    for bad_eos in ('WolfBower2018:MgSiO3', 'none', ''):
        cfg_bad = tmp_path / f'bad_{bad_eos or "empty"}.toml'
        cfg_bad.write_text(
            f'[interior_struct]\nmodule = "zalmoxis"\neos_dir = "PALEOS:MgSiO3"\n'
            f'[interior_struct.zalmoxis]\nmantle_eos = "{bad_eos}"\n',
            encoding='utf-8',
        )
        with pytest.raises(RuntimeError, match='Invalid EOS source_table'):
            builder._resolve_source_table(cfg_bad)

    cfg_no_mantle = tmp_path / 'no_mantle_eos.toml'
    cfg_no_mantle.write_text('[interior_struct]\nmodule = "zalmoxis"\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Invalid EOS source_table'):
        builder._resolve_source_table(cfg_no_mantle)

    fake_aragog_file = tmp_path / 'aragog' / 'eos' / 'entropy.py'
    fake_aragog_file.parent.mkdir(parents=True, exist_ok=True)
    fake_aragog_file.write_text('# stub\n', encoding='utf-8')
    valid_desc = '26.05.10-47-g075fcb05ce035183100c9bec47b7dcc7d769c2d6'

    calls = []

    def _mock_run_ok(cmd, **_kwargs):
        calls.append(list(cmd))
        if 'rev-parse' in cmd:
            return SimpleNamespace(stdout=f'{tmp_path / "aragog"}\n')
        return SimpleNamespace(stdout=f'{valid_desc}\n')

    assert (
        builder._resolve_aragog_commit(aragog_file=fake_aragog_file, run_cmd=_mock_run_ok)
        == valid_desc
    )
    repo_dir = str(fake_aragog_file.resolve().parent)
    assert calls[0] == ['git', '-C', repo_dir, 'rev-parse', '--show-toplevel']
    assert calls[1][:4] == ['git', '-C', repo_dir, 'describe']
    for flag in ('--long', '--always', '--dirty', '--abbrev=40'):
        assert flag in calls[1]

    # Default lookup: the checkout of the imported aragog.eos.entropy module.
    calls.clear()
    monkeypatch.setattr(builder.aragog.eos.entropy, '__file__', str(fake_aragog_file))
    assert builder._resolve_aragog_commit(run_cmd=_mock_run_ok) == valid_desc
    assert calls[1][:3] == ['git', '-C', repo_dir]

    for bad_desc in (
        f'{valid_desc}-dirty',
        '26.05.10-47-g075fcb05',
        '075fcb05ce035183100c9bec47b7dcc7d769c2d6',
        'x' * 56,
    ):

        def _mock_run_bad_desc(cmd, _d=bad_desc, **_kwargs):
            if 'rev-parse' in cmd:
                return SimpleNamespace(stdout=f'{tmp_path / "aragog"}\n')
            return SimpleNamespace(stdout=f'{_d}\n')

        with pytest.raises(RuntimeError, match='Expected clean aragog git describe'):
            builder._resolve_aragog_commit(
                aragog_file=fake_aragog_file, run_cmd=_mock_run_bad_desc
            )

    def _mock_run_wrong_top(cmd, **_kwargs):
        if 'rev-parse' in cmd:
            return SimpleNamespace(stdout=f'{tmp_path / "proteus"}\n')
        return SimpleNamespace(stdout=f'{valid_desc}\n')

    with pytest.raises(RuntimeError, match='Expected git toplevel for aragog'):
        builder._resolve_aragog_commit(
            aragog_file=fake_aragog_file, run_cmd=_mock_run_wrong_top
        )

    def _mock_run_missing_git(_cmd, **_kwargs):
        raise FileNotFoundError('git not found')

    with pytest.raises(RuntimeError, match='Failed to resolve aragog git commit'):
        builder._resolve_aragog_commit(
            aragog_file=fake_aragog_file, run_cmd=_mock_run_missing_git
        )

    def _mock_run_called_error(cmd, **_kwargs):
        raise subprocess.CalledProcessError(128, cmd)

    with pytest.raises(RuntimeError, match='Failed to resolve aragog git commit'):
        builder._resolve_aragog_commit(
            aragog_file=fake_aragog_file, run_cmd=_mock_run_called_error
        )

    monkeypatch.setattr(builder.aragog.eos.entropy, '__file__', None)
    with pytest.raises(RuntimeError, match='Cannot locate aragog'):
        builder._resolve_aragog_commit(aragog_file=None, run_cmd=_mock_run_ok)
