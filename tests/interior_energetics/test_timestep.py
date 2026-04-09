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

pytestmark = pytest.mark.unit


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
        minimum=100.0,
        minimum_rel=0.005,
        maximum=dt_max,
        initial=10.0,
        mushy_maximum=mushy_maximum,
        mushy_upper=mushy_upper,
        hysteresis_iters=hysteresis_iters,
        hysteresis_sfinc=hysteresis_sfinc,
    )
    stop_solid = SimpleNamespace(enabled=True, phi_crit=phi_crit)
    stop_radeqm = SimpleNamespace(enabled=False)
    stop_escape = SimpleNamespace(enabled=False)
    stop = SimpleNamespace(
        solid=stop_solid,
        radeqm=stop_radeqm,
        escape=stop_escape,
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
    return SimpleNamespace(dt_hysteresis_remaining=0, solver_stiffness=0.0)


# ---------------------------------------------------------------------------
# Mushy-regime automatic dt cap
# ---------------------------------------------------------------------------


class TestMushyCap:
    """Verify the Phi-aware dt cap activates only inside the mushy band."""

    def test_disabled_when_mushy_maximum_is_zero(self, tmp_path):
        """``mushy_maximum = 0`` must preserve legacy behaviour — no cap."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(mushy_maximum=0.0)
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=0.5)
        hf_row = {
            'Time': float(hf_all['Time'].iloc[-1]) + 5.0e3,
            'F_atm': 1.0e4,
            'Phi_global': 0.5,  # inside mushy band
        }
        dt = next_step(
            config, {}, hf_row, hf_all, 1.0,
            interior_o=_make_interior_o(),
        )
        # SFINC * dt_prev = 1.6 * 5e3 = 8e3. No cap applied since
        # mushy_maximum = 0 disables the feature.
        assert dt == pytest.approx(8.0e3, rel=1e-6), (
            f'Expected 8e3, got {dt}'
        )

    def test_cap_active_when_phi_in_band(self, tmp_path):
        """Cap kicks in when Phi_global is inside (phi_crit, mushy_upper)."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(
            mushy_maximum=4.0e3, mushy_upper=0.99, phi_crit=0.05,
        )
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=0.5)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 0.5}
        dt = next_step(
            config, {}, hf_row, hf_all, 1.0,
            interior_o=_make_interior_o(),
        )
        # 1.6 * 5e3 = 8e3 would be chosen; cap to 4e3.
        assert dt == pytest.approx(4.0e3, rel=1e-6), (
            f'Expected 4e3 (mushy cap), got {dt}'
        )

    def test_cap_inactive_when_phi_above_upper(self):
        """Pure-liquid (Phi > mushy_upper) must NOT trigger the cap."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(mushy_maximum=4.0e3, mushy_upper=0.99)
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=1.0)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        dt = next_step(
            config, {}, hf_row, hf_all, 1.0,
            interior_o=_make_interior_o(),
        )
        # Cap not active because Phi = 1.0 >= mushy_upper = 0.99.
        assert dt > 4.0e3, (
            f'Expected dt > 4e3 (cap inactive), got {dt}'
        )

    def test_cap_inactive_when_phi_below_phi_crit(self):
        """Solidified (Phi < phi_crit) must NOT trigger the cap."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(
            mushy_maximum=4.0e3, mushy_upper=0.99, phi_crit=0.05,
        )
        hf_all = _make_hf_all(n_rows=12, dt_prev=5.0e3, phi=0.02)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 0.02}
        dt = next_step(
            config, {}, hf_row, hf_all, 1.0,
            interior_o=_make_interior_o(),
        )
        # Cap not active because Phi = 0.02 <= phi_crit = 0.05.
        # But dt is still limited by the termination estimator — grab
        # its contribution by running with cap disabled for comparison.
        config_no_cap = _make_config(
            mushy_maximum=0.0, mushy_upper=0.99, phi_crit=0.05,
        )
        dt_no_cap = next_step(
            config_no_cap, {}, hf_row, hf_all, 1.0,
            interior_o=_make_interior_o(),
        )
        # The cap must NOT have made dt smaller than the un-capped
        # version; they should be identical.
        assert dt == pytest.approx(dt_no_cap, rel=1e-6), (
            f'Cap should be inactive below phi_crit, but dt={dt} vs '
            f'dt_no_cap={dt_no_cap}'
        )


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
            f'Expected counter=3 after slow-down, got '
            f'{interior_o.dt_hysteresis_remaining}'
        )

    def test_active_hysteresis_replaces_sfinc(self):
        """While counter > 0, the speed-up factor is ``hysteresis_sfinc``."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(
            hysteresis_iters=3, hysteresis_sfinc=1.1,
            mushy_maximum=0.0,  # disable mushy cap for this test
        )
        hf_all = self._hf_forcing_speed_up(dt_prev=1.0e3)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        interior_o = _make_interior_o()
        interior_o.dt_hysteresis_remaining = 2  # simulate mid-window

        dt = next_step(
            config, {}, hf_row, hf_all, 1.0, interior_o=interior_o,
        )
        # With hysteresis_sfinc = 1.1 and dt_prev = 1e3, expect dt =
        # 1.1 * 1e3 = 1.1e3. Without hysteresis it would be
        # SFINC * 1e3 = 1.6e3.
        assert dt == pytest.approx(1.1e3, rel=1e-6), (
            f'Expected gentler dt = 1.1e3 while hysteresis active, '
            f'got {dt}'
        )
        # Counter was 2, should now be 1 after this speed-up call.
        assert interior_o.dt_hysteresis_remaining == 1, (
            f'Expected counter decremented to 1, got '
            f'{interior_o.dt_hysteresis_remaining}'
        )

    def test_zero_hysteresis_preserves_legacy_sfinc(self):
        """``hysteresis_iters = 0`` keeps the full SFINC = 1.6."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(
            hysteresis_iters=0, hysteresis_sfinc=1.1,
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
            config, {}, hf_row, hf_all, 1.0, interior_o=interior_o,
        )
        # 1.6 * 1e3 = 1.6e3.
        assert dt == pytest.approx(1.6e3, rel=1e-6), (
            f'Expected full SFINC dt = 1.6e3, got {dt}'
        )

    def test_counter_decrements_to_zero(self):
        """After enough speed-ups the counter returns to 0 and SFINC
        goes back to its full value."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(
            hysteresis_iters=2, hysteresis_sfinc=1.1,
            mushy_maximum=0.0,
        )
        hf_all = self._hf_forcing_speed_up(dt_prev=1.0e3)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        interior_o = _make_interior_o()
        interior_o.dt_hysteresis_remaining = 2

        dts = []
        for _ in range(4):
            dt = next_step(
                config, {}, hf_row, hf_all, 1.0, interior_o=interior_o,
            )
            dts.append(dt)
            # Simulate PROTEUS advancing: next dtprev would be the
            # dt we just computed, so update hf_all tail time by dt.
            last = hf_all.iloc[-1].copy()
            last['Time'] = float(last['Time']) + dt
            hf_all = pd.concat(
                [hf_all, pd.DataFrame([last])], ignore_index=True,
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

    def test_none_interior_o_disables_hysteresis(self):
        """Passing ``interior_o=None`` must preserve pre-2026-04-09
        behaviour: no hysteresis, no counter, full SFINC."""
        from proteus.interior_energetics.timestep import next_step

        config = _make_config(
            hysteresis_iters=3, hysteresis_sfinc=1.1,
            mushy_maximum=0.0,
        )
        hf_all = self._hf_forcing_speed_up(dt_prev=1.0e3)
        hf_row = {'Time': 1e5, 'F_atm': 1.0e4, 'Phi_global': 1.0}
        dt = next_step(config, {}, hf_row, hf_all, 1.0, interior_o=None)
        # Without interior_o the hysteresis machinery is fully skipped.
        assert dt == pytest.approx(1.6e3, rel=1e-6)
