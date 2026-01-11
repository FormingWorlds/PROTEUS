"""Unit tests for termination logic in `proteus.utils.terminate`.

Exercises convergence/termination criteria for solidification, energy balance,
volatile escape, disintegration, time/iteration limits, and keepalive guard.
Follows PROTEUS testing standards (see docs/test_infrastructure.md and
docs/test_building_strategy.md).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import proteus.utils.terminate as terminate


def _cfg(**kwargs: Any) -> Any:
    """Build a minimal config-like namespace with defaults and overrides."""

    def ns(**kw):
        return SimpleNamespace(**kw)

    stop = ns(
        solid=ns(enabled=True, phi_crit=0.3),
        radeqm=ns(enabled=True, rtol=0.1, atol=1.0),
        escape=ns(enabled=True, p_stop=1.0),
        disint=ns(
            enabled=True,
            roche_enabled=True,
            spin_enabled=True,
            offset_roche=0.0,
            offset_spin=0.0,
        ),
        time=ns(enabled=True, maximum=100.0, minimum=0.0),
        iters=ns(enabled=True, total_loops=5, total_min=1),
        strict=False,
    )
    params = ns(stop=stop)
    return ns(params=params, atmos_clim=ns(prevent_warming=False), **kwargs)


def _handler(cfg: Any, *, phi_global: float = 0.4) -> Any:
    """Create a minimal handler stub with hf_row/loops flags."""

    hf_row = {
        'Phi_global': phi_global,
        'F_atm': 0.0,
        'F_tidal': 0.0,
        'F_radio': 0.0,
        'P_surf': 10.0,
        'separation': 2.0,
        'roche_limit': 1.0,
        'axial_period': 10.0,
        'breakup_period': 5.0,
        'Time': 0.0,
    }
    loops = {
        'total': 0,
        'total_loops': cfg.params.stop.iters.total_loops,
        'total_min': cfg.params.stop.iters.total_min,
    }
    handler = SimpleNamespace(
        hf_row=hf_row,
        config=cfg,
        directories=Path('/tmp'),
        lockfile='/tmp/keepalive.lock',
        loops=loops,
        finished_prev=False,
        finished_both=False,
    )
    return handler


@pytest.fixture(autouse=True)
def patch_statusfile(monkeypatch):
    """Patch UpdateStatusfile to avoid filesystem writes and capture calls."""

    calls: list[tuple[Path, int]] = []

    def _fake_update(path: Path, code: int):
        calls.append((Path(path), code))

    monkeypatch.setattr(terminate, 'UpdateStatusfile', _fake_update)
    return calls


@pytest.mark.unit
def test_check_solid_triggers_when_phi_below_crit(monkeypatch, patch_statusfile):
    """Solidification: phi_global below threshold should terminate with code 10."""
    cfg = _cfg()
    h = _handler(cfg, phi_global=0.2)
    assert terminate._check_solid(h) is True
    assert patch_statusfile[-1][1] == 10


@pytest.mark.unit
def test_check_solid_skips_when_above_crit(patch_statusfile):
    """Solidification: above threshold keeps simulation running."""
    cfg = _cfg()
    h = _handler(cfg, phi_global=0.5)
    assert terminate._check_solid(h) is False
    assert patch_statusfile == []


@pytest.mark.unit
def test_check_radeqm_hits_energy_balance(patch_statusfile):
    """Energy balance: F_atm == F_tidal yields convergence with status 14."""
    cfg = _cfg()
    h = _handler(cfg)
    h.hf_row['F_atm'] = 1.0
    h.hf_row['F_tidal'] = 1.0
    h.hf_row['F_radio'] = 0.0
    assert terminate._check_radeqm(h) is True
    assert patch_statusfile[-1][1] == 14


@pytest.mark.unit
def test_check_radeqm_prevent_warming_triggers(monkeypatch, patch_statusfile):
    """Energy balance: prevent_warming=True exits when cooling stops (status 14)."""
    cfg = _cfg(atmos_clim=SimpleNamespace(prevent_warming=True))
    h = _handler(cfg)
    h.hf_row['F_atm'] = 0.0
    h.hf_row['F_tidal'] = 1.0
    h.hf_row['F_radio'] = 0.0
    assert terminate._check_radeqm(h) is True
    assert patch_statusfile[-1][1] == 14


@pytest.mark.unit
def test_check_escape_triggers_when_pressure_low(patch_statusfile):
    """Escape: surface pressure below stop threshold exits with status 15."""
    cfg = _cfg()
    h = _handler(cfg)
    h.hf_row['P_surf'] = 0.5
    assert terminate._check_escape(h) is True
    assert patch_statusfile[-1][1] == 15


@pytest.mark.unit
def test_check_escape_not_triggered_when_pressure_high(patch_statusfile):
    """Escape: high surface pressure keeps simulation running."""
    cfg = _cfg()
    h = _handler(cfg)
    h.hf_row['P_surf'] = 5.0
    assert terminate._check_escape(h) is False
    assert patch_statusfile == []


@pytest.mark.unit
def test_check_separation_triggers_roche_limit(patch_statusfile):
    """Disintegration: separation below Roche limit exits with status 16."""
    cfg = _cfg()
    h = _handler(cfg)
    h.hf_row['separation'] = 0.9
    h.hf_row['roche_limit'] = 1.0
    assert terminate._check_separation(h) is True
    assert patch_statusfile[-1][1] == 16


@pytest.mark.unit
def test_check_spinrate_triggers_breakup(patch_statusfile):
    """Disintegration: spin faster than breakup exits with status 16."""
    cfg = _cfg()
    h = _handler(cfg)
    h.hf_row['axial_period'] = 4.0
    h.hf_row['breakup_period'] = 5.0
    assert terminate._check_spinrate(h) is True
    assert patch_statusfile[-1][1] == 16


@pytest.mark.unit
def test_check_maxtime_triggers(patch_statusfile):
    """Time limit: exceeding maximum time exits with status 13."""
    cfg = _cfg()
    h = _handler(cfg)
    h.hf_row['Time'] = 200.0
    assert terminate._check_maxtime(h) is True
    assert patch_statusfile[-1][1] == 13


@pytest.mark.unit
def test_check_mintime_blocks_until_min_elapsed(patch_statusfile):
    """Minimum time: exit is blocked and status 1 written if finished early."""
    cfg = _cfg()
    cfg.params.stop.time.minimum = 10.0
    h = _handler(cfg)
    h.hf_row['Time'] = 5.0
    assert terminate._check_mintime(h, finished=True) is False
    assert patch_statusfile[-1][1] == 1


@pytest.mark.unit
def test_check_mintime_allows_when_above_min(patch_statusfile):
    """Minimum time: when above minimum, allow exit without touching statusfile."""
    cfg = _cfg()
    cfg.params.stop.time.minimum = 1.0
    h = _handler(cfg)
    h.hf_row['Time'] = 2.0
    assert terminate._check_mintime(h, finished=True) is True
    assert patch_statusfile == []


@pytest.mark.unit
def test_check_maxiter_triggers_when_exceeding_limit(patch_statusfile):
    """Iteration cap: exceeding loop limit exits with status 12."""
    cfg = _cfg()
    h = _handler(cfg)
    h.loops['total'] = 6
    assert terminate._check_maxiter(h) is True
    assert patch_statusfile[-1][1] == 12


@pytest.mark.unit
def test_check_miniter_blocks_when_below_minimum(patch_statusfile):
    """Iteration floor: block exit and write status 1 when below minimum loops."""
    cfg = _cfg()
    h = _handler(cfg)
    h.loops['total'] = 1
    assert terminate._check_miniter(h, finished=True) is False
    assert patch_statusfile[-1][1] == 1


@pytest.mark.unit
def test_check_miniter_allows_when_above_minimum(patch_statusfile):
    """Iteration floor: allow exit silently once minimum loop count reached."""
    cfg = _cfg()
    h = _handler(cfg)
    h.loops['total'] = 3
    assert terminate._check_miniter(h, finished=True) is True
    assert patch_statusfile == []


@pytest.mark.unit
def test_check_keepalive_triggers_when_file_missing(monkeypatch, patch_statusfile):
    """Keepalive: missing lockfile forces termination with status 25."""
    cfg = _cfg()
    h = _handler(cfg)
    monkeypatch.setattr(terminate.os.path, 'exists', lambda _: False)
    assert terminate._check_keepalive(h) is True
    assert patch_statusfile[-1][1] == 25


@pytest.mark.unit
def test_check_keepalive_returns_false_when_present(monkeypatch, patch_statusfile):
    """Keepalive: existing lockfile leaves simulation running."""
    cfg = _cfg()
    h = _handler(cfg)
    monkeypatch.setattr(terminate.os.path, 'exists', lambda _: True)
    assert terminate._check_keepalive(h) is False
    assert patch_statusfile == []


@pytest.mark.unit
def test_check_termination_non_strict_exits_on_first_success(monkeypatch, patch_statusfile):
    """Non-strict: any single criterion triggers immediate exit and flags."""
    cfg = _cfg()
    h = _handler(cfg)
    h.hf_row['Time'] = 200.0  # triggers max time
    monkeypatch.setattr(terminate.os.path, 'exists', lambda _: True)
    assert terminate.check_termination(h) is True
    assert h.finished_both is True
    assert h.finished_prev is True
    assert patch_statusfile[-1][1] == 13


@pytest.mark.unit
def test_check_termination_strict_requires_two_iterations(monkeypatch, patch_statusfile):
    """Strict mode: requires two consecutive satisfied iterations to exit."""
    cfg = _cfg()
    cfg.params.stop.strict = True
    h = _handler(cfg)
    h.hf_row['Time'] = 200.0
    monkeypatch.setattr(terminate.os.path, 'exists', lambda _: True)

    # First pass sets finished_prev but does not exit
    assert terminate.check_termination(h) is False
    assert h.finished_prev is True
    # Second pass with same condition should exit
    assert terminate.check_termination(h) is True
    assert h.finished_both is True
    assert patch_statusfile[-1][1] == 13


@pytest.mark.unit
def test_check_termination_strict_resets_if_condition_lost(monkeypatch, patch_statusfile):
    """Strict mode: losing condition resets finished_prev and continues run."""
    cfg = _cfg()
    cfg.params.stop.strict = True
    h = _handler(cfg)
    h.hf_row['Time'] = 200.0
    monkeypatch.setattr(terminate.os.path, 'exists', lambda _: True)

    assert terminate.check_termination(h) is False
    # Condition removed
    h.hf_row['Time'] = 0.0
    assert terminate.check_termination(h) is False
    assert h.finished_prev is False
    # No new statusfile entries added beyond initial attempt
    assert patch_statusfile[-1][1] == 13
