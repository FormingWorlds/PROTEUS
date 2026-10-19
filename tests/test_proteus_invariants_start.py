"""Tests for Proteus.start lifecycle under run-row physical invariant enforcement.

Covers:
- Status family 31 (mass conservation invariant stopping violation)
- Status family 32 (state bounds invariant stopping violation)
- Enforcement mode 'off' execution vs main run
- Resume reading helpfile written and flushed to disk by a prior stop
- Multiple invariant violations on a single row (primary status resolution)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import proteus.proteus as proteus_mod
from proteus import Proteus
from proteus.interior_energetics.common import Interior_t
from proteus.utils.invariants import (
    InvariantViolation,
    InvariantViolationError,
    handle_violations,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _build_dummy_runner(
    tmp_path: Path, run_name: str, max_iters: int = 3
) -> tuple[Proteus, Path, Path]:
    """Helper to construct a lightweight all-dummy Proteus runner isolated in tmp_path."""
    out_dir = tmp_path / run_name
    dummy_toml = Path(__file__).resolve().parents[1] / 'input' / 'dummy.toml'
    text = dummy_toml.read_text(encoding='utf-8')
    cfg_path = tmp_path / f'{run_name}.toml'
    cfg_path.write_text(
        text.replace('path = "auto"', f'path = "{out_dir}"', 1), encoding='utf-8'
    )

    runner = Proteus(config_path=cfg_path)
    runner.config.params.stop.solid.enabled = False
    runner.config.params.stop.iters.minimum = 1
    runner.config.params.stop.iters.maximum = max_iters
    runner.config.params.stop.time.minimum = 0.0
    runner.config.params.stop.time.maximum = 500.0
    runner.config.params.out.plot_mod = None
    runner.config.params.out.archive_mod = 'none'
    return runner, cfg_path, out_dir


def test_proteus_start_status_family_31_mass_invariant(tmp_path, monkeypatch):
    """Verify Proteus.start writes status 31 and flushes helpfile on mass invariant violation."""
    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    runner, _, out_dir = _build_dummy_runner(tmp_path, 'run_status_31', max_iters=3)
    runner.config.invariants.mode = 'stop'

    def synthetic_check_row(hf_row, config, **kwargs):
        v = InvariantViolation(
            name='M_planet_bounds',
            status_code=31,
            value=6.5e24,
            tolerance=5.97e24,
            message='Planet mass exceeds upper bound',
            report_only=False,
        )
        handle_violations(
            [v],
            mode='stop',
            directories=kwargs.get('directories'),
            tracker=kwargs.get('tracker'),
        )
        return [v]

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', synthetic_check_row)

    with pytest.raises(InvariantViolationError) as exc_info:
        runner.start(resume=False, offline=True)

    assert exc_info.value.status_code == 31
    assert exc_info.value.invariant_name == 'M_planet_bounds'
    assert exc_info.value.violations[0].status_code == 31
    assert exc_info.value.violations[0].name == 'M_planet_bounds'

    status_file = out_dir / 'status'
    assert status_file.is_file()
    status_lines = status_file.read_text(encoding='utf-8').splitlines()
    assert status_lines[0].strip() == '31'
    assert 'Mass conservation invariant' in status_lines[1]

    helpfile = out_dir / 'runtime_helpfile.csv'
    assert helpfile.is_file()
    df = pd.read_csv(helpfile, sep=None, engine='python')
    assert len(df) >= 1


def test_proteus_start_status_family_32_state_bounds_invariant(tmp_path, monkeypatch):
    """Verify Proteus.start writes status 32 and flushes helpfile on state bounds violation."""
    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    runner, _, out_dir = _build_dummy_runner(tmp_path, 'run_status_32', max_iters=3)
    runner.config.invariants.mode = 'stop'

    def synthetic_check_row(hf_row, config, **kwargs):
        v = InvariantViolation(
            name='T_surf_bounds',
            status_code=32,
            value=-15.0,
            tolerance=0.0,
            message='Surface temperature below absolute zero',
            report_only=False,
        )
        handle_violations(
            [v],
            mode='stop',
            directories=kwargs.get('directories'),
            tracker=kwargs.get('tracker'),
        )
        return [v]

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', synthetic_check_row)

    with pytest.raises(InvariantViolationError) as exc_info:
        runner.start(resume=False, offline=True)

    assert exc_info.value.status_code == 32
    assert exc_info.value.invariant_name == 'T_surf_bounds'
    assert exc_info.value.violations[0].status_code == 32
    assert exc_info.value.violations[0].name == 'T_surf_bounds'

    status_file = out_dir / 'status'
    assert status_file.is_file()
    status_lines = status_file.read_text(encoding='utf-8').splitlines()
    assert status_lines[0].strip() == '32'
    assert 'State bounds invariant' in status_lines[1]

    helpfile = out_dir / 'runtime_helpfile.csv'
    assert helpfile.is_file()
    df = pd.read_csv(helpfile, sep=None, engine='python')
    assert len(df) >= 1


def test_proteus_start_mode_off_vs_main(tmp_path, monkeypatch):
    """Verify mode 'off' bypasses invariant evaluations and completes cleanly."""
    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    runner, _, out_dir = _build_dummy_runner(tmp_path, 'run_mode_off', max_iters=2)
    runner.config.invariants.mode = 'off'

    real_check_row = proteus_mod.check_row_invariants
    call_count = 0

    def spy_check_row(hf_row, config, **kwargs):
        nonlocal call_count
        call_count += 1
        res = real_check_row(hf_row, config, **kwargs)
        assert res == []
        return res

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_check_row)

    runner.start(resume=False, offline=True)

    assert call_count >= 2
    status_file = out_dir / 'status'
    assert status_file.is_file()
    status_lines = status_file.read_text(encoding='utf-8').splitlines()
    assert status_lines[0].strip() == '12'
    assert 'Completed' in status_lines[1]

    helpfile = out_dir / 'runtime_helpfile.csv'
    assert helpfile.is_file()
    df = pd.read_csv(helpfile, sep=None, engine='python')
    assert len(df) == 4


def test_proteus_start_resume_reads_helpfile_written_by_prior_stop(tmp_path, monkeypatch):
    """Verify Proteus.start(resume=True) cleanly reads helpfile flushed by prior invariant stop."""
    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    runner, cfg_path, out_dir = _build_dummy_runner(tmp_path, 'run_resume_stop', max_iters=6)
    runner.config.invariants.mode = 'stop'

    loop_count = 0

    def intermittent_check_row(hf_row, config, **kwargs):
        nonlocal loop_count
        loop_count += 1
        if loop_count == 5:
            v = InvariantViolation(
                name='M_planet_bounds',
                status_code=31,
                value=7.0e24,
                tolerance=5.97e24,
                message='Stopping violation on loop 5',
                report_only=False,
            )
            handle_violations(
                [v],
                mode='stop',
                directories=kwargs.get('directories'),
                tracker=kwargs.get('tracker'),
            )
        return []

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', intermittent_check_row)

    with pytest.raises(InvariantViolationError):
        runner.start(resume=False, offline=True)

    status_file = out_dir / 'status'
    assert status_file.read_text(encoding='utf-8').splitlines()[0].strip() == '31'
    helpfile = out_dir / 'runtime_helpfile.csv'
    assert helpfile.is_file()
    initial_rows = len(pd.read_csv(helpfile, sep=None, engine='python'))
    assert initial_rows == 5

    # Resume the stopped run with invariants mode relaxed to 'off'
    monkeypatch.undo()
    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)

    resume_runner = Proteus(config_path=cfg_path)
    resume_runner.config.invariants.mode = 'off'
    resume_runner.config.params.stop.solid.enabled = False
    resume_runner.config.params.stop.iters.minimum = 1
    resume_runner.config.params.stop.iters.maximum = 7
    resume_runner.config.params.stop.time.minimum = 0.0
    resume_runner.config.params.stop.time.maximum = 500.0
    resume_runner.config.params.out.plot_mod = None
    resume_runner.config.params.out.archive_mod = 'none'

    resume_runner.start(resume=True, offline=True)

    final_status = status_file.read_text(encoding='utf-8').splitlines()
    assert final_status[0].strip() == '12'
    assert 'Completed' in final_status[1]

    df_final = pd.read_csv(helpfile, sep=None, engine='python')
    assert len(df_final) > initial_rows


def test_proteus_start_row_failing_two_families_primary_status(tmp_path, monkeypatch):
    """Verify primary status selection when a row fails both status family 31 and 32."""
    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    runner, _, out_dir = _build_dummy_runner(tmp_path, 'run_multi_family', max_iters=2)
    runner.config.invariants.mode = 'stop'

    v_mass = InvariantViolation(
        name='M_planet_bounds',
        status_code=31,
        value=6.5e24,
        tolerance=5.97e24,
        message='Mass invariant failure',
        report_only=False,
    )
    v_bounds = InvariantViolation(
        name='T_surf_bounds',
        status_code=32,
        value=-10.0,
        tolerance=0.0,
        message='State bounds invariant failure',
        report_only=False,
    )

    def multi_violation_check_row(hf_row, config, **kwargs):
        handle_violations(
            [v_mass, v_bounds],
            mode='stop',
            directories=kwargs.get('directories'),
            tracker=kwargs.get('tracker'),
        )
        return [v_mass, v_bounds]

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', multi_violation_check_row)

    with pytest.raises(InvariantViolationError) as exc_info:
        runner.start(resume=False, offline=True)

    assert exc_info.value.status_code == 31
    assert exc_info.value.invariant_name == 'M_planet_bounds'
    assert len(exc_info.value.violations) == 2
    assert exc_info.value.violations[0].status_code == 31
    assert exc_info.value.violations[1].status_code == 32

    status_file = out_dir / 'status'
    assert status_file.is_file()
    status_lines = status_file.read_text(encoding='utf-8').splitlines()
    assert status_lines[0].strip() == '31'
    assert 'Mass conservation invariant' in status_lines[1]

    # Discrimination canary: if state bounds is first, primary status must be 32
    def reversed_violation_check_row(hf_row, config, **kwargs):
        handle_violations(
            [v_bounds, v_mass],
            mode='stop',
            directories=kwargs.get('directories'),
            tracker=kwargs.get('tracker'),
        )
        return [v_bounds, v_mass]

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', reversed_violation_check_row)
    with pytest.raises(InvariantViolationError) as exc_canary:
        runner.start(resume=False, offline=True)

    assert exc_canary.value.status_code == 32
    assert exc_canary.value.invariant_name == 'T_surf_bounds'
    assert exc_canary.value.violations[0].status_code == 32
    assert exc_canary.value.violations[1].status_code == 31
    status_lines_canary = status_file.read_text(encoding='utf-8').splitlines()
    assert status_lines_canary[0].strip() == '32'
    assert 'State bounds invariant' in status_lines_canary[1]
