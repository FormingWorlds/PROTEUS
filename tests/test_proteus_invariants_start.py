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
            value=0.0,
            tolerance=0.0,
            message='Time=1.000000e+00 yr: M_planet=0.000000e+00 kg violates M_planet > 0.0 kg',
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
            message='Time=1.000000e+00 yr: T_surf=-1.500000e+01 K violates T_surf > 0.0 K',
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

    real_check_row = proteus_mod.check_row_invariants
    loop_count = 0

    def intermittent_check_row(hf_row, config, **kwargs):
        nonlocal loop_count
        loop_count += 1
        if loop_count == 5:
            v = InvariantViolation(
                name='M_planet_bounds',
                status_code=31,
                value=-1.0e24,
                tolerance=0.0,
                message='Time=1.000000e+02 yr: M_planet=-1.000000e+24 kg violates M_planet > 0.0 kg',
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
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', real_check_row)

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
        value=0.0,
        tolerance=0.0,
        message='Time=1.000000e+00 yr: M_planet=0.000000e+00 kg violates M_planet > 0.0 kg',
        report_only=False,
    )
    v_bounds = InvariantViolation(
        name='T_surf_bounds',
        status_code=32,
        value=-10.0,
        tolerance=0.0,
        message='Time=1.000000e+00 yr: T_surf=-1.000000e+01 K violates T_surf > 0.0 K',
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


def test_proteus_start_passes_init_stage_and_interior_o_to_check_row_invariants(
    tmp_path, monkeypatch
):
    """Proteus.start passes init_stage (True on init rows, False after) and runner.interior_o to check_row_invariants."""
    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    runner, _, out_dir = _build_dummy_runner(tmp_path, 'run_init_stage_spy', max_iters=5)
    runner.config.invariants.mode = 'report'

    real_check_row = proteus_mod.check_row_invariants
    recorded_calls: list[tuple[int, bool | None, object]] = []

    def spy_check_row(hf_row, config, **kwargs):
        recorded_calls.append(
            (runner.loops['total'], kwargs.get('init_stage'), kwargs.get('interior_o'))
        )
        return real_check_row(hf_row, config, **kwargs)

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_check_row)
    runner.start(resume=False, offline=True)

    init_loops = runner.loops['init_loops']
    assert len(recorded_calls) >= init_loops + 3
    for loop_idx, is_init, int_o in recorded_calls:
        assert int_o is runner.interior_o
        if loop_idx <= init_loops + 1:
            assert is_init is True
        else:
            assert is_init is False


def test_proteus_start_assert_mass_conservation_writes_status_31(tmp_path, monkeypatch):
    """Proteus.start writes status 31 in its except RuntimeError block when assert_mass_conservation fails."""
    import proteus.outgas.wrapper as outgas_wrapper

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    runner, _, out_dir = _build_dummy_runner(tmp_path, 'run_coupler_status_31', max_iters=2)
    real_outgas = outgas_wrapper.run_outgassing

    def corrupt_outgas(dirs, config, hf_row, *args, **kwargs):
        real_outgas(dirs, config, hf_row, *args, **kwargs)
        hf_row['M_atm'] = 1.0e26

    monkeypatch.setattr(outgas_wrapper, 'run_outgassing', corrupt_outgas)
    with pytest.raises(RuntimeError, match='exceeds M_planet'):
        runner.start(resume=False, offline=True)

    status_lines = (out_dir / 'status').read_text(encoding='utf-8').splitlines()
    assert status_lines[0].strip() == '31'
    assert 'Mass conservation invariant' in status_lines[1]


def _setup_zalmoxis_baseline_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_name: str,
    *,
    max_iters: int = 6,
    temperature_mode: str = 'isothermal',
    update_interval: float = 0.0,
    mode: str = 'stop',
) -> tuple[Proteus, Path, Path]:
    """Build a Zalmoxis dummy-interior runner isolated in its own directory."""
    import proteus.interior_energetics.wrapper as int_wrap

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    monkeypatch.setattr(Proteus, '_require_paleos_tables', lambda self: None)
    runner, cfg_path, out_dir = _build_dummy_runner(tmp_path, run_name, max_iters=max_iters)
    runner.config.invariants.mode = mode
    runner.config.interior_struct.module = 'zalmoxis'
    runner.config.interior_struct.zalmoxis.equilibrate_init = False
    runner.config.interior_struct.zalmoxis.update_interval = update_interval
    runner.config.planet.temperature_mode = temperature_mode

    def fake_solve_structure(dirs, config, hf_all, hf_row, outdir, thermal_solve=True):
        int_wrap.determine_interior_radius_with_dummy(
            dirs, config, hf_all, hf_row, outdir, thermal_solve=thermal_solve
        )
        hf_row['M_int_target'] = hf_row['M_int']
        hf_row['M_core_target'] = hf_row['M_core']
        hf_row['_structure_rollback'] = False

    monkeypatch.setattr(int_wrap, 'solve_structure', fake_solve_structure)
    return runner, cfg_path, out_dir


def test_proteus_start_zalmoxis_failed_baseline_keeps_row_4_init_stage_true(
    tmp_path, monkeypatch
):
    """Proteus.start with zalmoxis keeps init_stage=True on row 4 when the forced baseline fails, then clears it on row 5 when the retry succeeds."""
    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        resume_baseline_structure_done,
    )

    runner, _, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path, monkeypatch, 'run_failed_baseline_row4', max_iters=6
    )

    def fake_solve_structure(dirs, config, hf_all, hf_row, outdir, thermal_solve=True):
        int_wrap.determine_interior_radius_with_dummy(
            dirs, config, hf_all, hf_row, outdir, thermal_solve=thermal_solve
        )
        hf_row['M_int_target'] = hf_row['M_int']
        hf_row['M_core_target'] = hf_row['M_core']
        hf_row['_structure_rollback'] = False
        Path(get_file_structure_rollback(outdir)).write_text('1\n', encoding='utf-8')
        Path(get_file_baseline_structure_done(outdir)).write_text('1\n', encoding='utf-8')

    monkeypatch.setattr(int_wrap, 'solve_structure', fake_solve_structure)
    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)

    baseline_attempts = 0

    def fake_update_structure(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        nonlocal baseline_attempts
        baseline_attempts += 1
        if baseline_attempts == 1:
            Path(get_file_baseline_structure_done(dirs['output'])).unlink(missing_ok=True)
            interior_o.structure_stale = True
            interior_o.structure_rollback_active = True
            hf_row['_structure_rollback'] = True
            interior_o.write_structure_stale(dirs['output'])
            return lt, lT, lP
        interior_o.structure_stale = False
        interior_o.structure_rollback_active = False
        hf_row['_structure_rollback'] = False
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        hf_row['struct_mass_desync_frac'] = 1.0e-3
        interior_o.write_structure_stale(dirs['output'])
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', fake_update_structure)

    real_interior = int_wrap.run_interior

    def interior_with_init_desync(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        if runner.loops['total'] <= 4:
            hf_row['struct_mass_desync_frac'] = 1.123e-2
        else:
            hf_row['struct_mass_desync_frac'] = 1.0e-3

    monkeypatch.setattr(int_wrap, 'run_interior', interior_with_init_desync)

    real_check_row = proteus_mod.check_row_invariants
    recorded_init_stages: list[tuple[int, bool | None, bool]] = []

    def spy_check_row(hf_row, config, **kwargs):
        recorded_init_stages.append(
            (
                runner.loops['total'],
                kwargs.get('init_stage'),
                bool(runner._baseline_structure_done),
            )
        )
        if runner.loops['total'] == 1:
            assert (
                Path(get_file_structure_rollback(str(out_dir)))
                .read_text(encoding='utf-8')
                .strip()
                == '0 0'
            )
            assert (
                Path(get_file_baseline_structure_done(str(out_dir)))
                .read_text(encoding='utf-8')
                .strip()
                == '0 0'
            )
        if runner.loops['total'] == 5:
            baseline_file = Path(get_file_baseline_structure_done(str(out_dir)))
            assert baseline_file.is_file()
            assert baseline_file.read_text(encoding='utf-8').strip() == '0 4'
            assert resume_baseline_structure_done(str(out_dir)) is False
        return real_check_row(hf_row, config, **kwargs)

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_check_row)
    runner.start(resume=False, offline=True)

    stage_by_loop = {
        loop_idx: (is_init, done) for loop_idx, is_init, done in recorded_init_stages
    }
    assert stage_by_loop[4] == (True, False)
    assert stage_by_loop[5] == (True, False)
    assert stage_by_loop[6] == (False, True)
    assert (
        Path(get_file_baseline_structure_done(str(out_dir))).read_text(encoding='utf-8').strip()
        == '1 5'
    )
    assert resume_baseline_structure_done(str(out_dir)) is True


def test_proteus_start_zalmoxis_resume_without_baseline_file_treats_baseline_as_done(
    tmp_path, monkeypatch
):
    """An older run directory with single-token flag files and no baseline_structure_done.dat treats the baseline as done on a normal resume."""
    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path, monkeypatch, 'run_missing_baseline_file', max_iters=5
    )
    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)

    def init_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        return lt, lT, lP

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', init_update)
    runner.start(resume=False, offline=True)

    Path(get_file_baseline_structure_done(str(out_dir))).unlink(missing_ok=True)
    Path(get_file_structure_stale(str(out_dir))).write_text('1\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(out_dir))).write_text('0\n', encoding='utf-8')

    missing_file_resume, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path, monkeypatch, 'run_missing_baseline_file', max_iters=6
    )
    missing_file_resume.config_path = cfg_path

    missing_file_forced_calls: list[bool] = []
    missing_file_stages: list[tuple[bool | None, bool]] = []
    real_check_row = proteus_mod.check_row_invariants

    def spy_missing_file_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        missing_file_forced_calls.append(bool(force))
        return lt, lT, lP

    def spy_missing_file_check_row(hf_row, config, **kwargs):
        missing_file_stages.append(
            (kwargs.get('init_stage'), bool(missing_file_resume._baseline_structure_done))
        )
        return real_check_row(hf_row, config, **kwargs)

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', spy_missing_file_update)
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_missing_file_check_row)

    missing_file_resume.start(resume=True, offline=True)
    assert missing_file_resume._baseline_structure_done is True
    assert missing_file_forced_calls == []
    assert missing_file_stages[0] == (False, True)


def test_proteus_start_zalmoxis_resume_retries_uncommitted_baseline_when_file_holds_zero(
    tmp_path, monkeypatch
):
    """A run directory where baseline_structure_done.dat holds 0 at the resume row retries the baseline until it succeeds."""
    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        resume_baseline_structure_done,
        write_baseline_structure_done,
    )

    runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path, monkeypatch, 'run_uncommitted_baseline_resume', max_iters=5
    )
    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)

    def init_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        return lt, lT, lP

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', init_update)
    runner.start(resume=False, offline=True)

    write_baseline_structure_done(str(out_dir), False, row_index=5)
    runner.interior_o.structure_stale = True
    runner.interior_o.structure_rollback_active = True
    runner.interior_o.write_structure_stale(str(out_dir), row_index=5)

    resume_runner, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path, monkeypatch, 'run_uncommitted_baseline_resume', max_iters=7
    )
    resume_runner.config_path = cfg_path

    resumed_stages: list[tuple[bool | None, bool]] = []
    real_check_row = proteus_mod.check_row_invariants

    def spy_resumed_check_row(hf_row, config, **kwargs):
        resumed_stages.append(
            (kwargs.get('init_stage'), bool(resume_runner._baseline_structure_done))
        )
        return real_check_row(hf_row, config, **kwargs)

    resumed_baseline_calls = 0

    def fake_resumed_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        nonlocal resumed_baseline_calls
        resumed_baseline_calls += 1
        if resumed_baseline_calls == 1:
            interior_o.structure_stale = True
            interior_o.structure_rollback_active = True
            hf_row['struct_mass_desync_frac'] = 1.123e-2
            return lt, lT, lP
        interior_o.structure_stale = False
        interior_o.structure_rollback_active = False
        hf_row['_structure_rollback'] = False
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        hf_row['struct_mass_desync_frac'] = 1.0e-3
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', fake_resumed_update)
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_resumed_check_row)

    resume_runner.start(resume=True, offline=True)
    assert resumed_stages[0] == (True, False)
    assert resumed_stages[1] == (False, True)
    assert resume_baseline_structure_done(str(out_dir)) is True
    assert (
        Path(get_file_baseline_structure_done(str(out_dir))).read_text(encoding='utf-8').strip()
        == '1 7'
    )
    assert (
        Path(get_file_structure_rollback(str(out_dir))).read_text(encoding='utf-8').strip()
        == '0 7'
    )


def test_proteus_start_zalmoxis_dropped_snapshot_newer_event_forces_resolve_and_clears_on_success(
    tmp_path, monkeypatch
):
    """When a dropped snapshot leaves a newer flag event on disk, Proteus sets rollback=True and baseline-done=False on entry and clears rollback when the forced re-solve succeeds."""
    import proteus.interior_energetics.wrapper as int_wrap
    import proteus.utils.coupler as coupler_mod
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path, monkeypatch, 'run_dropped_newer_success', max_iters=6
    )
    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)

    def write_flags_on_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        interior_o.structure_stale = False
        interior_o.structure_rollback_active = False
        hf_row['_structure_rollback'] = False
        interior_o.write_structure_stale(dirs['output'])
        return lt, lT, lP

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', write_flags_on_update)
    runner.start(resume=False, offline=True)

    # Simulate a structure event written on row 6 (the last row), which select_resumable_snapshot drops:
    Path(get_file_structure_stale(str(out_dir))).write_text('0 6\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(out_dir))).write_text('0 6\n', encoding='utf-8')
    Path(get_file_baseline_structure_done(str(out_dir))).write_text('1 6\n', encoding='utf-8')

    real_select = coupler_mod.select_resumable_snapshot

    def fake_select_drops_snapshot(outdir, hf_all, **kwargs):
        trimmed, _ = real_select(outdir, hf_all, **kwargs)
        return trimmed.iloc[:-1].copy(), [float(trimmed.iloc[-1]['Time'])]

    monkeypatch.setattr(coupler_mod, 'select_resumable_snapshot', fake_select_drops_snapshot)

    resume_entry_files: list[tuple[str, str]] = []

    def fake_succeeding_baseline(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        resume_entry_files.append(
            (
                Path(get_file_structure_rollback(dirs['output']))
                .read_text(encoding='utf-8')
                .strip(),
                Path(get_file_baseline_structure_done(dirs['output']))
                .read_text(encoding='utf-8')
                .strip(),
            )
        )
        interior_o.structure_stale = False
        hf_row['struct_mass_desync_frac'] = 1.0e-3
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    real_interior = int_wrap.run_interior

    def interior_with_target_mismatch(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        hf_row['struct_mass_desync_frac'] = 1.0e-3
        hf_row['M_int'] = 5.97e24
        hf_row['M_int_target'] = 5.97e24 * 1.02

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', fake_succeeding_baseline)
    monkeypatch.setattr(int_wrap, 'run_interior', interior_with_target_mismatch)

    resumed, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path, monkeypatch, 'run_dropped_newer_success', max_iters=7
    )
    resumed.config_path = cfg_path

    with pytest.raises(InvariantViolationError) as exc_after_baseline_clear:
        resumed.start(resume=True, offline=True)
    assert resume_entry_files == [('1 5', '0 5')]
    assert exc_after_baseline_clear.value.invariant_name == 'M_int_target_rel_err'
    assert resumed.interior_o.structure_rollback_active is False
    assert resumed._baseline_structure_done is True
    assert (
        Path(get_file_structure_rollback(str(out_dir))).read_text(encoding='utf-8').strip()
        == '0 6'
    )
    assert (
        Path(get_file_baseline_structure_done(str(out_dir))).read_text(encoding='utf-8').strip()
        == '1 6'
    )


def test_proteus_start_zalmoxis_dropped_snapshot_with_older_indexed_flags_skips_forced_resolve(
    tmp_path, monkeypatch
):
    """When a resume drops a snapshot and all flag files carry row indices at or before the resume row, rollback stays False and no forced re-solve runs."""
    import proteus.interior_energetics.wrapper as int_wrap
    import proteus.utils.coupler as coupler_mod
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path, monkeypatch, 'run_dropped_older_indexed', max_iters=6, mode='report'
    )
    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)

    def init_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        return lt, lT, lP

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', init_update)
    runner.start(resume=False, offline=True)

    # All 3 flag files hold row 4 <= resume_row 5 (after dropping row 6):
    Path(get_file_structure_stale(str(out_dir))).write_text('0 4\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(out_dir))).write_text('0 4\n', encoding='utf-8')
    Path(get_file_baseline_structure_done(str(out_dir))).write_text('1 4\n', encoding='utf-8')

    real_select = coupler_mod.select_resumable_snapshot

    def fake_select_drops_snapshot(outdir, hf_all, **kwargs):
        trimmed, _ = real_select(outdir, hf_all, **kwargs)
        return trimmed.iloc[:-1].copy(), [float(trimmed.iloc[-1]['Time'])]

    monkeypatch.setattr(coupler_mod, 'select_resumable_snapshot', fake_select_drops_snapshot)

    no_newer_calls: list[bool] = []

    def spy_no_newer_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        no_newer_calls.append(bool(force))
        return lt, lT, lP

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', spy_no_newer_update)

    no_newer_resume, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path, monkeypatch, 'run_dropped_older_indexed', max_iters=7, mode='report'
    )
    no_newer_resume.config_path = cfg_path
    no_newer_resume.start(resume=True, offline=True)
    assert no_newer_resume.interior_o.structure_rollback_active is False
    assert no_newer_resume._baseline_structure_done is True
    assert no_newer_calls == []


def test_proteus_start_zalmoxis_ahead_of_helpfile_flag_event_forces_resolve_on_resume(
    tmp_path, monkeypatch
):
    """A kill between the flag write and the helpfile write (dropped_snapshots == [], flag row > resume_row) forces 1 re-solve at resume and clears rollback on success."""
    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path, monkeypatch, 'run_ahead_flag_event', max_iters=5, mode='report'
    )
    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)

    def init_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        return lt, lT, lP

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', init_update)
    runner.start(resume=False, offline=True)

    # Helpfile has 6 rows (0..5, resume_row=5); flags record row 6 > 5:
    Path(get_file_structure_stale(str(out_dir))).write_text('0 6\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(out_dir))).write_text('0 6\n', encoding='utf-8')
    Path(get_file_baseline_structure_done(str(out_dir))).write_text('1 4\n', encoding='utf-8')

    ahead_forced_calls: list[tuple[bool, bool]] = []

    def spy_ahead_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        ahead_forced_calls.append((bool(force), bool(interior_o.structure_rollback_active)))
        interior_o.structure_stale = False
        hf_row['struct_mass_desync_frac'] = 1.0e-3
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', spy_ahead_update)

    ahead_resume, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path, monkeypatch, 'run_ahead_flag_event', max_iters=6, mode='report'
    )
    ahead_resume.config_path = cfg_path
    ahead_resume.start(resume=True, offline=True)
    assert ahead_forced_calls == [(True, True)]
    assert ahead_resume.interior_o.structure_rollback_active is False
    assert ahead_resume._baseline_structure_done is True
    assert (
        Path(get_file_structure_rollback(str(out_dir))).read_text(encoding='utf-8').strip()
        == '0 6'
    )


@pytest.mark.parametrize('temperature_mode', ['adiabatic_from_cmb', 'liquidus_super'])
def test_proteus_start_resume_dropped_snapshots_with_unindexed_flag_file_sets_rollback(
    tmp_path, monkeypatch, temperature_mode
):
    """An unindexed directory with dropped snapshots sets rollback at resume, forces 1 re-solve, and checks M_int/M_core afterwards, matching the continuous run."""
    import proteus.interior_energetics.wrapper as int_wrap
    import proteus.utils.coupler as coupler_mod
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    monkeypatch.setattr(
        int_wrap,
        '_use_superliquidus_adiabat_ic',
        lambda cfg: cfg.planet.temperature_mode == 'liquidus_super',
    )

    def fake_accepting_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        interior_o.structure_stale = False
        interior_o.structure_rollback_active = False
        hf_row['_structure_rollback'] = False
        interior_o.write_structure_stale(dirs['output'])
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    real_interior = int_wrap.run_interior
    real_check_row = proteus_mod.check_row_invariants

    # 1. Continuous reference run (rows 0..6 in stop mode, with a 2% M_int_target error injected on row 6):
    cont_runner, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        f'run_arch_r3_cont_{temperature_mode}',
        max_iters=6,
        temperature_mode=temperature_mode,
        update_interval=0.0,
        mode='stop',
    )

    def cont_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        hf_row['M_int'] = 5.97e24
        hf_row['M_int_target'] = (
            5.97e24 * 1.02 if hf_all is not None and len(hf_all) == 6 else 5.97e24
        )

    cont_seq: list[tuple[int, bool, bool]] = []

    def spy_cont_check(hf_row, config, **kwargs):
        row_i = len(cont_runner.hf_all) - 1
        if row_i >= 4:
            cont_seq.append(
                (
                    row_i,
                    bool(kwargs.get('init_stage')),
                    bool(cont_runner.interior_o.structure_rollback_active),
                )
            )
        return real_check_row(hf_row, config, **kwargs)

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', fake_accepting_update)
    monkeypatch.setattr(int_wrap, 'run_interior', cont_interior)
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_cont_check)

    cont_stop_exc: InvariantViolationError | None = None
    try:
        cont_runner.start(resume=False, offline=True)
    except InvariantViolationError as exc:
        cont_stop_exc = exc
    assert cont_stop_exc is not None
    assert cont_stop_exc.invariant_name == 'M_int_target_rel_err'
    assert cont_seq == [(4, False, False), (5, False, False), (6, False, False)]

    # 2. Resumed run from an arch_r3_front-like directory with a dropped snapshot (row 6 dropped -> resume_row 5):
    monkeypatch.setattr(int_wrap, 'run_interior', real_interior)
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', real_check_row)
    runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        f'run_arch_r3_front_like_{temperature_mode}',
        max_iters=6,
        temperature_mode=temperature_mode,
        update_interval=0.0,
        mode='report',
    )
    runner.start(resume=False, offline=True)

    # Match arch_r3_front layout: only structure_stale.dat '0\n' on disk, no row index, no other flag files:
    Path(get_file_structure_stale(str(out_dir))).write_text('0\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(out_dir))).unlink(missing_ok=True)
    Path(get_file_baseline_structure_done(str(out_dir))).unlink(missing_ok=True)
    (out_dir / 'data' / 'structure_forced_pending.dat').unlink(missing_ok=True)

    real_select = coupler_mod.select_resumable_snapshot

    def fake_select_drops_snapshot(outdir, hf_all, **kwargs):
        trimmed, _ = real_select(outdir, hf_all, **kwargs)
        return trimmed.iloc[:-1].copy(), [float(trimmed.iloc[-1]['Time'])]

    monkeypatch.setattr(coupler_mod, 'select_resumable_snapshot', fake_select_drops_snapshot)

    forced_calls: list[bool] = []
    entry_rollback_disk: list[str] = []

    def spy_forced_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        forced_calls.append(bool(force))
        entry_rollback_disk.append(
            Path(get_file_structure_rollback(dirs['output']))
            .read_text(encoding='utf-8')
            .strip()
        )
        return fake_accepting_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=force)

    def resumed_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        hf_row['M_int'] = 5.97e24
        hf_row['M_int_target'] = 5.97e24 * 1.02

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', spy_forced_update)
    monkeypatch.setattr(int_wrap, 'run_interior', resumed_interior)

    resumed_seq: list[tuple[int, bool, bool]] = [(4, False, False), (5, False, False)]
    resumed, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        f'run_arch_r3_front_like_{temperature_mode}',
        max_iters=7,
        temperature_mode=temperature_mode,
        update_interval=0.0,
        mode='stop',
    )
    resumed.config_path = cfg_path

    def spy_resumed_check(hf_row, config, **kwargs):
        resumed_seq.append(
            (
                len(resumed.hf_all) - 1,
                bool(kwargs.get('init_stage')),
                bool(resumed.interior_o.structure_rollback_active),
            )
        )
        return real_check_row(hf_row, config, **kwargs)

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_resumed_check)

    resumed_stop_exc: InvariantViolationError | None = None
    try:
        resumed.start(resume=True, offline=True)
    except InvariantViolationError as exc:
        resumed_stop_exc = exc

    assert forced_calls == [True]
    assert entry_rollback_disk == ['1 5']
    assert resumed_seq == cont_seq
    assert resumed_stop_exc is not None
    assert (
        resumed_stop_exc.invariant_name
        == cont_stop_exc.invariant_name
        == 'M_int_target_rel_err'
    )
    assert resumed.interior_o.structure_rollback_active is False
    assert (
        Path(get_file_structure_rollback(str(out_dir))).read_text(encoding='utf-8').strip()
        == '0 6'
    )


def test_proteus_start_loop_top_sets_structure_row_index_for_dynamic_resolves_and_resume(
    tmp_path, monkeypatch
):
    """The main-loop top sets interior_o._structure_row_index = len(hf_all) on every iteration so dynamic re-solves write '<flag> <len(hf_all)>' and a dropped last row forces 1 re-solve on resume."""
    import proteus.interior_energetics.wrapper as int_wrap
    import proteus.utils.coupler as coupler_mod
    from proteus.interior_energetics.common import (
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_loop_top_row_idx',
        max_iters=6,
        temperature_mode='liquidus_super',
        update_interval=1.0,
        mode='report',
    )

    written_records: list[tuple[int, str, str]] = []

    def fake_dynamic_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        interior_o.structure_stale = False
        interior_o.structure_rollback_active = False
        hf_row['_structure_rollback'] = False
        interior_o.write_structure_stale(dirs['output'])
        written_records.append(
            (
                len(runner.hf_all),
                Path(get_file_structure_stale(dirs['output']))
                .read_text(encoding='utf-8')
                .strip(),
                Path(get_file_structure_rollback(dirs['output']))
                .read_text(encoding='utf-8')
                .strip(),
            )
        )
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', fake_dynamic_update)
    runner.start(resume=False, offline=True)

    # Post-init iterations run at len(hf_all) == 4, 5, 6 and each writes '<flag> <len(hf_all)>':
    assert written_records == [
        (4, '0 4', '0 4'),
        (5, '0 5', '0 5'),
        (6, '0 6', '0 6'),
    ]

    # Drop the last row (row 6 -> resume_row 5) and resume:
    real_select = coupler_mod.select_resumable_snapshot

    def fake_select_drops_last(outdir, hf_all, **kwargs):
        trimmed, _ = real_select(outdir, hf_all, **kwargs)
        return trimmed.iloc[:-1].copy(), [float(trimmed.iloc[-1]['Time'])]

    monkeypatch.setattr(coupler_mod, 'select_resumable_snapshot', fake_select_drops_last)

    resumed_calls: list[tuple[bool, int | None]] = []
    resumed, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_loop_top_row_idx',
        max_iters=6,
        temperature_mode='liquidus_super',
        update_interval=1.0,
        mode='report',
    )
    resumed.config_path = cfg_path

    def spy_resumed_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        resumed_calls.append((bool(force), interior_o._structure_row_index))
        interior_o.structure_stale = False
        interior_o.structure_rollback_active = False
        hf_row['_structure_rollback'] = False
        interior_o.write_structure_stale(dirs['output'])
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', spy_resumed_update)
    resumed.start(resume=True, offline=True)

    # Exactly 1 forced re-solve at the resume (the dynamic call is skipped in that iteration),
    # and the loop top set _structure_row_index to 6 (len(hf_all)):
    assert resumed_calls == [(True, 6)]


@pytest.mark.parametrize('outcome', ['accept', 'upstep_reject', 'solver_failure'])
def test_proteus_start_newer_event_resume_forced_resolve_three_outcomes_with_static_structure(
    tmp_path, monkeypatch, caplog, outcome
):
    """On a newer-event resume with update_interval=0, Proteus makes 1 forced attempt across >= 9 resumed rows, restores targets on reject, and writes rollback/stale flags on failure."""
    import logging

    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        f'run_newer_resume_{outcome}',
        max_iters=5,
        temperature_mode='liquidus_super',
        update_interval=0.0,
        mode='report',
    )
    runner.start(resume=False, offline=True)

    # Seed a newer flag event at row 6 > resume_row 5 with structure_stale=1 on disk:
    Path(get_file_structure_stale(str(out_dir))).write_text('1 6\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(out_dir))).write_text('1 6\n', encoding='utf-8')
    Path(get_file_baseline_structure_done(str(out_dir))).write_text('1 4\n', encoding='utf-8')

    forced_calls: list[bool] = []

    def fake_forced_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        forced_calls.append(bool(force))
        if outcome == 'accept':
            interior_o.structure_stale = False
            interior_o.structure_rollback_active = False
            hf_row['_structure_rollback'] = False
            interior_o.write_structure_stale(dirs['output'])
        elif outcome == 'upstep_reject':
            # Mirror update_structure_from_interior up-step reject: zalmoxis writes new
            # targets (+0.4% M_int_target, +4% M_core_target) before the reject sets _structure_rollback=True:
            hf_row['M_int_target'] = float(hf_row['M_int']) * 1.004
            hf_row['M_core_target'] = float(hf_row['M_core']) * 1.04
            hf_row['_structure_rollback'] = True
            interior_o.structure_rollback_active = True
            interior_o.write_structure_stale(dirs['output'])
        else:
            interior_o.zalmoxis_fail_count += 1
            if interior_o.zalmoxis_fail_count >= 8:
                raise RuntimeError('Zalmoxis failed 8 consecutive times')
            interior_o.structure_stale = True
            interior_o.structure_rollback_active = False
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    real_interior = int_wrap.run_interior

    def interior_sets_targets(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        hf_row['M_int_target'] = float(hf_row['M_int'])
        hf_row['M_core_target'] = float(hf_row['M_core'])

    monkeypatch.setattr(int_wrap, 'run_interior', interior_sets_targets)
    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', fake_forced_update)

    # Run 9 resumed steps (> 8 consecutive Zalmoxis failure budget) to prove the forced
    # attempt runs once and never repeats or aborts on subsequent static steps:
    resumed, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        f'run_newer_resume_{outcome}',
        max_iters=14,
        temperature_mode='liquidus_super',
        update_interval=0.0,
        mode='report',
    )
    resumed.config_path = cfg_path
    resumed.config.params.stop.time.maximum = 1.0e9

    row6_targets: list[tuple[float, float]] = []
    real_check_row = proteus_mod.check_row_invariants

    def spy_row6_check(hf_row, config, **kwargs):
        if len(resumed.hf_all) - 1 == 6:
            row6_targets.append(
                (
                    float(hf_row.get('M_int_target', 0.0)),
                    float(hf_row.get('M_core_target', 0.0)),
                )
            )
        return real_check_row(hf_row, config, **kwargs)

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_row6_check)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='fwl.proteus'):
        resumed.start(resume=True, offline=True)

    assert len(resumed.hf_all) - 6 >= 9
    assert forced_calls == [True]
    assert resumed._baseline_structure_done is True
    assert (
        Path(get_file_baseline_structure_done(str(out_dir))).read_text(encoding='utf-8').strip()
        == '1 6'
    )
    resume_warns = [
        r.message
        for r in caplog.records
        if r.levelname == 'WARNING'
        and 'resume' in r.message.lower()
        and 'structure' in r.message.lower()
    ]
    if outcome in ('accept', 'upstep_reject'):
        assert resumed.interior_o.structure_rollback_active is False
        assert resumed.interior_o.structure_stale is False
        assert (
            Path(get_file_structure_rollback(str(out_dir))).read_text(encoding='utf-8').strip()
            == '0 6'
        )
        assert resume_warns == []
        if outcome == 'upstep_reject':
            assert row6_targets[0][0] == pytest.approx(float(resumed.hf_all.iloc[6]['M_int']))
            assert row6_targets[0][1] == pytest.approx(float(resumed.hf_all.iloc[6]['M_core']))
    else:
        assert resumed.interior_o.structure_rollback_active is True
        assert resumed.interior_o.structure_stale is True
        assert (
            Path(get_file_structure_stale(str(out_dir))).read_text(encoding='utf-8').strip()
            == '1 6'
        )
        assert (
            Path(get_file_structure_rollback(str(out_dir))).read_text(encoding='utf-8').strip()
            == '1 6'
        )
        assert len(resume_warns) == 1


def test_proteus_start_resume_after_dropped_impact_row_runs_only_impact_structure_solve(
    tmp_path, monkeypatch
):
    """When a resume drops a snapshot row on which an impact occurred, the accepted impact solve on the first resumed iteration satisfies the forced re-solve so update_structure_from_interior is not also called."""
    from types import SimpleNamespace

    import proteus.accretion.wrapper as acc_wrap
    import proteus.interior_energetics.wrapper as int_wrap
    import proteus.utils.coupler as coupler_mod
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
    )

    runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_dropped_impact_row',
        max_iters=5,
        temperature_mode='isothermal',
        update_interval=0.0,
        mode='report',
    )
    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)

    impact_event = SimpleNamespace(
        time=50.0,
        id_target=1,
        id_impactor=2,
        M_target_before=5.97e24,
        M_impactor=1.0e22,
        mass_delta=1.0e22,
        v_impact=1.2e4,
        b_impact=0.3,
        semimajoraxis_ratio=1.0,
        semimajoraxis_after=1.0,
        eccentricity_change=0.0,
        eccentricity_after=0.01,
        dE_total=1.0e29,
        dE_core=0.0,
        f_atm_loss=None,
    )
    monkeypatch.setattr(int_wrap, 'remelt_mantle', lambda *a, **kw: None)

    solve_calls: list[str] = []
    real_dummy_solve = int_wrap.determine_interior_radius_with_dummy

    def spy_solve_structure(dirs, config, hf_all, hf_row, outdir, thermal_solve=True):
        solve_calls.append('impact' if not thermal_solve else 'init')
        real_dummy_solve(dirs, config, hf_all, hf_row, outdir, thermal_solve=thermal_solve)
        hf_row['M_int_target'] = hf_row['M_int']
        hf_row['M_core_target'] = hf_row['M_core']
        hf_row['_structure_rollback'] = False

    def spy_update_structure(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        solve_calls.append('forced_baseline' if force else 'dynamic')
        interior_o.structure_stale = False
        interior_o.structure_rollback_active = False
        hf_row['_structure_rollback'] = False
        interior_o.write_structure_stale(dirs['output'])
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    monkeypatch.setattr(int_wrap, 'solve_structure', spy_solve_structure)
    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', spy_update_structure)
    monkeypatch.setattr(acc_wrap, 'init_accretion', lambda h: [impact_event])
    monkeypatch.setattr(acc_wrap, 'restore_accretion_state', lambda h: None)

    runner.start(resume=False, offline=True)
    # Continuous run did 'init' (row 0), 'forced_baseline' (row 4), and 'impact' (row 5):
    assert solve_calls == ['init', 'forced_baseline', 'impact']

    # Now drop row 5 (the impact row) on resume and re-run step 5:
    real_select = coupler_mod.select_resumable_snapshot

    def fake_select_drops_impact_row(outdir, hf_all, **kwargs):
        trimmed, _ = real_select(outdir, hf_all, **kwargs)
        return trimmed.iloc[:-1].copy(), [float(trimmed.iloc[-1]['Time'])]

    monkeypatch.setattr(coupler_mod, 'select_resumable_snapshot', fake_select_drops_impact_row)

    solve_calls.clear()
    resumed, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_dropped_impact_row',
        max_iters=5,
        temperature_mode='isothermal',
        update_interval=0.0,
        mode='report',
    )
    resumed.config_path = cfg_path
    monkeypatch.setattr(int_wrap, 'solve_structure', spy_solve_structure)
    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', spy_update_structure)

    resumed.start(resume=True, offline=True)
    # First resumed iteration (row 5) must run ONLY the impact solve, not also forced_baseline:
    assert solve_calls == ['impact']
    assert resumed._baseline_structure_done is True
    assert resumed.interior_o.structure_rollback_active is False
    assert (
        Path(get_file_structure_rollback(str(out_dir))).read_text(encoding='utf-8').strip()
        == '0 5'
    )
    assert (
        Path(get_file_baseline_structure_done(str(out_dir))).read_text(encoding='utf-8').strip()
        == '1 5'
    )


def test_proteus_start_syncs_rollback_flag_across_init_equilibrate_and_impact(
    tmp_path, monkeypatch
):
    """Proteus.start syncs interior_o.structure_rollback_active after solve_structure, equilibrate_initial_state, and apply_impact, leaving it False for liquidus_super + update_interval=0 so a post-init target error stops."""
    from types import SimpleNamespace

    import proteus.accretion.wrapper as acc_wrap
    import proteus.interior_energetics.wrapper as int_wrap
    import proteus.interior_struct.zalmoxis as zalmoxis_mod
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    monkeypatch.setattr(Proteus, '_require_paleos_tables', lambda self: None)

    # Part A: solve_structure sets _structure_rollback=True with equilibrate_init=False -> copied to runner.interior_o,
    # and the super-liquidus baseline path writes 1 to baseline_structure_done.dat:
    runner_a, _, out_dir_a = _build_dummy_runner(tmp_path, 'run_sync_init_only', max_iters=5)
    runner_a.config.invariants.mode = 'report'
    runner_a.config.interior_struct.module = 'zalmoxis'
    runner_a.config.interior_struct.zalmoxis.equilibrate_init = False
    runner_a.config.interior_struct.zalmoxis.update_interval = 0.0
    runner_a.config.planet.temperature_mode = 'liquidus_super'

    def fake_solve_sets_rollback(dirs, config, hf_all, hf_row, outdir, thermal_solve=True):
        int_wrap.determine_interior_radius_with_dummy(
            dirs, config, hf_all, hf_row, outdir, thermal_solve=thermal_solve
        )
        hf_row['_structure_rollback'] = True

    monkeypatch.setattr(int_wrap, 'solve_structure', fake_solve_sets_rollback)
    runner_a.start(resume=False, offline=True)
    assert runner_a.interior_o.structure_rollback_active is True
    assert (
        Path(get_file_structure_rollback(str(out_dir_a))).read_text(encoding='utf-8').strip()
        == '1 0'
    )
    assert (
        Path(get_file_baseline_structure_done(str(out_dir_a)))
        .read_text(encoding='utf-8')
        .strip()
        == '1 4'
    )

    # Part B: solve_structure sets _structure_rollback=True (and writes '1 0' to disk), then equilibrate_initial_state
    # clears it to False and overwrites structure_rollback.dat with '0 0':
    runner_b, _, out_dir_b = _build_dummy_runner(tmp_path, 'run_sync_equil_clears', max_iters=5)
    runner_b.config.invariants.mode = 'stop'
    runner_b.config.interior_struct.module = 'zalmoxis'
    runner_b.config.interior_struct.zalmoxis.equilibrate_init = True
    runner_b.config.interior_struct.zalmoxis.update_interval = 0.0
    runner_b.config.planet.temperature_mode = 'liquidus_super'

    def fake_equilibrate_clears(dirs, config, hf_row, outdir):
        assert (
            Path(get_file_structure_rollback(outdir)).read_text(encoding='utf-8').strip()
            == '1 0'
        )
        hf_row['_structure_rollback'] = False

    monkeypatch.setattr(int_wrap, 'equilibrate_initial_state', fake_equilibrate_clears)

    real_interior = int_wrap.run_interior

    def interior_injects_post_init_target_err(
        dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs
    ):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        hf_row['M_core'] = 1.94e24
        if runner_b.loops['total'] == 4:
            # First post-init row (row 4): inject a 5% M_core_target mismatch (> 2.5e-2 tol)
            hf_row['M_core_target'] = 1.94e24 * 1.05
        else:
            hf_row['M_core_target'] = 1.94e24

    monkeypatch.setattr(int_wrap, 'run_interior', interior_injects_post_init_target_err)
    with pytest.raises(InvariantViolationError) as exc_b:
        runner_b.start(resume=False, offline=True)
    assert runner_b.interior_o.structure_rollback_active is False
    assert (
        Path(get_file_structure_rollback(str(out_dir_b))).read_text(encoding='utf-8').strip()
        == '0 0'
    )
    assert exc_b.value.invariant_name == 'M_core_target_rel_err'
    assert exc_b.value.status_code == 31

    # Part C: apply_impact on a stale structure clears structure_stale, structure_rollback_active,
    # and zalmoxis_fail_count when the impact structure solve is accepted (_structure_rollback=False),
    # and preserves structure_stale and zalmoxis_fail_count when _structure_rollback=True:
    from proteus.utils.constants import element_list

    def fake_solve_clears_rollback(dirs, config, hf_all, hf_row, outdir, thermal_solve=True):
        hf_row['R_int'] = 6.37e6
        hf_row['R_core'] = 3.48e6
        hf_row['M_int'] = 5.97e24
        hf_row['M_int_target'] = 5.97e24 * 1.01
        hf_row['M_core'] = 1.94e24
        hf_row['M_mantle'] = 4.03e24
        hf_row['_structure_rollback'] = False

    monkeypatch.setattr(int_wrap, 'solve_structure', fake_solve_clears_rollback)
    monkeypatch.setattr(int_wrap, 'remelt_mantle', lambda *a, **kw: None)

    impact_out = tmp_path / 'impact_out'
    (impact_out / 'data').mkdir(parents=True, exist_ok=True)
    (impact_out / 'data' / 'zalmoxis_output.dat').write_text('0.0 0.0\n', encoding='utf-8')
    Path(get_file_structure_stale(str(impact_out))).write_text('1\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(impact_out))).write_text('1\n', encoding='utf-8')

    impact_cfg = runner_b.config
    impact_int_o = Interior_t(5)
    impact_int_o.ic = 2
    impact_int_o.radius = pd.Series([3.48e6, 4.0e6, 5.0e6, 6.0e6, 6.37e6]).to_numpy()
    impact_int_o.temp = pd.Series([4000.0, 3500.0, 3000.0, 2500.0]).to_numpy()
    impact_int_o.structure_stale = True
    impact_int_o.structure_rollback_active = True
    impact_int_o.zalmoxis_fail_count = int_wrap._ZALMOXIS_MAX_CONSECUTIVE_FAILS - 1
    impact_hf_row = {f'{e}_kg_atm': 0.0 for e in element_list}
    impact_hf_row.update({f'{e}_kg_total': 0.0 for e in element_list})
    impact_hf_row.update(
        {
            'Time': 500.0,
            'T_surf': 2000.0,
            'T_magma': 2500.0,
            'T_cmb': 4000.0,
            'P_surf': 100.0,
            'P_cmb': 1.35e11,
            'P_center': 3.6e11,
            'gravity': 9.81,
            'rho_avg': 5515.0,
            'Phi_global': 0.8,
            'M_planet': 5.972e24,
            'M_int': 5.97e24,
            'M_core': 1.94e24,
            'M_core_target': 1.94e24,
            'M_mantle': 4.03e24,
            'R_int': 6.37e6,
            'R_core': 3.48e6,
            'M_atm': 1.0e20,
            'M_vol_atm': 1.0e20,
            'M_vaps': 0.0,
            'M_accreted_rock': 0.0,
            'n_impacts_applied': 0,
            'H2O_kg_atm': 1.0e20,
            'H_kg_total': 1.1e19,
            'H_kg_atm': 1.1e19,
            'O_kg_total': 8.9e19,
            'O_kg_atm': 8.9e19,
            'semimajorax': 1.496e11,
            'eccentricity': 0.01,
            '_structure_rollback': True,
        }
    )
    handler = SimpleNamespace(
        config=impact_cfg,
        directories={'output': str(impact_out), 'output/data': str(impact_out / 'data')},
        hf_all=pd.DataFrame([impact_hf_row]),
        hf_row=impact_hf_row,
        interior_o=impact_int_o,
        init_stage=False,
        crystallized=False,
        desiccated=False,
    )
    event = SimpleNamespace(
        time=500.0,
        id_target=1,
        id_impactor=2,
        M_target_before=5.97e24,
        M_impactor=1.0e22,
        mass_delta=1.0e22,
        v_impact=1.2e4,
        b_impact=0.3,
        semimajoraxis_ratio=1.0,
        semimajoraxis_after=1.0,
        eccentricity_change=0.0,
        eccentricity_after=0.01,
        dE_total=1.0e29,
        dE_core=0.0,
        f_atm_loss=None,
    )
    acc_wrap.apply_impact(handler, event)
    assert impact_hf_row.get('_structure_rollback') is False
    assert handler.interior_o.structure_stale is False
    assert handler.interior_o.structure_rollback_active is False
    assert handler.interior_o.zalmoxis_fail_count == 0
    assert (
        Path(get_file_structure_stale(str(impact_out))).read_text(encoding='utf-8').strip()
        == '0 1'
    )
    assert (
        Path(get_file_structure_rollback(str(impact_out))).read_text(encoding='utf-8').strip()
        == '0 1'
    )

    resumed_after_impact = Interior_t(5)
    resumed_after_impact.resume_structure_stale(str(impact_out), resume_row=1)
    assert resumed_after_impact.structure_stale is False
    assert resumed_after_impact.structure_rollback_active is False

    resumed_impact_row = {k: v for k, v in impact_hf_row.items() if k != '_structure_rollback'}
    with pytest.raises(InvariantViolationError) as exc_impact_cont:
        proteus_mod.check_row_invariants(
            impact_hf_row,
            impact_cfg,
            init_stage=False,
            interior_o=handler.interior_o,
        )
    with pytest.raises(InvariantViolationError) as exc_impact_resumed:
        proteus_mod.check_row_invariants(
            resumed_impact_row,
            impact_cfg,
            init_stage=False,
            interior_o=resumed_after_impact,
        )
    assert exc_impact_cont.value.invariant_name == 'M_int_target_rel_err'
    assert exc_impact_resumed.value.invariant_name == 'M_int_target_rel_err'
    assert (
        [v.report_only for v in exc_impact_cont.value.violations]
        == [v.report_only for v in exc_impact_resumed.value.violations]
        == [False]
    )

    # A single failed update_structure_from_interior solve after the accepted impact
    # increments zalmoxis_fail_count from 0 to 1 rather than aborting at 8:
    def failing_zalmoxis_after_impact(*a, **kw):
        raise RuntimeError('post-impact zalmoxis non-convergence')

    monkeypatch.setattr(zalmoxis_mod, 'zalmoxis_solver', failing_zalmoxis_after_impact)
    int_wrap.update_structure_from_interior(
        handler.directories,
        impact_cfg,
        impact_hf_row,
        handler.interior_o,
        last_struct_time=0.0,
        last_Tmagma=2600.0,
        last_Phi=0.85,
        force=True,
    )
    assert handler.interior_o.zalmoxis_fail_count == 1


def test_proteus_start_forced_upstep_reject_restores_m_int_and_m_core_targets(
    tmp_path, monkeypatch
):
    """When a forced resume re-solve takes the up-step reject path, M_int_target and M_core_target are restored to their pre-solve values so a 0.4% rejected target change does not reach hf_row or stop the run."""
    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_forced_reject_restores_targets',
        max_iters=5,
        temperature_mode='liquidus_super',
        update_interval=0.0,
        mode='report',
    )
    runner.start(resume=False, offline=True)

    # Seed a newer flag event at row 6 > resume_row 5:
    Path(get_file_structure_stale(str(out_dir))).write_text('1 6\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(out_dir))).write_text('1 6\n', encoding='utf-8')
    Path(get_file_baseline_structure_done(str(out_dir))).write_text('1 4\n', encoding='utf-8')

    real_interior = int_wrap.run_interior

    def interior_sets_targets(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        hf_row['M_int_target'] = float(hf_row['M_int'])
        hf_row['M_core_target'] = float(hf_row['M_core'])

    def fake_reject_changes_targets(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        hf_row['M_int_target'] = float(hf_row['M_int']) * 1.004
        hf_row['M_core_target'] = float(hf_row['M_core']) * 1.04
        hf_row['_structure_rollback'] = True
        interior_o.structure_rollback_active = True
        interior_o.write_structure_stale(dirs['output'])
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    monkeypatch.setattr(int_wrap, 'run_interior', interior_sets_targets)
    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', fake_reject_changes_targets)

    resumed, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_forced_reject_restores_targets',
        max_iters=6,
        temperature_mode='liquidus_super',
        update_interval=0.0,
        mode='stop',
    )
    resumed.config_path = cfg_path

    stop_exc: InvariantViolationError | None = None
    try:
        resumed.start(resume=True, offline=True)
    except InvariantViolationError as exc:
        stop_exc = exc

    assert stop_exc is None
    assert float(resumed.hf_row['M_int_target']) == pytest.approx(
        float(resumed.hf_row['M_int'])
    )
    assert float(resumed.hf_row['M_core_target']) == pytest.approx(
        float(resumed.hf_row['M_core'])
    )
    assert resumed.interior_o.structure_rollback_active is False


@pytest.mark.parametrize('outcome', ['accept', 'upstep_reject', 'solver_failure'])
def test_proteus_start_superliquidus_resume_with_no_newer_event_three_outcomes(
    tmp_path, monkeypatch, caplog, outcome
):
    """For liquidus_super on resume with baseline '0 R', rollback '1 R', and no newer event (R == resume_row), the superliquidus clause in is_resume_forced runs 1 forced attempt across >= 9 resumed rows for all 3 outcomes."""
    import logging

    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        f'run_superliq_no_newer_{outcome}',
        max_iters=5,
        temperature_mode='liquidus_super',
        update_interval=0.0,
        mode='report',
    )
    runner.start(resume=False, offline=True)

    # Seed baseline '0 5' and rollback '1 5' at R == resume_row == 5 (no newer event, no pending file):
    Path(get_file_structure_stale(str(out_dir))).write_text('1 5\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(out_dir))).write_text('1 5\n', encoding='utf-8')
    Path(get_file_baseline_structure_done(str(out_dir))).write_text('0 5\n', encoding='utf-8')
    pending_file = out_dir / 'data' / 'structure_forced_pending.dat'
    pending_file.unlink(missing_ok=True)

    real_interior = int_wrap.run_interior

    def interior_sets_targets(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        hf_row['M_int_target'] = float(hf_row['M_int'])
        hf_row['M_core_target'] = float(hf_row['M_core'])

    forced_calls: list[bool] = []

    def fake_forced_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        forced_calls.append(bool(force))
        if outcome == 'accept':
            interior_o.structure_stale = False
            interior_o.structure_rollback_active = False
            hf_row['_structure_rollback'] = False
            interior_o.write_structure_stale(dirs['output'])
        elif outcome == 'upstep_reject':
            hf_row['M_int_target'] = float(hf_row['M_int']) * 1.004
            hf_row['M_core_target'] = float(hf_row['M_core']) * 1.04
            hf_row['_structure_rollback'] = True
            interior_o.structure_rollback_active = True
            interior_o.write_structure_stale(dirs['output'])
        else:
            interior_o.zalmoxis_fail_count += 1
            if interior_o.zalmoxis_fail_count >= 8:
                raise RuntimeError('Zalmoxis failed 8 consecutive times')
            interior_o.structure_stale = True
            interior_o.structure_rollback_active = False
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    monkeypatch.setattr(int_wrap, 'run_interior', interior_sets_targets)
    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', fake_forced_update)

    resumed, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        f'run_superliq_no_newer_{outcome}',
        max_iters=14,
        temperature_mode='liquidus_super',
        update_interval=0.0,
        mode='report',
    )
    resumed.config_path = cfg_path
    resumed.config.params.stop.time.maximum = 1.0e9

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='fwl.proteus'):
        resumed.start(resume=True, offline=True)

    assert len(resumed.hf_all) - 6 >= 9
    assert forced_calls == [True]
    assert resumed._baseline_structure_done is True
    assert (
        Path(get_file_baseline_structure_done(str(out_dir))).read_text(encoding='utf-8').strip()
        == '1 6'
    )
    assert pending_file.is_file()
    assert pending_file.read_text(encoding='utf-8').strip() == '0 6'
    if outcome in ('accept', 'upstep_reject'):
        assert resumed.interior_o.structure_rollback_active is False
        assert resumed.interior_o.structure_stale is False
        assert (
            Path(get_file_structure_rollback(str(out_dir))).read_text(encoding='utf-8').strip()
            == '0 6'
        )
        if outcome == 'upstep_reject':
            assert float(resumed.hf_row['M_int_target']) == pytest.approx(
                float(resumed.hf_row['M_int'])
            )
    else:
        assert resumed.interior_o.structure_rollback_active is True
        assert resumed.interior_o.structure_stale is True
        assert (
            Path(get_file_structure_stale(str(out_dir))).read_text(encoding='utf-8').strip()
            == '1 6'
        )
        assert (
            Path(get_file_structure_rollback(str(out_dir))).read_text(encoding='utf-8').strip()
            == '1 6'
        )


def test_proteus_start_dynamic_interval_skips_second_call_in_forced_attempt_iteration(
    tmp_path, monkeypatch
):
    """When update_interval > 0 and _solve_structure_baseline_if_needed runs a failing forced solve, the dynamic call in the same iteration is skipped so Zalmoxis is not called twice in 1 iteration."""
    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_dyn_skip_forced_iter',
        max_iters=5,
        temperature_mode='liquidus_super',
        update_interval=0.0,
        mode='report',
    )
    runner.start(resume=False, offline=True)

    # Seed a newer flag event at row 6 > resume_row 5:
    Path(get_file_structure_stale(str(out_dir))).write_text('1 6\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(out_dir))).write_text('1 6\n', encoding='utf-8')
    Path(get_file_baseline_structure_done(str(out_dir))).write_text('1 4\n', encoding='utf-8')

    calls_on_row_6: list[bool] = []

    def fake_failing_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        calls_on_row_6.append(bool(force))
        interior_o.zalmoxis_fail_count += 1
        interior_o.structure_stale = True
        return lt, lT, lP

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', fake_failing_update)

    resumed, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_dyn_skip_forced_iter',
        max_iters=6,
        temperature_mode='liquidus_super',
        update_interval=1.0,
        mode='report',
    )
    resumed.config_path = cfg_path
    resumed.config.interior_struct.zalmoxis.update_interval = 1.0
    resumed.start(resume=True, offline=True)

    assert calls_on_row_6 == [True]
    assert resumed.interior_o.zalmoxis_fail_count == 1


def test_proteus_start_newer_event_resume_with_uncommitted_baseline_keeps_retry_rule_and_matches_continuous(
    tmp_path, monkeypatch
):
    """When baseline_structure_done.dat holds '0 R' on disk at a newer-event resume, Proteus keeps the initial-baseline retry rule (and impact shortcut rule) rather than marking the baseline done on failure, matching the continuous run."""
    from types import SimpleNamespace

    import proteus.accretion.wrapper as acc_wrap
    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)
    real_interior = int_wrap.run_interior
    real_check_row = proteus_mod.check_row_invariants

    def inject_desync_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        hf_row['M_int'] = float(hf_row.get('M_int_target') or hf_row.get('M_int', 5.97e24))
        hf_row['struct_mass_desync_frac'] = 2.0e-2

    monkeypatch.setattr(int_wrap, 'run_interior', inject_desync_interior)

    # 1. Continuous run: baseline fails on rows 4 and 5 (init_stage=True, report_only=True) and succeeds on row 6 (init_stage=False, stops on row 6):
    cont_runner, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_r109a_cont',
        max_iters=6,
        temperature_mode='isothermal',
        update_interval=0.0,
        mode='stop',
    )

    def cont_baseline_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        if len(cont_runner.hf_all) in (4, 5):
            interior_o.structure_stale = True
            interior_o.structure_rollback_active = True
            hf_row['_structure_rollback'] = True
            interior_o.write_structure_stale(dirs['output'])
            return lt, lT, lP
        interior_o.structure_stale = False
        interior_o.structure_rollback_active = False
        hf_row['_structure_rollback'] = False
        interior_o.write_structure_stale(dirs['output'])
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    cont_seq: list[tuple[int, bool, bool]] = []

    def spy_cont_check(hf_row, config, **kwargs):
        row_i = len(cont_runner.hf_all) - 1
        try:
            vs = real_check_row(hf_row, config, **kwargs)
        except InvariantViolationError as exc:
            if row_i >= 4:
                cont_seq.append(
                    (
                        row_i,
                        bool(kwargs.get('init_stage')),
                        all(v.report_only for v in exc.violations),
                    )
                )
            raise
        if row_i >= 4:
            cont_seq.append(
                (row_i, bool(kwargs.get('init_stage')), all(v.report_only for v in vs))
            )
        return vs

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', cont_baseline_update)
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_cont_check)

    cont_stop_exc: InvariantViolationError | None = None
    try:
        cont_runner.start(resume=False, offline=True)
    except InvariantViolationError as exc:
        cont_stop_exc = exc
    assert cont_stop_exc is not None
    assert cont_stop_exc.invariant_name == 'struct_mass_desync_frac'
    assert cont_seq == [(4, True, True), (5, True, True), (6, False, False)]

    # 2. Resumed run: run rows 0..4 (baseline fails on row 4), seed newer flag event at dropped row 5 with baseline '0 5', and resume for rows 5..6:
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', real_check_row)
    seed_runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_r109a_resumed',
        max_iters=4,
        temperature_mode='isothermal',
        update_interval=0.0,
        mode='stop',
    )

    def seed_failing_baseline(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        interior_o.structure_stale = True
        interior_o.structure_rollback_active = True
        hf_row['_structure_rollback'] = True
        interior_o.write_structure_stale(dirs['output'])
        return lt, lT, lP

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', seed_failing_baseline)
    seed_runner.start(resume=False, offline=True)

    # Seed newer flag event at row 5 > resume_row 4 while baseline_structure_done.dat is still '0 5' (uncommitted):
    Path(get_file_structure_stale(str(out_dir))).write_text('1 5\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(out_dir))).write_text('1 5\n', encoding='utf-8')
    Path(get_file_baseline_structure_done(str(out_dir))).write_text('0 5\n', encoding='utf-8')

    resumed_runner, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_r109a_resumed',
        max_iters=6,
        temperature_mode='isothermal',
        update_interval=0.0,
        mode='stop',
    )
    resumed_runner.config_path = cfg_path

    def resumed_baseline_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        if len(resumed_runner.hf_all) == 5:
            interior_o.structure_stale = True
            interior_o.structure_rollback_active = True
            hf_row['_structure_rollback'] = True
            interior_o.write_structure_stale(dirs['output'])
            return lt, lT, lP
        interior_o.structure_stale = False
        interior_o.structure_rollback_active = False
        hf_row['_structure_rollback'] = False
        interior_o.write_structure_stale(dirs['output'])
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    resumed_seq: list[tuple[int, bool, bool]] = [(4, True, True)]

    def spy_resumed_check(hf_row, config, **kwargs):
        row_i = len(resumed_runner.hf_all) - 1
        try:
            vs = real_check_row(hf_row, config, **kwargs)
        except InvariantViolationError as exc:
            resumed_seq.append(
                (
                    row_i,
                    bool(kwargs.get('init_stage')),
                    all(v.report_only for v in exc.violations),
                )
            )
            raise
        resumed_seq.append(
            (row_i, bool(kwargs.get('init_stage')), all(v.report_only for v in vs))
        )
        return vs

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', resumed_baseline_update)
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_resumed_check)

    resumed_stop_exc: InvariantViolationError | None = None
    try:
        resumed_runner.start(resume=True, offline=True)
    except InvariantViolationError as exc:
        resumed_stop_exc = exc

    assert resumed_seq == cont_seq
    assert resumed_stop_exc is not None
    assert (
        resumed_stop_exc.invariant_name
        == cont_stop_exc.invariant_name
        == 'struct_mass_desync_frac'
    )

    # 3. Impact shortcut rule when baseline_structure_done.dat is '0 R' (uncommitted):
    # An impact on the first resumed step does NOT skip the uncommitted initial baseline solve:
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', real_check_row)
    monkeypatch.setattr(int_wrap, 'run_interior', real_interior)
    imp_seed, imp_cfg_path, imp_out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_r109a_impact_uncommitted',
        max_iters=4,
        temperature_mode='isothermal',
        update_interval=0.0,
        mode='report',
    )
    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', seed_failing_baseline)
    imp_seed.start(resume=False, offline=True)

    Path(get_file_structure_stale(str(imp_out_dir))).write_text('0 5\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(imp_out_dir))).write_text('0 5\n', encoding='utf-8')
    Path(get_file_baseline_structure_done(str(imp_out_dir))).write_text(
        '0 5\n', encoding='utf-8'
    )

    impact_event = SimpleNamespace(
        time=10.0,
        id_target=1,
        id_impactor=2,
        M_target_before=5.97e24,
        M_impactor=1.0e22,
        mass_delta=1.0e22,
        v_impact=1.2e4,
        b_impact=0.3,
        semimajoraxis_ratio=1.0,
        semimajoraxis_after=1.0,
        eccentricity_change=0.0,
        eccentricity_after=0.01,
        dE_total=1.0e29,
        dE_core=0.0,
        f_atm_loss=None,
    )
    monkeypatch.setattr(int_wrap, 'remelt_mantle', lambda *a, **kw: None)
    monkeypatch.setattr(acc_wrap, 'init_accretion', lambda h: [impact_event])
    monkeypatch.setattr(acc_wrap, 'restore_accretion_state', lambda h: None)
    real_dummy_solve = int_wrap.determine_interior_radius_with_dummy

    imp_calls: list[str] = []

    def spy_imp_solve(dirs, config, hf_all, hf_row, outdir, thermal_solve=True):
        imp_calls.append('impact' if not thermal_solve else 'init')
        real_dummy_solve(dirs, config, hf_all, hf_row, outdir, thermal_solve=thermal_solve)
        hf_row['_structure_rollback'] = False

    def spy_imp_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        imp_calls.append('baseline')
        interior_o.structure_stale = False
        interior_o.structure_rollback_active = False
        hf_row['_structure_rollback'] = False
        interior_o.write_structure_stale(dirs['output'])
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    imp_resumed, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_r109a_impact_uncommitted',
        max_iters=5,
        temperature_mode='isothermal',
        update_interval=0.0,
        mode='report',
    )
    imp_resumed.config_path = imp_cfg_path
    monkeypatch.setattr(int_wrap, 'solve_structure', spy_imp_solve)
    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', spy_imp_update)
    imp_resumed.start(resume=True, offline=True)
    assert imp_calls == ['impact', 'baseline']


def test_proteus_start_kill_after_first_resume_preserves_pending_forced_resolve_and_matches_continuous(
    tmp_path, monkeypatch
):
    """When a process is killed after the first resume setup before the forced re-solve completes, structure_forced_pending.dat preserves the pending state so the second resume makes 1 attempt (not up to 8) and matches the continuous run."""
    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        get_file_structure_stale,
    )

    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)
    real_interior = int_wrap.run_interior
    real_check_row = proteus_mod.check_row_invariants

    # 1. Continuous run (rows 0..14, update_interval=0.0, mode='stop'):
    # Row 4 baseline succeeds; row 6 has a failed structure event leaving rollback=True and +2% M_int_target mismatch:
    cont_runner, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_r109b_cont',
        max_iters=14,
        temperature_mode='isothermal',
        update_interval=0.0,
        mode='stop',
    )
    cont_runner.config.params.stop.time.maximum = 1.0e9

    def cont_baseline_succeeds(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        interior_o.structure_stale = False
        interior_o.structure_rollback_active = False
        hf_row['_structure_rollback'] = False
        interior_o.write_structure_stale(dirs['output'])
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    def cont_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        hf_row['M_int'] = 5.97e24
        if hf_all is not None and len(hf_all) >= 6:
            interior_o.structure_stale = True
            interior_o.structure_rollback_active = True
            hf_row['M_int_target'] = 5.97e24 * 1.02
        else:
            hf_row['M_int_target'] = 5.97e24

    cont_seq: list[tuple[int, bool, bool]] = []

    def spy_cont_check(hf_row, config, **kwargs):
        row_i = len(cont_runner.hf_all) - 1
        vs = real_check_row(hf_row, config, **kwargs)
        if row_i >= 4:
            cont_seq.append(
                (
                    row_i,
                    bool(kwargs.get('init_stage')),
                    bool(cont_runner.interior_o.structure_rollback_active),
                )
            )
        return vs

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', cont_baseline_succeeds)
    monkeypatch.setattr(int_wrap, 'run_interior', cont_interior)
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_cont_check)
    cont_runner.start(resume=False, offline=True)
    assert len(cont_runner.hf_all) - 6 >= 9

    # 2. Seed run (rows 0..5, baseline committed '1 4' at row 4), then newer flag event at dropped row 6:
    monkeypatch.setattr(int_wrap, 'run_interior', real_interior)
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', real_check_row)
    seed_runner, cfg_path, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_r109b_resumed',
        max_iters=5,
        temperature_mode='isothermal',
        update_interval=0.0,
        mode='report',
    )
    seed_runner.config.params.stop.time.maximum = 1.0e9
    seed_runner.start(resume=False, offline=True)

    Path(get_file_structure_stale(str(out_dir))).write_text('1 6\n', encoding='utf-8')
    Path(get_file_structure_rollback(str(out_dir))).write_text('1 6\n', encoding='utf-8')
    Path(get_file_baseline_structure_done(str(out_dir))).write_text('1 4\n', encoding='utf-8')

    # First resume is killed inside run_interior on the first resumed step (after resume setup, before _solve_structure_baseline_if_needed):
    class _SimulatedKillAfterResumeSetup(BaseException):
        pass

    def kill_on_first_resume_interior(*args, **kwargs):
        raise _SimulatedKillAfterResumeSetup('killed after first resume setup')

    monkeypatch.setattr(int_wrap, 'run_interior', kill_on_first_resume_interior)
    first_resume, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_r109b_resumed',
        max_iters=14,
        temperature_mode='isothermal',
        update_interval=0.0,
        mode='stop',
    )
    first_resume.config_path = cfg_path
    first_resume.config.params.stop.time.maximum = 1.0e9
    with pytest.raises(_SimulatedKillAfterResumeSetup):
        first_resume.start(resume=True, offline=True)

    # Launch the second resume with a failing forced structure solve:
    forced_calls: list[bool] = []

    def failing_forced_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        forced_calls.append(bool(force))
        interior_o.zalmoxis_fail_count += 1
        if interior_o.zalmoxis_fail_count >= 8:
            raise RuntimeError('Zalmoxis failed 8 consecutive times')
        interior_o.structure_stale = True
        interior_o.structure_rollback_active = False
        return lt, lT, lP

    def resumed_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        hf_row['M_core'] = float(hf_row.get('M_core_target') or hf_row.get('M_core', 0.0))
        hf_row['M_int'] = 5.97e24
        hf_row['M_int_target'] = 5.97e24 * 1.02

    resumed_seq: list[tuple[int, bool, bool]] = [(4, False, False), (5, False, False)]
    second_resume, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_r109b_resumed',
        max_iters=14,
        temperature_mode='isothermal',
        update_interval=0.0,
        mode='stop',
    )
    second_resume.config_path = cfg_path
    second_resume.config.params.stop.time.maximum = 1.0e9
    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', failing_forced_update)
    monkeypatch.setattr(int_wrap, 'run_interior', resumed_interior)

    def spy_resumed_check(hf_row, config, **kwargs):
        row_i = len(second_resume.hf_all) - 1
        vs = real_check_row(hf_row, config, **kwargs)
        resumed_seq.append(
            (
                row_i,
                bool(kwargs.get('init_stage')),
                bool(second_resume.interior_o.structure_rollback_active),
            )
        )
        return vs

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_resumed_check)

    runtime_err: RuntimeError | None = None
    try:
        second_resume.start(resume=True, offline=True)
    except RuntimeError as exc:
        runtime_err = exc

    assert forced_calls == [True]
    assert runtime_err is None
    assert len(second_resume.hf_all) - 6 >= 9
    assert resumed_seq == cont_seq
    pending_file = out_dir / 'data' / 'structure_forced_pending.dat'
    assert pending_file.is_file()
    assert pending_file.read_text(encoding='utf-8').strip() == '0 6'
