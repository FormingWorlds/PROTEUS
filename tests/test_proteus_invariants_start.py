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


def test_proteus_start_continuous_run_failing_baseline_with_dynamic_interval_runs_both_calls(
    tmp_path, monkeypatch
):
    """On a continuous run with update_interval > 0 and a failing initial baseline, _solve_structure_baseline_if_needed returns False so both the baseline retry and the dynamic call run on each iteration and abort at row 7 after 8 consecutive failures."""
    import proteus.interior_energetics.wrapper as int_wrap

    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)

    runner, _, _ = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_cont_failing_baseline_dyn',
        max_iters=10,
        temperature_mode='isothermal',
        update_interval=1.0,
        mode='report',
    )
    runner.config.params.stop.time.maximum = 1.0e9

    call_sequence: list[tuple[int, bool]] = []

    def fake_failing_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        call_sequence.append((len(runner.hf_all), bool(force)))
        interior_o.zalmoxis_fail_count += 1
        if interior_o.zalmoxis_fail_count >= 8:
            raise RuntimeError('Zalmoxis failed 8 consecutive times')
        interior_o.structure_stale = True
        interior_o.structure_rollback_active = True
        hf_row['_structure_rollback'] = True
        interior_o.write_structure_stale(dirs['output'])
        return lt, lT, lP

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', fake_failing_update)

    runtime_err: RuntimeError | None = None
    try:
        runner.start(resume=False, offline=True)
    except RuntimeError as exc:
        runtime_err = exc

    assert call_sequence == [
        (4, True),
        (4, False),
        (5, True),
        (5, False),
        (6, True),
        (6, False),
        (7, True),
        (7, False),
    ]
    assert len(runner.hf_all) == 7
    assert runtime_err is not None
    expected_rows = [
        (4, 2.0, 3999.2966793951055, 6284776.941068927, 5.972e24),
        (5, 102.0, 3929.0141255951376, 6284776.941068927, 5.972e24),
        (6, 202.0, 3863.5485955804766, 6284776.941068927, 5.972e24),
    ]
    for row_idx, exp_time, exp_tmagma, exp_rint, exp_mint in expected_rows:
        row = runner.hf_all.iloc[row_idx]
        assert float(row['Time']) == pytest.approx(exp_time, rel=1e-12)
        assert float(row['T_magma']) == pytest.approx(exp_tmagma, rel=1e-12)
        assert float(row['R_int']) == pytest.approx(exp_rint, rel=1e-12)
        assert float(row['M_int']) == pytest.approx(exp_mint, rel=1e-12)


def test_proteus_start_continuous_run_succeeding_baseline_with_dynamic_interval_runs_dynamic_call(
    tmp_path, monkeypatch
):
    """On a continuous run with update_interval > 0 and a succeeding initial baseline on row 4, _solve_structure_baseline_if_needed returns False so the dynamic structure call also runs on row 4 and on subsequent rows."""
    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        write_baseline_structure_done,
    )

    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)

    runner, _, out_dir = _setup_zalmoxis_baseline_runner(
        tmp_path,
        monkeypatch,
        'run_cont_succeeding_baseline_dyn',
        max_iters=7,
        temperature_mode='isothermal',
        update_interval=1.0,
        mode='report',
    )
    runner.config.params.stop.time.maximum = 1.0e9

    call_sequence: list[tuple[int, bool]] = []

    def fake_succeeding_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        call_sequence.append((len(runner.hf_all), bool(force)))
        interior_o.structure_stale = False
        interior_o.structure_rollback_active = False
        hf_row['_structure_rollback'] = False
        interior_o.write_structure_stale(dirs['output'])
        return float(hf_row.get('Time', 0.0)), float(hf_row.get('T_magma', 2000.0)), 0.5

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', fake_succeeding_update)
    runner.start(resume=False, offline=True)

    assert call_sequence == [
        (4, True),
        (4, False),
        (5, False),
        (6, False),
        (7, False),
    ]
    assert len(runner.hf_all) == 8
    assert runner._baseline_structure_done is True
    write_baseline_structure_done(str(out_dir), True, row_index=7)
    assert (
        Path(get_file_baseline_structure_done(str(out_dir))).read_text(encoding='utf-8').strip()
        == '1 4'
    )
