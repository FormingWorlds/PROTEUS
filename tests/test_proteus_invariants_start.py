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


def test_proteus_start_zalmoxis_failed_baseline_keeps_row_4_init_stage_true_and_persists_on_resume(
    tmp_path, monkeypatch
):
    """Proteus.start with zalmoxis keeps init_stage=True on row 4 when the forced baseline fails, restores _baseline_structure_done=False only when baseline_structure_done.dat is 0, and treats absent file as done."""
    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
        get_file_baseline_structure_done,
        get_file_structure_rollback,
        resume_baseline_structure_done,
        write_baseline_structure_done,
    )

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    runner, cfg_path, out_dir = _build_dummy_runner(
        tmp_path, 'run_zalmoxis_failed_baseline', max_iters=6
    )
    runner.config.invariants.mode = 'stop'
    runner.config.interior_struct.module = 'zalmoxis'
    runner.config.interior_struct.zalmoxis.equilibrate_init = False
    runner.config.interior_struct.zalmoxis.update_interval = 0.0
    runner.config.planet.temperature_mode = 'isothermal'
    monkeypatch.setattr(Proteus, '_require_paleos_tables', lambda self: None)

    def fake_solve_structure(dirs, config, hf_all, hf_row, outdir, thermal_solve=True):
        int_wrap.determine_interior_radius_with_dummy(
            dirs, config, hf_all, hf_row, outdir, thermal_solve=thermal_solve
        )
        hf_row['M_int_target'] = hf_row['M_int']
        hf_row['M_core_target'] = hf_row['M_core']
        hf_row['_structure_rollback'] = False
        # Pre-seed 1 so the test verifies that start() overwrites both flag files with 0:
        Path(get_file_structure_rollback(outdir)).write_text('1\n', encoding='utf-8')
        Path(get_file_baseline_structure_done(outdir)).write_text('1\n', encoding='utf-8')

    monkeypatch.setattr(int_wrap, 'solve_structure', fake_solve_structure)
    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)

    baseline_attempts = 0

    def fake_update_structure(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        nonlocal baseline_attempts
        baseline_attempts += 1
        if baseline_attempts == 1:
            # Remove baseline_structure_done.dat before returning so the test checks
            # that _solve_structure_baseline_if_needed writes 0 on a failed baseline:
            Path(get_file_baseline_structure_done(dirs['output'])).unlink(missing_ok=True)
            interior_o.structure_stale = True
            interior_o.structure_rollback_active = True
            hf_row['_structure_rollback'] = True
            interior_o.write_structure_stale(dirs['output'])
            return lt, lT, lP
        # Second attempt (row 5, loop 6) succeeds
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
            # Init rows 0-3 and failed-baseline row 4 carry pre-baseline desync > 1e-2 stop tol
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
                == '0'
            )
            assert (
                Path(get_file_baseline_structure_done(str(out_dir)))
                .read_text(encoding='utf-8')
                .strip()
                == '0'
            )
        if runner.loops['total'] == 5:
            # Right after the failed baseline on row 4, the persisted baseline-done file must exist with 0:
            baseline_file = Path(get_file_baseline_structure_done(str(out_dir)))
            assert baseline_file.is_file()
            assert baseline_file.read_text(encoding='utf-8').strip() == '0'
            assert resume_baseline_structure_done(str(out_dir)) is False
        return real_check_row(hf_row, config, **kwargs)

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_check_row)
    runner.start(resume=False, offline=True)

    # Loop 5 (row 4, post-init iteration where baseline failed) had init_stage=True;
    # loop 6 (row 5, after baseline succeeded) had init_stage=False:
    stage_by_loop = {
        loop_idx: (is_init, done) for loop_idx, is_init, done in recorded_init_stages
    }
    assert stage_by_loop[4] == (True, False)
    assert stage_by_loop[5] == (True, False)
    assert stage_by_loop[6] == (False, True)
    assert resume_baseline_structure_done(str(out_dir)) is True

    # Case 1: A run directory with structure_stale.dat = 1 and no baseline_structure_done.dat
    # resumes with _baseline_structure_done = True and does not run a forced baseline solve:
    Path(get_file_baseline_structure_done(str(out_dir))).unlink(missing_ok=True)
    runner.interior_o.structure_stale = True
    runner.interior_o.structure_rollback_active = False
    runner.interior_o.write_structure_stale(str(out_dir))

    missing_file_resume = Proteus(config_path=cfg_path)
    missing_file_resume.config.invariants.mode = 'stop'
    missing_file_resume.config.interior_struct.module = 'zalmoxis'
    missing_file_resume.config.interior_struct.zalmoxis.equilibrate_init = False
    missing_file_resume.config.interior_struct.zalmoxis.update_interval = 0.0
    missing_file_resume.config.planet.temperature_mode = 'isothermal'
    missing_file_resume.config.params.stop.solid.enabled = False
    missing_file_resume.config.params.stop.iters.minimum = 1
    missing_file_resume.config.params.stop.iters.maximum = 7
    missing_file_resume.config.params.stop.time.minimum = 0.0
    missing_file_resume.config.params.stop.time.maximum = 5000.0
    missing_file_resume.config.params.out.plot_mod = None
    missing_file_resume.config.params.out.archive_mod = 'none'

    missing_file_forced_calls: list[bool] = []
    missing_file_stages: list[tuple[bool | None, bool]] = []

    def spy_missing_file_update(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        missing_file_forced_calls.append(bool(force))
        return lt, lT, lP

    def spy_missing_file_check_row(hf_row, config, **kwargs):
        missing_file_stages.append(
            (kwargs.get('init_stage'), bool(missing_file_resume._baseline_structure_done))
        )
        return real_check_row(hf_row, config, **kwargs)

    monkeypatch.setattr(int_wrap, 'update_structure_from_interior', spy_missing_file_update)
    monkeypatch.setattr(int_wrap, 'run_interior', real_interior)
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_missing_file_check_row)

    missing_file_resume.start(resume=True, offline=True)
    assert missing_file_resume._baseline_structure_done is True
    assert missing_file_forced_calls == []
    assert missing_file_stages[0] == (False, True)

    # Case 2: A run directory where baseline_structure_done.dat holds 0 retries the baseline on resume:
    write_baseline_structure_done(str(out_dir), False)
    runner.interior_o.structure_stale = True
    runner.interior_o.structure_rollback_active = True
    runner.interior_o.write_structure_stale(str(out_dir))

    resume_runner = Proteus(config_path=cfg_path)
    resume_runner.config.invariants.mode = 'stop'
    resume_runner.config.interior_struct.module = 'zalmoxis'
    resume_runner.config.interior_struct.zalmoxis.equilibrate_init = False
    resume_runner.config.interior_struct.zalmoxis.update_interval = 0.0
    resume_runner.config.planet.temperature_mode = 'isothermal'
    resume_runner.config.params.stop.solid.enabled = False
    resume_runner.config.params.stop.iters.minimum = 1
    resume_runner.config.params.stop.iters.maximum = 9
    resume_runner.config.params.stop.time.minimum = 0.0
    resume_runner.config.params.stop.time.maximum = 5000.0
    resume_runner.config.params.out.plot_mod = None
    resume_runner.config.params.out.archive_mod = 'none'

    resumed_stages: list[tuple[bool | None, bool]] = []

    def spy_resumed_check_row(hf_row, config, **kwargs):
        resumed_stages.append(
            (kwargs.get('init_stage'), bool(resume_runner._baseline_structure_done))
        )
        return real_check_row(hf_row, config, **kwargs)

    # First resumed step fails baseline again (stays init_stage=True), next succeeds:
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
    monkeypatch.setattr(int_wrap, 'run_interior', real_interior)
    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_resumed_check_row)

    resume_runner.start(resume=True, offline=True)
    assert resumed_stages[0] == (True, False)
    assert resumed_stages[1] == (False, True)
    assert resume_baseline_structure_done(str(out_dir)) is True

    # Case 3: A resume that drops trailing snapshots sets structure_rollback_active=True
    # and persists 1 to structure_rollback.dat so a second resume keeps report_only=True:
    runner.interior_o.structure_stale = False
    runner.interior_o.structure_rollback_active = False
    runner.interior_o.write_structure_stale(str(out_dir))
    assert (
        Path(get_file_structure_rollback(str(out_dir))).read_text(encoding='utf-8').strip()
        == '0'
    )

    import proteus.utils.coupler as coupler_mod

    real_select = coupler_mod.select_resumable_snapshot

    def fake_select_drops_snapshot(outdir, hf_all, **kwargs):
        trimmed, _ = real_select(outdir, hf_all, **kwargs)
        return trimmed.iloc[:-1].copy(), [float(trimmed.iloc[-1]['Time'])]

    monkeypatch.setattr(coupler_mod, 'select_resumable_snapshot', fake_select_drops_snapshot)

    dropped_resume = Proteus(config_path=cfg_path)
    dropped_resume.config.invariants.mode = 'report'
    dropped_resume.config.interior_struct.module = 'zalmoxis'
    dropped_resume.config.interior_struct.zalmoxis.equilibrate_init = False
    dropped_resume.config.interior_struct.zalmoxis.update_interval = 0.0
    dropped_resume.config.planet.temperature_mode = 'isothermal'
    dropped_resume.config.params.stop.solid.enabled = False
    dropped_resume.config.params.stop.iters.minimum = 1
    dropped_resume.config.params.stop.iters.maximum = 9
    dropped_resume.config.params.stop.time.minimum = 0.0
    dropped_resume.config.params.stop.time.maximum = 5000.0
    dropped_resume.config.params.out.plot_mod = None
    dropped_resume.config.params.out.archive_mod = 'none'

    dropped_flags: list[bool] = []

    def spy_dropped_check_row(hf_row, config, **kwargs):
        dropped_flags.append(bool(dropped_resume.interior_o.structure_rollback_active))
        return real_check_row(hf_row, config, **kwargs)

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_dropped_check_row)
    dropped_resume.start(resume=True, offline=True)
    assert dropped_flags[0] is True
    assert dropped_resume.interior_o.structure_rollback_active is True
    assert (
        Path(get_file_structure_rollback(str(out_dir))).read_text(encoding='utf-8').strip()
        == '1'
    )

    # A second resume without dropped snapshots reads the persisted rollback flag ('1')
    # and keeps M_int_target_rel_err report_only=True in stop mode:
    monkeypatch.setattr(coupler_mod, 'select_resumable_snapshot', real_select)

    def interior_with_target_mismatch(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs):
        real_interior(dirs, cfg_, hf_all, hf_row, interior_o, *args, **kwargs)
        hf_row['M_int'] = 5.97e24
        hf_row['M_int_target'] = 5.97e24 * 1.02

    monkeypatch.setattr(int_wrap, 'run_interior', interior_with_target_mismatch)

    second_resume = Proteus(config_path=cfg_path)
    second_resume.config.invariants.mode = 'stop'
    second_resume.config.interior_struct.module = 'zalmoxis'
    second_resume.config.interior_struct.zalmoxis.equilibrate_init = False
    second_resume.config.interior_struct.zalmoxis.update_interval = 0.0
    second_resume.config.planet.temperature_mode = 'isothermal'
    second_resume.config.params.stop.solid.enabled = False
    second_resume.config.params.stop.iters.minimum = 1
    second_resume.config.params.stop.iters.maximum = 10
    second_resume.config.params.stop.time.minimum = 0.0
    second_resume.config.params.stop.time.maximum = 5000.0
    second_resume.config.params.out.plot_mod = None
    second_resume.config.params.out.archive_mod = 'none'

    second_resume_report_only: list[list[bool]] = []

    def spy_second_resume_check_row(hf_row, config, **kwargs):
        viols = real_check_row(hf_row, config, **kwargs)
        target_viols = [v for v in viols if v.name == 'M_int_target_rel_err']
        second_resume_report_only.append([v.report_only for v in target_viols])
        return viols

    monkeypatch.setattr(proteus_mod, 'check_row_invariants', spy_second_resume_check_row)
    second_resume.start(resume=True, offline=True)
    assert second_resume.interior_o.structure_rollback_active is True
    assert second_resume_report_only[0] == [True]


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
        == '1'
    )
    assert (
        Path(get_file_baseline_structure_done(str(out_dir_a)))
        .read_text(encoding='utf-8')
        .strip()
        == '1'
    )

    # Part B: solve_structure sets _structure_rollback=True (and writes 1 to disk), then equilibrate_initial_state
    # clears it to False and overwrites structure_rollback.dat with 0:
    runner_b, _, out_dir_b = _build_dummy_runner(tmp_path, 'run_sync_equil_clears', max_iters=5)
    runner_b.config.invariants.mode = 'stop'
    runner_b.config.interior_struct.module = 'zalmoxis'
    runner_b.config.interior_struct.zalmoxis.equilibrate_init = True
    runner_b.config.interior_struct.zalmoxis.update_interval = 0.0
    runner_b.config.planet.temperature_mode = 'liquidus_super'

    def fake_equilibrate_clears(dirs, config, hf_row, outdir):
        assert (
            Path(get_file_structure_rollback(outdir)).read_text(encoding='utf-8').strip() == '1'
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
        == '0'
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
        == '0'
    )
    assert (
        Path(get_file_structure_rollback(str(impact_out))).read_text(encoding='utf-8').strip()
        == '0'
    )

    resumed_after_impact = Interior_t(5)
    resumed_after_impact.resume_structure_stale(str(impact_out))
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

    # When solve_structure sets _structure_rollback=True during apply_impact,
    # structure_stale and zalmoxis_fail_count are preserved and rollback is persisted as 1:
    def fake_solve_keeps_rollback(dirs, config, hf_all, hf_row, outdir, thermal_solve=True):
        hf_row['_structure_rollback'] = True

    monkeypatch.setattr(int_wrap, 'solve_structure', fake_solve_keeps_rollback)
    handler.interior_o.structure_stale = True
    handler.interior_o.structure_rollback_active = False
    handler.interior_o.zalmoxis_fail_count = 7
    acc_wrap.apply_impact(handler, event)
    assert handler.interior_o.structure_stale is True
    assert handler.interior_o.structure_rollback_active is True
    assert handler.interior_o.zalmoxis_fail_count == 7
    assert (
        Path(get_file_structure_stale(str(impact_out))).read_text(encoding='utf-8').strip()
        == '1'
    )
    assert (
        Path(get_file_structure_rollback(str(impact_out))).read_text(encoding='utf-8').strip()
        == '1'
    )
