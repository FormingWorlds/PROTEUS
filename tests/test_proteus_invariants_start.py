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
    """Proteus.start with zalmoxis keeps init_stage=True on row 4 when the forced baseline fails, and restores _baseline_structure_done=False on resume."""
    import proteus.interior_energetics.wrapper as int_wrap
    from proteus.interior_energetics.common import (
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

    monkeypatch.setattr(int_wrap, 'solve_structure', fake_solve_structure)
    monkeypatch.setattr(int_wrap, '_use_superliquidus_adiabat_ic', lambda cfg: False)

    baseline_attempts = 0

    def fake_update_structure(dirs, config, hf_row, interior_o, lt, lT, lP, force=False):
        nonlocal baseline_attempts
        baseline_attempts += 1
        if baseline_attempts == 1:
            # First post-init forced baseline (row 4, loop 5) fails and falls back
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
        if runner.loops['total'] == 5:
            # Right after the failed baseline on row 4, the persisted baseline-done flag must be False
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

    # Simulate resuming a run that stopped while _baseline_structure_done was still False:
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
    resume_runner.config.params.stop.iters.maximum = 7
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

    resume_runner.config.params.stop.iters.maximum = 8
    resume_runner.start(resume=True, offline=True)
    assert resumed_stages[0] == (True, False)
    assert resumed_stages[1] == (False, True)
    assert resume_baseline_structure_done(str(out_dir)) is True


def test_proteus_start_syncs_rollback_flag_across_init_equilibrate_and_impact(
    tmp_path, monkeypatch
):
    """Proteus.start syncs interior_o.structure_rollback_active after solve_structure, equilibrate_initial_state, and apply_impact, leaving it False for liquidus_super + update_interval=0 so a post-init target error stops."""
    from types import SimpleNamespace

    import proteus.accretion.wrapper as acc_wrap
    import proteus.interior_energetics.wrapper as int_wrap

    monkeypatch.setattr(Interior_t, '_load_ps_table', lambda self, *a, **kw: None)
    monkeypatch.setattr(Proteus, '_require_paleos_tables', lambda self: None)

    # Part A: solve_structure sets _structure_rollback=True with equilibrate_init=False -> copied to runner.interior_o
    runner_a, _, _ = _build_dummy_runner(tmp_path, 'run_sync_init_only', max_iters=2)
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

    # Part B: solve_structure sets _structure_rollback=True, then equilibrate_initial_state clears it to False
    # on a liquidus_super + update_interval=0 run (cmb_regression_singleimpact shape); flag is False after init
    # and a real target error on a post-init row stops!
    runner_b, _, _ = _build_dummy_runner(tmp_path, 'run_sync_equil_clears', max_iters=5)
    runner_b.config.invariants.mode = 'stop'
    runner_b.config.interior_struct.module = 'zalmoxis'
    runner_b.config.interior_struct.zalmoxis.equilibrate_init = True
    runner_b.config.interior_struct.zalmoxis.update_interval = 0.0
    runner_b.config.planet.temperature_mode = 'liquidus_super'

    def fake_equilibrate_clears(dirs, config, hf_row, outdir):
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
    assert exc_b.value.invariant_name == 'M_core_target_rel_err'
    assert exc_b.value.status_code == 31

    # Part C: apply_impact calls solve_structure(thermal_solve=False) and syncs handler.interior_o.structure_rollback_active
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

    impact_cfg = runner_b.config
    impact_int_o = Interior_t(5)
    impact_int_o.structure_rollback_active = True
    impact_hf_row = {f'{e}_kg_atm': 0.0 for e in element_list}
    impact_hf_row.update({f'{e}_kg_total': 0.0 for e in element_list})
    impact_hf_row.update(
        {
            'Time': 500.0,
            'T_surf': 2000.0,
            'T_magma': 2500.0,
            'P_surf': 100.0,
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
        directories={'output': str(tmp_path), 'output/data': str(tmp_path)},
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
    assert handler.interior_o.structure_rollback_active is False
    with pytest.raises(InvariantViolationError) as exc_impact:
        proteus_mod.check_row_invariants(
            impact_hf_row,
            impact_cfg,
            init_stage=False,
            interior_o=handler.interior_o,
        )
    assert exc_impact.value.invariant_name == 'M_int_target_rel_err'
