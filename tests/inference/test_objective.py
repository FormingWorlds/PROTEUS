"""
Unit tests for inference objective helpers and simulator wrapping.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import logging
import subprocess

import pandas as pd
import pytest
import toml

# The Bayesian-optimisation stack ships as the optional `inference` extra,
# which installs all three together. Guarding the whole stack keeps a
# partial environment skipping rather than failing collection.
torch = pytest.importorskip('torch')
pytest.importorskip('botorch')
pytest.importorskip('gpytorch')

import proteus.inference.failures as failures_mod  # noqa: E402
import proteus.inference.objective as objective_mod  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_log_warp_monotonic_decreasing_in_squared_distance():
    """``log_warp(sq_dist)`` returns -log10(sq_dist + 1e-10): values
    closer to the target (sq_dist near 0) score higher than distant
    ones. Discrimination: a regression that flipped the sign would
    invert the ranking; a regression that dropped the offset 1e-10
    would diverge at sq_dist=0.
    """
    near = torch.tensor([1e-4], dtype=torch.double)
    far = torch.tensor([1.0], dtype=torch.double)
    score_near = objective_mod.log_warp(near)
    score_far = objective_mod.log_warp(far)
    # Closer (smaller sq_dist) -> larger score
    assert score_near.item() > score_far.item()
    # Scale guards: -log10(1e-4) ~ 4, -log10(1.0) ~ 0
    assert 3 < score_near.item() < 5
    assert -0.5 < score_far.item() < 0.5


@pytest.mark.unit
def test_log_warp_finite_at_exact_zero():
    """``log_warp(0.0)`` does not diverge: the 1e-10 offset guarantees
    finite output. Discrimination: a regression that removed the offset
    would emit -inf, which would NaN-poison downstream BO maths.
    """
    val = objective_mod.log_warp(torch.tensor([0.0], dtype=torch.double))
    assert torch.isfinite(val).all()
    # Expected ~ -log10(1e-10) = 10
    assert 9 < val.item() < 11


@pytest.mark.unit
def test_update_toml_updates_nested_keys(tmp_path):
    """``update_toml`` applies dotted-key overrides on the loaded config
    (e.g. ``section.value=2``) and creates intermediate nesting for keys
    that did not exist in the base (``new.branch.leaf=3``).
    """
    base_cfg = {'section': {'value': 1}}
    config_file = tmp_path / 'base.toml'
    out_file = tmp_path / 'nested' / 'updated.toml'
    config_file.write_text(toml.dumps(base_cfg), encoding='utf-8')

    objective_mod.update_toml(
        str(config_file),
        {'section.value': 2, 'new.branch.leaf': 3},
        str(out_file),
    )

    loaded = toml.loads(out_file.read_text(encoding='utf-8'))
    assert loaded['section']['value'] == 2
    assert loaded['new']['branch']['leaf'] == 3


@pytest.mark.unit
def test_apply_nested_updates_mutates_in_place_and_rejects_value_paths():
    """``apply_nested_updates`` writes dotted keys into the dict it was given,
    creating the sections a new key needs, and refuses a path that descends
    through an entry holding a value. The refusal matters because a swept
    parameter name is user-supplied: ``planet.mass_tot.value`` would otherwise
    fail with an attribute error naming nothing the user wrote.
    """
    config = {'section': {'value': 1}}
    returned = objective_mod.apply_nested_updates(
        config, {'section.value': 2, 'new.branch.leaf': 3}
    )
    assert config['section']['value'] == 2
    assert config['new']['branch']['leaf'] == 3
    # In-place: the same object is handed back, so a caller holding the
    # original reference sees the updates.
    assert returned is config

    with pytest.raises(ValueError, match="'section.value' holds a value"):
        objective_mod.apply_nested_updates(config, {'section.value.deeper': 4})
    # The refused key is not written. Updates are applied as they are walked,
    # so an earlier key in the same call would already have been applied; this
    # pins only that the rejected one was not.
    assert config['section']['value'] == 2


@pytest.mark.unit
def test_run_proteus_success_handles_escaped_atmosphere(monkeypatch, tmp_path):
    """``run_proteus`` handles the escaped-atmosphere case (P_surf=0):
    the observable dictionary is populated with zeros instead of NaN,
    and ``update_toml`` is invoked exactly twice (once per simulator pass)
    so the inversion harness sees a numeric value.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    pd.DataFrame([{'P_surf': 0.0, 'atm_kg_per_mol': 44.0}]).to_csv(
        out_abs / 'runtime_helpfile.csv', sep=' ', index=False
    )

    updates = []
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(
        objective_mod,
        'update_toml',
        lambda config_file, values, output_file: updates.append(
            (config_file, values, output_file)
        ),
    )
    monkeypatch.setattr(objective_mod.subprocess, 'run', lambda *args, **kwargs: None)

    parameters = {}
    obs, status = objective_mod.run_proteus(
        parameters=parameters,
        worker=1,
        iter=2,
        observables=['P_surf', 'atm_kg_per_mol'],
        ref_config='reference.toml',
        output='dummy_output',
    )

    assert obs['P_surf'] == pytest.approx(0.0)
    assert obs['atm_kg_per_mol'] == pytest.approx(0.0)
    assert len(updates) == 2
    # No status file was written, which is reported as such rather than as a
    # generic error: a run that dies during start-up and a run that reaches
    # the main loop and fails there call for different investigations.
    assert status == objective_mod.STATUS_MISSING
    assert status != 20


@pytest.mark.unit
def test_run_proteus_raises_when_command_missing(monkeypatch, tmp_path):
    """A ``FileNotFoundError`` from ``subprocess.run`` (i.e. the proteus
    binary is not on PATH) is wrapped as ``RuntimeError`` with a
    'command not found' message.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)

    run_calls = []

    def _fake_run(*args, **kwargs):
        run_calls.append((args, kwargs))
        raise FileNotFoundError('missing')

    monkeypatch.setattr(objective_mod.subprocess, 'run', _fake_run)

    with pytest.raises(RuntimeError, match='command not found') as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    # Cause-preservation guard: the original FileNotFoundError must be
    # chained via __cause__. A regression that swallowed the cause and
    # raised a bare RuntimeError would still match the 'command not found'
    # text but lose the traceback the operator needs.
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)
    # Side-effect guard: subprocess.run must have been invoked exactly
    # once. A regression that short-circuited before dispatch would
    # still raise but with a different (constant) error path.
    assert len(run_calls) == 1


@pytest.mark.unit
def test_run_proteus_raises_when_command_fails(monkeypatch, tmp_path):
    """A non-zero exit from the proteus binary is reported as a
    ``ProteusRunFailure`` naming the run, its exit code, and the status the
    simulator recorded for itself, so the failure mode can be diagnosed
    without opening the study by hand.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    # The simulator recorded an atmosphere-model error before exiting. The
    # report must carry this, not a code inferred from the exit status.
    (out_abs / 'status').write_text('22\nError (Atmosphere model)\n', encoding='utf-8')
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)

    def _fake_run(*_args, **kwargs):
        # The real simulator writes to the stream it is handed before it dies.
        kwargs['stdout'].write('boom\n')
        kwargs['stdout'].flush()
        raise subprocess.CalledProcessError(returncode=3, cmd=['proteus'])

    monkeypatch.setattr(objective_mod.subprocess, 'run', _fake_run)

    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        objective_mod.run_proteus(
            parameters={'planet.mass_tot': 1.25},
            worker=0,
            iter=0,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    failure = excinfo.value
    # Cause-preservation guard: the original CalledProcessError must be
    # chained via __cause__ so the operator sees the failing command.
    assert isinstance(failure.__cause__, subprocess.CalledProcessError)
    # Exit-code-fidelity guard: a regression that always reported
    # 'exit code 0' or hardcoded a different code would still pass a
    # plain regex match if loose, so pin the integer through the cause.
    assert failure.__cause__.returncode == 3
    assert failure.exit_code == 3
    # Status fidelity: the status file is read on the failure path. A
    # regression that raised before reading it would report the missing
    # sentinel, and one that kept the old hard-coded fallback would report 20.
    assert failure.status == 22
    assert 'Atmosphere' in failure.status_desc
    # The swept parameter is named; the fixed per-run overrides are not,
    # because they carry no information about which sample failed.
    assert failure.parameters == {'planet.mass_tot': pytest.approx(1.25)}
    assert 'params.out.path' not in failure.parameters
    # The capture is kept beside the run folder, not inside it: the simulator
    # empties its own output directory once it starts, which would unlink a
    # file held open there.
    console = out_abs.parent / f'{out_abs.name}{objective_mod.CHILD_CONSOLE_SUFFIX}'
    assert console.is_file()
    assert 'boom' in console.read_text(encoding='utf-8')
    assert not (out_abs / console.name).exists()
    # The failure names that capture rather than copying its contents. A run
    # that dies before its own logger exists leaves nothing else to read, so a
    # report that named no path would leave the cause unreachable.
    assert failure.console_path == str(console)
    rendered = failure.report()
    assert 'worker=0 iter=0' in rendered
    assert 'planet.mass_tot=1.25' in rendered
    assert str(console) in rendered


@pytest.mark.unit
def test_run_proteus_raises_on_missing_observable(monkeypatch, tmp_path):
    """Requesting an observable that the simulator did not write to the
    helpfile raises ``KeyError`` with a 'Requested observable' message,
    so a typo in the inference config fails loudly rather than producing
    silent NaN results.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    pd.DataFrame([{'P_surf': 1.0}]).to_csv(
        out_abs / 'runtime_helpfile.csv', sep=' ', index=False
    )

    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(objective_mod.subprocess, 'run', lambda *args, **kwargs: None)

    with pytest.raises(KeyError, match='Requested observable') as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['not_present'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    # Identity guard: the raised KeyError must name the offending
    # observable explicitly. A regression that emitted a generic
    # 'Requested observable not found' without the field name would
    # match the regex above but lose the diagnostic information.
    assert 'not_present' in str(excinfo.value)
    # Discrimination: a valid observable on the same helpfile must
    # complete normally. This rules out a regression that hard-raises
    # KeyError on every input regardless of the observables list.
    obs, status = objective_mod.run_proteus(
        parameters={},
        worker=0,
        iter=0,
        observables=['P_surf'],
        ref_config='reference.toml',
        output='dummy_output',
    )
    assert obs['P_surf'] == pytest.approx(1.0)
    assert status == objective_mod.STATUS_MISSING


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_eval_obj_mixes_log_and_linear_variables(monkeypatch):
    """``eval_obj`` evaluates log-relative residuals for log-scaled
    observables and linear-relative residuals for linear ones, then
    returns ``-log10(sum_sq + 1e-10)``. The mixed-mode arithmetic is the
    point of this test: log and linear contributions enter the sum
    using different normalisations.
    """
    monkeypatch.setattr(objective_mod, 'variable_is_logarithmic', lambda key: key == 'P_surf')

    sim = {'P_surf': 1e-6, 'R_obs': 2.0}
    tru = {'P_surf': 1e-5, 'R_obs': 1.0}

    value = objective_mod.eval_obj(sim, tru)

    expected_sq = ((1.0 - (-6.0 / -5.0)) ** 2) + ((1.0 - 2.0 / 1.0) ** 2)
    expected = -torch.log10(torch.tensor([[expected_sq + 1e-10]], dtype=torch.double))
    assert value.item() == pytest.approx(expected.item())
    # Discrimination guard: a regression that treated P_surf as linear
    # (1e-6 vs 1e-5: relative residual 0.9) would land at a very
    # different objective than the log-mode (-6/-5 = 1.2: residual
    # 0.04). Pin the magnitude with a wrong-mode counter-value.
    sim_lin = {'P_surf': 1e-6, 'R_obs': 2.0}
    expected_sq_wrong = ((1.0 - 1e-6 / 1e-5) ** 2) + ((1.0 - 2.0 / 1.0) ** 2)
    expected_wrong = -torch.log10(
        torch.tensor([[expected_sq_wrong + 1e-10]], dtype=torch.double)
    )
    assert abs(value.item() - expected_wrong.item()) > 0.1
    # Sign / boundedness guard: the objective is -log10(sum_sq + 1e-10).
    # With sum_sq > 0 (mismatched sim vs tru), the inner argument
    # exceeds 1e-10 and the result is finite. A regression that
    # produced NaN or inf would fail an isfinite check.
    assert torch.isfinite(value).all()
    # Identical sim == tru produces sum_sq = 0, hence -log10(1e-10) = 10.
    value_match = objective_mod.eval_obj(sim_lin, sim_lin)
    assert value_match.item() == pytest.approx(10.0, rel=1e-6)


@pytest.mark.unit
def test_eval_obj_handles_zero_true_value():
    """Zero-valued observables should use an EPS_CLIP offset denominator
    to avoid division-by-zero.

    Discrimination: a denominator of exactly 0.0 would produce +inf or NaN;
    the EPS_CLIP offset must make the result finite.
    """
    sim = {'R_obs': 2.0}
    tru = {'R_obs': 0.0}

    value = objective_mod.eval_obj(sim, tru)

    denom = 0.0 + objective_mod.EPS_CLIP
    expected_sq = (1.0 - 2.0 / denom) ** 2
    expected = -torch.log10(
        torch.tensor([[expected_sq + objective_mod.EPS_CLIP]], dtype=torch.double)
    )
    assert value.item() == pytest.approx(expected.item())
    # Without EPS_CLIP the result would be +inf or NaN; finite output
    # is the key contract of the zero-denominator guard.
    assert torch.isfinite(value).all()


@pytest.mark.unit
def test_prot_builder_unnormalizes_and_calls_J(monkeypatch):
    """``prot_builder`` returns a closure that un-normalises an x in
    [0, 1]^d to the physical parameter ranges (so x=0.5 with bounds
    [0, 10] maps to 5.0) before calling the inner objective ``J``.
    """
    captured = {}

    def fake_J(x, **kwargs):
        captured['x'] = x
        captured['kwargs'] = kwargs
        return torch.tensor([[5.0]], dtype=torch.double)

    monkeypatch.setattr(objective_mod, 'J', fake_J)

    f = objective_mod.prot_builder(
        parameters={'a': [0.0, 10.0], 'b': [2.0, 4.0]},
        observables={'obs': 1.0},
        worker=7,
        iter=9,
        output='out_dir',
        ref_config='ref.toml',
        failure_codes=[0, 1],
    )

    y = f(torch.tensor([[0.5, 0.25]], dtype=torch.double))

    assert y.item() == pytest.approx(5.0)
    assert captured['x'][0, 0].item() == pytest.approx(5.0)
    assert captured['x'][0, 1].item() == pytest.approx(2.5)


@pytest.mark.unit
def test_prot_builder_unnormalizes_log_scaled_parameter(monkeypatch):
    """Surface pressure spans orders of magnitude, so log scaling must round-trip."""
    captured = {}

    def fake_J(x, **kwargs):
        captured['x'] = x
        return torch.tensor([[1.0]], dtype=torch.double)

    monkeypatch.setattr(objective_mod, 'J', fake_J)

    f = objective_mod.prot_builder(
        parameters={'P_surf': [1e-3, 1e3], 'struct.mass_tot': [1.0, 3.0]},
        observables={'obs': 1.0},
        worker=0,
        iter=0,
        output='out_dir',
        ref_config='ref.toml',
        failure_codes=[0, 1],
    )

    f(torch.tensor([[0.5, 0.25]], dtype=torch.double))

    assert captured['x'][0, 0].item() == pytest.approx(1.0)
    assert captured['x'][0, 1].item() == pytest.approx(1.5)


# ============================================================================
# Failure reporting for a single simulator run
# ============================================================================


@pytest.mark.unit
def test_run_proteus_failure_distinguishes_a_missing_status_from_a_generic_error(
    monkeypatch, tmp_path
):
    """A run that dies before writing a status file is reported as having
    written none, rather than as a generic configuration error. The two call
    for different investigations: the first points at the simulator's start-up
    (environment, reference data), the second at the model configuration.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)

    def _fake_run(*_args, **kwargs):
        kwargs['stdout'].write('Error: no\n')
        kwargs['stdout'].flush()
        raise subprocess.CalledProcessError(returncode=1, cmd=['proteus'])

    monkeypatch.setattr(objective_mod.subprocess, 'run', _fake_run)

    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=3,
            iter=4,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    failure = excinfo.value
    assert failure.status == objective_mod.STATUS_MISSING
    assert 'no readable status file' in failure.status_desc
    # Discrimination: the previous behaviour reported code 20 for this case,
    # which reads as a configuration fault the user does not have.
    assert failure.status != 20
    assert 'Generic' not in failure.status_desc
    # No logfile exists either, so the report must not invent one.
    assert failure.log_path is None
    assert 'logfile' not in failure.report()

    # Edge case: the same run with a status file present reports that status,
    # which proves the sentinel above came from the absent file and not from a
    # reader that always fails.
    (out_abs / 'status').write_text('21\nError (Interior model)\n', encoding='utf-8')
    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=3,
            iter=4,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    assert excinfo.value.status == 21


@pytest.mark.unit
def test_run_proteus_failure_points_at_the_simulator_logfile(monkeypatch, tmp_path):
    """When the failed run left a logfile, the report names it. That file holds
    the traceback the simulator captured for itself, and is the only place the
    cause of a mid-run crash is recorded.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    (out_abs / 'proteus_00.log').write_text('early\n', encoding='utf-8')
    (out_abs / 'proteus_01.log').write_text('CRITICAL Uncaught exception\n', encoding='utf-8')
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        objective_mod.subprocess,
        'run',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(returncode=1, cmd=['proteus'])
        ),
    )

    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    # The newest logfile is the one the failed run wrote; an earlier one
    # belongs to a previous attempt in the same folder.
    assert excinfo.value.log_path.endswith('proteus_01.log')
    assert 'proteus_01.log' in excinfo.value.report()


@pytest.mark.unit
def test_run_proteus_reports_a_clean_exit_that_produced_no_output(monkeypatch, tmp_path):
    """A run that exits zero but writes no readable helpfile is reported as a
    failed sample rather than crashing the study with a bare parser error. The
    exit code is recorded as zero so the report does not suggest a crash.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(objective_mod.subprocess, 'run', lambda *args, **kwargs: None)

    # No helpfile at all.
    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    assert excinfo.value.exit_code == 0
    assert 'no readable output' in excinfo.value.reason

    # Edge case: a helpfile that exists but holds no rows.
    (out_abs / 'runtime_helpfile.csv').write_text('', encoding='utf-8')
    with pytest.raises(objective_mod.ProteusRunFailure):
        objective_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )

    # Discrimination: a helpfile with a usable row completes normally, so the
    # two failures above come from the output and not from an unconditional
    # raise on this code path.
    pd.DataFrame([{'P_surf': 2.5}]).to_csv(
        out_abs / 'runtime_helpfile.csv', sep=' ', index=False
    )
    obs, _status = objective_mod.run_proteus(
        parameters={},
        worker=0,
        iter=0,
        observables=['P_surf'],
        ref_config='reference.toml',
        output='dummy_output',
    )
    assert obs['P_surf'] == pytest.approx(2.5)


@pytest.mark.unit
def test_J_scores_a_failed_run_badly_and_keeps_the_study_running(monkeypatch, tmp_path, caplog):
    """A parameter combination the simulator cannot integrate is scored as a
    poor sample so the sweep continues, and the failure is reported once in
    full and recorded for the end-of-study tally. Aborting instead would end a
    study on the first unphysical corner of the parameter box.
    """
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(tmp_path)}
    )
    failure = objective_mod.ProteusRunFailure(
        reason='the simulator exited with an error',
        worker=1,
        iter=2,
        out_dir='/study/workers/w_1/i_2',
        exit_code=1,
        status=21,
        parameters={'planet.mass_tot': 3.0},
    )

    def _fail(**_kwargs):
        raise failure

    monkeypatch.setattr(objective_mod, 'run_proteus', _fail)
    monkeypatch.setenv(failures_mod._ABORT_ON_FAILURE_ENV, '0')

    with caplog.at_level('WARNING'):
        value = objective_mod.J(
            x=torch.tensor([[0.5]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'P_surf': 1.0},
            worker=1,
            iter=2,
            output='dummy_output',
            ref_config='reference.toml',
        )

    assert value.shape == (1, 1)
    assert value.item() == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    # The score must be far below any value a successful run can produce, or
    # the optimiser would be drawn toward the region that fails.
    assert value.item() < -10.0
    # Reported once, in full: the status description and the output folder are
    # what let the user find the run.
    reported = '\n'.join(record.getMessage() for record in caplog.records)
    assert 'Interior model' in reported
    assert '/study/workers/w_1/i_2' in reported

    # The same failure is left on disk for the end-of-study tally, because a
    # log line scrolls past and a study that failed mostly needs a count.
    recorded = failures_mod.read_failure_records(tmp_path)
    assert [(r['worker'], r['iter'], r['status']) for r in recorded] == [(1, 2, 21)]
    assert recorded[0]['planet.mass_tot'] == pytest.approx(3.0)

    # Opting in turns the same failure into a hard stop. `set_abort_on_failure`
    # is the writer under test; monkeypatch restores the variable afterwards.
    failures_mod.set_abort_on_failure(True)
    with pytest.raises(objective_mod.ProteusRunFailure):
        objective_mod.J(
            x=torch.tensor([[0.5]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'P_surf': 1.0},
            worker=1,
            iter=2,
            output='dummy_output',
            ref_config='reference.toml',
        )


@pytest.mark.unit
def test_J_scores_a_clean_run_that_stopped_in_an_error_state(monkeypatch, tmp_path, caplog):
    """A run that exits cleanly but records an error status is scored badly and
    named in the log. Status 25 is the only error code reachable this way: it
    is written when a run is stopped through its keepalive file, and the
    simulator then terminates normally.

    'R_obs' is used as the observable because it is compared linearly, which
    gives the exact-match objective a closed form to pin against.
    """
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(tmp_path)}
    )
    monkeypatch.setattr(
        objective_mod,
        'run_proteus',
        lambda **_kwargs: ({'R_obs': 9.25e6}, 25),
    )
    monkeypatch.setenv(failures_mod._ABORT_ON_FAILURE_ENV, '0')

    with caplog.at_level('WARNING'):
        value = objective_mod.J(
            x=torch.tensor([[0.5]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'R_obs': 9.25e6},
            worker=0,
            iter=0,
            output='dummy_output',
            ref_config='reference.toml',
        )
    assert value.item() == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    assert 'status 25' in '\n'.join(r.getMessage() for r in caplog.records)
    # Counted in the end-of-study tally alongside the runs that crashed. A
    # tally that covered only crashes would understate a study stopped by hand.
    recorded = failures_mod.read_failure_records(tmp_path)
    assert [r['status'] for r in recorded] == [25]
    assert recorded[0]['exit_code'] == 0

    # Discrimination: the same observables under a completed status (13,
    # "target time reached") are scored normally, which rules out a regression
    # that returns the failure score for every run.
    monkeypatch.setattr(
        objective_mod,
        'run_proteus',
        lambda **_kwargs: ({'R_obs': 9.25e6}, 13),
    )
    good = objective_mod.J(
        x=torch.tensor([[0.5]], dtype=torch.double),
        parameters=['planet.mass_tot'],
        true_observables={'R_obs': 9.25e6},
        worker=0,
        iter=0,
        output='dummy_output',
        ref_config='reference.toml',
        failure_codes=[],
    )
    # Closed form for an exact match on a linear observable: the normalised
    # difference is zero, so sq_dist is zero and the score is
    # -log10(0 + EPS_CLIP) = -log10(1e-10) = 10.
    assert good.item() == pytest.approx(10.0, rel=1e-9)
    # Sign guard: a flipped objective would land at -10, which is still above
    # BAD_OBJ_VALUE and would pass a bare "better than failure" assertion.
    assert good.item() > 0
    # Scale guard: the failure score is -20, so the two are far apart.
    assert good.item() - objective_mod.BAD_OBJ_VALUE > 25.0


@pytest.mark.unit
def test_J_aborts_on_a_clean_run_that_stopped_in_an_error_state(monkeypatch, tmp_path):
    """`abort_on_failure` stops the study on a run that exited cleanly but
    recorded an error status, the same way it stops on a run that crashed.
    Both are faults; only the route by which the simulator reported them
    differs, so honouring the setting on one and not the other would let a
    study set up with `abort_on_failure = true` run to completion on a
    reference config that fails every evaluation.

    The asymmetry the setting must keep: an excluded status completed
    normally, so it is scored as a poor sample and the study carries on even
    with aborting enabled.
    """
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(tmp_path)}
    )

    def _run(status, worker, iter, codes=()):
        monkeypatch.setattr(
            objective_mod,
            'run_proteus',
            lambda **_kwargs: ({'R_obs': 9.25e6}, status),
        )
        return objective_mod.J(
            x=torch.tensor([[0.5]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'R_obs': 9.25e6},
            worker=worker,
            iter=iter,
            output='dummy_output',
            ref_config='reference.toml',
            failure_codes=list(codes),
        )

    monkeypatch.setenv(failures_mod._ABORT_ON_FAILURE_ENV, '1')

    # Status 25: written when a run is stopped through its keepalive file, so
    # the simulator exits 0 and the fault is visible only in the status file.
    with pytest.raises(objective_mod.ProteusRunFailure) as caught:
        _run(25, worker=0, iter=0)
    assert caught.value.status == 25
    assert caught.value.category == objective_mod.CATEGORY_FAILURE
    # Exit code 0 is the whole point of this path: the abort must not depend
    # on the child having exited non-zero.
    assert caught.value.exit_code == 0

    # Boundary of the failure set: STATUS_MISSING is the lowest code treated
    # as a fault, and the run's own account of itself is absent, so it cannot
    # be scored. A range check written as `20 <= status <= 28` alone would
    # miss it.
    with pytest.raises(objective_mod.ProteusRunFailure) as missing:
        _run(objective_mod.STATUS_MISSING, worker=0, iter=1)
    assert missing.value.status == objective_mod.STATUS_MISSING

    # The record is written before the abort, so an aborted study still says
    # on disk what stopped it rather than leaving only the traceback.
    recorded = failures_mod.read_failure_records(tmp_path)
    # Ordered by (worker, iter), so the status-25 run at iter 0 comes first.
    assert [r['status'] for r in recorded] == [25, objective_mod.STATUS_MISSING]

    # Discrimination against a fix that aborts on `failed or excluded`: an
    # excluded status is scored as a poor sample and returns normally.
    excluded = _run(11, worker=1, iter=0, codes=(11,))
    assert excluded.item() == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    # Boundedness: the failure score sits far below anything a completed run
    # can reach, so the optimiser is not drawn toward the excluded region.
    assert excluded.item() < -10.0
    assert failures_mod.read_failure_records(tmp_path)[-1]['category'] == (
        objective_mod.CATEGORY_EXCLUDED
    )

    # Discrimination against a regression that raises unconditionally: with
    # the setting off, the same error status is scored and the study goes on.
    monkeypatch.setenv(failures_mod._ABORT_ON_FAILURE_ENV, '0')
    scored = _run(25, worker=2, iter=0)
    assert scored.item() == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    assert scored.item() < -10.0


@pytest.mark.unit
def test_J_treats_the_documented_error_codes_as_failures(monkeypatch, tmp_path):
    """The failure range covers the error statuses the simulator can record.
    Code 28 is the highest error the status table defines; 29 is a completion
    ('planet evaporated') and no current code path writes it, so it must not
    be scored as a failure by an off-by-one in the range bound.
    """
    monkeypatch.setenv(failures_mod._ABORT_ON_FAILURE_ENV, '0')
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(tmp_path)}
    )

    def _score(status):
        monkeypatch.setattr(
            objective_mod,
            'run_proteus',
            lambda **_kwargs: ({'R_obs': 9.25e6}, status),
        )
        return objective_mod.J(
            x=torch.tensor([[0.5]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'R_obs': 9.25e6},
            worker=0,
            iter=0,
            output='dummy_output',
            ref_config='reference.toml',
        ).item()

    # Highest defined error code, and the escape-model error below it.
    assert _score(28) == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    assert _score(21) == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    # Completion codes are scored on their observables.
    assert _score(29) == pytest.approx(10.0, rel=1e-9)
    assert _score(13) == pytest.approx(10.0, rel=1e-9)
    # A run that never updated its status past 'Running' died mid-flight.
    assert _score(1) == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    # An unreadable status file is treated as a failure, because the run's own
    # account of itself is missing and its output cannot be trusted.
    assert _score(objective_mod.STATUS_MISSING) == pytest.approx(objective_mod.BAD_OBJ_VALUE)


@pytest.mark.unit
def test_J_separates_an_excluded_outcome_from_a_failed_run(monkeypatch, tmp_path, caplog):
    """A status named in `failure_codes` marks an outcome the study does not fit
    against, not a fault. A run stopped by its clock limit (status 11) completed
    normally, so it is scored as a poor sample and reported at info level, while
    an error status (21, interior model) is reported as a run that produced
    nothing usable. Reporting the first as the second sends the user looking for
    a bug in a run that did exactly what it was configured to do.
    """
    monkeypatch.setenv(failures_mod._ABORT_ON_FAILURE_ENV, '0')
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(tmp_path)}
    )

    def _score(status, worker):
        monkeypatch.setattr(
            objective_mod,
            'run_proteus',
            lambda **_kwargs: ({'R_obs': 9.25e6}, status),
        )
        return objective_mod.J(
            x=torch.tensor([[0.5]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'R_obs': 9.25e6},
            worker=worker,
            iter=0,
            output='dummy_output',
            ref_config='reference.toml',
            failure_codes=[11],
        ).item()

    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.objective'):
        excluded = _score(11, worker=0)

    # The optimiser must still be steered away from the excluded region, so the
    # score is the same one a failure carries.
    assert excluded == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    assert not [r for r in caplog.records if r.levelname in ('WARNING', 'ERROR')]
    reported = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'excludes' in reported
    assert 'maximum clock runtime' in reported
    assert 'failure state' not in reported
    assert 'failed for worker=0' not in reported

    # The record is kept for the end-of-study tally, labelled so the tally can
    # count it apart from the runs that genuinely failed.
    recorded = failures_mod.read_failure_records(tmp_path)
    assert [(r['status'], r['category']) for r in recorded] == [
        (11, objective_mod.CATEGORY_EXCLUDED)
    ]

    # Discrimination: an error status under the same call is still a failure,
    # warned about and recorded under the other category. Without this the test
    # would pass against a regression that labelled every run 'excluded'.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.objective'):
        failed = _score(21, worker=1)
    assert failed == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    warnings = [r for r in caplog.records if r.levelname == 'WARNING']
    assert len(warnings) == 1
    assert 'failed for worker=1' in warnings[0].getMessage()
    assert 'stopped in a failure state' in warnings[0].getMessage()
    assert 'status 21' in warnings[0].getMessage()
    recorded = failures_mod.read_failure_records(tmp_path)
    assert [(r['status'], r['category']) for r in recorded] == [
        (11, objective_mod.CATEGORY_EXCLUDED),
        (21, objective_mod.CATEGORY_FAILURE),
    ]

    # Discrimination: a completion status that the study does not exclude is
    # scored on its observables and leaves no record at all. The exact match on
    # a linear observable has the closed form -log10(0 + 1e-10) = 10.
    assert _score(13, worker=2) == pytest.approx(10.0, rel=1e-9)
    assert len(failures_mod.read_failure_records(tmp_path)) == 2


# ============================================================================
# Failure records: written per evaluation, read back for the study summary
# ============================================================================


@pytest.mark.unit
def test_run_output_dir_names_the_folder_the_simulator_is_given(monkeypatch, tmp_path):
    """The per-evaluation folder is derived in one place, so the path a failure
    report names is the path the simulator was told to write to. Initial
    samples use worker -1, which must survive the same construction.
    """
    monkeypatch.setattr(
        objective_mod,
        'get_proteus_directories',
        lambda path: {'output': str(tmp_path / path)},
    )

    rel, absolute = objective_mod.run_output_dir('study', 2, 7)
    assert rel.as_posix() == 'study/workers/w_2/i_7'
    assert absolute == tmp_path / 'study' / 'workers' / 'w_2' / 'i_7'

    # Initial sampling identifies itself with worker -1 rather than a worker
    # index, and must land in its own folder rather than colliding with w_1.
    rel_init, _ = objective_mod.run_output_dir('study', -1, 7)
    assert rel_init.as_posix() == 'study/workers/w_-1/i_7'
    assert rel_init != rel
