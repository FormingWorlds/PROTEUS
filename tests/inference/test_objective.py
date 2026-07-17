"""
Unit tests for inference objective helpers and simulator wrapping.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import subprocess

import pandas as pd
import pytest
import toml
import torch

import proteus.inference.objective as objective_mod

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
    assert status == 20


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
    """A non-zero exit from the proteus binary is wrapped as
    ``RuntimeError`` with an 'exit code N' message; the exit code is
    surfaced so the caller can diagnose the failure mode.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        objective_mod.subprocess,
        'run',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(returncode=3, cmd=['proteus'])
        ),
    )

    with pytest.raises(RuntimeError, match='exit code 3') as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    # Cause-preservation guard: the original CalledProcessError must be
    # chained via __cause__ so the operator sees the failing command.
    assert isinstance(excinfo.value.__cause__, subprocess.CalledProcessError)
    # Exit-code-fidelity guard: a regression that always reported
    # 'exit code 0' or hardcoded a different code would still pass a
    # plain regex match if loose, so pin the integer through the cause.
    assert excinfo.value.__cause__.returncode == 3


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
    assert status == 20


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
