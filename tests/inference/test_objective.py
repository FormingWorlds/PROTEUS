"""
Unit tests for inference objective helpers and simulator wrapping.

References:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import subprocess

import pandas as pd
import pytest
import toml
import torch

import proteus.inference.objective as objective_mod


@pytest.mark.unit
def test_update_toml_updates_nested_keys(tmp_path):
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
    obs = objective_mod.run_proteus(
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


@pytest.mark.unit
def test_run_proteus_raises_when_command_missing(monkeypatch, tmp_path):
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        objective_mod.subprocess,
        'run',
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError('missing')),
    )

    with pytest.raises(RuntimeError, match='command not found'):
        objective_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )


@pytest.mark.unit
def test_run_proteus_raises_when_command_fails(monkeypatch, tmp_path):
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

    with pytest.raises(RuntimeError, match='exit code 3'):
        objective_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )


@pytest.mark.unit
def test_run_proteus_raises_on_missing_observable(monkeypatch, tmp_path):
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

    with pytest.raises(KeyError, match='Requested observable'):
        objective_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['not_present'],
            ref_config='reference.toml',
            output='dummy_output',
        )


@pytest.mark.unit
def test_eval_obj_mixes_log_and_linear_variables(monkeypatch):
    monkeypatch.setattr(objective_mod, 'variable_is_logarithmic', lambda key: key == 'P_surf')

    sim = {'P_surf': 1e-6, 'R_obs': 2.0}
    tru = {'P_surf': 1e-5, 'R_obs': 1.0}

    value = objective_mod.eval_obj(sim, tru)

    expected_sq = ((1.0 - (-6.0 / -5.0)) ** 2) + ((1.0 - 2.0 / 1.0) ** 2)
    expected = -torch.log10(torch.tensor([[expected_sq + 1e-10]], dtype=torch.double))
    assert value.item() == pytest.approx(expected.item())


@pytest.mark.unit
def test_eval_obj_handles_zero_true_value():
    """Zero-valued observables should use an offset denominator."""
    sim = {'R_obs': 2.0}
    tru = {'R_obs': 0.0}

    value = objective_mod.eval_obj(sim, tru)

    denom = 0.0 + objective_mod.EPS_CLIP
    expected_sq = (1.0 - 2.0 / denom) ** 2
    expected = -torch.log10(
        torch.tensor([[expected_sq + objective_mod.EPS_CLIP]], dtype=torch.double)
    )
    assert value.item() == pytest.approx(expected.item())


@pytest.mark.unit
def test_prot_builder_unnormalizes_and_calls_J(monkeypatch):
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
