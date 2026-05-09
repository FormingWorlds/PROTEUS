"""
Unit tests for inference utility helpers.

References:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

import proteus.inference.utils as utils_mod


@pytest.mark.unit
def test_save_dataset_csv_validates_input_shapes(tmp_path):
    out = tmp_path / 'dataset.csv'

    with pytest.raises(ValueError, match='Expected X to be 2D'):
        utils_mod.save_dataset_csv(
            torch.tensor([1.0], dtype=torch.double), torch.zeros((1, 1)), str(out)
        )

    with pytest.raises(ValueError, match='Expected Y shape'):
        utils_mod.save_dataset_csv(torch.zeros((2, 1)), torch.zeros(2), str(out))

    with pytest.raises(ValueError, match='row counts differ'):
        utils_mod.save_dataset_csv(torch.zeros((2, 1)), torch.zeros((3, 1)), str(out))


@pytest.mark.unit
def test_load_dataset_csv_validates_required_columns(tmp_path):

    # check parameter key
    no_x = tmp_path / 'no_x.csv'
    pd.DataFrame({'y': [1.0]}).to_csv(no_x, index=False)
    with pytest.raises(ValueError, match='x_<index>'):
        utils_mod.load_dataset_csv(str(no_x))

    # check objective value key
    no_y = tmp_path / 'no_y.csv'
    pd.DataFrame({'x_0': [1.0]}).to_csv(no_y, index=False)
    with pytest.raises(ValueError, match="Missing 'y' column"):
        utils_mod.load_dataset_csv(str(no_y))


@pytest.mark.unit
def test_get_obj_reads_square_worker_grid(monkeypatch, tmp_path):
    seen_paths: list[Path] = []

    def fake_get_obs(out_csv, observables):
        seen_paths.append(Path(out_csv))
        assert observables == {'P_surf'}
        return {'P_surf': float(len(seen_paths))}

    def fake_eval_obj(sim_obs, _true_obs):
        return torch.tensor([[sim_obs['P_surf']]], dtype=torch.double)

    monkeypatch.setattr(utils_mod, 'get_obs', fake_get_obs)
    monkeypatch.setattr(utils_mod, 'eval_obj', fake_eval_obj)

    # try evaluating the objective on a 2x2 worker grid, with Psurf observable only
    result = utils_mod.get_obj({'P_surf': 1.0}, n=2, path=tmp_path)

    # worker grid
    assert result.shape == (2, 2)
    assert result.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    # worker results
    assert seen_paths[0] == tmp_path / 'workers' / 'w_-1' / 'i_0' / 'runtime_helpfile.csv'
    assert seen_paths[-1] == tmp_path / 'workers' / 'w_-1' / 'i_3' / 'runtime_helpfile.csv'
