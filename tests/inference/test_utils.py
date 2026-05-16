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

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_save_dataset_csv_validates_input_shapes(tmp_path):
    """``save_dataset_csv`` rejects malformed inputs with specific
    ValueError messages: X must be 2D, Y must have matching column rank,
    and the row counts of X and Y must agree.
    """
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
    """``load_dataset_csv`` raises if the CSV is missing either an
    ``x_<index>`` parameter column or the ``y`` objective column.
    """
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
    """``get_obj`` walks an n*n worker grid, reads each worker's
    ``runtime_helpfile.csv``, evaluates the objective, and returns the
    values as an n*n tensor. Worker paths follow the
    ``workers/w_-1/i_<k>/`` convention.
    """
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


# ---------------------------------------------------------------------------
# Pure-Python helpers: str_time, get_nested, flatten
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_str_time_returns_iso_like_string_with_timezone():
    """``str_time`` returns the current wall-clock time formatted as
    'YYYY-MM-DD HH:MM:SS TZ'. Discrimination: a regression that dropped
    the seconds field or the timezone abbreviation would not match
    the expected length+structure.
    """
    s = utils_mod.str_time()
    # Expected shape: 19 chars for date+time + 1 space + timezone abbrev
    assert isinstance(s, str)
    # Discrimination: must contain at least one ':' (HH:MM:SS) and one '-' (YYYY-MM-DD)
    assert s.count(':') >= 2
    assert s.count('-') >= 2
    # Discrimination: must be at least 20 chars (full date+time) and at most ~30 chars
    assert 20 <= len(s) <= 40


@pytest.mark.unit
def test_get_nested_drills_into_nested_dict():
    """``get_nested`` resolves a dot-separated key path through a nested
    dict. Discrimination: get_nested(d, 'a.b.c') must return the value
    at d['a']['b']['c'], not a different branch.
    """
    config = {'a': {'b': {'c': 42, 'd': 'wrong'}}, 'x': 'unrelated'}
    assert utils_mod.get_nested(config, 'a.b.c') == 42
    # Discrimination: the sibling key 'a.b.d' must resolve independently
    assert utils_mod.get_nested(config, 'a.b.d') == 'wrong'


@pytest.mark.unit
def test_get_nested_accepts_custom_separator():
    """``get_nested`` honours the ``sep`` argument so callers can use a
    different separator (e.g. '/'). Discrimination: with sep='/' the
    function must use '/' for splitting (not '.'); the same key under
    the default separator would NOT resolve because 'a/b' is a single
    literal segment under sep='.'.
    """
    config = {'a': {'b': 7}}
    assert utils_mod.get_nested(config, 'a/b', sep='/') == 7
    with pytest.raises(KeyError):
        utils_mod.get_nested(config, 'a/b')


@pytest.mark.unit
def test_get_nested_raises_keyerror_for_missing_path():
    """``get_nested`` raises KeyError when any segment in the path is
    absent. Discrimination: a regression that silently returned None
    would mask user typos in config keys; verify both that the present
    sibling still resolves AND that a missing leaf and a missing root
    both raise.
    """
    config = {'a': {'b': 1}}
    assert utils_mod.get_nested(config, 'a.b') == 1
    with pytest.raises(KeyError):
        utils_mod.get_nested(config, 'a.missing')
    with pytest.raises(KeyError):
        utils_mod.get_nested(config, 'missing.b')


@pytest.mark.unit
def test_flatten_handles_nested_dict_with_dot_separator():
    """``flatten`` produces a single-level dict whose keys are the
    dot-joined paths from the original nested dict. Discrimination:
    a regression that dropped the parent_key would produce raw leaf
    keys (no 'a.b.' prefix); the top-level non-dict 'd' must survive
    untouched (a regression that nested all keys would mangle it).
    """
    config = {'a': {'b': 1, 'c': 2}, 'd': 3}
    flat = utils_mod.flatten(config)
    assert flat == {'a.b': 1, 'a.c': 2, 'd': 3}
    assert 'd' in flat and flat['d'] == 3


@pytest.mark.unit
def test_flatten_descends_through_multiple_levels():
    """``flatten`` recurses through arbitrarily deep nesting.
    Discrimination: a regression that flattened only one level would
    leave the deepest dicts intact; assert the result is fully flat
    (no dict values remain).
    """
    config = {'a': {'b': {'c': {'d': 99}}}}
    flat = utils_mod.flatten(config)
    assert flat == {'a.b.c.d': 99}
    assert all(not isinstance(v, dict) for v in flat.values())


@pytest.mark.unit
def test_flatten_honours_custom_separator():
    """``flatten`` uses the ``sep`` argument for path joins. With sep='/'
    the keys must use '/' (e.g. 'a/b'), not '.'. A regression that
    hardcoded '.' would fail this discrimination.
    """
    config = {'a': {'b': 1}}
    flat = utils_mod.flatten(config, sep='/')
    assert flat == {'a/b': 1}
    # Discrimination: the same input under default sep gives a different key
    flat_default = utils_mod.flatten(config)
    assert flat_default == {'a.b': 1}


@pytest.mark.unit
def test_flatten_returns_empty_dict_for_empty_input():
    """``flatten({})`` returns ``{}`` (the identity on empty dicts).
    Discrimination: a regression that returned None or raised would
    fail; the result must be a real dict (not a falsy stand-in like
    None or an empty list).
    """
    result = utils_mod.flatten({})
    assert result == {}
    assert isinstance(result, dict)
