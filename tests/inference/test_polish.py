"""
Unit tests for ``src/proteus/inference/polish.py``.

PROTEUS runs are replaced by an analytic chi-squared, so the search, its budget and the
island choice are checked without a simulation.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')

import torch  # noqa: E402

import proteus.inference.polish as pol  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

MINIMUM = np.array([0.62, 0.41])


def objective(x):
    """-log10 of a chi-squared with its minimum of 1 at MINIMUM."""
    return -np.log10(1.0 + 100.0 * np.sum((np.asarray(x) - MINIMUM) ** 2, axis=-1))


@pytest.fixture
def fake_proteus(monkeypatch):
    """Replace prot_builder by `objective`; returns the (worker, iter) of each run."""
    calls = []

    def builder(worker, iter, **kw):
        calls.append((worker, iter))
        return lambda x: torch.tensor([[objective(x.numpy()[0])]], dtype=torch.double)

    monkeypatch.setattr(pol, 'prot_builder', builder)
    monkeypatch.setattr(pol, 'close_worker_runner', lambda: None)
    return calls


@pytest.mark.parametrize('n', [None, 0, 25])
def test_validate_polish_runs_accepts(n):
    """None and 0 turn the polish off; a positive count passes unchanged."""
    out = pol.validate_polish_runs(n, {'R_obs': 0.1})
    assert out == (n or 0)
    assert type(out) is int


@pytest.mark.parametrize('n', [-1, 2.5, True, '25'])
def test_validate_polish_runs_rejects(n):
    """Negative, fractional, boolean and string counts are refused."""
    with pytest.raises(ValueError, match='non-negative integer'):
        pol.validate_polish_runs(n, {'R_obs': 0.1})
    with pytest.raises(ValueError, match='non-negative integer'):
        pol.validate_polish_runs(n, None)


def test_validate_polish_runs_needs_sigma():
    """Islands are chosen by chi-squared, so a positive count needs [sigma]."""
    assert pol.validate_polish_runs(0, None) == 0
    with pytest.raises(ValueError, match='sigma'):
        pol.validate_polish_runs(25, None)


def test_find_islands_thresholds_and_separation():
    """Islands are within DELTA_CHI2 of the best fit and ISLAND_SEPARATION apart, best first."""
    X = np.array([[0.1, 0.1], [0.12, 0.1], [0.5, 0.5], [0.9, 0.9], [0.9, 0.1]])
    chi2 = np.array([2.0, 1.5, 8.0, 11.9, 12.1])
    # Index 0 is within 0.15 of the best (1); index 4 is past best + DELTA_CHI2 = 11.5
    assert pol.find_islands(X, -np.log10(chi2)) == [1, 2]
    assert pol.find_islands(X[:2], -np.log10(chi2[:2])) == [1]  # 0.02 apart: one island


def test_find_islands_absolute_limit_and_count():
    """Nothing above MAX_CHI2 is an island, and at most MAX_ISLANDS are chosen."""
    X = np.eye(5)
    assert pol.find_islands(X, -np.log10(np.array([16.0, 17.0, 20, 30, 40]))) == []
    assert len(pol.find_islands(X, -np.log10(np.ones(5)))) == pol.MAX_ISLANDS


def test_polish_island_improves_within_budget(fake_proteus):
    """The search gets close to the minimum of a quadratic chi-squared within 25 runs."""
    x0 = np.array([0.5, 0.5])
    runs = pol.polish_island(1, x0, float(objective(x0)), 25, {})
    assert len(runs) == len(fake_proteus) <= 25
    assert {w for w, _ in fake_proteus} == {-3}
    assert [i for _, i in fake_proteus] == list(range(len(runs)))
    assert not any(np.array_equal(x, x0) for x, _ in runs)  # the start is not rerun
    assert all(((x >= 0) & (x <= 1)).all() for x, _ in runs)
    chi2_start, chi2_best = 10 ** -objective(x0), 10 ** -max(y for _, y in runs)
    # Start at chi2 3.25; a search that only shrinks the simplex would stay above 2
    assert chi2_best < 1.3 < chi2_start


def test_polish_island_steps_inward_at_a_bound(fake_proteus):
    """A start in a corner of the cube keeps every run inside it and spends the budget."""
    x0 = np.array([1.0, 0.0])
    runs = pol.polish_island(0, x0, float(objective(x0)), 3, {})
    assert len(runs) == 3
    assert all(((x >= 0) & (x <= 1)).all() for x, _ in runs)


def test_polish_writes_runs_and_extends_data(fake_proteus, monkeypatch, tmp_path):
    """Every island run goes to polish.csv and is appended to the data, with its objective."""

    class SerialPool:
        def __init__(self, n):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def starmap(self, f, args):
            return list(itertools.starmap(f, args))

    monkeypatch.setattr(pol, 'Pool', SerialPool)
    X = torch.tensor([[0.5, 0.5], [0.8, 0.6], [0.95, 0.05]], dtype=torch.double)
    Y = torch.tensor(objective(X.numpy())[:, None])
    # chi2 3.25, 7.85 and 24.9: the third is past best + DELTA_CHI2
    D = pol.polish({'X': X, 'Y': Y}, 10, {}, str(tmp_path))

    P = pd.read_csv(tmp_path / 'polish.csv')
    assert list(P.columns) == ['island', 'x_0', 'x_1', 'y']
    assert sorted(P.island.unique()) == [0, 1]
    assert len(P) == len(fake_proteus) <= 20
    assert len(D['X']) == len(D['Y']) == 3 + len(P)
    np.testing.assert_allclose(D['Y'][3:, 0].numpy(), P.y)
    np.testing.assert_allclose(D['Y'][3:, 0].numpy(), objective(D['X'][3:].numpy()))


def test_polish_without_islands_returns_data(fake_proteus, tmp_path):
    """Without a fit below MAX_CHI2 no run starts and no file is written."""
    D = {'X': torch.rand(4, 2, dtype=torch.double), 'Y': torch.full((4, 1), -2.0)}
    assert pol.polish(D, 10, {}, str(tmp_path)) is D
    assert not fake_proteus and not (tmp_path / 'polish.csv').exists()
