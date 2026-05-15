"""
Unit tests for Bayesian optimization core utilities.

References:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest
import torch

import proteus.inference.BO as bo_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


class _DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyGP:
    def __init__(self):
        self.likelihood = object()

    def posterior(self, xs):
        class _Posterior:
            mean = torch.zeros(xs.shape[0], dtype=torch.double)
            variance = torch.ones(xs.shape[0], dtype=torch.double) * 0.1

        return _Posterior()


@pytest.mark.unit
def test_unit_bounds_returns_hypercube_tensor():
    bounds = bo_mod.unit_bounds(3)
    assert tuple(bounds.shape) == (2, 3)
    assert bounds[0].tolist() == [0.0, 0.0, 0.0]
    assert bounds[1].tolist() == [1.0, 1.0, 1.0]


@pytest.mark.unit
def test_bo_step_with_x_in_skips_gp_fitting():
    D = {
        'X': torch.tensor([[0.1]], dtype=torch.double),
        'Y': torch.tensor([[0.2]], dtype=torch.double),
    }
    B = {}

    x, y, *_rest, dist = bo_mod.BO_step(
        D=D,
        B=B,
        f=lambda _x: torch.tensor([[0.7]], dtype=torch.double),
        k=object(),
        acqf='LogEI',
        lock=_DummyLock(),
        worker_id=0,
        x_in=torch.tensor([[0.4]], dtype=torch.double),
    )

    assert x[0, 0].item() == pytest.approx(0.4)
    assert y[0, 0].item() == pytest.approx(0.7)
    assert dist is None
    assert B[0][0, 0].item() == pytest.approx(0.4)


@pytest.mark.unit
def test_bo_step_raises_for_unknown_acquisition(monkeypatch):
    monkeypatch.setattr(bo_mod, 'SingleTaskGP', lambda **kwargs: _DummyGP())
    monkeypatch.setattr(bo_mod, 'ExactMarginalLogLikelihood', lambda _lik, _gp: object())
    monkeypatch.setattr(bo_mod, 'fit_gpytorch_mll', lambda *args, **kwargs: None)

    D = {
        'X': torch.tensor([[0.1]], dtype=torch.double),
        'Y': torch.tensor([[0.2]], dtype=torch.double),
    }
    B = {
        0: torch.tensor([[0.15]], dtype=torch.double),
        1: torch.tensor([[0.20]], dtype=torch.double),
    }

    with pytest.raises(ValueError, match='Unknown acquisition function'):
        bo_mod.BO_step(
            D=D,
            B=B,
            f=lambda _x: torch.tensor([[0.0]], dtype=torch.double),
            k=object(),
            acqf='BAD-ACQF',
            lock=_DummyLock(),
            worker_id=0,
        )


@pytest.mark.unit
def test_bo_step_ucb_path_computes_distance(monkeypatch):
    monkeypatch.setattr(bo_mod, 'SingleTaskGP', lambda **kwargs: _DummyGP())
    monkeypatch.setattr(bo_mod, 'ExactMarginalLogLikelihood', lambda _lik, _gp: object())
    monkeypatch.setattr(bo_mod, 'fit_gpytorch_mll', lambda *args, **kwargs: None)
    monkeypatch.setattr(bo_mod, 'UpperConfidenceBound', lambda _gp, beta: object())
    monkeypatch.setattr(
        bo_mod,
        'optimize_acqf',
        lambda **kwargs: (torch.tensor([[0.8]], dtype=torch.double), None),
    )
    monkeypatch.setattr(bo_mod, 'plot_iter', lambda **kwargs: None)

    D = {
        'X': torch.tensor([[0.1]], dtype=torch.double),
        'Y': torch.tensor([[1.0]], dtype=torch.double),
    }
    B = {
        0: torch.tensor([[0.1]], dtype=torch.double),
        1: torch.tensor([[0.3]], dtype=torch.double),
    }

    _x, y, *_rest, dist = bo_mod.BO_step(
        D=D,
        B=B,
        f=lambda _x: torch.tensor([[0.9]], dtype=torch.double),
        k=object(),
        acqf='UCB',
        lock=_DummyLock(),
        worker_id=0,
    )

    assert y[0, 0].item() == pytest.approx(0.9)
    assert dist == pytest.approx(0.5)


@pytest.mark.unit
def test_init_locs_returns_batch_candidates(monkeypatch):
    monkeypatch.setattr(bo_mod, 'get_kernel_w_prior', lambda *args, **kwargs: object())
    monkeypatch.setattr(bo_mod, 'SingleTaskGP', lambda **kwargs: _DummyGP())
    monkeypatch.setattr(bo_mod, 'ExactMarginalLogLikelihood', lambda _lik, _gp: object())
    monkeypatch.setattr(bo_mod, 'fit_gpytorch_mll', lambda *args, **kwargs: None)
    monkeypatch.setattr(bo_mod, 'qLogExpectedImprovement', lambda _gp, best_f: object())
    monkeypatch.setattr(
        bo_mod,
        'optimize_acqf',
        lambda **kwargs: (torch.tensor([[0.2], [0.7]], dtype=torch.double), None),
    )

    D = {
        'X': torch.tensor([[0.1], [0.4]], dtype=torch.double),
        'Y': torch.tensor([[0.2], [0.6]], dtype=torch.double),
    }
    out = bo_mod.init_locs(2, D)
    assert tuple(out.shape) == (2, 1)
    assert out[0, 0].item() == pytest.approx(0.2)
    assert out[1, 0].item() == pytest.approx(0.7)


@pytest.mark.unit
def test_plot_iter_writes_figure(tmp_path):
    gp = _DummyGP()

    class _Acqf:
        def __call__(self, x):
            return torch.ones((x.shape[0], 1), dtype=torch.double)

    bo_mod.plot_iter(
        gp=gp,
        acqf=_Acqf(),
        X=torch.tensor([[0.1], [0.8]], dtype=torch.double),
        Y=torch.tensor([[0.2], [0.3]], dtype=torch.double),
        next_x=torch.tensor([[0.5]], dtype=torch.double),
        busys=torch.tensor([[0.2]], dtype=torch.double),
        dir=str(tmp_path),
        name='iter.png',
    )

    assert (tmp_path / 'iter.png').is_file()
