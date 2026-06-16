"""
Unit tests for Bayesian optimization core utilities.

Covers: ``unit_bounds``, ``get_acqf`` dispatch and error contract,
``BO_step`` with explicit x_in / UCB path, ``init_locs`` with acqf
propagation, and the ``plot_iter`` file-I/O leg.

Physics invariants exercised:
  - Acquisition-parameter contract: correct beta (UCB) and best_f
    (LogEI, LogPI) are forwarded; wrong values shift the optimisation target.
  - Error-before-evaluation: unsupported acqf raises before any objective
    call so no evaluation budget is wasted.
  - Distance metric: UCB path returns the correct minimum distance to
    busy points for the diversity-aware scheduler.

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
    """``unit_bounds(d)`` returns a (2, d) tensor whose rows are all-zeros
    and all-ones, i.e. the lower and upper corners of the d-dimensional
    unit hypercube the BO loop optimises over.
    """
    bounds = bo_mod.unit_bounds(3)
    assert tuple(bounds.shape) == (2, 3)
    assert bounds[0].tolist() == [0.0, 0.0, 0.0]
    assert bounds[1].tolist() == [1.0, 1.0, 1.0]


@pytest.mark.unit
def test_bo_step_with_x_in_skips_gp_fitting():
    """When ``BO_step`` is called with an explicit ``x_in`` (an
    externally-suggested candidate), it skips the GP fit + acquisition
    optimisation, evaluates ``f(x_in)`` directly, and registers the
    candidate in the busy dict ``B``.
    """
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
    """An unsupported acquisition function name raises ValueError with
    'Unsupported acquisition function' rather than silently dispatching to
    a default.
    """
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

    f_calls: list = []

    def _f(_x):
        f_calls.append(_x)
        return torch.tensor([[0.0]], dtype=torch.double)

    with pytest.raises(ValueError, match='Unsupported acquisition function'):
        bo_mod.BO_step(
            D=D,
            B=B,
            f=_f,
            k=object(),
            acqf='BAD-ACQF',
            lock=_DummyLock(),
            worker_id=0,
        )
    # Discrimination: the guard must fire BEFORE the expensive objective
    # call. A regression that evaluated `f` first and then raised would
    # waste a costly simulator call per misconfigured worker.
    assert f_calls == []


@pytest.mark.unit
def test_bo_step_ucb_path_computes_distance(monkeypatch):
    """The UCB acquisition path returns the proposed candidate, the
    evaluated ``y`` at that candidate, and the minimum distance to any
    other worker's busy point (used by the diversity-aware scheduler).
    """
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
    """``init_locs(n, D)`` returns an n*d tensor of initial candidate
    locations for the n workers; the returned rows are exactly the
    output of ``optimize_acqf`` over the LogExpectedImprovement
    acquisition.
    """
    monkeypatch.setattr(bo_mod, 'get_kernel_w_prior', lambda *args, **kwargs: object())
    monkeypatch.setattr(bo_mod, 'SingleTaskGP', lambda **kwargs: _DummyGP())
    monkeypatch.setattr(bo_mod, 'ExactMarginalLogLikelihood', lambda _lik, _gp: object())
    monkeypatch.setattr(bo_mod, 'fit_gpytorch_mll', lambda *args, **kwargs: None)
    monkeypatch.setattr(bo_mod, 'LogExpectedImprovement', lambda _gp, best_f: object())
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
    """``plot_iter`` writes the BO-iteration diagnostic figure to disk
    under the given directory and filename. Verifies the file-IO leg of
    the diagnostic plotting pipeline.
    """
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
    # Discrimination: a regression that wrote an empty file (matplotlib
    # closed before save) would still satisfy `is_file()`. Pin a non-zero
    # file size to catch that mode.
    assert (tmp_path / 'iter.png').stat().st_size > 0


# ---------------------------------------------------------------------------
# get_acqf factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_get_acqf_ucb_uses_beta_2(monkeypatch):
    """get_acqf('UCB') instantiates UpperConfidenceBound with beta=2.0.

    beta controls the exploration-exploitation trade-off; the configured
    default is 2.0. A wrong beta shifts candidate ranking away from the
    intended strategy and is not caught by any downstream assertion.
    """
    calls: list = []

    def _mock_ucb(gp, beta):
        calls.append({'gp': gp, 'beta': beta})
        return object()

    monkeypatch.setattr(bo_mod, 'UpperConfidenceBound', _mock_ucb)
    mock_gp = object()
    result = bo_mod.get_acqf('UCB', mock_gp, best=0.5)

    assert len(calls) == 1
    assert calls[0]['gp'] is mock_gp
    assert calls[0]['beta'] == pytest.approx(2.0)
    # Discrimination: beta=1.0 (pessimistic) and beta=0.0 (pure exploit)
    # are both valid but would produce different candidate rankings.
    assert calls[0]['beta'] != pytest.approx(1.0)
    assert calls[0]['beta'] != pytest.approx(0.0)
    assert result is not None


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_get_acqf_log_ei_forwards_best_f(monkeypatch):
    """get_acqf('LogEI') instantiates LogExpectedImprovement with the correct best_f.

    LogEI measures improvement relative to the current best observation;
    using the wrong best_f would bias candidate selection toward already-explored
    regions or miss genuine improvements.
    """
    calls: list = []

    def _mock_lei(gp, best_f):
        calls.append({'gp': gp, 'best_f': best_f})
        return object()

    monkeypatch.setattr(bo_mod, 'LogExpectedImprovement', _mock_lei)
    mock_gp = object()
    best = 0.73  # non-trivial value; T=0 or T=1 both collapse exponent differences
    result = bo_mod.get_acqf('LogEI', mock_gp, best=best)

    assert len(calls) == 1
    assert calls[0]['gp'] is mock_gp
    assert calls[0]['best_f'] == pytest.approx(best)
    # Discrimination: best_f=0.0 would produce different improvement thresholds.
    assert abs(calls[0]['best_f'] - 0.0) > 0.1
    assert result is not None


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_get_acqf_log_pi_forwards_best_f(monkeypatch):
    """get_acqf('LogPI') instantiates LogProbabilityOfImprovement with the correct best_f.

    LogPI is the log-probability of strictly exceeding the current best; like
    LogEI it is anchored at best_f. Verifies the new LogPI branch is wired
    to the right constructor with the right threshold argument.
    """
    calls: list = []

    def _mock_lpi(gp, best_f):
        calls.append({'gp': gp, 'best_f': best_f})
        return object()

    monkeypatch.setattr(bo_mod, 'LogProbabilityOfImprovement', _mock_lpi)
    mock_gp = object()
    best = 0.61  # non-trivial value well away from 0 and 1
    result = bo_mod.get_acqf('LogPI', mock_gp, best=best)

    assert len(calls) == 1
    assert calls[0]['gp'] is mock_gp
    assert calls[0]['best_f'] == pytest.approx(best)
    # Discrimination: best_f=0.0 collapses to a trivial improvement threshold.
    assert abs(calls[0]['best_f'] - 0.0) > 0.1
    assert result is not None


@pytest.mark.unit
def test_get_acqf_raises_for_unsupported_name_before_any_gp_call(monkeypatch):
    """get_acqf raises ValueError for names outside {'UCB', 'LogEI', 'LogPI'}.

    The error must fire before any GP or acquisition constructor is called
    so misconfigured runs fail immediately rather than after the first GP fit.
    Exercises both a formerly-supported name ('E-LogEI') and a blank string.
    """
    gp_calls: list = []

    def _track_gp(*args, **kwargs):
        gp_calls.append(True)
        return object()

    monkeypatch.setattr(bo_mod, 'UpperConfidenceBound', _track_gp)
    monkeypatch.setattr(bo_mod, 'LogExpectedImprovement', _track_gp)
    monkeypatch.setattr(bo_mod, 'LogProbabilityOfImprovement', _track_gp)
    mock_gp = object()

    with pytest.raises(ValueError, match='Unsupported acquisition function'):
        bo_mod.get_acqf('E-LogEI', mock_gp, best=0.5)  # removed in this branch

    with pytest.raises(ValueError, match='Unsupported acquisition function'):
        bo_mod.get_acqf('', mock_gp, best=0.5)

    # Discrimination: a regression where get_acqf always raises would still
    # satisfy the two assertions above. Verify each supported name succeeds.
    assert gp_calls == [], 'Acquisition constructors must not be called on unsupported name'
    for name in ('UCB', 'LogEI', 'LogPI'):
        result = bo_mod.get_acqf(name, mock_gp, best=0.5)
        assert result is not None, (
            f'Supported name {name!r} unexpectedly raised or returned None'
        )


# ---------------------------------------------------------------------------
# init_locs acqf propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_init_locs_propagates_acqf_to_log_pi(monkeypatch):
    """init_locs with acqf='LogPI' routes through LogProbabilityOfImprovement.

    The initial candidate locations must use the same acquisition as the main
    BO loop; a mismatch would bias early exploration differently from
    steady-state sampling, producing an inconsistent search trajectory.
    """
    lpi_calls: list = []
    lei_calls: list = []

    def _mock_lpi(gp, best_f):
        lpi_calls.append(best_f)
        return object()

    def _mock_lei(gp, best_f):
        lei_calls.append(best_f)
        return object()

    monkeypatch.setattr(bo_mod, 'get_kernel_w_prior', lambda *a, **kw: object())
    monkeypatch.setattr(bo_mod, 'SingleTaskGP', lambda **kw: _DummyGP())
    monkeypatch.setattr(bo_mod, 'ExactMarginalLogLikelihood', lambda _l, _g: object())
    monkeypatch.setattr(bo_mod, 'fit_gpytorch_mll', lambda *a, **kw: None)
    monkeypatch.setattr(bo_mod, 'LogProbabilityOfImprovement', _mock_lpi)
    monkeypatch.setattr(bo_mod, 'LogExpectedImprovement', _mock_lei)
    monkeypatch.setattr(
        bo_mod,
        'optimize_acqf',
        lambda **kw: (torch.tensor([[0.3], [0.6]], dtype=torch.double), None),
    )

    D = {
        'X': torch.tensor([[0.1], [0.4]], dtype=torch.double),
        'Y': torch.tensor([[0.2], [0.6]], dtype=torch.double),
    }
    out = bo_mod.init_locs(2, D, acqf='LogPI')

    assert len(lpi_calls) == 1, 'LogProbabilityOfImprovement was not called for acqf=LogPI'
    assert len(lei_calls) == 0, 'LogExpectedImprovement must not be called when acqf=LogPI'
    # Returned shape: (n_workers, d)
    assert tuple(out.shape) == (2, 1)
    assert out[0, 0].item() == pytest.approx(0.3)
    assert out[1, 0].item() == pytest.approx(0.6)
