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
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
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
    num_outputs = 1  # required by AnalyticAcquisitionFunction.__init__

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
    monkeypatch.setattr(bo_mod, 'get_acqf', lambda *args, **kwargs: object())
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
    """init_locs(n, D) returns an (n, d) tensor for n workers by calling
    optimize_acqf once per worker with q=1 and stacking the results.

    Analytic acquisition functions require q=1; each candidate is
    optimised independently.
    """
    monkeypatch.setattr(bo_mod, 'get_kernel', lambda *args, **kwargs: object())
    monkeypatch.setattr(bo_mod, 'SingleTaskGP', lambda **kwargs: _DummyGP())
    monkeypatch.setattr(bo_mod, 'ExactMarginalLogLikelihood', lambda _lik, _gp: object())
    monkeypatch.setattr(bo_mod, 'fit_gpytorch_mll', lambda *args, **kwargs: None)
    monkeypatch.setattr(bo_mod, 'get_acqf', lambda *args, **kwargs: object())

    q_values: list = []
    results = [
        torch.tensor([[0.2]], dtype=torch.double),
        torch.tensor([[0.7]], dtype=torch.double),
    ]
    call_idx = [0]

    def _mock_optimize(**kwargs):
        q_values.append(kwargs.get('q'))
        result = results[call_idx[0]]
        call_idx[0] += 1
        return (result, None)

    monkeypatch.setattr(bo_mod, 'optimize_acqf', _mock_optimize)

    D = {
        'X': torch.tensor([[0.1], [0.4]], dtype=torch.double),
        'Y': torch.tensor([[0.2], [0.6]], dtype=torch.double),
    }
    out = bo_mod.init_locs(2, D)

    assert len(q_values) == 2, 'Expected one optimize_acqf call per worker'
    assert all(q == 1 for q in q_values), 'Each call must use q=1 (analytic acqf constraint)'
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

# get_acqf lives in utils.py and uses lazy per-branch imports, so we test
# by inspecting the returned object type and its stored buffers rather than
# patching import targets.


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_get_acqf_ucb_uses_beta_2():
    """get_acqf('UCB') returns an UpperConfidenceBound with beta=2.0.

    beta controls the exploration-exploitation trade-off; the configured
    default is 2.0. A wrong beta shifts candidate ranking away from the
    intended strategy and is not caught by any downstream assertion.
    """
    from botorch.acquisition.analytic import UpperConfidenceBound

    result = bo_mod.get_acqf('UCB', _DummyGP(), best=0.5)

    assert isinstance(result, UpperConfidenceBound)
    # beta is stored as a tensor buffer; use .item() for scalar comparison.
    assert result.beta.item() == pytest.approx(2.0)
    # Discrimination: beta=1.0 (pessimistic) and beta=0.0 (pure exploit)
    # are both valid but would produce different candidate rankings.
    assert result.beta.item() != pytest.approx(1.0)
    assert result.beta.item() != pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_get_acqf_log_ei_forwards_best_f():
    """get_acqf('LogEI') returns a LogExpectedImprovement anchored at best_f.

    LogEI measures improvement relative to the current best observation;
    a wrong best_f would bias candidate selection toward already-explored
    regions or miss genuine improvements.
    """
    from botorch.acquisition.analytic import LogExpectedImprovement

    best = 0.73  # non-trivial: T=0 or T=1 collapse exponent differences
    result = bo_mod.get_acqf('LogEI', _DummyGP(), best=best)

    assert isinstance(result, LogExpectedImprovement)
    assert result.best_f.item() == pytest.approx(best)
    # Discrimination: best_f=0.0 would produce different improvement thresholds.
    assert abs(result.best_f.item()) > 0.1


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_get_acqf_log_pi_forwards_best_f():
    """get_acqf('LogPI') returns a LogProbabilityOfImprovement anchored at best_f.

    LogPI is the log-probability of strictly exceeding the current best; like
    LogEI it is anchored at best_f. Verifies the new LogPI branch is wired
    to the right constructor with the right threshold argument.
    """
    from botorch.acquisition.analytic import LogProbabilityOfImprovement

    best = 0.61  # non-trivial value well away from 0 and 1
    result = bo_mod.get_acqf('LogPI', _DummyGP(), best=best)

    assert isinstance(result, LogProbabilityOfImprovement)
    assert result.best_f.item() == pytest.approx(best)
    # Discrimination: best_f=0.0 collapses to a trivial improvement threshold.
    assert abs(result.best_f.item()) > 0.1


@pytest.mark.unit
def test_get_acqf_raises_for_unsupported_name():
    """get_acqf raises ValueError for names outside {'UCB', 'LogEI', 'LogPI'}.

    The ValueError must include the offending name in the message so that a
    mis-configured infer.toml surfaces a clear diagnostic rather than a
    cryptic traceback.
    """
    with pytest.raises(ValueError, match='Unsupported acquisition function: E-LogEI'):
        bo_mod.get_acqf('E-LogEI', object(), best=0.5)  # removed in this branch

    with pytest.raises(ValueError, match='Unsupported acquisition function'):
        bo_mod.get_acqf('', object(), best=0.5)

    # Discrimination: a regression that always raises would still satisfy the
    # two assertions above. Verify each supported name succeeds without raising.
    for name in ('UCB', 'LogEI', 'LogPI'):
        result = bo_mod.get_acqf(name, _DummyGP(), best=0.5)
        assert result is not None, f'Supported name {name!r} unexpectedly returned None'


# ---------------------------------------------------------------------------
# init_locs acqf propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_init_locs_propagates_acqf_to_log_pi(monkeypatch):
    """init_locs with acqf='LogPI' passes 'LogPI' to get_acqf for every worker slot.

    The initial candidate locations must use the same acquisition as the main
    BO loop; a mismatch would bias early exploration differently from
    steady-state sampling, producing an inconsistent search trajectory.
    With 2 workers the loop calls get_acqf twice, once per candidate.
    """
    acqf_names: list = []

    def _mock_get_acqf(name, gp, best):
        acqf_names.append(name)
        return object()

    monkeypatch.setattr(bo_mod, 'get_kernel', lambda *a, **kw: object())
    monkeypatch.setattr(bo_mod, 'SingleTaskGP', lambda **kw: _DummyGP())
    monkeypatch.setattr(bo_mod, 'ExactMarginalLogLikelihood', lambda _l, _g: object())
    monkeypatch.setattr(bo_mod, 'fit_gpytorch_mll', lambda *a, **kw: None)
    monkeypatch.setattr(bo_mod, 'get_acqf', _mock_get_acqf)

    results = [
        torch.tensor([[0.3]], dtype=torch.double),
        torch.tensor([[0.6]], dtype=torch.double),
    ]
    call_idx = [0]

    def _mock_optimize(**kw):
        result = results[call_idx[0]]
        call_idx[0] += 1
        return (result, None)

    monkeypatch.setattr(bo_mod, 'optimize_acqf', _mock_optimize)

    D = {
        'X': torch.tensor([[0.1], [0.4]], dtype=torch.double),
        'Y': torch.tensor([[0.2], [0.6]], dtype=torch.double),
    }
    out = bo_mod.init_locs(2, D, acqf='LogPI')

    # The loop calls get_acqf once per worker: 2 workers -> 2 calls.
    assert acqf_names == ['LogPI', 'LogPI'], (
        'get_acqf must be called with LogPI for every worker slot'
    )
    # Returned shape: (n_workers, d)
    assert tuple(out.shape) == (2, 1)
    assert out[0, 0].item() == pytest.approx(0.3)
    assert out[1, 0].item() == pytest.approx(0.6)
