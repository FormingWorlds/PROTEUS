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

import warnings

import pytest

# The Bayesian-optimisation stack ships as the optional `inference` extra,
# which installs all three together. Guarding the whole stack keeps a
# partial environment skipping rather than failing collection.
torch = pytest.importorskip('torch')
pytest.importorskip('botorch')
pytest.importorskip('gpytorch')

import proteus.inference.BO as bo_mod  # noqa: E402

from ._bo_helpers import make_quadratic_objective  # noqa: E402

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
    pending = []
    monkeypatch.setattr(
        bo_mod, 'get_acqf', lambda *args, X_pending=None: pending.append(X_pending)
    )
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
    # Only worker 1's point is pending; worker 0's own entry is excluded
    assert pending[0].tolist() == [[0.3]]


@pytest.mark.unit
def test_init_locs_returns_batch_candidates(monkeypatch):
    """init_locs(n, D) returns an (n, d) tensor for n workers by calling
    optimize_acqf once per worker with q=1 and stacking the results.

    Each call gets the candidates chosen before it as pending points.
    """
    monkeypatch.setattr(bo_mod, 'get_kernel', lambda *args, **kwargs: object())
    monkeypatch.setattr(bo_mod, 'SingleTaskGP', lambda **kwargs: _DummyGP())
    monkeypatch.setattr(bo_mod, 'ExactMarginalLogLikelihood', lambda _lik, _gp: object())
    monkeypatch.setattr(bo_mod, 'fit_gpytorch_mll', lambda *args, **kwargs: None)
    pending = []
    monkeypatch.setattr(
        bo_mod, 'get_acqf', lambda *args, X_pending=None: pending.append(X_pending)
    )

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
    assert pending[0] is None
    assert pending[1].tolist() == [[0.2]]


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

    def _mock_get_acqf(name, gp, best, X_pending=None):
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


# ---------------------------------------------------------------------------
# Synthetic objective shared with the slow convergence tier
# ---------------------------------------------------------------------------
def test_quadratic_objective_returns_zero_at_target():
    """The synthetic objective evaluates to exactly 0 at its target point.

    The slow-tier convergence tests optimize against this objective, so a bug
    in the helper would make them pass while testing nothing. Pinning it at
    unit tier keeps the helper honest without waiting for the nightly.
    """
    target = torch.tensor([0.3, 0.7], dtype=torch.double)
    objective = make_quadratic_objective(target)
    y_at_target = objective(target.unsqueeze(0))
    assert y_at_target.item() == pytest.approx(0.0, abs=1e-12)
    # Off-target value is strictly negative
    y_off = objective(torch.tensor([[0.0, 0.0]], dtype=torch.double))
    assert y_off.item() < 0
    # Quadratic: doubling the distance quadruples the magnitude
    y_far = objective(torch.tensor([[0.6, 0.4]], dtype=torch.double))
    y_near = objective(torch.tensor([[0.45, 0.55]], dtype=torch.double))
    # ratio of (far-target)^2 to (near-target)^2 = ((0.3,0.3))^2 / ((0.15,0.15))^2 = 4
    assert y_near.item() / y_far.item() == pytest.approx(0.25, rel=1e-9)


# ============================================================================
# Busy-point bookkeeping when workers come and go
# ============================================================================


def _patched_bo_step_deps(monkeypatch, candidate=0.8):
    """Replace the GP fit and acquisition optimisation with fixed stand-ins.

    Leaves the busy-point handling under test as the only live logic.
    """
    monkeypatch.setattr(bo_mod, 'SingleTaskGP', lambda **kwargs: _DummyGP())
    monkeypatch.setattr(bo_mod, 'ExactMarginalLogLikelihood', lambda _lik, _gp: object())
    monkeypatch.setattr(bo_mod, 'fit_gpytorch_mll', lambda *args, **kwargs: None)
    monkeypatch.setattr(bo_mod, 'get_acqf', lambda *args, **kwargs: object())
    monkeypatch.setattr(
        bo_mod,
        'optimize_acqf',
        lambda **kwargs: (torch.tensor([[candidate]], dtype=torch.double), None),
    )
    monkeypatch.setattr(bo_mod, 'plot_iter', lambda **kwargs: None)


@pytest.mark.unit
def test_bo_step_identifies_busy_points_by_worker_id_not_position(monkeypatch):
    """Busy points are matched to their owner by worker id. Once a worker has
    stopped and released its claim, the remaining entries no longer sit at the
    position their worker id implies, so a positional lookup reads another
    worker's point as its own.
    """
    _patched_bo_step_deps(monkeypatch, candidate=0.8)

    D = {
        'X': torch.tensor([[0.1]], dtype=torch.double),
        'Y': torch.tensor([[1.0]], dtype=torch.double),
    }
    # Worker 1 has stopped and released its point; worker 2 is the caller, so
    # its own claim is excluded. Two other points keep the nearest distinct
    # from the furthest.
    B = {
        0: torch.tensor([[0.1]], dtype=torch.double),
        2: torch.tensor([[0.75]], dtype=torch.double),
        3: torch.tensor([[0.79]], dtype=torch.double),
    }

    _x, y, *_rest, dist = bo_mod.BO_step(
        D=D,
        B=B,
        f=lambda _x: torch.tensor([[0.9]], dtype=torch.double),
        k=object(),
        acqf='UCB',
        lock=_DummyLock(),
        worker_id=2,
    )

    assert y[0, 0].item() == pytest.approx(0.9)
    # Nearest other claim is worker 3 at 0.79, from the candidate at 0.8.
    assert dist == pytest.approx(0.01)
    # Nearest, not furthest: worker 0 sits at 0.1, giving 0.7. A regression to
    # torch.max would report that instead.
    assert abs(dist - 0.7) > 0.5
    # The caller's own claim at 0.75 is excluded. Including it would give
    # 0.05, which is neither of the two values above.
    assert abs(dist - 0.05) > 0.02


@pytest.mark.unit
def test_bo_step_reports_no_distance_when_no_other_worker_is_busy(monkeypatch):
    """With no other worker running, there is no nearest busy point and the
    distance is undefined rather than zero. This is the steady state of a
    single-worker study and the tail of every multi-worker one.
    """
    _patched_bo_step_deps(monkeypatch, candidate=0.4)

    D = {
        'X': torch.tensor([[0.1]], dtype=torch.double),
        'Y': torch.tensor([[1.0]], dtype=torch.double),
    }
    B = {0: torch.tensor([[0.2]], dtype=torch.double)}

    x, y, *_rest, dist = bo_mod.BO_step(
        D=D,
        B=B,
        f=lambda _x: torch.tensor([[0.6]], dtype=torch.double),
        k=object(),
        acqf='UCB',
        lock=_DummyLock(),
        worker_id=0,
    )

    # The step still completes and proposes its candidate.
    assert x[0, 0].item() == pytest.approx(0.4)
    assert y[0, 0].item() == pytest.approx(0.6)
    # Undefined, not zero: a zero would read as another worker sitting exactly
    # on this candidate and would suppress the diversity term.
    assert dist is None

    # Edge case: an entirely empty busy map behaves the same way.
    _x2, _y2, *_rest2, dist2 = bo_mod.BO_step(
        D=D,
        B={},
        f=lambda _x: torch.tensor([[0.6]], dtype=torch.double),
        k=object(),
        acqf='UCB',
        lock=_DummyLock(),
        worker_id=0,
    )
    assert dist2 is None


def _two_peak_data():
    """1D data around two unsampled maxima, at x = 0.2 (higher) and x = 0.8."""
    X = torch.tensor([[0.0], [0.1], [0.3], [0.5], [0.7], [0.9], [1.0]], dtype=torch.double)
    Y = torch.exp(-(((X - 0.2) / 0.1) ** 2)) + 0.8 * torch.exp(-(((X - 0.8) / 0.1) ** 2))
    return {'X': X, 'Y': Y}


def _fitted_gp(D):
    """GP on `D` with fixed hyperparameters, so the test needs no fit."""
    from botorch.models import SingleTaskGP

    gp = SingleTaskGP(D['X'], D['Y'], train_Yvar=torch.full_like(D['Y'], 1e-4))
    gp.covar_module.lengthscale = 0.1  # the width of the peaks in _two_peak_data
    return gp.eval()


def _propose(gp, D, X_pending, name='LogEI'):
    from botorch.optim import optimize_acqf

    torch.manual_seed(0)  # fixes the Monte Carlo base samples of the batch versions
    acqf = bo_mod.get_acqf(name, gp, D['Y'].max().item(), X_pending=X_pending)
    x, _ = optimize_acqf(acqf, bo_mod.unit_bounds(1), q=1, num_restarts=2, raw_samples=64)
    return x


@pytest.mark.unit
def test_get_acqf_with_pending_uses_monte_carlo_versions():
    """With pending points, LogEI, UCB and LogPI become qLogEI, qUCB and qPI
    holding those points, with best_f carried over where it applies."""
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.acquisition.monte_carlo import (
        qProbabilityOfImprovement,
        qUpperConfidenceBound,
    )

    D = _two_peak_data()
    gp = _fitted_gp(D)
    pending = torch.tensor([[0.2]], dtype=torch.double)

    ei = bo_mod.get_acqf('LogEI', gp, best=0.73, X_pending=pending)
    ucb = bo_mod.get_acqf('UCB', gp, best=0.73, X_pending=pending)
    pi = bo_mod.get_acqf('LogPI', gp, best=0.73, X_pending=pending)

    assert isinstance(ei, qLogExpectedImprovement)
    assert ei.best_f.item() == pytest.approx(0.73)
    assert isinstance(ucb, qUpperConfidenceBound)
    assert isinstance(pi, qProbabilityOfImprovement)
    assert pi.best_f.item() == pytest.approx(0.73)
    for acqf in (ei, ucb, pi):
        assert acqf.X_pending.tolist() == [[0.2]]


@pytest.mark.unit
# PI peaks beside the best observation at x = 0.1, EI further into the peak.
@pytest.mark.parametrize(('name', 'x_first'), [('LogEI', 0.2), ('LogPI', 0.12)])
def test_pending_point_moves_the_proposal_away(name, x_first):
    """The acquisition proposes the same point again while it is busy unless
    that point is pending; with it pending, the proposal moves to the other
    maximum.

    Without pending points, two workers would both propose a point on the
    higher peak at 0.2.
    """
    D = _two_peak_data()
    gp = _fitted_gp(D)

    first = _propose(gp, D, X_pending=None, name=name)
    again = _propose(gp, D, X_pending=None, name=name)
    second = _propose(gp, D, X_pending=first, name=name)

    assert first.item() == pytest.approx(x_first, abs=0.05)
    assert again.item() == pytest.approx(first.item(), abs=1e-3), 'without pending: a duplicate'
    assert second.item() == pytest.approx(0.8, abs=0.05), 'with pending: the other maximum'


# botorch 0.18.1 warning texts, as they appear in an inference log
_FIRST_TRY = (
    'Optimization failed in `gen_candidates_scipy` with the following warning(s):\n'
    "[OptimizationWarning('Optimization failed within `scipy.optimize.minimize` with "
    "status 2 and message ABNORMAL: .'), OptimizationWarning('Optimization failed within "
    "`scipy.optimize.minimize` with status 2 and message ABNORMAL: .')]\n"
    'Trying again with a new set of initial conditions.'
)
_SECOND_TRY = (
    'Optimization failed on the second try, after generating a new set of initial conditions.'
)


def _patch_optimizer(monkeypatch, *messages):
    """Make optimize_acqf emit `messages` as (text, category) warnings and return x = [[0.25, 0.75]]."""

    def _optimize(**kwargs):
        for text, category in messages:
            warnings.warn(text, category)
        return torch.tensor([[0.25, 0.75]], dtype=torch.double), None

    monkeypatch.setattr(bo_mod, 'optimize_acqf', _optimize)


@pytest.mark.unit
def test_optimize_acqf_logged_folds_both_retry_warnings_into_one_warning_line(
    monkeypatch, caplog, recwarn
):
    """Both botorch retry warnings become one WARNING log line naming the worker,
    the failed starts and the candidate, and neither is re-raised as a warning.
    """
    _patch_optimizer(monkeypatch, (_FIRST_TRY, RuntimeWarning), (_SECOND_TRY, RuntimeWarning))

    with caplog.at_level('INFO', logger='fwl.proteus.inference.BO'):
        x = bo_mod.optimize_acqf_logged(acq_function=None, d=2, who='worker 3')

    assert x.tolist() == [[0.25, 0.75]]
    assert len(caplog.records) == 1
    rec = caplog.records[0]
    assert rec.levelname == 'WARNING'
    assert rec.getMessage() == (
        'Acquisition optimiser (worker 3): L-BFGS-B stopped early in 2 start(s) '
        '[status 2 (ABNORMAL)], failed again after a retry; proposing x = [0.25, 0.75]'
    )
    assert len(recwarn) == 0


@pytest.mark.unit
def test_optimize_acqf_logged_reports_successful_retry_at_info(monkeypatch, caplog):
    """A first-try failure that the retry fixes is logged at INFO, not WARNING."""
    _patch_optimizer(monkeypatch, (_FIRST_TRY, RuntimeWarning))

    with caplog.at_level('INFO', logger='fwl.proteus.inference.BO'):
        bo_mod.optimize_acqf_logged(acq_function=None, d=2, who='initial point 0')

    assert [r.levelname for r in caplog.records] == ['INFO']
    assert 'the retry succeeded' in caplog.records[0].getMessage()
    assert 'initial point 0' in caplog.records[0].getMessage()


_NO_STATUS = (
    "OptimizationWarning('Optimization failed within `scipy.optimize.minimize` with no "
    "status returned to `res.`')"
)


@pytest.mark.unit
def test_optimize_acqf_logged_names_starts_that_returned_no_status(monkeypatch, caplog):
    """A start that returned no status is counted and named, alone and next to
    a status failure, not logged as '0 start(s) []'."""
    head = 'Optimization failed in `gen_candidates_scipy` with the following warning(s):\n['
    tail = ']\nTrying again with a new set of initial conditions.'
    status = (
        "OptimizationWarning('Optimization failed within `scipy.optimize.minimize` with "
        "status 2 and message ABNORMAL: .')"
    )

    with caplog.at_level('INFO', logger='fwl.proteus.inference.BO'):
        _patch_optimizer(monkeypatch, (head + _NO_STATUS + tail, RuntimeWarning))
        bo_mod.optimize_acqf_logged(acq_function=None, d=2, who='worker 1')
        _patch_optimizer(
            monkeypatch, (head + _NO_STATUS + ', ' + status + tail, RuntimeWarning)
        )
        bo_mod.optimize_acqf_logged(acq_function=None, d=2, who='worker 1')

    lines = [r.getMessage() for r in caplog.records]
    assert lines[0].startswith(
        'Acquisition optimiser (worker 1): L-BFGS-B stopped early in 1 start(s) '
        '[no status returned], the retry succeeded'
    )
    assert 'in 2 start(s) [no status returned, status 2 (ABNORMAL)]' in lines[1]


@pytest.mark.unit
def test_optimize_acqf_logged_falls_back_to_a_plain_line_for_an_unknown_format(
    monkeypatch, caplog
):
    """An unparseable first-try warning is still logged, without a count or reasons."""
    _patch_optimizer(
        monkeypatch,
        (
            'Optimization failed in `gen_candidates_scipy` with the following warning(s):\n[]',
            RuntimeWarning,
        ),
    )

    with caplog.at_level('INFO', logger='fwl.proteus.inference.BO'):
        bo_mod.optimize_acqf_logged(acq_function=None, d=2, who='worker 2')

    assert [r.getMessage() for r in caplog.records] == [
        'Acquisition optimiser (worker 2): L-BFGS-B stopped early, the retry succeeded; '
        'proposing x = [0.25, 0.75]'
    ]


@pytest.mark.unit
def test_optimize_acqf_logged_passes_each_other_warning_on_once(monkeypatch):
    """A warning repeated on every BO step is passed on once; a different one
    still is."""
    monkeypatch.setattr(bo_mod, '_PASSED_ON', set())
    _patch_optimizer(monkeypatch, ('GP fit is ill-conditioned', UserWarning))

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('default')
        for _ in range(3):
            bo_mod.optimize_acqf_logged(acq_function=None, d=2, who='worker 0')
        _patch_optimizer(monkeypatch, ('input is not standardised', UserWarning))
        bo_mod.optimize_acqf_logged(acq_function=None, d=2, who='worker 0')

    assert [str(w.message) for w in seen] == [
        'GP fit is ill-conditioned',
        'input is not standardised',
    ]


@pytest.mark.unit
def test_optimize_acqf_logged_passes_other_warnings_through_and_logs_nothing(
    monkeypatch, caplog
):
    """A warning that is not botorch's retry message is re-raised unchanged and
    produces no log line; a clean optimisation logs nothing either.
    """
    monkeypatch.setattr(bo_mod, '_PASSED_ON', set())
    _patch_optimizer(monkeypatch, ('GP fit is ill-conditioned', UserWarning))

    with caplog.at_level('INFO', logger='fwl.proteus.inference.BO'):
        with pytest.warns(UserWarning, match='ill-conditioned'):
            bo_mod.optimize_acqf_logged(acq_function=None, d=2, who='worker 0')
        _patch_optimizer(monkeypatch)
        bo_mod.optimize_acqf_logged(acq_function=None, d=2, who='worker 0')

    assert caplog.records == []
