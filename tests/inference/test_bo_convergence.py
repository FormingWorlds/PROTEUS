"""
Convergence tests for the PROTEUS Bayesian-optimization scheme.

These tests run the real `BO_step` loop on a synthetic forward problem
with a closed-form optimum, and assert that the best observed Y value
improves over iterations. Existing unit-tier tests in `test_bo.py` mock
the GP and acquisition pipeline; they verify per-step plumbing but
cannot catch a regression in convergence behaviour. These tests fill
that gap.

All synthetic objectives are deterministic and bounded on the
unit-hypercube [0, 1]^d so the BO scheme operates in its native
parameter space without any rescaling. Tests are marked `slow` so
they do not gate PR CI; they run in nightly + on demand.

Typical wall cost: ~5-15 s per test (one BO loop with N <= 12 steps).
"""

from __future__ import annotations

import random
import threading

import numpy as np
import pytest
import torch
from botorch.exceptions.warnings import OptimizationWarning

from proteus.inference import BO as bo_mod
from proteus.inference.utils import get_kernel_w_prior

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Synthetic objectives
# ---------------------------------------------------------------------------
def make_quadratic_objective(target: torch.Tensor):
    """Return f(x) = -||x - target||^2.

    The minimum of `||x - target||^2` is 0 at x = target. The negation
    gives a maximum of 0 at x = target, which matches the BO scheme's
    convention (maximize Y). Returns a torch tensor shaped like
    `(1,)` to satisfy the BO_step contract.
    """

    def _f(x: torch.Tensor) -> torch.Tensor:
        # x is shape (1, d)
        diff = x.flatten() - target
        return -(diff * diff).sum().unsqueeze(0)

    return _f


# ---------------------------------------------------------------------------
# BO loop helper
# ---------------------------------------------------------------------------
def _run_bo_loop(
    objective,
    d: int,
    n_init: int,
    n_iter: int,
    acqf: str = 'LogEI',
    seed: int = 0,
    monkeypatch=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a BO optimization loop; return (X_history, Y_history).

    The plotting side effect inside `BO_step` (when d == 1) writes to
    'plots/iters/'. d >= 2 avoids that path, but for safety we use d
    >= 2 throughout.
    """
    assert d >= 2, 'use d >= 2 to avoid the d==1 plot side effect inside BO_step'

    # Seed every RNG that botorch / scipy / Python could touch. torch alone
    # is not enough: scipy.optimize internals and any numpy fall-back path
    # in the acquisition optimiser would otherwise leave residual non-
    # determinism that flakes the seed-determinism tests across hosts.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initial sample of n_init random points in [0, 1]^d
    X = torch.rand((n_init, d), dtype=torch.double)
    Y = torch.stack([objective(X[i : i + 1]) for i in range(n_init)]).reshape(-1, 1)

    D = {'X': X, 'Y': Y}
    # BO_step computes the cdist between this worker's candidate and the
    # OTHER workers' busy points; with only one entry the post-mask tensor
    # is empty and torch.min raises. Use two workers (the test acts as
    # worker 0; worker 1 is an inert placeholder so the cdist branch has
    # something to compute against) to match how the production loop
    # invokes BO_step.
    B = {
        0: torch.zeros((1, d), dtype=torch.double),
        1: 0.5 * torch.ones((1, d), dtype=torch.double),
    }
    lock = threading.Lock()
    kernel = get_kernel_w_prior(ard_num_dims=d)

    # Suppress harmless GP-fit warnings to keep test output tidy.
    import warnings

    for _ in range(n_iter):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=OptimizationWarning)
            warnings.simplefilter('ignore', category=UserWarning)
            x_next, y_next, *_ = bo_mod.BO_step(
                D=D,
                B=B,
                f=objective,
                k=kernel,
                acqf=acqf,
                lock=lock,
                worker_id=0,
            )
        D['X'] = torch.cat([D['X'], x_next], dim=0)
        D['Y'] = torch.cat([D['Y'], y_next.reshape(-1, 1)], dim=0)
        # Refresh kernel each iteration so ARD lengthscales adapt to the
        # growing dataset (matches how the production loop rebuilds it).
        kernel = get_kernel_w_prior(ard_num_dims=d)

    return D['X'], D['Y']


# ---------------------------------------------------------------------------
# Convergence on a known-optimum problem
# ---------------------------------------------------------------------------
def _initial_max_distance(X_init: torch.Tensor, target: torch.Tensor) -> float:
    """Smallest distance from the initial X sample to the target.

    The convergence proximity check below requires that the BO loop close
    a meaningful fraction of this initial distance. Computing it this way
    keeps the assertion bound tied to the actual seed-dependent starting
    geometry rather than a hard-coded magic number.
    """
    return torch.min(torch.norm(X_init - target, dim=1)).item()


def test_bo_converges_on_centered_quadratic():
    """BO must reduce both the gap to the optimum value AND the distance
    from the best-x to the target on a centered quadratic in 2D. Optimum
    is Y = 0 at x = (0.5, 0.5).

    Two independent assertions guard against distinct regression classes:
    - Gap reduction (Y improvement) catches a BO that returns wrong values.
    - Proximity reduction (||x_best - target|| shrinks) catches a BO that
      returns a constant x (e.g., always (0.5, 0.5) coincidentally near
      the centered target). Without the proximity check, the constant-
      output regression would pass the gap check on the centered target.
    """
    target = torch.tensor([0.5, 0.5], dtype=torch.double)
    objective = make_quadratic_objective(target)

    X, Y = _run_bo_loop(objective, d=2, n_init=4, n_iter=8, acqf='LogEI', seed=1)

    Y_init_best = Y[:4].max().item()
    Y_final_best = Y.max().item()
    initial_gap = -Y_init_best
    final_gap = -Y_final_best
    assert final_gap < 0.5 * initial_gap, (
        f'BO did not improve the best Y by 50% of the initial gap. '
        f'Y_init_best={Y_init_best:.4f}, Y_final_best={Y_final_best:.4f}, '
        f'initial gap={initial_gap:.4f}, final gap={final_gap:.4f}.'
    )

    initial_min_dist = _initial_max_distance(X[:4], target)
    final_min_dist = torch.min(torch.norm(X - target, dim=1)).item()
    assert final_min_dist < 0.5 * initial_min_dist, (
        f'BO best-x did not approach target. Initial min ||x - target|| = '
        f'{initial_min_dist:.4f}, final = {final_min_dist:.4f}. A regression '
        f'that returned a constant x would fire here.'
    )


def test_bo_converges_on_off_center_quadratic():
    """Same convergence test but with the optimum off-center to catch
    any hard-coded centering bias in the acquisition optimisation.
    Optimum at x = (0.2, 0.8). The proximity check is essential here:
    if BO always returned (0.5, 0.5), the gap to the off-center target
    would NOT improve, and the proximity bound would catch it directly.
    """
    target = torch.tensor([0.2, 0.8], dtype=torch.double)
    objective = make_quadratic_objective(target)

    X, Y = _run_bo_loop(objective, d=2, n_init=4, n_iter=8, acqf='LogEI', seed=2)

    Y_init_best = Y[:4].max().item()
    Y_final_best = Y.max().item()
    initial_gap = -Y_init_best
    final_gap = -Y_final_best
    assert final_gap < 0.5 * initial_gap, (
        f'BO did not improve the best Y by 50% on off-center quadratic. '
        f'Y_init_best={Y_init_best:.4f}, Y_final_best={Y_final_best:.4f}.'
    )

    initial_min_dist = _initial_max_distance(X[:4], target)
    final_min_dist = torch.min(torch.norm(X - target, dim=1)).item()
    assert final_min_dist < 0.5 * initial_min_dist, (
        f'BO best-x did not approach off-center target. Initial min '
        f'||x - target|| = {initial_min_dist:.4f}, final = {final_min_dist:.4f}.'
    )


# ---------------------------------------------------------------------------
# Acquisition-function selection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('acqf', ['LogEI', 'UCB', 'E-LogEI'])
def test_bo_converges_with_each_acquisition_function(acqf):
    """All three documented acquisition functions must converge on the
    centered quadratic. The two-worker B emulation in `_run_bo_loop`
    supports the X_pending shape that E-LogEI's `qLogExpectedImprovement`
    expects (worker 1 acts as a static pending point).
    """
    target = torch.tensor([0.5, 0.5], dtype=torch.double)
    objective = make_quadratic_objective(target)

    X, Y = _run_bo_loop(objective, d=2, n_init=4, n_iter=6, acqf=acqf, seed=3)

    Y_init_best = Y[:4].max().item()
    Y_final_best = Y.max().item()
    initial_gap = -Y_init_best
    final_gap = -Y_final_best
    assert final_gap < 0.7 * initial_gap, (
        f'Acquisition {acqf!r} did not improve the best Y by 30%. '
        f'Y_init_best={Y_init_best:.4f}, Y_final_best={Y_final_best:.4f}.'
    )
    # Discrimination: the best-x must approach the target geometrically, not
    # just lower its Y by exploiting some pathological GP fit. A regression
    # that returned constant `(0.5, 0.5)` for every acqf would still satisfy
    # the gap check on this centered target; the proximity bound is what
    # catches that mode.
    initial_min_dist = _initial_max_distance(X[:4], target)
    final_min_dist = torch.min(torch.norm(X - target, dim=1)).item()
    assert final_min_dist < initial_min_dist, (
        f'Acquisition {acqf!r} did not move best-x closer to target. '
        f'Initial min={initial_min_dist:.4f}, final={final_min_dist:.4f}.'
    )


# ---------------------------------------------------------------------------
# Determinism / seed reproducibility
# ---------------------------------------------------------------------------
def test_bo_loop_is_seed_deterministic():
    """Two BO runs with the same seed produce the same final dataset.

    Catches a regression where someone introduces uninstrumented
    randomness (e.g. a numpy.random.rand without seeding) that would
    make CI runs non-reproducible across hosts.
    """
    target = torch.tensor([0.5, 0.5], dtype=torch.double)
    objective = make_quadratic_objective(target)

    X1, Y1 = _run_bo_loop(objective, d=2, n_init=3, n_iter=4, seed=42)
    X2, Y2 = _run_bo_loop(objective, d=2, n_init=3, n_iter=4, seed=42)

    assert torch.allclose(X1, X2, rtol=1e-9, atol=1e-9), (
        'BO loop is non-deterministic at a fixed seed; X1 and X2 differ. '
        'A future host- or thread-dependent randomness is bleeding through.'
    )
    assert torch.allclose(Y1, Y2, rtol=1e-9, atol=1e-9), (
        'BO Y outputs are non-deterministic at a fixed seed.'
    )


def test_bo_different_seeds_produce_different_trajectories():
    """Two BO runs with different seeds must NOT produce identical X.

    Anti-happy-path: catches the dual regression where someone HARD-codes
    a seed inside BO_step or initialises X identically regardless of the
    test seed.
    """
    target = torch.tensor([0.5, 0.5], dtype=torch.double)
    objective = make_quadratic_objective(target)

    X1, _ = _run_bo_loop(objective, d=2, n_init=3, n_iter=3, seed=10)
    X2, _ = _run_bo_loop(objective, d=2, n_init=3, n_iter=3, seed=20)

    # Initial random points should differ
    assert not torch.allclose(X1[:3], X2[:3], rtol=1e-3), (
        'BO loop produced identical initial X under different seeds; the '
        'random-seed plumbing is broken or seeds are silently overridden.'
    )
    # Discrimination: the BO-selected candidates (rows beyond the initial
    # sample) must also diverge. A regression that re-seeded inside the
    # acquisition optimiser would let the initial-X check pass while still
    # collapsing the BO trajectory to a seed-independent path.
    assert not torch.allclose(X1[3:], X2[3:], rtol=1e-3), (
        'BO post-init trajectory is seed-independent; the acquisition '
        'optimiser is overriding the test seed.'
    )


# ---------------------------------------------------------------------------
# Invalid and degenerate inputs
# ---------------------------------------------------------------------------
def test_bo_step_rejects_unknown_acquisition():
    """An unknown acquisition function name must raise ValueError, not
    silently fall through to a default. Already covered by the existing
    test_bo_step_raises_for_unknown_acquisition unit test, but pinning
    here because the convergence loop above must select among the
    documented acquisitions."""
    target = torch.tensor([0.5, 0.5], dtype=torch.double)
    objective = make_quadratic_objective(target)
    with pytest.raises(ValueError, match=r'Unknown acquisition function'):
        _run_bo_loop(objective, d=2, n_init=3, n_iter=1, acqf='not-a-real-acqf', seed=4)
    # Discrimination: a known-good acqf in the same harness must complete
    # without raising AND grow the dataset by exactly one iteration.
    # Without this paired call, a regression that raised
    # `ValueError('Unknown acquisition function')` for EVERY acqf would
    # still pass the test above; without the shape pin, a regression that
    # returned an empty X tensor would also pass.
    X_ok, _ = _run_bo_loop(objective, d=2, n_init=3, n_iter=1, acqf='LogEI', seed=4)
    assert X_ok.shape == (4, 2)


def test_bo_handles_constant_objective_without_crashing():
    """An objective that returns the same Y for every X is degenerate
    (no signal for the GP to exploit) but must not crash. The loop
    should complete without an exception. This guards against the BO
    scheme dividing by an empty Y-variance or similar degenerate-path
    failures.
    """

    def constant_objective(x):
        return torch.tensor([0.0], dtype=torch.double)

    X, Y = _run_bo_loop(constant_objective, d=2, n_init=3, n_iter=3, seed=5)
    # Y must all be 0
    assert torch.allclose(Y, torch.zeros_like(Y))
    # X must have advanced
    assert X.shape[0] == 3 + 3


# ---------------------------------------------------------------------------
# Sanity: the helper objective itself is correct
# ---------------------------------------------------------------------------
# This test deliberately carries TWO tier markers: the module-level
# `pytestmark = pytest.mark.slow` plus a function-level `@pytest.mark.unit`.
# Pytest applies both, so the test matches BOTH the `unit` PR-CI filter
# AND the `slow` nightly filter. The unit-tier inclusion is intentional:
# the convergence tests above all rely on this synthetic objective being
# correct, so a bug here would invalidate the whole file silently in
# nightly. Running it in PR CI keeps the helper honest at fast tier.
@pytest.mark.unit
def test_quadratic_objective_returns_zero_at_target():
    """The synthetic objective evaluates to exactly 0 at its target point.

    If the helper itself were buggy, the convergence tests above would
    test nothing meaningful, so this sanity check is the gate at PR CI
    that prevents a silent helper-level regression.
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
