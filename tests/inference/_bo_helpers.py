"""
Shared synthetic objectives for the Bayesian-optimization tests.

The quadratic objective lives here rather than in a test module because two
tiers need it: the unit tier pins its behaviour (``tests/inference/test_bo.py``)
and the slow tier optimizes against it (``tests/inference/test_bo_convergence.py``).
Keeping it in one place means the convergence tests and the test that guards
them cannot drift apart.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import torch


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
