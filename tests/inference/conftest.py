"""Fixtures shared by the inference tests.

Torch ignores OMP_NUM_THREADS once juliacall is loaded before it, as `import proteus` does,
and then uses every core; on a shared many-core node the small GP fits slow down a
hundredfold. Each test therefore runs on a fixed thread count, restored afterwards, so
a test that changes it (`async_BO.worker`) does not leak into the next.
"""

from __future__ import annotations

import pytest

# As tests/conftest.py sets OMP_NUM_THREADS
TEST_TORCH_THREADS = 2


@pytest.fixture(autouse=True)
def _pinned_torch_threads():
    """Run each test on TEST_TORCH_THREADS torch threads and restore the count after it."""
    try:
        import torch
    except ImportError:
        yield
        return
    before = torch.get_num_threads()
    torch.set_num_threads(TEST_TORCH_THREADS)
    yield
    torch.set_num_threads(before)
