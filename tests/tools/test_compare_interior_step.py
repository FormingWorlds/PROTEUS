"""Unit tests for tools/compare_interior_step.py comparison harness.

Testing standards: docs/How-to/testing.md,
docs/Explanations/test_framework.md
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / 'tools'
INTEGRATION = REPO / 'tests' / 'integration'


def _load_compare_tool():
    """Import tools/compare_interior_step.py by path."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        'compare_interior_step', TOOLS / 'compare_interior_step.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_recordings_self_parity():
    """Verify comparing a recording against itself reports bitwise match."""
    mod = _load_compare_tool()
    ref_file = INTEGRATION / 'aragog_compare_off.npz'
    if not ref_file.exists():
        pytest.skip(f'Reference file {ref_file} not found')

    result = mod.compare_recordings(ref_file, ref_file)
    assert result is True


def test_compare_recordings_detects_rheology_mismatch():
    """Verify comparing rheology-off against rheology-on detects the viscosity difference."""
    mod = _load_compare_tool()
    off_file = INTEGRATION / 'aragog_compare_off.npz'
    on_file = INTEGRATION / 'aragog_compare_on.npz'
    if not off_file.exists() or not on_file.exists():
        pytest.skip('Comparison recordings not found')

    result = mod.compare_recordings(off_file, on_file)
    assert result is False


def test_compare_recordings_shape_mismatch(tmp_path):
    """Verify comparing arrays with mismatched shapes fails cleanly."""
    mod = _load_compare_tool()
    p1 = tmp_path / 'rec1.npz'
    p2 = tmp_path / 'rec2.npz'

    np.savez(p1, arr=np.ones((10,)))
    np.savez(p2, arr=np.ones((20,)))

    result = mod.compare_recordings(p1, p2)
    assert result is False
