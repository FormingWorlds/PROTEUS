"""Smoke test: a real CALLIOPE cold start through PROTEUS is reproducible.

``calc_surface_pressures`` runs CALLIOPE's real equilibrium solver from a
cold start (no previous pressures). CALLIOPE draws that start guess from the
global ``np.random``, and the root it returns moves within the solver
tolerance with the guess, so the wrapper seeds the draw.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('calliope')

from proteus.outgas.calliope import calc_surface_pressures
from tests.outgas.test_calliope import _cold_start_config, _surface_pressure_hf_row

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]


def _cold_start(caller_seed):
    hf_row = _surface_pressure_hf_row()
    np.random.set_state(np.random.RandomState(caller_seed).get_state())
    calc_surface_pressures({'output': '/tmp/test'}, _cold_start_config(), hf_row)
    return hf_row


@pytest.mark.physics_invariant
def test_real_cold_start_is_identical_for_any_caller_rng_state(monkeypatch):
    """Two real cold starts under different caller RNG states agree in every
    copied value. Without the wrapper's seed they differ, and the seeded root
    lies inside the solver tolerance of the unseeded ones."""
    a, b = _cold_start(1), _cold_start(2)
    assert a['P_surf'] > 0.0
    assert [k for k in a if a[k] != b[k]] == []

    monkeypatch.setattr(np.random, 'seed', lambda seed: None)
    c, d = _cold_start(1), _cold_start(2)
    assert c['P_surf'] != d['P_surf']
    for raw in (c, d):
        assert a['P_surf'] == pytest.approx(raw['P_surf'], rel=1e-4)
