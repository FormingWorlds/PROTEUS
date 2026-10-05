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
from tests.outgas.test_calliope import (  # noqa: F401
    _cold_start_config,
    _restore_global_rng,
    _surface_pressure_hf_row,
)

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]


def _cold_start(caller_seed):
    hf_row = _surface_pressure_hf_row()
    np.random.set_state(np.random.RandomState(caller_seed).get_state())
    calc_surface_pressures({'output': '/tmp/test'}, _cold_start_config(), hf_row)
    return hf_row


def test_real_cold_start_is_identical_for_any_caller_rng_state(monkeypatch):
    """Two real cold starts under different caller RNG states agree in every
    copied value.

    With the wrapper's seed disabled the roots differ (negative control; if
    CALLIOPE seeds itself this fails and the wrapper's seed can go). The seeded
    root lies within rel 1e-4 of these two unseeded ones, a sanity bound on the
    seed, not a check of the physics.
    """
    a, b = _cold_start(1), _cold_start(2)
    assert a['P_surf'] > 0.0
    assert [k for k in a if a[k] != b[k]] == []

    monkeypatch.setattr(np.random, 'seed', lambda seed: None)
    c, d = _cold_start(1), _cold_start(2)
    assert c['P_surf'] != d['P_surf']
    for raw in (c, d):
        assert a['P_surf'] == pytest.approx(raw['P_surf'], rel=1e-4)


def test_failing_solve_draws_the_same_guesses_on_a_rerun(monkeypatch):
    """A solve that exhausts its attempts fails after the same sequence of
    start guesses whatever state the caller's RNG is in, so a rerun of the
    input fails the same way."""
    import calliope.solve as calliope_solve

    config = _cold_start_config()
    config.outgas.calliope.nguess = 3
    config.outgas.calliope.nsolve = 1
    real = calliope_solve.get_initial_pressures
    runs = []
    for caller_seed in (1, 2):
        draws = []
        monkeypatch.setattr(
            calliope_solve,
            'get_initial_pressures',
            lambda *a, **k: draws.append(real(*a, **k)) or draws[-1],
        )
        np.random.set_state(np.random.RandomState(caller_seed).get_state())
        with pytest.raises(RuntimeError, match='Could not find solution'):
            calc_surface_pressures({'output': '/tmp/test'}, config, _surface_pressure_hf_row())
        runs.append(draws)
    # The cold-start guess plus one redraw after each of the 3 failed attempts.
    assert len(runs[0]) == 4
    assert runs[0] == runs[1]
