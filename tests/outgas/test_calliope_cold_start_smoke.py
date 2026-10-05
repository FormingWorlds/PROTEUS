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

    monkeypatch.setattr(np.random, 'seed', lambda *a, **k: None)
    c, d = _cold_start(1), _cold_start(2)
    assert c['P_surf'] != d['P_surf']
    for raw in (c, d):
        assert a['P_surf'] == pytest.approx(raw['P_surf'], rel=1e-4)


def test_failing_solve_draws_the_same_guesses_on_a_rerun(monkeypatch, tmp_path):
    """A solve that exhausts its attempts fails after the same sequence of
    start guesses whatever state the caller's RNG is in, so a rerun of the
    input fails the same way."""
    import calliope.solve as calliope_solve

    config = _cold_start_config()
    config.outgas.calliope.nguess = 3
    config.outgas.calliope.nsolve = 1
    real, draws = calliope_solve.get_initial_pressures, []
    monkeypatch.setattr(
        calliope_solve,
        'get_initial_pressures',
        lambda *a, **k: draws.append(real(*a, **k)) or draws[-1],
    )
    for caller_seed in (1, 2):
        np.random.set_state(np.random.RandomState(caller_seed).get_state())
        with pytest.raises(RuntimeError, match='Could not find solution'):
            calc_surface_pressures(
                {'output': str(tmp_path)}, config, _surface_pressure_hf_row()
            )
    # Per run: the cold-start guess plus one redraw after each of the 3 failed attempts.
    assert len(draws) == 8
    assert draws[:4] == draws[4:]


@pytest.mark.physics_invariant
def test_seeded_low_h_cold_start_returns_the_physical_root(tmp_path):
    """A low-H, C-rich inventory (H 1e19, C 5e19, N 1e17, S 1e18 kg) at IW+2 and
    1500 K, whose seeded cold start CALLIOPE can end on a root with negative
    pH2O, returns the physical root with positive H2O and H2."""
    config = _cold_start_config()
    config.outgas.fO2_shift_IW = 2.0
    config.outgas.calliope.nguess = 1000
    config.outgas.calliope.nsolve = 3000
    hf_row = _surface_pressure_hf_row()
    hf_row.update(
        T_magma=1500.0,
        H_kg_total=1.0e19,
        C_kg_total=5.0e19,
        N_kg_total=1.0e17,
        S_kg_total=1.0e18,
    )
    calc_surface_pressures({'output': str(tmp_path)}, config, hf_row)
    assert hf_row['H2O_bar'] > 0.0
    assert hf_row['H2_bar'] > 0.0
    assert hf_row['P_surf'] == pytest.approx(24.8673, rel=1e-4)
    # Discrimination guard: the negative-water root gives 18.60 bar.
    assert abs(hf_row['P_surf'] - 18.60) > 1.0
