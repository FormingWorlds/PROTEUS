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

from calliope.solve import equilibrium_atmosphere

from proteus.outgas.calliope import calc_surface_pressures, construct_options
from tests.outgas.test_calliope import _cold_start_config, _surface_pressure_hf_row

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]


def _cold_start(caller_seed):
    hf_row = _surface_pressure_hf_row()
    np.random.seed(caller_seed)
    calc_surface_pressures({'output': '/tmp/test'}, _cold_start_config(), hf_row)
    return hf_row


@pytest.mark.physics_invariant
def test_real_cold_start_is_identical_for_any_caller_rng_state():
    """Two real cold starts under different caller RNG states agree in every
    copied value, and the unseeded solver does not, so the check can fail."""
    a, b = _cold_start(1), _cold_start(2)
    assert a['P_surf'] > 0.0
    assert [k for k in a if a[k] != b[k]] == []

    # Discrimination: the same solve without the wrapper's seed moves with the caller RNG.
    config, hf_row = _cold_start_config(), _surface_pressure_hf_row()
    opts = construct_options({}, config, hf_row)
    target = {e: hf_row[e + '_kg_total'] for e in ('H', 'C', 'N', 'S')}
    raw = []
    for caller_seed in (1, 2):
        np.random.seed(caller_seed)
        raw.append(
            equilibrium_atmosphere(
                target,
                opts,
                xtol=1e-6,
                rtol=1e-4,
                atol=1.0e16,
                nguess=100,
                nsolve=500,
                p_guess=None,
                p_guess_max=1.0e5,
                print_result=False,
                opt_solver=False,
            )['P_surf']
        )
    assert raw[0] != raw[1]
    assert raw[0] == pytest.approx(raw[1], rel=1e-4)
