"""
Integration test: aragog interior coupled to atmodeller outgas.

Exercises aragog (real T-P interior energetics) with atmodeller (JAX-based
real outgas chemistry, Bower+2025 ApJ 995:59). Atmosphere, star, escape,
and structure stay on dummy backends so the test isolates the
interior + atmodeller coupling interface.

This is the second real-real interior + outgas pair tested in the suite
after aragog + calliope. The two pairs together stress every code path
in the aragog wrapper that interacts with a real outgas backend, and
between them they push the integration-tier coverage on
``src/proteus/interior_energetics/aragog.py`` and
``src/proteus/outgas/atmodeller.py`` toward the 90 % target.

Invariants asserted:
- Per-element mass closure for H, C, N, S, O at the final row.
- Positivity of every reservoir mass and physical scalar.
- ``T_magma`` and ``Phi_global`` continuity between consecutive iters.
- mass conservation + stability cross-cutting helpers.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('atmodeller')

from tests.integration.conftest import (  # noqa: E402
    validate_mass_conservation,
    validate_stability,
)

# Module-level timeout raised from the 300 s integration default because
# aragog's real-binary first-call dominates wall time and atmodeller's
# first JAX compile adds another ~15-30 s. 192 s observed on a fast
# local Mac Studio; 300 s exceeded on the macos-latest GHA runner.
# 600 s gives ~1.5x headroom on macOS without crossing into the
# slow-tier bucket.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_aragog_atmodeller_two_timesteps(proteus_multi_timestep_run):
    """Two-step PROTEUS run with aragog + atmodeller on a dummy structure.

    Physical scenario: Earth-mass, 0.5 AU, IW+2 fO2 shift, 3000 ppmw H
    budget. Aragog steps the entropy solver; atmodeller partitions
    volatiles at the new T, P state. The two real solvers must agree on
    the magma-ocean state by construction; this test pins the basic
    contract that they produce a finite, physical, mass-closing
    trajectory together.

    Verifies:
    - At least 2 helpfile rows.
    - Per-element mass closure for H, C, N, S, O at the final row within
      rel=1e-2 — the equality form is the conservation invariant.
    - Positivity of every reservoir mass and every physical scalar
      (sign guard).
    - ``T_magma`` jump between consecutive iters bounded by 1000 K
      (entropy-solver runaway guard).
    - ``Phi_global`` in [0, 1].
    - mass conservation + stability cross-cutting helpers.

    Runtime budget: ~120-180 s. The first call dominates wall time
    (aragog EOS load + JAX compile in atmodeller); the second call hits
    both caches.
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
        interior_energetics__module='aragog',
        # Legacy fallback EOS path is reached when eos_dir=None +
        # interior_struct=dummy (the aragog.py:642 else-branch).
        interior_struct__melting_dir='Monteux-600',
        outgas__module='atmodeller',
        # Keep atmodeller in basic mode + single restart to amortize
        # the JAX compile cost; both the basic and robust solver paths
        # exercise the same wrapper.
        outgas__atmodeller__solver_mode='basic',
        outgas__atmodeller__solver_multistart=1,
    )

    hf = runner.hf_all
    assert hf is not None, 'helpfile should be created'
    assert len(hf) >= 2, f'expected >= 2 rows, got {len(hf)}'

    final = hf.iloc[-1]

    for elt in ('H', 'C', 'N', 'S', 'O'):
        atm_key = f'{elt}_kg_atm'
        liq_key = f'{elt}_kg_liquid'
        sol_key = f'{elt}_kg_solid'
        tot_key = f'{elt}_kg_total'
        if not all(k in final for k in (atm_key, liq_key, sol_key, tot_key)):
            continue
        atm = float(final[atm_key])
        liq = float(final[liq_key])
        sol = float(final[sol_key])
        tot = float(final[tot_key])
        assert atm >= 0, f'{atm_key} negative: {atm:.3e}'
        assert liq >= 0, f'{liq_key} negative: {liq:.3e}'
        assert sol >= 0, f'{sol_key} negative: {sol:.3e}'
        if tot > 0:
            assert atm + liq + sol == pytest.approx(tot, rel=1e-2), (
                f'{elt} closure: atm+liq+sol={atm + liq + sol:.3e}, total={tot:.3e}'
            )

    for col in ('T_magma', 'P_surf', 'R_int', 'M_int', 'gravity'):
        if col not in hf.columns:
            continue
        vals = hf[col].to_numpy()
        assert np.all(np.isfinite(vals)), f'{col}: NaN or Inf'
        assert np.all(vals > 0), f'{col}: non-positive value present, min={vals.min():.3e}'

    if 'Phi_global' in hf.columns:
        phi = hf['Phi_global'].to_numpy()
        assert np.all((0 <= phi) & (phi <= 1)), (
            f'Phi_global out of [0,1], observed [{phi.min():.3e}, {phi.max():.3e}]'
        )

    if 'T_magma' in hf.columns and len(hf) >= 2:
        dT = np.diff(hf['T_magma'].to_numpy())
        assert np.all(np.abs(dT) < 1000.0), (
            f'T_magma jump too large: max(|dT|)={np.max(np.abs(dT)):.1f} K'
        )

    mass_results = validate_mass_conservation(hf, tolerance=0.2)
    assert mass_results.get('masses_positive', True), 'mass reservoirs negative'

    stability = validate_stability(hf, max_temp=1e6, max_pressure=1e10)
    assert stability['temps_stable'], 'temperature instability detected'
    assert stability['pressures_stable'], 'pressure instability detected'
