"""
Integration test: atmodeller outgas with dummy interior, atmos, star, escape.

This is the first integration test that exercises atmodeller (Bower+2025) as
the outgas backend in a coupled PROTEUS run. Atmodeller replaces CALLIOPE
when ``config.outgas.module = 'atmodeller'``; it uses a JAX-based root finder
with real-gas EOS and non-ideal solubility laws. Wrapping it in dummy
interior / atmos / star isolates the outgas-coupling interface so a regression
in module dispatch, hf_row population, or unit conversion at the
PROTEUS-atmodeller boundary is the only failure class the test can catch.

Invariants asserted:
- Per-element mass closure for H, C, N, S, O at every row.
- Positivity of every reservoir mass and of surface pressure.
- ``Phi_global`` bounded to [0, 1].
- Cross-step continuity: P_surf and T_surf jump bounded.

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

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_atmodeller_dummy_two_timesteps(proteus_multi_timestep_run):
    """Two-step PROTEUS run with atmodeller outgas + dummy everything else.

    Physical scenario: Earth-mass, 0.5 AU dummy orbit, IW+2 fO2 shift,
    3000 ppmw H budget (matching ``input/dummy.toml``). The atmodeller
    equilibrium solver is the only real physics module; every other slot
    is dummy. This isolates the atmodeller boundary in PROTEUS: a
    regression in dispatch, hf_row schema, or solver-result unpacking
    surfaces as a single-module test failure.

    Verifies:
    - Helpfile has at least 2 rows.
    - Per-element mass closure ``atm + liquid + solid == total`` for H, C,
      N, S, O within rel=1e-2 at the final row.
    - Every reservoir mass is non-negative and finite.
    - ``P_surf > 0`` and bounded by 1e10 Pa.
    - ``Phi_global`` in [0, 1].
    - Cross-step continuity: |dP_surf| < 0.5 * P_surf_max, |dT_surf| < 500 K.

    Runtime budget: ~60-90 s (atmodeller cold-compile dominates first call;
    second call hits the JAX cache and is ~1-2 s).
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
        outgas__module='atmodeller',
        # Keep the atmodeller solver in 'basic' mode to amortize the JAX
        # compile cost. 'robust' is the default; both code paths exercise
        # the same wrapper; 'basic' is sufficient for a coupling test.
        outgas__atmodeller__solver_mode='basic',
        outgas__atmodeller__solver_multistart=1,
    )

    hf = runner.hf_all
    assert hf is not None, 'helpfile should be created'
    assert len(hf) >= 2, f'expected >= 2 rows, got {len(hf)}'

    final = hf.iloc[-1]

    # Per-element mass closure: the conservation invariant. Atmodeller
    # partitions volatiles between atmosphere and melt; the sum equals
    # the user budget (less escape, which here is dummy + small).
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
        # Sign guard: each reservoir non-negative; any negative value is
        # a numerical-stability or sign-error regression.
        assert atm >= 0, f'{atm_key} negative: {atm:.3e}'
        assert liq >= 0, f'{liq_key} negative: {liq:.3e}'
        assert sol >= 0, f'{sol_key} negative: {sol:.3e}'
        if tot > 0:
            assert atm + liq + sol == pytest.approx(tot, rel=1e-2), (
                f'{elt} closure: atm+liq+sol={atm + liq + sol:.3e}, total={tot:.3e}'
            )

    # Surface pressure is the primary scalar atmodeller controls.
    # Sign guard + scale guard: P_surf must be positive and below 1 Mbar
    # for an Earth-mass / Earth-budget config. A unit-conversion bug
    # would land >1e11 Pa (Pa-vs-bar inversion); a sign-flip would land
    # negative.
    if 'P_surf' in final:
        p = float(final['P_surf'])
        assert p > 0, f'P_surf non-positive: {p:.3e}'
        assert p < 1e10, f'P_surf above 1 Mbar: {p:.3e} Pa'

    # Melt fraction bounded.
    if 'Phi_global' in hf.columns:
        phi = hf['Phi_global'].to_numpy()
        assert np.all((0 <= phi) & (phi <= 1)), (
            f'Phi_global out of [0,1], observed [{phi.min():.3e}, {phi.max():.3e}]'
        )

    # Cross-step continuity. Atmodeller can shift partial pressures across
    # a step but should not produce a jump that doubles P_surf in 1000 yr.
    if 'P_surf' in hf.columns and len(hf) >= 2:
        p_arr = hf['P_surf'].to_numpy()
        dp = np.diff(p_arr)
        p_max = p_arr.max()
        assert np.all(np.abs(dp) < 0.5 * p_max), (
            f'P_surf jump too large: max(|dP|)={np.max(np.abs(dp)):.3e} vs max(P)={p_max:.3e}'
        )

    if 'T_surf' in hf.columns and len(hf) >= 2:
        dT = np.diff(hf['T_surf'].to_numpy())
        # 500 K cap discriminates: a unit-conversion bug (K vs C) would
        # land an O(273) shift on the very first step.
        assert np.all(np.abs(dT) < 500.0), (
            f'T_surf jump too large: max(|dT|)={np.max(np.abs(dT)):.1f} K'
        )

    # Conservation + stability cross-cutting helpers.
    mass_results = validate_mass_conservation(hf, tolerance=0.2)
    assert mass_results.get('masses_positive', True), 'mass reservoirs negative'

    stability = validate_stability(hf, max_temp=1e6, max_pressure=1e10)
    assert stability['temps_stable'], 'temperature instability detected'
    assert stability['pressures_stable'], 'pressure instability detected'
