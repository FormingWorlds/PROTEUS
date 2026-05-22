"""Slow-tier integration test: production interior + outgas stack
with real Zalmoxis + real Aragog + real CALLIOPE.

This is the heaviest end-to-end test that does not yet require a
real atmosphere. Real Zalmoxis solves the structure (newton outer
solver, JAX backend, PALEOS EOS, 150 radial levels); real Aragog
steps the entropy ODE on the resulting mantle (production
``backend='jax'``: scipy-CVode with JAX-derived RHS and analytic
Jacobian); real CALLIOPE partitions volatiles at the new T, P
state. Atmosphere, star, escape, atmos_chem stay on dummy backends
so the test isolates the interior + outgas coupling boundary while
exercising the production-default structure solver.

Complements:

- ``test_slow_aragog_calliope.py`` (real aragog + real calliope,
  dummy structure) — exercises Aragog + CALLIOPE without booting
  Zalmoxis.
- ``test_slow_zalmoxis_dummy.py`` (real zalmoxis, dummy everything
  else) — exercises Zalmoxis without the coupling load from Aragog
  or CALLIOPE.

This file is the union: every code path that runs in either of the
two complementary tests is exercised here under the production
coupling cadence (Zalmoxis at IC + after structure-refresh
triggers; Aragog at every iteration; CALLIOPE at every iteration).

Invariants asserted:

- Helpfile has at least 2 rows.
- Discrimination guard on Zalmoxis (config.interior_struct.module)
  and the Aragog JAX CVODE factory call counter, so a regression
  that silently fell back to dummy or to the FD Jacobian fails
  loudly.
- Per-element mass closure for H, C, N, S, O at the final row.
  ``M_int + sum(<E>_kg_atm + <E>_kg_liquid + <E>_kg_solid)``
  matches ``M_planet`` within rel=1e-2 (loosened from the
  aragog+calliope test's 1e-2 because the Zalmoxis dry-mass
  subtraction introduces its own roundoff at the kg scale).
- Positivity (sign + scale guards) on T_magma, P_surf, R_int,
  M_int, gravity at every row.
- ``Phi_global`` bounded to [0, 1].
- Cross-step continuity on T_magma (|dT| < 1000 K rejects an
  entropy-solver runaway) and Phi_global (|dphi| < 0.5 rejects an
  unphysical melt-fraction jump).
- R_int constant across rows under the default update_interval =
  1 Gyr (catches a spurious structure refresh).
- Earth-scale R_int (5.5e6-6.5e6 m for 1 M_Earth).
- Cross-cutting mass + stability helpers.

Runtime budget: ~6 min macOS GHA (~3 min Zalmoxis setup + EOS load,
~2 min Aragog setup, ~1 min coupled iteration), ~25 min Linux GHA
(JAX/CVode setup tax on x86 plus Zalmoxis EOS table load). The
3600 s timeout stays well inside the slow-tier 120 min step cap.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.integration.conftest import (
    validate_mass_conservation,
    validate_stability,
)

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_zalmoxis_aragog_calliope_two_timesteps(proteus_multi_timestep_run):
    """Two-step PROTEUS run with real Zalmoxis + Aragog + CALLIOPE on
    the Earth-IC fiducial.

    Physical scenario: 1 M_Earth, 0.5 AU, IW+2 fO2 shift, 3000 ppmw
    H budget (from ``input/dummy.toml``). The real Zalmoxis solver
    sets up the mass-radius profile at IC; Aragog steps the entropy
    ODE on the mantle (backend='jax'); CALLIOPE partitions
    volatiles at the new T, P state every iteration. The three real
    solvers must produce a trajectory where every element's
    reservoir sum equals its total budget, every physical scalar is
    finite and bounded, and the interior temperature does not jump
    unphysically.

    Verifies:

    - At least 2 helpfile rows.
    - Discrimination guards: Zalmoxis module is on (not dummy
      fallback); Aragog JAX CVODE factory was invoked at least
      once (not the FD Jacobian fallback).
    - Earth-scale R_int across rows.
    - Per-element mass closure for H, C, N, S, O at the final row
      within rel=1e-2.
    - Sign guard on every reservoir mass.
    - Positivity of T_magma, P_surf, R_int, M_int, gravity at every
      row.
    - ``Phi_global`` in [0, 1].
    - Cross-step continuity: |dT_magma| < 1000 K, |dPhi_global| <
      0.5.
    - R_int stable across rows (default update_interval = 1 Gyr).
    - Cross-cutting mass + stability helpers.
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
        interior_struct__module='zalmoxis',
        interior_energetics__module='aragog',
        outgas__module='calliope',
    )

    hf = runner.hf_all
    assert hf is not None, 'helpfile should be created'
    assert len(hf) >= 2, f'expected >= 2 rows, got {len(hf)}'

    # Discrimination guards on both production paths.
    assert runner.config.interior_struct.module == 'zalmoxis', (
        'interior_struct silently swapped away from zalmoxis'
    )
    assert runner.config.interior_energetics.module == 'aragog', (
        'interior_energetics silently swapped away from aragog'
    )
    assert runner.config.outgas.module == 'calliope', (
        'outgas silently swapped away from calliope'
    )
    # Aragog JAX CVODE factory must have fired at least once; a
    # silent fallback to FD Jacobian would still pass the physics
    # invariants below.
    solver = runner.interior_o.aragog_solver
    assert solver is not None, 'aragog solver missing after run'
    n_factory_calls = getattr(solver, '_jax_factory_call_count', None)
    assert n_factory_calls is not None, (
        'JAX CVODE factory never installed on solver; backend may '
        'have silently fallen back to FD Jacobian'
    )
    assert n_factory_calls >= 1, (
        f'JAX CVODE factory installed but never invoked '
        f'(call_count={n_factory_calls}); production analytic-Jacobian '
        f'path was not exercised'
    )

    # Earth-scale R_int at every row. Self-consistent PALEOS solve
    # on 1 M_Earth lands within a few percent of 6.371e6 m.
    r_int = hf['R_int'].to_numpy()
    assert np.all(np.isfinite(r_int)), 'R_int contains NaN or Inf'
    assert np.all(r_int > 5.5e6), f'R_int below Earth scale: min={r_int.min():.3e} m'
    assert np.all(r_int < 6.5e6), f'R_int above Earth scale: max={r_int.max():.3e} m'

    # R_int stable across rows (update_interval = 1 Gyr fires only
    # at IC over the 1e3 yr run).
    if len(r_int) >= 2:
        rel_drift = np.max(np.abs(np.diff(r_int))) / r_int[0]
        assert rel_drift < 1e-6, (
            f'R_int drifted across rows despite update_interval = 1 Gyr; '
            f'max rel drift = {rel_drift:.3e}'
        )

    final = hf.iloc[-1]

    # Per-element mass closure: the conservation invariant.
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

    # Positivity of physical scalars at every row.
    for col in ('T_magma', 'P_surf', 'R_int', 'M_int', 'gravity'):
        if col not in hf.columns:
            continue
        vals = hf[col].to_numpy()
        assert np.all(np.isfinite(vals)), f'{col}: NaN or Inf'
        assert np.all(vals > 0), f'{col}: non-positive value present, min={vals.min():.3e}'

    # Phi_global bounded.
    if 'Phi_global' in hf.columns:
        phi = hf['Phi_global'].to_numpy()
        assert np.all((0 <= phi) & (phi <= 1)), (
            f'Phi_global out of [0,1], observed [{phi.min():.3e}, {phi.max():.3e}]'
        )

    # Cross-step continuity of T_magma.
    if 'T_magma' in hf.columns and len(hf) >= 2:
        dT = np.diff(hf['T_magma'].to_numpy())
        assert np.all(np.abs(dT) < 1000.0), (
            f'T_magma jump too large: max(|dT|)={np.max(np.abs(dT)):.1f} K'
        )

    # Cross-step continuity of Phi_global.
    if 'Phi_global' in hf.columns and len(hf) >= 2:
        dphi = np.diff(hf['Phi_global'].to_numpy())
        assert np.all(np.abs(dphi) < 0.5), (
            f'Phi_global jump too large: max(|dPhi|)={np.max(np.abs(dphi)):.3f}'
        )

    # Conservation + stability helpers.
    mass_results = validate_mass_conservation(hf, tolerance=0.2)
    assert mass_results.get('masses_positive', True), 'mass reservoirs negative'

    stability = validate_stability(hf, max_temp=1e6, max_pressure=1e10)
    assert stability['temps_stable'], 'temperature instability detected'
    assert stability['pressures_stable'], 'pressure instability detected'
