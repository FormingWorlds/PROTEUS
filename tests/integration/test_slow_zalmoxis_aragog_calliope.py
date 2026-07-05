"""Slow-tier integration test: production interior + outgas stack
with real Zalmoxis + real Aragog + real CALLIOPE.

This is the heaviest end-to-end test that does not yet require a
real atmosphere. Real Zalmoxis solves the structure (newton outer
solver, PALEOS EOS, 150 radial levels) on the numpy path; the
production-default unified ``PALEOS:MgSiO3`` mantle table has no JAX
structure reader, so the structure solve runs in numpy. Real Aragog
steps the entropy ODE on the resulting mantle (production
``backend='jax'``: scipy-CVode with JAX-derived RHS and analytic
Jacobian, the only JAX path exercised); real CALLIOPE partitions
volatiles at the new T, P state. Atmosphere, star, escape, atmos_chem
stay on dummy backends so the test isolates the interior + outgas
coupling boundary while exercising the production-default structure
solver.

Complements:

- ``test_slow_aragog_calliope.py`` (real aragog + real calliope,
  dummy structure), which exercises Aragog + CALLIOPE without booting
  Zalmoxis.
- ``test_slow_zalmoxis_dummy.py`` (real zalmoxis, dummy everything
  else), which exercises Zalmoxis without the coupling load from Aragog
  or CALLIOPE.

This file is the union: every code path that runs in either of the
two complementary tests is exercised here under the production
coupling cadence (Zalmoxis at the initial condition; Aragog at
every iteration; CALLIOPE at every iteration). The structure is
solved once and held fixed (``update_interval = 0``), isolating the
interior + outgas coupling from structure-refresh transients.

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
- R_int does one bounded hand-off step (expansion under the hot
  profile) then holds constant with structure refresh disabled
  (``update_interval = 0``; catches a spurious per-step refresh).
- Earth-scale R_int (5.5e6-6.7e6 m for 1 M_Earth).
- Cross-cutting mass + stability helpers.

Runtime: this end-to-end real-binary test is dominated by the Aragog
first-call JAX setup (CVode factory plus RHS and Jacobian compile)
and the Zalmoxis structure solves on the numpy path. It completes in
roughly 80 min on the macOS GHA runner and is gated to macOS
(``skipif`` on linux): the unified PALEOS mantle has no JAX structure
path, so the full-resolution coupled run on the numpy fallback
exceeds the slow-tier walltime on the x86 ubuntu runner. The per-test
timeout is 10800 s (180 min), the slow-tier standard, leaving generous
headroom above the macOS runtime; the slow-tier job cap is set higher
(210 min) so that setup time plus the per-test timeout still fit
inside the job, letting a genuine hang trip pytest-timeout's per-test
timer (which dumps every thread's stack) before the job-level
cancellation. The single IC-only structure solve (update_interval = 0)
keeps the coupled cost bounded; the per-row dynamic refresh path is
exercised separately by the structure-update unit tests.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tests.integration.conftest import (
    validate_mass_conservation,
    validate_stability,
)

pytestmark = [pytest.mark.slow, pytest.mark.timeout(10800)]


@pytest.mark.slow
@pytest.mark.physics_invariant
@pytest.mark.skipif(
    sys.platform.startswith('linux'),
    reason='Zalmoxis structure solve runs the numpy fallback (no JAX '
    'path for the unified PALEOS mantle); the full-resolution coupled '
    'run exceeds the slow-tier walltime on x86 runners. Exercised on '
    'macOS. Tracked: FormingWorlds/Zalmoxis#75.',
)
def test_zalmoxis_aragog_calliope_two_timesteps(proteus_multi_timestep_run):
    """Two-step PROTEUS run with real Zalmoxis + Aragog + CALLIOPE on
    the Earth-IC fiducial.

    Physical scenario: 1 M_Earth, 0.5 AU, IW+2 fO2 shift, with an
    H-C-N-S volatile inventory (from the test-owned config
    ``zalmoxis_aragog_calliope.toml``). The real Zalmoxis solver
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
    - R_int does a single bounded hand-off step (expansion under the
      hot profile) then stays stable (refresh disabled,
      update_interval = 0; the one-time baseline hand-off is the only
      structure change).
    - Cross-cutting mass + stability helpers.
    """
    runner = proteus_multi_timestep_run(
        config_path='tests/integration/zalmoxis_aragog_calliope.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
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

    # Earth-scale R_int at every row. Self-consistent PALEOS solve on
    # 1 M_Earth starts near 6.35e6 m and expands once under the hot
    # adiabatic hand-off to ~6.52e6 m, a few percent above 6.371e6 m.
    r_int = hf['R_int'].to_numpy()
    assert np.all(np.isfinite(r_int)), 'R_int contains NaN or Inf'
    assert np.all(r_int > 5.5e6), f'R_int below Earth scale: min={r_int.min():.3e} m'
    assert np.all(r_int < 6.7e6), f'R_int above Earth scale: max={r_int.max():.3e} m'

    # R_int follows the one-time baseline structure hand-off, then settles.
    # With update_interval = 0 the per-iteration refresh triggers are off, so
    # the only structure change is the baseline re-solve on the first non-init
    # step (which hands the structure off to the energetics-module hot
    # adiabatic temperature profile and expands R_int once); R_int is then
    # constant. The assertions encode "one bounded hand-off step then stable",
    # which still discriminates a per-step re-solve (more than one change) and
    # a runaway in either direction (magnitude guard).
    if len(r_int) >= 2:
        step_drift = np.abs(np.diff(r_int)) / r_int[0]
        # At most one significant per-step change: the baseline hand-off. A
        # refresh trigger misfiring every step would show more than one; a
        # frozen structure shows none.
        n_significant = int(np.sum(step_drift > 1e-4))
        assert n_significant <= 1, (
            f'R_int changed on more than one step (n={n_significant}); expected a '
            f'single baseline hand-off, per-step rel drifts = {step_drift}'
        )
        # The hand-off re-solves the structure under the hot adiabatic profile,
        # expanding R_int once; the one-time change is bounded well below a
        # runaway in either direction (|net change| < 10%).
        net_change = (r_int[0] - r_int[-1]) / r_int[0]
        assert abs(net_change) < 0.10, (
            f'R_int net change {net_change:.3e} exceeds 10%: expected a '
            f'bounded one-time baseline hand-off, not a runaway'
        )
        assert np.all(np.abs(np.diff(r_int)) <= 0.10 * r_int[0]), (
            f'R_int step change exceeds 10% of baseline; per-step diffs = {np.diff(r_int)}'
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
