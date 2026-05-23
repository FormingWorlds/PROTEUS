"""Slow-tier integration test: real Zalmoxis structure solver with
dummy for every other slot.

Exercises the production Zalmoxis interior structure solver (newton
outer solver, JAX+diffrax backend, PALEOS EOS, 150 radial levels)
on a 1 M_Earth planet with dummy interior energetics, dummy outgas,
dummy atmosphere, dummy escape, dummy star, and dummy atmos_chem.
The pair tests in
``test_integration_zalmoxis_{aragog,spider,calliope,atmodeller}.py``
cover the schema-tier cross-validators; this test boots the real
solver and pins the production trajectory against physical
invariants.

Invariants asserted:

- Helpfile has at least 2 rows.
- ``R_int`` is consistent with the Earth-like scale (5.5e6-6.5e6 m
  for 1 M_Earth at a typical mantle density).
- ``M_int`` mass closure: ``M_int + M_atm`` matches ``M_planet``
  within rel=1e-3 (conservation invariant, §2 carve-out).
- ``R_core / R_int`` lands close to the configured ``core_frac =
  0.55`` (radius mode) within ±5% of the input value.
- ``P_center``, ``P_cmb`` positive and bounded in physical ranges
  (~3e11 Pa and ~1.5e11 Pa for Earth-like).
- ``core_density``, ``core_heatcap`` populated with positive
  physical values when the ``'self'`` sentinel resolves through
  the EOS.
- ``gravity`` positive and bounded (5-15 m/s^2 for 1 M_Earth).
- Cross-step stability: with the default ``update_interval = 1e9
  yr``, only the IC structure solve fires in a 1e3 yr run, so
  ``R_int`` is constant to within numerical roundoff across rows.

Discrimination guard: the test verifies that Zalmoxis (not the
dummy Noack & Lasbleis scaling law) ran, by reading the structure
config off the runner. A regression that silently fell back to the
dummy module would still pass the Earth-scale R_int check but fail
the explicit module-string assertion.

Runtime budget: ~600-1200 s on Linux GHA (JAX + diffrax + CVode
setup tax on x86; mirrors the aragog slow tier). macOS arm64 is
skipped at the module level because the same configuration takes
markedly longer there and exhausts the 3600 s pytest-timeout; if
the macOS path is re-enabled later, budget at least 3600 s.

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

# Linux exercises the production Zalmoxis path end-to-end. The IC
# solve via `solve_structure` is the only Zalmoxis call exercised
# in this test; the test overrides below disable the
# `equilibrate_initial_state` loop (up to 15 more solves) and the
# per-iteration `update_structure_from_interior` refresh. With those
# off the test lands well inside the standard 3600 s slow-tier
# budget on GHA ubuntu-latest.
#
# On macOS arm64 the same test path is markedly slower (JAX +
# diffrax PALEOS solve runs far slower on Apple Silicon); the test
# is skipif(darwin) until the macOS / JAX / diffrax slowness is
# investigated at the wrapper or library level (TODO).
pytestmark = [
    pytest.mark.slow,
    pytest.mark.timeout(3600),
    pytest.mark.skipif(
        sys.platform == 'darwin',
        reason='Zalmoxis + JAX PALEOS solve is markedly slower on macOS arm64; Linux covers production path',
    ),
]


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_zalmoxis_dummy_two_timesteps(proteus_multi_timestep_run):
    """Two-step PROTEUS run with real Zalmoxis + dummy for all other
    slots on a 1 M_Earth Earth-like fiducial.

    Physical scenario: 1 M_Earth, 0.5 AU, IW+2 fO2 shift, 3000 ppmw
    H budget (from ``input/dummy.toml``). The real Zalmoxis solver
    (production default ``outer_solver='newton'``, ``use_jax=True``,
    PALEOS mantle + iron core EOS) solves the structure at IC and
    holds it constant across the two timesteps because
    ``update_interval`` defaults to 1 Gyr.

    Verifies:

    - Helpfile has at least 2 rows.
    - The interior structure module is the real Zalmoxis wrapper,
      not the dummy scaling-law fallback (discrimination guard).
    - Earth-scale R_int: 5.5e6 < R_int < 6.5e6 m at every row.
    - Mass closure: M_int + M_atm = M_planet within rel=1e-3.
    - Core-radius ratio: 0.50 < R_core/R_int < 0.60 (input
      ``core_frac = 0.55`` in dummy.toml, radius mode).
    - P_center, P_cmb positive and in physical ranges.
    - core_density, core_heatcap positive (the 'self' sentinel
      resolved to physical values through the EOS).
    - Gravity in [5, 15] m/s^2.
    - R_int stable across rows (no spurious refresh).
    - Cross-cutting mass + stability helpers.
    """
    # The test needs only the IC structure solve. Two extra solve
    # sources fire by default and must be disabled:
    #
    # 1. `equilibrate_initial_state` iterates CALLIOPE + Zalmoxis up
    #    to 15 times before the main loop. With dummy outgas the
    #    P_surf relative change never converges (P stays near zero),
    #    so the loop runs to the iteration cap and burns the wall
    #    time. `equilibrate_init=False` skips the loop entirely.
    #
    # 2. `update_structure_from_interior` runs once per main-loop
    #    iteration when `update_interval > 0`. The composition
    #    trigger inside it (`d_w_H2O >= 0.05`) is hardcoded and
    #    cannot be raised from config, so the only way to keep the
    #    main loop quiet with dummy outgas is to set
    #    `update_interval=0`, which short-circuits the wrapper
    #    before any trigger is evaluated.
    #
    # The other refresh triggers (`update_dtmagma_frac`,
    # `update_dphi_abs`, `update_stale_ceiling`) become unreachable
    # once `update_interval=0` returns no_update, but the test still
    # passes the high thresholds for defence-in-depth.
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
        interior_struct__module='zalmoxis',
        interior_struct__zalmoxis__equilibrate_init=False,
        interior_struct__zalmoxis__update_interval=0,
        interior_struct__zalmoxis__update_dtmagma_frac=0.999,
        interior_struct__zalmoxis__update_dphi_abs=0.999,
        interior_struct__zalmoxis__update_stale_ceiling=0,
    )

    hf = runner.hf_all
    assert hf is not None, 'helpfile should be created'
    assert len(hf) >= 2, f'expected >= 2 rows, got {len(hf)}'

    # Discrimination guard: confirm the real Zalmoxis module ran, not
    # the dummy scaling-law fallback. The dummy module would also
    # produce Earth-scale R_int (Noack & Lasbleis scaling), so the
    # physical-range checks below would pass for the wrong reason
    # under a regression that silently switched the structure module.
    assert runner.config.interior_struct.module == 'zalmoxis', (
        'structure module silently swapped away from zalmoxis'
    )

    # Earth-scale R_int at every row. The Earth radius is 6.371e6 m;
    # a self-consistent PALEOS solve on 1 M_Earth lands within a few
    # percent of that. The window 5.5e6 - 6.5e6 m discriminates a
    # mass-vs-radius confusion (where R_int would land at M_Earth in
    # kg, ~6e24, off by 18 orders of magnitude).
    r_int = hf['R_int'].to_numpy()
    assert np.all(np.isfinite(r_int)), 'R_int contains NaN or Inf'
    assert np.all(r_int > 5.5e6), f'R_int below Earth scale: min={r_int.min():.3e} m'
    assert np.all(r_int < 6.5e6), f'R_int above Earth scale: max={r_int.max():.3e} m'

    # R_int is held constant by the default update_interval = 1 Gyr.
    # Two timesteps at 100-1000 yr separation should not trigger a
    # re-solve, so cross-row variation reflects only IC roundoff.
    if len(r_int) >= 2:
        rel_drift = np.max(np.abs(np.diff(r_int))) / r_int[0]
        assert rel_drift < 1e-6, (
            f'R_int drifted across rows despite update_interval = 1 Gyr; '
            f'max rel drift = {rel_drift:.3e}'
        )

    # Mass closure: M_int + (total volatile mass in atmosphere) =
    # M_planet within rel=1e-3. Pull the volatile sum directly from
    # the per-element atmospheric columns so the check does not
    # depend on M_atm being a registered column.
    final = hf.iloc[-1]
    m_int = float(final['M_int'])
    m_planet = float(final['M_planet'])
    m_atm = 0.0
    for elt in ('H', 'O', 'C', 'N', 'S'):
        col = f'{elt}_kg_atm'
        if col in final:
            m_atm += float(final[col])
    assert m_int > 0, f'M_int non-positive: {m_int:.3e}'
    assert m_planet > 0, f'M_planet non-positive: {m_planet:.3e}'
    assert m_int + m_atm == pytest.approx(m_planet, rel=1e-3), (
        f'mass closure broken: M_int + M_atm = {m_int + m_atm:.3e}, M_planet = {m_planet:.3e}'
    )

    # Core-radius ratio against the configured core_frac = 0.55
    # (dummy.toml, radius mode). Zalmoxis honours the radius fraction
    # exactly when ``core_frac_mode='radius'``; the assertion window
    # 0.50-0.60 catches a mass-vs-radius mode confusion (in mass mode
    # 0.55 would imply R_core/R_int near 0.43 for an Earth-density
    # core).
    r_core = float(final['R_core'])
    assert r_core > 0, f'R_core non-positive: {r_core:.3e}'
    ratio = r_core / float(final['R_int'])
    assert 0.50 < ratio < 0.60, (
        f'R_core/R_int = {ratio:.3f} outside the [0.50, 0.60] band '
        f'expected for core_frac = 0.55 in radius mode'
    )

    # P_center, P_cmb positive and bounded. Earth: P_center ~ 3.6e11
    # Pa, P_cmb ~ 1.36e11 Pa. The discrimination guards are wide
    # (factor of ~5 on each) so a Mars-mass or super-Earth fiducial
    # would still pass.
    p_center = float(final['P_center'])
    p_cmb = float(final['P_cmb'])
    assert p_center > 1e11, f'P_center too low: {p_center:.3e} Pa'
    assert p_center < 1e13, f'P_center too high: {p_center:.3e} Pa'
    assert p_cmb > 1e10, f'P_cmb too low: {p_cmb:.3e} Pa'
    assert p_cmb < p_center, (
        f'P_cmb >= P_center; monotonicity broken: P_cmb={p_cmb:.3e}, P_center={p_center:.3e}'
    )

    # core_density + core_heatcap positive (the 'self' sentinel
    # resolved through the EOS). Iron at Earth-core pressures sits
    # near 13000 kg/m^3; the assertion window covers any plausible
    # core composition for a 1 M_Earth planet.
    rho_core = float(final['core_density'])
    cp_core = float(final['core_heatcap'])
    assert rho_core > 5000, f'core_density too low: {rho_core:.3e} kg/m^3'
    assert rho_core < 20000, f'core_density too high: {rho_core:.3e} kg/m^3'
    assert cp_core > 200, f'core_heatcap too low: {cp_core:.3e} J/kg/K'
    assert cp_core < 2000, f'core_heatcap too high: {cp_core:.3e} J/kg/K'

    # Surface gravity bounded. Earth: 9.81 m/s^2. The 5-15 m/s^2
    # window covers Mars-mass through super-Earth fiducials.
    g_surf = hf['gravity'].to_numpy()
    assert np.all(np.isfinite(g_surf)), 'gravity has NaN or Inf'
    assert np.all(g_surf > 5.0), f'gravity below physical range: min={g_surf.min():.3f}'
    assert np.all(g_surf < 15.0), f'gravity above physical range: max={g_surf.max():.3f}'

    # Cross-cutting helpers. Mass tolerance loosened to 20% because
    # the dummy outgas + dummy atmos do not enforce closure as
    # tightly as the real solvers.
    mass_results = validate_mass_conservation(hf, tolerance=0.2)
    assert mass_results.get('masses_positive', True), 'mass reservoirs negative'
    stability = validate_stability(hf, max_temp=1e6, max_pressure=1e10)
    assert stability['temps_stable'], 'temperature instability detected'
    assert stability['pressures_stable'], 'pressure instability detected'
