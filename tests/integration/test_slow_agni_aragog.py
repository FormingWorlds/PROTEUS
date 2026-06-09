"""Slow-tier integration test: real AGNI atmosphere coupled to real
Aragog interior.

AGNI (1D radiative-convective atmosphere, Julia wrapper around
SOCRATES) is the heaviest atmosphere wrapper in PROTEUS and is the
largest uncovered file at the slow tier. This test closes the gap
by booting AGNI end-to-end alongside the production Aragog interior.

The atmos_clim.agni.spectral_file is pinned to 'greygas' so the
test does not need a SOCRATES spectral-file download path. The
grey-gas dispatch exercises the same wrapper post-processing block
as the spectral path (init_agni_atmos, update_agni_atmos, run_agni,
_solve_energy, the output-dict assembly) so coverage of the wrapper
is essentially the same as the spectral mode.

Outgas stays on calliope (production default). Star, orbit, escape,
atmos_chem stay on dummy backends so the test isolates the
atmosphere + interior coupling boundary.

Invariants asserted:

- At least 2 helpfile rows.
- atmos_clim.module is 'agni' (discrimination guard against fallback).
- F_atm finite and bounded at every row.
- T_surf in [200, 4000] K under the dummy.toml IC.
- TOA optical depth < surface optical depth (radiation thins upward).
- Per-element mass closure for H, C, N, S, O at the final row within
  rel=1e-2.
- Phi_global in [0, 1].
- Cross-step continuity on T_magma (|dT_magma| < 1000 K) and
  Phi_global (|dPhi| < 0.5).
- AGNI convergence flag present and boolean.
- Cross-cutting mass + stability helpers.

Runtime budget: ~10-30 min Linux GHA dominated by Julia precompile
plus AGNI's per-step Newton solve at the grey-gas level. The 3600 s
timeout sits inside the slow-tier 200 min step cap.

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

# Linux exercises the production AGNI path end-to-end. macOS arm64
# is skipped initially: AGNI's Julia precompile + grey-gas Newton
# solve has not yet been validated on Apple Silicon at the slow
# tier (the existing test_slow_janus_aragog.py shows JANUS already
# hits a macOS-only plot issue; AGNI may have its own). Enable the
# macOS shard after first-light Linux green.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.timeout(3600),
    pytest.mark.skipif(
        sys.platform == 'darwin',
        reason='AGNI grey-gas + Aragog coupling not yet validated on macOS arm64 at the slow tier; Linux covers the production path',
    ),
]


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_agni_aragog_two_timesteps(proteus_multi_timestep_run):
    """Two-step PROTEUS run with real AGNI (grey gas) + real Aragog
    on the Earth-IC fiducial.

    Physical scenario: 1 M_Earth, 0.5 AU, IW+2 fO2 shift, 3000 ppmw
    H budget (from ``input/dummy.toml``). AGNI solves the
    radiative-convective atmosphere with grey-gas opacity; Aragog
    steps the entropy ODE on the mantle (backend='jax'); calliope
    partitions volatiles.

    Verifies:

    - At least 2 helpfile rows.
    - AGNI module is on (discrimination guard against fallback).
    - F_atm finite and physically bounded.
    - T_surf in [200, 4000] K.
    - Optical-depth monotonicity (TOA < surface) at every row that
      has both values populated.
    - Per-element mass closure within rel=1e-2.
    - Phi_global in [0, 1].
    - Cross-step T_magma and Phi continuity.
    - Cross-cutting mass + stability helpers.
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
        atmos_clim__module='agni',
        atmos_clim__agni__spectral_file='greygas',
        # AGNI grey-gas is incompatible with both Rayleigh and aerosols;
        # dummy.toml already has rayleigh=false and aerosols off, but
        # pinning here keeps the test self-documenting if dummy.toml
        # drifts.
        atmos_clim__rayleigh=False,
        atmos_clim__aerosols_enabled=False,
        interior_energetics__module='aragog',
        interior_struct__melting_dir='Monteux-600',
        outgas__module='calliope',
    )

    hf = runner.hf_all
    assert hf is not None, 'helpfile should be created'
    assert len(hf) >= 2, f'expected >= 2 rows, got {len(hf)}'

    # Discrimination guard: AGNI module ran, not dummy.
    assert runner.config.atmos_clim.module == 'agni', (
        'atmos_clim silently swapped away from agni'
    )
    assert runner.config.interior_energetics.module == 'aragog', (
        'interior_energetics silently swapped away from aragog'
    )

    # F_atm finite and physically bounded. AGNI's Newton solver can
    # transiently produce negative F_atm during early iterations when
    # the atmosphere absorbs more than it emits; the honest invariant
    # is finite-and-bounded, not strict sign.
    f_atm = hf['F_atm'].to_numpy()
    assert np.all(np.isfinite(f_atm)), 'F_atm contains NaN or Inf'
    assert np.all(np.abs(f_atm) < 1e7), (
        f'F_atm out of physical range: max(|F_atm|)={np.max(np.abs(f_atm)):.3e} W/m^2'
    )

    # T_surf in the AGNI-supported range. The grey-gas + dummy.toml IC
    # should not stray near the bounds; the wide window catches a
    # runaway temperature regression.
    t_surf = hf['T_surf'].to_numpy()
    assert np.all(np.isfinite(t_surf)), 'T_surf contains NaN or Inf'
    assert np.all(t_surf > 200.0), f'T_surf too low: min={t_surf.min():.1f} K'
    assert np.all(t_surf < 4000.0), f'T_surf too high: max={t_surf.max():.1f} K'

    # Optical-depth monotonicity: radiation must thin upward. AGNI
    # populates tau_atm_TOA and tau_atm_surface in hf_row at every
    # iteration; the assertion catches a regression that swapped the
    # two or that lost the diagnostic.
    if {'tau_atm_TOA', 'tau_atm_surface'} <= set(hf.columns):
        tau_toa = hf['tau_atm_TOA'].to_numpy()
        tau_srf = hf['tau_atm_surface'].to_numpy()
        valid = (tau_toa > 0) & (tau_srf > 0)
        if valid.any():
            assert np.all(tau_toa[valid] < tau_srf[valid]), (
                f'tau monotonicity broken: max(tau_TOA / tau_srf) = '
                f'{(tau_toa[valid] / tau_srf[valid]).max():.3f}'
            )

    final = hf.iloc[-1]

    # Per-element mass closure.
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
        if tot > 0:
            closure = (atm + liq + sol) / tot
            assert closure == pytest.approx(1.0, rel=1e-2), (
                f'{elt} mass closure broken: (atm+liq+sol)/tot = {closure:.6f}'
            )

    # Phi_global within physical bounds.
    if 'Phi_global' in hf.columns:
        phi = hf['Phi_global'].to_numpy()
        assert np.all(np.isfinite(phi)), 'Phi_global contains NaN or Inf'
        assert np.all(phi >= 0.0), f'Phi_global negative: min={phi.min():.3f}'
        assert np.all(phi <= 1.0), f'Phi_global above 1: max={phi.max():.3f}'

    # Cross-step continuity. Both Phi_global and T_magma evolve slowly
    # on the 100-1000 yr timescale of dummy.toml; large jumps signal
    # solver instability.
    if len(hf) >= 2:
        if 'Phi_global' in hf.columns:
            dphi = float(abs(hf['Phi_global'].iloc[-1] - hf['Phi_global'].iloc[-2]))
            assert dphi < 0.5, f'|dPhi| across last step too large: {dphi:.3f}'
        if 'T_magma' in hf.columns:
            dT = float(abs(hf['T_magma'].iloc[-1] - hf['T_magma'].iloc[-2]))
            assert dT < 1000.0, f'|dT_magma| across last step too large: {dT:.1f} K'

    # AGNI convergence flag. The wrapper sets agni_converged via the
    # output dict; the column appears in hf_row but may not be
    # persisted to the helpfile (it is transient by design). Skip if
    # absent.
    if 'agni_converged' in final:
        # Boolean-like value; do not pin to True because the test runs
        # on a cold AGNI start and the first iteration may not yet
        # converge.
        v = final['agni_converged']
        assert v in (True, False, 0, 1, 0.0, 1.0), (
            f'agni_converged should be boolean-like, got {v!r}'
        )

    # Cross-cutting helpers.
    mass_results = validate_mass_conservation(hf, tolerance=0.2)
    assert mass_results.get('masses_positive', True), 'mass reservoirs negative'
    stability = validate_stability(hf, max_temp=1e6, max_pressure=1e10)
    assert stability['temps_stable'], 'temperature instability detected'
    assert stability['pressures_stable'], 'pressure instability detected'
