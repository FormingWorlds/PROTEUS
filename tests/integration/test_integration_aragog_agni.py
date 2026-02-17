"""
Integration test: ARAGOG + AGNI (interior–atmosphere coupling).

Validates multi-timestep coupling between ARAGOG (interior thermal evolution)
and AGNI (radiative-convective atmosphere). Uses the shared integration
infrastructure and validation helpers.

**Purpose**: Phase 2 §2 — first of three module-combination integration tests.
- Validates interior–atmosphere flux exchange (F_atm, F_int) with real modules
- Tests stability and conservation over several timesteps
- Establishes pattern for ARAGOG+AGNI runs in nightly CI

**Runtime**: ~3–6 h in full (AGNI is Julia-based); keep timesteps/iter limits low
for CI. Uses config derived from aragog_janus with atmos_clim switched to AGNI.

**Documentation**:
- docs/test_infrastructure.md
- docs/test_categorization.md
- docs/test_building.md (Integration Prompt)

**Data and environment**:
- ARAGOG lookup data and melting curves are downloaded automatically by the
  integration fixture when interior.module is "aragog" (locally and in CI).
  FWL_DATA must be set; the download runs once per run and is a no-op if
  data already exists.
- Julia and AGNI must be installed for the atmosphere step (see .github/copilot-instructions.md).
  Run with: pytest tests/integration/test_integration_aragog_agni.py -v -p no:faulthandler
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.integration.conftest import (
    validate_energy_conservation,
    validate_mass_conservation,
    validate_stability,
)


@pytest.mark.integration
def test_integration_aragog_agni_multi_timestep(proteus_multi_timestep_run):
    """Test multi-timestep ARAGOG + AGNI interior–atmosphere coupling.

    Physical scenario: ARAGOG provides interior structure and magma-surface
    temperature; AGNI solves radiative-convective atmosphere and returns
    F_atm. Validates that the coupled run proceeds for several timesteps
    without NaNs/Infs and that flux/temperature evolution is stable.

    Validates:
    - Simulation runs for multiple timesteps without errors
    - Helpfile has F_atm, F_int, T_surf, P_surf (or equivalents)
    - Stability: temperatures and pressures within bounds
    - Energy conservation (flux balance) within tolerance where columns exist
    - Mass conservation within tolerance where columns exist

    Config: tests/integration/aragog_janus.toml with atmos_clim overridden
    to AGNI (interior remains ARAGOG). Other modules (orbit=none, outgas=calliope,
    etc.) as in that config.
    """
    runner = proteus_multi_timestep_run(
        config_path='tests/integration/aragog_janus.toml',
        num_timesteps=4,
        max_time=1e6,
        min_time=1e2,
        # Switch atmosphere from JANUS to AGNI; interior stays ARAGOG
        atmos_clim__module='agni',
        # Ensure first atmosphere has at least one 'safe' gas (dry + opacity + thermo)
        # so AGNI's allocate! check passes. N2 is dry and has opacity in Frostflow.
        delivery__initial='volatiles',
        delivery__volatiles__N2=0.01,
        # Allow allocate when composition has no AGNI "safe" gas (e.g. spectral set
        # or first-step state). Prefer fixing composition; this is a fallback for CI.
        atmos_clim__agni__check_safe_gas=False,
    )

    assert runner.hf_all is not None, 'Helpfile should be created'
    assert len(runner.hf_all) >= 2, (
        f'Helpfile should have at least 2 timesteps, got {len(runner.hf_all)}'
    )

    # Stability checks (required columns checked inside validate_stability)
    stability_results = validate_stability(
        runner.hf_all,
        max_temp=1e6,
        max_pressure=1e10,
    )
    assert stability_results['temps_stable'], 'Temperatures should be within bounds'
    assert stability_results['pressures_stable'], 'Pressures should be within bounds'
    assert stability_results['no_unbounded_growth'], 'No unbounded growth detected'

    # Energy and mass validation (skip internally if required columns missing)
    # During magma-ocean cooling F_int >> F_atm is expected; use loose balance tolerance.
    validate_energy_conservation(
        runner.hf_all,
        tolerance=2.0,  # ARAGOG+AGNI transient: interior flux dominates until radeqm
    )
    validate_mass_conservation(
        runner.hf_all,
        tolerance=0.2,
    )

    # Sanity: time should advance
    if 'Time' in runner.hf_all.columns and len(runner.hf_all) >= 2:
        assert runner.hf_all['Time'].iloc[-1] > runner.hf_all['Time'].iloc[0], (
            'Time should progress'
        )

    # Fluxes should be finite where present (negative values are physically
    # valid, e.g. net cooling or tidal heating scenarios)
    for col in ('F_atm', 'F_int', 'F_ins'):
        if col in runner.hf_all.columns:
            vals = runner.hf_all[col].values
            assert np.all(np.isfinite(vals)), f'{col} must be finite'
