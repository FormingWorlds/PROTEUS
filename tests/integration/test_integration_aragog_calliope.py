"""
Integration test: aragog interior coupled to calliope outgas.

Exercises the aragog (real interior energetics, T-P formalism) coupling
with calliope (real outgas chemistry) over two timesteps. Atmosphere,
star, escape, and structure stay on the dummy backends so the test
isolates the interior-outgas interface.

The aragog wrapper resolves its EOS lookup table via the legacy
``1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa`` path when
``interior_struct.eos_dir`` is None and PALEOS is not in scope. This
test pins that resolution end to end: a regression in the EOS-path
fallback, or in the entropy-solver IC, surfaces here.

Invariants asserted:
- Per-element mass closure: ``atm + liquid + solid ~= total`` for H, C,
  N, S, O within 1 % at the final row.
- Positivity of every reservoir mass and every physical scalar.
- ``T_magma`` and ``Phi_global`` continuity between consecutive iterations.
- Stability via ``validate_stability``.

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

# Module-level timeout raised from the 300 s integration default because
# aragog's real-binary first-call dominates wall time (EOS table load +
# entropy IC build + first solver step). 180 s observed on a fast local
# Mac Studio; 300 s exceeded on the macos-latest GHA runner, which has
# slower CPU. 600 s gives ~1.5x headroom on macOS without crossing into
# the slow-tier bucket.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_aragog_calliope_two_timesteps(proteus_multi_timestep_run):
    """Two-step PROTEUS run with aragog + calliope on a dummy structure.

    Physical scenario: Earth-mass, 0.5 AU dummy orbit, IW+2 fO2 shift,
    3000 ppmw H budget (matching ``input/dummy.toml``). The real interior
    solver and the real outgas solver must produce a trajectory where
    every element's reservoir sum equals its total budget (mass closure),
    every physical scalar is positive and finite, and the interior
    temperature does not jump unphysically between iterations.

    Verifies:
    - At least 2 helpfile rows.
    - Per-element mass closure for H, C, N, S, O at the final row within
      rel=1e-2 — the equality form discriminates exponent / factor errors
      in the partition step.
    - Positivity of every reservoir mass and every physical scalar (sign
      guard).
    - ``T_magma`` jump between consecutive iters bounded by 1000 K (catches
      a runaway entropy-solver step).
    - ``Phi_global`` in [0, 1].
    - mass conservation + stability cross-cutting helpers.

    Runtime budget: ~60-120 s. The first aragog call dominates wall time
    via the EOS table load + entropy IC build; subsequent steps are
    seconds.
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
        interior_energetics__module='aragog',
        # Use the real EOS table path. With interior_struct.module='dummy'
        # and eos_dir=None the wrapper's else-branch falls back to the
        # legacy 1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa tree
        # shipped in FWL_DATA. const_properties=True is the alternative
        # but its default S_ref + T_ref do not match the CMB-anchored
        # IC, producing T_surf ~1.5e6 K on this dummy structure.
        # melting_dir is required when interior_struct is not zalmoxis;
        # Monteux-600 ships under FWL_DATA/interior_lookup_tables/Melting_curves/.
        interior_struct__melting_dir='Monteux-600',
        outgas__module='calliope',
    )

    hf = runner.hf_all
    assert hf is not None, 'helpfile should be created'
    assert len(hf) >= 2, f'expected >= 2 rows, got {len(hf)}'

    final = hf.iloc[-1]

    # Per-element mass closure. CALLIOPE partitions volatiles between
    # reservoirs but the total is the user-supplied budget; the sum is
    # the discriminator for any prefactor or unit-conversion bug.
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

    # Cross-step continuity of T_magma. A jump > 1000 K between
    # consecutive iters on this scenario implies the entropy solver
    # took a wildly out-of-range step.
    if 'T_magma' in hf.columns and len(hf) >= 2:
        dT = np.diff(hf['T_magma'].to_numpy())
        assert np.all(np.abs(dT) < 1000.0), (
            f'T_magma jump too large: max(|dT|)={np.max(np.abs(dT)):.1f} K'
        )

    # Conservation + stability cross-cutting helpers.
    mass_results = validate_mass_conservation(hf, tolerance=0.2)
    assert mass_results.get('masses_positive', True), 'mass reservoirs negative'

    stability = validate_stability(hf, max_temp=1e6, max_pressure=1e10)
    assert stability['temps_stable'], 'temperature instability detected'
    assert stability['pressures_stable'], 'pressure instability detected'
