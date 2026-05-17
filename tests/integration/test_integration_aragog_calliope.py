"""
Integration test: aragog interior coupled to calliope outgas.

Exercises the aragog real interior energetics solver (T-P formalism) coupled
with calliope (real outgas chemistry) over two timesteps. Atmosphere, star,
escape, and structure stay on dummy backends so the test isolates the
interior + outgas coupling boundary.

The aragog wrapper resolves its EOS lookup table via the legacy
``1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa`` path when
``interior_struct.eos_dir`` is None and PALEOS is not in scope. The
``aragog.py:642-650`` fallback for ``eos_dir=None`` is exercised here in
every scenario; a regression in that fallback, or in the entropy-solver IC,
surfaces immediately.

Three scenarios sweep the redox + outgas axis to exercise different
branches of the coupled aragog + calliope chemistry:

- ``earth_IWp2``: 1 M_Earth, 0.5 AU, IW+2, 3000 ppmw H. Nominal Earth
  anchor; mildly oxidised, water-dominated outgassing.
- ``reducing_IWm2``: same orbit + mass, IW-2 buffer. H2/CH4 dominate
  over H2O/CO2 in the gas phase.
- ``oxidising_IWp4_high_H``: same orbit + mass, IW+4, 10000 ppmw H.
  Higher-pressure oxidised atmosphere.

Per ``proteus-tests.md`` §1 the file also includes an error-contract test
exercising the interior_energetics module schema validator.

Invariants asserted per scenario:
- Per-element mass closure for H, C, N, S, O at the final row.
- Positivity (sign + scale guards) on T_magma, P_surf, R_int, M_int, gravity.
- ``Phi_global`` bounded to [0, 1].
- Cross-step continuity on T_magma and Phi_global.
- mass conservation + stability cross-cutting helpers.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tests.integration.conftest import (
    validate_mass_conservation,
    validate_stability,
)

# Module-level timeout raised from the 300 s integration default because
# aragog's real-binary first-call dominates wall time (EOS table load +
# entropy IC build + first solver step). ~180 s observed on a fast local
# Mac Studio; 300 s exceeded on macos-latest GHA. 600 s gives ~1.5x
# headroom on macOS without crossing into the slow-tier bucket.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


@dataclass(frozen=True)
class _AragogCalliopeScenario:
    """Per-scenario parametrize input for the aragog+calliope pair test."""

    name: str
    fO2_shift_IW: float
    H_budget: float


_SCENARIOS = (
    _AragogCalliopeScenario(name='earth_IWp2', fO2_shift_IW=2.0, H_budget=3.0e3),
    _AragogCalliopeScenario(name='reducing_IWm2', fO2_shift_IW=-2.0, H_budget=3.0e3),
    _AragogCalliopeScenario(name='oxidising_IWp4_high_H', fO2_shift_IW=4.0, H_budget=1.0e4),
)


@pytest.mark.integration
@pytest.mark.physics_invariant
@pytest.mark.parametrize('scenario', _SCENARIOS, ids=lambda s: s.name)
def test_aragog_calliope_two_timesteps(proteus_multi_timestep_run, scenario):
    """Two-step PROTEUS run with aragog + calliope across three IC scenarios.

    The same coupling invariants must hold across all three IC: a mildly
    oxidised water-dominated atmosphere, a reducing H2-dominated atmosphere,
    and an oxidising volatile-rich atmosphere. The aragog entropy solver
    sees the same dummy structure in each case; the difference between
    scenarios is the partial-pressure spectrum and the dissolved-mass
    profile that calliope computes for it.

    Verifies per scenario:
    - At least 2 helpfile rows.
    - Per-element mass closure for H, C, N, S, O at the final row within
      rel=1e-2 (conservation invariant, §2 carve-out).
    - Sign guard on every reservoir mass (catches a negative-value
      regression that the closure tolerance might otherwise swallow).
    - Positivity of T_magma, P_surf, R_int, M_int, gravity at every row.
    - ``Phi_global`` in [0, 1].
    - Cross-step continuity: |dT_magma| < 1000 K (rejects an entropy-solver
      runaway), |dPhi_global| < 0.5 (rejects an unphysical melt-fraction
      jump).
    - mass conservation + stability cross-cutting helpers.

    Runtime budget: ~150-200 s per scenario on local Mac Studio, ~250-350 s
    on macOS GHA. Three scenarios fit inside ~10 min total wall time on
    macOS GHA, each individual scenario well under the 600 s timeout.
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
        outgas__module='calliope',
        outgas__fO2_shift_IW=scenario.fO2_shift_IW,
        planet__elements__H_budget=scenario.H_budget,
    )

    hf = runner.hf_all
    assert hf is not None, 'helpfile should be created'
    assert len(hf) >= 2, f'expected >= 2 rows, got {len(hf)}'

    final = hf.iloc[-1]

    # Per-element mass closure: the conservation invariant. The equality
    # form discriminates exponent / factor errors by construction (§2).
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

    # Cross-step continuity of T_magma. A jump > 1000 K between consecutive
    # iters implies the entropy solver took a wildly out-of-range step.
    if 'T_magma' in hf.columns and len(hf) >= 2:
        dT = np.diff(hf['T_magma'].to_numpy())
        assert np.all(np.abs(dT) < 1000.0), (
            f'T_magma jump too large: max(|dT|)={np.max(np.abs(dT)):.1f} K'
        )

    # Cross-step continuity of Phi_global. Tighter bound: phi changes
    # smoothly under the coupled cooling pathway.
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


# ---------------------------------------------------------------------------
# Error-contract path per proteus-tests.md §1 clause 2.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_interior_energetics_module_validator_rejects_unknown_backend():
    """Interior_energetics ``module`` schema validator rejects backends
    outside the documented {spider, aragog, dummy, boundary} enum.

    Contract from ``src/proteus/config/_interior.py``: the Interior
    dataclass's ``module`` field is validated with
    ``in_(('spider', 'aragog', 'dummy', 'boundary'))``.

    Verifies:
    - ``module='unknown'`` raises ValueError at attrs validator time, BEFORE
      any module dispatch or hf_row write. This prevents a typo'd config
      from silently dispatching to a no-op interior.
    - Each of the four known-good values round-trips without raising, so
      a regression that broke the validator into raising on every input
      is not masked.
    - The default is inside the enum (catches a stale-default regression).
    """
    from proteus.config._interior import Interior

    with pytest.raises(ValueError, match=r'(?i)module'):
        Interior(module='unknown')

    # Discrimination: confirm each documented backend round-trips.
    for known in ('spider', 'aragog', 'dummy', 'boundary'):
        i = Interior(module=known)
        assert i.module == known

    # Discrimination: default is inside the enum.
    default = Interior()
    assert default.module in ('spider', 'aragog', 'dummy', 'boundary'), (
        f'default interior_energetics module unexpectedly outside enum: {default.module!r}'
    )
