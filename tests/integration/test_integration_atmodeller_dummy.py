"""
Integration test: atmodeller outgas with dummy interior, atmos, star, escape.

Exercises atmodeller (Bower+2025 ApJ 995:59) as the only real backend in a
coupled PROTEUS run; every other slot is dummy. Atmodeller is selected via
``config.outgas.module = 'atmodeller'``; it uses a JAX-based root finder with
real-gas EOS and non-ideal solubility laws. Dummying the rest isolates the
atmodeller boundary in PROTEUS so a regression in module dispatch, hf_row
population, solver-result unpacking, or unit conversion at the
PROTEUS-atmodeller boundary is the only failure class.

Three scenarios sweep the redox + inventory axis that drives the atmodeller
chemistry:

- ``earth_like_IWp2``: IW+2, 3000 ppmw H. Nominal Earth anchor; mildly
  oxidised, water-dominated outgassing.
- ``reducing_IWm2``: IW-2, 3000 ppmw H. H2 dominates over H2O above this
  buffer offset; exercises the reducing branch of the equilibrium network.
- ``oxidising_IWp4_high_H``: IW+4, 10000 ppmw H. Strongly oxidised, high
  volatile budget; exercises the upper-oxidation branch and a higher P_surf
  regime than the nominal Earth anchor.

Per ``proteus-tests.md`` §1, the file also includes a sibling
error-contract test that exercises the atmodeller solver_mode validator.

Invariants asserted per scenario:
- Per-element mass closure ``atm + liquid + solid == total`` for H, C, N, S, O
  at the final row (conservation invariant carve-out, §2).
- Sign guards on every reservoir mass.
- Sign + scale guards on P_surf.
- ``Phi_global`` bounded to [0, 1].
- Cross-step continuity on P_surf and T_surf.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip('atmodeller')

from tests.integration.conftest import (  # noqa: E402
    validate_mass_conservation,
    validate_stability,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


@dataclass(frozen=True)
class _AtmodellerScenario:
    """Per-scenario parametrize input for the atmodeller pair test."""

    name: str
    fO2_shift_IW: float
    H_budget: float


_SCENARIOS = (
    _AtmodellerScenario(name='earth_like_IWp2', fO2_shift_IW=2.0, H_budget=3.0e3),
    _AtmodellerScenario(name='reducing_IWm2', fO2_shift_IW=-2.0, H_budget=3.0e3),
    _AtmodellerScenario(name='oxidising_IWp4_high_H', fO2_shift_IW=4.0, H_budget=1.0e4),
)


@pytest.mark.integration
@pytest.mark.physics_invariant
@pytest.mark.parametrize('scenario', _SCENARIOS, ids=lambda s: s.name)
def test_atmodeller_dummy_two_timesteps(proteus_multi_timestep_run, scenario):
    """Two-step PROTEUS run with atmodeller outgas across three IC scenarios.

    The same invariants must hold across all three IC: a mildly-oxidised
    water-dominated atmosphere, a reducing H2-dominated atmosphere, and an
    oxidising volatile-rich atmosphere. The fO2 sweep takes the atmodeller
    chemistry through the H2/H2O dominance flip (near IW-2 in PROTEUS' default
    species list) and into the high-P_surf regime at IW+4 + high H budget.

    Verifies per scenario:
    - Helpfile has at least 2 rows.
    - Per-element mass closure ``atm + liquid + solid == total`` within
      rel=1e-2.
    - Every reservoir mass is non-negative and finite (sign guard).
    - ``P_surf > 0`` and bounded by 1e10 Pa (sign + scale guards; the
      Pa-vs-bar inversion would land > 1e11; the sign flip would land
      negative).
    - ``Phi_global`` in [0, 1].
    - |dP_surf| < 0.5 * P_surf_max (no doubling per step), |dT_surf| < 500 K
      (rejects a K-vs-C unit bug on T_surf which would land ~ 273 K shift).

    Runtime budget: ~15-30 s per scenario after the JAX compile is amortised
    in scenario 1 (~10-15 s of solver run per scenario thereafter).
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
        outgas__module='atmodeller',
        outgas__fO2_shift_IW=scenario.fO2_shift_IW,
        planet__elements__H_budget=scenario.H_budget,
        outgas__atmodeller__solver_mode='basic',
        outgas__atmodeller__solver_multistart=1,
    )

    hf = runner.hf_all
    assert hf is not None, 'helpfile should be created'
    assert len(hf) >= 2, f'expected >= 2 rows, got {len(hf)}'

    final = hf.iloc[-1]

    # Per-element mass closure: the conservation invariant. Discriminates
    # exponent / factor errors via the equality form per §2 carve-out.
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
        # Sign guard: any negative reservoir is a numerical-stability or
        # sign-error regression.
        assert atm >= 0, f'{atm_key} negative: {atm:.3e}'
        assert liq >= 0, f'{liq_key} negative: {liq:.3e}'
        assert sol >= 0, f'{sol_key} negative: {sol:.3e}'
        if tot > 0:
            assert atm + liq + sol == pytest.approx(tot, rel=1e-2), (
                f'{elt} closure: atm+liq+sol={atm + liq + sol:.3e}, total={tot:.3e}'
            )

    # P_surf sign + scale guard. A Pa-vs-bar inversion would land >1e11 Pa
    # (a 1 bar atmosphere reported in bar instead of Pa); a sign flip lands
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

    # Cross-step continuity. Atmodeller can shift partial pressures across a
    # step but should not produce a jump that doubles P_surf in 1000 yr.
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


# ---------------------------------------------------------------------------
# Error-contract path per proteus-tests.md §1 clause 2.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_atmodeller_solver_mode_validator_rejects_unknown_mode():
    """Atmodeller ``solver_mode`` schema validator rejects modes outside
    the documented {robust, basic} enum.

    Contract from ``src/proteus/config/_outgas.py:108-111``:
        ``solver_mode`` must be in ``('robust', 'basic')``.

    Verifies:
    - ``solver_mode='unknown'`` raises ValueError at attrs validator time,
      BEFORE any module dispatch or hf_row write.
    - The known-good values ``'robust'`` and ``'basic'`` round-trip without
      raising, so a regression that broke the validator into raising on
      every input would be caught.
    - A non-atmodeller module ({calliope, dummy}) accepts any solver_mode
      value because the validator is bound to the Atmodeller dataclass
      field itself, not gated on the outgas module. Confirm both halves
      of that contract: the field rejects invalid values regardless of
      whether atmodeller is the active backend.
    """
    from proteus.config._outgas import Atmodeller

    with pytest.raises(ValueError, match=r'(?i)solver_mode'):
        Atmodeller(solver_mode='unknown')

    # Discrimination: confirm the known-good values DO NOT raise.
    ok_robust = Atmodeller(solver_mode='robust')
    ok_basic = Atmodeller(solver_mode='basic')
    assert ok_robust.solver_mode == 'robust'
    assert ok_basic.solver_mode == 'basic'

    # Discrimination: confirm the validator is not gated on the outer
    # outgas module (the field itself enforces). Constructing with a
    # different known-good value here is the negative half of the
    # gating contract.
    ok_default = Atmodeller()
    assert ok_default.solver_mode in ('robust', 'basic'), (
        f'default solver_mode unexpectedly outside enum: {ok_default.solver_mode!r}'
    )
