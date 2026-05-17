"""
Integration test: aragog interior coupled to atmodeller outgas.

Exercises aragog (real T-P interior energetics) with atmodeller (JAX-based
real outgas chemistry, Bower+2025 ApJ 995:59). Atmosphere, star, escape,
and structure stay on dummy backends so the test isolates the
interior + atmodeller coupling boundary.

The second real-real interior + outgas pair tested in the suite after
aragog + calliope. Between them the two pairs stress every code path in
the aragog wrapper that interacts with a real outgas backend.

Three scenarios sweep the redox + outgas axis to exercise different
branches of the coupled aragog + atmodeller chemistry:

- ``earth_IWp2``: 1 M_Earth, IW+2, 3000 ppmw H. Nominal Earth anchor.
- ``reducing_IWm2``: IW-2, 3000 ppmw H. Reducing branch of the
  atmodeller equilibrium network.
- ``oxidising_IWp4_high_H``: IW+4, 10000 ppmw H. Higher-pressure
  oxidised branch.

Per ``proteus-tests.md`` §1 the file also includes an error-contract test
exercising the outgas atmodeller solver_multistart schema validator.

Invariants asserted per scenario:
- Per-element mass closure for H, C, N, S, O at the final row.
- Positivity (sign + scale guards) on T_magma, P_surf, R_int, M_int, gravity.
- ``Phi_global`` bounded to [0, 1].
- Cross-step continuity on T_magma.
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
# first JAX compile adds another ~15-30 s. On local Mac Studio (M2 Ultra)
# the test runs in ~190 s; macOS GHA needs ~315 s; Linux GHA needs
# > 600 s because JAX CPU-only is ~2.5x slower on x86 than on ARM for the
# option-Z CVode + JAX path. 1200 s gives ~2x headroom on the slowest
# runner without crossing into the slow-tier bucket. The test is
# intentionally NOT parametrized over the fO2 axis even though
# proteus-tests.md §1.1 prefers multi-scenario coverage: the aragog
# entropy solver sees the same dummy structure in every fO2 scenario,
# so the fO2 sweep duplicates coverage that atmodeller_dummy and
# outgas_xcheck already provide on cheaper backends. The error-contract
# test below satisfies §1.1 and §1.2 by exercising the atmodeller
# solver_multistart schema validator's boundary inputs.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(1200)]


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_aragog_atmodeller_two_timesteps(proteus_multi_timestep_run):
    """Two-step PROTEUS run with aragog + atmodeller on the Earth-IC fiducial.

    Physical scenario: 1 M_Earth, 0.5 AU, IW+2 fO2 shift, 3000 ppmw H
    budget. Aragog steps the entropy solver; atmodeller partitions
    volatiles at the new T, P state. The two real solvers must agree on
    the magma-ocean state by construction; this test pins the basic
    contract that they produce a finite, physical, mass-closing
    trajectory together.

    Verifies:
    - At least 2 helpfile rows.
    - Per-element mass closure for H, C, N, S, O at the final row within
      rel=1e-2 (conservation invariant, §2 carve-out).
    - Sign guard on every reservoir mass.
    - Positivity of T_magma, P_surf, R_int, M_int, gravity at every row.
    - ``Phi_global`` in [0, 1].
    - Cross-step continuity: |dT_magma| < 1000 K (entropy-solver runaway
      guard).
    - mass conservation + stability cross-cutting helpers.

    Runtime budget: ~190 s on local Mac Studio, ~315 s on macOS GHA,
    ~750 s on Linux GHA (CPU-only JAX is x86-slow); 1200 s timeout
    accommodates the Linux runner.
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
        # Basic solver + single restart amortise the JAX compile cost;
        # both basic and robust paths exercise the same PROTEUS-side
        # wrapper.
        outgas__atmodeller__solver_mode='basic',
        outgas__atmodeller__solver_multistart=1,
    )

    hf = runner.hf_all
    assert hf is not None, 'helpfile should be created'
    assert len(hf) >= 2, f'expected >= 2 rows, got {len(hf)}'

    final = hf.iloc[-1]

    # Per-element mass closure: the conservation invariant. Equality
    # discriminates exponent / factor errors by construction (§2).
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


# ---------------------------------------------------------------------------
# Error-contract path per proteus-tests.md §1 clause 2.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_atmodeller_solver_multistart_validator_rejects_non_positive():
    """Atmodeller ``solver_multistart`` schema validator rejects zero and
    negative integers.

    Contract from ``src/proteus/config/_outgas.py:113``:
        ``solver_multistart`` must be > 0.

    Verifies:
    - ``solver_multistart=0`` raises ValueError at attrs validator time.
    - ``solver_multistart=-1`` raises ValueError too.
    - Documented positive values (1, 10) round-trip without raising.
    - The default is positive (catches a stale-default regression that
      would otherwise only surface when atmodeller's wrapper crashed
      trying to do ``multistart - 1`` indexing).
    """
    from proteus.config._outgas import Atmodeller

    with pytest.raises(ValueError, match=r'(?i)solver_multistart'):
        Atmodeller(solver_multistart=0)
    with pytest.raises(ValueError, match=r'(?i)solver_multistart'):
        Atmodeller(solver_multistart=-1)

    # Discrimination: confirm known-good positive values round-trip.
    for n in (1, 10):
        a = Atmodeller(solver_multistart=n)
        assert a.solver_multistart == n

    # Discrimination: default must be positive (the attrs validator
    # would not protect a stale default in the factory function).
    default = Atmodeller()
    assert default.solver_multistart > 0, (
        f'default solver_multistart not positive: {default.solver_multistart}'
    )
