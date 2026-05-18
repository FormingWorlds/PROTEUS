"""
Slow-tier integration test: aragog interior coupled to atmodeller outgas.

Exercises the production aragog interior energetics solver (backend
``'jax'``, the schema default in ``config/_interior.py``) with
atmodeller (JAX-based real outgas chemistry, Bower+2025 ApJ 995:59).
Atmosphere, star, escape, and structure stay on dummy backends so the
test isolates the interior + atmodeller coupling boundary.

The second real-real interior + outgas pair tested in the suite after
aragog + calliope. Between them the two pairs stress every code path in
the aragog wrapper that interacts with a real outgas backend.

Runs in the nightly slow tier because Linux GHA needs > 1200 s for the
first aragog setup + first solver step + atmodeller JAX compile
combined on x86, whereas macOS GHA finishes the same test in ~440 s.
The 360 s setup phase on Linux (EOS table load + EntropySolver
construction inside the aragog library) plus the atmodeller JAX
compile dominate. Diagnostic timing in
``src/proteus/interior_energetics/aragog.py`` records per-phase wall
time on every nightly run when ``PROTEUS_CI_NIGHTLY=1``.

Invariants asserted:
- Per-element mass closure for H, C, N, S, O at the final row.
- Positivity (sign + scale guards) on T_magma, P_surf, R_int, M_int, gravity.
- ``Phi_global`` bounded to [0, 1].
- Cross-step continuity on T_magma.
- mass conservation + stability cross-cutting helpers.

The error-contract sibling test (atmodeller solver_multistart schema
validator) lives in ``test_integration_aragog_atmodeller.py`` at the
integration tier since it is a sub-second config-only check.

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

# Slow tier: this test sits in the nightly slow-tier file list in
# ``.github/workflows/ci-nightly.yml`` and is excluded from the PR-CI
# integration step (``pytest -m "integration and not slow"``). The 2400 s
# timeout sits above the macOS GHA wall time (~440 s) and the projected
# Linux GHA wall time (~1800-2200 s) with a margin; well under the
# slow-tier 3600 s budget cap from proteus-tests.md section 7.
pytestmark = [pytest.mark.slow, pytest.mark.timeout(2400)]


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_aragog_atmodeller_two_timesteps(proteus_multi_timestep_run):
    """Two-step PROTEUS run with aragog + atmodeller on the Earth-IC fiducial.

    Physical scenario: 1 M_Earth, 0.5 AU, IW+2 fO2 shift, 3000 ppmw H
    budget. Aragog steps the entropy solver (production default
    ``backend='jax'``: scipy-CVode with JAX-derived RHS and analytic
    Jacobian); atmodeller partitions volatiles at the new T, P state.
    The two real solvers must agree on the magma-ocean state by
    construction; this test pins the basic contract that they produce a
    finite, physical, mass-closing trajectory together.

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

    Runtime budget: ~190 s on local Mac Studio, ~440 s on macOS GHA,
    ~1800-2200 s on Linux GHA (the 360 s setup phase on Linux x86 is
    the dominant overhead). 2400 s timeout accommodates the Linux runner.
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

    # Production-path discrimination guard. The aragog wrapper installs a
    # JAX RHS + analytic-Jacobian factory on the CVODE solver when
    # backend='jax' (the production default). If the JAX import or pytree
    # construction silently failed, CVODE would still run on its FD
    # Jacobian fallback and the physics invariants below would pass for
    # the wrong reason. The counter is incremented exactly once per
    # CVODE solve(), so >= 1 proves the analytic-Jacobian factory was
    # consumed at least once during the two-step run.
    solver = runner.interior_o.aragog_solver
    assert solver is not None, 'aragog solver missing after run'
    n_factory_calls = getattr(solver, '_jax_factory_call_count', None)
    assert n_factory_calls is not None, (
        'JAX CVODE factory never installed on solver; backend may have '
        'silently fallen back to FD Jacobian'
    )
    assert n_factory_calls >= 1, (
        f'JAX CVODE factory installed but never invoked '
        f'(call_count={n_factory_calls}); production analytic-Jacobian '
        f'path was not exercised'
    )

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
