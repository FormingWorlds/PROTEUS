"""
Integration test: calliope vs atmodeller at the same initial condition.

Runs PROTEUS twice from an identical IC, swapping only the outgas backend
between calliope (Fischer 2011 IW buffer by default, fsolve-based, ideal
gas) and atmodeller (Hirschmann composite buffer, JAX-based, ideal gas
where the EOS options are toggled off). The CALLIOPE
``cross_backend_comparison`` docs page (Fig. 3, Fig. 4 bar 2) reports that
with the Fischer default the two backends agree to within ~0.16 dex in
ΔIW at Earth-like inputs and below ~0.3 dex in magnitude across the
1800-3000 K magma-ocean range; the previous O'Neill default had a much
larger ~1 dex gap that motivated the buffer-default flip in CALLIOPE PR
#20. This test pins a factor-of-3 window on the P_surf ratio, which is
~0.48 dex and gives ~1.6x headroom over the documented worst-case at
3000 K and ~2.8x headroom over the observed ~0.17 dex on the dummy IC.

A regression that swapped CALLIOPE back to O'Neill, or that introduced
a sign / unit error in either backend, would push the ratio outside the
factor-of-3 window.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- CALLIOPE docs/Explanations/cross_backend_comparison.md
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('atmodeller')

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


_BASE_OVERRIDES: dict[str, object] = {
    # Identical IC: 3000 ppmw H, C/H=1, 100 ppmw N, S/H=1, IW+2 buffer,
    # 0.5 AU dummy orbit (all from input/dummy.toml unless overridden).
    'outgas__fO2_shift_IW': 2.0,
}


def _run_with_backend(proteus_multi_timestep_run, backend: str, **extra):
    """Helper: run a 2-step PROTEUS sim with the requested outgas backend."""
    overrides = dict(_BASE_OVERRIDES)
    overrides.update(extra)
    return proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
        outgas__module=backend,
        **overrides,
    )


def _per_element_closure(hf_row, rel: float = 2e-2) -> None:
    """Assert per-element mass closure on the final row of a helpfile."""
    for elt in ('H', 'C', 'N', 'S', 'O'):
        atm_key = f'{elt}_kg_atm'
        liq_key = f'{elt}_kg_liquid'
        sol_key = f'{elt}_kg_solid'
        tot_key = f'{elt}_kg_total'
        if not all(k in hf_row for k in (atm_key, liq_key, sol_key, tot_key)):
            continue
        atm = float(hf_row[atm_key])
        liq = float(hf_row[liq_key])
        sol = float(hf_row[sol_key])
        tot = float(hf_row[tot_key])
        assert atm >= 0 and liq >= 0 and sol >= 0, f'{elt}: negative reservoir mass'
        if tot > 0:
            assert atm + liq + sol == pytest.approx(tot, rel=rel), (
                f'{elt}: closure broken (sum={atm + liq + sol:.3e}, total={tot:.3e})'
            )


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_calliope_atmodeller_cross_consistency(proteus_multi_timestep_run):
    """Same IC, two backends, Fischer-buffer-tight consistency.

    Physical scenario: Earth-mass, IW+2 fO2 shift, 3000 ppmw H budget. Run
    PROTEUS twice; once with outgas=calliope (Fischer 2011 buffer by
    default) and once with outgas=atmodeller (Hirschmann composite).
    Compare the final surface pressure of both runs.

    Verifies:
    - Both runs complete and produce >= 2 helpfile rows.
    - Both P_surf are finite, positive, and below 1 Mbar (1e10 Pa).
    - The ratio of the two P_surf values is within [1/3, 3] (≈0.48 dex).
      Bound derived from the CALLIOPE ``cross_backend_comparison`` doc:
      raw ΔIW gap with Fischer default is 0.16 dex at canonical Earth
      fiducial, and below 0.3 dex everywhere across 1800-3000 K. The
      P_surf ratio inherits the same scale because partial pressures
      track the buffered chemistry. Factor-of-3 leaves ~1.6x headroom
      over the documented worst-case at 3000 K and ~2.8x over the
      observed ~0.17 dex at this dummy IC.
    - Per-element mass closure holds in both runs independently.
    - Discrimination: ``p_calliope > 0`` and ``p_atmodeller > 0`` are
      asserted separately so a regression that zeroed out one backend's
      result cannot be hidden by the ratio guard.

    Runtime budget: ~30-60 s for the pair (calliope ~5 s, atmodeller
    first-call ~15-30 s including JAX compile, second-call ~1 s).
    """
    runner_c = _run_with_backend(proteus_multi_timestep_run, 'calliope')
    runner_a = _run_with_backend(
        proteus_multi_timestep_run,
        'atmodeller',
        outgas__atmodeller__solver_mode='basic',
        outgas__atmodeller__solver_multistart=1,
    )

    hf_c = runner_c.hf_all
    hf_a = runner_a.hf_all

    assert hf_c is not None and hf_a is not None, 'both runs must produce a helpfile'
    assert len(hf_c) >= 2, f'calliope run wrote {len(hf_c)} rows (<2)'
    assert len(hf_a) >= 2, f'atmodeller run wrote {len(hf_a)} rows (<2)'

    final_c = hf_c.iloc[-1]
    final_a = hf_a.iloc[-1]

    p_c = float(final_c['P_surf'])
    p_a = float(final_a['P_surf'])

    # Sign guard (both backends): catches a regression that zeroed out
    # P_surf in either run. The ratio assertion below would silently pass
    # if p_a == 0 (ratio = inf would just fail; ratio = 0/x = 0 would
    # fail too) — but a positive sign-guard makes intent explicit.
    assert p_c > 0, f'calliope P_surf non-positive: {p_c:.3e}'
    assert p_a > 0, f'atmodeller P_surf non-positive: {p_a:.3e}'
    assert np.isfinite(p_c) and np.isfinite(p_a), 'P_surf non-finite in one run'

    # Scale guard: both runs must land below 1 Mbar at this IC.
    assert p_c < 1e10, f'calliope P_surf above 1 Mbar: {p_c:.3e}'
    assert p_a < 1e10, f'atmodeller P_surf above 1 Mbar: {p_a:.3e}'

    # Cross-backend agreement: factor-of-3 (~0.48 dex). Derived from
    # the CALLIOPE cross_backend_comparison doc: with the Fischer 2011
    # default buffer the raw ΔIW gap against atmodeller is 0.16 dex at
    # canonical Earth fiducial, climbing to ~0.3 dex only at 3000 K
    # (Figure 3 panel b). The observed log10(p_atmodeller / p_calliope)
    # on this dummy IC is ~0.17 dex; factor-of-3 leaves enough headroom
    # for the documented hot-end envelope while still discriminating a
    # regression that swapped CALLIOPE back to the O'Neill buffer (which
    # would push the ratio to 5x-10x at 2000-3000 K) or that introduced
    # a sign / unit error in either backend.
    ratio = p_a / p_c
    assert (1.0 / 3.0) < ratio < 3.0, (
        f'cross-backend P_surf disagreement out of window: '
        f'p_calliope={p_c:.3e}, p_atmodeller={p_a:.3e}, ratio={ratio:.3f} '
        f'(log10 = {np.log10(ratio):+.3f} dex)'
    )

    # Per-element closure must hold INDEPENDENTLY in each backend; a
    # bookkeeping bug in one run alone would slip through the cross-
    # backend ratio.
    _per_element_closure(final_c, rel=2e-2)
    _per_element_closure(final_a, rel=2e-2)
