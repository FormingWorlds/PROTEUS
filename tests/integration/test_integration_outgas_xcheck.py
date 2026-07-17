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
#20.

The test sweeps the fO2 axis across three scenarios so the cross-backend
ratio is checked along the redox dimension that the docs identify as the
primary driver of divergence:

- ``earth_IWp2``: nominal Earth anchor, IW+2.
- ``reducing_IWm2``: IW-2, exercises the reducing branch of the chemistry
  where H2 dominates over H2O.
- ``oxidising_IWp4``: IW+4, exercises the upper-oxidation branch.

A factor-of-3 (~0.48 dex) window is pinned on the P_surf ratio in each
scenario. The window is ~1.6x the documented worst-case (0.3 dex at 3000 K
in Figure 3 panel b) and ~2.8x the observed ~0.17 dex at the nominal
IW+2 fiducial. A regression that swapped CALLIOPE back to the O'Neill
buffer (5-10x ratio at hot oxidising conditions), or that introduced a
sign / unit error in either backend, would push the ratio outside the
window.

Per ``proteus-tests.md`` §1 the file also includes an error-contract test
exercising the outgas module schema validator.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
- CALLIOPE docs/Explanations/cross_backend_comparison.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip('atmodeller')

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


@dataclass(frozen=True)
class _XcheckScenario:
    """Per-scenario parametrize input for the calliope-vs-atmodeller xcheck."""

    name: str
    fO2_shift_IW: float
    H_budget: float


_SCENARIOS = (
    _XcheckScenario(name='earth_IWp2', fO2_shift_IW=2.0, H_budget=3.0e3),
    _XcheckScenario(name='reducing_IWm2', fO2_shift_IW=-2.0, H_budget=3.0e3),
    _XcheckScenario(name='oxidising_IWp4', fO2_shift_IW=4.0, H_budget=3.0e3),
)


def _run_with_backend(
    proteus_multi_timestep_run,
    backend: str,
    scenario: _XcheckScenario,
    **extra,
):
    """Helper: run a 2-step PROTEUS sim with the requested outgas backend."""
    overrides = dict(extra)
    return proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
        outgas__module=backend,
        outgas__fO2_shift_IW=scenario.fO2_shift_IW,
        planet__elements__H_budget=scenario.H_budget,
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
@pytest.mark.parametrize('scenario', _SCENARIOS, ids=lambda s: s.name)
def test_calliope_atmodeller_cross_consistency(proteus_multi_timestep_run, scenario):
    """Same IC, two backends, factor-of-3 P_surf agreement across fO2 axis.

    For each scenario, run PROTEUS twice: once with outgas=calliope (Fischer
    2011 buffer by default) and once with outgas=atmodeller (Hirschmann
    composite). Compare the final surface pressure of both runs.

    Verifies per scenario:
    - Both runs complete and produce >= 2 helpfile rows.
    - Both P_surf are finite, positive, and below 1 Mbar (sign + scale).
    - The ratio of the two P_surf values is within [1/3, 3] (~0.48 dex).
      The window is derived from the CALLIOPE cross_backend_comparison
      doc and gives ~1.6x headroom over the documented hot-end (0.3 dex
      at 3000 K) and ~2.8x over the observed ~0.17 dex at the IW+2
      fiducial. Holding the bound across reducing, nominal, and oxidising
      scenarios checks that the agreement is robust to the redox axis
      that drives most of the documented divergence.
    - Per-element mass closure holds INDEPENDENTLY in each run (a per-run
      bookkeeping bug would slip through the cross-backend ratio).
    - Sign guards on both P_surf separately, so a zero-result regression
      in one backend cannot hide behind a small ratio.

    Runtime budget: ~30-60 s per scenario (calliope ~5 s, atmodeller
    first-call ~15-30 s with JAX compile then ~5 s thereafter).
    """
    runner_c = _run_with_backend(proteus_multi_timestep_run, 'calliope', scenario)
    runner_a = _run_with_backend(
        proteus_multi_timestep_run,
        'atmodeller',
        scenario,
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
    # P_surf in either run. A 0 / x = 0 result would fail the ratio
    # bound below too, but the explicit sign guard makes intent clear
    # and surfaces the failure on the right line.
    assert p_c > 0, f'calliope P_surf non-positive: {p_c:.3e}'
    assert p_a > 0, f'atmodeller P_surf non-positive: {p_a:.3e}'
    assert np.isfinite(p_c) and np.isfinite(p_a), 'P_surf non-finite in one run'

    # Scale guard: both runs land below 1 Mbar at this dummy IC.
    assert p_c < 1e10, f'calliope P_surf above 1 Mbar: {p_c:.3e}'
    assert p_a < 1e10, f'atmodeller P_surf above 1 Mbar: {p_a:.3e}'

    # Cross-backend agreement: factor-of-3 (~0.48 dex). See file docstring
    # for the derivation; the bound holds across the fO2 axis because the
    # documented divergence (~0.3 dex worst case) is driven by the buffer
    # choice and the S2 solubility law, both of which scale with redox
    # but stay inside the window across [IW-2, IW+4] at T_magma ~ 4000 K.
    ratio = p_a / p_c
    assert (1.0 / 3.0) < ratio < 3.0, (
        f'cross-backend P_surf disagreement out of window: '
        f'p_calliope={p_c:.3e}, p_atmodeller={p_a:.3e}, ratio={ratio:.3f} '
        f'(log10 = {np.log10(ratio):+.3f} dex) at fO2_shift_IW={scenario.fO2_shift_IW}'
    )

    # Per-element closure must hold INDEPENDENTLY in each backend; a
    # bookkeeping bug in one run would slip through the cross-backend ratio.
    _per_element_closure(final_c, rel=2e-2)
    _per_element_closure(final_a, rel=2e-2)


# ---------------------------------------------------------------------------
# Error-contract path per proteus-tests.md §1 clause 2.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_outgas_module_validator_rejects_unknown_backend():
    """Outgas ``module`` schema validator rejects backends outside the
    documented {calliope, atmodeller, dummy} enum.

    Contract from ``src/proteus/config/_outgas.py:158-160``:
        ``Outgas.module`` must be in ``('calliope', 'atmodeller', 'dummy')``.

    Verifies:
    - ``module='unknown'`` raises ValueError at attrs validator time, BEFORE
      any module dispatch or hf_row write.
    - The three known-good values round-trip without raising, so a
      regression that broke the validator into raising on every input
      is not masked.
    - The default value is inside the enum (catches a stale-default
      regression that would otherwise only surface when the test fixture
      hits a default-construction path).
    """
    from proteus.config._outgas import Outgas

    with pytest.raises(ValueError, match=r'(?i)module'):
        Outgas(module='unknown')

    # Discrimination: confirm the three documented values DO NOT raise.
    for known in ('calliope', 'atmodeller', 'dummy'):
        o = Outgas(module=known)
        assert o.module == known

    # Discrimination: confirm the default is inside the enum.
    default = Outgas()
    assert default.module in ('calliope', 'atmodeller', 'dummy'), (
        f'default outgas module unexpectedly outside enum: {default.module!r}'
    )
