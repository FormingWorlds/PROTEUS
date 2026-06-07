"""
Conservation-invariant assertions for PROTEUS smoke tests.

Centralised so every smoke test (existing + future hypothesis-parametrised
ones) checks the same physics surface. Each assertion picks a discriminating
condition and includes diagnostic context in its failure message so a
regression points at the broken invariant by name and value, not at a bare
`assert False`.

Philosophy (per .github/.claude/rules/proteus-tests.md and .github/.claude/rules/proteus-code-review.md):

- Conservation invariants hold for ANY valid simulation, regardless of
  which modules are active. They are the strongest class of assertion
  because they catch bugs in code paths the smoke test was not specifically
  designed to exercise.
- Per-element mass closure (e_kg_total ≈ e_kg_atm + e_kg_solid + e_kg_liquid)
  is the symmetric form of issue #677's M_atm > M_planet bug; if a future
  refactor re-introduces the H/C/N/S/O asymmetry one of these checks fires.
- Whole-planet closure (M_planet = M_int + M_ele) is a cross-module
  consistency check. M_int comes from the structure module; M_ele comes
  from element bookkeeping. They must agree.
- The escape ≤ atmospheric mass check uses the per-step dt to bound the
  cumulative loss. It catches escape rates that are unphysically large
  (e.g., >> M_atm / dt would indicate a sign or unit bug).
- All checks tolerate `M_planet == 0` or `M_atm == 0` as "uninitialised";
  these are valid pre-IC states and asserting on them would false-alarm.

Relationship to `tests/integration/conftest.py`:

- `validate_mass_conservation` in the integration conftest is a softer,
  scope-narrower check kept for the multi-timestep integration tests.
  It asserts finiteness and positivity but tolerates any tolerance
  violation. It is not a substitute for the strict per-element closure
  here.
- This helper is the strict version: every assertion fires on a
  meaningful tolerance violation. New smoke and integration tests
  should call `assert_smoke_conservation_invariants` (this module) for
  conservation checks, and use the conftest helpers only for the
  fixture mechanics they bundle.

Underscore-prefixed module name keeps pytest from auto-collecting this
file as a test module (it has no `test_*` functions).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from proteus.utils.constants import element_list, gas_list, secs_per_year

# Tolerance defaults. Chosen so they admit float-rounding noise from the
# per-species sums but not any physically meaningful drift.
DEFAULT_REL_TOL = 1e-6
DEFAULT_ABS_TOL_KG = 1.0  # kg; some bookkeeping fields legitimately go to 0.0


# ---------------------------------------------------------------------------
# Finiteness and sign
# ---------------------------------------------------------------------------
def assert_no_nan_inf(hf_row: pd.Series, columns: list[str] | None = None) -> None:
    """Every numeric column in `hf_row` (or in the given subset) is finite.

    NaN and Inf in a helpfile column always indicate a bug: a divide-by-zero,
    a missing fallback, or a propagation from an upstream solver that did not
    converge. Smoke tests should never accept either.
    """
    if columns is None:
        columns = [c for c, v in hf_row.items() if isinstance(v, (int, float, np.floating))]
    bad = []
    for c in columns:
        v = hf_row.get(c)
        if v is None:
            continue
        # Strings/objects pass through; only check numerics.
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            bad.append(f'{c}=NaN')
        elif math.isinf(f):
            bad.append(f'{c}=Inf')
    assert not bad, f'Non-finite helpfile values: {", ".join(bad)}'


# Temperatures that are always meaningful in any simulation. Module-specific
# fields (T_solvus, T_cmb, T_cmb_initial, etc.) are excluded; they are
# legitimately 0.0 when the relevant module/feature is inactive. Add to this
# list only if the field is required for every smoke run.
ALWAYS_POSITIVE_TEMPS = ('T_surf', 'T_magma', 'T_star')


def assert_temperatures_positive(hf_row: pd.Series, columns: list[str] | None = None) -> None:
    """Every required temperature column is strictly > 0 Kelvin.

    T = 0 K is unphysical for required state fields. T < 0 K indicates a sign
    or unit bug. Catches the easy class of "computed in Celsius, written as
    Kelvin" mistakes. Module-specific temperatures (e.g., T_solvus, T_cmb,
    T_cmb_initial) that legitimately remain 0 when their module is inactive
    are excluded from the default surface; pass an explicit `columns` list
    if a smoke test needs to assert one of those.
    """
    if columns is None:
        columns = list(ALWAYS_POSITIVE_TEMPS)
    bad = []
    for c in columns:
        v = hf_row.get(c)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(f):
            continue  # caught by assert_no_nan_inf
        if f <= 0:
            bad.append(f'{c}={f:.3e} K')
    assert not bad, f'Non-positive required temperatures: {", ".join(bad)}'


def assert_pressures_non_negative(hf_row: pd.Series, columns: list[str] | None = None) -> None:
    """Every pressure column is >= 0 bar.

    Strictly positive only when the atmosphere is non-empty; an empty
    atmosphere (e.g. desiccated, dummy w/ no volatiles) legitimately has
    P_surf = 0. Negative pressures are always a bug.
    """
    if columns is None:
        columns = [
            c for c in hf_row.index if c.endswith('_bar') or c == 'P_surf' or c == 'p_xuv'
        ]
    bad = []
    for c in columns:
        v = hf_row.get(c)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(f):
            continue
        if f < 0:
            bad.append(f'{c}={f:.3e} bar')
    assert not bad, f'Negative pressures: {", ".join(bad)}'


# ---------------------------------------------------------------------------
# Mass closure (per-element and whole-planet)
# ---------------------------------------------------------------------------
def assert_per_element_mass_closure(
    hf_row: pd.Series,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol_kg: float = DEFAULT_ABS_TOL_KG,
) -> None:
    """For every tracked element e: e_kg_total ≈ e_kg_atm + e_kg_solid + e_kg_liquid.

    This is the per-element symmetric form of the issue #677 invariant. If
    a future refactor re-introduces an O-skip in any aggregation site, the
    O-row check fires. Same with the C, N, S analogues if any of them are
    later added to the asymmetric set.

    Tolerance: max(rel_tol * |total|, abs_tol_kg) to handle the all-zeros
    case (an element not present in this simulation).
    """
    bad = []
    for e in element_list:
        total = hf_row.get(f'{e}_kg_total')
        if total is None:
            continue
        atm = float(hf_row.get(f'{e}_kg_atm', 0.0))
        sol = float(hf_row.get(f'{e}_kg_solid', 0.0))
        liq = float(hf_row.get(f'{e}_kg_liquid', 0.0))
        total_f = float(total)
        if not math.isfinite(total_f):
            continue
        partition_sum = atm + sol + liq
        tol = max(rel_tol * abs(total_f), abs_tol_kg)
        if abs(partition_sum - total_f) > tol:
            bad.append(
                f'{e}: total={total_f:.3e}, '
                f'atm+solid+liquid={partition_sum:.3e}, '
                f'diff={partition_sum - total_f:+.3e} kg (tol {tol:.3e})'
            )
    assert not bad, (
        'Per-element mass closure violated; one of the partition sites '
        'is dropping the element silently:\n  ' + '\n  '.join(bad)
    )


def assert_per_species_mass_closure(
    hf_row: pd.Series,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol_kg: float = DEFAULT_ABS_TOL_KG,
) -> None:
    """For every gas species s: s_kg_total ≈ s_kg_atm + s_kg_solid + s_kg_liquid.

    Same shape as the per-element check but for molecular species. Catches
    bookkeeping bugs in the outgassing/dissolution path.
    """
    bad = []
    for s in gas_list:
        total = hf_row.get(f'{s}_kg_total')
        if total is None:
            continue
        atm = float(hf_row.get(f'{s}_kg_atm', 0.0))
        sol = float(hf_row.get(f'{s}_kg_solid', 0.0))
        liq = float(hf_row.get(f'{s}_kg_liquid', 0.0))
        total_f = float(total)
        if not math.isfinite(total_f):
            continue
        partition_sum = atm + sol + liq
        tol = max(rel_tol * abs(total_f), abs_tol_kg)
        if abs(partition_sum - total_f) > tol:
            bad.append(
                f'{s}: total={total_f:.3e}, atm+solid+liquid={partition_sum:.3e}, '
                f'diff={partition_sum - total_f:+.3e} kg (tol {tol:.3e})'
            )
    assert not bad, (
        'Per-species mass closure violated; one of the outgassing partition '
        'sites is dropping the species:\n  ' + '\n  '.join(bad)
    )


def assert_atmosphere_element_sum_matches_M_atm(
    hf_row: pd.Series, rel_tol: float = 1e-3
) -> None:
    """sum over tracked elements of e_kg_atm ≈ M_atm.

    Cross-check between the per-element bookkeeping and the M_atm scalar.
    If they disagree, either an aggregation loop is missing an element or
    M_atm is being computed via a different path (issue #677 root cause).

    The default tolerance (0.1%) is looser than the per-element closure
    check because M_atm is a separately-computed scalar that accumulates
    small float-rounding drift across iterations. The issue #677 failure
    mode produced > 70% disagreement, so the looser tolerance still
    catches real regressions while ignoring numerical noise.
    """
    M_atm = float(hf_row.get('M_atm', 0.0))
    if M_atm <= 0.0:
        return
    elem_sum = sum(float(hf_row.get(f'{e}_kg_atm', 0.0)) for e in element_list)
    rel = abs(elem_sum - M_atm) / M_atm
    assert rel < rel_tol, (
        f'M_atm={M_atm:.3e} kg disagrees with sum over elements '
        f'({elem_sum:.3e} kg) by {rel * 100:.4f}% (tol {rel_tol * 100:.4f}%). '
        f'A per-element kg_atm field is stale or M_atm is computed via a '
        f'different path.'
    )


def assert_element_sum_matches_species_sum(hf_row: pd.Series, rel_tol: float = 1e-3) -> None:
    """sum over elements of e_kg_atm ≈ sum over species of s_kg_atm.

    Cross-tree consistency: PROTEUS maintains both per-element and
    per-species mass trees. A bug in the species-to-element distribution
    step (e.g., O atoms in H2O mis-allocated to atomic O budget) could
    leave each tree internally consistent (per-element closure passes,
    per-species closure passes) while the two trees disagree with each
    other. This check catches the cross-tree drift directly, closing the
    gap that the per-element and per-species closure checks leave open.
    """
    elem_sum = sum(float(hf_row.get(f'{e}_kg_atm', 0.0)) for e in element_list)
    spec_sum = sum(float(hf_row.get(f'{s}_kg_atm', 0.0)) for s in gas_list)
    if elem_sum <= 0.0 and spec_sum <= 0.0:
        return
    denom = max(elem_sum, spec_sum)
    rel = abs(elem_sum - spec_sum) / denom
    assert rel < rel_tol, (
        f'Element sum ({elem_sum:.3e} kg) disagrees with species sum '
        f'({spec_sum:.3e} kg) by {rel * 100:.4f}% (tol {rel_tol * 100:.4f}%). '
        f'A species-to-element mass distribution step is dropping or '
        f'mis-allocating mass between the two bookkeeping trees.'
    )


def assert_M_atm_le_M_planet(hf_row: pd.Series, rel_tol: float = DEFAULT_REL_TOL) -> None:
    """M_atm <= M_planet, the issue #677 hard invariant."""
    M_atm = float(hf_row.get('M_atm', 0.0))
    M_planet = float(hf_row.get('M_planet', 0.0))
    if M_planet <= 0.0:
        return
    assert M_atm <= M_planet * (1.0 + rel_tol), (
        f'M_atm={M_atm:.3e} kg exceeds M_planet={M_planet:.3e} kg '
        f'by {(M_atm / M_planet - 1) * 100:.4f}% (issue #677 regression). '
        f'Likely cause: an aggregation site re-introduced the '
        f'"if e == \'O\': continue" skip.'
    )


def assert_M_planet_matches_M_int_plus_M_ele(hf_row: pd.Series, rel_tol: float = 1e-3) -> None:
    """M_planet ≈ M_int + M_ele (whole-planet element + interior closure).

    The structure module computes M_int. Element bookkeeping computes M_ele.
    They must agree on M_planet. A 0.1% tolerance admits the small
    bookkeeping noise from atmosphere not being part of M_int (since M_int
    is the structure mass below the surface). Tighten as PROTEUS gets more
    consistent here.
    """
    M_int = float(hf_row.get('M_int', 0.0))
    M_ele = float(hf_row.get('M_ele', 0.0))
    M_planet = float(hf_row.get('M_planet', 0.0))
    if M_planet <= 0.0:
        return
    expected = M_int + M_ele
    rel = abs(expected - M_planet) / M_planet
    assert rel < rel_tol, (
        f'M_planet={M_planet:.3e} kg does not match M_int + M_ele = '
        f'{expected:.3e} kg (rel diff {rel * 100:.4f}%, tol {rel_tol * 100:.4f}%). '
        f'Either the structure module and element bookkeeping disagree on '
        f'where mass lives, or M_atm has been double-counted.'
    )


# ---------------------------------------------------------------------------
# Escape bound
# ---------------------------------------------------------------------------
def assert_escape_within_atmospheric_budget(
    hf_row: pd.Series, dt_s: float | None = None, slack_factor: float = 10.0
) -> None:
    """esc_rate_total * dt < slack_factor * M_atm.

    A finite-step escape cannot remove more than the atmosphere contains.
    The slack_factor (default 10x) is generous: PROTEUS smoke tests use
    overridden tsurf_init / dummy modules that may produce unrealistic
    escape rates. The check still catches unphysically huge rates (e.g.
    sign-flip bugs producing ~M_atm / second).

    When dt_s is None, falls back to checking that esc_rate_total is finite
    and non-negative.
    """
    esc = float(hf_row.get('esc_rate_total', 0.0))
    M_atm = float(hf_row.get('M_atm', 0.0))
    assert math.isfinite(esc), f'esc_rate_total = {esc} (not finite)'
    assert esc >= 0.0, f'esc_rate_total = {esc:.3e} kg/s (negative)'
    if dt_s is None or M_atm <= 0.0:
        return
    cumulative_loss = esc * dt_s
    upper = slack_factor * M_atm
    assert cumulative_loss <= upper, (
        f'Escape over one step ({cumulative_loss:.3e} kg) exceeds '
        f'{slack_factor:.0f}x atmospheric mass ({M_atm:.3e} kg, dt={dt_s:.3e} s). '
        f'Likely a sign or unit bug in the escape backend.'
    )


# ---------------------------------------------------------------------------
# Composite check
# ---------------------------------------------------------------------------
def assert_smoke_conservation_invariants(hf_all: pd.DataFrame) -> None:
    """Run every conservation-invariant check on the FINAL row of hf_all.

    Single entry point so smoke tests can call this once and get every
    invariant in one assertion path. The dt for the escape bound is taken
    from the last two helpfile rows (Time delta in years, converted to
    seconds); if there is only one row, dt is None and the escape check
    falls back to finiteness only.
    """
    assert len(hf_all) > 0, 'helpfile is empty'
    final_row = hf_all.iloc[-1]

    # dt in seconds, from the last two helpfile rows. Time is in years.
    # Use the canonical PROTEUS constant so any future change to the
    # year-length convention propagates here automatically.
    dt_s = None
    if len(hf_all) >= 2:
        dt_yr = float(final_row['Time']) - float(hf_all.iloc[-2]['Time'])
        if dt_yr > 0:
            dt_s = dt_yr * secs_per_year

    assert_no_nan_inf(final_row)
    assert_temperatures_positive(final_row)
    assert_pressures_non_negative(final_row)
    assert_per_element_mass_closure(final_row)
    assert_per_species_mass_closure(final_row)
    assert_atmosphere_element_sum_matches_M_atm(final_row)
    assert_element_sum_matches_species_sum(final_row)
    assert_M_atm_le_M_planet(final_row)
    assert_M_planet_matches_M_int_plus_M_ele(final_row)
    assert_escape_within_atmospheric_budget(final_row, dt_s=dt_s)
