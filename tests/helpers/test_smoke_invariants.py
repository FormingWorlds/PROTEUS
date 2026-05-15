"""
Unit tests for the conservation-invariant assertion helper.

Each helper function in `tests/helpers/_smoke_invariants.py` is exercised
here with a synthetic helpfile row that either satisfies the invariant
(the function must NOT raise) or violates it (the function MUST raise
with a specific message). Without these tests, an off-by-one in a
tolerance formula or a logic inversion in the comparison would only
surface the day a real conservation regression slips through a smoke
test.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from _smoke_invariants import (
    assert_atmosphere_element_sum_matches_M_atm,
    assert_element_sum_matches_species_sum,
    assert_escape_within_atmospheric_budget,
    assert_M_atm_le_M_planet,
    assert_M_planet_matches_M_int_plus_M_ele,
    assert_no_nan_inf,
    assert_per_element_mass_closure,
    assert_per_species_mass_closure,
    assert_pressures_non_negative,
    assert_smoke_conservation_invariants,
    assert_temperatures_positive,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Builder for synthetic, fully-self-consistent helpfile rows
# ---------------------------------------------------------------------------
def _good_row() -> pd.Series:
    """Return a synthetic helpfile row that satisfies every invariant.

    Numbers are chosen to be at the right physical scale (Earth-like) so
    that any tolerance bug surfaces with a meaningful magnitude rather than
    in the noise. Asymmetric so a transposition or off-by-one in any helper
    produces a different value at every check.
    """
    M_planet = 5.972e24
    M_int = 5.96e24
    M_ele = 1.2e22
    M_atm = 8.0e21
    # Per-element atmospheric mass: H + O + C dominate.
    H_atm, O_atm, C_atm = 5.0e21, 2.5e21, 5.0e20
    # All other elements absent (kg_total = 0)
    H_total, O_total, C_total = 6.0e21, 5.0e21, 1.0e21
    H_liquid, O_liquid, C_liquid = 1.0e21, 2.5e21, 5.0e20
    # Per-species atmospheric mass; sums to M_atm = 8.0e21.
    H2O_atm = 6.0e21
    CO2_atm = 1.5e21
    N2_atm = 5.0e20
    # Per-species totals
    H2O_total, CO2_total, N2_total = 8.0e21, 2.0e21, 5.0e20

    row = {
        'Time': 1000.0,  # yr
        'M_planet': M_planet,
        'M_int': M_int,
        'M_ele': M_ele,
        'M_atm': M_atm,
        'T_surf': 1500.0,
        'T_magma': 2000.0,
        'T_star': 5772.0,
        'P_surf': 100.0,
        'p_xuv': 1e-6,
        'esc_rate_total': 1.0e3,  # kg/s, well under M_atm/dt
        # Per-element masses; H_total = H_atm + H_liquid + H_solid (=0)
        'H_kg_atm': H_atm,
        'H_kg_liquid': H_liquid,
        'H_kg_solid': 0.0,
        'H_kg_total': H_total,
        'O_kg_atm': O_atm,
        'O_kg_liquid': O_liquid,
        'O_kg_solid': 0.0,
        'O_kg_total': O_total,
        'C_kg_atm': C_atm,
        'C_kg_liquid': C_liquid,
        'C_kg_solid': 0.0,
        'C_kg_total': C_total,
        # Per-species masses
        'H2O_kg_atm': H2O_atm,
        'H2O_kg_liquid': H2O_total - H2O_atm,
        'H2O_kg_solid': 0.0,
        'H2O_kg_total': H2O_total,
        'CO2_kg_atm': CO2_atm,
        'CO2_kg_liquid': CO2_total - CO2_atm,
        'CO2_kg_solid': 0.0,
        'CO2_kg_total': CO2_total,
        'N2_kg_atm': N2_atm,
        'N2_kg_liquid': N2_total - N2_atm,
        'N2_kg_solid': 0.0,
        'N2_kg_total': N2_total,
    }
    # Sanity: M_atm should equal sum of (atom counts) so the cross-checks pass.
    elem_sum = H_atm + O_atm + C_atm
    spec_sum = H2O_atm + CO2_atm + N2_atm
    assert math.isclose(elem_sum, M_atm, rel_tol=1e-9)
    assert math.isclose(spec_sum, M_atm, rel_tol=1e-9)
    assert math.isclose(M_int + M_ele, M_planet, rel_tol=1e-9)
    return pd.Series(row)


# ---------------------------------------------------------------------------
# assert_no_nan_inf
# ---------------------------------------------------------------------------
def test_assert_no_nan_inf_accepts_clean_row():
    """A fully-consistent synthetic row passes ``assert_no_nan_inf``
    without raising; pairs with the NaN / Inf flag tests below.
    """
    assert_no_nan_inf(_good_row())


def test_assert_no_nan_inf_flags_nan():
    """A NaN value in T_surf trips ``assert_no_nan_inf`` with a
    ``T_surf=NaN`` message identifying the offending column.
    """
    row = _good_row()
    row['T_surf'] = float('nan')
    with pytest.raises(AssertionError, match=r'T_surf=NaN'):
        assert_no_nan_inf(row)


def test_assert_no_nan_inf_flags_inf():
    """An Inf value in F_atm trips ``assert_no_nan_inf`` with an
    ``F_atm=Inf`` message; the helper distinguishes Inf from NaN in the
    error message so the user sees which class of non-finite value fired.
    """
    row = _good_row()
    row['F_atm'] = float('inf')
    row['F_atm'] = float('inf')  # add new column
    with pytest.raises(AssertionError, match=r'F_atm=Inf'):
        assert_no_nan_inf(pd.Series(row))


# ---------------------------------------------------------------------------
# assert_temperatures_positive
# ---------------------------------------------------------------------------
def test_assert_temperatures_positive_accepts_positive():
    """All-positive temperatures in the synthetic row pass
    ``assert_temperatures_positive`` without raising.
    """
    assert_temperatures_positive(_good_row())


def test_assert_temperatures_positive_flags_zero():
    """``T_surf = 0`` trips the helper with a formatted message including
    the offending value and the K unit, so the user sees the exact
    column and magnitude that violated the positivity contract.
    """
    row = _good_row()
    row['T_surf'] = 0.0
    with pytest.raises(AssertionError, match=r'T_surf=0.000e\+00 K'):
        assert_temperatures_positive(row)


def test_assert_temperatures_positive_flags_negative():
    """A negative T_magma trips the helper; ensures the sign comparison
    is ``> 0`` (strict) rather than ``>= 0`` (which would silently
    accept the zero case tested above).
    """
    row = _good_row()
    row['T_magma'] = -100.0
    with pytest.raises(AssertionError, match=r'T_magma=-1.000e\+02 K'):
        assert_temperatures_positive(row)


def test_assert_temperatures_positive_skips_module_specific_zeros():
    """Optional fields like T_solvus that legitimately stay at 0 must NOT trip."""
    row = _good_row()
    row['T_solvus'] = 0.0
    row['T_core'] = 0.0
    row['T_cmb_initial'] = 0.0
    assert_temperatures_positive(row)  # must not raise


# ---------------------------------------------------------------------------
# assert_pressures_non_negative
# ---------------------------------------------------------------------------
def test_assert_pressures_non_negative_accepts_zero():
    """P_surf = 0 is legitimate (empty atmosphere); the check is non-negative."""
    row = _good_row()
    row['P_surf'] = 0.0
    assert_pressures_non_negative(row)


def test_assert_pressures_non_negative_flags_negative_pressure():
    """Any negative ``*_bar`` partial pressure trips the helper; the
    error message names the column and the offending value.
    """
    row = _good_row()
    row['H2O_bar'] = -0.5
    with pytest.raises(AssertionError, match=r'H2O_bar=-5.000e-01 bar'):
        assert_pressures_non_negative(row)


# ---------------------------------------------------------------------------
# Per-element mass closure
# ---------------------------------------------------------------------------
def test_per_element_mass_closure_accepts_consistent_row():
    """A fully-consistent row where each ``<E>_kg_total = atm + solid + liquid``
    passes the closure check without raising.
    """
    assert_per_element_mass_closure(_good_row())


def test_per_element_mass_closure_flags_dropped_partition():
    """Set H_kg_total higher than the sum of partitions; closure must fail."""
    row = _good_row()
    row['H_kg_total'] = row['H_kg_total'] * 2.0
    with pytest.raises(AssertionError, match=r'H: total=.*atm\+solid\+liquid'):
        assert_per_element_mass_closure(row)


def test_per_element_mass_closure_tolerates_float_noise():
    """A fractional drift below rel_tol must NOT fire."""
    row = _good_row()
    # 1e-9 relative drift is well below rel_tol = 1e-6
    row['H_kg_total'] = row['H_kg_total'] * (1.0 + 1e-9)
    assert_per_element_mass_closure(row)


# ---------------------------------------------------------------------------
# Per-species mass closure
# ---------------------------------------------------------------------------
def test_per_species_mass_closure_accepts_consistent_row():
    """A fully-consistent row where each species' ``kg_total = kg_atm
    + kg_liquid + kg_solid`` passes the closure check.
    """
    assert_per_species_mass_closure(_good_row())


def test_per_species_mass_closure_flags_inconsistent_species():
    """Inflating ``H2O_kg_total`` without changing the partition values
    breaks closure; the helper raises with a message naming H2O.
    """
    row = _good_row()
    row['H2O_kg_total'] = row['H2O_kg_total'] + 1e22  # 10 EkG drop
    with pytest.raises(AssertionError, match=r'H2O: total='):
        assert_per_species_mass_closure(row)


# ---------------------------------------------------------------------------
# assert_atmosphere_element_sum_matches_M_atm
# ---------------------------------------------------------------------------
def test_atmosphere_element_sum_matches_M_atm_accepts_consistent_row():
    """A consistent row where ``sum(<E>_kg_atm) == M_atm`` passes the
    cross-tree check without raising.
    """
    assert_atmosphere_element_sum_matches_M_atm(_good_row())


def test_atmosphere_element_sum_matches_M_atm_flags_drift():
    """Set M_atm 1% higher than the per-element sum; the 0.1% tolerance fires."""
    row = _good_row()
    row['M_atm'] = row['M_atm'] * 1.01
    with pytest.raises(AssertionError, match=r'M_atm=.*disagrees with sum over elements'):
        assert_atmosphere_element_sum_matches_M_atm(row)


def test_atmosphere_element_sum_matches_M_atm_skips_when_M_atm_zero():
    """Pre-IC state (M_atm = 0) must not false-alarm."""
    row = _good_row()
    row['M_atm'] = 0.0
    # element sums are non-zero but the function returns early
    assert_atmosphere_element_sum_matches_M_atm(row)


# ---------------------------------------------------------------------------
# assert_element_sum_matches_species_sum (cross-tree consistency)
# ---------------------------------------------------------------------------
def test_element_sum_matches_species_sum_accepts_consistent_row():
    """A consistent row where the element-tree atmospheric sum matches
    the species-tree atmospheric sum (the two independent
    representations agree) passes the cross-tree check.
    """
    assert_element_sum_matches_species_sum(_good_row())


def test_element_sum_matches_species_sum_flags_distribution_bug():
    """Inflate one species without inflating any element; cross-tree check fires."""
    row = _good_row()
    row['H2O_kg_atm'] = row['H2O_kg_atm'] * 1.5
    with pytest.raises(AssertionError, match=r'Element sum.*disagrees with species sum'):
        assert_element_sum_matches_species_sum(row)


def test_element_sum_matches_species_sum_skips_when_both_zero():
    """No volatile inventory: both sums are 0, function returns early."""
    row = pd.Series({})
    assert_element_sum_matches_species_sum(row)


# ---------------------------------------------------------------------------
# assert_M_atm_le_M_planet
# ---------------------------------------------------------------------------
def test_M_atm_le_M_planet_accepts_realistic_row():
    """An Earth-like row where M_atm << M_planet passes
    ``assert_M_atm_le_M_planet`` without raising; pairs with the
    violation test below.
    """
    assert_M_atm_le_M_planet(_good_row())


def test_M_atm_le_M_planet_flags_violation():
    """M_atm > M_planet: the issue #677 hard invariant must fire."""
    row = _good_row()
    row['M_atm'] = row['M_planet'] * 1.5
    with pytest.raises(AssertionError, match=r'issue #677 regression'):
        assert_M_atm_le_M_planet(row)


def test_M_atm_le_M_planet_admits_float_rounding():
    """A drift at the rel_tol boundary must NOT fire."""
    row = _good_row()
    row['M_atm'] = row['M_planet'] * (1.0 + 5e-7)  # below 1e-6 tol
    assert_M_atm_le_M_planet(row)


def test_M_atm_le_M_planet_skips_when_M_planet_zero():
    """A pre-IC state where ``M_planet = 0`` (no structure yet) returns
    early without firing; otherwise the helper would false-alarm on
    every simulation's first iteration.
    """
    row = _good_row()
    row['M_planet'] = 0.0
    assert_M_atm_le_M_planet(row)


# ---------------------------------------------------------------------------
# assert_M_planet_matches_M_int_plus_M_ele
# ---------------------------------------------------------------------------
def test_M_planet_matches_M_int_plus_M_ele_accepts_consistent_row():
    """A consistent row where ``M_planet == M_int + M_ele`` passes the
    interior-vs-element bookkeeping check.
    """
    assert_M_planet_matches_M_int_plus_M_ele(_good_row())


def test_M_planet_matches_M_int_plus_M_ele_flags_disagreement():
    """Halving M_int leaves ``M_planet > M_int + M_ele``; the helper
    raises with a message naming the two sides of the comparison.
    """
    row = _good_row()
    row['M_int'] = row['M_int'] * 0.5  # halve the interior, planet now wrong
    with pytest.raises(AssertionError, match=r'M_planet=.*does not match M_int \+ M_ele'):
        assert_M_planet_matches_M_int_plus_M_ele(row)


# ---------------------------------------------------------------------------
# assert_escape_within_atmospheric_budget
# ---------------------------------------------------------------------------
def test_escape_bound_accepts_realistic_rate():
    """A physically plausible escape rate (1e3 kg/s for one year) is well
    within the 10*M_atm cap and passes the bound check.
    """
    row = _good_row()
    # 1 yr of escape at 1e3 kg/s = ~3.15e10 kg, well under 10x M_atm
    assert_escape_within_atmospheric_budget(row, dt_s=3.156e7)


def test_escape_bound_flags_unphysical_rate():
    """Sign-flip-style bug producing ~M_atm/second escape: must fire."""
    row = _good_row()
    row['esc_rate_total'] = row['M_atm'] * 100  # would empty atmosphere in 0.01s
    with pytest.raises(AssertionError, match=r'Escape over one step'):
        assert_escape_within_atmospheric_budget(row, dt_s=1.0)


def test_escape_bound_flags_negative_rate():
    """A negative escape rate is unphysical (atmosphere can only leave,
    not arrive); the helper trips with the value in the message.
    """
    row = _good_row()
    row['esc_rate_total'] = -1.0e3
    with pytest.raises(AssertionError, match=r'esc_rate_total = -1.000e\+03 kg/s'):
        assert_escape_within_atmospheric_budget(row, dt_s=1.0)


def test_escape_bound_skips_when_dt_unknown():
    """dt_s = None falls back to finiteness/sign only, not bound."""
    row = _good_row()
    row['esc_rate_total'] = row['M_atm'] * 1e6  # huge but no dt provided
    assert_escape_within_atmospheric_budget(row, dt_s=None)


def test_escape_bound_flags_nonfinite_rate():
    """A NaN escape rate is caught by the bound check (which guards
    against any non-finite value as well as out-of-bound rates).
    """
    row = _good_row()
    row['esc_rate_total'] = float('nan')
    with pytest.raises(AssertionError, match=r'esc_rate_total = nan'):
        assert_escape_within_atmospheric_budget(row, dt_s=1.0)


# ---------------------------------------------------------------------------
# Composite check
# ---------------------------------------------------------------------------
def test_composite_check_accepts_clean_dataframe():
    """``assert_smoke_conservation_invariants`` accepts a two-row
    DataFrame where every per-row invariant passes and the dt-dependent
    escape bound is satisfied.
    """
    row = _good_row()
    # Two rows so dt is computed (dt = 1000 yr, well under 10x M_atm)
    df = pd.DataFrame([row.copy(), row.copy()])
    df.loc[0, 'Time'] = 0.0
    df.loc[1, 'Time'] = 1000.0
    assert_smoke_conservation_invariants(df)


def test_composite_check_rejects_empty_dataframe():
    """An empty helpfile DataFrame is rejected with a 'helpfile is
    empty' message rather than silently passing on a zero-row check.
    """
    with pytest.raises(AssertionError, match=r'helpfile is empty'):
        assert_smoke_conservation_invariants(pd.DataFrame())


def test_composite_check_falls_back_when_single_row():
    """A 1-row helpfile can't compute dt, so the escape bound must skip
    cleanly. Other invariants still run.
    """
    df = pd.DataFrame([_good_row()])
    assert_smoke_conservation_invariants(df)


def test_composite_check_propagates_first_failure():
    """When one invariant fires, the composite check raises with that
    failure's message (not a generic wrapper). Triggering a NaN, which
    `assert_no_nan_inf` runs first in the composite chain."""
    row = _good_row()
    row['T_surf'] = float('nan')
    df = pd.DataFrame([row, row.copy()])
    df.loc[0, 'Time'] = 0.0
    df.loc[1, 'Time'] = 1000.0
    with pytest.raises(AssertionError, match=r'T_surf=NaN'):
        assert_smoke_conservation_invariants(df)


# ---------------------------------------------------------------------------
# dt unit conversion: years to seconds via secs_per_year
# ---------------------------------------------------------------------------
def test_dt_conversion_uses_canonical_year_length():
    """The composite check converts years to seconds via secs_per_year.
    Verify the conversion produces the expected magnitude (a 1000-year dt
    should map to about 3.15e10 s).
    """
    from proteus.utils.constants import secs_per_year

    df = pd.DataFrame([_good_row(), _good_row()])
    df.loc[0, 'Time'] = 0.0
    df.loc[1, 'Time'] = 1000.0
    expected_dt_s = 1000.0 * secs_per_year
    # 1 year in seconds = 3.156e7 (Julian year via 365.25 days)
    assert math.isclose(expected_dt_s, 3.15576e10, rel_tol=1e-3)
    # Composite check must accept this dt without firing the escape bound
    assert_smoke_conservation_invariants(df)


# ---------------------------------------------------------------------------
# Edge cases that historically caused false positives
# ---------------------------------------------------------------------------
def test_all_zero_helpfile_does_not_false_alarm():
    """A pre-IC row where every mass is 0 must pass every check
    (the early-return branches handle this)."""
    row = pd.Series(
        {
            'Time': 1.0,
            'M_planet': 0.0,
            'M_int': 0.0,
            'M_ele': 0.0,
            'M_atm': 0.0,
            'T_surf': 1500.0,
            'T_magma': 2000.0,
            'T_star': 5772.0,
            'esc_rate_total': 0.0,
        }
    )
    df = pd.DataFrame([row])
    assert_smoke_conservation_invariants(df)


def test_zero_volatile_planet_passes_per_element_closure():
    """If H/O/C kg_total are all 0, per-element closure passes via the
    abs_tol_kg = 1.0 floor (the rel_tol branch goes to 0)."""
    row = _good_row()
    for e in ('H', 'O', 'C'):
        for k in ('atm', 'solid', 'liquid', 'total'):
            row[f'{e}_kg_{k}'] = 0.0
    assert_per_element_mass_closure(row)


def test_string_columns_in_helpfile_are_skipped():
    """assert_no_nan_inf must skip non-numeric columns silently
    (some helpfile fields like config_path are strings)."""
    row = _good_row()
    row['config_path'] = '/tmp/some/path'  # noqa: S108  test-only path
    assert_no_nan_inf(row)


def test_uses_canonical_constant_for_year_length():
    """Catches a regression that re-introduces an inline 365.25*24*3600
    constant instead of importing secs_per_year."""
    import _smoke_invariants as helper

    src = open(helper.__file__).read()
    assert 'secs_per_year' in src, (
        '_smoke_invariants must import and use the canonical secs_per_year '
        'from proteus.utils.constants instead of reinventing the conversion.'
    )
    assert 'years_per_sec' not in src, (
        '_smoke_invariants previously had an inverted-naming variable '
        '`years_per_sec` for year-to-second conversion. The fix uses '
        '`secs_per_year` from proteus.utils.constants. Do not reintroduce.'
    )
