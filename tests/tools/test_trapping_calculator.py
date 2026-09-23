"""Unit tests for ``tools/trapping_calculator.py``.

The module under test computes solid-phase volatile trapping offline from a
PROTEUS helpfile: for each crystallisation increment it buries a species at the
effective partition coefficient ``D_eff = (1 - F_tl) * D_Z + F_tl`` times the
melt concentration. ``F_tl`` comes from one of three modes: ``none`` traps
nothing at all, ``constant`` supplies a fixed scalar, and ``dynamic`` recomputes
it every step from the secular cooling rate after Sim et al. (2024), Section
2.2. These tests exercise:

* the three modes on one helpfile, so ``none`` is exactly zero, ``constant``
  matches the closed form, and ``dynamic`` matches a hand-computed ``F_tl(t)``,
* the dynamic fraction against the published model the source paper was run
  with, transcribed in its own unit system,
* the ``F_tl`` clamp at the disaggregation melt fraction and at zero, the
  warming-step branch where the cooling rate turns non-negative, and the
  zero-length step PROTEUS writes on restart,
* the analytic limits of ``D_eff`` (``D_Z = 0`` reduces it to ``F_tl``,
  ``D_Z = 1`` gives no fractionation) and its validation contract,
* the closed-form trapped mass for two species with different ``D_Z``,
  including one at ``D_Z = 0``, against hand-computed values,
* the remelting clamp, so a shrinking solid mantle never reduces the total,
* the fully-crystallised limit, where no melt is available to draw on,
* the supply cap, so no step buries more of a species than the melt holds,
* the default coefficient set, which spans every volatile the helpfile can
  supply and is non-zero only for water, whose coefficient is elemental,
* the trapping of species at ``D_Z = 0``, which the interstitial melt buries at
  ``F_tl * C_Z * dM_RM`` although no crystal term applies to them,
* the separation of ``D_Z = 0``, ``F_tl = 0`` and the none mode, which are three
  different results rather than three spellings of one,
* the missing-column error contract and the skip path for absent species, both
  for a single species and across the full default set,
* the plot cap and the summary grouping that keep a full-species run readable,
* delimiter detection, since PROTEUS writes the helpfile tab-separated,
* the figures, which carry trapped mass on one axis and nothing else.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    """Load a tools/ script by path; the directory is not an importable package."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / 'tools' / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_tc = _load('trapping_calculator')

# Hand-built helpfile. The melt concentrations are constant by construction
# (H2O at 1e-3, CO2 at 2e-3 mass fraction) so the trapped mass per step reduces
# to D_eff * C_Z * dM_RM with no interpolation to reason about. The solid mantle
# shrinks on the last step, which exercises the remelting clamp.
_TIME = [0.0, 1.0e2, 2.0e2, 3.0e2]
_M_SOLID = [0.0, 1.0e21, 3.0e21, 2.5e21]
_M_LIQUID = [4.0e21, 3.0e21, 1.0e21, 1.5e21]


def _synthetic_helpfile() -> pd.DataFrame:
    """Four-row helpfile with constant melt concentrations for two species."""
    return pd.DataFrame(
        {
            'Time': _TIME,
            'M_mantle': [4.0e21] * 4,
            'M_mantle_solid': _M_SOLID,
            'M_mantle_liquid': _M_LIQUID,
            'Phi_global': [liq / 4.0e21 for liq in _M_LIQUID],
            'H2O_kg_liquid': [4.0e18, 3.0e18, 1.0e18, 1.5e18],
            'H2O_kg_total': [5.0e18] * 4,
            'CO2_kg_liquid': [8.0e18, 6.0e18, 2.0e18, 3.0e18],
            'CO2_kg_total': [1.0e19] * 4,
        }
    )


def test_effective_partition_limits_bracket_the_trapped_melt_fraction():
    """A perfectly incompatible species is buried only as trapped melt, and a
    species that does not fractionate is buried whole, so D_eff is bounded by
    F_tl below and 1 above for any physical D_Z."""
    f_tl = 0.02
    assert _tc.effective_partition(f_tl, 0.0) == pytest.approx(f_tl, rel=1e-12)
    assert _tc.effective_partition(f_tl, 1.0) == pytest.approx(1.0, rel=1e-12)
    # Intermediate D_Z stays strictly inside the bracket. A regression that
    # dropped the (1 - F_tl) weight would give 0.0217 here, not 0.021666.
    mid = _tc.effective_partition(f_tl, 0.0017)
    assert mid == pytest.approx(0.021666, rel=1e-12)
    assert abs(mid - (0.0017 + f_tl)) > 1.0e-5
    assert f_tl < mid < 1.0

    # Validation contract: an out-of-range trapped fraction and a negative
    # partition coefficient are both refused rather than silently accepted.
    with pytest.raises(ValueError, match='F_tl'):
        _tc.effective_partition(1.5, 0.0017)
    with pytest.raises(ValueError, match='D_Z'):
        _tc.effective_partition(f_tl, -0.1)


@pytest.mark.physics_invariant
def test_trapped_mass_matches_closed_form_across_two_partition_coefficients():
    """Two species with different D_Z, one of them zero, are buried at
    D_eff * C_Z * dM_RM over the two crystallising steps of the helpfile.

    H2O: D_eff = 0.98 * 0.0017 + 0.02 = 0.021666, C_Z = 1e-3, dM_RM summing to
    3e21 kg, giving 6.4998e16 kg. CO2: D_eff = 0.02 exactly, C_Z = 2e-3, giving
    1.2e17 kg. Carbon is buried despite D_Z = 0 because interstitial melt is
    buried whatever the crystal chemistry does.
    """
    result = _tc.compute_trapping(
        _synthetic_helpfile(), f_tl=0.02, d_z={'H2O': 0.0017, 'CO2': 0.0}
    )
    by_name = {species.name: species for species in result.species}

    water = by_name['H2O']
    assert water.total_trapped == pytest.approx(6.4998e16, rel=1e-12)
    # Formula guard: omitting the trapped-melt term entirely (D_eff = D_Z)
    # lands at 5.1e15 kg, two orders below the correct value.
    assert abs(water.total_trapped - 0.0017 * 1.0e-3 * 3.0e21) > 1.0e16
    # Weight guard: dropping the (1 - F_tl) factor lands at 6.51e16 kg.
    assert abs(water.total_trapped - 0.0217 * 1.0e-3 * 3.0e21) > 1.0e14
    # Sign and scale guards: mass is buried, not released, and the magnitude is
    # ~1e17 kg rather than ~1e20 (melt mass mistaken for melt concentration).
    assert water.total_trapped > 0.0
    assert 1.0e16 < water.total_trapped < 1.0e17
    assert water.inventory_fraction == pytest.approx(6.4998e16 / 5.0e18, rel=1e-12)

    carbon = by_name['CO2']
    assert carbon.d_eff == pytest.approx(0.02, rel=1e-12)
    assert carbon.total_trapped == pytest.approx(1.2e17, rel=1e-12)
    # Cross-species discrimination: CO2 outruns H2O purely on melt
    # concentration, so the ratio is fixed by D_eff * C_Z alone.
    assert carbon.total_trapped > water.total_trapped
    ratio = carbon.total_trapped / water.total_trapped
    assert ratio == pytest.approx((0.02 * 2.0e-3) / (0.021666 * 1.0e-3), rel=1e-12)


@pytest.mark.physics_invariant
def test_remelting_step_is_clamped_and_never_reduces_the_cumulative_total():
    """The solid mantle shrinks over the last step of the helpfile. Remelting
    releases volatiles rather than burying them, so that step contributes
    nothing and the cumulative trapped mass stays non-decreasing."""
    frame = _synthetic_helpfile()
    dm_rm, n_negative = _tc.crystallised_mass(frame['M_mantle_solid'].to_numpy(dtype=float))
    assert n_negative == 1
    assert dm_rm[-1] == pytest.approx(0.0, abs=1.0e-6)
    # The crystallising steps are untouched by the clamp.
    assert dm_rm[1] == pytest.approx(1.0e21, rel=1e-12)
    assert dm_rm[2] == pytest.approx(2.0e21, rel=1e-12)

    result = _tc.compute_trapping(frame, f_tl=0.02, d_z={'H2O': 0.0017})
    water = result.species[0]
    assert result.n_negative_dm == 1
    assert water.dm_trap[-1] == pytest.approx(0.0, abs=1.0e-6)
    assert np.all(np.diff(water.cumulative) >= 0.0)
    # Edge case: a single-row helpfile has no step to difference at all.
    single, single_negative = _tc.crystallised_mass(np.array([1.0e21]))
    assert single_negative == 0
    assert single[0] == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.physics_invariant
def test_fully_crystallised_step_traps_nothing_and_stays_finite():
    """Once the melt is gone there is no reservoir to draw on, so the melt
    concentration is zero rather than a division by zero, and no further mass
    is buried even while the solid mantle keeps growing."""
    frame = pd.DataFrame(
        {
            'Time': [0.0, 1.0e2, 2.0e2],
            'M_mantle_solid': [0.0, 2.0e21, 4.0e21],
            'M_mantle_liquid': [4.0e21, 0.0, 0.0],
            'Phi_global': [1.0, 0.0, 0.0],
            'H2O_kg_liquid': [4.0e18, 0.0, 0.0],
            'H2O_kg_total': [5.0e18] * 3,
        }
    )
    result = _tc.compute_trapping(frame, f_tl=0.02, d_z={'H2O': 0.0017})
    water = result.species[0]
    assert result.n_zero_melt == 2
    assert np.all(np.isfinite(water.c_z))
    assert np.all(np.isfinite(water.dm_trap))
    assert water.total_trapped == pytest.approx(0.0, abs=1.0e-6)
    # The first row carries melt but no crystallisation increment, so its
    # concentration is the physical 1e-3 while its trapped mass is still zero.
    assert water.c_z[0] == pytest.approx(1.0e-3, rel=1e-12)
    assert water.c_z[1] == pytest.approx(0.0, abs=1.0e-15)


@pytest.mark.physics_invariant
def test_trapped_mass_never_exceeds_the_melt_it_draws_on():
    """A coarse step can ask for more of a species than the melt holds. The
    burial is capped at the available melt mass, so the trapped mass respects
    the reservoir it comes from and the capped step is reported."""
    frame = pd.DataFrame(
        {
            'Time': [0.0, 1.0e2],
            'M_mantle_solid': [0.0, 1.0e21],
            'M_mantle_liquid': [1.0e21, 1.0e20],
            'Phi_global': [1.0, 0.1],
            'H2O_kg_liquid': [2.0e18, 1.0e18],
            'H2O_kg_total': [3.0e18, 3.0e18],
        }
    )
    # D_eff = 0.5 with D_Z = 0, C_Z = 0.01, dM_RM = 1e21 asks for 5e18 kg from
    # a melt holding 1e18 kg, so the cap binds by a factor of five.
    result = _tc.compute_trapping(frame, f_tl=0.5, d_z={'H2O': 0.0})
    water = result.species[0]
    assert water.n_supply_clamped == 1
    assert water.total_trapped == pytest.approx(1.0e18, rel=1e-12)
    assert water.total_trapped <= frame['H2O_kg_liquid'].to_numpy(dtype=float)[1]
    # Without the cap the uncapped product would be 5e18 kg, well clear of the
    # capped answer, so this is not a tolerance-level distinction.
    assert abs(water.total_trapped - 5.0e18) > 1.0e18


def test_missing_required_column_names_the_column_and_computes_nothing():
    """A helpfile without the melt mass cannot supply a melt concentration.
    The calculation refuses it by name rather than guessing a substitute."""
    frame = _synthetic_helpfile().drop(columns=['M_mantle_liquid'])
    with pytest.raises(_tc.MissingColumnError, match='M_mantle_liquid'):
        _tc.compute_trapping(frame, f_tl=0.02, d_z={'H2O': 0.0017})

    # Every required column is checked, not just the first, and the message
    # lists them together so one run names the whole gap.
    stripped = _synthetic_helpfile().drop(columns=['Time', 'Phi_global'])
    with pytest.raises(_tc.MissingColumnError) as excinfo:
        _tc.compute_trapping(stripped, f_tl=0.02, d_z={'H2O': 0.0017})
    assert 'Time' in str(excinfo.value)
    assert 'Phi_global' in str(excinfo.value)


def test_species_without_helpfile_columns_are_skipped_not_fatal():
    """A requested species the helpfile never tracked is reported and passed
    over, so a partition dictionary can name more species than a given run
    carried without failing the calculation."""
    result = _tc.compute_trapping(
        _synthetic_helpfile(), f_tl=0.02, d_z={'H2O': 0.0017, 'N2': 0.004}
    )
    assert result.skipped == ['N2']
    assert [species.name for species in result.species] == ['H2O']
    # The surviving species is computed exactly as it would be on its own.
    assert result.species[0].total_trapped == pytest.approx(6.4998e16, rel=1e-12)
    # Edge case: nothing requested that the helpfile supports leaves an empty
    # result rather than raising.
    empty = _tc.compute_trapping(_synthetic_helpfile(), f_tl=0.02, d_z={'N2': 0.004})
    assert empty.species == []
    assert empty.skipped == ['N2']


def test_helpfile_delimiter_is_detected_for_tab_and_comma_files(tmp_path):
    """PROTEUS writes runtime_helpfile.csv tab-separated despite the extension.
    Reading it as comma-separated would collapse it to one column, so the
    delimiter is taken from the header line."""
    frame = _synthetic_helpfile()
    tabbed = tmp_path / 'runtime_helpfile.csv'
    frame.to_csv(tabbed, index=False, sep='\t', float_format='%.10e')
    read_tab = _tc.read_helpfile(tabbed)
    assert list(read_tab.columns) == list(frame.columns)
    assert read_tab['M_mantle_liquid'].iloc[1] == pytest.approx(3.0e21, rel=1e-9)

    comma = tmp_path / 'comma_helpfile.csv'
    frame.to_csv(comma, index=False, float_format='%.10e')
    read_comma = _tc.read_helpfile(comma)
    assert list(read_comma.columns) == list(frame.columns)
    assert read_comma['H2O_kg_liquid'].iloc[2] == pytest.approx(1.0e18, rel=1e-9)


# Second hand-built helpfile, for the dynamic mode. The solid mantle grows by
# the same 1e21 kg on every step so the trapped mass isolates F_tl(t), and the
# melt concentrations are again constant (H2O at 1e-3, CO2 at 2e-3).
#
# T_pot is chosen so the three steps land in the three regimes of Sim et al.
# (2024) Eq. 7 in turn: inside the valid window, above the disaggregation limit,
# and warming. With phi_c = 0.3, tau = 1e3 yr and DeltaT = 100 K the prefactor
# phi_c * tau / DeltaT is 3 yr K-1, so F_tl = -3 * dT/dt:
#
#   step 1: dT/dt = -0.01 K/yr  ->  F_tl = 0.03   (inside the window)
#   step 2: dT/dt = -0.50 K/yr  ->  F_tl = 1.5, clamped down to phi_c = 0.3
#   step 3: dT/dt = +0.10 K/yr  ->  F_tl = -0.3, clamped up to 0 (warming)
#
# tau is 1e3 yr here rather than the paper's 1 Ma so the three regimes are
# reached with temperature steps of order 1 to 50 K over 100 yr, which keeps the
# expected values readable. The paper's own defaults are pinned separately in
# test_paper_default_parameters_put_the_clamp_at_the_stated_cooling_rate.
_DYN_PHI_C = 0.3
_DYN_TAU_YR = 1.0e3
_DYN_DELTA_T_K = 100.0
_DYN_D_Z = {'H2O': 0.0017, 'CO2': 0.0}
_DYN_KWARGS = {
    'd_z': _DYN_D_Z,
    'phi_c': _DYN_PHI_C,
    'tau': _DYN_TAU_YR,
    'delta_t': _DYN_DELTA_T_K,
}


def _dynamic_helpfile() -> pd.DataFrame:
    """Four-row helpfile whose cooling rate sweeps the three F_tl regimes."""
    return pd.DataFrame(
        {
            'Time': [0.0, 1.0e2, 2.0e2, 3.0e2],
            'M_mantle': [4.0e21] * 4,
            'M_mantle_solid': [0.0, 1.0e21, 2.0e21, 3.0e21],
            'M_mantle_liquid': [4.0e21, 3.0e21, 2.0e21, 1.0e21],
            'Phi_global': [1.0, 0.75, 0.5, 0.25],
            'T_pot': [2000.0, 1999.0, 1949.0, 1959.0],
            'H2O_kg_liquid': [4.0e18, 3.0e18, 2.0e18, 1.0e18],
            'H2O_kg_total': [5.0e18] * 4,
            'CO2_kg_liquid': [8.0e18, 6.0e18, 4.0e18, 2.0e18],
            'CO2_kg_total': [1.0e19] * 4,
        }
    )


@pytest.mark.physics_invariant
def test_no_trapping_mode_buries_nothing_while_still_reporting_every_step():
    """The none mode is the absence of the process, not F_tl = 0. Every species
    is buried at exactly zero even where the crystal partition coefficient is
    non-zero and melt is present, and the per-step table is still emitted at
    full length so the three modes stay directly comparable."""
    frame = _dynamic_helpfile()
    result = _tc.compute_trapping(frame, mode='none', **_DYN_KWARGS)
    by_name = {species.name: species for species in result.species}

    assert result.mode == 'none'
    assert result.f_tl is None
    for name in ('H2O', 'CO2'):
        species = by_name[name]
        assert species.total_trapped == pytest.approx(0.0, abs=1.0e-30)
        assert np.all(species.dm_trap == pytest.approx(0.0, abs=1.0e-30))
        # The bracket is never evaluated, so there is no effective partition
        # coefficient to report. A regression that routed none through
        # F_tl = 0 would leave D_eff = D_Z here and bury 5.1e15 kg of H2O.
        assert species.d_eff is None
    assert by_name['H2O'].total_trapped < 1.0e-30 * 5.0e18

    # The melt concentrations are still computed, so the table carries the
    # same diagnostic columns as the other modes rather than an empty frame.
    assert by_name['H2O'].c_z[1] == pytest.approx(1.0e-3, rel=1e-12)
    table = _tc.build_table(result, frame)
    assert len(table) == len(frame)
    assert list(table['mode'].unique()) == ['none']

    # Edge case: a helpfile with melt but no crystallisation at all still
    # returns a full-length zero series rather than raising.
    flat = frame.copy()
    flat['M_mantle_solid'] = 0.0
    flat_result = _tc.compute_trapping(flat, mode='none', **_DYN_KWARGS)
    assert flat_result.species[0].cumulative.size == len(flat)
    assert flat_result.species[0].total_trapped == pytest.approx(0.0, abs=1.0e-30)


@pytest.mark.physics_invariant
def test_dynamic_trapped_fraction_matches_hand_computed_cooling_rates():
    """F_tl(t) is -(phi_c * tau / DeltaT) * dT/dt, clamped to [0, phi_c]. On the
    dynamic helpfile the prefactor is 3 yr K-1 and the three steps cool at
    -0.01 and -0.5 K/yr and then warm at +0.1 K/yr, so the unclamped fractions
    are 0.03, 1.5 and -0.3 and the clamped series is 0.03, 0.3 and 0."""
    frame = _dynamic_helpfile()
    dynamic = _tc.dynamic_trapped_fraction(
        frame['Time'].to_numpy(dtype=float),
        frame['T_pot'].to_numpy(dtype=float),
        phi_c=_DYN_PHI_C,
        tau=_DYN_TAU_YR,
        delta_t=_DYN_DELTA_T_K,
    )

    np.testing.assert_allclose(dynamic.dt_dt, [0.0, -0.01, -0.5, 0.1], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(dynamic.f_tl_raw, [0.0, 0.03, 1.5, -0.3], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(dynamic.f_tl, [0.0, 0.03, 0.3, 0.0], rtol=1e-12, atol=1e-15)

    # Sign guard: the leading minus is what makes a cooling mantle trap melt.
    # Dropping it would put the cooling steps at -0.03 and -1.5 and leave only
    # the warming step positive, so the first cooling step must be positive.
    assert dynamic.f_tl_raw[1] > 0.0
    assert dynamic.f_tl_raw[3] < 0.0
    # Prefactor guard: omitting tau / DeltaT leaves the bare phi_c * dT/dt, which
    # puts step 1 at 0.003 rather than 0.03, an order of magnitude away.
    assert abs(dynamic.f_tl_raw[1] - _DYN_PHI_C * 0.01) > 1.0e-3
    # Boundedness invariant: the series actually used never leaves the window.
    assert np.all(dynamic.f_tl >= 0.0)
    assert np.all(dynamic.f_tl <= _DYN_PHI_C)

    # Edge case: a single-row helpfile has no step to difference, so the rate is
    # zero everywhere and no step is counted either way.
    lone = _tc.dynamic_trapped_fraction(np.array([0.0]), np.array([2000.0]))
    assert lone.dt_dt.size == 1
    assert lone.n_warming == 0
    assert lone.f_tl[0] == pytest.approx(0.0, abs=1.0e-15)

    # Validation contract: the three parameters are refused out of range rather
    # than silently producing a nonsensical fraction.
    with pytest.raises(ValueError, match='phi_c'):
        _tc.dynamic_trapped_fraction(np.array([0.0, 1.0]), np.array([2.0, 1.0]), phi_c=0.0)
    with pytest.raises(ValueError, match='tau'):
        _tc.dynamic_trapped_fraction(np.array([0.0, 1.0]), np.array([2.0, 1.0]), tau=-1.0)
    with pytest.raises(ValueError, match='DeltaT'):
        _tc.dynamic_trapped_fraction(np.array([0.0, 1.0]), np.array([2.0, 1.0]), delta_t=0.0)


@pytest.mark.physics_invariant
def test_dynamic_mode_trapped_mass_matches_the_hand_computed_per_step_series():
    """With F_tl running 0.03, 0.3 and 0 over three steps of dM_RM = 1e21 kg,
    H2O at D_Z = 0.0017 and C_Z = 1e-3 is buried at 3.1649e16, 3.0119e17 and
    1.7e15 kg, summing to 3.34539e17 kg. CO2 at D_Z = 0 is buried at F_tl * C_Z
    * dM_RM with C_Z = 2e-3, so 6e16, 6e17 and exactly nothing on the warming
    step, summing to 6.6e17 kg."""
    result = _tc.compute_trapping(_dynamic_helpfile(), mode='dynamic', **_DYN_KWARGS)
    by_name = {species.name: species for species in result.species}
    assert result.mode == 'dynamic'

    water = by_name['H2O']
    np.testing.assert_allclose(
        water.dm_trap, [0.0, 3.1649e16, 3.0119e17, 1.7e15], rtol=1e-12, atol=1.0
    )
    assert water.total_trapped == pytest.approx(3.34539e17, rel=1e-12)
    # Clamp guard: leaving F_tl unclamped at 1.5 on step 2 buries 1.49915e18 kg
    # on that step alone, putting the total near 1.53e18 rather than 3.3e17.
    assert abs(water.total_trapped - 1.5325e18) > 1.0e18
    # Constant-mode guard: this is not the fixed-F_tl answer in disguise. At
    # F_tl = 0.02 over the same three steps the total is 6.4998e16 kg.
    assert abs(water.total_trapped - 6.4998e16) > 2.0e17
    # Sign and scale guards: mass is buried, and the magnitude is ~1e17 kg
    # rather than ~1e20 (melt mass mistaken for melt concentration).
    assert water.total_trapped > 0.0
    assert 1.0e17 < water.total_trapped < 1.0e18

    carbon = by_name['CO2']
    np.testing.assert_allclose(carbon.dm_trap, [0.0, 6.0e16, 6.0e17, 0.0], rtol=1e-12, atol=1.0)
    assert carbon.total_trapped == pytest.approx(6.6e17, rel=1e-12)
    # D_eff varies step to step in this mode, and for D_Z = 0 it is F_tl itself.
    np.testing.assert_allclose(carbon.d_eff, [0.0, 0.03, 0.3, 0.0], rtol=1e-12, atol=1e-15)

    # Conservation: no species is buried beyond the inventory it came from, and
    # the cumulative series never decreases.
    for species in (water, carbon):
        assert species.total_trapped < species.inventory_final
        assert np.all(np.diff(species.cumulative) >= 0.0)


@pytest.mark.physics_invariant
def test_warming_step_drops_the_trapped_melt_but_keeps_crystal_partitioning():
    """A warming step makes the dynamic formula negative, so F_tl clamps to zero
    and no interstitial melt is retained. Crystal partitioning still runs: H2O
    is buried at D_Z * C_Z * dM_RM = 1.7e15 kg on that step, while CO2 at
    D_Z = 0 is buried at exactly nothing. This is what separates a warming step
    in dynamic mode from the none mode, where both are zero."""
    frame = _dynamic_helpfile()
    dynamic = _tc.compute_trapping(frame, mode='dynamic', **_DYN_KWARGS)
    none = _tc.compute_trapping(frame, mode='none', **_DYN_KWARGS)
    dyn_by_name = {s.name: s for s in dynamic.species}
    none_by_name = {s.name: s for s in none.species}

    assert dynamic.dynamic.n_warming == 1
    assert dynamic.dynamic.f_tl[3] == pytest.approx(0.0, abs=1.0e-15)
    # H2O still buries at the bare partition coefficient on the warming step.
    assert dyn_by_name['H2O'].dm_trap[3] == pytest.approx(1.7e15, rel=1e-12)
    # Scale guard: 1.7e15 kg is the D_Z-only answer, two orders below the
    # 3.0119e17 kg the preceding clamped step buried.
    assert dyn_by_name['H2O'].dm_trap[3] < 0.01 * dyn_by_name['H2O'].dm_trap[2]
    assert dyn_by_name['H2O'].dm_trap[3] > 0.0
    # CO2 is perfectly incompatible, so with no trapped melt it buries nothing.
    assert dyn_by_name['CO2'].dm_trap[3] == pytest.approx(0.0, abs=1.0e-30)
    # The none mode zeroes both, which is the distinction under test.
    assert none_by_name['H2O'].dm_trap[3] == pytest.approx(0.0, abs=1.0e-30)

    # Edge case: an isothermal step has dT/dt of exactly zero, which counts as
    # non-cooling and yields F_tl = 0 without tripping the lower clamp.
    flat = frame.copy()
    flat['T_pot'] = [2000.0, 2000.0, 1999.0, 1998.0]
    flat_dynamic = _tc.compute_trapping(flat, mode='dynamic', **_DYN_KWARGS).dynamic
    assert flat_dynamic.n_warming == 1
    assert flat_dynamic.n_clamped_low == 0
    assert flat_dynamic.f_tl[1] == pytest.approx(0.0, abs=1.0e-15)


@pytest.mark.physics_invariant
def test_clamped_and_unusable_steps_are_counted_and_reported():
    """The run reports how far outside the valid window it strayed: one step
    above the disaggregation limit, one below zero, one warming, and, on a
    helpfile that repeats a timestamp the way PROTEUS does at restart, one step
    of zero length whose cooling rate cannot be formed."""
    dynamic = _tc.compute_trapping(_dynamic_helpfile(), mode='dynamic', **_DYN_KWARGS).dynamic
    assert dynamic.n_clamped_high == 1
    assert dynamic.n_clamped_low == 1
    assert dynamic.n_clamped == 2
    assert dynamic.n_warming == 1
    assert dynamic.n_unusable == 0
    assert dynamic.column == 'T_pot'

    # A repeated timestamp gives a zero-length step. Dividing by it would put
    # an infinity into F_tl; the step is excluded and counted instead.
    repeated = _dynamic_helpfile()
    repeated['Time'] = [0.0, 0.0, 1.0e2, 2.0e2]
    result = _tc.compute_trapping(repeated, mode='dynamic', **_DYN_KWARGS)
    assert result.dynamic.n_unusable == 1
    assert np.all(np.isfinite(result.dynamic.f_tl))
    assert np.all(np.isfinite(result.species[0].dm_trap))
    assert result.dynamic.f_tl[1] == pytest.approx(0.0, abs=1.0e-15)
    # The excluded step is not counted as warming either, so the counters do
    # not double-report it.
    assert result.dynamic.n_warming == 1

    # Edge case: a non-finite temperature is excluded on the same footing
    # rather than propagating a NaN through every trapped mass.
    holed = _dynamic_helpfile()
    holed['T_pot'] = [2000.0, np.nan, 1949.0, 1959.0]
    holed_result = _tc.compute_trapping(holed, mode='dynamic', **_DYN_KWARGS)
    assert holed_result.dynamic.n_unusable == 2
    assert np.all(np.isfinite(holed_result.species[0].dm_trap))


_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
# Constants as they appear in the model the source paper was run with,
# https://github.com/joycesim/MOE at commit e7edd1c, file mars_module.py:
# phic dimensionless (line 331), tau in seconds (line 330, the constructor
# default), deltaT in K (line 332), and dT/dt in K/s (the rhs docstring).
_MOE_PHI_C = 0.3
_MOE_TAU_S = 3.15e13
_MOE_DELTA_T_K = 100.0


def _moe_trapped_fraction(dt_dt_per_second: float) -> float:
    """Trapped melt fraction as the published model computes it.

    Transcribed from mars_module.py line 1241 and its clamp at lines 1247
    to 1248:

        self.Ftl[ii] = -self.phic * self.tau * self.dTdt[ii] / self.deltaT
        if self.Ftl[ii] >= 0.3:
            self.Ftl[ii] = 0.3
    """
    f_tl = -_MOE_PHI_C * _MOE_TAU_S * dt_dt_per_second / _MOE_DELTA_T_K
    return min(f_tl, 0.3)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_dynamic_fraction_reproduces_the_published_model_across_cooling_rates():
    """Cross-implementation check against the model behind Sim, Hirschmann and
    Hier-Majumder (2024), JGR Planets 129, e2024JE008346, published at
    https://github.com/joycesim/MOE (commit e7edd1c, mars_module.py line 1241).

    That model carries tau in seconds and dT/dt in K/s; this one carries them in
    years to match the helpfile Time column. Feeding both the same physical
    cooling rate must give the same dimensionless fraction, which fixes the
    prefactor and the unit conversion together. The rates span the valid window
    and cross the disaggregation limit, so the clamp is compared too.
    """
    tau_years = _MOE_TAU_S / _SECONDS_PER_YEAR
    dt_years = 1.0e4

    for rate_per_year, expected in (
        (-1.0e-5, 0.029965753424657536),
        (-5.0e-5, 0.14982876712328766),
        (-9.0e-5, 0.2696917808219178),
        (-2.0e-4, 0.3),
    ):
        time = np.array([0.0, dt_years])
        temperature = np.array([2000.0, 2000.0 + rate_per_year * dt_years])
        ours = _tc.dynamic_trapped_fraction(
            time,
            temperature,
            phi_c=_MOE_PHI_C,
            tau=tau_years,
            delta_t=_MOE_DELTA_T_K,
        )
        reference = _moe_trapped_fraction(rate_per_year / _SECONDS_PER_YEAR)
        assert ours.f_tl[1] == pytest.approx(reference, rel=1e-12)
        # Independent pin, so the test does not rest solely on the transcription
        # agreeing with itself. The literals are 0.3 * (3.15e13 / 31536000) *
        # |rate| / 100 evaluated separately, hence the closed-form tolerance.
        assert ours.f_tl[1] == pytest.approx(expected, rel=1e-12)

    # The fastest rate is the one the published model clamps, and the slowest is
    # far below the limit, so the comparison spans both branches.
    assert _moe_trapped_fraction(-2.0e-4 / _SECONDS_PER_YEAR) == pytest.approx(0.3, rel=1e-12)
    assert _moe_trapped_fraction(-1.0e-5 / _SECONDS_PER_YEAR) < 0.1

    # Unit guard: reading tau as years while the rate is per second, or the
    # reverse, moves the answer by the 3.16e7 s/yr conversion rather than a
    # tolerance. Feeding the seconds-valued tau to the years-based function
    # saturates the clamp at every rate above.
    mismatched = _tc.dynamic_trapped_fraction(
        np.array([0.0, dt_years]),
        np.array([2000.0, 2000.0 - 1.0e-5 * dt_years]),
        phi_c=_MOE_PHI_C,
        tau=_MOE_TAU_S,
        delta_t=_MOE_DELTA_T_K,
    )
    assert mismatched.f_tl[1] == pytest.approx(_MOE_PHI_C, rel=1e-12)
    assert abs(mismatched.f_tl[1] - 0.029965753424657536) > 0.2


@pytest.mark.reference_pinned
def test_paper_default_parameters_put_the_clamp_at_the_stated_cooling_rate():
    """Sim et al. (2024) Section 2.2 give phi_c = 0.3, tau = 1 Ma and
    DeltaT = 100 C, and bound F_tl above by phi_c. The published model at
    https://github.com/joycesim/MOE sets the same three, its tau of 3.15e13 s
    being 0.999 Ma. Those defaults make the
    prefactor 3000 yr K-1, so a mantle cooling at exactly DeltaT / tau, which is
    1e-4 K/yr, sits on the disaggregation limit without being clamped."""
    assert _tc.DEFAULT_PHI_C == pytest.approx(0.3, rel=1e-12)
    assert _tc.DEFAULT_TAU_YR == pytest.approx(1.0e6, rel=1e-12)
    assert _tc.DEFAULT_DELTA_T_K == pytest.approx(100.0, rel=1e-12)
    assert _tc.DEFAULT_TEMPERATURE_COLUMN == 'T_pot'

    # Boundary value: 1e4 yr at -1 K gives dT/dt = -1e-4 K/yr exactly.
    boundary = _tc.dynamic_trapped_fraction(np.array([0.0, 1.0e4]), np.array([2000.0, 1999.0]))
    assert boundary.f_tl_raw[1] == pytest.approx(_tc.DEFAULT_PHI_C, rel=1e-12)
    assert boundary.n_clamped_high == 0
    # Unit guard: had tau been read as 1 Ma in seconds against a Time column in
    # years, the prefactor would be ~3e10 and this step would clamp instead.
    assert boundary.f_tl[1] < 1.0

    # Doubling the cooling rate doubles the unclamped fraction, which fixes the
    # linearity of the relation independently of the prefactor.
    faster = _tc.dynamic_trapped_fraction(np.array([0.0, 1.0e4]), np.array([2000.0, 1998.0]))
    assert faster.f_tl_raw[1] == pytest.approx(2.0 * boundary.f_tl_raw[1], rel=1e-12)
    assert faster.n_clamped_high == 1


@pytest.mark.physics_invariant
def test_three_modes_share_the_mass_balance_and_order_by_trapped_fraction():
    """All three modes run the same bracket on the same helpfile and differ only
    in F_tl, so on a monotonically cooling run whose dynamic fractions exceed
    the constant one, the totals order none < constant < dynamic for every
    species."""
    frame = _dynamic_helpfile()
    results = _tc.compare_modes(frame, f_tl=0.02, **_DYN_KWARGS)
    assert sorted(results) == sorted(_tc.TRAPPING_MODES)

    for name, constant_total, dynamic_total in (
        ('H2O', 6.4998e16, 3.34539e17),
        ('CO2', 1.2e17, 6.6e17),
    ):
        totals = {
            mode: next(s.total_trapped for s in results[mode].species if s.name == name)
            for mode in _tc.TRAPPING_MODES
        }
        assert totals['none'] == pytest.approx(0.0, abs=1.0e-30)
        assert totals['constant'] == pytest.approx(constant_total, rel=1e-12)
        assert totals['dynamic'] == pytest.approx(dynamic_total, rel=1e-12)
        assert totals['none'] < totals['constant'] < totals['dynamic']

    # The comparison summary names every mode and species it reports on.
    text = _tc.summarise_comparison(results)
    for token in ('none', 'constant', 'dynamic', 'H2O', 'CO2'):
        assert token in text

    # Error contract: an unknown mode is refused by name rather than falling
    # back to a default, so a typo cannot silently produce constant-mode output.
    with pytest.raises(ValueError, match='Unknown trapping mode'):
        _tc.compute_trapping(frame, mode='Dynamic', **_DYN_KWARGS)


def test_dynamic_mode_requires_the_temperature_column_by_name():
    """Dynamic mode differences a named temperature column, and a helpfile
    without it cannot supply dT/dt. The calculation refuses it by name instead
    of substituting another temperature, while the modes that never form a
    cooling rate run on the same helpfile unaffected."""
    stripped = _dynamic_helpfile().drop(columns=['T_pot'])
    with pytest.raises(_tc.MissingColumnError, match='T_pot'):
        _tc.compute_trapping(stripped, mode='dynamic', **_DYN_KWARGS)

    # The other two modes do not read the column, so they still compute.
    for mode in ('none', 'constant'):
        result = _tc.compute_trapping(stripped, mode=mode, f_tl=0.02, **_DYN_KWARGS)
        assert result.dynamic is None
        assert len(result.species) == 2

    # An explicitly selected column is honoured, and a missing one is named in
    # the same way as the default.
    with_magma = _dynamic_helpfile().rename(columns={'T_pot': 'T_magma'})
    result = _tc.compute_trapping(
        with_magma, mode='dynamic', temperature_column='T_magma', **_DYN_KWARGS
    )
    assert result.dynamic.column == 'T_magma'
    assert result.species[0].total_trapped == pytest.approx(3.34539e17, rel=1e-12)
    with pytest.raises(_tc.MissingColumnError, match='T_surf'):
        _tc.compute_trapping(
            with_magma, mode='dynamic', temperature_column='T_surf', **_DYN_KWARGS
        )


def test_per_step_table_carries_the_mode_and_the_dynamic_diagnostics():
    """The table names the mode on every row so runs can be concatenated and
    grouped, and dynamic runs also carry dT/dt and both the raw and clamped
    F_tl so the fraction that produced each mass can be inspected."""
    frame = _dynamic_helpfile()
    dynamic_table = _tc.build_table(
        _tc.compute_trapping(frame, mode='dynamic', **_DYN_KWARGS), frame
    )
    assert list(dynamic_table['mode'].unique()) == ['dynamic']
    for column in ('T_pot_dT_dt', 'F_tl_raw', 'F_tl', 'dm_trap_H2O', 'cum_trap_H2O'):
        assert column in dynamic_table.columns
    np.testing.assert_allclose(
        dynamic_table['F_tl'].to_numpy(dtype=float),
        [0.0, 0.03, 0.3, 0.0],
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        dynamic_table['T_pot_dT_dt'].to_numpy(dtype=float),
        [0.0, -0.01, -0.5, 0.1],
        rtol=1e-12,
        atol=1e-15,
    )

    # Constant mode reports the scalar it used, broadcast over every row, so the
    # three tables share a schema up to the dynamic-only columns.
    constant_table = _tc.build_table(
        _tc.compute_trapping(frame, mode='constant', f_tl=0.02, **_DYN_KWARGS), frame
    )
    assert constant_table['F_tl'].to_numpy(dtype=float) == pytest.approx(0.02, rel=1e-12)
    assert 'F_tl_raw' not in constant_table.columns

    # Edge case: the none mode reports a zero fraction rather than omitting the
    # column, so a concatenation of all three modes has no missing values.
    none_table = _tc.build_table(_tc.compute_trapping(frame, mode='none', **_DYN_KWARGS), frame)
    assert none_table['F_tl'].to_numpy(dtype=float) == pytest.approx(0.0, abs=1.0e-30)
    assert (
        not pd.concat([dynamic_table, constant_table, none_table])['dm_trap_H2O'].isna().any()
    )


@pytest.mark.physics_invariant
def test_both_mode_pairs_the_prescriptions_without_altering_either():
    """The combined mode reports two independent runs rather than a blend of
    them, so each half reproduces the single-mode total for that species
    exactly and the pair still orders constant below dynamic on a cooling run."""
    frame = _dynamic_helpfile()
    combined = _tc.compute_both(frame, f_tl=0.02, **_DYN_KWARGS)

    assert combined.mode == 'both'
    assert list(combined.results()) == list(_tc.COMBINED_PAIR)
    assert combined.constant.mode == 'constant'
    assert combined.dynamic.mode == 'dynamic'

    # Each half equals what that mode produces alone. The values are the same
    # ones pinned in the three-mode comparison, so a regression that averaged
    # the two would land at 2.0e17 for H2O and match neither number here.
    for name, constant_total, dynamic_total in (
        ('H2O', 6.4998e16, 3.34539e17),
        ('CO2', 1.2e17, 6.6e17),
    ):
        from_both = {
            mode: next(s.total_trapped for s in result.species if s.name == name)
            for mode, result in combined.results().items()
        }
        alone = {
            mode: next(
                s.total_trapped
                for s in _tc.compute_trapping(
                    frame, mode=mode, f_tl=0.02, **_DYN_KWARGS
                ).species
                if s.name == name
            )
            for mode in _tc.COMBINED_PAIR
        }
        assert from_both['constant'] == pytest.approx(constant_total, rel=1e-12)
        assert from_both['dynamic'] == pytest.approx(dynamic_total, rel=1e-12)
        assert from_both['constant'] == pytest.approx(alone['constant'], rel=1e-15)
        assert from_both['dynamic'] == pytest.approx(alone['dynamic'], rel=1e-15)
        assert from_both['constant'] < from_both['dynamic']
        # Neither half exceeds the inventory it draws on.
        for mode in _tc.COMBINED_PAIR:
            inventory = next(
                s.inventory_final for s in combined.results()[mode].species if s.name == name
            )
            assert from_both[mode] <= inventory

    # Error contract: the combined mode is not a single result, and routing it
    # through the single-result path is refused by name rather than silently
    # falling back to constant.
    with pytest.raises(ValueError, match='compute_both'):
        _tc.compute_trapping(frame, mode='both', **_DYN_KWARGS)
    assert 'both' not in _tc.TRAPPING_MODES
    assert 'both' in _tc.CLI_MODES

    # Edge case: the dynamic half needs a temperature column, so a helpfile
    # without one fails the pair rather than returning a usable constant half.
    with pytest.raises(_tc.MissingColumnError, match='T_pot'):
        _tc.compute_both(frame.drop(columns=['T_pot']), f_tl=0.02, **_DYN_KWARGS)


def test_combined_table_stacks_both_modes_and_keeps_their_own_columns():
    """The combined table is the two single-mode tables stacked and told apart
    by the mode column, so the dynamic diagnostics survive and the constant
    rows carry the scalar rather than being padded with the dynamic series."""
    frame = _dynamic_helpfile()
    combined = _tc.compute_both(frame, f_tl=0.02, **_DYN_KWARGS)
    table = _tc.build_combined_table(combined, frame)

    assert len(table) == 2 * len(frame)
    assert list(table['mode'].unique()) == ['constant', 'dynamic']

    constant_rows = table[table['mode'] == 'constant']
    dynamic_rows = table[table['mode'] == 'dynamic']
    assert constant_rows['F_tl'].to_numpy(dtype=float) == pytest.approx(0.02, rel=1e-12)
    np.testing.assert_allclose(
        dynamic_rows['F_tl'].to_numpy(dtype=float),
        [0.0, 0.03, 0.3, 0.0],
        rtol=1e-12,
        atol=1e-15,
    )
    # The dynamic-only diagnostics exist and belong to the dynamic rows alone.
    # A regression that broadcast one mode's fraction across both blocks would
    # leave F_tl_raw populated here.
    assert constant_rows['F_tl_raw'].isna().all()
    assert np.isfinite(dynamic_rows['F_tl_raw'].to_numpy(dtype=float)).all()
    assert dynamic_rows['F_tl_raw'].to_numpy(dtype=float)[2] == pytest.approx(1.5, rel=1e-12)

    # The trapped masses differ between the blocks, which is the point of
    # running both: identical blocks would mean one mode overwrote the other.
    assert constant_rows['cum_trap_H2O'].to_numpy(dtype=float)[-1] == pytest.approx(
        6.4998e16, rel=1e-12
    )
    assert dynamic_rows['cum_trap_H2O'].to_numpy(dtype=float)[-1] == pytest.approx(
        3.34539e17, rel=1e-12
    )
    # Edge case: the stacked frame has no missing trapped mass anywhere, so a
    # reader can total the column without first filling gaps.
    assert not table['dm_trap_H2O'].isna().any()


def test_both_mode_summary_reports_each_mode_and_their_ratio():
    """The combined summary carries both modes' own reports plus the two-mode
    totals table, and the comparison covers exactly the pair that was run."""
    frame = _dynamic_helpfile()
    combined = _tc.compute_both(frame, f_tl=0.02, **_DYN_KWARGS)
    text = _tc.summarise_both(combined)

    assert 'constant, F_tl = 0.02' in text
    assert 'dynamic, F_tl in [0 to 0.3]' in text
    for token in ('H2O', 'CO2', 'Trapped mass by mode', 'Relative to constant mode'):
        assert token in text
    # The pair excludes the zero baseline, so the none column must not appear
    # in the two-mode totals block even though the mode still exists.
    totals_block = text.split('Trapped mass by mode [kg]')[1]
    assert 'none' not in totals_block

    # The dynamic-to-constant ratio is the quotient of the pinned totals,
    # 3.34539e17 / 6.4998e16 = 5.1469..., so the ratio row is not a copy of
    # the constant column.
    ratio_line = [line for line in text.splitlines() if line.strip().startswith('H2O')][-1]
    assert '5.1469' in ratio_line

    # Restricting the general comparison to one mode still works, which is the
    # contract the pair relies on.
    single = _tc.summarise_comparison(combined.results(), modes=('dynamic',))
    assert 'dynamic' in single and 'constant' not in single.split('\n')[2]


def _captured_figure(monkeypatch, plt, draw):
    """Run a plot function and return the figure it drew, still open.

    Both plot functions close their figure before returning, so the only way to
    inspect what they drew is to intercept the close. The file is still written.
    """
    captured = []
    monkeypatch.setattr(plt, 'close', captured.append)
    draw()
    assert len(captured) == 1
    return captured[0]


def test_plots_carry_trapped_mass_alone_on_a_single_axis(tmp_path, monkeypatch):
    """Both figures draw cumulative trapped mass and nothing else: one axis, one
    unit, and no trapped-melt-fraction series or disaggregation bound overlaid on
    it. Those are drawn by tools/plot_trapping_fraction.py instead."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frame = _dynamic_helpfile()
    combined = _tc.compute_both(frame, f_tl=0.02, **_DYN_KWARGS)
    destination = tmp_path / 'combined.png'
    figure = _captured_figure(
        monkeypatch, plt, lambda: _tc.plot_combined(combined, destination)
    )

    assert destination.exists()
    written = destination.read_bytes()
    assert written[:8] == b'\x89PNG\r\n\x1a\n'
    assert len(written) > 10_000

    # A twin axis would appear here as a second entry; the F_tl overlay was the
    # only thing that ever created one.
    assert len(figure.axes) == 1
    axes = figure.axes[0]
    assert axes.get_ylabel() == 'Cumulative trapped mass [kg]'
    # Two species drawn for each of the two modes, and nothing else: an axhline
    # for phi_c or a level line for the constant fraction would raise the count.
    assert len(axes.lines) == 4
    # Scale guard on what was drawn. Trapped masses here are of order 1e16 kg,
    # while every fraction the removed overlay drew is bounded by phi_c = 0.3,
    # so a fraction series reintroduced on these axes would show up as a curve
    # whose largest value is far below any mass curve's.
    peaks = [float(np.max(line.get_ydata())) for line in axes.lines]
    assert min(peaks) > 1.0e15
    plt.close(figure)

    # The single-mode figure drew the conditional overlay when the mode was
    # dynamic, so that is the case to check: mass curves only, still one axis.
    dynamic_plot = tmp_path / 'dynamic.png'
    figure = _captured_figure(
        monkeypatch, plt, lambda: _tc.plot_cumulative(combined.dynamic, dynamic_plot)
    )
    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 2
    assert min(float(np.max(line.get_ydata())) for line in figure.axes[0].lines) > 1.0e15
    plt.close(figure)

    # Limit case: the no-trapping mode traps identically zero, so every curve is
    # flat at the origin. The figure is still one axis with one curve per
    # species rather than an empty or a rescaled one.
    empty = _tc.compute_trapping(frame, mode='none', **_DYN_KWARGS)
    figure = _captured_figure(
        monkeypatch, plt, lambda: _tc.plot_cumulative(empty, tmp_path / 'none.png')
    )
    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 2
    assert max(float(np.max(line.get_ydata())) for line in figure.axes[0].lines) == 0.0
    plt.close(figure)


def test_log_time_axis_leaves_a_run_without_positive_times_linear():
    """The time axis goes logarithmic from the first positive sample, since a
    helpfile starts at t = 0, and a run with no positive time at all is left on
    the linear default rather than raising or producing an empty axis."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    _tc.log_time_axis(axes, np.array([0.0, 1.0e2, 2.0e2]))
    assert axes.get_xscale() == 'log'
    # The lower limit is the first positive time, not the first row, which a
    # log axis could not display.
    assert axes.get_xlim()[0] == pytest.approx(1.0e2, rel=1e-12)
    plt.close(figure)

    # Edge case: every time is zero, so there is nothing a log axis can show.
    figure, axes = plt.subplots()
    _tc.log_time_axis(axes, np.zeros(3))
    assert axes.get_xscale() == 'linear'
    plt.close(figure)

    # Edge case: an empty series is handled the same way rather than indexing
    # into an empty array.
    figure, axes = plt.subplots()
    _tc.log_time_axis(axes, np.array([]))
    assert axes.get_xscale() == 'linear'
    plt.close(figure)


def test_both_mode_runs_end_to_end_from_the_command_line(tmp_path):
    """The command line accepts the combined mode, writes one stacked table and
    one plot for it, and refuses a helpfile whose temperature column is absent
    rather than quietly reporting only the constant half."""
    frame = _dynamic_helpfile()
    helpfile = tmp_path / 'runtime_helpfile.csv'
    frame.to_csv(helpfile, sep='\t', index=False)
    table_path = tmp_path / 'table.csv'
    plot_path = tmp_path / 'plot.png'

    code = _tc.main(
        [
            '-i',
            str(helpfile),
            '--mode',
            'both',
            '--f-tl',
            '0.02',
            '--phi-c',
            str(_DYN_PHI_C),
            '--tau',
            str(_DYN_TAU_YR),
            '--delta-t',
            str(_DYN_DELTA_T_K),
            '--output-csv',
            str(table_path),
            '--plot',
            str(plot_path),
        ]
    )
    assert code == 0
    assert plot_path.exists() and plot_path.stat().st_size > 10_000
    written = pd.read_csv(table_path)
    assert sorted(written['mode'].unique()) == ['constant', 'dynamic']
    assert len(written) == 2 * len(frame)

    # Edge case: --no-plot writes the table and nothing else, so the combined
    # mode is usable where matplotlib is unwanted.
    second_plot = tmp_path / 'absent.png'
    code = _tc.main(
        [
            '-i',
            str(helpfile),
            '--mode',
            'both',
            '--no-plot',
            '--output-csv',
            str(tmp_path / 'table2.csv'),
            '--plot',
            str(second_plot),
        ]
    )
    assert code == 0
    assert not second_plot.exists()

    # Error contract: the dynamic half needs its temperature column, and the
    # run fails by name instead of degrading to a constant-only report.
    bare = tmp_path / 'bare.csv'
    frame.drop(columns=['T_pot']).to_csv(bare, sep='\t', index=False)
    with pytest.raises(_tc.MissingColumnError, match='T_pot'):
        _tc.main(
            [
                '-i',
                str(bare),
                '--mode',
                'both',
                '--no-plot',
                '--output-csv',
                str(tmp_path / 'table3.csv'),
            ]
        )


# Helpfile carrying four of the tracked volatiles with melt concentrations that
# are constant by construction, so every trapped mass below reduces to
# D_eff * C_Z * dM_RM with no interpolation to reason about. H2O is the only
# species with a non-zero partition coefficient; CO and N2 stand for the
# molecular species that carry none; Xe is present but never outgassed, which is
# the case a full-species run meets on most helpfiles.
_MULTI_TIME = [0.0, 1.0e2, 2.0e2, 3.0e2]
_MULTI_SOLID = [0.0, 1.0e21, 2.0e21, 3.0e21]
_MULTI_LIQUID = [4.0e21, 3.0e21, 2.0e21, 1.0e21]
# Melt mass fractions the species columns are built to hold at every row.
_MULTI_C_Z = {'H2O': 1.0e-3, 'CO': 5.0e-4, 'N2': 2.0e-4, 'Xe': 0.0}
_MULTI_TOTAL = {'H2O': 5.0e18, 'CO': 2.5e18, 'N2': 1.0e18, 'Xe': 1.0e15}
# Every step crystallises the same mass, which makes the per-step trapped mass
# constant and the total a clean multiple of it.
_MULTI_DM_RM = 1.0e21
_MULTI_STEPS = 3


def _multi_species_helpfile() -> pd.DataFrame:
    """Four-row helpfile over four species, three of them at D_Z = 0."""
    data: dict[str, list[float]] = {
        'Time': list(_MULTI_TIME),
        'M_mantle': [4.0e21] * 4,
        'M_mantle_solid': list(_MULTI_SOLID),
        'M_mantle_liquid': list(_MULTI_LIQUID),
        'Phi_global': [liquid / 4.0e21 for liquid in _MULTI_LIQUID],
        'T_pot': [3000.0, 2900.0, 2800.0, 2700.0],
    }
    for name, concentration in _MULTI_C_Z.items():
        data[f'{name}_kg_liquid'] = [concentration * melt for melt in _MULTI_LIQUID]
        data[f'{name}_kg_total'] = [_MULTI_TOTAL[name]] * 4
    return pd.DataFrame(data)


def _wide_helpfile(names: list[str]) -> pd.DataFrame:
    """Helpfile over an arbitrary species list, each at a distinct concentration.

    Concentrations decrease down the list so the trapped-mass ranking the plot
    cap applies is known in advance and is not the declaration order.
    """
    data: dict[str, list[float]] = {
        'Time': list(_MULTI_TIME),
        'M_mantle': [4.0e21] * 4,
        'M_mantle_solid': list(_MULTI_SOLID),
        'M_mantle_liquid': list(_MULTI_LIQUID),
        'Phi_global': [liquid / 4.0e21 for liquid in _MULTI_LIQUID],
        'T_pot': [3000.0, 2900.0, 2800.0, 2700.0],
    }
    for rank, name in enumerate(names):
        concentration = 1.0e-3 / float(rank + 1)
        data[f'{name}_kg_liquid'] = [concentration * melt for melt in _MULTI_LIQUID]
        data[f'{name}_kg_total'] = [concentration * 5.0e21] * 4
    return pd.DataFrame(data)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_every_tracked_volatile_carries_a_coefficient_and_only_water_is_non_zero():
    """The default set covers every volatile the helpfile can supply: the
    CALLIOPE volatile species plus the noble gases. Only hydrogen has a measured
    crystal/melt coefficient, so water alone is non-zero and every molecular
    species sits at zero as a statement about lattice incorporation."""
    expected = {
        'H2O',
        'CO2',
        'O2',
        'H2',
        'CH4',
        'CO',
        'N2',
        'NH3',
        'S2',
        'SO2',
        'H2S',
        'He',
        'Ne',
        'Ar',
        'Kr',
        'Xe',
    }
    assert set(_tc.DEFAULT_D_Z) == expected
    # The olivine/melt hydrogen coefficient of Aubaud et al. (2004). Pinned
    # because the whole crystal term scales with it.
    assert _tc.DEFAULT_D_Z['H2O'] == pytest.approx(0.0017, rel=1e-12)
    # Discrimination guard: the pyroxene coefficients of the same work (opx
    # 0.019, cpx 0.023) are an order of magnitude larger, so a default swapped
    # to one of those would land far outside this bound.
    assert 1.0e-4 < _tc.DEFAULT_D_Z['H2O'] < 5.0e-3
    # Every other species is exactly zero, not merely small: a value smuggled in
    # for CO or NH3 would have no experimental basis.
    others = [value for name, value in _tc.DEFAULT_D_Z.items() if name != 'H2O']
    assert max(abs(value) for value in others) < 1.0e-18
    assert len(others) == len(expected) - 1
    # Water stays first so the two-species behaviour the defaults previously
    # described is the leading edge of the wider set rather than a reordering.
    assert list(_tc.DEFAULT_D_Z)[:2] == ['H2O', 'CO2']


@pytest.mark.physics_invariant
def test_zero_coefficient_species_still_trap_the_interstitial_melt():
    """A species with no lattice incorporation is still buried, because the
    trapped interstitial melt carries whatever is dissolved in it. On a helpfile
    with constant melt concentrations the buried mass is F_tl * C_Z * dM_RM per
    step, which is hand-computable and independent of the crystal chemistry."""
    frame = _multi_species_helpfile()
    result = _tc.compute_trapping(frame, mode='constant', f_tl=0.02, d_z=dict(_tc.DEFAULT_D_Z))
    by_name = {species.name: species for species in result.species}
    assert sorted(by_name) == ['CO', 'H2O', 'N2', 'Xe']

    # CO carries D_Z = 0, so D_eff collapses to F_tl and the trapped mass is
    # 0.02 * 5e-4 * 1e21 = 1e16 kg per step over three crystallising steps.
    carbon = by_name['CO']
    expected_co = 0.02 * _MULTI_C_Z['CO'] * _MULTI_DM_RM * _MULTI_STEPS
    assert carbon.total_trapped == pytest.approx(expected_co, rel=1e-12)
    assert carbon.d_eff == pytest.approx(0.02, rel=1e-12)
    # Sign guard: burial adds mass to the solid, so the total is positive.
    assert carbon.total_trapped > 0.0
    # Scale guard: 3e16 kg, not 3e13 (a gram/kilogram slip) or 3e19 (the melt
    # inventory itself rather than the trapped share).
    assert 1.0e16 < carbon.total_trapped < 1.0e17
    # Formula guard: a regression that used D_Z in place of the bracket would
    # bury nothing at all here, since D_Z is zero for CO.
    assert carbon.total_trapped > 1.0e16

    # N2 is also at D_Z = 0 but at two-fifths the concentration, so the ratio of
    # the two totals is the ratio of the concentrations and nothing else.
    nitrogen = by_name['N2']
    assert nitrogen.total_trapped / carbon.total_trapped == pytest.approx(
        _MULTI_C_Z['N2'] / _MULTI_C_Z['CO'], rel=1e-12
    )

    # Water is the one species with a crystal term, so it is buried at a larger
    # effective coefficient than the shared interstitial one.
    water = by_name['H2O']
    assert water.d_eff == pytest.approx(0.98 * 0.0017 + 0.02, rel=1e-12)
    assert water.d_eff > carbon.d_eff
    # Weighting guard: dropping the (1 - F_tl) factor would give 0.0217 and a
    # total 3.3e13 kg larger, which the tolerance above resolves.
    unweighted = (0.0017 + 0.02) * _MULTI_C_Z['H2O'] * _MULTI_DM_RM * _MULTI_STEPS
    assert abs(water.total_trapped - unweighted) > 1.0e13

    # Boundedness: no species is buried beyond the inventory it belongs to.
    for species in result.species:
        assert species.total_trapped <= species.inventory_final

    # Edge case: a species present in the helpfile but never outgassed has zero
    # melt concentration, so the interstitial term buries nothing for it.
    assert by_name['Xe'].total_trapped == pytest.approx(0.0, abs=1.0e-30)
    assert by_name['Xe'].d_eff == pytest.approx(0.02, rel=1e-12)


@pytest.mark.physics_invariant
def test_zero_coefficient_and_the_no_trapping_mode_are_different_results():
    """D_Z = 0 removes the crystal term, not the trapping: the interstitial melt
    is still buried. The none mode removes the process itself. Collapsing the two
    would silently zero every molecular species, which is the whole species set
    bar water, so the distinction is pinned on both sides."""
    frame = _multi_species_helpfile()
    coefficients = {'H2O': 0.0017, 'CO': 0.0}
    trapped = _tc.compute_trapping(frame, mode='constant', f_tl=0.02, d_z=coefficients)
    absent = _tc.compute_trapping(frame, mode='none', f_tl=0.02, d_z=coefficients)
    by_trapped = {s.name: s for s in trapped.species}
    by_absent = {s.name: s for s in absent.species}

    # Interstitial-only trapping for the D_Z = 0 species against exactly zero.
    expected_co = 0.02 * _MULTI_C_Z['CO'] * _MULTI_DM_RM * _MULTI_STEPS
    assert by_trapped['CO'].total_trapped == pytest.approx(expected_co, rel=1e-12)
    assert by_absent['CO'].total_trapped == pytest.approx(0.0, abs=1.0e-30)
    assert by_trapped['CO'].total_trapped - by_absent['CO'].total_trapped > 2.9e16
    # The bracket is evaluated in one case and not the other, which is the
    # structural difference behind the numeric one.
    assert by_trapped['CO'].d_eff == pytest.approx(0.02, rel=1e-12)
    assert by_absent['CO'].d_eff is None
    assert trapped.f_tl == pytest.approx(0.02, rel=1e-12)
    assert absent.f_tl is None

    # The none mode is not the crystal term on its own either: water carries a
    # non-zero D_Z and is still buried at exactly nothing.
    assert by_absent['H2O'].total_trapped == pytest.approx(0.0, abs=1.0e-30)

    # Limit input: F_tl = 0 in the constant mode is the third case, distinct
    # from both. The bracket is evaluated and collapses to D_Z, so water is
    # still buried at 0.0017 * 1e-3 * 1e21 * 3 = 5.1e15 kg while CO, having no
    # crystal term to fall back on, is buried at nothing.
    edge = _tc.compute_trapping(frame, mode='constant', f_tl=0.0, d_z=coefficients)
    by_edge = {s.name: s for s in edge.species}
    expected_water = 0.0017 * _MULTI_C_Z['H2O'] * _MULTI_DM_RM * _MULTI_STEPS
    assert by_edge['H2O'].total_trapped == pytest.approx(expected_water, rel=1e-12)
    assert 1.0e15 < by_edge['H2O'].total_trapped < 1.0e16
    assert by_edge['CO'].total_trapped == pytest.approx(0.0, abs=1.0e-30)
    assert by_edge['CO'].d_eff == pytest.approx(0.0, abs=1.0e-18)


def test_absent_species_are_skipped_without_changing_the_species_that_remain():
    """Running the full default set against a helpfile that tracked only a few
    species reports the rest as skipped and leaves the surviving totals bit for
    bit what a run naming only those species produces."""
    frame = _multi_species_helpfile()
    full = _tc.compute_trapping(frame, mode='constant', f_tl=0.02)
    present = ['H2O', 'CO', 'N2', 'Xe']

    # Order follows the default dictionary, not the helpfile column order.
    assert [species.name for species in full.species] == present
    assert full.skipped == [name for name in _tc.DEFAULT_D_Z if name not in present]
    assert len(full.skipped) == len(_tc.DEFAULT_D_Z) - len(present)

    restricted = _tc.compute_trapping(
        frame,
        mode='constant',
        f_tl=0.02,
        d_z={name: _tc.DEFAULT_D_Z[name] for name in present},
    )
    for wide, narrow in zip(full.species, restricted.species):
        assert wide.name == narrow.name
        assert wide.total_trapped == pytest.approx(narrow.total_trapped, rel=1e-12)
        np.testing.assert_allclose(wide.dm_trap, narrow.dm_trap, rtol=1e-12, atol=0.0)
    assert restricted.skipped == []

    # Error contract: a species needs both of its columns. One alone is not
    # enough to form a concentration, so it is skipped rather than half-used.
    half = frame.drop(columns=['CO_kg_total'])
    partial = _tc.compute_trapping(half, mode='constant', f_tl=0.02)
    assert 'CO' in partial.skipped
    assert [species.name for species in partial.species] == ['H2O', 'N2', 'Xe']
    # Edge case: the surviving species are untouched by the loss of a neighbour.
    assert partial.species[0].total_trapped == pytest.approx(
        full.species[0].total_trapped, rel=1e-12
    )


def test_plot_cap_ranks_by_trapped_mass_and_never_repeats_a_style(tmp_path, monkeypatch):
    """The full species set outnumbers the colour palette, so the figure draws
    the largest contributors and names the rest instead of wrapping the palette
    and giving two species the same colour."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = ['H2O', 'CO2', 'O2', 'H2', 'CH4', 'CO', 'N2', 'NH3', 'S2', 'SO2']
    frame = _wide_helpfile(names)
    result = _tc.compute_trapping(frame, mode='constant', f_tl=0.02)
    assert len(result.species) == len(names)

    drawn, omitted = _tc.rank_species(result.species, _tc.DEFAULT_PLOT_TOP)
    # Concentrations fall down the list, so the ranking is the declaration order
    # here and the two smallest contributors are the ones dropped.
    assert [species.name for species in drawn] == names[: _tc.DEFAULT_PLOT_TOP]
    assert [species.name for species in omitted] == names[_tc.DEFAULT_PLOT_TOP :]
    assert drawn[0].total_trapped > drawn[-1].total_trapped
    assert min(s.total_trapped for s in drawn) >= max(s.total_trapped for s in omitted)

    note = _tc.describe_omitted(drawn, omitted)
    for name in names[_tc.DEFAULT_PLOT_TOP :]:
        assert name in note
    assert '8 of 10' in note
    assert _tc.describe_omitted(result.species, []) == ''

    figure = _captured_figure(
        monkeypatch, plt, lambda: _tc.plot_cumulative(result, tmp_path / 'wide.png')
    )
    axes = figure.axes[0]
    assert len(axes.lines) == _tc.DEFAULT_PLOT_TOP
    colours = [line.get_color() for line in axes.lines]
    assert len(set(colours)) == len(colours)
    assert axes.get_legend().get_title().get_text() == '+2 not shown'
    plt.close(figure)

    # Edge case: lifting the cap draws every species, and the style cycle keeps
    # each curve distinguishable past the eight colours the palette holds.
    figure = _captured_figure(
        monkeypatch, plt, lambda: _tc.plot_cumulative(result, tmp_path / 'all.png', top=0)
    )
    axes = figure.axes[0]
    assert len(axes.lines) == len(names)
    pairs = [(line.get_color(), line.get_linestyle()) for line in axes.lines]
    assert len(set(pairs)) == len(pairs)
    plt.close(figure)

    # Line style is spent on the mode in the combined figure, so the cap there
    # cannot exceed the palette however it is asked for.
    combined = _tc.compute_both(
        frame, f_tl=0.02, phi_c=_DYN_PHI_C, tau=_DYN_TAU_YR, delta_t=_DYN_DELTA_T_K
    )
    assert len(combined.constant.species) == len(names)
    figure = _captured_figure(
        monkeypatch,
        plt,
        lambda: _tc.plot_combined(combined, tmp_path / 'combined_wide.png', top=0),
    )
    axes = figure.axes[0]
    assert len(axes.lines) == 2 * len(_tc.WONG_COLOURS)
    assert len({line.get_color() for line in axes.lines}) == len(_tc.WONG_COLOURS)
    plt.close(figure)


def test_summary_groups_species_that_trap_nothing_but_keeps_the_zero_baseline_whole():
    """Most of the species set traps nothing on a given run, so those are named
    on one line while the species that do trap keep their full report. A run in
    which nothing traps anywhere is the zero baseline itself and keeps every
    block, so the baseline stays legible rather than collapsing to a name list."""
    frame = _multi_species_helpfile()
    result = _tc.compute_trapping(frame, mode='constant', f_tl=0.02)
    text = _tc.summarise(result)

    # Xe is tracked but never outgassed, so it traps nothing and is grouped.
    assert 'trapped nothing            : Xe' in text
    assert 'Xe: D_Z' not in text
    for name in ('H2O', 'CO', 'N2'):
        assert f'{name}: D_Z' in text
    # The skipped species are a different category and stay on their own line:
    # absent from the helpfile, not merely un-outgassed.
    assert 'skipped (no columns)' in text
    assert 'CO2' in text.split('skipped (no columns)')[1].split('\n')[0]

    # Limit case: the none mode buries nothing anywhere, so no species is
    # singled out as the one that trapped nothing and every block is kept.
    baseline = _tc.summarise(_tc.compute_trapping(frame, mode='none', f_tl=0.02))
    assert 'trapped nothing' not in baseline
    for name in ('H2O', 'CO', 'N2', 'Xe'):
        assert f'{name}: D_Z' in baseline
    assert baseline.count('supply-clamped') == 4
