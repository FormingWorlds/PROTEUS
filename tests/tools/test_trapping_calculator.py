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
* the missing-column error contract and the skip path for absent species,
* delimiter detection, since PROTEUS writes the helpfile tab-separated.

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
