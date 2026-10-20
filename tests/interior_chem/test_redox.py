"""Melt Fe3+/Fe2+ redox tracking (``src/proteus/interior_chem/redox.py``).

Covers the initial ferric fraction ``f_0`` now that it is a user-settable
config field (``planet.ferric_fraction_initial``) rather than a module
constant. Contract clauses exercised here:

* Step 1-2 (Eq 1-3): Fe2+ is seeded from the ferrous FeO fraction,
  ``W_FET * M_melt / MU_FEO``, and Fe3+ is added on top so that the ferric
  fraction equals the value the caller asked for.
* The open-interval domain of ``f_0``: the tracker forms ``f/(1-f)``, so
  the endpoints are rejected rather than silently producing a degenerate
  or infinite redox ratio.
* End-to-end propagation: a non-default value in the config reaches
  ``hf_row['ferric_frac_mantle']`` on the first coupling step.
* The no-op contract: the tracker does nothing unless
  ``planet.fO2_source == 'from_mantle_redox'``.
* Persistence: the Fe-metal diagnostics are helpfile schema columns and
  survive the write to and read from ``runtime_helpfile.csv``.

See docs/How-to/testing.md and docs/Explanations/test_framework.md.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from proteus.interior_chem.disproportionation import activity_Fe_metal
from proteus.interior_chem.redox import (
    BULK_D_FE2,
    BULK_D_FE3,
    BULK_P_BOUNDS_GPA,
    D_FE2_BRG,
    D_FE2_SHALLOW,
    D_FE3_BRG,
    D_FE3_SHALLOW,
    MU_FEO,
    MU_FEO15,
    P_CUTOFF_GPA,
    W_FET,
    _fe_into_new_solid,
    _init_state,
    _metal_saturation_step,
    _partition_coefficients,
    update_melt_redox,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Asymmetric mesh: one fully molten cell, one half molten, one solid, with
# unequal cell masses. Symmetric or all-equal input would let a bug that
# drops the phi weighting still reproduce the right melt mass.
_PHI = np.array([1.0, 0.5, 0.0])
_MASS = np.array([2.0e21, 4.0e21, 6.0e21])
# Straddles P_CUTOFF_GPA so both the Cpx/Opx and bridgmanite branches of
# the partition-coefficient split are exercised in every test.
_PRES = np.array([10.0e9, 30.0e9, 50.0e9])
# Monotonically increasing with depth and spanning 1200 K, so the coolest and
# hottest cells give clearly different equilibrium constants and the binding
# cell the metal step selects is unambiguous.
_TEMP = np.array([2200.0, 2800.0, 3400.0])

# Melt inventory for the direct chemistry comparisons below: bulk-silicate
# iron on a single-cation oxide basis, in moles per 100 g of melt.
_N_FET_TEST = 7.82 / 71.844
_N_SIL_TEST = _N_FET_TEST + 1.85945

# M_melt = 1.0*2e21 + 0.5*4e21 + 0.0*6e21 = 4e21 kg
_M_MELT = 4.0e21


def _iw_oneill_eggins_2002(T):
    """log10 fO2 of IW, O'Neill & Eggins (2002) Eq 11 [T in K]: an independent
    1 bar calibration, and the buffer the surface Delta-IW used before H21."""
    return 2 * (-244118 + 115.559 * T - 8.474 * T * np.log(T)) / (np.log(10) * 8.31441 * T)


def _make_config(
    f_0: float, source: str = 'from_mantle_redox', metal_saturation: bool = True
) -> MagicMock:
    """Minimal config exposing only the fields the tracker reads."""
    config = MagicMock()
    config.planet.fO2_source = source
    config.planet.ferric_fraction_initial = f_0
    config.planet.metal_saturation = metal_saturation
    config.planet.partition_coefficients = 'mgsio3'
    config.outgas.T_floor = 700.0  # the outgassing temperature floor default
    return config


def _make_interior() -> MagicMock:
    """Interior_t stand-in carrying the three radial profiles and no state."""
    interior = MagicMock()
    interior.phi = _PHI
    interior.mass = _MASS
    interior.pres = _PRES
    interior.temp = _TEMP
    interior.redox_state = None
    return interior


@pytest.mark.physics_invariant
def test_init_state_splits_the_melt_iron_inventory_by_the_requested_ferric_fraction():
    """Step 1-2: the ferrous FeO fraction fixes Fe2+ at W_FET * M_melt / MU_FEO,
    and Fe3+ is added on top so that Fe3+ / (Fe2+ + Fe3+) equals f_0.

    Uses f_0 = 0.25 rather than 0.5 so a swapped Fe3+/Fe2+ assignment lands
    at 0.75 and cannot pass, and so the two readings of W_FET (ferrous only
    versus total iron) differ by a factor 1 / (1 - f_0) = 1.33 in total iron.
    """
    f_0 = 0.25
    state = _init_state(_PHI, _MASS, _PRES, f_0)

    n_fe2_expected = W_FET * _M_MELT / MU_FEO  # mol of Fe2+ in the melt
    n_total = state.n_fe3_melt + state.n_fe2_melt

    # Fe2+ is set by the ferrous FeO fraction alone.
    assert state.n_fe2_melt == pytest.approx(n_fe2_expected, rel=1e-12)
    # The split is the requested fraction, not the module default of 0.1.
    assert state.n_fe3_melt / n_total == pytest.approx(f_0, rel=1e-12)
    assert state.ferric_frac == pytest.approx(f_0, rel=1e-12)
    # Closure: total iron is Fe2+ / (1 - f_0), i.e. Fe3+ sits on top of the
    # ferrous inventory rather than being carved out of it.
    assert n_total == pytest.approx(n_fe2_expected / (1.0 - f_0), rel=1e-12)
    # Discrimination guard: reading W_FET as total iron would give a total of
    # n_fe2_expected, 25% below the correct value, far outside rel=1e-12.
    assert abs(n_total / n_fe2_expected - 1.0) > 0.3
    # The FeO1.5 mass fraction carried into the metal step matches Fe3+.
    assert state.w_feo15 == pytest.approx(state.n_fe3_melt * MU_FEO15 / _M_MELT, rel=1e-12)
    # Swapped-assignment guard: Fe2+ is the majority reservoir at f_0 = 0.25,
    # so a transposed split would put 0.75 here and miss by 0.5.
    assert abs(state.n_fe3_melt / n_total - (1.0 - f_0)) > 0.4
    # Sign guard: mole counts are physical amounts, never negative.
    assert state.n_fe3_melt > 0
    assert state.n_fe2_melt > 0
    # Scale guard: ~5.9e21 mol, not ~5.9e18 (a g-vs-kg slip in MU_FEO) or
    # ~5.9e24 (dropping the phi weighting and using the whole mantle mass).
    assert 1e21 < n_total < 1e22


@pytest.mark.physics_invariant
def test_redox_ratio_is_the_odds_form_of_the_configured_ferric_fraction():
    """The seeded redox ratio is f/(1-f), and it increases monotonically with
    f_0. Pins the mapping the Hirschmann fO2 expression consumes via log10.

    f_0 = 0.2 gives 0.25 and f_0 = 0.8 gives 4.0: a factor of 16 apart, so a
    regression that returned f_0 itself (0.2 vs 0.8, factor 4) cannot pass.
    """
    low = _init_state(_PHI, _MASS, _PRES, 0.2)
    high = _init_state(_PHI, _MASS, _PRES, 0.8)

    assert low.redox_ratio == pytest.approx(0.2 / 0.8, rel=1e-12)
    assert high.redox_ratio == pytest.approx(0.8 / 0.2, rel=1e-12)
    # Monotonicity: a more oxidised melt has the larger Fe3+/Fe2+ ratio.
    assert high.redox_ratio > low.redox_ratio
    # Discrimination: returning f_0 unchanged would give 0.2 and 0.8, a
    # ratio of 4; the odds form gives 0.25 and 4.0, a ratio of 16.
    assert high.redox_ratio / low.redox_ratio == pytest.approx(16.0, rel=1e-12)
    # Both ratios are strictly positive, so log10 in the fO2 term is defined.
    assert low.redox_ratio > 0


@pytest.mark.parametrize('bad_f0', [0.0, 1.0, -0.1, 1.5, 2.0])
def test_init_state_rejects_a_ferric_fraction_outside_the_open_unit_interval(bad_f0):
    """Error contract: the endpoints and any value outside (0, 1) raise rather
    than seeding a degenerate state.

    f_0 = 1 divides by zero forming f/(1-f); f_0 = 0 sends log10(ratio) to
    -inf inside the fO2 expression. Both would otherwise propagate as a nan
    or inf Delta-IW several steps downstream of the real mistake.
    """
    with pytest.raises(ValueError, match=r'\(0, 1\)'):
        _init_state(_PHI, _MASS, _PRES, bad_f0)


def test_init_state_accepts_values_just_inside_the_open_interval():
    """Edge case: the interval is open, not empty. Values adjacent to the
    rejected endpoints are valid and produce a finite redox ratio.

    Pairs with the rejection test above: together they pin the boundary at
    exactly 0 and 1 rather than at some wider interior margin.
    """
    near_zero = _init_state(_PHI, _MASS, _PRES, 1e-6)
    near_one = _init_state(_PHI, _MASS, _PRES, 1.0 - 1e-6)

    assert np.isfinite(near_zero.redox_ratio)
    assert np.isfinite(near_one.redox_ratio)
    # The odds form diverges at 1, so the near-one ratio is enormous but
    # still finite; near zero it is tiny but still strictly positive.
    assert near_zero.redox_ratio > 0
    assert near_one.redox_ratio > 1e5


@pytest.mark.physics_invariant
def test_partition_coefficients_switch_at_the_bridgmanite_pressure_cutoff():
    """Step 1-2: the per-cell Fe2+ and Fe3+ partition coefficients take the
    bridgmanite values at and below the cutoff depth and the Cpx/Opx means
    above it.

    The mesh straddles P_CUTOFF_GPA (10, 30, 50 GPa against a 22 GPa cutoff),
    so a dropped or inverted comparison changes at least one cell.
    """
    state = _init_state(_PHI, _MASS, _PRES, 0.3)

    # 10 GPa is shallower than the 22 GPa cutoff: Cpx/Opx assemblage.
    assert state.D_fe3_cell[0] == pytest.approx(D_FE3_SHALLOW, rel=1e-12)
    assert state.D_fe2_cell[0] == pytest.approx(D_FE2_SHALLOW, rel=1e-12)
    # 30 and 50 GPa are deeper: bridgmanite.
    assert state.D_fe3_cell[1] == pytest.approx(D_FE3_BRG, rel=1e-12)
    assert state.D_fe3_cell[2] == pytest.approx(D_FE3_BRG, rel=1e-12)
    np.testing.assert_allclose(state.D_fe2_cell[1:], D_FE2_BRG, rtol=1e-12)
    # Bridgmanite takes up Fe3+ more readily than Fe2+ and the pyroxenes the
    # reverse, so Fe3+/FeT falls in the lower mantle and rises above it.
    assert np.all(state.D_fe3_cell[1:] > state.D_fe2_cell[1:] + 0.2)
    assert state.D_fe2_cell[0] > state.D_fe3_cell[0] + 0.1
    # Guard: the bulk peridotite D(FeO) = 0.85 would invert the deep order.
    assert np.all(state.D_fe2_cell < 0.6)
    # Inverted-comparison guard: the two branches differ by ~0.4, far beyond
    # any tolerance, so a flipped >= would fail the assertions above.
    assert abs(D_FE3_BRG - D_FE3_SHALLOW) > 0.3
    # Partition coefficients are bounded fractions, not free parameters.
    assert 0.0 < D_FE3_SHALLOW < 1.0
    assert _PRES[1] > P_CUTOFF_GPA * 1e9


@pytest.mark.parametrize('f_0', [0.02, 0.1, 0.45])
def test_configured_ferric_fraction_reaches_the_helpfile_on_the_first_step(f_0):
    """End-to-end: planet.ferric_fraction_initial is what the tracker seeds and
    reports, so a parameter sweep over it actually varies the model.

    Sweeps a reducing, the default, and a strongly oxidised melt. Before this
    field existed every case would have reported 0.1.
    """
    interior = _make_interior()
    hf_row = {'T_magma': 2000.0}

    update_melt_redox(interior, hf_row, _make_config(f_0))

    assert hf_row['ferric_frac_mantle'] == pytest.approx(f_0, rel=1e-12)
    # The state really was seeded from it, not just echoed into hf_row.
    assert interior.redox_state.ferric_frac == pytest.approx(f_0, rel=1e-12)
    # Delta-IW is finite and physically sized: magma-ocean surface values sit
    # within a few log units of the buffer, not at nan or hundreds.
    assert np.isfinite(hf_row['fO2_shift_IW_mantle'])
    assert -20.0 < hf_row['fO2_shift_IW_mantle'] < 20.0


def test_a_more_oxidised_initial_melt_gives_a_higher_surface_delta_iw():
    """Monotonicity across the new parameter: raising f_0 at fixed T_magma
    raises the surface Delta-IW, because Delta-IW goes as log10(f/(1-f))/a
    with a > 0 in Hirschmann Eq 21.

    This is the property a parameter study over f_0 relies on; a wiring bug
    that ignored the config would make the two runs identical.
    """
    hf_reduced: dict = {'T_magma': 2000.0}
    hf_oxidised: dict = {'T_magma': 2000.0}

    update_melt_redox(_make_interior(), hf_reduced, _make_config(0.05))
    update_melt_redox(_make_interior(), hf_oxidised, _make_config(0.40))

    assert hf_oxidised['fO2_shift_IW_mantle'] > hf_reduced['fO2_shift_IW_mantle']
    # Discrimination: the separation is log10(odds ratio)/a, which for these
    # two fractions is over 4 log units. A config that failed to propagate
    # would give exactly 0.0 separation.
    separation = hf_oxidised['fO2_shift_IW_mantle'] - hf_reduced['fO2_shift_IW_mantle']
    assert separation > 1.0
    # Same temperature, so the IW buffer term cancels and the difference is
    # attributable to the redox ratio alone.
    assert hf_reduced['T_magma'] == pytest.approx(hf_oxidised['T_magma'], rel=1e-12)


@pytest.mark.parametrize('source', ['user_constant', 'from_O_budget'])
def test_tracker_is_a_no_op_and_writes_nothing_under_other_fo2_sources(source):
    """Error/guard contract: under any fO2_source other than from_mantle_redox
    the tracker returns without seeding state or writing helpfile columns.

    Verifies the guard has no side effect, not merely that it returns: a
    partially seeded state would make a later switch of source inconsistent.
    """
    interior = _make_interior()
    hf_row = {'T_magma': 2000.0}

    update_melt_redox(interior, hf_row, _make_config(0.25, source=source))

    assert interior.redox_state is None
    assert 'ferric_frac_mantle' not in hf_row
    assert 'fO2_shift_IW_mantle' not in hf_row


# ── Fe metal saturation (Steps 9a-9h) ──────────────────────────────────────


def _crystallise(f_0, n_steps=6, metal_saturation=True):
    """Run the tracker through a solidifying mantle and return (state, rows).

    Melt fraction falls monotonically so Fe3+ is progressively partitioned
    into the solid, which is what drives the melt towards or away from the
    metal saturation boundary.
    """
    config = _make_config(f_0, metal_saturation=metal_saturation)
    interior = _make_interior()
    rows = []
    for step in range(n_steps):
        interior.phi = np.clip(_PHI - np.array([0.12, 0.07, 0.0]) * step, 0.0, None)
        hf_row = {'T_magma': 2200.0}
        update_melt_redox(interior, hf_row, config)
        rows.append(hf_row)
    return interior.redox_state, rows


@pytest.mark.physics_invariant
def test_reduced_melt_saturates_in_metal_and_becomes_more_oxidised():
    """A melt below the saturation threshold exsolves iron metal, and doing so
    raises its ferric fraction because the metal removes iron while leaving the
    oxygen behind."""
    f_0 = 0.005
    reduced, rows = _crystallise(f_0)

    assert rows[-1]['n_fe_metal_mantle'] > 0.0
    # The reaction brought the melt down onto the saturation boundary: it
    # started an order of magnitude supersaturated and must not be left so.
    assert rows[-1]['a_fe_max_mantle'] <= 1.0
    # Discriminating against partitioning alone, which lowers the ferric
    # fraction on this mesh; only the metal reaction can double it.
    assert reduced.ferric_frac > 2.0 * f_0


@pytest.mark.physics_invariant
def test_disabled_metal_saturation_skips_the_metal_check(monkeypatch):
    """planet.metal_saturation = False: the same reduced melt that exsolves
    metal above forms none, a_Fe is never evaluated (the activity function
    is not called, the columns hold 0 and -1), and the melt ends below f_0.
    Without metal only partitioning acts: each step the 30 GPa bridgmanite
    cell crystallises 2.8e20 kg with D(Fe3+) - D(Fe2+) = 0.31, which outweighs
    the 2.4e20 kg of the 10 GPa pyroxene cell with D(Fe2+) - D(Fe3+) = 0.15."""
    from proteus.interior_chem import redox

    f_0 = 0.005
    on, rows_on = _crystallise(f_0)

    def _not_called(*args, **kwargs):
        raise AssertionError('activity_Fe_metal called with metal_saturation off')

    monkeypatch.setattr(redox.dispro, 'activity_Fe_metal', _not_called)
    off, rows = _crystallise(f_0, metal_saturation=False)

    for row in rows:
        assert row['n_fe_metal_step_mantle'] == 0.0
        assert row['n_fe_metal_mantle'] == 0.0
        assert row['fe_metal_kg_mantle'] == 0.0
        assert row['a_fe_max_mantle'] == 0.0
        assert row['a_fe_max_cell_mantle'] == -1.0
    np.testing.assert_array_equal(off.n_fe_metal_cell, 0.0)
    np.testing.assert_array_equal(off.a_fe_cell, 0.0)
    assert off.a_fe_max_cell == -1
    # Discrimination guard: with saturation on, the same melt does report a_Fe.
    assert rows_on[-1]['a_fe_max_mantle'] > 0.0
    # Partitioning alone reduces this melt; the metal reaction oxidises it.
    assert off.ferric_frac < f_0 < on.ferric_frac


@pytest.mark.physics_invariant
def test_metal_diagnostics_reach_the_helpfile_csv(tmp_path, caplog):
    """The Fe-metal diagnostics the tracker writes into hf_row are
    helpfile schema columns, so they pass the row filter, are written to
    runtime_helpfile.csv and read back unchanged.

    Uses the reduced melt (f_0 = 0.005) so metal actually forms and the
    columns carry non-zero values; an all-zero column would also pass a
    presence check if the values were being dropped and re-seeded with 0.
    """
    from proteus.utils.coupler import (
        CreateHelpfileFromDict,
        ExtendHelpfile,
        ReadHelpfileFromCSV,
        WriteHelpfileToCSV,
        ZeroHelpfileRow,
    )

    metal_keys = (
        'a_fe_max_mantle',
        'a_fe_max_cell_mantle',
        'fe_metal_kg_mantle',
        'n_fe_metal_mantle',
        'n_fe_metal_step_mantle',
    )
    config = _make_config(0.005)
    interior = _make_interior()
    rows = []
    hf_all = None
    with caplog.at_level('WARNING'):
        for step in range(6):
            interior.phi = np.clip(_PHI - np.array([0.12, 0.07, 0.0]) * step, 0.0, None)
            hf_row = ZeroHelpfileRow()
            hf_row['Time'] = float(step)
            hf_row['T_magma'] = 2200.0
            update_melt_redox(interior, hf_row, config)
            rows.append(hf_row)
            hf_all = (
                CreateHelpfileFromDict(hf_row)
                if hf_all is None
                else ExtendHelpfile(hf_all, hf_row)
            )
    # The row filter no longer reports the metal keys as unknown.
    assert not any('not declared in GetHelpfileKeys' in r.message for r in caplog.records)

    WriteHelpfileToCSV(str(tmp_path), hf_all)
    csv = ReadHelpfileFromCSV(str(tmp_path))
    for key in metal_keys:
        assert key in csv.columns
        # Round trip: every step's value survives, not just the column.
        # rel=1e-9 because the CSV stores 11 significant digits (%.10e).
        np.testing.assert_allclose(
            csv[key].to_numpy(), [r[key] for r in rows], rtol=1e-9, atol=0.0
        )

    # Edge case: the first call runs no metal check, so all three are zero.
    assert csv['n_fe_metal_mantle'].iloc[0] == pytest.approx(0.0, abs=1e-30)
    assert csv['a_fe_max_mantle'].iloc[0] == pytest.approx(0.0, abs=1e-30)
    # No cell is tested on the first call, so the index is the -1 sentinel,
    # not 0 (which would name a real cell).
    assert csv['a_fe_max_cell_mantle'].iloc[0] == -1.0
    # Metal formed and was recorded, so a zero-filled column cannot pass.
    assert csv['n_fe_metal_mantle'].iloc[-1] > 0.0
    # Conservation: with no redissolution, the cumulative metal equals the
    # running sum of the per-step metal. A column holding a different
    # quantity (for example the metal activity) would break this closure.
    np.testing.assert_allclose(
        csv['n_fe_metal_mantle'].to_numpy(),
        np.cumsum(csv['n_fe_metal_step_mantle'].to_numpy()),
        rtol=1e-9,
        atol=0.0,
    )
    assert np.all(np.diff(csv['n_fe_metal_mantle'].to_numpy()) >= 0.0)

    # The mass column is the cumulative moles times the molar mass of Fe
    # metal, 0.055845 kg/mol.
    kg = csv['fe_metal_kg_mantle'].to_numpy()
    mol = csv['n_fe_metal_mantle'].to_numpy()
    np.testing.assert_allclose(kg, mol * 0.055845, rtol=1e-9, atol=0.0)
    # Discrimination guard: using the FeO molar mass (0.07184 kg/mol) instead
    # would overstate the mass by 29 %, far outside rtol=1e-9.
    assert abs(kg[-1] / (mol[-1] * 0.07184) - 1.0) > 0.2

    # Error contract: the schema is enforced, so a row that lacks one of the
    # metal columns is rejected rather than written with a gap.
    broken = dict(rows[-1])
    del broken['n_fe_metal_step_mantle']
    with pytest.raises(Exception, match='missing expected keys'):
        ExtendHelpfile(hf_all, broken)


@pytest.mark.physics_invariant
def test_surface_delta_iw_is_eq13_at_1_bar_and_the_outgassing_temperature(caplog):
    """The offset handed to the outgassing is Schaefer et al.'s surface fO2
    (fO2lowP_H22.m): Eq 13 at 1 bar, int(dV dP) = 0, with the melt ratio,
    minus IW, both at T_out = max(T_magma, outgas.T_floor) -- the
    temperature CALLIOPE and atmodeller solve at. 302 K is a surface
    temperature an interior step can return; at that temperature the
    relation gives an offset about 8 dex below the 700 K one.
    IW is Hirschmann (2021) at 1 bar, Schaefer et al.'s convention.
    """
    from proteus.interior_chem.redox import _iw_buffer_hirschmann2021, _log10_fO2_surface

    def diw_at(T, state):
        return _log10_fO2_surface(state.redox_ratio, T, state.X) - float(
            _iw_buffer_hirschmann2021(T, 1.0e-4)
        )

    config = _make_config(0.1344)
    results = {}
    for T in (302.342, 700.0, 2200.0, 3500.0):
        interior = _make_interior()
        hf_row = {'T_magma': T}
        caplog.clear()
        with caplog.at_level('WARNING'):
            update_melt_redox(interior, hf_row, config)
        warned = any('below outgas.T_floor' in r.message for r in caplog.records)
        results[T] = (hf_row['fO2_shift_IW_mantle'], interior.redox_state, warned)

    for T, (diw, state, warned) in results.items():
        T_out = max(T, 700.0)
        assert diw == pytest.approx(diw_at(T_out, state), rel=1e-12), T
        assert warned == (T < 700.0)
    # Discrimination guard: at the raw 302 K the offset is far lower, so a
    # value evaluated at T_magma itself could not pass the check above.
    cold, cold_state, _ = results[302.342]
    assert diw_at(302.342, cold_state) < cold - 5.0
    # Buffer guard: at 3500 K the O'Neill & Eggins buffer sits 1.1 dex below
    # H21, so an offset taken against it would fail the check above.
    hot, hot_state, _ = results[3500.0]
    bower = _log10_fO2_surface(hot_state.redox_ratio, 3500.0, hot_state.X)
    assert bower - _iw_oneill_eggins_2002(3500.0) > hot + 1.0
    # The temperature genuinely enters: at a fixed ratio the offset differs
    # between 700, 2200 and 3500 K (it is not monotonic in T, so only
    # distinctness is asserted).
    vals = [results[T][0] for T in (700.0, 2200.0, 3500.0)]
    assert min(abs(x - y) for i, x in enumerate(vals) for y in vals[i + 1 :]) > 0.5


@pytest.mark.physics_invariant
def test_freezing_the_shallow_melt_does_not_shift_the_surface_delta_iw():
    """Regression for the uppermost-melt-cell convention, which jumped by
    tens of log units once the shallow cells froze: the deepest remaining
    melt cell then carried a large int(dV dP) against a 1-bar buffer. The
    melt is homogeneous, so the surface offset may depend only on its
    Fe3+/Fe2+ and T_out, not on where melt remains. On the first call the
    ratio is exactly f_0 whatever phi is, so the two meshes below carry the
    same melt redox state.
    """
    config = _make_config(0.1)
    pres = np.array([0.3e9, 30.0e9, 80.0e9])
    temp = np.array([2000.0, 3200.0, 3900.0])
    mass = np.array([1.0e21, 3.0e21, 5.0e21])
    out = {}
    for name, phi in (
        ('molten', np.array([1.0, 1.0, 1.0])),
        ('shallow frozen', np.array([0.0, 0.0, 1.0])),
    ):
        interior = _make_interior()
        interior.phi, interior.pres, interior.temp, interior.mass = phi, pres, temp, mass
        hf_row = {'T_magma': 2500.0}
        update_melt_redox(interior, hf_row, config)
        out[name] = (hf_row['fO2_shift_IW_mantle'], interior.redox_state)

    (d_m, s_m), (d_f, s_f) = out['molten'], out['shallow frozen']
    assert s_m.redox_ratio == pytest.approx(s_f.redox_ratio, rel=1e-15)
    assert d_f == pytest.approx(d_m, rel=1e-12)
    # Discrimination guard: the diagnostic profile does see the deep cell,
    # whose fO2 sits many log units above the 1-bar value at the same ratio,
    # so evaluating Delta-IW there would have failed the check above.
    assert s_f.fO2_cell == 2
    assert s_f.log10_fO2_cell[2] - s_m.log10_fO2_cell[0] > 5.0


@pytest.mark.physics_invariant
def test_eq13_pressure_term_matches_the_uncompressed_volume_analytic_limit():
    """At the Deng reference temperature T0 both endmembers sit at V0 at
    1 bar, so for small P dV(FeO1.5 - FeO) is the uncompressed value
    dV0 = (V0_ox - V0_red)/2 and Eq 13 shifts log10 fO2 by
    dV0 (P - P0) / (a R T ln10) relative to 1 bar. The expectation is
    built by hand from that closed form.
    """
    import math

    from proteus.interior_chem import eos_deng
    from proteus.interior_chem.redox import _A, _R, _log10_fO2_profile, _log10_fO2_surface

    T = eos_deng._T0
    P = np.array([1.0e-4, 0.02, 0.05])  # compression of order P/K0 ~ 0.2 % at most
    dV0 = 0.5 * (eos_deng._V0[1] - eos_deng._V0[0])
    ratio = 0.1 / 0.9
    state = _init_state(_PHI, _MASS, _PRES, 0.1)
    prof, valid = _log10_fO2_profile(ratio, np.full(3, T), P, state.X)
    assert np.all(valid)
    shift = dV0 * (P - 1.0e-4) / (_A * _R * T * math.log(10))
    expected = _log10_fO2_surface(ratio, T, state.X) + shift
    # 1 % of the shift covers the neglected compression of dV.
    np.testing.assert_allclose(prof, expected, rtol=0, atol=0.01 * shift[-1] + 1e-12)
    # P -> 1 bar recovers the surface relation (pressure term vanishes).
    assert prof[0] == pytest.approx(_log10_fO2_surface(ratio, T, state.X), abs=1e-9)
    # Order of magnitude guard: ~0.03 log units at 0.05 GPa here.
    assert 0.01 < prof[2] - prof[0] < 0.1


@pytest.mark.physics_invariant
def test_fo2_profile_is_resolved_per_melt_cell_and_nan_in_solid():
    """The Step 10 profile covers every melt cell at its own (T, P); solid
    cells carry NaN. At a uniform ratio and fixed T, fO2 rises with P
    because dV(FeO1.5 - FeO) > 0 over the mantle range."""
    from proteus.interior_chem.redox import _log10_fO2_profile

    config = _make_config(0.1)
    interior = _make_interior()
    update_melt_redox(interior, {'T_magma': 2200.0}, config)
    prof = interior.redox_state.log10_fO2_cell
    assert prof.shape == _PHI.shape
    assert np.all(np.isfinite(prof[_PHI > 0]))
    assert np.all(np.isnan(prof[_PHI == 0]))
    # Same (T, P) per cell as a direct call.
    direct, _ = _log10_fO2_profile(
        interior.redox_state.redox_ratio, _TEMP, _PRES / 1e9, interior.redox_state.X
    )
    np.testing.assert_allclose(prof[_PHI > 0], direct[_PHI > 0], rtol=1e-12)

    flat, _ = _log10_fO2_profile(
        0.1 / 0.9, np.full(4, 3000.0), np.array([0.1, 5.0, 25.0, 100.0]), interior.redox_state.X
    )
    assert np.all(np.diff(flat) > 0)


@pytest.mark.physics_invariant
def test_uppermost_melt_cell_index_is_found_whatever_the_grid_order():
    """fO2_top_index (diagnostic) is the lowest-pressure melt cell whether
    the backend orders cells surface-first (SPIDER) or CMB-first (Aragog),
    and a solid lid is skipped; the surface offset does not depend on the
    order either."""
    config = _make_config(0.1)
    phi = np.array([0.0, 1.0, 0.5])  # solid lid on top
    pres = np.array([0.5e9, 3.0e9, 30.0e9])
    temp = np.array([1600.0, 2500.0, 3300.0])
    mass = np.array([1.0e21, 3.0e21, 5.0e21])

    offsets = []
    for order in (slice(None), slice(None, None, -1)):
        interior = _make_interior()
        interior.phi, interior.pres = phi[order], pres[order]
        interior.temp, interior.mass = temp[order], mass[order]
        hf_row = {'T_magma': 2000.0}
        update_melt_redox(interior, hf_row, config)
        assert interior.pres[interior.redox_state.fO2_cell] == pytest.approx(3.0e9)
        offsets.append(hf_row['fO2_shift_IW_mantle'])
    assert offsets[0] == pytest.approx(offsets[1], rel=1e-12)


def test_oxidised_melt_stays_below_metal_saturation():
    """A melt at the published bulk-silicate ferric fraction sits far from the
    saturation boundary and forms no metal even with the check enabled.

    This is the boundary case on the other side of the threshold: the physics
    is on, and the answer is still zero.
    """
    state, rows = _crystallise(0.10)

    assert rows[-1]['n_fe_metal_mantle'] == pytest.approx(0.0, abs=1e-30)
    # The diagnostic activity is reported and is well under unity, so the
    # zero comes from an evaluated criterion, not a skipped branch.
    assert 0.0 < rows[-1]['a_fe_max_mantle'] < 1.0


@pytest.mark.physics_invariant
def test_metal_step_conserves_iron_and_oxygen():
    """Exsolving metal moves iron out of the melt without creating or
    destroying iron or oxygen.

    Conservation closure is the primary assertion, so it also serves as the
    exponent guard: any stoichiometric coefficient error breaks it.
    """
    state, _ = _crystallise(0.005, n_steps=3)
    fe_before = state.n_fe2_melt + state.n_fe3_melt + float(np.sum(state.n_fe_metal_cell))
    o_before = state.n_fe2_melt + 1.5 * state.n_fe3_melt

    xi = _metal_saturation_step(state, _TEMP, _PRES, _PHI, _MASS)

    fe_after = state.n_fe2_melt + state.n_fe3_melt + float(np.sum(state.n_fe_metal_cell))
    o_after = state.n_fe2_melt + 1.5 * state.n_fe3_melt

    assert fe_after == pytest.approx(fe_before, rel=1e-12)
    assert o_after == pytest.approx(o_before, rel=1e-12)
    assert np.isfinite(xi)


def test_metal_is_deposited_in_a_single_cell_not_smeared_across_the_mantle():
    """Metal accumulates where it formed rather than being distributed over
    every cell, because only the most supersaturated cell hosts equilibrium
    while the melt itself is homogeneous.

    The resulting radial field is what a later transport model consumes, so
    the locality matters as much as the total.
    """
    state, _ = _crystallise(0.005, n_steps=3)
    occupied = int(np.count_nonzero(state.n_fe_metal_cell))

    assert occupied == 1
    # The occupied cell is one that actually holds melt: a solid cell cannot
    # exsolve metal from a liquid.
    assert state.n_fe_metal_cell[np.argmax(state.n_fe_metal_cell)] > 0.0
    assert float(np.sum(state.n_fe_metal_cell)) > 0.0


@pytest.mark.reference_pinned
def test_the_pressure_term_moves_metal_formation_away_from_the_coolest_cell():
    """The reaction volume change relocates the most supersaturated cell off
    the shallow cool end of the mantle and down into it.

    Anchor: Schaefer et al. (2024) JGR Planets 129, e2023JE008262, Section
    3.1.1, where metal forms at the base of the whole-mantle magma ocean and
    not at all in the 500 km models. On standard-state energies alone the
    activity depends on temperature only and the trend runs the other way,
    which is why the pressure term is not optional. The comparison is driven
    through the chemistry directly, since the model always includes it.
    """
    n2, n3 = 0.99 * _N_FET_TEST, 0.01 * _N_FET_TEST
    without_P, _ = activity_Fe_metal(
        _TEMP, n2, n3, _N_SIL_TEST, P_gpa=_PRES / 1e9, use_P_term=False
    )
    with_P, _ = activity_Fe_metal(
        _TEMP, n2, n3, _N_SIL_TEST, P_gpa=_PRES / 1e9, use_P_term=True
    )

    assert int(np.argmax(without_P)) == 0
    assert int(np.argmax(with_P)) > 0
    # Both are live comparisons rather than an artefact of one being all-zero.
    assert float(np.max(without_P)) > 0.0
    assert float(np.max(with_P)) > 0.0


def test_metal_saturation_does_not_run_before_any_crystallisation():
    """The first coupling step seeds the reservoirs and forms no metal, since
    no solid has yet grown to drive the melt anywhere.

    This is the limit-input contract at the start of a run: Schaefer likewise
    begins checking only after the first solid layer appears.
    """
    config = _make_config(0.005)
    interior = _make_interior()
    hf_row = {'T_magma': 2200.0}
    update_melt_redox(interior, hf_row, config)

    assert hf_row['n_fe_metal_mantle'] == pytest.approx(0.0, abs=1e-30)
    # The reservoirs were still seeded, so the step was skipped rather than
    # the whole tracker having been short-circuited.
    assert interior.redox_state.n_fe3_melt > 0.0


@pytest.mark.physics_invariant
def test_a_fe_max_cell_is_the_cell_that_receives_the_metal():
    """The binding-cell index names the cell holding the largest a_Fe, and on
    every step that forms metal it is exactly the cell whose inventory grew.
    A column filled with argmax over the wrong array, or left stale from an
    earlier step, would fail one of the two checks."""
    config = _make_config(0.005)
    interior = _make_interior()
    formed_any = False
    for step in range(6):
        interior.phi = np.clip(_PHI - np.array([0.12, 0.07, 0.0]) * step, 0.0, None)
        before = (
            None
            if interior.redox_state is None
            else interior.redox_state.n_fe_metal_cell.copy()
        )
        hf_row = {'T_magma': 2200.0}
        update_melt_redox(interior, hf_row, config)
        idx = hf_row['a_fe_max_cell_mantle']
        state = interior.redox_state
        if before is None:
            assert idx == -1.0
            continue
        assert idx == float(np.argmax(state.a_fe_cell))
        assert state.a_fe_cell[int(idx)] == hf_row['a_fe_max_mantle']
        grown = np.flatnonzero(state.n_fe_metal_cell > before)
        if hf_row['n_fe_metal_step_mantle'] > 0.0:
            formed_any = True
            np.testing.assert_array_equal(grown, [int(idx)])
        else:
            assert grown.size == 0
    # Guard against a fixture that never saturates, which would make the
    # metal-location check above vacuous.
    assert formed_any


def _make_int_snapshot(path, n_stag):
    """Minimal Aragog-style _int.nc: a staggered dimension and one field."""
    import netCDF4 as nc

    with nc.Dataset(path, mode='w') as ds:
        ds.createDimension('staggered', n_stag)
        v = ds.createVariable('temp_s', np.float64, ('staggered',))
        v[:] = np.arange(n_stag, dtype=float)


def test_fo2_profile_round_trips_through_the_interior_snapshot(tmp_path):
    """The Step 10 profile appended to an _int.nc reads back cell for cell,
    NaN (solid) included, with the top-cell index; the backend's own fields
    are untouched, and a second append overwrites rather than failing."""
    import netCDF4 as nc

    from proteus.interior_chem.redox import write_fO2_profile_ncdf

    fpath = str(tmp_path / '1000_int.nc')
    _make_int_snapshot(fpath, _PHI.size)
    interior = _make_interior()
    update_melt_redox(interior, {'T_magma': 2200.0}, _make_config(0.1))
    state = interior.redox_state

    assert write_fO2_profile_ncdf(fpath, state)
    with nc.Dataset(fpath) as ds:
        got = np.asarray(ds['log10_fO2_s'][:], dtype=float)
        assert ds['log10_fO2_s'].units == 'log10(bar)'
        assert int(ds['fO2_top_index'][...]) == state.fO2_cell
        np.testing.assert_array_equal(ds['temp_s'][:], np.arange(_PHI.size))
    np.testing.assert_array_equal(np.isnan(got), _PHI == 0)
    np.testing.assert_allclose(got[_PHI > 0], state.log10_fO2_cell[_PHI > 0], rtol=0, atol=0)

    # The Schaefer-convention profile: log10_fO2_s minus IW_H21 at each cell's
    # (T, P), with the same NaN cells.
    from proteus.interior_chem.redox import _iw_buffer_hirschmann2021

    with nc.Dataset(fpath) as ds:
        diw = np.asarray(ds['dIW_H21_s'][:], dtype=float)
        assert ds['dIW_H21_s'].units == 'log10 units rel. IW'
    np.testing.assert_array_equal(np.isnan(diw), np.isnan(got))
    expected = got - _iw_buffer_hirschmann2021(_TEMP, _PRES / 1e9)
    np.testing.assert_allclose(diw[_PHI > 0], expected[_PHI > 0], rtol=0, atol=1e-12)

    # Second step on the same file: values replaced, no duplicate-variable error.
    state.log10_fO2_cell = state.log10_fO2_cell + 1.0
    assert write_fO2_profile_ncdf(fpath, state)
    with nc.Dataset(fpath) as ds:
        again = np.asarray(ds['log10_fO2_s'][:], dtype=float)
    np.testing.assert_allclose(again[_PHI > 0], got[_PHI > 0] + 1.0, rtol=0, atol=1e-12)
    assert ds_attr(fpath, 'log10_fO2_s', 'pressure_term') == 1


def ds_attr(fpath, var, attr):
    """Read one attribute of one variable from a NetCDF file."""
    import netCDF4 as nc

    with nc.Dataset(fpath) as ds:
        return ds[var].getncattr(attr)


@pytest.mark.physics_invariant
def test_without_metal_saturation_the_fo2_profiles_are_pressure_free(tmp_path):
    """metal_saturation = False: log10_fO2_cell is Eq 13 at each cell's T with
    int(dV dP) = 0, and dIW_H21 is taken against IW_H21 at 1 bar, so both sides
    drop their pressure dependence. A cell above the Deng envelope (7000 K) is
    no longer NaN, since no EOS is evaluated."""
    from proteus.interior_chem.redox import (
        P_1BAR_GPA,
        _iw_buffer_hirschmann2021,
        _log10_fO2_surface,
        write_fO2_profile_ncdf,
    )

    temp = np.array([2200.0, 3500.0, 7000.0])
    profiles = {}
    for react in (False, True):
        interior = _make_interior()
        interior.phi = np.array([1.0, 0.5, 0.4])
        interior.temp = temp
        update_melt_redox(
            interior, {'T_magma': 2200.0}, _make_config(0.1, metal_saturation=react)
        )
        profiles[react] = interior.redox_state
    off, on = profiles[False], profiles[True]

    expected = np.array([_log10_fO2_surface(off.redox_ratio, T, off.X) for T in temp])
    np.testing.assert_allclose(off.log10_fO2_cell, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        off.dIW_H21_cell,
        expected - _iw_buffer_hirschmann2021(temp, P_1BAR_GPA),
        rtol=0,
        atol=1e-12,
    )
    assert not off.profile_pressure_term and on.profile_pressure_term
    # Discrimination guard: with the pressure term the 30 GPa cell sits several
    # log units higher, and the 7000 K cell is outside the Deng envelope.
    assert on.log10_fO2_cell[1] - off.log10_fO2_cell[1] > 2.0
    assert np.isnan(on.log10_fO2_cell[2]) and np.isfinite(off.log10_fO2_cell[2])

    fpath = str(tmp_path / '1000_int.nc')
    _make_int_snapshot(fpath, temp.size)
    assert write_fO2_profile_ncdf(fpath, off)
    assert ds_attr(fpath, 'log10_fO2_s', 'pressure_term') == 0
    assert ds_attr(fpath, 'dIW_H21_s', 'pressure_term') == 0
    assert '1 bar' in ds_attr(fpath, 'dIW_H21_s', 'long_name')


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_hirschmann_iw_buffer_matches_oneill_eggins_at_one_bar_and_the_solid_volume_slope():
    """The Hirschmann (2021) IW buffer agrees with the independent O'Neill &
    Eggins (2002) calibration at 1 bar near 1500 K, and its pressure slope at
    2000 K matches 2 dV/(R T ln10) for Fe + 1/2 O2 = FeO with the ambient
    molar volumes of wustite (12.06 cm3/mol) and iron (7.09 cm3/mol).

    Both are calibrations of the same equilibrium, so they differ by tenths
    of a log unit; a wrong coefficient or a missing T ln T term moves the
    1-bar value by whole log units. The slope estimate neglects compression
    and thermal expansion, hence the 25 % tolerance.
    """
    import math

    from proteus.interior_chem.redox import _iw_buffer_hirschmann2021

    T = 1500.0
    h21 = float(_iw_buffer_hirschmann2021(T, 1.0e-4))
    assert h21 == pytest.approx(_iw_oneill_eggins_2002(T), abs=0.3)
    # Scale guard: IW at 1500 K sits near log10 fO2 = -10 to -11.
    assert -12.0 < h21 < -9.0

    T = 2000.0
    slope = float(_iw_buffer_hirschmann2021(T, 11.0) - _iw_buffer_hirschmann2021(T, 9.0)) / 2.0
    expected = 2.0 * (12.06 - 7.09) * 1e3 / (8.31447 * T * math.log(10))  # cm3 GPa = 1e3 J
    assert slope == pytest.approx(expected, rel=0.25)
    assert slope > 0.0  # IW rises with pressure


@pytest.mark.physics_invariant
def test_hirschmann_iw_buffer_is_continuous_at_the_iron_phase_switch_and_broadcasts():
    """The fcc/bcc and hcp coefficient sets meet at the transition pressure
    within a few hundredths of a log unit, and array input of mixed branches
    gives the same values as scalar calls."""
    from proteus.interior_chem.redox import _iw_buffer_hirschmann2021

    for T in (2000.0, 3000.0):
        p_tr = -18.640 + 0.04359 * T - 5.069e-6 * T**2
        jump = float(
            _iw_buffer_hirschmann2021(T, p_tr + 1e-6)
            - _iw_buffer_hirschmann2021(T, p_tr - 1e-6)
        )
        assert abs(jump) < 0.05

    T = np.array([1500.0, 2500.0, 3500.0])
    P = np.array([1.0e-4, 30.0, 90.0])  # fcc, fcc, hcp at these temperatures
    vec = _iw_buffer_hirschmann2021(T, P)
    assert vec.shape == (3,)
    np.testing.assert_allclose(
        vec, [float(_iw_buffer_hirschmann2021(t, p)) for t, p in zip(T, P)], rtol=0, atol=1e-12
    )


def test_fo2_profile_is_not_written_to_a_missing_or_mismatched_snapshot(tmp_path):
    import netCDF4 as nc

    from proteus.interior_chem.redox import write_fO2_profile_ncdf

    interior = _make_interior()
    update_melt_redox(interior, {'T_magma': 2200.0}, _make_config(0.1))
    state = interior.redox_state

    assert not write_fO2_profile_ncdf(str(tmp_path / 'absent_int.nc'), state)
    assert not write_fO2_profile_ncdf(str(tmp_path / 'x_int.nc'), None)

    fpath = str(tmp_path / 'wrong_int.nc')
    _make_int_snapshot(fpath, _PHI.size + 2)
    assert not write_fO2_profile_ncdf(fpath, state)
    with nc.Dataset(fpath) as ds:
        assert 'log10_fO2_s' not in ds.variables


def test_standalone_redox_snapshot_is_self_contained(tmp_path):
    """SPIDER has no _int.nc, so its <time>_redox.nc must carry the profile
    together with the P, T, phi it was computed on and the step's scalars,
    so the file alone can reproduce the reported offset's input cell."""
    import netCDF4 as nc

    from proteus.interior_chem.redox import write_redox_ncdf

    interior = _make_interior()
    hf_row = {'T_magma': 2200.0}
    update_melt_redox(interior, hf_row, _make_config(0.1))
    state = interior.redox_state

    fpath = str(tmp_path / '884p700_redox.nc')
    assert write_redox_ncdf(fpath, state, 884.7, interior, hf_row)
    with nc.Dataset(fpath) as ds:
        assert len(ds.dimensions['staggered']) == _PHI.size
        np.testing.assert_allclose(ds['pres_s'][:], _PRES / 1e9, rtol=1e-15)
        assert ds['pres_s'].units == 'GPa'
        np.testing.assert_array_equal(ds['temp_s'][:], _TEMP)
        np.testing.assert_array_equal(ds['phi_s'][:], _PHI)
        got = np.asarray(ds['log10_fO2_s'][:], dtype=float)
        top = int(ds['fO2_top_index'][...])
        assert float(ds['time'][...]) == pytest.approx(884.7)
        assert float(ds['fO2_shift_IW_mantle'][...]) == hf_row['fO2_shift_IW_mantle']
        assert float(ds['ferric_frac_mantle'][...]) == hf_row['ferric_frac_mantle']
        P_top = float(ds['pres_s'][top])
    np.testing.assert_array_equal(np.isnan(got), _PHI == 0)
    np.testing.assert_allclose(got[_PHI > 0], state.log10_fO2_cell[_PHI > 0], rtol=0, atol=0)
    # The stored top index points at the lowest-pressure melt cell.
    assert P_top == pytest.approx(np.min(_PRES[_PHI > 0]) / 1e9)
    assert not write_redox_ncdf(str(tmp_path / 'none_redox.nc'), None, 0.0, interior, hf_row)


@pytest.mark.physics_invariant
def test_cells_below_the_solid_threshold_carry_no_melt_iron():
    """phi < PHI_SOLID counts as solid: such a cell is left out of the
    initial melt inventory exactly as a phi = 0 cell is."""
    from proteus.interior_chem.redox import PHI_SOLID

    phi_trace = np.array([1.0, 0.5, 0.5 * PHI_SOLID])
    phi_zero = np.array([1.0, 0.5, 0.0])
    out = []
    for phi in (phi_trace, phi_zero):
        interior = _make_interior()
        interior.phi = phi
        update_melt_redox(interior, {'T_magma': 2200.0}, _make_config(0.1))
        out.append(interior.redox_state)
    assert out[0].n_fe2_melt == pytest.approx(out[1].n_fe2_melt, rel=1e-15)
    assert out[0].n_fe2_melt == pytest.approx(W_FET * _M_MELT / MU_FEO, rel=1e-12)
    # The interior's own phi is not modified.
    assert interior.phi is phi_zero


@pytest.mark.physics_invariant
def test_a_trace_melt_cell_is_not_tested_for_metal_and_has_no_fo2():
    """A deep, hot cell with phi just below PHI_SOLID would otherwise be
    the binding cell (the pressure term favours depth); as solid it is
    skipped by the metal check and left NaN in the fO2 profile."""
    from proteus.interior_chem.redox import PHI_SOLID

    pres = np.array([10.0e9, 30.0e9, 60.0e9])
    temp = np.array([2200.0, 3000.0, 3800.0])
    mass = np.array([2.0e21, 4.0e21, 6.0e21])
    config = _make_config(0.01)

    def run(phi_deep):
        interior = _make_interior()
        interior.pres, interior.temp, interior.mass = pres, temp, mass
        interior.phi = np.array([1.0, 1.0, 1.0])
        update_melt_redox(interior, {'T_magma': 2200.0}, config)
        interior.phi = np.array([0.95, 0.95, phi_deep])
        update_melt_redox(interior, {'T_magma': 2200.0}, config)
        return interior.redox_state

    st_melt = run(2.0 * PHI_SOLID)
    st_trace = run(0.5 * PHI_SOLID)
    # Discrimination guard: as melt, the deep cell is the most saturated one.
    assert st_melt.a_fe_max_cell == 2
    assert st_trace.a_fe_cell[2] == 0.0
    assert st_trace.a_fe_max_cell != 2
    assert np.isnan(st_trace.log10_fO2_cell[2])
    assert np.isfinite(st_melt.log10_fO2_cell[2])


@pytest.mark.physics_invariant
def test_crossing_the_solid_threshold_crystallises_the_remaining_melt():
    """A cell dropping from phi = 0.5 to just below PHI_SOLID crystallises
    all of its remaining melt that step: the reservoirs end up the same as
    for a drop straight to phi = 0."""
    from proteus.interior_chem.redox import PHI_SOLID

    out = []
    for phi_last in (0.8 * PHI_SOLID, 0.0):
        interior = _make_interior()
        update_melt_redox(interior, {'T_magma': 2200.0}, _make_config(0.1))
        interior.phi = np.array([1.0, phi_last, 0.0])
        update_melt_redox(interior, {'T_magma': 2200.0}, _make_config(0.1))
        st = interior.redox_state
        out.append(
            (st.n_fe2_melt + float(np.sum(st.n_fe_metal_cell)), st.n_fe3_melt, st.phi_prev[1])
        )
    (fe2_a, fe3_a, prev_a), (fe2_b, fe3_b, prev_b) = out
    assert prev_a == 0.0 and prev_b == 0.0
    assert fe2_a == pytest.approx(fe2_b, rel=1e-12)
    assert fe3_a == pytest.approx(fe3_b, rel=1e-12)


@pytest.mark.physics_invariant
def test_without_metal_saturation_a_trace_melt_cell_keeps_its_iron():
    """metal_saturation = False drops the PHI_SOLID threshold: a cell at
    phi = PHI_SOLID / 2 joins the initial melt inventory with its real melt
    mass, whereas with saturation on it counts as solid."""
    from proteus.interior_chem.redox import PHI_SOLID

    phi_deep = 0.5 * PHI_SOLID
    phi = np.array([1.0, 0.5, phi_deep])
    n_fe2 = {}
    for react in (False, True):
        interior = _make_interior()
        interior.phi = phi
        update_melt_redox(
            interior, {'T_magma': 2200.0}, _make_config(0.1, metal_saturation=react)
        )
        n_fe2[react] = interior.redox_state.n_fe2_melt
    m_melt_real = _M_MELT + phi_deep * _MASS[2]
    assert n_fe2[False] == pytest.approx(W_FET * m_melt_real / MU_FEO, rel=1e-12)
    assert n_fe2[True] == pytest.approx(W_FET * _M_MELT / MU_FEO, rel=1e-12)


@pytest.mark.physics_invariant
def test_without_metal_saturation_crystallisation_follows_the_real_melt_fraction():
    """With metal_saturation = False a cell dropping from phi = 0.5 to
    0.8 * PHI_SOLID crystallises only that decrease (Eq 12-19), not its whole
    remaining melt, and phi_prev keeps the solver value."""
    from proteus.interior_chem.redox import PHI_SOLID

    f_0, phi_last = 0.1, 0.8 * PHI_SOLID
    config = _make_config(f_0, metal_saturation=False)
    interior = _make_interior()
    update_melt_redox(interior, {'T_magma': 2200.0}, config)
    interior.phi = np.array([1.0, phi_last, 0.0])
    update_melt_redox(interior, {'T_magma': 2200.0}, config)
    st = interior.redox_state

    n2_0 = W_FET * _M_MELT / MU_FEO
    n3_0 = n2_0 * f_0 / (1.0 - f_0)
    dM = (0.5 - phi_last) * _MASS[1]  # cell 1 sits at 30 GPa: bridgmanite D
    assert st.n_fe2_melt == pytest.approx(n2_0 - D_FE2_BRG * n2_0 / _M_MELT * dM, rel=1e-12)
    assert st.n_fe3_melt == pytest.approx(n3_0 - D_FE3_BRG * n3_0 / _M_MELT * dM, rel=1e-12)
    assert st.phi_prev[1] == phi_last
    assert np.all(st.n_fe_metal_cell == 0.0)
    # Discrimination guard: the thresholded drop to 0 removes more Fe2+.
    dM_thr = 0.5 * _MASS[1]
    assert st.n_fe2_melt > n2_0 - D_FE2_BRG * n2_0 / _M_MELT * dM_thr * (1.0 - 1e-9)


# ---------------------------------------------------------------------------
# Edge states: no iron, no melt, melt outside the EOS, a solidified mantle
# ---------------------------------------------------------------------------


def test_ratios_are_left_unchanged_when_the_melt_holds_no_iron():
    """f = n3/(n2+n3) is undefined with no iron, so the last ratios stand."""
    from proteus.interior_chem.redox import _update_ratios

    state = _init_state(_PHI, _MASS, _PRES, 0.1)
    state.n_fe2_melt = state.n_fe3_melt = 0.0
    _update_ratios(state)
    assert state.ferric_frac == pytest.approx(0.1, rel=1e-15)


def test_metal_step_does_nothing_without_melt():
    state = _init_state(_PHI, _MASS, _PRES, 0.01)
    xi = _metal_saturation_step(state, _TEMP, _PRES, np.zeros(3), _MASS)
    assert xi == 0.0 and state.a_fe_max_cell == -1


def test_melt_hotter_than_the_eos_ceiling_is_not_tested_and_warned_once(caplog):
    """All melt above T_CEILING: no cell can be tested, so no metal forms,
    a_Fe stays 0 everywhere, and the warning is logged once per run."""
    import logging

    from proteus.interior_chem.eos_deng import T_CEILING

    state = _init_state(_PHI, _MASS, _PRES, 0.001)
    hot = np.full(3, T_CEILING + 500.0)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_chem.redox'):
        for _ in range(2):
            n2 = state.n_fe2_melt
            assert _metal_saturation_step(state, hot, _PRES, _PHI, _MASS) == 0.0
            assert state.n_fe2_melt == n2
    assert np.all(state.a_fe_cell == 0.0) and state.a_fe_max_cell == -1
    warned = [r for r in caplog.records if 'envelope' in r.message]
    assert len(warned) == 1


@pytest.mark.physics_invariant
def test_a_solidified_mantle_freezes_the_reservoirs(caplog):
    """Once no cell holds melt the tracker stops: the reservoirs and the
    ferric fraction are frozen, the metal check is reported as skipped,
    and the surface offset is still evaluated from the frozen ratio."""
    import logging

    config = _make_config(0.1)
    interior = _make_interior()
    update_melt_redox(interior, {'T_magma': 2200.0}, config)
    interior.phi = np.array([0.8, 0.3, 0.0])
    update_melt_redox(interior, {'T_magma': 2200.0}, config)
    st = interior.redox_state
    # The step on which the last melt disappears still crystallises it.
    interior.phi = np.zeros(3)
    update_melt_redox(interior, {'T_magma': 2200.0}, config)
    assert not st.melt_exhausted
    frozen = (st.n_fe2_melt, st.n_fe3_melt, st.ferric_frac)

    with caplog.at_level(logging.INFO, logger='fwl.proteus.interior_chem.redox'):
        update_melt_redox(interior, {'T_magma': 2200.0}, config)  # flags it
        hf_row = {'T_magma': 2200.0}
        update_melt_redox(interior, hf_row, config)  # frozen step
    assert st.melt_exhausted
    assert (st.n_fe2_melt, st.n_fe3_melt, st.ferric_frac) == frozen
    assert hf_row['ferric_frac_mantle'] == frozen[2]
    assert hf_row['a_fe_max_cell_mantle'] == -1.0
    assert np.isfinite(hf_row['fO2_shift_IW_mantle'])
    msgs = ' '.join(r.message for r in caplog.records)
    assert 'mantle fully solidified' in msgs
    assert 'not checked (mantle solidified)' in msgs


# ---------------------------------------------------------------------------
# store_profile_snapshot: which file, for which interior module
# ---------------------------------------------------------------------------


def _snapshot_config(module, source='from_mantle_redox'):
    config = _make_config(0.1, source=source)
    config.interior_energetics.module = module
    return config


def test_snapshot_appends_to_the_aragog_int_file(tmp_path):
    import netCDF4 as nc

    from proteus.interior_chem.redox import store_profile_snapshot

    (tmp_path / 'data').mkdir()
    fpath = tmp_path / 'data' / '884p700_int.nc'
    _make_int_snapshot(str(fpath), _PHI.size)
    interior = _make_interior()
    hf_row = {'T_magma': 2200.0}
    update_melt_redox(interior, hf_row, _snapshot_config('aragog'))

    out = store_profile_snapshot(
        _snapshot_config('aragog'), {'output': str(tmp_path)}, 884.7, interior, hf_row
    )
    assert out == str(fpath)
    with nc.Dataset(fpath) as ds:
        assert 'log10_fO2_s' in ds.variables
    assert not (tmp_path / 'data' / '884p700_redox.nc').exists()


def test_snapshot_writes_a_standalone_file_for_spider(tmp_path):
    import netCDF4 as nc

    from proteus.interior_chem.redox import store_profile_snapshot

    (tmp_path / 'data').mkdir()
    interior = _make_interior()
    hf_row = {'T_magma': 2200.0}
    update_melt_redox(interior, hf_row, _snapshot_config('spider'))

    out = store_profile_snapshot(
        _snapshot_config('spider'), {'output': str(tmp_path)}, 884.7, interior, hf_row
    )
    assert out == str(tmp_path / 'data' / '884p700_redox.nc')
    with nc.Dataset(out) as ds:
        assert float(ds['time'][...]) == pytest.approx(884.7)


def test_snapshot_writes_nothing_for_other_sources_modules_or_missing_files(tmp_path):
    from proteus.interior_chem.redox import store_profile_snapshot

    (tmp_path / 'data').mkdir()
    interior = _make_interior()
    hf_row = {'T_magma': 2200.0}
    update_melt_redox(interior, hf_row, _snapshot_config('aragog'))
    dirs = {'output': str(tmp_path)}
    # Another fO2 source: the tracker never ran, nothing to store.
    assert (
        store_profile_snapshot(
            _snapshot_config('aragog', 'user_constant'), dirs, 1.0, interior, hf_row
        )
        is None
    )
    # A module without radial output.
    assert (
        store_profile_snapshot(_snapshot_config('dummy'), dirs, 1.0, interior, hf_row) is None
    )
    # Aragog step that wrote no snapshot (dt_write throttle).
    assert (
        store_profile_snapshot(_snapshot_config('aragog'), dirs, 1.0, interior, hf_row) is None
    )
    assert list((tmp_path / 'data').iterdir()) == []


def test_a_saturated_cell_with_no_reaction_extent_leaves_the_melt_unchanged(monkeypatch):
    """If the extent solver finds nothing to react (xi <= 0, e.g. a melt
    exactly at a_Fe = 1), no metal is deposited and the reservoirs stand."""
    from proteus.interior_chem import redox

    state = _init_state(_PHI, _MASS, _PRES, 0.001)  # strongly supersaturated
    monkeypatch.setattr(redox.dispro, 'solve_extent', lambda *a, **k: 0.0)
    n2, n3 = state.n_fe2_melt, state.n_fe3_melt
    xi = _metal_saturation_step(state, _TEMP, _PRES, _PHI, _MASS)
    assert xi == 0.0
    assert state.a_fe_cell[state.a_fe_max_cell] >= 1.0  # the check did run
    assert (state.n_fe2_melt, state.n_fe3_melt) == (n2, n3)
    assert np.all(state.n_fe_metal_cell == 0.0)


# ── Tracker state in the snapshot, restored on resume ──────────────────────


def _phi_at(step):
    """Melt fraction of the solidifying three-cell mantle at ``step``."""
    return np.clip(_PHI - np.array([0.12, 0.07, 0.0]) * step, 0.0, None)


def _resume_config(module='aragog'):
    """A reduced melt (f_0 = 0.005) with metal saturation on, so the state
    carries metal as well as crystallization history."""
    config = _make_config(0.005, metal_saturation=True)
    config.interior_energetics.module = module
    return config


def _run_steps(interior, config, steps):
    for step in steps:
        interior.phi = _phi_at(step)
        update_melt_redox(interior, {'T_magma': 2200.0}, config)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('module', ['aragog', 'spider'])
def test_a_resume_continues_the_tracker_exactly_as_an_unbroken_run(tmp_path, module):
    """A run stored at step 3 and resumed from its snapshot reaches the same
    reservoirs, metal and Fe3+/FeT after step 6 as a run that never stopped.
    Without the restore the tracker reseeds from f_0 and loses the history."""
    from proteus.interior_chem.redox import restore_tracker_state, store_profile_snapshot

    config = _resume_config(module)
    dirs = {'output': str(tmp_path)}
    (tmp_path / 'data').mkdir()
    if module == 'aragog':
        _make_int_snapshot(str(tmp_path / 'data' / '1722p000_int.nc'), _PHI.size)

    unbroken = _make_interior()
    _run_steps(unbroken, config, range(6))

    first = _make_interior()
    _run_steps(first, config, range(3))
    assert store_profile_snapshot(
        config,
        dirs,
        1722.0,
        first,
        {'fO2_shift_IW_mantle': 0.0, 'ferric_frac_mantle': first.redox_state.ferric_frac},
    )
    resumed = _make_interior()
    assert restore_tracker_state(config, dirs, 1722.0, resumed)
    _run_steps(resumed, config, range(3, 6))

    a, b = unbroken.redox_state, resumed.redox_state
    assert b.n_fe2_melt == pytest.approx(a.n_fe2_melt, rel=1e-12)
    assert b.n_fe3_melt == pytest.approx(a.n_fe3_melt, rel=1e-12)
    assert b.ferric_frac == pytest.approx(a.ferric_frac, rel=1e-12)
    np.testing.assert_allclose(b.n_fe_metal_cell, a.n_fe_metal_cell, rtol=1e-12, atol=0)
    assert np.sum(a.n_fe_metal_cell) > 0.0  # the history being carried includes metal

    # Discrimination guard: a reseeded tracker ends far from the unbroken run.
    reseeded = _make_interior()
    _run_steps(reseeded, config, range(3, 6))
    assert abs(reseeded.redox_state.ferric_frac - a.ferric_frac) > 0.1 * a.ferric_frac


def test_restored_state_matches_every_stored_field(tmp_path):
    """read_tracker_state returns each MeltRedoxState field the tracker
    carries between steps, with types preserved (flags as bool, indices as
    int); the per-step fO2 profiles are not stored."""
    from proteus.interior_chem.redox import read_tracker_state, write_fO2_profile_ncdf

    interior = _make_interior()
    _run_steps(interior, _resume_config(), range(4))
    st = interior.redox_state
    st.melt_exhausted, st.eos_coverage_logged = True, True  # non-default flags
    fpath = str(tmp_path / '1_int.nc')
    _make_int_snapshot(fpath, _PHI.size)
    assert write_fO2_profile_ncdf(fpath, st)

    got = read_tracker_state(fpath)
    assert np.sum(st.n_fe2_solid_cell) > 0.0  # the solid iron being stored is not trivial
    for name in (
        'phi_prev',
        'D_fe3_cell',
        'D_fe2_cell',
        'n_fe_metal_cell',
        'm_solid_cell',
        'n_fe2_solid_cell',
        'n_fe3_solid_cell',
        'a_fe_cell',
    ):
        np.testing.assert_array_equal(getattr(got, name), getattr(st, name))
    for name in ('n_fe3_melt', 'n_fe2_melt', 'ferric_frac', 'redox_ratio', 'w_feo15'):
        assert getattr(got, name) == getattr(st, name), name
    assert (got.a_fe_max_cell, got.fO2_cell) == (st.a_fe_max_cell, st.fO2_cell)
    assert got.melt_exhausted is True and got.eos_coverage_logged is True
    assert got.profile_pressure_term is st.profile_pressure_term
    assert got.log10_fO2_cell is None


def test_a_state_stored_without_per_cell_d_fe2_rebuilds_it_from_the_d_fe3_regime(
    tmp_path, monkeypatch, caplog
):
    """A snapshot written before D(Fe2+) was a per-cell field still resumes:
    each cell gets the D(Fe2+) of the assemblage its stored D(Fe3+) marks,
    and the substitution is logged."""
    from proteus.interior_chem import redox

    interior = _make_interior()
    _run_steps(interior, _resume_config(), range(2))
    st = interior.redox_state
    fpath = str(tmp_path / '1_int.nc')
    _make_int_snapshot(fpath, _PHI.size)
    with monkeypatch.context() as m:
        m.delitem(redox._STATE_CELL, 'D_fe2_cell')
        assert redox.write_fO2_profile_ncdf(fpath, st)

    with caplog.at_level('WARNING'):
        got = redox.read_tracker_state(fpath)
    np.testing.assert_array_equal(got.D_fe2_cell, st.D_fe2_cell)
    # The shallow cell takes the Cpx/Opx value, not the bridgmanite one.
    assert got.D_fe2_cell[0] == pytest.approx(D_FE2_SHALLOW, rel=1e-12)
    assert abs(got.D_fe2_cell[0] - D_FE2_BRG) > 0.02
    assert got.n_fe2_melt == st.n_fe2_melt
    assert sum('no per-cell D(Fe2+)' in r.message for r in caplog.records) == 1


def test_a_snapshot_without_state_warns_and_leaves_the_tracker_unset(tmp_path, caplog):
    """A resume onto a snapshot written before the state was stored (or a
    missing file) cannot restore: it says so and the tracker restarts."""
    from proteus.interior_chem.redox import restore_tracker_state

    config = _resume_config()
    (tmp_path / 'data').mkdir()
    _make_int_snapshot(str(tmp_path / 'data' / '1722p000_int.nc'), _PHI.size)
    interior = _make_interior()
    with caplog.at_level('WARNING'):
        assert not restore_tracker_state(config, {'output': str(tmp_path)}, 1722.0, interior)
        assert not restore_tracker_state(config, {'output': str(tmp_path)}, 99.0, interior)
    assert interior.redox_state is None
    assert sum('no melt-redox tracker state' in r.message for r in caplog.records) == 2
    # Another fO2 source never looks for a state.
    other = _resume_config()
    other.planet.fO2_source = 'user_constant'
    assert not restore_tracker_state(other, {'output': str(tmp_path)}, 1722.0, interior)


def test_a_restored_state_on_a_different_grid_is_refused():
    """A state whose cell count differs from the interior grid cannot be
    continued; update_melt_redox raises instead of broadcasting."""
    config = _resume_config()
    interior = _make_interior()
    _run_steps(interior, config, range(2))
    interior.phi = np.array([1.0, 0.5, 0.2, 0.0])
    interior.mass = np.full(4, 1.0e21)
    interior.pres = np.linspace(1.0e9, 6.0e10, 4)
    interior.temp = np.full(4, 2500.0)
    n2 = interior.redox_state.n_fe2_melt
    with pytest.raises(ValueError, match='3 cells but the interior grid has 4'):
        update_melt_redox(interior, {'T_magma': 2200.0}, config)
    assert interior.redox_state.n_fe2_melt == n2


def test_a_state_stored_without_solid_iron_resumes_with_none_and_warns(
    tmp_path, monkeypatch, caplog
):
    """A snapshot written before the solid iron was stored still resumes: the
    solid of every cell starts with no tracked iron, and this is logged."""
    from proteus.interior_chem import redox

    interior = _make_interior()
    _run_steps(interior, _resume_config(), range(2))
    st = interior.redox_state
    fpath = str(tmp_path / '1_int.nc')
    _make_int_snapshot(fpath, _PHI.size)
    with monkeypatch.context() as m:
        for name in ('m_solid_cell', 'n_fe2_solid_cell', 'n_fe3_solid_cell'):
            m.delitem(redox._STATE_CELL, name)
        assert redox.write_fO2_profile_ncdf(fpath, st)

    with caplog.at_level('WARNING'):
        got = redox.read_tracker_state(fpath)
    for name in ('m_solid_cell', 'n_fe2_solid_cell', 'n_fe3_solid_cell'):
        assert np.all(getattr(got, name) == 0.0), name
    assert got.n_fe2_solid_cell.shape == st.phi_prev.shape
    assert got.n_fe2_melt == st.n_fe2_melt
    assert sum('no solid iron per cell' in r.message for r in caplog.records) == 1


# ── Remelting returns the iron of the remelted solid (Step 3b) ─────────────


def _remelt_run(phis, f_0=0.1):
    """Tracker without metal saturation through the melt-fraction profiles
    ``phis``; returns the interior after the last one."""
    config = _make_config(f_0, metal_saturation=False)
    interior = _make_interior()
    for phi in phis:
        interior.phi = np.asarray(phi, dtype=float)
        update_melt_redox(interior, {'T_magma': 2200.0}, config)
    return interior


def _fe_total(st):
    return st.n_fe2_melt + st.n_fe3_melt + np.sum(st.n_fe2_solid_cell + st.n_fe3_solid_cell)


@pytest.mark.physics_invariant
def test_iron_is_conserved_between_melt_and_solid_through_freezing_and_remelting():
    """Without metal, melt plus solid Fe2+ and Fe3+ stay at the initial melt
    inventory on every step of a freeze then remelt, and each species is
    conserved on its own."""
    phis = [_PHI, [0.7, 0.3, 0.0], [0.4, 0.1, 0.0], [0.6, 0.35, 0.0], [0.9, 0.5, 0.0]]
    n2_0 = W_FET * _M_MELT / MU_FEO
    n3_0 = n2_0 * 0.1 / 0.9
    for k in range(1, len(phis) + 1):
        st = _remelt_run(phis[:k]).redox_state
        assert _fe_total(st) == pytest.approx(n2_0 + n3_0, rel=1e-12)
        assert st.n_fe2_melt + np.sum(st.n_fe2_solid_cell) == pytest.approx(n2_0, rel=1e-12)
        assert np.all(st.n_fe2_solid_cell >= 0.0) and np.all(st.n_fe3_solid_cell >= 0.0)
    # Guard: the remelt steps put iron back; dropping it leaves the melt short.
    frozen = _remelt_run(phis[:3]).redox_state
    assert st.n_fe2_melt > frozen.n_fe2_melt * 1.2


@pytest.mark.physics_invariant
def test_remelting_back_to_the_initial_melt_fraction_restores_the_initial_ferric_fraction():
    """Solid that formed from the melt and remelts completely returns all its
    iron, so the reservoirs and Fe3+/FeT go back to their initial values
    whatever D the solid took up."""
    st = _remelt_run([_PHI, [0.6, 0.2, 0.0], [0.3, 0.05, 0.0], _PHI]).redox_state
    n2_0 = W_FET * _M_MELT / MU_FEO
    assert st.n_fe2_melt == pytest.approx(n2_0, rel=1e-12)
    assert st.n_fe3_melt == pytest.approx(n2_0 * 0.1 / 0.9, rel=1e-12)
    assert st.ferric_frac == pytest.approx(0.1, rel=1e-12)
    np.testing.assert_allclose(st.n_fe2_solid_cell, 0.0, atol=1e-12 * n2_0)
    # Guard: before the remelt the ratio had moved well away from f_0.
    frozen = _remelt_run([_PHI, [0.6, 0.2, 0.0], [0.3, 0.05, 0.0]]).redox_state
    assert abs(frozen.ferric_frac - 0.1) > 0.002
    # Guard: spreading the iron of cell 1 over its initial solid as well
    # returns only 0.45 / 0.95 of it, 5 % of n2_0 short.
    diluted = (
        frozen.n_fe2_melt
        + frozen.n_fe2_solid_cell[0]
        + 0.45 / 0.95 * frozen.n_fe2_solid_cell[1]
    )
    assert n2_0 - diluted > 0.01 * n2_0


@pytest.mark.physics_invariant
def test_a_partial_remelt_returns_the_remelted_fraction_of_the_cell_solid():
    """A cell at phi = 0.2 that rises to 0.6 has remelted half of its solid,
    (0.6 - 0.2) / (1 - 0.2), and returns half of the iron that solid holds;
    the other cells are untouched."""
    before = _remelt_run([_PHI, [0.2, 0.5, 0.0]]).redox_state
    s2, s3 = before.n_fe2_solid_cell.copy(), before.n_fe3_solid_cell.copy()
    st = _remelt_run([_PHI, [0.2, 0.5, 0.0], [0.6, 0.5, 0.0]]).redox_state

    assert st.n_fe2_melt == pytest.approx(before.n_fe2_melt + 0.5 * s2[0], rel=1e-12)
    assert st.n_fe3_melt == pytest.approx(before.n_fe3_melt + 0.5 * s3[0], rel=1e-12)
    assert st.n_fe2_solid_cell[0] == pytest.approx(0.5 * s2[0], rel=1e-12)
    np.testing.assert_array_equal(st.n_fe2_solid_cell[1:], s2[1:])
    # Guard: returning (phi - phi_prev) = 0.4 of the solid iron instead is 20 % short.
    assert abs(st.n_fe2_melt - (before.n_fe2_melt + 0.4 * s2[0])) > 0.05 * s2[0]


@pytest.mark.physics_invariant
def test_solid_present_at_the_first_step_returns_no_iron_on_remelting():
    """The deep cell starts solid with no tracked iron: remelting it adds melt
    mass but no iron, so the reservoirs and Fe3+/FeT stay as they were. A
    cell remelting past the solid the tracker formed returns that iron once."""
    before = _remelt_run([_PHI]).redox_state
    st = _remelt_run([_PHI, [1.0, 0.5, 0.5]]).redox_state
    assert (st.n_fe2_melt, st.n_fe3_melt) == (before.n_fe2_melt, before.n_fe3_melt)
    assert st.ferric_frac == before.ferric_frac
    assert np.all(st.m_solid_cell == 0.0)

    # Edge: remelting past the tracked solid of a cell returns all of its iron
    # once, and the initial solid underneath adds none.
    frozen = _remelt_run([_PHI, [1.0, 0.2, 0.0]]).redox_state
    over = _remelt_run([_PHI, [1.0, 0.2, 0.0], [1.0, 0.9, 0.0]]).redox_state
    assert over.n_fe2_melt == pytest.approx(
        frozen.n_fe2_melt + frozen.n_fe2_solid_cell[1], rel=1e-12
    )
    assert over.m_solid_cell[1] == 0.0 and over.n_fe2_solid_cell[1] == 0.0
    assert frozen.m_solid_cell[1] == pytest.approx(0.3 * _MASS[1], rel=1e-12)


@pytest.mark.physics_invariant
def test_a_remelt_after_full_solidification_restarts_the_tracker(caplog):
    """A fully solidified mantle that remelts, for example after a giant
    impact, gets its melt reservoirs back from the remelted solid and the
    tracker runs again instead of staying frozen."""
    config = _make_config(0.1, metal_saturation=False)
    interior = _make_interior()
    for phi in (_PHI, [0.5, 0.2, 0.0], np.zeros(3), np.zeros(3)):
        interior.phi = np.asarray(phi)
        update_melt_redox(interior, {'T_magma': 2200.0}, config)
    st = interior.redox_state
    assert st.melt_exhausted
    total = _fe_total(st)
    frozen_f = st.ferric_frac

    with caplog.at_level('INFO', logger='fwl.proteus.interior_chem.redox'):
        interior.phi = np.array([1.0, 0.5, 0.0])
        update_melt_redox(interior, {'T_magma': 2200.0}, config)
    assert not st.melt_exhausted
    assert _fe_total(st) == pytest.approx(total, rel=1e-12)
    assert np.sum(st.n_fe2_solid_cell[0]) == pytest.approx(0.0, abs=1e-12 * total)
    assert abs(st.ferric_frac - frozen_f) > 1e-4  # the returned solid has its own ratio
    assert any('remelts, tracking resumes' in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Crystallisation step against Schaefer et al. (2024)
# ---------------------------------------------------------------------------

# One converged run of the archived Schaefer et al. (2024) 1000 km model;
# fixtures/make_schaefer2024_replay.py writes it from their output.mat.
_SCHAEFER = np.loadtxt(
    Path(__file__).parent / 'fixtures' / 'schaefer2024_1000km_replay.csv', delimiter=','
)
_SCH_P, _SCH_S, _SCH_D2, _SCH_D3, _SCH_F = _SCHAEFER[:, 1:].T


def _replay(D_fe2, D_fe3, f_0=0.1):
    """Molar Fe3+/FeT of one well-mixed melt after each step of the Schaefer
    sequence, crystallised with _fe_into_new_solid; the magma ocean mass
    shrinks by the solid fraction s of each step."""
    n2, n3, M, out = 1.0 - f_0, f_0, 1.0, []
    for s, d2, d3 in zip(_SCH_S, D_fe2, D_fe3):
        dn2, dn3 = _fe_into_new_solid(
            n2, n3, np.array([M]), np.array([s * M]), np.array([d2]), np.array([d3])
        )
        n2, n3, M = n2 - dn2, n3 - dn3, M * (1.0 - s)
        out.append(n3 / (n2 + n3))
    return np.array(out)


@pytest.mark.physics_invariant
def test_new_solid_takes_up_d_times_the_pre_step_melt_concentration():
    """Step 4-6: with two melt cells of unequal mass and D, the solid takes
    n * sum_c(D_c * m_c) / M_melt, a cell without melt contributes nothing,
    and no new solid moves no iron."""
    M = np.array([3.0e21, 1.0e21, 0.0])  # kg of melt per cell
    m_new = np.array([0.6e21, 0.5e21, 2.0e21])  # kg of new solid per cell
    D2, D3 = np.array([0.3, 0.9, 5.0]), np.array([0.7, 0.2, 5.0])
    dn2, dn3 = _fe_into_new_solid(100.0, 10.0, M, m_new, D2, D3)

    # 100 mol * (0.3 * 0.6 + 0.9 * 0.5) / 4 = 15.75 mol; the empty cell is ignored.
    assert dn2 == pytest.approx(15.75, rel=1e-12)
    assert dn3 == pytest.approx(10.0 * (0.7 * 0.6 + 0.2 * 0.5) / 4.0, rel=1e-12)
    # Guard: weighting by the cell's own melt mass alone (no 1/M_melt) is off by 4.
    assert abs(dn2 - 4.0 * 15.75) > 10.0
    assert _fe_into_new_solid(100.0, 10.0, M, np.zeros(3), D2, D3) == (0.0, 0.0)
    # Limit: D = 1 and every kg of melt crystallising removes all the iron.
    full = _fe_into_new_solid(100.0, 10.0, M[:2], M[:2], np.ones(2), np.ones(2))
    assert full == pytest.approx((100.0, 10.0), rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_crystallisation_step_replays_the_schaefer_2024_run():
    """The tracker's crystallisation step, fed the bulk D(FeO), D(FeO1.5)
    and solid fractions of a run of Schaefer et al. (2024, archived Earth
    code, 1000 km, Fe3+/FeT = 0.1), reproduces its liquid Fe3+/FeT.

    Schaefer equilibrates the solid with the residual liquid; the tracker
    uses the melt before the step. Per step the moles left differ by a
    relative s**2 * |D (1 - D)| / ((1 - s + s D)(1 - s D)), so f differs by
    at most f (1 - f) times the running sum of that over both species.
    """
    f = _replay(_SCH_D2, _SCH_D3)

    # The fixture is Schaefer's equilibrium mass balance step by step.
    n2, n3 = np.cumprod((1 - _SCH_S) / (1 - _SCH_S + _SCH_S * np.stack([_SCH_D2, _SCH_D3])), 1)
    np.testing.assert_allclose(0.1 * n3 / (0.9 * n2 + 0.1 * n3), _SCH_F, rtol=1e-10)

    def rel(D):
        s = _SCH_S
        return s**2 * np.abs(D * (1 - D)) / ((1 - s + s * D) * (1 - s * D))

    bound = _SCH_F * (1 - _SCH_F) * np.cumsum(rel(_SCH_D2) + rel(_SCH_D3))
    assert np.all(np.abs(f - _SCH_F) <= bound)
    # Below 22 GPa the steps are small and the two agree to 1e-4.
    assert np.max(np.abs(f - _SCH_F)[_SCH_P >= P_CUTOFF_GPA]) < 1e-4
    # Guards: the bulk peridotite D(FeO) = 0.85, or the two D swapped, miss
    # by more than ten times the largest bound.
    assert np.max(np.abs(_replay(np.full_like(_SCH_D2, 0.85), _SCH_D3) - _SCH_F)) > 0.05
    assert np.max(np.abs(_replay(_SCH_D3, _SCH_D2) - _SCH_F)) > 0.05
    assert bound.max() < 0.007


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_bridgmanite_coefficients_follow_schaefer_2024_in_the_lower_mantle():
    """With the tracker's own MgSiO3 coefficients on the Schaefer et al.
    (2024) crystallisation sequence, the lower mantle (P >= 22 GPa) loses
    ferric iron as in their run: Fe3+/FeT falls monotonically, by 60-100 %
    of their drop, and stays within 5e-3 of it (3.4e-3 at 22 GPa). The drop
    is smaller because their layer averages D(Fe3+) over truncated random
    draws (bulk 0.87, against 0.75 here) and holds 5 % magnesiowuestite.
    """
    deep = _SCH_P >= P_CUTOFF_GPA
    D2 = np.where(deep, D_FE2_BRG, D_FE2_SHALLOW)
    D3 = np.where(deep, D_FE3_BRG, D_FE3_SHALLOW)
    f = _replay(D2, D3)[deep]

    assert np.all(np.diff(f) < 0.0) and f[0] < 0.1
    drop_ratio = (0.1 - f[-1]) / (0.1 - _SCH_F[deep][-1])
    assert 0.6 < drop_ratio < 1.0
    assert np.max(np.abs(f - _SCH_F[deep])) < 5e-3
    # Guard: the former D(Fe2+) = 0.85 raises Fe3+/FeT where Schaefer's falls.
    old = _replay(np.full_like(D2, 0.85), D3)[deep]
    assert old[-1] > 0.1 and old[-1] - _SCH_F[deep][-1] > 0.01


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_pyroxene_coefficients_give_the_shallow_rise_of_schaefer_2024():
    """Below 15 GPa Fe3+/FeT rises in the Schaefer et al. (2024) run (mainly
    olivine, which takes up no Fe3+), and with the tracker's pyroxene
    coefficients it rises too, because their Table S1 D(FeO) exceeds D(Fe3+).
    The turning point sits at the 22 GPa cutoff here and near 15 GPa in
    their run, since the transition-zone phases are absent; the rise is
    larger here (0.019 against 0.008 from 15 to 2.5 GPa).
    """
    deep = _SCH_P >= P_CUTOFF_GPA
    shallow = (_SCH_P < 15.0) & (_SCH_P >= 2.5)
    D3 = np.where(deep, D_FE3_BRG, D_FE3_SHALLOW)
    f = _replay(np.where(deep, D_FE2_BRG, D_FE2_SHALLOW), D3)

    rise, rise_sch = f[shallow][-1] - f[shallow][0], _SCH_F[shallow][-1] - _SCH_F[shallow][0]
    assert rise_sch > 0.0
    assert np.all(np.diff(f[shallow]) > 0.0)
    assert rise_sch < rise < 3.0 * rise_sch
    assert 15.0 <= _SCH_P[np.argmin(_SCH_F)] < _SCH_P[np.argmin(f)] < 23.0
    # Guard: the pyroxene D(FeO) her model produces against a magma-ocean
    # liquid (Cpx 0.33, Opx 0.36) lies below D(Fe3+) and gives a fall instead.
    model_run = _replay(np.where(deep, D_FE2_BRG, 0.345), D3)
    assert model_run[shallow][-1] < model_run[shallow][0]


@pytest.mark.physics_invariant
def test_bulk_prescription_takes_the_schaefer_layer_of_each_cell():
    """'bulk_BSE' gives each cell the bulk D of the Schaefer et al. (2024)
    layer that holds its pressure; a boundary pressure belongs to the deeper
    layer, and the layer below 1 GPa repeats the 1-2.5 GPa values."""
    P_gpa = np.array([0.5, 1.0, 2.0, 2.5, 10.0, 15.0, 17.0, 18.0, 20.0, 22.0, 60.0])
    D2, D3 = _partition_coefficients(P_gpa * 1e9, 'bulk_BSE')

    layer = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    np.testing.assert_array_equal(D2, np.asarray(BULK_D_FE2)[layer])
    np.testing.assert_array_equal(D3, np.asarray(BULK_D_FE3)[layer])
    assert BULK_P_BOUNDS_GPA == (1.0, 2.5, 15.0, 18.0, 22.0)
    # Transition-zone layers hold Fe3+ more strongly than the melt; the
    # 2.5-15 GPa layer (olivine-rich) rejects it.
    assert np.all(D3[5:9] > 1.0) and D3[4] < D2[4]
    # Guard: the mgsio3 prescription differs at 17 GPa (no transition-zone phase).
    assert abs(_partition_coefficients(17e9, 'mgsio3')[1] - D3[6]) > 0.5


def test_an_unknown_prescription_is_refused_before_the_tracker_starts():
    """A prescription name the tracker does not know raises, and the
    interior keeps no half-built state."""
    with pytest.raises(ValueError, match='unknown partition-coefficient set'):
        _init_state(_PHI, _MASS, _PRES, 0.1, 'bulk_bse')
    interior = _make_interior()
    config = _make_config(0.1)
    config.planet.partition_coefficients = 'schaefer'
    with pytest.raises(ValueError, match="'schaefer'"):
        update_melt_redox(interior, {'T_magma': 2200.0}, config)
    assert interior.redox_state is None


@pytest.mark.physics_invariant
def test_configured_prescription_sets_the_coefficients_of_the_run():
    """planet.partition_coefficients = 'bulk_BSE' reaches the tracker state:
    the 10 GPa cell takes the 2.5-15 GPa bulk values and the 30 and 50 GPa
    cells the lower-mantle ones, and the first crystallisation step uses them."""
    config = _make_config(0.1, metal_saturation=False)
    config.planet.partition_coefficients = 'bulk_BSE'
    interior = _make_interior()
    update_melt_redox(interior, {'T_magma': 2200.0}, config)
    st = interior.redox_state
    np.testing.assert_array_equal(st.D_fe2_cell, [BULK_D_FE2[2], BULK_D_FE2[5], BULK_D_FE2[5]])
    np.testing.assert_array_equal(st.D_fe3_cell, [BULK_D_FE3[2], BULK_D_FE3[5], BULK_D_FE3[5]])

    n2_0, n3_0 = st.n_fe2_melt, st.n_fe3_melt
    interior.phi = np.array([0.8, 0.5, 0.0])  # 0.2 x 2e21 kg crystallises at 10 GPa
    update_melt_redox(interior, {'T_magma': 2200.0}, config)
    dM = 0.2 * _MASS[0]
    assert st.n_fe2_melt == pytest.approx(n2_0 * (1 - BULK_D_FE2[2] * dM / _M_MELT), rel=1e-12)
    assert st.n_fe3_melt == pytest.approx(n3_0 * (1 - BULK_D_FE3[2] * dM / _M_MELT), rel=1e-12)
    # Guard: with the mgsio3 pyroxene D(Fe3+) the Fe3+ loss differs by 22 %.
    assert abs(BULK_D_FE3[2] - D_FE3_SHALLOW) / D_FE3_SHALLOW > 0.2


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_bulk_prescription_follows_the_schaefer_2024_run():
    """With 'bulk_BSE' on the Schaefer et al. (2024) crystallisation
    sequence, Fe3+/FeT falls through the lower mantle and transition zone to
    a minimum between 12 and 18 GPa and then rises, as in their run, and
    stays within 0.01 of run 570 at every step (constant layer medians
    against the per-step values of one draw; their mean is within 0.005).
    """
    P = _SCH_P * 1e9
    f = _replay(*_partition_coefficients(P, 'bulk_BSE'))

    assert np.max(np.abs(f - _SCH_F)) < 0.01
    assert 12.0 < _SCH_P[np.argmin(f)] < 18.0
    assert 12.0 < _SCH_P[np.argmin(_SCH_F)] < 18.0
    assert f[-1] > f.min() + 0.01
    # Guard: the mgsio3 coefficients miss by more than 0.02 and turn at 22 GPa.
    f_mg = _replay(*_partition_coefficients(P, 'mgsio3'))
    assert np.max(np.abs(f_mg - _SCH_F)) > 0.02
    assert _SCH_P[np.argmin(f_mg)] > 21.0
