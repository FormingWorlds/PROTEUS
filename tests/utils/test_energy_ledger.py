"""Tests of ``proteus.utils.energy_ledger.whole_run_ledger``: the whole-run energy ledger of a helpfile.

Invariants checked: a synthetic run whose booked terms close returns a planet residual of 0 to round-off;
an injected residual comes back with its size; every term enters with its size and sign (a ledger without
it moves the closure by that size); the coupling identity G = 0 holds on every no-lid step and its failure
raises; the mantle residual equals the stored cumulative column; bad input raises and returns no number.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from proteus.utils.constants import secs_per_year
from proteus.utils.energy_ledger import whole_run_ledger
from tests.utils._ledger_tables import _table

pytest.importorskip('hypothesis')
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.physics_invariant
@pytest.mark.parametrize('core_bc', ['energy_balance', 'core_module'])
def test_a_closed_synthetic_run_closes_and_returns_an_injected_residual(core_bc):
    """The booked terms of the synthetic run close: R_p is 0 to 1e-12 of the throughput T on both
    core paths; 1e24 J added to the mantle heat of one step comes back as R_p and as R_m."""
    led = whole_run_ledger(_table(core_bc), core_bc)
    assert abs(led.R_p.sum()) <= 1e-12 * led.throughput
    assert abs(led.closure) <= 1e-12
    assert led.throughput > 1e27  # the terms are summed, not left at 0
    hf = _table(core_bc)
    hf.loc[5, 'step_dE_state_heat_J'] += 1e24
    hf.loc[5:, 'E_residual_cons_J'] += 1e24
    led = whole_run_ledger(hf, core_bc)
    assert led.R_p.sum() == pytest.approx(1e24, rel=1e-9)
    assert led.R_m.sum() == pytest.approx(1e24, rel=1e-9)
    assert led.R_c.sum() == pytest.approx(0.0, abs=1e-12 * led.throughput)


@pytest.mark.physics_invariant
def test_radiogenic_heat_enters_the_mantle_ledger():
    """Q enters the mantle balance: a ledger without Q would move R_m by -sum Q (5e24 J here)."""
    hf = _table()
    led = whole_run_ledger(hf, 'energy_balance')
    Q = hf['step_dE_Q_radio_J'].iloc[3:].sum()
    assert led.terms['Q'].sum() == pytest.approx(Q, rel=1e-12)
    assert abs(led.R_m.sum()) <= 1e-12 * led.throughput
    assert abs(led.R_m.sum() + Q) > 0.5 * Q  # a dropped Q fails by its own size


@pytest.mark.physics_invariant
def test_tidal_heat_enters_the_mantle_ledger():
    """Tidal heat is booked with Q: 1e24 J of tidal heat in one step, balanced in the mantle heat, closes."""
    hf = _table()
    hf.loc[4, 'step_dE_Q_tidal_J'] = 1e24
    hf.loc[4, 'step_dE_state_heat_J'] += 1e24
    led = whole_run_ledger(hf, 'energy_balance')
    assert abs(led.R_m.sum()) <= 1e-12 * led.throughput
    assert led.terms['Q'][1] == pytest.approx(hf.loc[4, 'step_dE_Q_radio_J'] + 1e24, rel=1e-12)


@pytest.mark.physics_invariant
def test_the_cmb_heat_enters_the_core_side():
    """The CMB heat closes the core: R_c is 0 with C_cmb, and a core ledger without it would miss by
    sum C_cmb (about 2e27 J)."""
    hf = _table()
    led = whole_run_ledger(hf, 'energy_balance')
    C = hf['step_dE_F_cmb_J'].iloc[3:].sum()
    assert abs(led.R_c.sum()) <= 1e-12 * led.throughput
    assert led.terms['D_core'].sum() == pytest.approx(-C, rel=1e-9)
    assert C > 1e26


@pytest.mark.physics_invariant
def test_core_module_books_the_structure_update_of_the_core():
    """A core heat jump at an update that core_module books in step_dE_struct_core_J closes R_c;
    without the booked term the core residual would be the jump (3e26 J)."""
    hf = _table('core_module')
    hf.loc[6, 'step_dE_core_J'] += 3e26
    hf.loc[6, 'step_dE_struct_core_J'] = 3e26
    led = whole_run_ledger(hf, 'core_module')
    assert abs(led.R_c.sum()) <= 1e-12 * led.throughput
    assert led.terms['B_core'].sum() == pytest.approx(3e26, rel=1e-12)


@pytest.mark.physics_invariant
def test_the_loss_to_space_uses_the_area_of_the_start_row():
    """A step runs on the structure of its start row: with R_int changed at the end row of a step,
    E_atm uses 4 pi R_int(start)^2 and G stays 0; the end-row area would give G = -2.2 % of that L_int."""
    hf = _table()
    hf.loc[6:, 'R_int'] = 6.47e6  # update at row 6: the step ending at row 6 started on 6.4e6
    k = 6
    A0 = 4.0 * np.pi * 6.4e6**2
    for j in range(k, len(hf) - 1):  # the steps that start on the new structure
        dt_j = (hf.loc[j + 1, 'Time'] - hf.loc[j, 'Time']) * secs_per_year
        hf.loc[j, 'F_atm'] = -hf.loc[j + 1, 'step_dE_F_int_J'] / (
            4.0 * np.pi * 6.47e6**2 * dt_j
        )
    led = whole_run_ledger(hf, 'energy_balance')
    assert abs(led.G.sum()) <= 1e-12 * led.throughput
    assert led.classes['update'][k - 3]  # the step that ends at row 6 ends on a new structure
    dt = (hf.loc[k, 'Time'] - hf.loc[k - 1, 'Time']) * secs_per_year
    G_end = led.terms['L_int'][k - 3] - hf.loc[k - 1, 'F_atm'] * 4.0 * np.pi * 6.47e6**2 * dt
    assert G_end / led.terms['L_int'][k - 3] == pytest.approx(
        1.0 - (6.47e6 / 6.4e6) ** 2, rel=1e-9
    )
    assert (
        abs(led.terms['E_atm'][k - 3] - hf.loc[k - 1, 'F_atm'] * A0 * dt)
        <= 1e-12 * A0 * dt * hf.loc[k - 1, 'F_atm']
    )


@pytest.mark.physics_invariant
def test_a_lid_at_the_start_row_exempts_the_step_from_the_coupling_identity():
    """With a live lid at the start row of a step, F_int differs from F_atm on purpose: the step is in
    the lid class and G is reported, not refused; the same gap on a no-lid step raises."""
    hf = _table()
    hf.loc[4, 'lid_weight'] = 0.3
    hf.loc[5, 'step_dE_F_int_J'] *= 1.01
    hf.loc[5, 'step_dE_state_heat_J'] = (
        -(-hf.loc[5, 'step_dE_F_int_J'])
        + hf.loc[5, 'step_dE_F_cmb_J']
        + hf.loc[5, 'step_dE_Q_radio_J']
    )
    led = whole_run_ledger(hf, 'energy_balance')
    assert led.classes['lid'][2]  # step 2 ends at row 5 and starts at row 4
    assert abs(led.G[2]) > 1e-3 * abs(led.terms['L_int'][2])
    hf.loc[4, 'lid_weight'] = 0.0
    with pytest.raises(RuntimeError, match='coupling identity'):
        whole_run_ledger(hf, 'energy_balance')


@pytest.mark.physics_invariant
def test_the_mantle_residual_must_equal_the_stored_cumulative_column():
    """R_m over the window equals E_residual_cons_J(end) - E_residual_cons_J(start); a stored column
    that disagrees (a ledger that drops a term) raises."""
    hf = _table()
    led = whole_run_ledger(hf, 'energy_balance')
    assert led.R_m.sum() == pytest.approx(0.0, abs=1e-12 * led.throughput)
    hf.loc[7:, 'E_residual_cons_J'] = 5e25
    with pytest.raises(RuntimeError, match='E_residual_cons_J'):
        whole_run_ledger(hf, 'energy_balance')


def test_the_stored_residual_check_allows_the_round_off_of_the_cumulative_columns():
    """E_residual_cons_J is the difference of 2 whole-run sums of about 1e32 J, so its window difference
    carries their round-off: 5e16 J (about 2 eps of 1e32) passes, 1e19 J (above 7 steps x eps x 2e32)
    raises."""
    hf = _table()
    hf['E_state_heat_cons_J'] = -1.1e32
    hf['dE_predicted_cons_J'] = -1.1e32
    hf.loc[7:, 'E_residual_cons_J'] = 5e16
    led = whole_run_ledger(hf, 'energy_balance')
    assert abs(led.R_m.sum()) <= 1e-12 * led.throughput
    hf.loc[7:, 'E_residual_cons_J'] = 1e19
    with pytest.raises(RuntimeError, match='E_residual_cons_J'):
        whole_run_ledger(hf, 'energy_balance')


def test_a_helpfile_without_the_stored_residual_column_is_ledgered():
    """An older helpfile without E_residual_cons_J has no stored residual to compare: the ledger still
    closes the run and returns R_m, and an injected mantle residual still comes back."""
    hf = _table().drop(columns=['E_residual_cons_J'])
    led = whole_run_ledger(hf, 'energy_balance')
    assert abs(led.R_m.sum()) <= 1e-12 * led.throughput
    hf.loc[5, 'step_dE_state_heat_J'] += 1e24
    assert whole_run_ledger(hf, 'energy_balance').R_m.sum() == pytest.approx(1e24, rel=1e-9)


def test_the_mantle_mass_change_of_each_step_is_reported():
    """The mantle mass gained at a structure update is reported per step in kg and as a fraction of
    the mantle mass; the steps without a change report 0."""
    hf = _table()
    hf.loc[6:, 'M_mantle'] = 4.0e24 + 6.9e20
    led = whole_run_ledger(hf, 'energy_balance')
    assert led.dM_mantle[3] == pytest.approx(6.9e20, rel=1e-6)
    assert led.dM_mantle_frac[3] == pytest.approx(6.9e20 / 4.0e24, rel=1e-6)
    assert np.count_nonzero(led.dM_mantle) == 1


def test_steps_are_classed_by_update_remap_zero_energy_and_impact():
    """Classes: a step that ends at a new structure is 'update', the step after it 'remap'; a step
    with dt > 0 and no energy is 'zero'; a step with impact energy is 'impact'."""
    hf = _table()
    hf.loc[6:, 'R_core'] = 3.49e6
    hf.loc[8, 'step_dE_impact_J'] = 1e25  # booked on both sides of the mantle balance
    led = whole_run_ledger(hf, 'energy_balance')
    assert led.classes['update'].tolist() == [False, False, False, True, False, False, False]
    assert led.classes['remap'].tolist() == [False, False, False, False, True, False, False]
    assert led.classes['impact'].tolist() == [False, False, False, False, False, True, False]
    assert not led.classes['zero'].any()


def test_the_window_selects_the_steps_between_start_and_end_rows():
    """start_row and end_row bound the steps: rows 4 to 7 give the 3 steps ending at rows 5, 6, 7."""
    hf = _table()
    led = whole_run_ledger(hf, 'energy_balance', start_row=4, end_row=7)
    assert led.time.tolist() == hf['Time'].iloc[5:8].tolist()
    assert led.R_p.size == 3


@pytest.mark.parametrize(
    'kwargs, match',
    [
        ({'start_row': -1}, 'start_row'),
        ({'start_row': 9}, 'start_row'),
        ({'end_row': 2}, 'end_row'),
        ({'end_row': 10}, 'end_row'),
    ],
    ids=['negative-start', 'start-at-last-row', 'end-not-after-start', 'end-past-table'],
)
def test_rows_out_of_range_raise(kwargs, match):
    """A window outside the table raises ValueError that names the argument."""
    hf = _table()
    before = hf.copy()
    with pytest.raises(ValueError, match=match):
        whole_run_ledger(hf, 'energy_balance', **kwargs)
    assert hf.equals(before)  # a refused call leaves its input unchanged


@pytest.mark.parametrize('column', ['step_dE_F_int_J', 'F_atm', 'T_cmb_node', 'M_mantle'])
def test_a_missing_column_raises_and_names_it(column):
    """An older helpfile without a required column raises ValueError naming the column."""
    hf = _table().drop(columns=[column])
    before = hf.copy()
    with pytest.raises(ValueError, match=column):
        whole_run_ledger(hf, 'energy_balance')
    assert hf.equals(before)


def test_core_module_needs_its_core_columns():
    """core_module reads step_dE_core_J; a table without it raises."""
    hf = _table('core_module').drop(columns=['step_dE_core_J'])
    before = hf.copy()
    with pytest.raises(ValueError, match='step_dE_core_J'):
        whole_run_ledger(hf, 'core_module')
    assert hf.equals(before)


@pytest.mark.parametrize(
    'core_bc', ['bower2018', 'quasi_steady', 'gradient', '', 'Energy_Balance']
)
def test_an_unknown_core_bc_raises(core_bc):
    """Only energy_balance and core_module have a core ledger here; bower2018 keeps its own core
    temperature state (T_cmb) without the core temperature factor, so it raises with the others."""
    hf = _table()
    before = hf.copy()
    with pytest.raises(ValueError, match='core_bc'):
        whole_run_ledger(hf, core_bc)
    assert hf.equals(before)


@pytest.mark.parametrize(
    'bad', [math.nan, math.inf, -math.inf], ids=['nan', 'inf', 'minus-inf']
)
def test_a_non_finite_term_raises(bad):
    """NaN or inf in a summed term raises ValueError naming the column and the row."""
    hf = _table()
    hf.loc[6, 'step_dE_F_cmb_J'] = bad
    before = hf.copy()
    with pytest.raises(ValueError, match=r'step_dE_F_cmb_J.*row 6'):
        whole_run_ledger(hf, 'energy_balance')
    assert hf.equals(before)


def test_time_that_runs_backwards_raises():
    """A row earlier than its predecessor raises; a duplicated time with energy raises too."""
    hf = _table()
    hf.loc[6, 'Time'] = hf.loc[5, 'Time'] - 1.0
    with pytest.raises(ValueError, match='Time'):
        whole_run_ledger(hf, 'energy_balance')
    hf = _table()
    hf.loc[6, 'Time'] = hf.loc[5, 'Time']
    with pytest.raises(ValueError, match='Time'):
        whole_run_ledger(hf, 'energy_balance')


def test_a_negative_loss_to_space_is_flagged_not_refused():
    """F_atm < 0 (the atmosphere heats the interior) is a valid state: the step is flagged in its class
    and the ledger still closes."""
    hf = _table()
    k = 5
    hf.loc[k + 1, 'step_dE_F_int_J'] *= -1
    hf.loc[k, 'F_atm'] *= -1
    hf.loc[k + 1, 'step_dE_state_heat_J'] = (
        hf.loc[k + 1, 'step_dE_F_int_J']
        + hf.loc[k + 1, 'step_dE_F_cmb_J']
        + hf.loc[k + 1, 'step_dE_Q_radio_J']
    )
    led = whole_run_ledger(hf, 'energy_balance')
    assert led.classes['negative_F_atm'][k - 2]
    assert abs(led.R_p.sum()) <= 1e-12 * led.throughput


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 10_000),
    n_steps=st.integers(1, 12),
    scale=st.floats(1e-6, 1e6, allow_nan=False, allow_infinity=False),
)
def test_any_closed_run_closes_whatever_its_size(seed, n_steps, scale):
    """Property: a closed synthetic run of 1 to 12 steps, with every energy scaled by 1e-6 to 1e6,
    has |R_p| <= 1e-12 T; the closure is a ratio, so it does not change with the scale. It runs the
    core_module path: on energy_balance the core heat is C dT_cmb_node, and at a scale of 1e-6 dT lies
    below the float resolution of a 4000 K temperature."""
    hf = _table('core_module', n_steps=n_steps, seed=seed)
    cols = [
        c for c in hf.columns if c.startswith('step_dE_') or c in ('F_atm', 'E_residual_cons_J')
    ]
    hf[cols] *= scale
    led = whole_run_ledger(hf, 'core_module')
    assert abs(led.R_p.sum()) <= 1e-12 * led.throughput
    assert led.R_p.size == n_steps


@settings(max_examples=40, deadline=None)
@given(
    value=st.sampled_from([math.nan, math.inf, -math.inf]),
    row=st.integers(3, 9),
    column=st.sampled_from(['step_dE_Q_radio_J', 'step_dE_F_int_J', 'F_atm', 'T_cmb_node']),
)
def test_any_non_finite_value_in_a_read_column_raises(value, row, column):
    """Property: NaN or inf in any row of the window of a float column the ledger reads raises
    ValueError naming that column, never a number."""
    hf = _table()
    hf.loc[row, column] = value
    with pytest.raises(ValueError, match=column):
        whole_run_ledger(hf, 'energy_balance')
    assert (
        hf[column].dtype == np.float64
    )  # the column stays a float column: the value is refused


@pytest.mark.parametrize('value', [True, 'x'], ids=['bool-column', 'string-column'])
def test_a_column_that_is_not_float_raises(value):
    """A column of booleans or strings is refused by its type, even where its values would convert
    to numbers (True reads as 1.0 without the check)."""
    hf = _table()
    hf['step_dE_Q_radio_J'] = value
    with pytest.raises(ValueError, match='not numeric'):
        whole_run_ledger(hf, 'energy_balance')
    assert (hf['step_dE_Q_radio_J'] == value).all()


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'key, new', [('R_core', 3.49e6), ('M_core', 2.0002e24), ('R_int', 6.41e6)]
)
def test_a_window_that_starts_on_an_update_row_classes_its_first_step_as_remap(key, new):
    """A window whose start row holds a new structure (a change of R_core, M_core or R_int): its first
    step runs on the remapped profile and is in class remap, read from the row before the window, as in
    the full run."""
    hf = _table()
    hf.loc[4:, key] = new  # structure update at row 4
    full = _closed_coupling(hf)
    led = whole_run_ledger(hf, 'energy_balance', start_row=4)
    assert led.classes['remap'][0] and not led.classes['update'][0]
    assert led.classes['remap'].tolist() == full.classes['remap'][2:].tolist()
    assert not whole_run_ledger(hf, 'energy_balance', start_row=5).classes['remap'][0]


@pytest.mark.parametrize('bad', [math.nan, math.inf], ids=['nan', 'inf'])
def test_a_non_finite_row_before_the_window_raises(bad):
    """The row before the window classes its first step, so a non-finite structure value there raises
    instead of classing the step as remap."""
    hf = _table()
    hf.loc[3, 'R_core'] = bad
    before = hf.copy()
    with pytest.raises(ValueError, match=r'R_core.*row 3'):
        whole_run_ledger(hf, 'energy_balance', start_row=4)
    assert hf.equals(before)


def test_a_window_without_an_energy_step_is_refused():
    """A window in which no step books interior heat (a dummy or boundary interior) has no ledger:
    it raises instead of a closure of 0."""
    hf = _table()
    for c in [c for c in hf.columns if c.startswith('step_dE_')]:
        hf[c] = 0.0
    hf['E_residual_cons_J'] = 0.0
    with pytest.raises(ValueError, match='no step'):
        whole_run_ledger(hf, 'energy_balance')
    led = whole_run_ledger(_table(), 'energy_balance')
    assert math.isnan(replace(led, throughput=0.0).closure)  # no throughput: nan


def test_the_throughput_is_the_sum_of_the_absolute_booked_terms():
    """The throughput T sums |S|, |L_int|, |C_cmb|, |Q|, |I_mantle| and |B_core| over the steps, and the
    closure is sum R_p / T; with an impact and a booked core term both enter."""
    hf = _table('core_module')
    hf.loc[6, 'step_dE_impact_J'] = -2e25
    hf.loc[7, 'step_dE_core_J'] += 3e26
    hf.loc[7, 'step_dE_struct_core_J'] = 3e26
    led = whole_run_ledger(hf, 'core_module')
    cols = ['step_dE_state_heat_J', 'step_dE_F_int_J', 'step_dE_F_cmb_J', 'step_dE_Q_radio_J']
    hand = (
        hf[cols].iloc[3:].abs().sum().sum() + 2e25 + 3e26
    )  # plus the impact and booked core heat
    assert led.throughput == pytest.approx(hand, rel=1e-12)
    assert led.closure == pytest.approx(led.R_p.sum() / hand, rel=1e-12, abs=1e-30)
    stored = (
        (  # R_p from the stored columns: S - Q + D_core - B_core + E_atm on the start-row area
            hf['step_dE_state_heat_J']
            - hf['step_dE_Q_radio_J']
            + hf['step_dE_core_J']
            - hf['step_dE_struct_core_J']
        )
        .iloc[3:]
        .to_numpy()
        + _loss_to_space(hf)[2:]
    )
    np.testing.assert_allclose(led.R_p, stored, rtol=1e-12, atol=1e-9 * led.throughput)


@pytest.mark.parametrize(
    'tfac', [0.0, -1.0, math.nan, math.inf], ids=['zero', 'negative', 'nan', 'inf']
)
def test_a_core_temperature_factor_out_of_range_raises(tfac):
    """core_tfac must be finite and positive, as the config requires of core_tfac_avg."""
    hf = _table()
    before = hf.copy()
    with pytest.raises(ValueError, match='core_tfac'):
        whole_run_ledger(hf, 'energy_balance', core_tfac=tfac)
    assert hf.equals(before)


@pytest.mark.physics_invariant
def test_the_core_temperature_factor_enters_the_core_heat():
    """A run whose core heat was booked with tfac 1.3 closes only with core_tfac = 1.3; with the
    default 1.147 the core misses by 1 - 1.147/1.3 = 11.8 % of the CMB heat."""
    hf = _table(tfac=1.3)
    led = whole_run_ledger(hf, 'energy_balance', core_tfac=1.3)
    assert abs(led.R_c.sum()) <= 1e-12 * led.throughput
    off = whole_run_ledger(hf, 'energy_balance')
    C = led.terms['C_cmb'].sum()
    assert off.R_c.sum() / C == pytest.approx(1.0 - 1.147 / 1.3, rel=1e-9)


@settings(max_examples=40, deadline=None)
@given(rows=st.sets(st.integers(3, 9), max_size=5))
def test_the_class_split_of_the_core_residual_sums_to_its_total(rows):
    """Property: for any set of update rows, consecutive ones included, the update, remap and other
    parts of R_c add up to sum R_c, and the remap steps are those after an update step."""
    hf = _table()
    hf.loc[3:, 'T_cmb_node'] += np.linspace(
        0.0, 2.0, 7
    )  # an unbooked core change in every step
    for r in rows:
        hf.loc[r:, 'R_int'] += 1e3
    led = _closed_coupling(hf)
    assert sum(led.split('R_c').values()) == pytest.approx(led.R_c.sum(), rel=1e-12)
    assert led.classes['remap'][1:].tolist() == led.classes['update'][:-1].tolist()
    assert int(led.classes['update'].sum()) == len(rows)


def _closed_coupling(hf):
    """Ledger of ``hf`` after resetting F_atm so that each step loses L_int on its start-row area."""
    for j in range(2, len(hf) - 1):
        dt_j = (hf.loc[j + 1, 'Time'] - hf.loc[j, 'Time']) * secs_per_year
        hf.loc[j, 'F_atm'] = -hf.loc[j + 1, 'step_dE_F_int_J'] / (
            4.0 * np.pi * hf.loc[j, 'R_int'] ** 2 * dt_j
        )
    return whole_run_ledger(hf, 'energy_balance')


def test_a_change_of_the_core_mass_alone_is_a_structure_update():
    """A structure solve that changes only M_core still moves the step to class update; an impact
    step with negative impact heat is in class impact."""
    hf = _table()
    hf.loc[6:, 'M_core'] = 2.0e24 * (1 + 1e-4)
    hf['T_cmb_node'] = 4000.0 - np.cumsum(hf['step_dE_F_cmb_J'].to_numpy()) / (
        hf['M_core'] * 800.0 * 1.147
    )
    hf.loc[8, 'step_dE_impact_J'] = -1e25
    led = whole_run_ledger(hf, 'energy_balance')
    assert led.classes['update'].tolist() == [False, False, False, True, False, False, False]
    assert led.classes['impact'].tolist() == [False, False, False, False, False, True, False]


@pytest.mark.parametrize(
    'column', ['step_dE_state_heat_J', 'step_dE_F_int_J', 'step_dE_F_cmb_J']
)
def test_a_step_with_no_time_and_any_energy_raises(column):
    """A step with dt = 0 that books energy in any one of the mantle heat, the heat to the atmosphere
    or the CMB heat is refused."""
    hf = _table()
    for c in (
        'step_dE_state_heat_J',
        'step_dE_F_int_J',
        'step_dE_F_cmb_J',
        'step_dE_Q_radio_J',
    ):
        hf.loc[6, c] = 0.0
    hf.loc[6, column] = 1e24
    hf.loc[6, 'Time'] = hf.loc[5, 'Time']
    before = hf.copy()
    with pytest.raises(ValueError, match='Time'):
        whole_run_ledger(hf, 'energy_balance', end_row=6)
    assert hf.equals(before)


def _older_run():
    """A closed synthetic run at the times of a long run (near 5e4 yr, steps of 25 to 31 yr), with the
    2 whole-run cumulative columns of which E_residual_cons_J is the difference."""
    hf = _table()
    rng = np.random.default_rng(5)
    hf['Time'] = np.r_[0.0, 0.0, 0.0, 5.08e4 + np.cumsum(rng.uniform(25.0, 31.0, 7))]
    _closed_coupling(hf)
    cum = -8e31 + np.r_[0.0, 0.0, 0.0, np.cumsum(hf['step_dE_state_heat_J'].iloc[3:])]
    hf['E_state_heat_cons_J'] = cum
    hf['dE_predicted_cons_J'] = cum
    return hf


def _written(hf, rows=slice(None)):
    """``hf`` with the given rows rounded to 11 significant digits, as the older writer stored them."""
    out = hf.copy()
    out.iloc[rows] = out.iloc[rows].map(lambda v: float(f'{v:.10e}'))
    out['E_residual_cons_J'] = out['E_state_heat_cons_J'] - out['dE_predicted_cons_J']
    return out


def test_a_helpfile_written_with_11_digits_is_ledgered():
    """Older helpfiles hold 11 significant digits (format %.10e). At t near 5e4 yr and dt near 28 yr the
    rounding of 2 times alone moves dt by up to 4e-9 of itself, so the coupling identity and the stored
    residual hold only to that precision, and the ledger accepts them; a gap of 1e-5 of L_int is still
    refused, and a full-precision file with a gap of 1e-10 is refused."""
    hf = _older_run()
    written = _written(hf)
    led = whole_run_ledger(written, 'energy_balance')
    assert abs(led.G.sum()) <= 1e-7 * led.terms['L_int'].sum()
    assert np.abs(written['Time'] - hf['Time']).max() > 0  # the times are rounded
    gap = written.copy()
    gap.loc[6, 'F_atm'] *= 1 - 1e-5
    with pytest.raises(RuntimeError, match='coupling identity'):
        whole_run_ledger(gap, 'energy_balance')
    hf.loc[5, 'F_atm'] *= 1 + 1e-10
    with pytest.raises(RuntimeError, match='coupling identity'):
        whole_run_ledger(hf, 'energy_balance')


def test_a_helpfile_with_older_and_newer_rows_is_ledgered():
    """A run started with the 11-digit writer and resumed with the full-precision one holds both kinds of
    rows: each step is checked to the coarser precision of its 2 rows, and the ledger accepts it."""
    hf = _older_run()
    mixed = _written(hf, rows=slice(0, 6))
    led = whole_run_ledger(mixed, 'energy_balance')
    assert abs(led.R_m.sum()) <= 1e-9 * led.terms['L_int'].sum()
    assert mixed.loc[8, 'F_atm'] == hf.loc[8, 'F_atm']  # the later rows keep full precision


def _loss_to_space(hf):
    """F_atm A dt of each step on the area of its start row, from the stored columns [J]."""
    dt = np.diff(hf['Time'].to_numpy()) * secs_per_year
    return (hf['F_atm'] * 4.0 * np.pi * hf['R_int'] ** 2).to_numpy()[:-1] * dt
