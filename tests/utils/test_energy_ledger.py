"""Tests of ``proteus.utils.energy_ledger.whole_run_ledger``: the whole-run energy ledger of a helpfile.

Invariants checked: a synthetic run whose booked terms close returns a planet residual of 0 to round-off;
an injected residual comes back with its size; every term enters with its size and sign (a ledger without
it moves the closure by that size); the coupling identity G = 0 holds on every no-lid step and its failure
raises; the mantle residual equals the stored cumulative column; bad input raises and returns no number.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from proteus.utils.constants import secs_per_year
from proteus.utils.energy_ledger import whole_run_ledger

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

C_CORE = 2.0e24 * 800.0 * 1.147  # M_core core_heatcap tfac [J/K]


def _table(core_bc='energy_balance', n_steps=7, seed=3):
    """Helpfile of 3 start rows at t = 0 and ``n_steps`` steps whose booked terms close exactly.

    Per step: mantle heat S = -L_int + C_cmb + Q; the loss to space F_atm(start) A(start) dt equals L_int;
    the core change is -C_cmb (energy_balance through T_cmb_node, core_module through step_dE_core_J).
    """
    rng = np.random.default_rng(seed)
    n = 3 + n_steps
    t = np.r_[0.0, 0.0, 0.0, np.cumsum(rng.uniform(5.0, 50.0, n_steps))]
    R_int = np.full(n, 6.4e6)
    A = 4.0 * np.pi * R_int**2
    L = np.r_[0.0, 0.0, 0.0, rng.uniform(1e27, 3e27, n_steps)]
    C = np.r_[0.0, 0.0, 0.0, rng.uniform(1e26, 4e26, n_steps)]
    Q = np.r_[0.0, 0.0, 0.0, rng.uniform(1e24, 5e24, n_steps)]
    F_atm = np.zeros(n)
    dt = np.diff(t) * secs_per_year
    F_atm[2:-1] = L[3:] / (A[2:-1] * dt[2:])
    d = {
        'Time': t,
        'R_int': R_int,
        'F_atm': F_atm,
        'step_dE_state_heat_J': -L + C + Q,
        'step_dE_F_int_J': -L,
        'step_dE_F_cmb_J': C,
        'step_dE_Q_radio_J': Q,
        'step_dE_Q_tidal_J': np.zeros(n),
        'step_dE_impact_J': np.zeros(n),
        'M_core': np.full(n, 2.0e24),
        'R_core': np.full(n, 3.5e6),
        'M_mantle': np.full(n, 4.0e24),
        'core_heatcap': np.full(n, 800.0),
        'lid_weight': np.zeros(n),
        'step_dE_compression_J': np.zeros(n),
    }
    if core_bc == 'core_module':
        d |= {
            'step_dE_core_J': -C,
            'step_dE_impact_core_J': np.zeros(n),
            'step_dE_impact_core_refit_J': np.zeros(n),
            'step_dE_struct_core_J': np.zeros(n),
            'T_cmb_node': np.full(n, 4000.0),
        }
    else:
        d['T_cmb_node'] = 4000.0 - np.cumsum(C) / C_CORE
    d['E_residual_cons_J'] = np.zeros(n)
    return pd.DataFrame(d)


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
    with pytest.raises(ValueError, match=match):
        whole_run_ledger(_table(), 'energy_balance', **kwargs)
    assert len(_table()) == 10


@pytest.mark.parametrize('column', ['step_dE_F_int_J', 'F_atm', 'T_cmb_node', 'M_mantle'])
def test_a_missing_column_raises_and_names_it(column):
    """An older helpfile without a required column raises ValueError naming the column."""
    hf = _table().drop(columns=[column])
    with pytest.raises(ValueError, match=column):
        whole_run_ledger(hf, 'energy_balance')
    assert column not in hf.columns


def test_core_module_needs_its_core_columns():
    """core_module reads step_dE_core_J; a table without it raises."""
    hf = _table('core_module').drop(columns=['step_dE_core_J'])
    with pytest.raises(ValueError, match='step_dE_core_J'):
        whole_run_ledger(hf, 'core_module')
    assert 'step_dE_struct_core_J' in hf.columns


@pytest.mark.parametrize('core_bc', ['quasi_steady', 'gradient', '', 'Energy_Balance'])
def test_an_unknown_core_bc_raises(core_bc):
    """Only energy_balance, bower2018 and core_module have a core ledger here; others raise."""
    with pytest.raises(ValueError, match='core_bc'):
        whole_run_ledger(_table(), core_bc)
    assert core_bc not in ('energy_balance', 'bower2018', 'core_module')


@pytest.mark.parametrize(
    'bad', [math.nan, math.inf, -math.inf], ids=['nan', 'inf', 'minus-inf']
)
def test_a_non_finite_term_raises(bad):
    """NaN or inf in a summed term raises ValueError naming the column and the row."""
    hf = _table()
    hf.loc[6, 'step_dE_F_cmb_J'] = bad
    with pytest.raises(ValueError, match=r'step_dE_F_cmb_J.*row 6'):
        whole_run_ledger(hf, 'energy_balance')
    assert not np.isfinite(hf.loc[6, 'step_dE_F_cmb_J'])


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
    value=st.one_of(
        st.just(math.nan),
        st.just(math.inf),
        st.just(-math.inf),
        st.booleans(),
        st.text(max_size=3),
    )
)
def test_any_bad_value_in_a_term_raises(value):
    """Property: NaN, inf, a boolean or a string in a summed column raises ValueError, never a number."""
    hf = _table().astype({'step_dE_Q_radio_J': object})
    hf.loc[7, 'step_dE_Q_radio_J'] = value
    with pytest.raises(ValueError):
        whole_run_ledger(hf, 'energy_balance')
    assert hf['step_dE_Q_radio_J'].dtype == object
