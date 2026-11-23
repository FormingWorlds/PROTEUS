"""Validation of ``proteus.utils.energy_ledger.whole_run_ledger`` on stored rows of coupled runs.

The rows are in ``tests/data/energy_ledger/`` (``build_cases.py`` there writes them from the source runs and
records the source sha256 and the code versions). Cases: the lid onset of a 1 Earth-mass run (the mantle
residual and the coupling term, reproduced from the earlier measurement of that window), the core ledger
of core_module across 3 structure updates (closed to 1e-7 of the throughput) and the unbooked core heat
jump of energy_balance at the same updates (the remap class). Residuals are relative to the heat to the
atmosphere L_int of the window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proteus.utils.constants import secs_per_year
from proteus.utils.energy_ledger import whole_run_ledger

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

DATA = Path(__file__).parents[1] / 'data' / 'energy_ledger'


def _case(name, start_row=0):
    """Ledger, stored table and provenance of a stored case."""
    d = json.loads((DATA / f'{name}.json').read_text())
    hf = pd.DataFrame(d['columns'])
    s = d['settings']
    return (
        whole_run_ledger(hf, s['core_bc'], s['core_tfac'], start_row=start_row),
        hf,
        d['provenance'],
    )


def _cumulative_figure(led, series, xlabel, ticks):
    """Cumulative series against time, each over L_int of the window, legend below the axes."""
    import matplotlib.pyplot as plt

    L = led.terms['L_int'].sum()
    t = led.time - led.time[0]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for label, x in series:
        ax.plot(t, np.cumsum(x) / L, lw=2, label=label)
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.set_yticks(ticks, [f'{v:g}' for v in ticks])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(xlabel)
    ax.set_ylabel('cumulative / L_int')
    fig.legend(loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(
        rect=(0.0, 0.06 + 0.055 * -(-len(series) // 2), 1.0, 1.0)
    )  # room per legend row
    return fig


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_lid_onset_window_reproduces_its_mantle_and_coupling_terms(validation_plot):
    """Lid onset of a 1 Earth-mass run (25 steps, all with a live lid, the first after a structure
    update at the start row): the mantle residual is -1.646e-6 and the coupling term G -5.698e-2 of
    L_int, as a separate script on the same helpfile columns gives; the loss to space on the end-row area
    instead of the start-row area would move G outside the tolerance. The core heat jump of the update,
    which energy_balance books nowhere, is 1.3557 of L_int and sits in the step after the update."""
    led, hf, prov = _case('r2rel_onset', start_row=1)
    hf = hf.iloc[1:]
    L = led.terms['L_int'].sum()
    assert led.R_m.sum() / L == pytest.approx(-1.646e-6, rel=1e-3)
    assert led.G.sum() / L == pytest.approx(-5.698e-2, rel=1e-3)
    assert led.classes['lid'].all() and led.R_p.size == 25
    A = 4.0 * np.pi * hf['R_int'].to_numpy() ** 2
    dt = np.diff(hf['Time'].to_numpy()) * secs_per_year
    G_end = L - float(np.sum(hf['F_atm'].to_numpy()[:-1] * A[1:] * dt))
    assert abs(G_end / L + 5.698e-2) > 1e-3 * 5.698e-2  # the area of the end row misses
    assert prov['rows'] == [872, 898]  # row 872 classes the first step of the window
    split = led.split('R_c')
    assert led.classes['remap'][0]
    assert split['remap'] / L == pytest.approx(1.3557209386827422, rel=1e-9)
    assert split['other'] / L == pytest.approx(-9.932024803468492e-4, rel=1e-6)

    def figure():
        return _cumulative_figure(
            led,
            [
                ('R_m', led.R_m),
                ('G', led.G),
                ('R_c, other', np.where(led.classes['remap'], 0.0, led.R_c)),
                ('R_c, after an update', np.where(led.classes['remap'], led.R_c, 0.0)),
            ],
            'time in the window [yr]',
            [-0.1, -1e-3, 0.0, 1e-3, 1.0],
        )

    validation_plot(figure, 'energy_ledger_lid_onset')


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_core_module_books_the_core_heat_of_each_structure_update():
    """core_module books the core heat change of each structure update (step_dE_struct_core_J): across
    3 updates in 450 yr the core residual is within 1e-7 of the throughput in every step class; without
    the booked term it would be the booked sum, 6.873e-3 of L_int."""
    led, _, _ = _case('cm_updates', start_row=2)
    T = led.throughput
    assert abs(led.R_c.sum()) <= 1e-7 * T  # measured 6e-9 of T
    assert all(abs(v) <= 1e-7 * T for v in led.split('R_c').values())
    B = led.terms['B_core'].sum()
    assert B / led.terms['L_int'].sum() == pytest.approx(6.872932228745487e-3, rel=1e-9)
    assert int(led.classes['update'].sum()) == 3


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_energy_balance_leaves_the_core_heat_jump_of_an_update_in_the_remap_class(
    validation_plot,
):
    """energy_balance books no core heat at a structure update, so the jump stays in R_c of the steps
    after an update: 5.948278897938286e-3 of L_int over 3 updates in 450 yr, the value of a separate
    measurement script on the same rows (same arithmetic, 1e-9). The other steps hold 1e-2 of
    it, and the planet residual R_p = R_m + R_c - G is 8.526e-3 of L_int."""
    led, hf, _ = _case('eb_updates', start_row=2)
    L = led.terms['L_int'].sum()
    split = led.split('R_c')
    assert split['remap'] / L == pytest.approx(5.948278897938286e-3, rel=1e-9)
    assert abs(split['other'] + split['update']) < 1e-2 * split['remap']
    assert led.terms['B_core'].sum() == 0.0
    assert led.R_p.sum() / L == pytest.approx(8.525868397324384e-3, rel=1e-9)
    cm, _, _ = _case('cm_updates', start_row=2)  # same setup with core_module, for the plot
    dt = np.diff(hf['Time'].to_numpy()) * secs_per_year
    E_atm = (hf['F_atm'] * 4.0 * np.pi * hf['R_int'] ** 2).to_numpy()[:-1] * dt
    D = (hf['M_core'] * hf['core_heatcap'] * 1.147).to_numpy()[1:] * np.diff(
        hf['T_cmb_node'].to_numpy()
    )
    S_minus_Q = (
        hf['step_dE_state_heat_J'] - hf['step_dE_Q_radio_J'] - hf['step_dE_Q_tidal_J']
    ).to_numpy()
    stored = (
        S_minus_Q[1:] + D + E_atm
    )  # R_p from the stored columns (nothing booked for the core)
    np.testing.assert_allclose(led.R_p, stored[2:], rtol=1e-9, atol=1e-12 * led.throughput)

    def figure():
        return _cumulative_figure(
            led,
            [
                ('R_c, after an update', np.where(led.classes['remap'], led.R_c, 0.0)),
                ('R_c, other', np.where(led.classes['remap'], 0.0, led.R_c)),
                ('R_c, core_module', cm.R_c * L / cm.terms['L_int'].sum()),
                ('G', led.G),
                ('R_m', led.R_m),
            ],
            'time [yr]',
            [-1e-3, -1e-5, 0.0, 1e-5, 1e-3],
        )

    validation_plot(figure, 'energy_ledger_core_remap')


# steps whose CMB node is in the mush, from the node entropy of the snapshots (the helpfile has no
# melt fraction of the node); the step ending at 1577 yr starts in a flat table cell at the liquidus
MUSH_YR = (1787.0, 14101.0)


def _core_miss(name):
    """Per-step core residual over the CMB heat, and the steps of class other in the mush window."""
    led, _, _ = _case(name, start_row=2)
    C = np.abs(led.terms['C_cmb'])
    x = np.divide(led.R_c, C, out=np.full(C.size, np.nan), where=C > 0)
    c = led.classes
    other = ~c['update'] & ~c['remap'] & ~c['zero']
    return led, x, other & (led.time >= MUSH_YR[0]) & (led.time <= MUSH_YR[1])


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_slope_closure_of_the_cmb_node_closes_the_core_ledger_in_the_mush(validation_plot):
    """The same 1 Earth-mass PALEOS run with the c_p/T closure of the CMB node (aragog 3a869bb1) and with
    the entropy slope of the phase blend (aragog fbf15478, https://github.com/FormingWorlds/aragog/issues/115):
    per step R_c/|C_cmb|, positive when the core loses less heat than crosses the CMB. In the mush the
    c_p/T run misses by a median of 0.091 +- 0.01 and the slope run by at most 1e-3, against the core
    balance of Bower et al. (2018, Eqs. 19-21, https://doi.org/10.1016/j.pepi.2017.11.004)."""
    led_b, x_b, mush_b = _core_miss('a115_base')
    led_s, x_s, mush_s = _core_miss('a115_branch')
    assert np.median(x_b[mush_b]) == pytest.approx(0.091, abs=0.01)
    assert np.max(np.abs(x_s[mush_s])) <= 1e-3
    assert mush_b.sum() >= 10 and mush_s.sum() >= 10
    assert np.min(x_b[mush_b]) > 50 * np.max(
        np.abs(x_s[mush_s])
    )  # the 2 closures are told apart

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        for led, x, label in ((led_b, x_b, 'c_p/T'), (led_s, x_s, 'slope')):
            c = led.classes
            keep = ~c['update'] & ~c['remap'] & ~c['zero']
            ax.plot(led.time[keep], x[keep], lw=1.5, label=label)
        ax.axvspan(*MUSH_YR, alpha=0.08, color='C2')
        ax.set_xscale('log')
        ax.set_yscale('symlog', linthresh=1e-4)
        ticks = [-0.1, -1e-3, 0.0, 1e-3, 0.1]
        ax.set_yticks(ticks, [f'{v:g}' for v in ticks])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel('time [yr]')
        ax.set_ylabel('R_c / |C_cmb|')
        fig.legend(loc='lower center', ncol=2, frameon=False)
        fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
        return fig

    validation_plot(figure, 'energy_ledger_cmb_closure')
