"""The lid seal on the stored rows of its tutorial run.

Invariants checked: replayed on the stored closing rows with escape off, the
leak starts for each species at the step the floor predicts; replayed over the
whole held span with the stored escape and melt fraction, the code gives the
held water, its floor and the atmospheric water of the run on every row, held
water that leaked ends at or above its floor, and the water in the atmosphere
lies above the run that holds it and crosses the open run once; the seal effect of the
printed run and of 5 round-off starts lies in the band the tutorial quotes.
The data are rows of the tutorial runs that their header names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proteus.outgas.lid_seal import leak_step
from proteus.outgas.wrapper import run_crystallized

from .test_lid_seal import _assert_legend_clear, _config, _eq_row

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

DATA = Path(__file__).parents[1] / 'data' / 'lid_seal'
CFG = _config(solubility=True, tau_leak=1.0e5)  # the tutorial settings, crust_rate 0


def _closing_row():
    """The stored closing row as a sealed helpfile row, with the frozen solve temperature."""
    rows = pd.read_csv(DATA / 'tutorial_closing_rows.csv', comment='#')
    row = {**_eq_row({}, {}), **rows.iloc[0].to_dict()}
    row['seal_Tmagma_close'] = max(row['T_magma'], CFG.outgas.T_floor)
    return row, rows


@pytest.mark.physics_invariant
def test_the_closing_row_of_the_tutorial_sits_on_its_floor():
    """Without trapping the floor reservoir is the melt, so on the stored closing
    row each held species is the dissolved mass of the closing solve and equals
    its floor (relative 1e-15): closure is a fixed point of the leak. Replayed
    with escape off over the stored melt fractions, H2O, CO2, CO and CH4 leak
    from the first held step, as the falling melt fraction lowers each floor."""
    row, rows = _closing_row()
    at_close = dict(row)
    leak_step(CFG, at_close, 1.0e-9)
    for s in ('H2O', 'CO2', 'CO', 'CH4'):
        assert at_close[f'{s}_kg_eq'] == pytest.approx(row[f'{s}_kg_sealed'], rel=1e-15, abs=0)
    first = {}
    for nxt in rows.iloc[1:].itertuples():
        dt = nxt.Time - row['Time']
        row.update({'Time': nxt.Time, 'Phi_global': nxt.Phi_global, 'M_mantle': nxt.M_mantle})
        before = {s: row[f'{s}_kg_sealed'] for s in ('H2O', 'CO2', 'CO', 'CH4')}
        leak_step(CFG, row, dt)
        for s, m in before.items():
            if row[f'{s}_kg_sealed'] < m:
                first.setdefault(s, int(nxt.row))
    assert first == dict.fromkeys(('H2O', 'CO2', 'CO', 'CH4'), int(rows['row'].iloc[0]) + 1)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_tutorial_atmosphere_lies_above_the_held_run_and_crosses_the_open_run(
    validation_plot,
):
    """The held span of the stored tutorial run (tau_leak 1e5 yr), replayed from
    its closing row through run_crystallized and leak_step with the stored escape
    and melt fraction of each step, gives the held water, its floor H2O_kg_eq and
    the water in the atmosphere of the run on every row (relative 1e-12). Every
    row on which the held water fell ends at or above its floor. From the closing
    row on, the water in the atmosphere is at or above the run that holds it
    (tau_leak inf) and 1.5 times it at the end. Against the open run it is lower
    on the first held steps, while the leak lags its falling floor, crosses it
    once, and ends more than 1 % above it."""
    w = pd.read_csv(DATA / 'tutorial_water.csv', comment='#')
    held = w[w['seal_active'] > 0.5]
    row, _ = _closing_row()
    assert row['Time'] == held['Time'].iloc[0]
    cfg, out = CFG, []
    for r in held.iloc[1:].itertuples():
        dt = r.Time - row['Time']
        row.update({'Time': r.Time, 'Phi_global': r.Phi_global, 'M_mantle': r.M_mantle})
        row.update({'esc_step_kg': r.esc_step_kg, 'esc_rate_total': r.esc_rate_total})
        run_crystallized(cfg, row, dt)
        leak_step(cfg, row, dt)
        out.append((row['H2O_kg_sealed'], row['H2O_kg_eq'], row['H2O_kg_atm']))
    stored = held[['seal_held', 'seal_eq', 'seal_atm']].values[1:]
    np.testing.assert_allclose(np.array(out), stored, rtol=1e-12, atol=0.0)
    fell = held['seal_held'].diff() < 0.0
    assert fell.sum() > 0.5 * len(held)
    assert np.all(held['seal_held'][fell] >= held['seal_eq'][fell] * (1.0 - 1e-12))
    assert np.all(held['seal_atm'] >= held['hold_atm'])
    gap = (held['seal_atm'] / held['open_atm']).values[1:] - 1.0
    k = int(np.argmax(gap > 0.0))
    assert k > 0 and np.all(gap[:k] < 0.0) and np.all(gap[k:] > 0.0)
    assert gap[-1] > 0.01
    assert held['seal_atm'].iloc[-1] > 1.5 * held['hold_atm'].iloc[-1]

    def figure():
        import matplotlib.pyplot as plt

        plt.rcParams['svg.fonttype'] = 'none'  # text as SVG text, not glyph paths
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)
        t = w['Time'] / 1e3
        ax.plot(t, w['open_atm'] / 1e15, label='open run')
        ax.plot(t, w['seal_atm'] / 1e15, label=r'seal, $\tau$ = 100 kyr')
        ax.plot(t, w['hold_atm'] / 1e15, '--', label=r'seal, $\tau$ infinite')
        ax.set_ylabel('H$_2$O in the atmosphere\n[$10^{15}$ kg]')
        ax.legend(loc='upper left')
        on = held[held['seal_eq'] > 0.0]
        gap = (on['seal_held'] - on['seal_eq']) / on['seal_held']
        ax2.plot(on['Time'] / 1e3, gap, color=ax.lines[1].get_color())
        ax2.axhline(0.0, ls=':', color=plt.rcParams['text.color'], alpha=0.6)
        ax2.set_ylabel('Held H$_2$O above its floor\n[fraction of held]')
        ax2.set_xlabel('Time [kyr]')
        fig.tight_layout()
        _assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'outgas_lid_seal_tutorial_water')


def test_the_seal_effect_of_the_tutorial_lies_in_one_band_over_round_off_starts():
    """The stored end values of the printed run and of 5 starts whose initial
    superheat differs by up to 2.5e-9 K, each with the seal on and off: the seal
    run holds 0.428 % to 0.438 % less water in the atmosphere at 140 kyr than the
    open run of its start, 0.44 % in the printed run; the water of the 5 starts
    spreads by 0.64 % and the printed run ends 2.0 % below their mean."""
    d = pd.read_csv(DATA / 'tutorial_ensemble.csv', comment='#')
    less = 100.0 * (1.0 - d['seal_atm'] / d['open_atm'])
    printed, starts = d['run'] == 'printed', d[d['run'] != 'printed']
    assert len(starts) == 5 and starts['dT'].max() < 2.5e-9 * (1 + 1e-6)
    assert (round(less.min(), 3), round(less.max(), 3), round(less[printed].item(), 2)) == (
        0.428,
        0.438,
        0.44,
    )
    assert round(100.0 * np.ptp(starts['seal_atm']) / starts['seal_atm'].mean(), 2) == 0.64
    below = 1.0 - d.loc[printed, 'seal_atm'].item() / starts['seal_atm'].mean()
    assert round(100.0 * below, 1) == 2.0
