"""The lid seal on the stored rows of its tutorial run.

Invariants checked: replayed on the stored closing rows with escape off, the
leak starts for each species at the step the floor predicts; replayed over the
whole held span with the stored escape and melt fraction, the code gives the
held water, its floor and the atmospheric water of the run on every row, held
water that leaked ends at or above its floor, and the water in the atmosphere
lies between the run that holds it and the open run.
The data are rows of the tutorial runs that their header names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proteus.outgas.lid_seal import leak_step
from proteus.outgas.wrapper import run_crystallized

from .test_lid_seal import D_H2O, _assert_legend_clear, _config, _eq_row

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

DATA = Path(__file__).parents[1] / 'data' / 'lid_seal'


def _tutorial_config():
    """The seal and CALLIOPE settings of the tutorial (tau_leak 1e5 yr, crust_rate 0)."""
    return _config(solubility=True, tau_leak=1.0e5)


def _closing_row():
    """The stored closing row as a sealed helpfile row, with the frozen solve temperature."""
    rows = pd.read_csv(DATA / 'tutorial_closing_rows.csv', comment='#')
    row = {**_eq_row({}, {}), **rows.iloc[0].to_dict()}
    row['seal_Tmagma_close'] = max(row['T_magma'], _tutorial_config().outgas.T_floor)
    return row, rows


@pytest.mark.physics_invariant
def test_the_first_held_steps_of_the_tutorial_follow_the_floor():
    """On the stored closing row the held water is the dissolved water of the
    melt, below its floor on melt plus crystals by 1 + D (1 - Phi)/Phi. Replayed
    with escape off over the stored melt fractions, CO2, CO and CH4 leak from the
    first held step, and water only from the first step with Phi below
    (Phi_c - D)/(1 - D), where the floor falls under it."""
    row, rows = _closing_row()
    phi_c = row['Phi_global']
    phi_star = (phi_c - D_H2O) / (1.0 - D_H2O)
    cfg, first = _tutorial_config(), {}
    for nxt in rows.iloc[1:].itertuples():
        dt = nxt.Time - row['Time']
        row.update({'Time': nxt.Time, 'Phi_global': nxt.Phi_global, 'M_mantle': nxt.M_mantle})
        before = {s: row[f'{s}_kg_sealed'] for s in ('H2O', 'CO2', 'CO', 'CH4')}
        leak_step(cfg, row, dt)
        for s, m in before.items():
            if row[f'{s}_kg_sealed'] < m:
                first.setdefault(s, int(nxt.row))
        if 'H2O' not in first:
            assert row['H2O_kg_eq'] > row['H2O_kg_sealed']
    k = int(rows['row'].iloc[0])
    h2o = int(rows['row'][rows['Phi_global'] < phi_star].iloc[0])
    assert first == {'CO2': k + 1, 'CO': k + 1, 'CH4': k + 1, 'H2O': h2o}
    assert h2o > k + 1


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_tutorial_atmosphere_lies_between_the_held_and_the_open_run(validation_plot):
    """The held span of the stored tutorial run (tau_leak 1e5 yr), replayed from
    its closing row through run_crystallized and leak_step with the stored escape
    and melt fraction of each step, gives the held water, its floor H2O_kg_eq and
    the water in the atmosphere of the run on every row (relative 1e-12). Every
    row on which the held water fell ends at or above its floor. From the closing
    row on, the water in the atmosphere is at or above the run that holds it
    (tau_leak inf) and 1.5 times it at the end; from the first held step on, it
    is below the open run."""
    w = pd.read_csv(DATA / 'tutorial_water.csv', comment='#')
    held = w[w['seal_active'] > 0.5]
    row, _ = _closing_row()
    assert row['Time'] == held['Time'].iloc[0]
    cfg, out = _tutorial_config(), []
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
    assert np.all(held['seal_atm'].iloc[1:] < held['open_atm'].iloc[1:])
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
