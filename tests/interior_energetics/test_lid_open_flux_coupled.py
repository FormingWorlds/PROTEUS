"""Validation of the in-step open part of the lid from stored coupled runs.

Invariant: in a 1 Earth-mass run through the lid onset, the step out of the live-lid row with
the open part following its node within the interior step books an interior loss inside the
atmosphere flux band of the step, passes with no clip of T_magma, and moves the surface
temperature less and keeps the magma cooler than the same step with the response off. The rows
are in ``tests/data/lid_open_flux/coupled_rows.json``; see
``docs/Validation/atmos_clim/common.md``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

DATA = Path(__file__).parents[1] / 'data' / 'lid_open_flux' / 'coupled_rows.json'
PAGE = Path(__file__).parents[2] / 'docs' / 'Validation' / 'atmos_clim' / 'common.md'
RUNS = json.loads(DATA.read_text())
PAIRS = (('seg1_response', 'seg1_off'), ('seg2_response', 'seg2_off'))
NAMES = tuple(n for pair in PAIRS for n in pair)
START = ('Time', 'T_surf', 'F_int', 'T_magma', 'lid_weight', 'E_residual_cons_frac')
LIVE = 1  # index of the live-lid row; the response acts on the step from it to row LIVE + 1
# Quadrature floor of the band check, relative to |F_atm| at the start of the step.
BAND_FLOOR = 2.4e-4
# The response step moves T_surf by at most this fraction of the step with the response off
# (0.59 and 0.61 in the stored pairs).
SWING_RATIO = 2.0 / 3.0
# Sanity bound on the energy residual: the largest of the 4 stored runs is 7.0e-6.
E_RES_BOUND = 1.0e-5


def _step(run, key):
    """Change of ``key`` over the step out of the live-lid row of ``run``."""
    return RUNS[run][key][LIVE + 1] - RUNS[run][key][LIVE]


def _after(run, key):
    """``key`` at the end of the step out of the live-lid row of ``run``."""
    return RUNS[run][key][LIVE + 1]


def _band(run):
    """Atmosphere flux band of the step out of the live-lid row, widened by the floor."""
    a, b = RUNS[run]['F_atm'][LIVE], RUNS[run]['F_atm'][LIVE + 1]
    pad = BAND_FLOOR * abs(a)
    return min(a, b) - pad, max(a, b) + pad


def _facts():
    """Phrases of the page with the numbers of the stored rows, each formatted as on the page."""
    e_max = max(abs(x) for n in NAMES for x in RUNS[n]['E_residual_cons_frac'])
    exp = int(np.floor(np.log10(e_max)))
    t = {
        s: [RUNS[f'{s}_response']['Time'][k] / 1e6 for k in (LIVE, LIVE + 1)]
        for s in ('seg1', 'seg2')
    }
    wm2 = 'W m$^{-2}$'
    facts = []
    for s in ('seg1', 'seg2'):
        r, o = f'{s}_response', f'{s}_off'
        lo, hi = RUNS[r]['F_atm'][LIVE], RUNS[r]['F_atm'][LIVE + 1]
        facts += [
            f'from {t[s][0]:.3f} to {t[s][1]:.3f} Myr',
            f'({_step(r, "T_surf"):.1f} K against {_step(o, "T_surf"):.1f} K)',
            f'({_after(r, "T_magma"):.0f} K against {_after(o, "T_magma"):.0f} K)',
            f'{_after(r, "F_int"):.2f} {wm2} in [{lo:.2f}, {hi:.2f}] {wm2}',
        ]
    return [*facts, f'${e_max / 10**exp:.1f} \\times 10^{{{exp}}}$']


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_open_part_carries_the_coupled_runs_through_the_live_lid_step(validation_plot):
    """The response acts on the step out of the live-lid row; each pair agrees up to that row.
    With the response, that step books an interior loss inside the atmosphere flux band of the
    step, moves T_surf by less than 2/3 of the step with the response off, and ends with
    cooler magma. No run clips T_magma. Later rows differ through the path of each run and are
    not credited to the response."""
    for a, b in PAIRS:
        assert all(RUNS[a][k][: LIVE + 1] == RUNS[b][k][: LIVE + 1] for k in START)
        assert RUNS[a]['lid_weight'][LIVE - 1] == 0.0 < RUNS[a]['lid_weight'][LIVE]
        lo, hi = _band(a)
        assert lo <= _after(a, 'F_int') <= hi
        assert abs(_step(a, 'T_surf')) < SWING_RATIO * abs(_step(b, 'T_surf'))
        assert _after(a, 'T_magma') < _after(b, 'T_magma')
    for name in NAMES:
        assert sum(RUNS[name]['T_magma_clipped']) == 0.0
        assert max(RUNS[name]['T_magma']) < 1600.0
        assert max(abs(x) for x in RUNS[name]['E_residual_cons_frac']) <= E_RES_BOUND

    def figure():
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.6))
        role = {'off': ('response off', ':', 'C0'), 'response': ('response', '-', 'C2')}
        for ax, names in zip(axes, PAIRS, strict=True):
            for name in names:
                label, style, color = role[name.split('_', 1)[1]]
                run = RUNS[name]
                t = np.asarray(run['Time']) / 1e6
                ax.plot(t, run['T_surf'], style, color=color, marker='.', ms=4, label=label)
            t_live = RUNS[names[0]]['Time'][LIVE] / 1e6
            ax.axvline(t_live, color='0.5', lw=0.8, ls='--', label='live-lid row')
            ax.set(ylabel='T_surf [K]')
            ax.legend(fontsize=8, loc='upper right')
            # 4 ticks per axis keep the pair well under the size cap.
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(MaxNLocator(4))
        axes[1].set(xlabel='Time [Myr]')
        fig.tight_layout()
        return fig

    validation_plot(figure, 'atmos_clim_lid_open_coupled')


@pytest.mark.physics_invariant
def test_the_page_states_the_numbers_of_the_stored_coupled_rows():
    """Each number of the coupled case on the Validation page, in its phrase, comes from the
    stored rows."""
    text = PAGE.read_text()
    assert text.count('## Coupled runs through the lid onset') == 1
    section = text.split('## Coupled runs through the lid onset', 1)[1].split('\n## ', 1)[0]
    missing = [f for f in _facts() if not re.search(r'(?<![\d.])' + re.escape(f), section)]
    assert missing == []
