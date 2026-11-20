"""Validation of the in-step open part of the lid from stored coupled runs.

Invariant: in a 1 Earth-mass run through the lid onset, the step out of the live-lid row with
the open part following its node within the interior step (and the node held at least at the
lid base) passes with no clip of T_magma, where the same row with the response off stops at a
T_magma clip (segment 1), and moves the surface temperature less and keeps the magma cooler
than the same step with the response off (segment 2). The rows are in
``tests/data/lid_open_flux/coupled_rows.json``; see ``docs/Validation/atmos_clim/common.md``.
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
STOP = RUNS['seg1_off_stop']
PAIRS = (('seg1_response', 'seg1_off'), ('seg2_response', 'seg2_off'))
NAMES = tuple(n for pair in PAIRS for n in pair)
START = ('Time', 'T_surf', 'F_int', 'T_magma', 'lid_weight', 'E_residual_cons_frac')
# Sanity bound on the energy residual: the largest of the 4 stored runs is 7.0e-6.
E_RES_BOUND = 1.0e-5


def _live(run):
    return [i for i, w in enumerate(RUNS[run]['lid_weight']) if w > 0.0]


def _step(run, key):
    """Change of ``key`` over the step out of the live-lid row of ``run``."""
    (k,) = _live(run)
    return RUNS[run][key][k + 1] - RUNS[run][key][k]


def _after(run, key):
    """``key`` at the end of the step out of the live-lid row of ``run``."""
    (k,) = _live(run)
    return RUNS[run][key][k + 1]


def _facts():
    """Phrases of the page with the numbers of the stored rows, each formatted as on the page."""
    e_max = max(abs(x) for n in NAMES for x in RUNS[n]['E_residual_cons_frac'])
    exp = int(np.floor(np.log10(e_max)))
    t1, t2 = (RUNS['seg2_response']['Time'][k] / 1e6 for k in (1, 2))
    wm2 = 'W m$^{-2}$'
    return [
        f'({_step("seg2_response", "T_surf"):.1f} K against {_step("seg2_off", "T_surf"):.1f} K)',
        f'({_after("seg2_response", "T_magma"):.0f} K against {_after("seg2_off", "T_magma"):.0f} K)',
        f'from {t1:.3f} to {t2:.3f} Myr',
        f'clip from {STOP["T_magma_from"]:.0f} K',
        f'at {STOP["step_start"] / 1e6:.2f} Myr',
        f'the magma at {_after("seg1_response", "T_magma"):.0f} K',
        f'{_after("seg2_response", "lid_F_diff"):.2f} {wm2} against '
        f'${_after("seg2_off", "lid_F_diff"):.2f}$ {wm2}',
        f'{_after("seg1_response", "lid_F_diff"):.2f} {wm2} in the first',
        f'${e_max / 10**exp:.1f} \\times 10^{{{exp}}}$',
    ]


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_open_part_carries_the_coupled_runs_through_the_live_lid_step(validation_plot):
    """The response acts on the one step out of the live-lid row; each pair agrees up to that
    row. Segment 1: the run without the response stops in that step at a T_magma clip, the run
    with it passes with no clip. Segment 2: with the response the step moves T_surf less than
    half as much and ends with cooler magma. Later rows differ through the path of each run and
    are not credited to the response."""
    for a, b in PAIRS:
        assert all(RUNS[a][k][:2] == RUNS[b][k][:2] for k in START)
        assert _live(a) == _live(b) == [1]
    assert len(RUNS['seg1_off']['Time']) == 2
    assert STOP['status'] == 21 and STOP['step_start'] == RUNS['seg1_off']['Time'][1]
    assert _after('seg1_response', 'T_magma') < 1600.0 < STOP['T_magma_from']
    # Ratio 0.163 on the stored pair; 5 round-off kicked starts of the response run spread by 4.5e-6.
    assert abs(_step('seg2_response', 'T_surf')) < 0.5 * abs(_step('seg2_off', 'T_surf'))
    assert _after('seg2_response', 'T_magma') < _after('seg2_off', 'T_magma')
    for name in ('seg1_response', 'seg2_response'):
        assert sum(RUNS[name]['T_magma_clipped']) == 0.0
        assert max(RUNS[name]['T_magma']) < 1600.0
    for name in NAMES:
        assert max(abs(x) for x in RUNS[name]['E_residual_cons_frac']) <= E_RES_BOUND

    def figure():
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.6))
        role = {
            'seg1_off': ('response off', ':', 'C0'),
            'seg2_off': ('response off', ':', 'C0'),
            'seg1_response': ('response', '-', 'C2'),
            'seg2_response': ('response', '-', 'C2'),
        }
        for ax, names in zip(axes, PAIRS, strict=True):
            for name in names:
                label, style, color = role[name]
                run = RUNS[name]
                ax.plot(
                    np.asarray(run['Time']) / 1e6,
                    run['T_surf'],
                    style,
                    color=color,
                    marker='.',
                    ms=4,
                    label=label,
                )
            if names[1] == 'seg1_off':
                t_stop, T_stop = (
                    RUNS['seg1_off']['Time'][-1] / 1e6,
                    RUNS['seg1_off']['T_surf'][-1],
                )
                ax.plot(t_stop, T_stop, 'x', color='C0', ms=9, mew=2, label='stop')
                lo, hi = (
                    min(RUNS['seg1_response']['T_surf']),
                    max(RUNS['seg1_response']['T_surf']),
                )
                ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.5 * (hi - lo))  # room for the legend
            ax.set(ylabel='T_surf [K]')
            ax.legend(fontsize=8)
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
    stored rows or the stored stop."""
    text = PAGE.read_text()
    section = text.split('## Coupled runs through the lid onset', 1)[1].split('\n## ', 1)[0]
    missing = [f for f in _facts() if not re.search(r'(?<![\d.])' + re.escape(f), section)]
    assert missing == []
