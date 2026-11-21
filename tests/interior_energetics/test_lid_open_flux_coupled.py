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
REFERENCE = Path(__file__).parents[2] / 'docs' / 'Reference' / 'lid_coupling.md'
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


def _live(run):
    """Indices of the rows of ``run`` with a live lid."""
    return [i for i, w in enumerate(RUNS[run]['lid_weight']) if w > 0.0]


def _later_steps(run):
    """Size of each change of ``T_surf`` between rows after the step out of the live-lid row."""
    T = RUNS[run]['T_surf']
    return [abs(T[i + 1] - T[i]) for i in range(LIVE + 1, len(T) - 1)]


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
        facts += [
            f'from {t[s][0]:.3f} to {t[s][1]:.3f} Myr',
            f'({_step(r, "T_surf"):.1f} K against {_step(o, "T_surf"):.1f} K)',
            f'({_step(r, "T_magma"):.1f} K against {_step(o, "T_magma"):.1f} K)',
            f'({_after(r, "T_magma"):.0f} K against {_after(o, "T_magma"):.0f} K)',
        ]
        for run in (r, o):
            lo, hi = RUNS[run]['F_atm'][LIVE], RUNS[run]['F_atm'][LIVE + 1]
            facts.append(f'{_after(run, "F_int"):.2f} {wm2} in [{lo:.2f}, {hi:.2f}] {wm2}')
    swings = [max(_later_steps(n)) for n in NAMES]
    ratios = [_step(a, 'T_surf') / _step(b, 'T_surf') for a, b in PAIRS]
    k2 = _live('seg1_response')[1]
    dT_end = _after('seg1_response', 'T_magma') - _after('seg1_response', 'T_surf')
    return [
        *facts,
        f'up to {swings[0]:.1f} and {swings[1]:.1f} K between rows in the first segment and '
        f'{swings[2]:.1f} and {swings[3]:.1f} K in the second',
        f'({RUNS["seg1_response"]["Time"][k2] / 1e6:.3f} Myr, $w$ = '
        f'{RUNS["seg1_response"]["lid_weight"][k2]:.3f})',
        f'the 2 stored ratios, {ratios[0]:.2f} and {ratios[1]:.2f}',
        f'lies {dT_end:.2f} K below the magma temperature',
        f'${e_max / 10**exp:.1f} \\times 10^{{{exp}}}$',
    ]


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_open_part_carries_the_coupled_runs_through_the_live_lid_step(validation_plot):
    """The response acts on the step out of the live-lid row; each pair agrees up to that row.
    Both runs book an interior loss inside the atmosphere flux band of that step (the energy
    closure). With the response the step moves T_surf and T_magma by less than 2/3 of the step
    with the response off and ends with cooler magma. No run clips T_magma. Later rows differ
    through the path of each run and are not credited to the response."""
    for a, b in PAIRS:
        assert all(RUNS[a][k][: LIVE + 1] == RUNS[b][k][: LIVE + 1] for k in START)
        assert RUNS[a]['lid_weight'][LIVE - 1] == 0.0 < RUNS[a]['lid_weight'][LIVE]
        for run in (a, b):  # the band holds the energy closure of both runs
            lo, hi = _band(run)
            assert lo <= _after(run, 'F_int') <= hi
        for key in ('T_surf', 'T_magma'):
            assert abs(_step(a, key)) < SWING_RATIO * abs(_step(b, key))
        assert _after(a, 'T_magma') < _after(b, 'T_magma')
    for name in NAMES:
        assert set(RUNS[name]['lid_slope_guard']) == {0.0}  # no row needs a fallback slope
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


@pytest.mark.physics_invariant
def test_the_reference_states_the_lid_onset_of_the_stored_rows():
    """The Reference page gives the onset of the stored segments: the weight of the step that
    switches the lid on, its largest weight in the second segment, the fall back to 0 in the
    next step (with the unconverged AGNI row of the first segment), the held surface (the row
    before) above the lid base, the top node at the end below it, and the later live row of
    the first segment. The numbers come from the stored rows."""
    text = REFERENCE.read_text()
    w1, w2 = (RUNS[f'{seg}_response']['lid_weight'][LIVE] for seg in ('seg1', 'seg2'))
    assert all(
        RUNS[n]['lid_weight'][LIVE - 1] == 0.0 < RUNS[n]['lid_weight'][LIVE] for n in NAMES
    )
    assert f'rose from 0 to {w1:.2f} in the first segment and to {w2:.2f} in the second' in text
    w_max = RUNS['seg2_response']['lid_weight_max'][LIVE]
    assert (
        w_max > w2 and f'reached {w_max:.2f} at the largest (column `lid_weight_max`)' in text
    )
    assert all(RUNS[n]['lid_weight'][LIVE + 1] == 0.0 for n in NAMES) and len(NAMES) == 4
    assert 'In the next step it fell back to 0 in all 4 runs' in text
    t_end = RUNS['seg1_response']['Time'][LIVE + 1] / 1e6
    assert f'{t_end:.6f} Myr in seg1_response' in RUNS['provenance']['note']
    assert (
        'in the first segment with the response the AGNI solve of that row did not converge'
        in (text)
    )
    held, node = [], []
    for name in ('seg1_response', 'seg2_response'):
        run = RUNS[name]
        T_b = run['lid_T_base'][LIVE]
        assert run['T_surf'][LIVE - 1] > T_b > run['lid_T_node'][LIVE]
        held.append(run['T_surf'][LIVE - 1] - T_b)
        node.append(T_b - run['lid_T_node'][LIVE])
    assert f'held surface lay {held[0]:.1f} and {held[1]:.1f} K above $T_b$' in text
    assert f'top node lay {node[0]:.1f} and {node[1]:.1f} K below $T_b$' in text
    run, i = RUNS['seg1_response'], LIVE + 2
    w, d = run['lid_weight'][i], run['lid_T_base'][i] - run['lid_T_node'][i]
    assert 0.0 < w < 1.0 and RUNS['seg1_off']['lid_weight'][i] == 0.0
    assert (
        f'rose again 2 rows later, to {w:.2f}, with the top node {d:.1f} K below $T_b$' in text
    )
