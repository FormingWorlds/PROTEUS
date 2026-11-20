"""Validation of the in-step open part of the lid from stored coupled runs.

Invariant: with the open part of the surface following its node within the interior step (and
the node held at least at the lid base), the step out of the live-lid row of a 1 Earth-mass run
moves the surface temperature less, gives off heat at the surface and keeps the magma cooler
than the same step with the response off, with no clip of T_magma. The rows are in
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
NAMES = ('seg1_response', 'seg1_lid_off', 'seg2_response', 'seg2_off')
START = ('Time', 'T_surf', 'F_int', 'T_magma', 'lid_weight', 'E_residual_cons_frac')
# Sanity bound on the energy residual: the largest of the 4 stored runs is 7.0e-6.
E_RES_BOUND = 1.0e-5


def _swing(run):
    return np.abs(np.diff(np.asarray(RUNS[run]['T_surf'])))


def _live(run):
    return [i for i, w in enumerate(RUNS[run]['lid_weight']) if w > 0.0]


def _step(run, key):
    """Change of ``key`` over the step out of row 1, the live-lid row of the response runs."""
    return RUNS[run][key][2] - RUNS[run][key][1]


def _facts():
    """Numbers the page states, each formatted as on the page, from the stored rows."""
    r1, l1, r2, o2 = (RUNS[n] for n in NAMES)
    e_max = max(abs(x) for n in NAMES for x in RUNS[n]['E_residual_cons_frac'])
    stop = re.search(r'from ([\d.]+) K at ([\d.]+) Myr', RUNS['provenance']['seg1_off_stop'])
    return [
        f'{_swing("seg1_response").max():.1f} K',
        f'{_swing("seg1_lid_off").max():.1f} K',
        f'{-min(r1["F_int"]):.2f} W m$^{{-2}}$',
        f'{-min(l1["F_int"]):.2f} W m$^{{-2}}$',
        f'{_step("seg2_response", "T_surf"):.1f} K',
        f'{_step("seg2_off", "T_surf"):.1f} K',
        f'{-o2["F_int"][2]:.2f} W m$^{{-2}}$',
        f'{r2["T_magma"][2]:.0f} K',
        f'{o2["T_magma"][2]:.0f} K',
        f'{r2["Time"][1] / 1e6:.3f} Myr',
        f'{r2["Time"][2] / 1e6:.3f} Myr',
        f'{float(stop.group(1)):.0f} K',
        f'{float(stop.group(2)):.2f} Myr',
        f'${e_max / 10 ** np.floor(np.log10(e_max)):.1f} \\times 10^{{{int(np.floor(np.log10(e_max)))}}}$',
    ]


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_open_part_keeps_the_coupled_runs_from_swinging(validation_plot):
    """The response acts on the one step out of the live-lid row. In segment 2 the runs with and
    without it agree up to that row; on that step the run with it moves T_surf less than half as
    much, gives off heat and keeps the magma cooler. In segment 1 (where the run without it stops
    at that step) it swings less and takes up less heat than the run with the lid off. Later rows
    differ through the path of each run and are not credited to the response."""
    for a, b in (('seg1_response', 'seg1_lid_off'), ('seg2_response', 'seg2_off')):
        assert all(RUNS[a][k][0] == RUNS[b][k][0] for k in START)
        assert _live(a) == [1]
    assert _live('seg1_lid_off') == []
    assert _live('seg2_off') == [1]
    assert all(RUNS['seg2_response'][k][:2] == RUNS['seg2_off'][k][:2] for k in START)
    # Ratio 0.163 on the stored pair; 5 round-off kicked starts of the response run spread by 4.5e-6.
    assert abs(_step('seg2_response', 'T_surf')) < 0.5 * abs(_step('seg2_off', 'T_surf'))
    assert RUNS['seg2_response']['F_int'][2] > 0.0 > RUNS['seg2_off']['F_int'][2]
    assert RUNS['seg2_response']['T_magma'][2] < RUNS['seg2_off']['T_magma'][2]
    assert _swing('seg1_response').max() < _swing('seg1_lid_off').max()
    assert min(RUNS['seg1_response']['F_int']) > -1.0 > min(RUNS['seg1_lid_off']['F_int'])
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
            'seg1_lid_off': ('lid off', '--', 'C1'),
            'seg2_off': ('response off', ':', 'C0'),
            'seg1_response': ('response', '-', 'C2'),
            'seg2_response': ('response', '-', 'C2'),
        }
        for ax, names in zip(
            axes,
            (('seg1_lid_off', 'seg1_response'), ('seg2_off', 'seg2_response')),
            strict=True,
        ):
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
            ax.set(ylabel='T_surf [K]')
            ax.legend(fontsize=8)
            # 4 ticks per axis keep the pair 5.9 kB under the 60 kB cap.
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(MaxNLocator(4))
        axes[1].set(xlabel='Time [Myr]')
        fig.tight_layout()
        return fig

    validation_plot(figure, 'atmos_clim_lid_open_coupled')


@pytest.mark.physics_invariant
def test_the_page_states_the_numbers_of_the_stored_coupled_rows():
    """Each number of the coupled case on the Validation page comes from the stored rows."""
    text = PAGE.read_text()
    section = text.split('## Coupled runs through the lid onset', 1)[1].split('\n## ', 1)[0]
    missing = [f for f in _facts() if f not in section]
    assert missing == []
