"""Validation of ``proteus.utils.energy_ledger.whole_run_ledger`` on stored rows of coupled runs.

The rows are in ``tests/data/energy_ledger/`` (``build_cases.py`` there writes them from the source runs and
records the source sha256 and the code versions). Cases: the lid onset of a 1 Earth-mass run (the mantle
residual and the coupling term, reproduced from the earlier measurement of that window), the core ledger
of core_module across 3 structure updates (closed to 1e-6 of the throughput) and the unbooked core heat
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
    fig.tight_layout(rect=(0.0, 0.17, 1.0, 1.0))
    return fig


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_lid_onset_window_reproduces_its_mantle_and_coupling_terms(validation_plot):
    """Lid onset of a 1 Earth-mass run (25 steps, all with a live lid): the mantle residual is
    -1.646e-6 and the coupling term G -5.698e-2 of L_int, as measured for that window before this
    function existed (lid-open-flux stream, ledger_window.py); the loss to space on the end-row area
    instead of the start-row area would move G outside the tolerance."""
    led, hf, prov = _case('r2rel_onset')
    L = led.terms['L_int'].sum()
    assert led.R_m.sum() / L == pytest.approx(-1.646e-6, rel=1e-3)
    assert led.G.sum() / L == pytest.approx(-5.698e-2, rel=1e-3)
    assert led.classes['lid'].all() and led.R_p.size == 25
    A = 4.0 * np.pi * hf['R_int'].to_numpy() ** 2
    dt = np.diff(hf['Time'].to_numpy()) * secs_per_year
    G_end = L - float(np.sum(hf['F_atm'].to_numpy()[:-1] * A[1:] * dt))
    assert abs(G_end / L + 5.698e-2) > 1e-3 * 5.698e-2  # the area of the end row misses
    assert prov['rows'] == [873, 898]

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
    3 updates in 450 yr the core residual is within 1e-6 of the throughput in every step class; without
    the booked term it would be the booked sum, 6.9e-3 of L_int."""
    led, _, _ = _case('cm_updates', start_row=2)
    T = led.throughput
    assert abs(led.R_c.sum()) <= 1e-6 * T
    assert all(abs(v) <= 1e-6 * T for v in led.split('R_c').values())
    B = led.terms['B_core'].sum()
    assert B / led.terms['L_int'].sum() == pytest.approx(6.92e-3, rel=1e-2)
    assert int(led.classes['update'].sum()) == 3


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_energy_balance_leaves_the_core_heat_jump_of_an_update_in_the_remap_class(
    validation_plot,
):
    """energy_balance books no core heat at a structure update, so the jump stays in R_c of the steps
    after an update: 5.948278897938286e-3 of L_int over 3 updates in 450 yr, the value of the separate
    E2 measurement script on the same rows (same arithmetic, 1e-9). The other steps hold 1e-2 of it."""
    led, _, _ = _case('eb_updates', start_row=2)
    L = led.terms['L_int'].sum()
    split = led.split('R_c')
    assert split['remap'] / L == pytest.approx(5.948278897938286e-3, rel=1e-9)
    assert abs(split['other'] + split['update']) < 1e-2 * split['remap']
    assert led.terms['B_core'].sum() == 0.0

    def figure():
        return _cumulative_figure(
            led,
            [
                ('R_c, after an update', np.where(led.classes['remap'], led.R_c, 0.0)),
                ('R_c, other', np.where(led.classes['remap'], 0.0, led.R_c)),
                ('G', led.G),
                ('R_m', led.R_m),
            ],
            'time [yr]',
            [-1e-3, -1e-5, 0.0, 1e-5, 1e-3],
        )

    validation_plot(figure, 'energy_ledger_core_remap')
