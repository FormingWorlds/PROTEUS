"""Validation of the in-step open part of the lid from stored coupled runs.

Invariant: with the open part of the surface following its node within the interior step (and
the node held at least at the lid base), a 1 Earth-mass run through the lid onset keeps the
surface temperature from swinging between coupled steps, the interior from taking up large
heat at the surface, and the magma temperature near the solidus, with no clip of T_magma. The
rows are in ``tests/data/lid_open_flux/coupled_rows.json``; see
``docs/Validation/atmos_clim/common.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

DATA = Path(__file__).parents[1] / 'data' / 'lid_open_flux' / 'coupled_rows.json'
RUNS = json.loads(DATA.read_text())
# Sanity bound on the energy residual: the largest of the 5 stored runs is 7.2e-6.
E_RES_BOUND = 1.0e-5


def _swing(run):
    return np.abs(np.diff(np.asarray(RUNS[run]['T_surf'])))


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_open_part_keeps_the_coupled_runs_from_swinging(validation_plot):
    """Both segments: the runs with the response swing less than half as much as the runs with
    the fixed open flux, and in the first segment less than the run with the lid off; they take up
    at most 1 W/m2 at the surface, keep T_magma below 1600 K, clip nothing and close the energy to
    the bound. The lid is live on 2 rows of the first segment and 1 of the second; later rows
    differ through the path of each run."""
    for response, fixed in (('seg1_response', 'seg1_fixed'), ('seg2_response', 'seg2_fixed')):
        f, b = RUNS[response], RUNS[fixed]
        assert sum(w > 0.0 for w in f['lid_weight']) in (1, 2)
        assert _swing(response).max() < 0.5 * _swing(fixed).max()
        assert min(f['F_int']) > -1.0 > min(b['F_int'])
        assert max(f['T_magma']) < 1600.0 < max(b['T_magma'])
        assert sum(f['T_magma_clipped']) == 0.0
        assert max(abs(x) for x in f['E_residual_cons_frac']) <= E_RES_BOUND
    assert _swing('seg1_response').max() < _swing('seg1_lid_off').max()

    def figure():
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.6))
        role = {
            'seg1_fixed': ('fixed open flux', ':', 'C0'),
            'seg2_fixed': ('fixed open flux', ':', 'C0'),
            'seg1_lid_off': ('lid off', '--', 'C1'),
            'seg1_response': ('in-step open part', '-', 'C2'),
            'seg2_response': ('in-step open part', '-', 'C2'),
        }
        for ax, names in zip(
            axes,
            (('seg1_fixed', 'seg1_lid_off', 'seg1_response'), ('seg2_fixed', 'seg2_response')),
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
        axes[1].set(xlabel='Time [Myr]', ylim=(1500.0, 1650.0))
        fig.tight_layout()
        return fig

    validation_plot(figure, 'atmos_clim_lid_open_coupled')
