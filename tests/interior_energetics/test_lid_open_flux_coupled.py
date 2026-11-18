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
# Energy residual of the uniform-mesh run of the same code version, the bound of the check.
E_RES_BOUND = 2.5e-5


def _swing(run):
    return np.abs(np.diff(np.asarray(RUNS[run]['T_surf'])))


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_open_part_keeps_the_coupled_runs_from_swinging(validation_plot):
    """Both segments: the runs with the response swing less than the runs with the fixed open
    flux, the lid-off run
    included for V1, take up at most 1 W/m2 at the surface, keep T_magma below 1600 K, clip
    nothing and close the energy to the bound."""
    for fixed, before in (('V1', 'R1_before'), ('V2', 'probe_before')):
        f, b = RUNS[fixed], RUNS[before]
        assert _swing(fixed).max() < 0.5 * _swing(before).max()
        assert min(f['F_int']) > -1.0 > min(b['F_int'])
        assert max(f['T_magma']) < 1600.0 < max(b['T_magma'])
        assert sum(f['T_magma_clipped']) == 0.0
        assert max(abs(x) for x in f['E_residual_cons_frac']) <= E_RES_BOUND
    assert _swing('V1').max() < _swing('R2_lid_off').max()

    def figure():
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.6))
        role = {
            'R1_before': ('fixed open flux', ':', 'C0'),
            'probe_before': ('fixed open flux', ':', 'C0'),
            'R2_lid_off': ('lid off', '--', 'C1'),
            'V1': ('in-step open part', '-', 'C2'),
            'V2': ('in-step open part', '-', 'C2'),
        }
        for ax, names in zip(
            axes, (('R1_before', 'R2_lid_off', 'V1'), ('probe_before', 'V2')), strict=True
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
