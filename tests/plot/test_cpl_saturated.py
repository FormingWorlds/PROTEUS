"""Unit tests for ``proteus.plot.cpl_saturated``.

Covers ``plot_saturated`` (skip when no condensate was solved for, the plotted
activity, condensed column and layer thickness, the masking of the solver floor
below saturation, and the written file) and the ``plot_saturated_entry``
wrapper.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import proteus.plot.cpl_saturated as sat_mod
from proteus.utils.constants import condensate_density, condensate_list

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

R_INT = 2.5e6  # [m]
FLOOR = 1.0e-6  # condensed mass the solver reports below saturation [kg]


def _make_hf_all(n: int = 8) -> pd.DataFrame:
    """Helpfile frame with graphite saturating halfway and sulfur never saturating.

    The first two rows are the init rows ``plot_saturated`` drops, the same as
    a real run writes them.
    """
    time = np.concatenate([[0.0, 0.0], np.logspace(1, 6, n - 2)])
    df = pd.DataFrame(
        {
            'Time': time,
            'T_magma': np.linspace(3000.0, 1500.0, n),
            'Phi_global': np.linspace(1.0, 0.1, n),
            'R_int': np.full(n, R_INT),
        }
    )
    for c in condensate_list:
        df[f'{c}_cd_activity'] = 0.0
        df[f'{c}_cd_kg'] = 0.0
    half = n // 2
    df['C_cd_activity'] = np.where(np.arange(n) < half, 0.4, 1.0)
    df['C_cd_kg'] = np.where(np.arange(n) < half, FLOOR, np.linspace(1e20, 4e20, n))
    df['S_cd_activity'] = np.full(n, 1.0e-3)
    df['S_cd_kg'] = np.full(n, FLOOR)
    return df


@pytest.mark.physics_invariant
def test_column_and_thickness_follow_mass_area_and_density(draw_real, tmp_path):
    """The column is mass over 4 pi R^2 and the thickness that over the density.

    Below saturation the solver's floor mass is plotted as zero, so graphite
    starts at zero and only carries a mass once its activity reaches 1.
    """
    (tmp_path / 'plots').mkdir()
    hf = _make_hf_all()
    fig = draw_real(sat_mod.plot_saturated, hf, str(tmp_path), plot_format='png')[0]
    ax_act, ax_col, ax_thick, _ = fig.axes[:4]

    crop = hf.iloc[2:]
    area = 4.0 * np.pi * R_INT**2
    saturated = crop['C_cd_activity'].to_numpy() >= sat_mod.SATURATED
    mass = np.where(saturated, crop['C_cd_kg'].to_numpy(), 0.0)

    graphite_col, sulfur_col = ax_col.lines[0].get_ydata(), ax_col.lines[1].get_ydata()
    assert graphite_col == pytest.approx(mass / area / 1e6, rel=1e-12)
    assert ax_thick.lines[0].get_ydata() == pytest.approx(
        mass / area / condensate_density['C'], rel=1e-12
    )
    # Discrimination: the undersaturated rows carry the floor mass, which must not show.
    assert graphite_col[0] == 0.0 and crop['C_cd_kg'].iloc[0] == FLOOR
    assert np.all(sulfur_col == 0.0)
    # Graphite and sulfur are the two condensates solved for; water and acid are absent.
    assert [line.get_label() for line in ax_act.lines[:2]] == ['Graphite', 'Sulfur']
    assert (tmp_path / 'plots' / 'plot_saturated.png').stat().st_size > 0


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'late_activity', [0.3, 0.0], ids=['activity-dropped', 'left-the-network']
)
def test_a_held_layer_is_drawn_below_saturation(draw_real, tmp_path, late_activity):
    """A layer kept by prevent_desaturation stays on the plot once activity drops.

    The inventory left in the solve can fall below saturation while the held layer
    persists, so a mass well above the solver floor is drawn at any activity, and
    the floor itself is still drawn as zero. Edge case: once the condensate leaves
    the network its activity is written as 0, and the held layer is still drawn.
    """
    (tmp_path / 'plots').mkdir()
    hf = _make_hf_all()
    n = len(hf)
    late = np.arange(n) >= n - 2  # the last two rows: layer held, activity has dropped
    hf['C_cd_activity'] = np.where(late, late_activity, hf['C_cd_activity'])
    hf['C_cd_kg'] = np.where(late, 4.0e20, hf['C_cd_kg'])
    fig = draw_real(sat_mod.plot_saturated, hf, str(tmp_path), plot_format='png')[0]

    column = fig.axes[1].lines[0].get_ydata()
    area = 4.0 * np.pi * R_INT**2
    assert column[-2:] == pytest.approx(4.0e20 / area / 1e6, rel=1e-12)
    assert column[0] == 0.0  # the floor at the first, undersaturated row


@pytest.mark.physics_invariant
def test_a_layer_held_after_a_single_saturated_point_is_drawn(draw_real, tmp_path):
    """A condensate saturated on one solve and held from then on is drawn.

    Edge case: once a held layer takes nearly all of its element, the condensate
    leaves the network and its activity is written as 0, so the run has only one
    point with a positive activity. The held points count towards the two points a
    condensate needs to be drawn, and the column carries the held mass throughout.
    """
    (tmp_path / 'plots').mkdir()
    hf = _make_hf_all()
    rows = np.arange(len(hf))
    first = 2  # the first row after the init rows
    hf['C_cd_activity'] = np.where(rows == first, 1.0, 0.0)
    hf['C_cd_kg'] = np.where(rows >= first, 4.0e20, 0.0)
    fig = draw_real(sat_mod.plot_saturated, hf, str(tmp_path), plot_format='png')[0]
    ax_act, ax_col = fig.axes[:2]

    assert [line.get_label() for line in ax_act.lines[:2]] == ['Graphite', 'Sulfur']
    area = 4.0 * np.pi * R_INT**2
    assert ax_col.lines[0].get_ydata() == pytest.approx(
        np.full(len(hf) - 2, 4.0e20 / area / 1e6), rel=1e-12
    )


def test_skipped_without_any_solved_condensate(tmp_path, monkeypatch):
    """No figure is made when no condensate column holds two solved points.

    Edge cases: a helpfile written before the condensate columns existed, and one
    where every activity and condensed mass is zero, as the wrapper writes them with
    condensates disabled.
    """
    mock_plt = MagicMock()
    monkeypatch.setattr(sat_mod, 'plt', mock_plt)
    base = _make_hf_all()

    legacy = base.drop(columns=[c for c in base.columns if '_cd_' in c])
    sat_mod.plot_saturated(legacy, str(tmp_path))
    disabled = base.copy()
    for c in condensate_list:
        disabled[f'{c}_cd_activity'] = 0.0
        disabled[f'{c}_cd_kg'] = 0.0
    sat_mod.plot_saturated(disabled, str(tmp_path))

    assert not mock_plt.subplots.called
    assert not list(tmp_path.iterdir())


def test_entry_reads_the_helpfile_and_plots_it(monkeypatch):
    """The entry point reads the run's helpfile and passes it on with the format."""
    hf = _make_hf_all()
    read = MagicMock(return_value=hf)
    plot = MagicMock()
    monkeypatch.setattr('proteus.utils.coupler.read_helpfile_table', read)
    monkeypatch.setattr(sat_mod, 'plot_saturated', plot)
    handler = SimpleNamespace(
        directories={'output': '/run'},
        config=SimpleNamespace(params=SimpleNamespace(out=SimpleNamespace(plot_fmt='pdf'))),
    )

    sat_mod.plot_saturated_entry(handler)

    assert read.call_args.args[0] == '/run/runtime_helpfile.csv'
    assert plot.call_args.kwargs == {'hf_all': hf, 'output_dir': '/run', 'plot_format': 'pdf'}
