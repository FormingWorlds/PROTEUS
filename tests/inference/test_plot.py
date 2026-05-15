"""
Unit tests for inference plotting utilities.

References:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
import toml

import proteus.inference.plot as plot_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_plots_perf_timeline_BO(tmp_path):
    """``plots_perf_timeline`` with an empty logs list produces no PNGs;
    the plotting pipeline gracefully no-ops rather than writing an
    empty figure.
    """
    plot_mod.plots_perf_timeline(logs=[], directory=str(tmp_path), n_init=0)
    assert not list((tmp_path / 'plots').glob('*.png'))


@pytest.mark.unit
def test_plot_result_correlation_BO(monkeypatch, tmp_path, caplog):
    """``plot_result_correlation`` reads each worker's helpfile, applies
    log/linear scaling per parameter via ``variable_is_logarithmic``,
    sets axis labels, draws the legend, and saves the figure. Missing
    helpfiles produce a warning log entry rather than crashing.
    """
    caplog.set_level('WARNING')
    workers = tmp_path / 'workers'

    # check helpfile reads ok
    case_ok = workers / 'w_0' / 'i_0'
    case_ok.mkdir(parents=True)
    (case_ok / 'init_coupler.toml').write_text(
        toml.dumps({'planet': {'mass_tot': 1.5}}),
        encoding='utf-8',
    )
    pd.DataFrame([{'P_surf': 1.0}]).to_csv(
        case_ok / 'runtime_helpfile.csv', sep=' ', index=False
    )

    # check known-missing case
    case_missing = workers / 'w_0' / 'i_1'
    case_missing.mkdir(parents=True)
    (case_missing / 'init_coupler.toml').write_text(
        toml.dumps({'planet': {'mass_tot': 2.0}}),
        encoding='utf-8',
    )

    axis = MagicMock()
    axis.__getitem__.return_value = axis
    fig = MagicMock()

    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (fig, axis)
    monkeypatch.setattr(plot_mod, 'plt', mock_plt)
    monkeypatch.setattr(plot_mod, 'variable_is_logarithmic', lambda _k: True)

    # try making the plot
    plot_mod.plot_result_correlation(
        pars={'planet.mass_tot': [0.7, 3.0]},
        obs={'P_surf': 1.0},
        directory=str(tmp_path),
    )

    # axis scaling and labels are defined by is_logarithmic function
    axis.set_xscale.assert_called_with('log')
    axis.set_yscale.assert_called_with('log')
    axis.legend.assert_called_once()
    axis.set_xlabel.assert_called_once()
    axis.set_ylabel.assert_called_once()
    fig.savefig.assert_called_once()
    assert 'Missing helpfile for' in caplog.text
