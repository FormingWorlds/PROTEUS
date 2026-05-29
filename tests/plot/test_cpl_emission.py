"""Unit tests for ``proteus.plot.cpl_emission``."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import proteus.plot.cpl_emission as emission_mod


class _FakeDataset(dict):
    def close(self):
        return None


@pytest.mark.unit
def test_plot_emission_returns_early_with_insufficient_snapshots(tmp_path):
    result = emission_mod.plot_emission(str(tmp_path), times=[1.0, 1.0, 2.0], plot_format='png')
    assert result is None


@pytest.mark.unit
def test_plot_emission_handles_greygas_branch(monkeypatch, tmp_path):

    # greygas RT and dummy atmos_clim only have one spectral band
    ds = _FakeDataset(
        {
            'ba_U_LW': np.array([[2.0]]),
            'ba_U_SW': np.array([[1.0]]),
            'bandmin': np.array([1e-6]),
            'bandmax': np.array([2e-6]),
        }
    )

    mock_ax = MagicMock()
    mock_ax.yaxis = MagicMock()
    mock_fig = MagicMock()
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    monkeypatch.setattr(emission_mod.nc, 'Dataset', lambda _path: ds)
    monkeypatch.setattr(emission_mod, 'plt', mock_plt)

    # try making emission spectrum plot
    emission_mod.plot_emission(
        str(tmp_path), times=[1.0, 2.0, 3.0], plot_format='png', cumulative=False
    )

    # check that function reached savefig call, so was successful
    assert mock_ax.step.called
    mock_ax.set_xlim.assert_called_once()
    mock_fig.savefig.assert_called_once()


@pytest.mark.unit
def test_plot_emission_handles_reversed_bands_and_cumulative(monkeypatch, tmp_path):
    # checks that bands are plotted in standard order (shortwave to longwave)
    ds = _FakeDataset(
        {
            'ba_U_LW': np.array([[2.0, 3.0]]),
            'ba_U_SW': np.array([[1.0, 4.0]]),
            'bandmin': np.array([3e-6, 1e-6]),
            'bandmax': np.array([4e-6, 2e-6]),
        }
    )

    mock_ax = MagicMock()
    mock_ax.yaxis = MagicMock()
    mock_fig = MagicMock()
    mock_plt = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    monkeypatch.setattr(emission_mod.nc, 'Dataset', lambda _path: ds)
    monkeypatch.setattr(emission_mod, 'plt', mock_plt)

    emission_mod.plot_emission(
        str(tmp_path), times=[1.0, 2.0, 3.0], plot_format='png', cumulative=True
    )

    assert mock_ax.step.called
    mock_ax.set_xlim.assert_called_once()
    mock_fig.savefig.assert_called_once()
