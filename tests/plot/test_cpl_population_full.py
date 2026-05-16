"""Full-path coverage for ``proteus.plot.cpl_population``.

Drives ``plot_population_mass_radius`` and ``plot_population_time_density``
end-to-end with mocked plt, synthetic DACE exoplanet rows, and synthetic
Zeng-2019 mass-radius curves. Complements the existing missing-data
guards in ``test_cpl_population.py``.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _hf_long(n=10, t_min=1e3, t_max=1e8):
    """Helpfile DataFrame with enough rows past t0=100 yr for the plot."""
    time = np.linspace(t_min, t_max, n)
    # M_planet in kg; M_earth ~ 5.972e24 -> 1 Mearth
    M_earth_kg = 5.972e24
    R_earth_m = 6.371e6
    return pd.DataFrame(
        {
            'Time': time,
            'M_planet': np.full(n, 1.5 * M_earth_kg),
            'R_obs': np.linspace(1.1 * R_earth_m, 1.0 * R_earth_m, n),
        }
    )


def _exo_df():
    """A two-row DACE-flavoured exoplanet DataFrame with the columns
    ``plot_population_*`` reads, in Jupiter units for mass/radius.
    """
    M_jup_in_Mearth = 317.8
    R_jup_in_Rearth = 11.21
    return pd.DataFrame(
        {
            'Planet Mass [Mjup]': [1.5 / M_jup_in_Mearth, 3.0 / M_jup_in_Mearth],
            'Planet Mass - Upper Unc [Mjup]': [0.1 / M_jup_in_Mearth, 0.1 / M_jup_in_Mearth],
            'Planet Mass - Lower Unc [Mjup]': [0.1 / M_jup_in_Mearth, 0.1 / M_jup_in_Mearth],
            'Planet Radius [Rjup]': [1.2 / R_jup_in_Rearth, 1.8 / R_jup_in_Rearth],
            'Planet Radius - Upper Unc [Rjup]': [
                0.05 / R_jup_in_Rearth,
                0.05 / R_jup_in_Rearth,
            ],
            'Planet Radius - Lower Unc [Rjup]': [
                0.05 / R_jup_in_Rearth,
                0.05 / R_jup_in_Rearth,
            ],
            'Stellar Age [Gyr]': [5.0, 3.0],
            'Planet Density [g/cm**3] - Computation': [5.5, 1.5],
        }
    )


def _mr_curves():
    """A minimal mass-radius-curve dict that the loader returns."""
    masses = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    return {
        '100% H2O': (masses, masses**0.27 * 1.3),
        '100% rock': (masses, masses**0.27 * 1.0),
    }


def test_plot_population_mass_radius_writes_figure_when_data_is_available(
    monkeypatch, tmp_path
):
    """With non-empty DACE + Zeng-2019 loaders and >= 3 hf rows past t0,
    ``plot_population_mass_radius`` reaches the savefig call. Discrimination:
    ``plt.subplots`` and ``fig.savefig`` must each be called exactly once.
    """
    import proteus.plot.cpl_population as pop

    mock_plt = MagicMock()
    fig = MagicMock()
    ax = MagicMock()
    mock_plt.subplots.return_value = (fig, ax)
    monkeypatch.setattr(pop, 'plt', mock_plt)
    monkeypatch.setattr(pop, '_get_exo_data', lambda _f: _exo_df())
    monkeypatch.setattr(pop, '_get_mr_data', lambda _f: _mr_curves())

    # The plot function expects an existing ``plots`` subfolder under
    # output_dir (it joins paths but does not mkdir).
    (tmp_path / 'plots').mkdir()

    pop.plot_population_mass_radius(
        _hf_long(),
        str(tmp_path),
        fwl_dir='ignored',
        plot_format='pdf',
    )

    assert mock_plt.subplots.call_count == 1
    assert fig.savefig.call_count == 1
    # Discriminator: the saved path must end with the correct extension.
    saved_path = fig.savefig.call_args.args[0]
    assert saved_path.endswith('plot_population_mass_radius.pdf')


def test_plot_population_mass_radius_uses_errorbars_when_show_errors_is_true(
    monkeypatch, tmp_path
):
    """``show_errors=True`` selects the errorbar branch; ``False`` (the
    default) selects scatter. Discrimination: ``ax.errorbar`` is called
    iff ``show_errors`` is True.
    """
    import proteus.plot.cpl_population as pop

    mock_plt = MagicMock()
    fig = MagicMock()
    ax = MagicMock()
    mock_plt.subplots.return_value = (fig, ax)
    monkeypatch.setattr(pop, 'plt', mock_plt)
    monkeypatch.setattr(pop, '_get_exo_data', lambda _f: _exo_df())
    monkeypatch.setattr(pop, '_get_mr_data', lambda _f: _mr_curves())
    (tmp_path / 'plots').mkdir()

    pop.plot_population_mass_radius(
        _hf_long(), str(tmp_path), fwl_dir='ignored', plot_format='pdf', show_errors=True
    )

    assert ax.errorbar.call_count == 1
    assert ax.scatter.call_count >= 1  # solar-system + end-of-sim markers


def test_plot_population_time_density_writes_figure(monkeypatch, tmp_path):
    """``plot_population_time_density`` reaches savefig when the DACE
    loader returns rows and >=3 hf samples sit past t0. Discrimination:
    the y-axis is set to log scale because density spans orders of
    magnitude.
    """
    import proteus.plot.cpl_population as pop

    mock_plt = MagicMock()
    fig = MagicMock()
    ax = MagicMock()
    mock_plt.subplots.return_value = (fig, ax)
    monkeypatch.setattr(pop, 'plt', mock_plt)
    monkeypatch.setattr(pop, '_get_exo_data', lambda _f: _exo_df())
    (tmp_path / 'plots').mkdir()

    pop.plot_population_time_density(
        _hf_long(), str(tmp_path), fwl_dir='ignored', plot_format='pdf'
    )

    assert fig.savefig.call_count == 1
    ax.set_yscale.assert_called_with('log')
    ax.set_xscale.assert_called_with('log')


def test_plot_population_time_density_skips_when_hf_too_short(monkeypatch, tmp_path):
    """Helpfile with < 3 rows past t0 short-circuits before any plot
    setup. Discrimination: ``plt.subplots`` is never called.
    """
    import proteus.plot.cpl_population as pop

    mock_plt = MagicMock()
    monkeypatch.setattr(pop, 'plt', mock_plt)
    monkeypatch.setattr(pop, '_get_exo_data', lambda _f: _exo_df())

    # Only two rows past t0=100 -> falls back to early-return branch.
    hf = _hf_long(n=2)
    pop.plot_population_time_density(hf, str(tmp_path), fwl_dir='ignored', plot_format='pdf')

    assert mock_plt.subplots.call_count == 0
    assert mock_plt.savefig.call_count == 0


def test_plot_population_entry_dispatches_to_both_plot_helpers(monkeypatch, tmp_path):
    """The entry wrapper reads ``runtime_helpfile.csv`` and calls each
    of the two plot helpers exactly once. Discrimination: a regression
    that dropped the second call would skip the time-density plot.
    """
    import proteus.plot.cpl_population as pop

    mr_calls = []
    td_calls = []

    def fake_mr(**kw):
        mr_calls.append(kw)

    def fake_td(**kw):
        td_calls.append(kw)

    monkeypatch.setattr(pop, 'plot_population_mass_radius', fake_mr)
    monkeypatch.setattr(pop, 'plot_population_time_density', fake_td)
    monkeypatch.setattr(pop.pd, 'read_csv', lambda *_a, **_kw: _hf_long())

    handler = MagicMock()
    handler.directories = {'output': str(tmp_path), 'fwl': 'fwl-dir'}
    handler.config.params.out.plot_fmt = 'pdf'

    pop.plot_population_entry(handler)

    assert len(mr_calls) == 1
    assert len(td_calls) == 1
    assert mr_calls[0]['plot_format'] == 'pdf'
    assert td_calls[0]['fwl_dir'] == 'fwl-dir'
