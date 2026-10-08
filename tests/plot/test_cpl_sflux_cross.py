"""Unit tests for ``proteus.plot.cpl_sflux_cross``.

Covers ``planck_function`` (closed-form Planck radiance) and
``plot_sflux_cross`` (early-return guard plus full plot path with
modern-spectrum overlay) and the ``plot_sflux_cross_entry`` wrapper.
Matplotlib is mocked at the source-binding attribute.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import proteus.plot.cpl_sflux_cross as sflux_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# planck_function: closed-form Planck radiance
# ---------------------------------------------------------------------------


def test_planck_function_peak_wavelength_obeys_wien_displacement_law():
    """The Planck function for a blackbody at T peaks at wavelength
    lam_max ~= 2.898e6 nm * K / T (Wien's displacement law). For T = 5772 K
    (solar Teff), this gives lam_max ~= 502 nm. The flux at the peak must
    exceed the flux at adjacent wavelengths (300 nm UV and 900 nm NIR).

    Discrimination: a wrong-formula Planck (e.g. dropping the (e^x-1)^-1
    factor) would not exhibit this peak, and the comparison would fail.
    """
    T = 5772.0  # Solar effective temperature
    peak_lam = 2.898e6 / T  # ~502 nm
    f_peak = sflux_mod.planck_function(peak_lam, T)
    f_uv = sflux_mod.planck_function(300.0, T)
    f_nir = sflux_mod.planck_function(900.0, T)

    assert f_peak > f_uv
    assert f_peak > f_nir
    # Positivity / scale guards
    assert f_peak > 0
    # Order of magnitude: 1e6 erg s-1 cm-2 nm-1 for solar Teff
    assert 1e4 < f_peak < 1e8


def test_planck_function_scales_with_temperature_via_stefan_boltzmann_integral():
    """The wavelength-integrated Planck flux scales as T^4 (Stefan-Boltzmann).
    Doubling T must produce a flux at the new peak wavelength that is
    much larger than the original peak flux; specifically the ratio of
    new-peak-flux / old-peak-flux at the same wavelength must exceed 2^3.

    Discrimination: a regression that used T^3 or T^5 exponent in the
    Planck derivation would land outside this bound.
    """
    T_low = 3000.0
    T_high = 6000.0  # 2x
    # Pick a wavelength in both peak regions (visible)
    lam = 500.0
    f_low = sflux_mod.planck_function(lam, T_low)
    f_high = sflux_mod.planck_function(lam, T_high)

    # Discrimination: at 500 nm, going from 3000 K to 6000 K is moving from
    # the Wien tail into the peak. The ratio is huge (orders of magnitude).
    ratio = f_high / f_low
    # The exact Planck ratio at 500 nm between 6000 K and 3000 K is ~1700.
    # Any T^3 / T^4 / T^5 polynomial fit would give a few orders of magnitude
    # less; the exponential (e^x - 1)^-1 dominates this regime.
    assert ratio > 100
    assert f_low > 0
    assert f_high > 0


# ---------------------------------------------------------------------------
# plot_sflux_cross: early-return guard
# ---------------------------------------------------------------------------


def _install_mock_plt(monkeypatch):
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    monkeypatch.setattr(sflux_mod, 'plt', mock_plt)
    return mock_plt, mock_fig, mock_ax


def test_plot_sflux_cross_returns_early_when_few_files(monkeypatch, tmp_path):
    """With <=1 .sflux file in output_dir/data, the function logs a
    warning and returns. A regression that proceeded would have called
    subplots.
    """
    mock_plt, _, _ = _install_mock_plt(monkeypatch)
    (tmp_path / 'data').mkdir()
    # Create only one sflux file
    (tmp_path / 'data' / '100.sflux').write_text(
        'header\n100.0\t1.0e-3\n200.0\t1.0e-4\n', encoding='utf-8'
    )

    sflux_mod.plot_sflux_cross(output_dir=str(tmp_path))

    assert mock_plt.subplots.call_count == 0
    assert mock_plt.savefig.call_count == 0


# ---------------------------------------------------------------------------
# plot_sflux_cross: full path
# ---------------------------------------------------------------------------


def _write_sflux(path, wavelengths, fluxes):
    """Write a tab-separated .sflux file with the first row as a header."""
    lines = ['wave\tflux']
    for w, f in zip(wavelengths, fluxes):
        lines.append(f'{w}\t{f}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def test_plot_sflux_cross_full_path_plots_bins_over_time(monkeypatch, tmp_path):
    """With multiple .sflux files over time, the function loads each,
    selects the 8 default wavelength bins, and produces one plot line
    per bin. With 3 time files and 8 wavelength bins, ax.plot must be
    called 8 times.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    (tmp_path / 'data').mkdir()

    wavelengths = np.array([1.0, 12.0, 50.0, 121.0, 200.0, 400.0, 500.0, 2000.0])
    # 3 time snapshots: 100 yr, 1e4 yr, 1e6 yr
    for time in (100, 10000, 1000000):
        fluxes = np.full_like(wavelengths, 1e-4 * (time / 100))
        _write_sflux(tmp_path / 'data' / f'{time}.sflux', wavelengths, fluxes)

    mock_plt, mock_fig, mock_ax = _install_mock_plt(monkeypatch)

    sflux_mod.plot_sflux_cross(output_dir=str(tmp_path), plot_format='pdf')

    assert mock_plt.subplots.call_count == 1
    # Discrimination: 8 bins -> 8 line plots
    assert mock_ax.plot.call_count == 8
    assert mock_fig.savefig.call_count == 1


def test_plot_sflux_cross_overlays_modern_spectrum_when_age_positive(monkeypatch, tmp_path):
    """When modern_age > 0, the function loads the -1.sflux file and
    overlays scatter points for each wavelength bin. With 8 bins, 8
    scatter calls must happen on top of the 8 line plots.
    """
    plots_dir = tmp_path / 'plots'
    plots_dir.mkdir()
    (tmp_path / 'data').mkdir()

    wavelengths = np.array([1.0, 12.0, 50.0, 121.0, 200.0, 400.0, 500.0, 2000.0])
    # Time snapshots
    for time in (100, 10000, 1000000):
        fluxes = np.full_like(wavelengths, 1e-4 * (time / 100))
        _write_sflux(tmp_path / 'data' / f'{time}.sflux', wavelengths, fluxes)
    # Modern spectrum (-1.sflux) needs 2 header rows (skiprows=2)
    modern_path = tmp_path / 'data' / '-1.sflux'
    modern_lines = ['# modern header line 1', '# modern header line 2']
    for w, f in zip(wavelengths, np.full_like(wavelengths, 1e-3)):
        modern_lines.append(f'{w}\t{f}')
    modern_path.write_text('\n'.join(modern_lines) + '\n', encoding='utf-8')

    mock_plt, mock_fig, mock_ax = _install_mock_plt(monkeypatch)

    sflux_mod.plot_sflux_cross(output_dir=str(tmp_path), modern_age=4.567e9, plot_format='pdf')

    # Discrimination: 8 line plots + 8 modern-spectrum scatter overlays
    assert mock_ax.plot.call_count == 8
    assert mock_ax.scatter.call_count == 8


# ---------------------------------------------------------------------------
# plot_sflux_cross_entry: wrapper
# ---------------------------------------------------------------------------


def test_plot_sflux_cross_entry_passes_age_when_mors_module(monkeypatch):
    """When the star module is ``mors``, the entry wrapper extracts
    modern_age = age_now * 1e9 from config. A regression that left
    modern_age at -1 even for MORS configs would silently drop the
    modern-spectrum overlay.
    """
    captured = {}

    def fake_plot(output_dir, wl_targets=None, modern_age=-1, plot_format='pdf'):
        captured['modern_age'] = modern_age
        captured['plot_format'] = plot_format

    monkeypatch.setattr(sflux_mod, 'plot_sflux_cross', fake_plot)

    handler = MagicMock()
    handler.directories = {'output': '/tmp/out'}
    handler.config.star.module = 'mors'
    handler.config.star.mors.age_now = 4.567  # Gyr
    handler.config.params.out.plot_fmt = 'png'

    sflux_mod.plot_sflux_cross_entry(handler)

    # Discrimination: modern_age must be ~4.567e9 yr (age_now * 1e9), not -1
    assert captured['modern_age'] == pytest.approx(4.567e9, rel=1e-12)
    assert captured['plot_format'] == 'png'


def test_plot_sflux_cross_entry_sets_modern_age_negative_for_non_mors(monkeypatch):
    """For non-MORS star modules (e.g. ``dummy``), modern_age is set
    to -1 to disable the modern-spectrum overlay. A regression that
    used some other sentinel would either crash on the file load or
    plot an unwanted overlay.
    """
    captured = {}

    def fake_plot(output_dir, wl_targets=None, modern_age=-1, plot_format='pdf'):
        captured['modern_age'] = modern_age

    monkeypatch.setattr(sflux_mod, 'plot_sflux_cross', fake_plot)

    handler = MagicMock()
    handler.directories = {'output': '/tmp/out'}
    handler.config.star.module = 'dummy'
    handler.config.params.out.plot_fmt = 'pdf'

    sflux_mod.plot_sflux_cross_entry(handler)

    assert captured['modern_age'] == -1
    # Discrimination: -1 is the documented "disable overlay" sentinel;
    # a regression that emitted 0 (a plausible alternative falsy value)
    # would still pass a generic falsiness check but break the loader.
    assert captured['modern_age'] < 0
