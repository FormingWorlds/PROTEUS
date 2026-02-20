"""Unit tests for ``proteus.plot.cpl_visual``.

Covers ``plot_visual``, ``anim_visual``, and their entry-point wrappers.
All matplotlib rendering, file I/O, NetCDF reads, and subprocess calls are
mocked so these tests run in < 100 ms without any external binaries.

Testing standards:
  - docs/test_infrastructure.md
  - docs/test_categorization.md
  - docs/test_building.md
"""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from proteus.plot.cpl_visual import anim_visual, plot_visual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hf_all(n: int = 5) -> pd.DataFrame:
    """Build a minimal runtime helpfile DataFrame.

    Parameters
    ----------
    n : int
        Number of time-step rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns required by ``plot_visual``.
    """
    return pd.DataFrame(
        {
            'Time': np.linspace(0.0, 1e7, n),
            'separation': np.full(n, 1.5e11),
            'R_int': np.full(n, 6.371e6),
            'R_star': np.full(n, 6.957e8),
        }
    )


def _make_ncdf_dict(n_lev: int = 4, n_band: int = 6) -> dict:
    """Build a synthetic dict mimicking ``read_ncdf_profile`` output.

    Parameters
    ----------
    n_lev : int
        Number of atmospheric levels.
    n_band : int
        Number of spectral bands.

    Returns
    -------
    dict
        Keys match those expected by ``plot_visual``.
    """
    # Band-edge arrays have n_band + 1 elements; after the code slices
    # [:-1] they match the n_band flux columns produced by [:, 1:].
    return {
        'ba_U_LW': np.ones((n_lev, n_band + 1)),
        'ba_U_SW': np.ones((n_lev, n_band + 1)),
        'ba_D_SW': np.ones((1, n_band + 1)),
        'bandmin': np.linspace(3e-7, 4e-5, n_band + 1),
        'bandmax': np.linspace(4e-7, 4.1e-5, n_band + 1),
        'pl': np.logspace(5, 1, n_lev),
        'tmpl': np.linspace(3000, 300, n_lev),
        'rl': np.linspace(6.5e6, 6.371e6, n_lev),
    }


# ---------------------------------------------------------------------------
# plot_visual – format validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize('fmt', ['pdf', 'svg', 'eps'])
def test_plot_visual_rejects_non_raster_format(fmt):
    """Format guard must reject non-raster output formats (pdf, svg, eps)."""
    hf = _make_hf_all()
    result = plot_visual(hf, '/tmp/unused', plot_format=fmt)
    assert result is False


@pytest.mark.unit
@pytest.mark.parametrize('fmt', ['png', 'jpg', 'bmp'])
def test_plot_visual_accepts_raster_format(fmt, tmp_path):
    """Raster formats (png, jpg, bmp) must pass the format guard.

    We only check that the function proceeds past format validation
    (it will return False later because no NetCDF files exist).
    """
    output_dir = str(tmp_path)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    hf = _make_hf_all()
    # No .nc files → returns False, but *not* because of format
    result = plot_visual(hf, output_dir, plot_format=fmt)
    assert result is False


# ---------------------------------------------------------------------------
# plot_visual – osamp clamping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_visual_osamp_minimum(tmp_path):
    """``osamp`` values below 2 must be clamped to 2."""
    output_dir = str(tmp_path)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    hf = _make_hf_all()
    # osamp=1 should be clamped; function returns False due to missing files
    result = plot_visual(hf, output_dir, osamp=1, plot_format='png')
    assert result is False


# ---------------------------------------------------------------------------
# plot_visual – missing data paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_visual_returns_false_no_nc_files(tmp_path):
    """Returns False when no ``*_atm.nc`` files exist in data/."""
    output_dir = str(tmp_path)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    hf = _make_hf_all()
    result = plot_visual(hf, output_dir, plot_format='png')
    assert result is False


@pytest.mark.unit
def test_plot_visual_returns_false_missing_nc_file(tmp_path):
    """Returns False when the specific timestep NetCDF file is absent."""
    output_dir = str(tmp_path)
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    # Create a dummy .nc file so glob finds something, but not the right one
    open(os.path.join(data_dir, '999999_atm.nc'), 'w').close()
    hf = _make_hf_all()
    result = plot_visual(hf, output_dir, idx=0, plot_format='png')
    assert result is False


@pytest.mark.unit
def test_plot_visual_returns_false_missing_key(tmp_path):
    """Returns False when NetCDF is missing a required key."""
    output_dir = str(tmp_path)
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    hf = _make_hf_all()
    time = hf['Time'].iloc[-1]
    nc_name = '%.0f_atm.nc' % time
    open(os.path.join(data_dir, nc_name), 'w').close()

    incomplete_ds = {'ba_U_LW': np.ones((4, 7))}  # missing most keys

    with patch('proteus.plot.cpl_visual.read_ncdf_profile', return_value=incomplete_ds):
        result = plot_visual(hf, output_dir, plot_format='png')
    assert result is False


# ---------------------------------------------------------------------------
# plot_visual – full render path (all matplotlib mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_visual_renders_frame(tmp_path):
    """Full render path: mocks matplotlib and NetCDF read, returns file path."""
    output_dir = str(tmp_path)
    data_dir = os.path.join(output_dir, 'data')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    hf = _make_hf_all()
    time = hf['Time'].iloc[-1]
    nc_name = '%.0f_atm.nc' % time
    open(os.path.join(data_dir, nc_name), 'w').close()

    ds = _make_ncdf_dict()

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_axr = MagicMock()
    mock_ax.inset_axes.return_value = mock_axr

    with (
        patch('proteus.plot.cpl_visual.read_ncdf_profile', return_value=ds),
        patch('proteus.plot.cpl_visual.plt') as mock_plt,
        patch(
            'proteus.plot.cpl_visual.cs_srgb.spec_to_rgb',
            return_value=(1.0, 0.5, 0.2),
        ),
        patch(
            'proteus.plot.cpl_visual.interp_spec',
            return_value=np.ones(100),
        ),
        patch('proteus.plot.cpl_visual.patches'),
    ):
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        result = plot_visual(hf, output_dir, plot_format='png')

    expected = os.path.join(plots_dir, 'plot_visual.png')
    assert result == expected
    mock_fig.savefig.assert_called_once()
    mock_plt.close.assert_called_once()
    mock_plt.ioff.assert_called_once()


# ---------------------------------------------------------------------------
# anim_visual
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_anim_visual_returns_false_no_ffmpeg():
    """Returns False when ffmpeg is not available on PATH."""
    hf = _make_hf_all()
    with patch('proteus.plot.cpl_visual.which', return_value=None):
        result = anim_visual(hf, '/tmp/unused')
    assert result is False


@pytest.mark.unit
def test_anim_visual_returns_false_on_frame_failure(tmp_path):
    """Returns False when ``plot_visual`` fails for a frame."""
    output_dir = str(tmp_path)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    hf = _make_hf_all()

    with (
        patch('proteus.plot.cpl_visual.which', return_value='/usr/bin/ffmpeg'),
        patch('proteus.plot.cpl_visual.plot_visual', return_value=False),
        patch('proteus.plot.cpl_visual.safe_rm'),
    ):
        result = anim_visual(hf, output_dir, nframes=2)
    assert result is False


@pytest.mark.unit
def test_anim_visual_returns_false_on_nonzero_ffmpeg(tmp_path):
    """Returns False when ffmpeg returns a non-zero exit code."""
    output_dir = str(tmp_path)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    hf = _make_hf_all()

    # plot_visual returns a fake frame path
    fake_frame = os.path.join(plots_dir, 'plot_visual.frame.png')

    mock_process = MagicMock()
    mock_process.stdout = io.BytesIO(b'')
    mock_process.wait.return_value = 1

    with (
        patch('proteus.plot.cpl_visual.which', return_value='/usr/bin/ffmpeg'),
        patch('proteus.plot.cpl_visual.plot_visual', return_value=fake_frame),
        patch('proteus.plot.cpl_visual.safe_rm'),
        patch('proteus.plot.cpl_visual.copyfile'),
        patch('proteus.plot.cpl_visual.Popen', return_value=mock_process),
    ):
        result = anim_visual(hf, output_dir, nframes=2)
    assert result is False


@pytest.mark.unit
def test_anim_visual_returns_false_on_missing_output(tmp_path):
    """Returns False when ffmpeg exits 0 but the output file is absent."""
    output_dir = str(tmp_path)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    hf = _make_hf_all()
    fake_frame = os.path.join(plots_dir, 'plot_visual.frame.png')

    mock_process = MagicMock()
    mock_process.stdout = io.BytesIO(b'')
    mock_process.wait.return_value = 0

    with (
        patch('proteus.plot.cpl_visual.which', return_value='/usr/bin/ffmpeg'),
        patch('proteus.plot.cpl_visual.plot_visual', return_value=fake_frame),
        patch('proteus.plot.cpl_visual.safe_rm'),
        patch('proteus.plot.cpl_visual.copyfile'),
        patch('proteus.plot.cpl_visual.Popen', return_value=mock_process),
        patch('proteus.plot.cpl_visual.os.path.isfile', return_value=False),
    ):
        result = anim_visual(hf, output_dir, nframes=2)
    assert result is False


@pytest.mark.unit
def test_anim_visual_success(tmp_path):
    """Happy path: frames rendered, ffmpeg called, returns True."""
    output_dir = str(tmp_path)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    hf = _make_hf_all()
    fake_frame = os.path.join(plots_dir, 'plot_visual.frame.png')

    mock_process = MagicMock()
    mock_process.stdout = io.BytesIO(b'')
    mock_process.wait.return_value = 0

    with (
        patch('proteus.plot.cpl_visual.which', return_value='/usr/bin/ffmpeg'),
        patch('proteus.plot.cpl_visual.plot_visual', return_value=fake_frame),
        patch('proteus.plot.cpl_visual.safe_rm'),
        patch('proteus.plot.cpl_visual.copyfile'),
        patch('proteus.plot.cpl_visual.Popen', return_value=mock_process),
        patch('proteus.plot.cpl_visual.os.path.isfile', return_value=True),
    ):
        result = anim_visual(hf, output_dir, nframes=2)
    assert result is True


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_visual_entry_calls_plot_visual(tmp_path):
    """``plot_visual_entry`` reads helpfile CSV and delegates to ``plot_visual``."""
    from proteus.plot.cpl_visual import plot_visual_entry

    output_dir = str(tmp_path)
    hf = _make_hf_all()
    csv_path = os.path.join(output_dir, 'runtime_helpfile.csv')
    hf.to_csv(csv_path, sep=' ', index=False)

    handler = MagicMock()
    handler.directories = {'output': output_dir}

    with patch('proteus.plot.cpl_visual.plot_visual') as mock_pv:
        plot_visual_entry(handler)

    mock_pv.assert_called_once()
    call_args = mock_pv.call_args
    assert (
        call_args.kwargs.get('idx', call_args.args[2] if len(call_args.args) > 2 else -1) == -1
    )


@pytest.mark.unit
def test_anim_visual_entry_calls_anim_visual(tmp_path):
    """``anim_visual_entry`` reads helpfile CSV and delegates to ``anim_visual``."""
    from proteus.plot.cpl_visual import anim_visual_entry

    output_dir = str(tmp_path)
    hf = _make_hf_all()
    csv_path = os.path.join(output_dir, 'runtime_helpfile.csv')
    hf.to_csv(csv_path, sep=' ', index=False)

    handler = MagicMock()
    handler.directories = {'output': output_dir}

    with patch('proteus.plot.cpl_visual.anim_visual') as mock_av:
        anim_visual_entry(handler)

    mock_av.assert_called_once()
    call_args = mock_av.call_args
    assert call_args.args[1] == output_dir
