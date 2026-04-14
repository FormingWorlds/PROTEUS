from __future__ import annotations

import glob
import logging
import os
from shutil import copyfile, which
from subprocess import PIPE, STDOUT, Popen
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches, ticker

from proteus.atmos_clim.common import read_ncdf_profile
from proteus.utils.constants import R_earth
from proteus.utils.helper import safe_rm
from proteus.utils.visual import cs_srgb, interp_spec

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger('fwl.' + __name__)


def plot_visual(
    hf_all: pd.DataFrame,
    output_dir: str,
    idx: int = -1,
    osamp: int = 3,
    view: float = 12.5,
    plot_format: str = 'png',
):
    """Render a visual snapshot of the planet-star system.

    Generates a single frame visualising the planet surface color (from the
    band-integrated upwelling flux) and surrounding atmospheric shells, along
    with the star and an inset spectrum derived from NetCDF outputs.

    Parameters
    ----------
    hf_all : pandas.DataFrame
        Runtime helpfile table used to select the time step and metadata
    output_dir : str
        Path to the run's output directory containing `data/` and `plots/`.

    idx : int, optional
        Row index into `hf_all` and the sorted NetCDF files.
    osamp : int, optional
        Radial oversampling factor for rendering outer atmospheric levels.
        Minimum of 2. Defaults to 3.
    view : float, optional
        Observer distance in units of planetary radii (`R_int * view`).
    plot_format : str, optional
        Raster format for the saved figure (e.g., 'jpg', 'png').

    Returns
    -------
    str or bool
        Path to the saved figure on success; False if required data are missing.
    """
    log.info('Plot visual')

    # Check frame format
    if not np.any(
        [
            plot_format.endswith(ext)
            for ext in (
                'png',
                'jpg',
                'bmp',
            )
        ]
    ):
        log.error(f"Visualisation must use raster format; got '{plot_format}'")
        return False

    osamp = max(osamp, 2)

    # Orbital separation
    sep = float(hf_all['separation'].iloc[idx])

    # Set viewing distance
    R_int = float(hf_all['R_int'].iloc[idx])
    obs = R_int * view

    # Get time at this index, and path to NetCDF file
    time = hf_all['Time'].iloc[idx]
    files = glob.glob(os.path.join(output_dir, 'data', '*_atm.nc'))
    if len(files) == 0:
        log.warning('No atmosphere NetCDF files found in output folder')
        if os.path.exists(os.path.join(output_dir, 'data', 'data.tar')):
            log.warning('You may need to extract archived data files')
        return False

    fpath = os.path.join(output_dir, 'data', '%.0f_atm.nc' % time)
    if not os.path.exists(fpath):
        log.warning(f'Cannot find file {fpath}')
        if os.path.exists(os.path.join(output_dir, 'data', 'data.tar')):
            log.warning('You may need to extract archived data files')
        return False

    # Read data
    keys = ['ba_U_LW', 'ba_U_SW', 'ba_D_SW', 'bandmin', 'bandmax', 'pl', 'tmpl', 'rl']
    ds = read_ncdf_profile(fpath, extra_keys=keys)

    # Check that we have all the keys
    for k in keys:
        if k not in ds.keys():
            log.error(f"Could not read key '{k}' from NetCDF file")
            return False

    scale = 1.7
    fig, ax = plt.subplots(1, 1, figsize=(4 * scale, 4 * scale))

    # read fluxes
    sw_arr = np.array(ds['ba_U_SW'][:, 1:])
    lw_arr = np.array(ds['ba_U_LW'][:, 1:])
    st_arr = np.array(ds['ba_D_SW'][0, 1:])

    # reversed?
    reversed = bool(ds['bandmin'][1] < ds['bandmin'][0])
    if reversed:
        bandmin = np.array(ds['bandmin'][::-1])
        bandmax = np.array(ds['bandmax'][::-1])
        sw_arr = sw_arr[:, ::-1]
        lw_arr = lw_arr[:, ::-1]
        st_arr = st_arr[::-1]
    else:
        bandmin = np.array(ds['bandmin'][:])
        bandmax = np.array(ds['bandmax'][:])
    bandmin = bandmin[:-1]
    bandmax = bandmax[:-1]

    # get spectrum
    wl = 0.5 * (bandmin + bandmax) * 1e9  # nm
    wd = (bandmax - bandmin) * 1e9  # nm
    st = st_arr / wd
    sw = sw_arr / wd
    lw = lw_arr / wd

    # radii
    r_arr = ds['rl'] / obs
    r_min = np.amin(r_arr)
    r_lim = 0.2
    n_lev = len(r_arr)

    # pressures
    p_arr = ds['pl']
    p_max = np.amax(p_arr)

    # plot base layer
    srf = patches.Circle((0, 0), radius=r_min, fc='#492410', zorder=8)
    ax.add_patch(srf)

    # plot surface of planet
    fl_srf = lw[-1, :] + sw[-1, :]
    col = cs_srgb.spec_to_rgb(interp_spec(wl, fl_srf))
    srf = patches.Circle((0, 0), radius=r_min, fc=col, zorder=n_lev + 1, alpha=0.9)
    ax.add_patch(srf)

    # level opacities
    gamma = 0.08
    a_arr = []
    for i, p in enumerate(p_arr):
        alp = p / p_max
        a_arr.append(alp**gamma)
    a_arr /= sum(a_arr)
    a_arr *= 0.90

    # plot outer levels
    for i in range(n_lev - 2, -1, -1):
        sw_lev = sw[i + 1, :] - sw[i, :]
        lw_lev = lw[i + 1, :] - lw[i, :]

        rad_c = r_arr[i]
        rad_l = r_arr[i + 1]

        spec = interp_spec(wl, sw_lev + lw_lev)
        col = cs_srgb.spec_to_rgb(spec)

        for rad in np.linspace(rad_c, rad_l, osamp):
            cir = patches.Circle((0, 0), radius=rad, fc=col, alpha=a_arr[i], zorder=3 + i)
            ax.add_patch(cir)

    # annotate planet
    ax.text(
        0,
        0.2 * R_int / obs,
        r'T$_\text{s}=$%.0f K' % ds['tmpl'][-1],
        color='white',
        fontsize=11,
        ha='center',
        va='bottom',
        zorder=999,
    )

    # annotate time and distance
    ann = r'Viewing from %.1f R$_\oplus$' % (obs / R_earth) + ' at %6.1f Myr' % (time / 1e6)
    ax.text(
        0.01,
        0.99,
        ann,
        color='white',
        fontsize=11,
        zorder=999,
        transform=ax.transAxes,
        ha='left',
        va='top',
    )

    # plot star
    col = cs_srgb.spec_to_rgb(interp_spec(wl, st))
    r_star = hf_all['R_star'].iloc[idx] / (sep + obs)
    x_star = r_lim * 0.75
    cir = patches.Circle(
        (x_star, x_star),
        radius=r_star,
        fc=col,
        zorder=2,
    )
    ax.add_patch(cir)
    ax.text(
        x_star,
        x_star - r_star,
        'Star',
        color='white',
        fontsize=11,
        ha='center',
        va='top',
        zorder=999,
    )

    # scale bar
    for r in np.arange(0, 20, 1):
        x = r * R_earth / obs / 2**0.5
        if abs(x) > r_lim:
            break
        ax.scatter(x, -x, s=20, color='w', zorder=999)
        if r > 0:
            ax.text(
                x,
                -x,
                r'  %.0f R$_\oplus$' % r,
                ha='left',
                va='center',
                fontsize=8,
                color='w',
                zorder=999,
            )
    ax.plot([0, x], [0, -x], lw=1, color='w', zorder=99)

    # decorate
    ax.set_facecolor('k')
    ax.set_xlim(-r_lim, r_lim)
    ax.set_ylim(-r_lim, r_lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # inset spectrum
    axr = ax.inset_axes((0.07, 0.04, 0.39, 0.21))
    axr.set_alpha(0.0)
    axr.set_facecolor((0, 0, 0, 0))
    #    crop to wavelength region
    fl = lw[0, :] + sw[0, :]
    wl = wl[:]  # nm
    wd = wd[:]  # nm

    #   plot and decorate
    axr.step(wl / 1e3, fl, where='mid', color='w', lw=1.3)
    axr.spines[['bottom', 'left']].set_color('w')
    axr.spines[['right', 'top']].set_visible(False)
    axr.tick_params(axis='both', colors='w', labelsize=8)

    axr.set_xlabel(r'$/\mu$m', color='w', fontsize=8)
    axr.xaxis.set_label_coords(1.12, -0.08)
    axr.set_xlim(left=0.3, right=40)
    axr.set_xscale('log')
    axr.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    axr.xaxis.set_minor_locator(ticker.LogLocator(numticks=1000))

    axr.set_yscale('log')
    axr.set_ylim(bottom=max(1e-10, np.amin(fl)))
    axr.set_ylabel(r'W/m$^2$/nm', color='w', fontsize=8, rotation=0)
    axr.yaxis.set_label_coords(0.01, 1.02)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, 'plots', 'plot_visual.%s' % plot_format)
    fig.savefig(fpath, dpi=250, bbox_inches='tight')

    return fpath


def plot_visual_entry(handler: Proteus):
    """Entry point to render a single visual frame."""
    # read helpfile
    hf_all = pd.read_csv(
        os.path.join(handler.directories['output'], 'runtime_helpfile.csv'), sep=r'\s+'
    )

    plot_visual(
        hf_all,
        handler.directories['output'],
        idx=-1,
    )


def anim_visual(
    hf_all: pd.DataFrame, output_dir: str, duration: float = 12.0, nframes: int = 80
):
    """Create an MP4 animation from visual frames.

    Renders a sequence of frames using `plot_visual` and assembles them into
    an animation via `ffmpeg`. Frame number can be downsampled to speed
    up rendering process. Requires `ffmpeg` to be available on PATH.

    Parameters
    ----------
    hf_all : pandas.DataFrame
        Runtime helpfile table used to select time steps and metadata.
    output_dir : str
        Path to the run's output directory.
    duration : float, optional
        Animation duration in seconds.
    nframes : int, optional
        Number of frames in animation.

    Returns
    -------
    bool
        Returns False on failure
    """

    # Animation options
    frame_fmt = 'frame.png'
    video_fmt = 'mp4'
    codec = 'h264'

    # check ffmpeg is installed
    ffmpeg = which('ffmpeg')
    if not ffmpeg:
        log.error('Program `ffmpeg` not found; cannot make animation')
        return False

    # make frames folder (safe if it already exists)
    framesdir = os.path.join(output_dir, 'plots', 'anim_frames')
    safe_rm(framesdir)
    os.makedirs(framesdir)

    # Work out downsampling factor
    niters = len(hf_all)
    log.info(f'Found dataframe with {niters} iterations')

    # For each index...
    target_times = np.linspace(0, np.amax(hf_all['Time']), nframes)
    idxs = [np.argmin(np.abs(t - hf_all['Time'].values)) for t in target_times]
    fps = max(1, nframes / duration)
    log.info(f'Will make {nframes} frames')
    for i, idx in enumerate(idxs):
        idx = max(0, min(idx, niters - 1))

        log.info(f'Plotting iteration {idx:<5d} (frame {i + 1:3d} / {nframes:<3d})')

        # plot the frame
        fpath = plot_visual(hf_all, output_dir, plot_format=frame_fmt, idx=idx)
        if not fpath:
            return False

        # move frame to subfolder
        copyfile(fpath, os.path.join(framesdir, f'{idx:05d}.{frame_fmt}'))
        safe_rm(fpath)

    # Path to animation video
    out_video = os.path.join(output_dir, 'plots', f'anim_visual.{video_fmt}')
    safe_rm(out_video)

    # ffmpeg input pattern: frames named 0.<ext>, 1.<ext>, ...
    input_pattern = os.path.join(framesdir, f'*.{frame_fmt}')

    # Command for subprocess
    cmd = [
        ffmpeg,
        '-y',
        f'-framerate {fps:.2f}',
        '-pattern_type glob',
        f"-i '{input_pattern}'",
        f'-c:v {codec}',
        "-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2'",
        '-pix_fmt yuv420p',
        f' {out_video}',
    ]
    cmd = ' '.join(cmd)

    # Wrapper for logging
    def _log_subprocess_output(pipe):
        for line in iter(pipe.readline, b''):  # b'\n'-separated lines
            log.debug(str(line.decode('utf-8')).replace('\n', '').strip())

    # Run ffmpeg to make animation
    log.info(f'Running ffmpeg to assemble video: {cmd}')
    try:
        process = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
        with process.stdout:
            _log_subprocess_output(process.stdout)
        ret = process.wait()
        if ret == 0:
            if os.path.isfile(out_video):
                log.info(f'Wrote animation to {out_video}')
            else:
                log.error('ffmpeg exited successfully but animation file not found')
                return False
        else:
            log.error(f'ffmpeg returned non-zero exit code: {ret}')
            return False

    except FileNotFoundError:
        log.error('Program `ffmpeg` not found in PATH; cannot make animation')
        return False

    except Exception as e:
        log.error(f'Error running ffmpeg: {e}')
        return False

    # Remove frames directory
    safe_rm(framesdir)

    return True


def anim_visual_entry(handler: Proteus):
    """Entry point to generate a visual animation.

    Loads the runtime helpfile from the handler's output directory and calls
    `anim_visual` to render frames and assemble the MP4 animation.

    Parameters
    ----------
    handler : Proteus
        Active run handler providing `directories`.

    Returns
    -------
    None
        This function triggers animation assembly and does not return a value.
    """

    # read helpfile
    hf_all = pd.read_csv(
        os.path.join(handler.directories['output'], 'runtime_helpfile.csv'), sep=r'\s+'
    )

    anim_visual(
        hf_all,
        handler.directories['output'],
    )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv

    handler = get_handler_from_argv()
    plot_visual_entry(handler)
