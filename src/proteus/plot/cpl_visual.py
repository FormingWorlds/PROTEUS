from __future__ import annotations

import glob
import logging
import os
from shutil import copyfile, rmtree
from subprocess import call
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches, ticker

from proteus.atmos_clim.common import read_ncdf_profile
from proteus.utils.constants import R_earth
from proteus.utils.visual import cs_srgb as colsys
from proteus.utils.visual import interp_spec

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger('fwl.' + __name__)


def plot_visual(
    hf_all: pd.DataFrame,
    output_dir: str,
    idx: int = -1,
    osamp: int = 3,
    view: float = 12.5,
    plot_format: str = 'pdf',
):
    """Render a visual snapshot of the planet–star system.

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
        Image format for the saved figure (e.g., 'pdf', 'png').

    Returns
    -------
    str or bool
        Path to the saved figure on success; False if required data are missing.
    """
    log.info('Plot visual')

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
    sw_arr = np.array(ds['ba_U_SW'][:, :])
    lw_arr = np.array(ds['ba_U_LW'][:, :])
    st_arr = np.array(ds['ba_D_SW'][0, :])

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

    # get spectrum
    wl = 0.5 * (bandmin + bandmax) * 1e9
    wd = (bandmax - bandmin) * 1e9
    st = st_arr
    sw = sw_arr
    lw = lw_arr

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
    col = colsys.spec_to_rgb(interp_spec(wl, fl_srf))
    srf = patches.Circle((0, 0), radius=r_min, fc=col, zorder=n_lev + 1, alpha=0.2)
    ax.add_patch(srf)

    # level opacities
    gamma = 0.12
    a_arr = []
    for i, p in enumerate(p_arr):
        alp = p / p_max
        a_arr.append(alp**gamma)
    a_arr /= sum(a_arr)
    a_arr *= 0.99

    # plot outer levels
    for i in range(n_lev - 2, -1, -1):
        sw_lev = sw[i + 1, :] - sw[i, :]
        lw_lev = lw[i + 1, :] - lw[i, :]

        rad_c = r_arr[i]
        rad_l = r_arr[i + 1]

        spec = interp_spec(wl, sw_lev + lw_lev)
        col = colsys.spec_to_rgb(spec)

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
    col = colsys.spec_to_rgb(interp_spec(wl, st))
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
    imax = np.argmin(np.abs(wl - 6e3))
    fl = lw[0, :imax] + sw[0, :imax]
    wl = wl[:imax]
    wd = wd[:imax]
    fl = fl / wd

    #    flux units
    if np.amax(fl) < 1:
        fl *= 1e3
        un = 'mW'
    elif np.amax(fl) > 1e3:
        fl /= 1e3
        un = 'kw'
    else:
        un = 'W'

    #   plot and decorate
    axr.bar(wl, fl, width=wd, color='w', lw=1.3)
    axr.spines[['bottom', 'left']].set_color('w')
    axr.spines[['right', 'top']].set_visible(False)
    axr.tick_params(axis='both', colors='w', labelsize=8)
    axr.set_xlabel(r'$/$nm', color='w', fontsize=8)
    axr.xaxis.set_label_coords(1.12, -0.08)
    axr.set_ylabel(r'%s/m$^2$/nm' % un, color='w', fontsize=8, rotation=0)
    axr.yaxis.set_label_coords(0.01, 1.02)

    axr.set_xlim(left=0)
    axr.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    axr.xaxis.set_minor_locator(ticker.MultipleLocator(200))
    axr.set_ylim(bottom=0)
    axr.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

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
        plot_format=handler.config.params.out.plot_fmt,
        idx=-1,
    )


def anim_visual(
    hf_all: pd.DataFrame, output_dir: str, duration: float = 8.0, nframes: int = 80
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

    # make frames folder (safe if it already exists)
    framesdir = os.path.join(output_dir, 'plots', 'anim_frames')
    if os.path.isdir(framesdir):
        rmtree(framesdir)
    os.makedirs(framesdir)

    # Must be raster format
    plot_fmt = 'png'

    # Work out downsampling factor
    niters = len(hf_all)
    print(f'Found dataframe with {niters} iterations')
    ds_factor = int(max(1, np.floor(niters / nframes)))
    print(f'Downsampling iterations by a factor of {ds_factor}')

    # For each index...
    idxs = range(0, niters, ds_factor)
    nframes = len(idxs)
    fps = int(max(1, round(nframes / duration)))
    for i, idx in enumerate(idxs):
        idx = max(0, min(idx, niters - 1))

        print(f'Plotting iteration {idx:5d} (frame {i + 1} / {nframes})')

        fpath = plot_visual(hf_all, output_dir, plot_format=plot_fmt, idx=idx)

        if not fpath:
            return False

        copyfile(fpath, os.path.join(framesdir, f'{idx:05d}.{plot_fmt}'))

    # Make animation
    out_video = os.path.join(output_dir, 'plots', 'anim_visual.mp4')

    # ffmpeg input pattern: frames named 0.<ext>, 1.<ext>, ...
    input_pattern = os.path.join(framesdir, f'*.{plot_fmt}')

    cmd = [
        'ffmpeg',
        '-y',
        f'-framerate {fps:d}',
        '-pattern_type glob',
        f"-i '{input_pattern}'",
        '-c:v libx264',
        "-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2'",
        '-pix_fmt yuv420p',
        f' {out_video}',
    ]

    cmd = ' '.join(cmd)
    print(f'Running ffmpeg to assemble video: {cmd}')
    try:
        ret = call([cmd], shell=True, stdout=None)
        if ret == 0:
            log.info(f'Wrote animation to {out_video}')
        else:
            log.error(f'ffmpeg returned non-zero exit code: {ret}')
    except FileNotFoundError:
        log.error('ffmpeg not found on PATH; cannot assemble animation')
    except Exception as e:
        log.error(f'Error running ffmpeg: {e}')

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
