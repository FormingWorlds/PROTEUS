from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from proteus.atmos_clim.common import read_atmosphere_data
from proteus.interior_energetics.wrapper import read_interior_data
from proteus.utils.constants import R_earth
from proteus.utils.plot import get_colour, latex_float, profile_sample_tmin, sample_times

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger('fwl.' + __name__)


def _interior_profile(ds, module: str) -> tuple[np.ndarray, np.ndarray]:
    """Radius [R_earth] and temperature [K] of an interior snapshot."""
    if module == 'aragog':
        # Entropy solver uses staggered variables (temp_s, radius_s) [K, km]
        key = 's' if 'temp_s' in ds else 'b'
        int_t = np.array(ds['temp_' + key][::-1], dtype=float)
        int_r = np.array(ds['radius_' + key][::-1], dtype=float) * 1e3
    else:
        int_t = np.array(ds.get_dict_values(['data', 'temp_b']), dtype=float)
        int_r = np.array(ds.get_dict_values(['data', 'radius_b']), dtype=float)
    return int_r / R_earth, int_t


def _has_skin(hf_all: pd.DataFrame, idx: int) -> bool:
    """Whether a helpfile row holds a conductive skin: T_surf below T_magma, and a
    positive boundary_layer_conductivity where the helpfile has that column."""
    if 'T_surf' not in hf_all.columns or 'T_magma' not in hf_all.columns:
        return False
    row = hf_all.iloc[idx]
    if 'boundary_layer_conductivity' in hf_all.columns:
        if not float(row['boundary_layer_conductivity']) > 0:
            return False
    return float(row['T_surf']) < float(row['T_magma'])


def plot_structure(
    hf_all: pd.DataFrame,
    output_dir: str,
    times: list,
    int_data: list,
    atm_data: list,
    module: str,
    plot_format: str = 'pdf',
):
    if (len(times) < 2) or (np.amax(times) < 2):
        log.debug('Insufficient data to make plot_structure')
        return

    if module not in ('spider', 'aragog'):
        log.debug("Cannot make structure plot with interior module '%s'" % module)
        return

    # A missing snapshot would pair a profile with the wrong time
    if atm_data is None or len(atm_data) != len(times) or len(int_data) != len(times):
        log.warning('Snapshot data do not match the plot times; skipping plot_structure')
        return

    log.info('Plot structure')

    # helpfile indicies
    hf_idxs = []
    hf_times = np.array(hf_all['Time'], dtype=float)
    for time in times:
        hf_idxs.append(np.argmin(np.abs(hf_times - time)))

    # helpfile data
    R_int = np.array(hf_all['R_int'], dtype=float)[hf_idxs]
    R_obs = np.array(hf_all['R_obs'], dtype=float)[hf_idxs]
    R_xuv = np.array(hf_all['R_xuv'], dtype=float)[hf_idxs]

    # scale to Earth radius
    for r in (R_int, R_obs, R_xuv):
        r /= R_earth

    scale = 0.9
    fig, ax = plt.subplots(1, 1, figsize=(10 * scale, 5 * scale))

    norm = mpl.colors.LogNorm(vmin=max(1, times[0]), vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    # limits
    r_max = 0.2
    t_max = 1000
    lw = 1.5
    ms = 40
    esc_m = '|'
    obs_m = '.'

    # loop over times
    any_skin = False
    for i, time in enumerate(times):
        # Get atmosphere data for this time
        prof = atm_data[i]
        atm_r = prof['z'] / R_earth + R_int[i]
        atm_t = prof['t']

        # Get temperature-depth interior data for this time
        int_r, int_t = _interior_profile(int_data[i], module)

        # Plot decorators
        label = latex_float(time) + ' yr'
        color = sm.to_rgba(time)

        # Plot atmosphere
        ax.plot(atm_r, atm_t, color=color, label=label, lw=lw, zorder=i)

        # Observed radius
        i_obs = np.argmin(np.abs(atm_r - R_obs[i]))
        ax.scatter(R_obs[i], atm_t[i_obs], color=color, s=ms, marker=obs_m, zorder=i)

        # Escape radius
        i_xuv = np.argmin(np.abs(atm_r - R_xuv[i]))
        ax.scatter(R_xuv[i], atm_t[i_xuv], color=color, s=ms, marker=esc_m, zorder=i)

        # Plot mantle
        R_mantle = np.amax(int_r)
        ax.plot(int_r, int_t, linestyle='solid', color=color, lw=lw, zorder=i)

        # Join the top of the mantle to the base of the atmosphere; dashed across a skin
        j_top, j_base = int(np.argmax(int_r)), int(np.argmin(atm_r))
        skin = _has_skin(hf_all, hf_idxs[i])
        any_skin = any_skin or skin
        ax.plot(
            [int_r[j_top], atm_r[j_base]],
            [int_t[j_top], atm_t[j_base]],
            linestyle='dashed' if skin else 'solid',
            color=color,
            lw=lw,
            zorder=i,
        )

        # Plot core
        R_core = np.amin(int_r)
        ax.plot([0, R_core], [int_t[-1], int_t[-1]], color=color, lw=lw)

        # update limits
        r_max = max(r_max, np.amax(atm_r))
        t_max = max(t_max, np.amax(int_t))
        t_max = max(t_max, np.amax(atm_t))

    r_max *= 1.05
    t_max *= 1.05

    # Decorate plot
    ax.set(xlabel=r'Radius [R$_{\oplus}$]', ylabel='Temperature [K]')
    # ax.legend()
    ax.grid(zorder=-2, alpha=0.1)
    leg = ax.legend(framealpha=1.0, loc='lower right')
    leg.set_zorder(99)
    ax.add_artist(leg)

    ax.xaxis.set_major_locator(MultipleLocator(0.20))
    ax.xaxis.set_minor_locator(MultipleLocator(0.10))
    ticks = [t for t in ax.get_xticks() if t >= 0]
    ax.set_xticks(ticks, ['%.2f' % t for t in ticks], rotation=-45, ha='center')
    ax.set_xlim(left=0.0, right=r_max)

    ax.yaxis.set_minor_locator(MultipleLocator(500))
    ax.set_ylim(top=0.0, bottom=t_max)

    # Filled regions
    ax.fill_betweenx([0, 1e20], x1=0, x2=R_core, color=get_colour('cor_bkg'), zorder=-3)
    ax.fill_betweenx([0, 1e20], x1=R_core, x2=R_mantle, color=get_colour('int_bkg'), zorder=-3)
    ax.fill_betweenx([0, 1e20], x1=R_mantle, x2=r_max, color=get_colour('atm_bkg'), zorder=-3)

    # Line style information
    hdls = []
    hdls.append(ax.scatter(-100, 0, c='k', s=ms, marker=esc_m, label='Escape level'))
    hdls.append(ax.scatter(-100, 0, c='k', s=ms, marker=obs_m, label='Observed level'))
    if any_skin:
        hdls.append(
            Line2D([], [], color='k', lw=lw, linestyle='dashed', label='Conductive skin')
        )
    ax.legend(handles=hdls, loc='lower center', framealpha=1.0).set_zorder(99)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, 'plots', 'plot_structure.%s' % plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def structure_plot_times(output_dir: str, int_module: str, nsamp: int = 8) -> list:
    """Sample the times [yr] that have both an atmosphere and an interior snapshot."""
    from proteus.atmos_clim.common import list_atmosphere_times
    from proteus.utils.coupler import select_profile_plot_times

    if int_module == 'spider':
        from proteus.interior_energetics.spider import get_all_output_times
    elif int_module == 'aragog':
        from proteus.interior_energetics.aragog import get_all_output_times
    else:
        return []
    times = select_profile_plot_times(
        list(get_all_output_times(output_dir)),
        list_atmosphere_times(output_dir),
        no_int_snapshots=False,
        interior_rounds_to_year=int_module == 'spider',
    )
    if not times:
        return []
    plot_times, _ = sample_times(times, nsamp, tmin=profile_sample_tmin(times))
    return plot_times


def plot_structure_entry(handler: Proteus):
    int_module = handler.config.interior_energetics.module
    output_dir = handler.directories['output']
    plot_times = structure_plot_times(output_dir, int_module)
    log.info('Snapshots: %s', plot_times)

    # read helpfile
    hf_all = pd.read_csv(os.path.join(output_dir, 'runtime_helpfile.csv'), sep=r'\s+')

    int_data = read_interior_data(output_dir, int_module, plot_times)
    atm_data = read_atmosphere_data(output_dir, plot_times)

    plot_structure(
        hf_all,
        output_dir,
        plot_times,
        int_data,
        atm_data,
        int_module,
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv

    handler = get_handler_from_argv()
    plot_structure_entry(handler)
