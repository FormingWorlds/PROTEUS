from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm
from matplotlib.ticker import MultipleLocator

from proteus.atmos_clim.common import read_atmosphere_data
from proteus.interior.wrapper import read_interior_data
from proteus.utils.plot import get_colour, latex_float, sample_output
from proteus.utils.constants import R_earth

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_structure(hf_all: pd.DataFrame, output_dir: str,
                        times: list, int_data:list, atm_data:list,
                        module:str, plot_format: str="pdf"):

    if np.amax(times) < 2:
        log.debug("Insufficient data to make plot_structure")
        return

    log.info("Plot structure")

    # helpfile indicies
    hf_idxs = []
    hf_times = np.array(hf_all["Time"], dtype=float)
    for time in times:
        hf_idxs.append(np.argmin(np.abs(hf_times - time)))

    # helpfile data
    R_int = np.array(hf_all["R_int"], dtype=float)[hf_idxs]
    R_obs = np.array(hf_all["z_obs"], dtype=float)[hf_idxs] + R_int
    R_xuv = np.array(hf_all["z_xuv"], dtype=float)[hf_idxs] + R_int

    # scale to Earth radius
    for r in (R_int, R_obs, R_xuv):
        r /= R_earth

    scale = 0.9
    fig,ax = plt.subplots(1,1, figsize=(10*scale,5*scale))

    norm = mpl.colors.LogNorm(vmin=max(1,times[0]), vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    # limits
    r_max = 0.2
    t_max = 1000
    lw = 1.5
    ms=20
    esc_m = '|'
    obs_m = 'x'

    # loop over times
    for i,time in enumerate(times):

        # Get atmosphere data for this time
        prof = atm_data[i]
        atm_r = prof["z"]/R_earth + R_int[i]
        atm_t = prof["t"]

        # Get temperuture-depth interior data for this time
        ds = int_data[i]
        if module == "aragog":
            int_t = ds["temp_b"][::-1]
            int_r = ds["radius_b"][::-1] * 1e3
        elif module == "spider":
            int_t = ds.get_dict_values(['data','temp_b'])
            int_r = ds.get_dict_values(['data','radius_b'])
        int_r /= R_earth

        # Determine mixed phase region
        if module == "aragog":
            yy = np.array(ds["phi_b"])
            MASK_SO = yy < 0.05
            MASK_MI = (0.05 <= yy) & ( yy <= 0.95)
            MASK_ME = yy > 0.95
        elif module == "spider":
            MASK_MI = ds.get_mixed_phase_boolean_array('basic')
            MASK_ME = ds.get_melt_phase_boolean_array( 'basic')
            MASK_SO = ds.get_solid_phase_boolean_array('basic')

        # overlap lines by 1 node
        MASK_MI[2:-2] = MASK_MI[2:-2] | MASK_MI[1:-3] | MASK_MI[3:-1]

        # Plot decorators
        label = latex_float(time)+" yr"
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
        ax.plot(int_r[MASK_SO], int_t[MASK_SO],  linestyle='solid',  color=color, lw=lw, zorder=i)
        ax.plot(int_r[MASK_MI], int_t[MASK_MI],  linestyle='dashed', color=color, lw=lw, zorder=i)
        ax.plot(int_r[MASK_ME], int_t[MASK_ME],  linestyle='dotted', color=color, lw=lw, zorder=i)

        # Plot core
        R_core = np.amin(int_r)
        ax.plot([0,R_core],[int_t[-1], int_t[-1]], color=color, lw=lw)

        # update limits
        r_max = max(r_max, np.amax(atm_r))
        t_max = max(t_max, np.amax(int_t))
        t_max = max(t_max, np.amax(atm_t))


    r_max *= 1.05
    t_max *= 1.05

    # Decorate plot
    ax.set(xlabel=r"Radius [R$_{\oplus}$]", ylabel="Temperature [K]")
    ax.set_xlim(left=0.0, right=r_max)
    ax.set_ylim(top=0.0, bottom=t_max)
    ax.legend()
    ax.grid(zorder=-1, alpha=0.1)
    leg = ax.legend(framealpha=1.0, loc='upper left')
    ax.add_artist(leg)

    # Filled regions
    ax.fill_betweenx([0,1e20], x1=0,         x2=R_core,    color=get_colour("cor_bkg"), alpha=0.8, zorder=-3)
    ax.fill_betweenx([0,1e20], x1=R_core,    x2=R_mantle, color=get_colour("int_bkg"), alpha=0.8, zorder=-3)
    ax.fill_betweenx([0,1e20], x1=R_mantle,  x2=r_max,      color=get_colour("atm_bkg"), alpha=1, zorder=-3)

    # Line style information
    hdls = []
    # hdls.append(ax.plot([-100,-200],[0,100], ls='solid',  c='k', lw=lw, label="Solid")[0])
    # hdls.append(ax.plot([-100,-200],[0,100], ls='dashed', c='k', lw=lw, label="Mush")[0] )
    # hdls.append(ax.plot([-100,-200],[0,100], ls='dotted', c='k', lw=lw, label="Melt")[0] )
    hdls.append(ax.scatter(-100, 0, c='k', s=ms, marker=esc_m, label="Escape level"))
    hdls.append(ax.scatter(-100, 0, c='k', s=ms, marker=obs_m, label="Observed level"))
    ax.legend(handles=hdls, loc='lower left', framealpha=1.0, )

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plot_structure.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_structure_entry(handler: Proteus):
    plot_times,_ = sample_output(handler, extension="_atm.nc", tmin=1e3)
    print("Snapshots:", plot_times)

    # read helpfile
    hf_all = pd.read_csv(os.path.join(handler.directories['output'], "runtime_helpfile.csv"), sep=r"\s+")

    int_module = handler.config.interior.module
    output_dir = handler.directories['output']

    int_data = read_interior_data(output_dir, int_module, plot_times)
    atm_data = read_atmosphere_data(output_dir, plot_times)

    plot_structure(hf_all,output_dir, plot_times, int_data, atm_data, int_module,
                    plot_format=handler.config.params.out.plot_fmt)


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_structure_entry(handler)
