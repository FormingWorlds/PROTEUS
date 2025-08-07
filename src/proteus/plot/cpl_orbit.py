from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_orbit(hf_all:pd.DataFrame, output_dir:str, plot_format:str="pdf", t0:float=100.0):

    time = np.array(hf_all["Time"] )
    if np.amax(time) < 2:
        log.debug("Insufficient data to make plot_orbit")
        return

    log.info("Plot orbit")

    # Plotting parameters
    lw=2.0
    fig_ratio=(1,2)
    fig_scale=4.0

    fig,axs = plt.subplots(1,2, figsize=(fig_ratio[0]*fig_scale, fig_ratio[1]*fig_scale), sharex=True)
    ax_t = axs[0][0]
    ax_b = axs[0][1]

    # left axis
    ax_t.plot(time, hf_all["semimajorax"], lw=lw, color='k')
    ax_t.set_ylabel("Semi-major axis [m]")

    # right axis
    ax_tr = ax_t.twinx()
    color = "tab:red"
    ax_tr.plot(time, hf_all["eccentricity"], lw=lw, color=color)
    ax_tr.set_ylabel("Eccentricity")
    ax_tr.yaxis.label.set_color(color)
    ax_tr.tick_params(axis='y', colors=color)

    # x-axis
    ax_t.set_xlabel("Time [yr]")
    ax_t.set_xscale("log")
    ax_t.set_xlim(left=t0, right=np.amax(time))
    ax_t.grid(alpha=0.2)


    # left axis
    ax_b.plot(time, hf_all["semimajorax_sat"], lw=lw, color='k')
    ax_b.set_ylabel("Semi-major axis [m]")

    # right axis
    ax_br = ax_b.twinx()
    color = "tab:red"
    ax_br.plot(time, hf_all["axial_period"]/3600, lw=lw, color=color)
    ax_br.set_ylabel("Length of Day")
    ax_br.yaxis.label.set_color(color)
    ax_br.tick_params(axis='y', colors=color)

    # x-axis
    ax_b.set_xlabel("Time [yr]")
    ax_b.set_xscale("log")
    ax_b.set_xlim(left=t0, right=np.amax(time))
    ax_b.grid(alpha=0.2)


    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plots", "plot_orbit.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_orbit_entry(handler: Proteus):
    # read helpfile
    hf_all = pd.read_csv(os.path.join(handler.directories['output'], "runtime_helpfile.csv"), sep=r"\s+")

    # make plot
    plot_orbit(
        hf_all,
        handler.directories["output"],
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_orbit_entry(handler)
