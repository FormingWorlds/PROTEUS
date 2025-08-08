from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import AU, secs_per_day, secs_per_hour

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_orbit(hf_all:pd.DataFrame, output_dir:str, plot_format:str="pdf", t0:float=100.0):

    time = np.array(hf_all["Time"] )
    if np.amax(time) <= t0:
        log.debug("Insufficient data to make plot_orbit")
        return

    log.info("Plot orbit")

    # Plotting parameters
    lw=2.0
    figscale = 1.4
    yext = 1.05

    fig,axs = plt.subplots(2,1, figsize=(5*figscale,4*figscale), sharex=True)
    ax_t = axs[0]
    ax_b = axs[1]

    # left axis
    y = hf_all["semimajorax"]/AU
    ax_t.plot(time, y, lw=lw, color='k')
    ax_t.set_ylabel("Planet semi-major axis [AU]")
    ax_t.set_ylim(0, np.amax(y)*yext)

    # right axis
    ax_tr = ax_t.twinx()
    color = "tab:red"
    y = hf_all["eccentricity"]
    ax_tr.plot(time, y, lw=lw, color=color)
    ax_tr.set_ylabel("Planet orbital ccentricity")
    ax_tr.yaxis.label.set_color(color)
    ax_tr.tick_params(axis='y', colors=color)
    ax_tr.set_ylim(np.amin(y)/yext, np.amax(y)*yext)

    # x-axis
    ax_t.set_xscale("log")
    ax_t.set_xlim(left=t0, right=np.amax(time))
    ax_t.grid(alpha=0.2)

    # left axis
    y = hf_all["semimajorax_sat"]/1e6
    ax_b.plot(time, y, lw=lw, color='k')
    ax_b.set_ylabel(r"Satellite semi-major axis [$10^6$m]")
    ax_b.set_ylim(0, np.amax(y)*yext)

    # right axis
    ax_br = ax_b.twinx()
    color = "tab:red"
    y = hf_all["axial_period"]/secs_per_hour
    ax_br.plot(time, y, lw=lw, color=color)
    ax_br.set_ylabel("Planet axial period [hours]")
    ax_br.yaxis.label.set_color(color)
    ax_br.tick_params(axis='y', colors=color)
    ax_br.set_ylim(0, np.amax(y)*yext)

    # x-axis
    ax_b.set_xlabel("Time [yr]")
    ax_b.set_xscale("log")
    ax_b.set_xlim(left=t0, right=np.amax(time))
    ax_b.grid(alpha=0.2)


    plt.close()
    plt.ioff()

    fig.tight_layout()

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
