from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from proteus.utils.constants import AU, secs_per_hour

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
    ax_tr.set_ylabel("Planet orbital eccentricity")
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

def plot_orbit_system(hf_all:pd.DataFrame, output_dir:str, plot_format:str="pdf", t0=1e3):

    if np.amax(hf_all["Time"]) <= t0+1:
        log.debug("Insufficient data to make plot_system")
        return

    log.info("Plot orbit_system")

    # Plotting parameters
    lw_pla = 1.2
    lw_sat = 0.8
    figscale = 1.4
    fig,ax = plt.subplots(1,1, figsize=(4*figscale,4*figscale))

    # plot star
    ax.scatter(0,0,color='orange', s=60, zorder=4, label='Star', marker='*')

    # Colors
    times = np.array(hf_all["Time"][:])
    norm = mpl.colors.LogNorm(vmin=t0, vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlow, norm=norm)
    sm.set_array([])

    # plot planet at time
    t = np.linspace(0, np.pi*2, 80)
    def _plot_planet(i):
        hf_row = hf_all.iloc[i]
        col = sm.to_rgba(hf_row["Time"])

        # planet orbit parameters
        a = hf_row["semimajorax"] / AU
        e = hf_row["eccentricity"]
        b = a * np.sqrt(1-e*e)

        # location of focus
        f = a*e

        # plot ellipse of planet orbit
        x = a * np.cos(t) - f
        y = b * np.sin(t)
        ax.plot(x, y, color=col, alpha=0.8, zorder=5, lw=lw_pla)

        # plot satellite orbit around planet
        asat = hf_row["semimajorax_sat"] / AU
        x0 = np.amin(x)
        xx = asat * np.cos(t) + x0
        yy = asat * np.sin(t)
        ax.plot(xx, yy, lw=lw_sat, color=col, alpha=0.4, zorder=5)

        return max(rmax, np.amax(np.abs(x)))

    # make orbits
    rmax = 0.01
    for i in range(len(hf_all)):
        rmax = max(_plot_planet(i), rmax)

    # roche radius of star
    roche = hf_all.iloc[-1]["roche_limit"] / AU
    ax.plot(roche * np.cos(t), roche * np.sin(t), ls='dashed',
                c='tab:red', label="Roche limit")


    # Plot colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=-0.2)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label("Time [yr]")

    # dummy labels
    ax.plot([],[],label="Planet orbit", c='purple',lw=lw_pla)
    ax.plot([],[],label="Moon orbit",   c='purple',lw=lw_sat)

    # decorate
    rmax *= 1.2
    lims = (-rmax, rmax)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticklabels([])
    ax.set_ylabel("Distance [AU]")
    ax.grid(zorder=0, alpha=0.3)
    ax.legend(loc='upper right')


    plt.close()
    plt.ioff()

    fig.tight_layout()

    fpath = os.path.join(output_dir, "plots", "plot_orbit_system.%s"%plot_format)
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
    plot_orbit_system(
        hf_all,
        handler.directories["output"],
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_orbit_entry(handler)
