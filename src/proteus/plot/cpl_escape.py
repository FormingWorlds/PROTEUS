from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import element_list, secs_per_year
from proteus.utils.plot import get_colour

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

def plot_escape(hf_all:pd.DataFrame, output_dir:str, plot_format="pdf") :

    time = hf_all["Time"]
    if (len(time) < 3) or (np.amax(time) < 2):
        log.debug("Insufficient data to make plot_escape")
        return

    log.info("Plot escape")

    hf_crop = hf_all.iloc[2:]
    time = np.array(hf_crop["Time"])

    # mass unit
    M_uval = 1e20     # kg
    M_ulbl = r"$10^{20}$kg"

    # create plot
    lw = 1.2
    scale = 1.2
    fig,axs = plt.subplots(2,1, figsize=(5*scale,5*scale), sharex=True)

    # get axes
    axt = axs[0]
    axb = axs[1]
    axr = axb.twinx()

    # By element
    total = np.zeros(len(time))
    for e in element_list:

        _lw = lw
        if e == 'H':
            _lw = lw * 1.8
        col = get_colour(e)

        # Plot planetary inventory of this element
        y = np.array(hf_crop[e+"_kg_total"])/M_uval
        total += y
        axt.plot(time, y, lw=_lw, ls='dotted', color=col)

        # Plot atmospheric inventory of this element
        y = np.array(hf_crop[e+"_kg_atm"])/M_uval
        axt.plot(time, y, lw=_lw, ls='solid', color=col, label=e)

    # Planetary element sum inventory
    axt.plot(time, total, lw=lw, ls='dotted',  label='Total',  c='k')

    # Atmosphere mass
    M_atm = np.array(hf_crop["M_atm"])/M_uval
    axt.plot(time, M_atm, lw=lw, ls='solid', label='Atm.', c='k')

    # Decorate top plot
    axt.set_ylabel(r"Mass [%s]"%M_ulbl)
    axt.set_yscale("symlog", linthresh=1e-4)
    axt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.02), labelspacing=0.2)

    # Plot escape rate (kg / yr)
    y = np.array(hf_crop['esc_rate_total']) * secs_per_year * 1e6 / M_uval
    axb.plot(time, y, lw=lw, color='k')
    axb.set_ylabel('Escape rate [%s / Myr]'%M_ulbl)
    axb.set_xlabel("Time [yr]")
    axb.set_xlim(left=0, right=np.amax(time)*1.01)

    # Plot surface pressure
    color = 'seagreen'
    alpha = 0.8

    y = np.array(hf_crop["P_surf"])
    axr.plot(time, y, lw=lw, color=color, alpha=alpha)
    tmax = time[np.argmax(y)]
    axr.axvline(tmax, color=color, alpha=alpha, ls='dashdot', label=r"Maximum P$_\text{surf}$")
    axt.axvline(tmax, color=color, alpha=alpha, ls='dashdot')

    axr.set_ylabel('Surface pressure [bar]', color=color)
    axr.tick_params(axis='y', color=color, labelcolor=color)
    axr.legend(loc='lower left')

    # Adjust
    fig.subplots_adjust(hspace=0.02)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plots", "plot_escape.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_escape_entry(handler: Proteus):
    # read helpfile
    hf_all = pd.read_csv(os.path.join(handler.directories['output'], "runtime_helpfile.csv"), sep=r"\s+")

    # make plot
    plot_escape(
        hf_all=hf_all,
        output_dir=handler.directories["output"],
        plot_format=handler.config.params.out.plot_fmt,
    )

if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_escape_entry(handler)
