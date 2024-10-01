from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import element_list

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

def plot_elements( hf_all:pd.DataFrame, output_dir:str, plot_format="pdf", t0=100.0):

    if np.amax(hf_all["Time"]) < 2:
        log.debug("Insufficient data to make plot_elements")
        return

    log.info("Plot elements")

    time = np.array(hf_all["Time"] )

    # make plot
    lw = 1.2
    scale = 1.1
    fig,ax = plt.subplots(1,1, figsize=(7*scale,4*scale))

    total = np.zeros(len(time))
    for e in element_list:
        y = hf_all[e+"_kg_total"]
        ax.plot(time, y, lw=lw, ls='solid', label="Total "+e)[0]

        total += y

    ax.plot(time, total,           lw=lw, ls='solid',  label='Total',  c='k')
    ax.plot(time, hf_all["M_atm"], lw=lw, ls='dotted', label='Atmos.', c='k')

    # decorate
    ax.set_ylabel("Inventory [kg]")
    ax.set_yscale("log")
    ax.set_xlabel("Time [yr]")
    ax.set_xscale("log")
    ax.set_xlim(left=t0, right=np.amax(time))
    ax.grid(alpha=0.2)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plot_elements.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_elements_entry(handler: Proteus):
    # read helpfile
    hf_all = pd.read_csv(os.path.join(handler.directories['output'], "runtime_helpfile.csv"), sep=r"\s+")

    # make plot
    plot_elements(
        hf_all=hf_all,
        output_dir=handler.directories["output"],
        plot_format=handler.config["plot_format"],
    )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_elements_entry(handler)
