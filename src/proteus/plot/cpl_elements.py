from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import M_earth, element_list

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
    lw = 1.8
    scale = 1.1
    fig,ax = plt.subplots(1,1, figsize=(6*scale,4*scale))

    # By element
    total = np.zeros(len(time))
    for e in element_list:

        _lw = lw
        if e == 'H':
            _lw = lw * 1.8

        # Plot planetary inventory of this element
        y = np.array(hf_all[e+"_kg_total"])/M_earth
        total += y
        line = ax.plot(time, y, lw=_lw, ls='solid',  label=e)[0]

        # Plot atmospheric inventory of this element
        y = np.array(hf_all[e+"_kg_atm"])/M_earth
        ax.plot(time, y, lw=_lw, ls='dotted', color=line.get_color())

    # Planetary element sum inventory
    ax.plot(time, total, lw=lw, ls='solid',  label='Total',  c='k')

    # Atmosphere mass
    M_atm = np.array(hf_all["M_atm"])/M_earth
    ax.plot(time, M_atm, lw=lw, ls='dotted', label='Atmosphere', c='k')

    # decorate
    ax.set_ylabel(r"Mass [M$_\oplus$]")
    ax.set_yscale("log")
    ax.set_xlabel("Time [yr]")
    ax.set_xscale("log")
    ax.set_xlim(left=t0, right=np.amax(time)*1.1)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=4)

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
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_elements_entry(handler)
