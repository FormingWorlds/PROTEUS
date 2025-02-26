from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_spectra(output_dir: str, plot_format: str="pdf", t0: float=100.0):

    log.info("Plot transit/eclipse spectra")

    # read data
    ftransit = os.path.join(output_dir, "data", "obs_synth_transit.csv")
    if not os.path.isfile(ftransit):
        log.warning(f"Could not find file '{ftransit}'")
        return
    else:
        data = np.loadtxt(ftransit)
        transit_wl = data[:,0]
        transit_de = data[:,1]

    feclipse = os.path.join(output_dir, "data", "obs_synth_eclipse.csv")
    if not os.path.isfile(feclipse):
        log.warning(f"Could not find file '{feclipse}'")
        return
    else:
        data = np.loadtxt(feclipse)
        eclipse_wl = data[:,0]
        eclipse_de = data[:,1]

    # make plot
    lw = 1.0
    scale = 1.1
    fig,axs = plt.subplots(2,1, figsize=(7*scale,7*scale), sharex=True)
    axt = axs[0]
    axb = axs[1]

    # transit
    axt.plot(transit_wl, transit_de, lw=lw, color='k')
    axt.set_ylabel("Transit depth [ppm]")

    # eclipse
    axb.plot(eclipse_wl, eclipse_de, lw=lw, color='k')
    axb.set_ylabel("Eclipse depth [ppm]")

    axb.set_xlabel(r"Wavelength [$\mu$m]")
    axb.set_xscale("log")
    axb.set_xlim(left=0.3, right=25.0)
    axb.set_xticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20])
    axb.xaxis.set_major_formatter(FormatStrFormatter("%g"))

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plot_spectra.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')

def plot_spectra_entry(handler: Proteus):
    # make plots
    plot_spectra(
        handler.directories["output"],
        plot_format=handler.config.params.out.plot_fmt,
    )

if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_spectra_entry(handler)
