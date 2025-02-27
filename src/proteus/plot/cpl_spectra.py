from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

from proteus.utils.plot import get_colour, latexify

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

WAVE_KEY = "Wavelength/um"

def plot_spectra(output_dir: str, plot_format: str="pdf",
                    wlmin:float=0.5, wlmax:float=20.0):

    log.info("Plot transit and eclipse spectra")

    # read data
    ftransit = os.path.join(output_dir, "data", "obs_synth_transit.csv")
    if not os.path.isfile(ftransit):
        log.warning(f"Could not find file '{ftransit}'")
        return
    else:
        df_transit = pd.read_csv(ftransit)

    feclipse = os.path.join(output_dir, "data", "obs_synth_eclipse.csv")
    if not os.path.isfile(feclipse):
        log.warning(f"Could not find file '{feclipse}'")
        return
    else:
        df_eclipse = pd.read_csv(feclipse)

    # make plot
    lw = 0.3
    scale = 1.1
    fig,axs = plt.subplots(2,1, figsize=(8*scale,7*scale), sharex=True)
    axt = axs[0]
    axb = axs[1]

    # transit
    for i,df in enumerate([df_transit, df_eclipse]):
        wl = df[WAVE_KEY]
        for key in df.keys():
            if key != WAVE_KEY:
                lbl = key.split("_")[-1].split("/")[0]
                if lbl == "None":
                    zorder = 6
                    color = 'black'
                else:
                    zorder = 4
                    color = get_colour(lbl)
                    lbl = latexify(lbl)
                axs[i].plot(wl, df[key], lw=lw, label=lbl, color=color, zorder=zorder)

        axs[i].grid(zorder=-2, alpha=0.3)

    axt.set_ylabel("Primary transit depth [ppm]")
    leg = axb.legend(ncols=3, title="Removed gases", loc='upper left')
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    axb.set_ylabel("Secondary eclipse depth [ppm]")
    axb.set_xlabel(r"Wavelength [$\mu$m]")
    axb.set_xscale("log")
    axb.set_xticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20])
    axb.set_xlim(left=wlmin, right=wlmax)
    axb.xaxis.set_major_formatter(FormatStrFormatter("%g"))

    fig.subplots_adjust(hspace=0.01)
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
