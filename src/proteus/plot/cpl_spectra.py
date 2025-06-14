from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from proteus.observe.common import read_eclipse, read_transit
from proteus.utils.plot import get_colour, latexify

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

WAVE_KEY = "Wavelength/um"

def plot_spectra(output_dir: str, plot_format: str="pdf",
                    wlmin:float=0.5, wlmax:float=20.0, source:str="profile"):
    '''
    Plot the transit and eclipse spectra.

    Parameters
    ----------
    output_dir : str
        Output directory for the PROTEUS run.
    plot_format : str
        Format of the plot.
    wlmin : float
        Minimum wavelength to plot.
    wlmax : float
        Maximum wavelength to plot.
    source : str
        Method for setting atmospheric composition: "outgas", "profile", "offchem".
    '''

    log.info("Plot transit and eclipse spectra")

    # read data
    try:
        df_transit = read_transit(output_dir, source, "synthesis")
    except FileNotFoundError:
        log.warning(f"Could not read synthetic transit spectrum ({source}) file")
        df_transit = None

    try:
        df_eclipse = read_eclipse(output_dir, source, "synthesis")
    except FileNotFoundError:
        log.warning(f"Could not read synthetic eclipse spectrum ({source}) file")
        df_eclipse = None

    if df_transit is None and df_eclipse is None:
        return

    # make plot
    lw = 0.3
    scale = 1.1
    fig,axs = plt.subplots(2,1, figsize=(8*scale,7*scale), sharex=True)
    axt = axs[0]
    axb = axs[1]

    # plot spectra
    for i,df in enumerate([df_transit, df_eclipse]):
        if df is None:
            continue
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
    leg.set_zorder(99)
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    axb.set_ylabel("Secondary eclipse depth [ppm]")
    axb.set_xlabel(r"Wavelength [$\mu$m]")
    axb.set_xscale("log")
    axb.set_xticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20])
    axb.set_xlim(left=wlmin, right=wlmax)
    axb.xaxis.set_major_formatter(FormatStrFormatter("%g"))

    fig.subplots_adjust(hspace=0.01)
    fpath = os.path.join(output_dir, "plots", "plot_spectra.%s"%plot_format)
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
