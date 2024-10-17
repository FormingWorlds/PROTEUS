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


def plot_observables(hf_all:pd.DataFrame, output_dir: str, plot_format: str="pdf", t0: float=100.0):

    time = np.array(hf_all["Time"] )
    if np.amax(time) < 2:
        log.debug("Insufficient data to make plot_observables")
        return

    log.info("Plot observables")

    # make plot
    lw = 1.2
    scale = 1.1
    fig,axl = plt.subplots(1,1, figsize=(7*scale,4*scale))

    # left axis
    axl.plot(time, hf_all["transit_depth"]*1e6, lw=lw, color='k')
    axl.set_ylabel("Transit depth [ppm]")

    # right axis
    axr = axl.twinx()
    color = "tab:red"
    axr.plot(time, hf_all["contrast_ratio"]*1e9, lw=lw, color=color)
    axr.set_ylabel("Contrast ratio [ppb]")
    axr.yaxis.label.set_color(color)
    axr.tick_params(axis='y', colors=color)

    # x-axis
    axl.set_xlabel("Time [yr]")
    axl.set_xscale("log")
    axl.set_xlim(left=t0, right=np.amax(time))
    axl.grid(alpha=0.2)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plot_observables.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_observables_entry(handler: Proteus):
    # read helpfile
    hf_all = pd.read_csv(os.path.join(handler.directories['output'], "runtime_helpfile.csv"), sep=r"\s+")

    # make plot
    plot_observables(
        hf_all=hf_all,
        output_dir=handler.directories["output"],
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_observables_entry(handler)
