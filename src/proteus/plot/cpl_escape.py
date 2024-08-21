#!/usr/bin/env python3
from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import M_earth

log = logging.getLogger("PROTEUS")

def plot_escape(output_dir, escape_model, plot_format="pdf", t0=100.0) :

    log.info("Plot escape")

    hf_all = pd.read_csv(output_dir+"/runtime_helpfile.csv", sep=r"\s+")

    time = np.array(hf_all["Time"])
    if len(time) < 3:
        log.warning("Cannot make plot with less than 3 samples")
        return

    # make plot
    lw = 1.2
    al = 0.5
    scale = 1.1
    fig,ax1 = plt.subplots(1,1, figsize=(7*scale,4*scale))

    if escape_model == 0 :
        escape_model_label = 'No escape'
    elif escape_model == 1 :
        escape_model_label = 'Energy-limited escape (Zephyrus)'
    elif escape_model == 2 :
        escape_model_label = 'Dummy escape'

    y = hf_all['esc_rate_total']
    l = ax1.plot(time, y, lw=lw, ls='solid', label=f'{escape_model_label}')


    # decorate
    ax1.set_ylabel(r'Mass loss rate [kg $s^{-1}$]')
    ax1.set_yscale("log")
    ax1.set_xlabel("Time [yr]")
    ax1.set_xscale("log")
    ax1.set_xlim(left=t0, right=np.amax(time))
    ax1.grid(alpha=0.2)
    ax1.legend()
    ax2 = ax1.twinx()
    ylims = ax1.get_ylim()
    ax2.set_ylim((ylims[0]/ s2yr) / M_earth,(ylims[1] / s2yr) / M_earth)
    ax2.set_yscale('log')
    ax2.set_ylabel(r'Mass loss rate [$M_{\oplus}$ $yr^{-1}$]')

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plot_escape.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')

#====================================================================
def main():
    from proteus.plot._cpl_helpers import get_options_dirs_from_argv

    options, dirs = get_options_dirs_from_argv()

    plot_escape(output_dir=dirs["output"], escape_model=options['escape_model'], plot_format=options["plot_format"])

#====================================================================

if __name__ == "__main__":

    main()
