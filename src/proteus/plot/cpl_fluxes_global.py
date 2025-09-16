from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.plot import get_colour

if TYPE_CHECKING:
    from proteus import Proteus
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def plot_fluxes_global(hf_all:pd.DataFrame, output_dir: str, config: Config, t0: float=100.0):

    # Get values
    hf_crop = hf_all.loc[hf_all["Time"]>t0]
    time = np.array(hf_crop["Time"])
    if len(time) < 3:
        log.warning("Cannot make plot_fluxes with less than 3 samples")
        return

    log.info("Plot global fluxes")

    F_net = np.array(hf_crop["F_atm"])
    F_asf = np.array(hf_crop["F_ins"]) * config.orbit.s0_factor * (1.0 - config.atmos_clim.albedo_pl) * np.cos(config.orbit.zenith_angle * np.pi/180.0)
    F_olr = np.array(hf_crop["F_olr"])
    F_upw = np.array(hf_crop["F_olr"]) + np.array(hf_crop["F_sct"])
    F_int = np.array(hf_crop["F_int"])
    F_tidal = np.array(hf_crop["F_tidal"])
    F_radio = np.array(hf_crop["F_radio"])

    # Create plot
    scale = 1.2
    fig,ax = plt.subplots(1,1,figsize=(6*scale,5*scale))
    lw = 2.0
    al = 0.96

    # Steam runaway line
    ax.axhline(y=280.0, color='black', lw=lw, linestyle='dashed', label="S-N limit", zorder=1)

    # Plot fluxes
    ax.plot(time, F_tidal, lw=lw, alpha=al, color=get_colour("tidal"),label="Tidal")
    ax.plot(time, F_radio, lw=lw, alpha=al, color=get_colour("radio"),label="Radio.")
    ax.plot(time, F_int,   lw=lw, alpha=al, color=get_colour("int"),  label="Net (int.)")
    ax.plot(time, F_net,   lw=lw, alpha=al, color=get_colour("atm"),  label="Net (atm.)")
    ax.plot(time, F_olr,   lw=lw, alpha=al, color=get_colour("OLR"),  label="OLR")
    ax.plot(time, F_upw,   lw=lw, alpha=al, color=get_colour("sct"),  label="OLR + Scat.")
    ax.plot(time, F_asf,   lw=lw, alpha=al, color=get_colour("ASF"),  label="ASF", linestyle='dotted')


    # Configure plot
    ax.set_yscale("symlog")
    ax.set_ylabel("Unsigned flux [W m$^{-2}$]")
    ax.set_xscale("log")
    ax.set_xlabel("Time [yr]")
    ax.set_xlim(time[0], time[-1])
    legend = ax.legend(loc='lower left')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.99))
    ax.grid(color='black', alpha=0.05)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plots", "plot_fluxes_global.%s"%config.params.out.plot_fmt)
    fig.savefig(fpath, bbox_inches='tight', dpi=200)


def plot_fluxes_global_entry(handler: Proteus):
    # read helpfile
    hf_all = pd.read_csv(os.path.join(handler.directories['output'], "runtime_helpfile.csv"), sep=r"\s+")

    # make plot
    plot_fluxes_global(
        hf_all=hf_all,
        output_dir=handler.directories["output"],
        config=handler.config,
    )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_fluxes_global_entry(handler)
