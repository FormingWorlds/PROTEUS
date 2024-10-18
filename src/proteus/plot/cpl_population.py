from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import (
    M_earth,
    R_earth,
    M_jupiter,
    R_jupiter
)

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

def _get_exo_data(fwl_dir:str):
    popfile = os.path.join(fwl_dir, "planet_reference", "Exoplanets", "DACE_PlanetS.csv")
    return pd.read_csv(popfile,comment="#")

def plot_population_mass_radius(hf_all:pd.DataFrame, output_dir: str, fwl_dir:str,
                                    plot_format:str, t0: float=100.0,
                                    show_errors:bool = False, r_max:float = 10.0):

    """
    Plot planetary evolution on a population mass-radius diagram.
    """

    # Get values from simulation
    hf_crop = hf_all.loc[hf_all["Time"]>t0]
    time = np.array(hf_crop["Time"])
    if len(time) < 3:
        log.warning("Cannot make plot with less than 3 samples")
        return

    log.info("Plot population (mass-radius)")
    sim_rad = np.array(hf_crop["z_obs"]    )/ R_earth
    sim_mas = np.array(hf_crop["M_planet"]) / M_earth

    # Get values from database
    popfile = os.path.join(fwl_dir, "planet_reference", "Exoplanets", "DACE_PlanetS.csv")
    exo = pd.read_csv(popfile,comment="#")

    # filter database
    exo = _get_exo_data(fwl_dir)
    exo = exo.loc[exo["Planet Mass [Mjup]"] * M_jupiter / M_earth <= r_max]

    exo_mas_val = exo["Planet Mass [Mjup]"]             * M_jupiter / M_earth
    exo_mas_upp = exo["Planet Mass - Upper Unc [Mjup]"] * M_jupiter / M_earth
    exo_mas_low = exo["Planet Mass - Lower Unc [Mjup]"] * M_jupiter / M_earth

    exo_rad_val = exo["Planet Radius [Rjup]"]             * R_jupiter / R_earth
    exo_rad_upp = exo["Planet Radius - Upper Unc [Rjup]"] * R_jupiter / R_earth
    exo_rad_low = exo["Planet Radius - Lower Unc [Rjup]"] * R_jupiter / R_earth

    # Create plot
    scale = 1.0
    fig,ax = plt.subplots(1,1,figsize=(6*scale,5*scale))

    # Exoplanets
    lbl = "Exoplanets"
    col = 'grey'
    ms=8
    if show_errors:
        ax.errorbar(exo_mas_val, exo_rad_val,
                    xerr=[exo_mas_low, exo_mas_upp],
                    yerr=[exo_rad_low, exo_rad_upp],
                    label=lbl, ls='none', ms=ms, color=col, elinewidth=0.9)
    else:
        ax.scatter(exo_mas_val, exo_rad_val, color=col, label=lbl, s=ms)

    # Solar system planets
    ax.scatter(1.0,  1.0, label="Earth", color='tab:red')

    # Simulation
    col='k'
    ax.plot(sim_mas, sim_rad, label="Simulation", color=col, lw=1.5, zorder=99)

    # Save figure
    ax.set_ylabel(r"Planet radius [$R_{\oplus}$]")
    ax.set_xlabel(r"Planet mass [$M_{\oplus}$]")

    ax.legend()
    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_population_mass_radius."+plot_format,
                bbox_inches='tight', dpi=200)


def plot_population_time_density(hf_all:pd.DataFrame, output_dir: str, fwl_dir:str,
                                 plot_format:str, t0: float=100.0):
    """
    Plot planetary evolution on population time-density diagram
    """

    # Get values from simulation
    hf_crop = hf_all.loc[hf_all["Time"]>t0]
    time = np.array(hf_crop["Time"])
    if len(time) < 3:
        log.warning("Cannot make plot with less than 3 samples")
        return

    log.info("Plot population (time-density)")
    sim_rad = np.array(hf_crop["z_obs"]    )
    sim_mas = np.array(hf_crop["M_planet"])
    sim_rho = 3*sim_mas/(4*np.pi*sim_rad**3)  * 0.001

    # Get values from database
    popfile = os.path.join(fwl_dir, "planet_reference", "Exoplanets", "DACE_PlanetS.csv")
    exo = pd.read_csv(popfile,comment="#")

    # filter database
    exo = _get_exo_data(fwl_dir)

    exo_mas_val = np.array(exo["Planet Mass [Mjup]"]  )  * M_jupiter
    exo_rad_val = np.array(exo["Planet Radius [Rjup]"])  * R_jupiter
    exo_age_val = np.array(exo["Stellar Age [Gyr]"]   )  * 1e9    # yr
    exo_rho_val = 3*exo_mas_val/(4*np.pi*exo_rad_val**3) * 0.001  # g cm-3

    xmax = max(np.nanmax(exo_age_val), np.amax(time))

    # Create plot
    scale = 1.0
    lw=1.8
    fig,ax = plt.subplots(1,1,figsize=(6*scale,5*scale))

    # Exoplanets
    lbl = "Exoplanets"
    col = 'grey'
    ms=8
    ax.scatter(exo_age_val, exo_rho_val, color=col, label=lbl, s=ms)

    # Other reference values
    ax.scatter(4.6e9, 5513*0.001, label="Earth", color='tab:red')
    ax.axhline(y=1.0, label=r"H$_2$O", color='tab:blue', lw=lw)
    ax.axhline(y=10738*0.001, label="Earth's core", color='tab:brown', lw=lw)

    # Simulation
    col='k'
    ax.plot(time, sim_rho, label="Simulation", color=col, lw=lw)
    ax.plot([time[-1],1e13],[sim_rho[-1],sim_rho[-1]], color=col, lw=lw, ls='dashed', label="Extrapolated")

    # Save figure
    ax.set_ylabel(r"Bulk density [g cm$^{-3}$]")
    ax.set_yscale("log")
    ax.set_xlabel(r"Time [yr]")
    ax.set_xscale("log")
    ax.set_xlim(left=time[0],right=xmax*1.2)

    ax.legend()
    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_population_time_density."+plot_format,
                bbox_inches='tight', dpi=200)


def plot_population_entry(handler: Proteus):
    # read helpfile
    hf_all = pd.read_csv(os.path.join(handler.directories['output'], "runtime_helpfile.csv"), sep=r"\s+")

    # make plots
    plot_population_mass_radius(
        hf_all=hf_all,
        output_dir=handler.directories["output"],
        fwl_dir=handler.directories["fwl"],
        plot_format=handler.config.params.out.plot_fmt
    )

    plot_population_time_density(
        hf_all=hf_all,
        output_dir=handler.directories["output"],
        fwl_dir=handler.directories["fwl"],
        plot_format=handler.config.params.out.plot_fmt
    )

if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_population_entry(handler)
