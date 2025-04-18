from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from proteus.utils.constants import M_earth, M_jupiter, R_earth, R_jupiter

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

EARTH_COLOUR="hotpink"
LEG_KWARGS = {
        "frameon":1,
        "fancybox":True,
        "framealpha":0.9,
        "labelspacing":0.2,
        "columnspacing":1.1,
        "handletextpad":0.7
    }

def _get_exo_data(fwl_dir:str):
    popfile = os.path.join(fwl_dir, "planet_reference", "Exoplanets", "DACE_PlanetS.csv")
    return pd.read_csv(popfile,comment="#")

def _get_mr_data(fwl_dir:str):
    z19 = os.path.join(fwl_dir, "mass_radius", "Mass-radius", "Zeng2019")

    # Set paths
    curves = {
        "Earth-like":       os.path.join(z19, "massradiusEarthlikeRocky.txt"),
        "Fe":               os.path.join(z19, "massradiusFe.txt"),
        r"MgSiO$_3$":       os.path.join(z19, "massradiusmgsio3.txt"),
        r"H$_2$O, 300 K":   os.path.join(z19, "massradius_100percentH2O_300K_1mbar.txt"),
        r"Cold H$_2$":      os.path.join(z19, "massradiushydrogen.txt"),
    }

    # Replace paths with the data
    for k in curves.keys():
        data = np.loadtxt(curves[k]).T
        mask = np.argsort(data[0])
        curves[k] = [data[0][mask],data[1][mask]]

    return curves

def plot_population_mass_radius(hf_all:pd.DataFrame, output_dir: str, fwl_dir:str,
                                    plot_format:str, t0: float=100.0,
                                    show_errors:bool = False,
                                    m_min:float=0.0, m_max:float = 8.0):

    """
    Plot planetary evolution on a population mass-radius diagram.
    """

    # Get values from simulation
    hf_crop = hf_all.loc[hf_all["Time"]>t0]
    time = np.array(hf_crop["Time"])
    if len(time) < 3:
        log.warning("Cannot make plot_population with less than 3 samples")
        return

    m_max = max(m_max, np.amax(hf_crop["M_planet"])/M_earth+1)
    m_min = min(m_min, np.amin(hf_crop["M_planet"])/M_earth)

    log.info("Plot population (mass-radius)")
    sim_rad = np.array(hf_crop["R_obs"])/ R_earth
    sim_mas = np.array(hf_crop["M_planet"]) / M_earth

    # Get exoplanet values from database
    exo = _get_exo_data(fwl_dir)
    exo = exo.loc[exo["Planet Mass [Mjup]"] * M_jupiter / M_earth <= m_max*1.1]

    exo_mas_val = exo["Planet Mass [Mjup]"]             * M_jupiter / M_earth
    exo_mas_upp = exo["Planet Mass - Upper Unc [Mjup]"] * M_jupiter / M_earth
    exo_mas_low = exo["Planet Mass - Lower Unc [Mjup]"] * M_jupiter / M_earth

    exo_rad_val = exo["Planet Radius [Rjup]"]             * R_jupiter / R_earth
    exo_rad_upp = exo["Planet Radius - Upper Unc [Rjup]"] * R_jupiter / R_earth
    exo_rad_low = exo["Planet Radius - Lower Unc [Rjup]"] * R_jupiter / R_earth

    # Get Mass-Radius curves from files
    mrdata = _get_mr_data(fwl_dir)

    # Create plot
    scale = 1.05
    fig,ax = plt.subplots(1,1,figsize=(6*scale,5*scale))

    # M-R curves
    for k in mrdata.keys():
        ax.plot(mrdata[k][0],mrdata[k][1], label=k, linestyle='dashed', alpha=0.8, zorder=9)

    # Radius valley
    ax.fill_betweenx([1.5, 2.0], m_min, m_max, zorder=8,
                        alpha=0.2, color='gold', ec='none', label="Radius valley")

    # Exoplanets
    lbl = "Exoplanets"
    col = 'grey'
    ms=6
    if show_errors:
        ax.errorbar(exo_mas_val, exo_rad_val,
                    xerr=[exo_mas_low, exo_mas_upp],
                    yerr=[exo_rad_low, exo_rad_upp],
                    label=lbl, ls='none', ms=ms, color=col, elinewidth=0.9, zorder=10)
    else:
        ax.scatter(exo_mas_val, exo_rad_val, color=col, label=lbl, s=ms, zorder=10)

    # Solar system planets
    ax.scatter(1.0,  1.0, label="Modern Earth", color=EARTH_COLOUR, zorder=11)

    # Simulation
    col='k'
    ax.plot(sim_mas, sim_rad, label="Simulation", color=col, lw=1.5, zorder=12)
    ax.scatter(sim_mas[-1], sim_rad[-1], color=col, s=ms, zorder=12, marker='D', label='End')

    # Save figure
    ax.set_ylabel(r"Planet radius [$R_{\oplus}$]")
    ax.set_ylim(bottom=0, top=max(np.amax(exo_rad_val), np.amax(sim_rad)))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.set_xlabel(r"Planet mass [$M_{\oplus}$]")
    ax.set_xlim(left=m_min,right=m_max)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(alpha=0.2, zorder=-2)

    ax.legend(loc='lower right', ncol=2, **LEG_KWARGS)
    fpath = os.path.join(output_dir, "plots", "plot_population_mass_radius.%s"%plot_format)
    fig.savefig(fpath, bbox_inches='tight', dpi=200)


def plot_population_time_density(hf_all:pd.DataFrame, output_dir: str, fwl_dir:str,
                                 plot_format:str, t0: float=100.0,
                                 xmin:float=1e5, ymin:float=0.1, ymax:float=10.0):
    """
    Plot planetary evolution on population time-density diagram
    """

    # Get values from simulation
    hf_crop = hf_all.loc[hf_all["Time"]>t0]
    time = np.array(hf_crop["Time"])
    if len(time) < 3:
        log.warning("Cannot make plot_population with less than 3 samples")
        return

    log.info("Plot population (time-density)")
    sim_rad = np.array(hf_crop["R_obs"])
    sim_mas = np.array(hf_crop["M_planet"])
    sim_rho = 3*sim_mas/(4*np.pi*sim_rad**3)  * 0.001

    # Get values from database
    exo = _get_exo_data(fwl_dir)
    exo_age_val = np.array(exo["Stellar Age [Gyr]"]   )  * 1e9    # yr
    exo_rho_val = np.array(exo["Planet Density [g/cm**3] - Computation"]) # g cm-3
    xmax = max(np.nanmax(exo_age_val), np.amax(time))

    # Make sure that we actually plot the simulation data
    if xmin > np.amax(time):
        xmin = time[0]
    ymin = min(ymin, np.amin(sim_rho[2:])/2)
    ymax = max(ymax, np.amax(sim_rho[2:])*2)

    # Create plot
    scale = 1.05
    lw=1.8
    fig,ax = plt.subplots(1,1,figsize=(6*scale,5*scale))

    # Reference values
    ax.axhline(y=1.0, label=r"H$_2$O", color='tab:blue', alpha=0.8, zorder=0, ls='dashed')

    # Exoplanets
    lbl = "Exoplanets"
    col = 'grey'
    ms=8
    ax.scatter(exo_age_val, exo_rho_val, color=col, label=lbl, s=ms, zorder=1)

    # Modern Earth
    ax.scatter(4.6e9, 5513*0.001, label="Modern Earth", color=EARTH_COLOUR, zorder=3)

    # Simulation
    col='k'
    ax.plot([time[-1],1e13],[sim_rho[-1],sim_rho[-1]], color=col, lw=lw, ls='dotted', label="Extrapolated", zorder=97)
    ax.plot(time, sim_rho, label="Simulation", color=col, lw=lw, zorder=98)
    ax.scatter(time[-1], sim_rho[-1], color=col, s=ms, zorder=99, marker='D', label='End')


    # Save figure
    ax.set_ylabel(r"Bulk density [g cm$^{-3}$]")
    ax.set_yscale("log")
    ax.set_ylim(bottom=ymin, top=ymax)

    ax.set_xlabel(r"Time [yr]")
    ax.set_xscale("log")
    ax.set_xlim(left=xmin,right=xmax*1.2)
    ax.grid(alpha=0.2, zorder=-2)

    ax.legend(**LEG_KWARGS)
    fpath = os.path.join(output_dir, "plots", "plot_population_time_density.%s"%plot_format)
    fig.savefig(fpath, bbox_inches='tight', dpi=200)


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
