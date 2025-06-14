from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm

from proteus.utils.constants import const_c, const_h, const_k
from proteus.utils.helper import find_nearest, natural_sort

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def planck_function(lam, T):
    """Plots stellar flux from output directory for a set of wavelength bins

    Planck function value at stellar surface
    lam in nm
    erg s-1 cm-2 nm-1
    """

    x = lam * 1.0e-9   # convert nm -> m
    hc_by_kT = const_h*const_c / (const_k*T)

    planck_func = 1.0/( (x ** 5.0) * ( np.exp( hc_by_kT/ x) - 1.0 ) )
    planck_func *= 2 * const_h * const_c * const_c #  w m-2 sr-1 s-1 m-1
    planck_func *= np.pi * 1.0e3 * 1.0e-9  # erg s-1 cm-2 nm-1

    return planck_func


def plot_sflux_cross(
        output_dir: str,
        wl_targets: list | None=None,
        modern_age: float=-1,
        plot_format="pdf",
    ):
    """Plots stellar flux vs time, for a set of wavelengths.

    Note that this function will plot the flux from EVERY file it finds.
    Saves plot as 'cpl_sflux_cross.pdf' in  the output directory.

    Parameters
    ----------
    output_dir : str
        Directory for both reading from and saving to.
    wl_targets : list | None
        List of wavelengths to plot [nm]
    modern_age : float
        Current age of star. If not provided, then won't be plotted
    """

    # Wavelength targets default value
    if not wl_targets:
        wl_targets = [1.0, 12.0, 50.0, 121.0, 200.0, 400.0, 500.0, 2000.0]

    # Find and sort files
    files = glob.glob(output_dir+"/data/*.sflux")
    files = natural_sort(files)

    if len(files) <= 1:
        log.warning("Insufficient data to make plot_sflux_cross")
        return

    log.info("Plot stellar flux (crossection)")

    # Arrays for storing data over time
    time_t = []
    wave_t = []
    flux_t = []
    for f in files:
        # Load data
        X = np.loadtxt(f,skiprows=1,delimiter='\t').T

        # Parse data
        time = int(f.split('/')[-1].split('.')[0])
        if (time < 0):
            continue
        wave = X[0]
        flux = X[1]

        # Save data
        time_t.append(time)
        wave_t.append(wave)
        flux_t.append(flux)

    time_t = np.array(time_t)
    wave_t = np.array(wave_t)
    flux_t = np.array(flux_t)

    # Create figure
    fig,ax = plt.subplots(1,1)

    ax.set_yscale("log")
    ax.set_ylabel("Flux [erg / (s cm$^2$ nm)]")

    ax.set_xscale("log")
    ax.set_xlabel("Time [yr]")
    ax.set_title("TOA flux versus time")

    vmin = max(time_t[0],1.0)
    vmax = time_t[-1]
    ax.set_xlim([vmin,vmax])

    # Find indices for wavelength bins
    wl_iarr = []
    wl_varr = []
    for w in wl_targets:
        wl_v, wl_i = find_nearest(wave,w)
        wl_iarr.append(wl_i)
        wl_varr.append(wl_v)
    N = len(wl_iarr)

    # Load modern spectrum
    if modern_age > 0:
        modern_fpath = os.path.join(output_dir, "data", "-1.sflux")
        X = np.loadtxt(modern_fpath, skiprows=2).T

    # Plot bins over time
    for i in range(N):

        fl = flux_t.T[wl_iarr[i]]
        c = cm.oleron(1.0*i/N)
        lbl = "%d"%max(1,round(wl_varr[i]))

        ax.plot(time_t,fl,color=c,lw=2.8,label=lbl)

        # Plot modern values
        if modern_age > 0:
            ax.scatter(modern_age,X[1][wl_iarr[i]],marker='o',color=c, s=40, zorder=4, edgecolors='white')

    leg = ax.legend(title=r"$\lambda$ [nm]", loc='center left',bbox_to_anchor=(1.02, 0.5))
    for legobj in leg.legend_handles:
        legobj.set_linewidth(4.0)

    plt.close()
    plt.ioff()
    fpath = os.path.join(output_dir, "plots", "plot_sflux_cross.%s"%plot_format)
    fig.savefig(fpath, bbox_inches='tight', dpi=200)



def plot_sflux_cross_entry(handler: Proteus):
    wl_targets = [1.0, 12.0, 50.0, 121.0, 200.0, 400.0, 500.0, 2000.0]

    if handler.config.star.module == "mors":
        modern_age = handler.config.star.mors.age_now * 1e9
    else:
        modern_age = -1

    plot_sflux_cross(
        output_dir=handler.directories['output'],
        wl_targets=wl_targets,
        modern_age=modern_age,
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == '__main__':
    print("Plotting stellar flux over time (bins)...")

    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_sflux_cross_entry(handler)

    print("Done!")
