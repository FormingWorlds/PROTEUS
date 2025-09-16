from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from proteus.utils.constants import const_c, const_h, const_k
from proteus.utils.helper import natural_sort

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

def planck_function(lam, T):
    """Plots stellar flux from `output/` versus time (colorbar)

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


def plot_sflux(output_dir: str, wl_max: float = 6000.0,
                plt_modern:bool=True, plot_format: str="pdf"):
    """Plots stellar flux vs time for all wavelengths

    Note that this function will plot the flux from EVERY file it finds.
    Saves plot as 'cpl_sflux.pdf' in  the output directory.

    Parameters
    ----------
    output_dir : str
        Directory for both reading from and saving to.

    wl_max : float
        Upper limit of wavelength axis [nm]
    plt_modern : bool
        Include modern spectrum in plot?
    plot_format: str
        Output figure file format

    """

    star_cmap = plt.get_cmap('Spectral')

    # Find and sort files
    files_unsorted = glob.glob(output_dir+"/data/*.sflux")
    files = natural_sort(files_unsorted)

    if (len(files) == 0):
        log.warning("Insufficient data to make plot_sflux")
        return

    log.info("Plot stellar flux")

    # Downsample data
    if (len(files) > 200):
        files = files[::2]
    if (len(files) > 500):
        files = files[::2]
    if (len(files) > 1000):
        files = files[::3]

    if (len(files) != len(files_unsorted)):
        log.debug("Downsampled data over time")

    # Arrays for storing data over time
    time_t = []
    wave_t = []
    flux_t = []
    for f in files:
        # Load data
        X = np.loadtxt(f,skiprows=1,delimiter='\t').T

        # Parse data
        time = int(f.split('/')[-1].split('.')[0])
        if time < 0:
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
    N = len(time_t)

    fig,ax = plt.subplots(1,1)

    # Colorbar
    justone = bool(N == 1)
    if not justone:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)

        vmin = max(time_t[0],1.0)
        vmax = time_t[-1]
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=star_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
        cbar.set_label("Time [yr]")
    else:
        log.warning("Only one spectrum was found")

    ax.set_yscale("log")
    ax.set_ylabel(r"TOA spectral flux density [erg / (s cm$^2$ nm)]")

    ax.set_xscale("log")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_xlim([0.5,max(1.0,wl_max)])

    # Plot historical spectra
    for i in range(N):
        if justone:
            c = 'tab:blue'
            label = "%.2e yr"%(time_t[i])
        else:
            c = sm.to_rgba(time_t[i])
            label = None
        ax.plot(wave_t[i],flux_t[i],color=c,lw=0.7,alpha=0.6, label=label)

    # Plot current spectrum (use the copy made in the output directory)
    if plt_modern:
        modern_fpath = os.path.join(output_dir, "data", "-1.sflux")
        if os.path.isfile(modern_fpath):
            X = np.loadtxt(modern_fpath,skiprows=2).T
            ax.plot(X[0],X[1],color='black',label='Modern (1 AU)',lw=0.8,alpha=0.9)
        else:
            log.warning(f"Could not find file {modern_fpath}")

    if plt_modern or justone:
        ax.legend(loc='lower left')

    plt.close()
    plt.ioff()
    fpath = os.path.join(output_dir, "plots", "plot_sflux.%s"%plot_format)
    fig.savefig(fpath, bbox_inches='tight', dpi=200)


def plot_sflux_entry(handler: Proteus):
    plot_sflux(
        output_dir=handler.directories['output'],
        plt_modern=handler.config.star.module == "mors",
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == '__main__':
    print("Plotting stellar flux over time (colorbar)...")

    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_sflux_entry(handler)

    print("Done!")
