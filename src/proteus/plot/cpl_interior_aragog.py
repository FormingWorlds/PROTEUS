from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import MultipleLocator
from proteus.utils.plot import latex_float, sample_output
from proteus.interior.aragog import read_ncdfs

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_interior_aragog(output_dir: str, times: list | np.ndarray, data:list, plot_format: str="pdf"):

    if np.amax(times) < 2:
        log.debug("Insufficient data to make plot_interior")
        return

    log.info("Plot interior")

    # Init figure
    scale = 1.0
    fig,axs = plt.subplots(1,4, figsize=(10*scale,6*scale), sharey=True)

    # Create colormapping stuff
    norm = mpl.colors.LogNorm(vmin=max(1,times[0]), vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    visc_min, visc_max = 0.1, 1.0

    lw=1.5

    # loop over times
    for i,time in enumerate(times):

        # Get decorators
        label = latex_float(time)+" yr"
        color = sm.to_rgba(time)

        # Get interior data for this time
        fpath = os.path.join(output_dir,"data","%d_int.nc"%time)
        ds = data[i]

        # Pressure [GPa] grid
        xx_pres = ds["pres_b"]

        # Phase masks
        yy = np.array(ds["phi_b"])
        MASK_SO = yy < 0.05
        MASK_MI = (0.05 <= yy) & ( yy <= 0.95)
        MASK_ME = yy > 0.95

        # Widen mixed phase region by 1 index so that lines are continuous on the plot
        MASK_MI[2:-2] = MASK_MI[2:-2] | MASK_MI[1:-3] | MASK_MI[3:-1]

        # Depth [km] grid
        xx_radius = ds["radius_b"][:]
        xx_depth = xx_radius[-1] - xx_radius

        # Plot temperature
        yy = ds["temp_b"][:] / 1e3 # convert to kK
        axs[0].plot( yy[MASK_SO], xx_pres[MASK_SO], ls='solid',  c=color, lw=lw, label=label )
        axs[0].plot( yy[MASK_MI], xx_pres[MASK_MI], ls='dashed', c=color, lw=lw)
        axs[0].plot( yy[MASK_ME], xx_pres[MASK_ME], ls='dotted', c=color, lw=lw)

        # Plot melt fraction
        yy = ds["phi_b"][:] * 100
        axs[1].plot( yy[MASK_SO], xx_pres[MASK_SO], ls='solid',  c=color, lw=lw)
        axs[1].plot( yy[MASK_MI], xx_pres[MASK_MI], ls='dashed',  c=color, lw=lw)
        axs[1].plot( yy[MASK_ME], xx_pres[MASK_ME], ls='dotted',  c=color, lw=lw)

        # Plot viscosity
        visc_const = 1 # this is used for the arcsinh scaling
        yy = 10**ds["log10visc_b"][:]
        axs[2].plot( yy[MASK_SO], xx_pres[MASK_SO], ls='solid',  c=color, lw=lw)
        axs[2].plot( yy[MASK_MI], xx_pres[MASK_MI], ls='dashed',  c=color, lw=lw)
        axs[2].plot( yy[MASK_ME], xx_pres[MASK_ME], ls='dotted',  c=color, lw=lw)
        visc_min = min(visc_min, np.amin(yy))
        visc_max = max(visc_max, np.amax(yy))

        # Plot convective flux
        yy = ds["Fconv_b"][:]
        axs[3].plot( yy[MASK_SO], xx_pres[MASK_SO], ls='solid',  c=color, lw=lw)
        axs[3].plot( yy[MASK_MI], xx_pres[MASK_MI], ls='dashed',  c=color, lw=lw)
        axs[3].plot( yy[MASK_ME], xx_pres[MASK_ME], ls='dotted',  c=color, lw=lw)

    # Decorate figure
    title = '(a) Temperature' #'(a) Temperature, {}'.format(units)
    axs[0].set( title=title, xlabel=r'$T$ [kK]', ylabel=r'$P$ [GPa]')
    axs[0].set_ylim(top=np.amin(xx_pres), bottom=np.amax(xx_pres))
    axs[0].yaxis.set_minor_locator(MultipleLocator(10.0))
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[0].legend( fontsize=8, fancybox=True, framealpha=0.9, loc='lower left')

    title = '(b) Melt fraction'
    axs[1].set(title=title, xlabel=r'$\phi$ [%]')
    axs[1].set_xlim(left=-5, right=105)
    axs[1].xaxis.set_major_locator(MultipleLocator(25))
    axs[1].xaxis.set_minor_locator(MultipleLocator(5))

    axs[1].plot([-100,-200],[-100,-200], ls='solid',  c='k', lw=lw, label="Solid")
    axs[1].plot([-100,-200],[-100,-200], ls='dashed', c='k', lw=lw, label="Mush")
    axs[1].plot([-100,-200],[-100,-200], ls='dotted', c='k', lw=lw, label="Melt")
    axs[1].legend(fontsize=8, fancybox=True, framealpha=0.9, loc='lower left')

    title = '(c) Viscosity'
    axs[2].set( title=title, xlabel=r'$\eta$ [Pa s]')
    if visc_max > 100.0*visc_min:
        axs[2].set_xscale("log")

    title = '(d) Convective flux'
    axs[3].set( title=title, xlabel=r'$F_c$ [W m$^{-2}$]')
    axs[3].set_xscale("symlog")

    # Pressure-depth conversion for y-axis
    ax3b = axs[3].twinx()
    ax3b.plot( yy, xx_depth, alpha=0.0)
    ax3b.set_ylim(top=np.amin(xx_depth), bottom=np.amax(xx_depth))
    ax3b.yaxis.set_minor_locator(MultipleLocator(100.0))
    ax3b.set_ylabel( '$d$ [km]')

    # Save figure
    fig.subplots_adjust(wspace=0.05)
    plt.close()
    plt.ioff()
    fpath = os.path.join(output_dir, "plot_interior.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


def plot_interior_aragog_entry(handler: Proteus):
    plot_times,_ = sample_output(handler, extension='_int.nc', tmin=1e3)
    print("Snapshots:", plot_times)

    ncdfs = read_ncdfs(handler.directories["output"],plot_times)

    plot_interior_aragog(
        output_dir=handler.directories['output'],
        times=plot_times,
        data=ncdfs,
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_interior_aragog_entry(handler)
