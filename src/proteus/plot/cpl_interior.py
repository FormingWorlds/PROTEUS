from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import MultipleLocator
from proteus.utils.plot import latex_float, sample_output, MyFuncFormatter
from proteus.interior.wrapper import read_interior_data

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

def plot_interior(output_dir: str, times: list | np.ndarray, data:list, module:str, plot_format: str="pdf"):

    if np.amax(times) < 2:
        log.debug("Insufficient data to make plot_interior")
        return
    if module not in ["spider","aragog"]:
        log.warning(f"Cannot make interior plot for module '{module}'")
        return

    log.info("Plot interior")

    # Init figure
    scale = 1.0
    fig,axs = plt.subplots(1,4, figsize=(10*scale,6*scale), sharey=True)

    # Create colormapping stuff
    norm = mpl.colors.LogNorm(vmin=max(1,times[0]), vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    visc_min, visc_max = 1e99, 0
    flux_min, flux_max = 1e99, 0

    lw=1.5

    # loop over times
    for i,time in enumerate(times):

        # Get decorators
        label = latex_float(time)+" yr"
        color = sm.to_rgba(time)

        # Get interior data for this time
        ds = data[i]

        # Pressure [GPa] grid
        if module == "aragog":
            xx_pres = ds["pres_b"]
        elif module == "spider":
            xx_pres = ds.get_dict_values(['data','pressure_b']) * 1e-9

        # Phase masks
        if module == "aragog":
            yy = np.array(ds["phi_b"])
            MASK_SO = yy < 0.05
            MASK_MI = (0.05 <= yy) & ( yy <= 0.95)
            MASK_ME = yy > 0.95
        elif module == "spider":
            MASK_MI = ds.get_mixed_phase_boolean_array( 'basic' )
            MASK_ME = ds.get_melt_phase_boolean_array(  'basic' )
            MASK_SO = ds.get_solid_phase_boolean_array( 'basic' )

        # Widen mixed phase region by 1 index so that lines are continuous on the plot
        MASK_MI[2:-2] = MASK_MI[2:-2] | MASK_MI[1:-3] | MASK_MI[3:-1]

        # Depth [km] grid
        if module == "aragog":
            xx_radius = ds["radius_b"][:]
            xx_depth = xx_radius[-1] - xx_radius
        elif module == "spider":
           xx_radius = ds.get_dict_values(['data','radius_b']) * 1e-3
           xx_depth = xx_radius[0] - xx_radius

        # Plot temperature
        if module == "aragog":
            yy = ds["temp_b"][:]
        elif module == "spider":
            yy = ds.get_dict_values(['data','temp_b'])
        yy /= 1e3 # convert to kK
        axs[0].plot( yy[MASK_SO], xx_pres[MASK_SO], ls='solid',  c=color, lw=lw, label=label )
        axs[0].plot( yy[MASK_MI], xx_pres[MASK_MI], ls='dashed', c=color, lw=lw)
        axs[0].plot( yy[MASK_ME], xx_pres[MASK_ME], ls='dotted', c=color, lw=lw)

        # Plot melt fraction
        if module == "aragog":
            yy = ds["phi_b"][:]
        elif module == "spider":
            yy = ds.get_dict_values(['data','phi_b'])
        yy *= 100.0 # convert to percentage
        axs[1].plot( yy[MASK_SO], xx_pres[MASK_SO], ls='solid',  c=color, lw=lw)
        axs[1].plot( yy[MASK_MI], xx_pres[MASK_MI], ls='dashed',  c=color, lw=lw)
        axs[1].plot( yy[MASK_ME], xx_pres[MASK_ME], ls='dotted',  c=color, lw=lw)

        # Plot viscosity
        if module == "aragog":
            yy = 10**ds["log10visc_b"][:]
        elif module == "spider":
            visc_const = 1 # this is used for the arcsinh scaling
            yy = ds.get_dict_values(['data','visc_b'], MyFuncFormatter( visc_const ))
        axs[2].plot( yy[MASK_SO], xx_pres[MASK_SO], ls='solid',  c=color, lw=lw)
        axs[2].plot( yy[MASK_MI], xx_pres[MASK_MI], ls='dashed',  c=color, lw=lw)
        axs[2].plot( yy[MASK_ME], xx_pres[MASK_ME], ls='dotted',  c=color, lw=lw)
        visc_min = min(visc_min, np.amin(yy))
        visc_max = max(visc_max, np.amax(yy))

        # Plot convective flux
        if module == "aragog":
            yy = ds["Fconv_b"][:]
        elif module == "spider":
            yy =  ds.get_dict_values(['data','Jconv_b'])
        axs[3].plot( yy[MASK_SO], xx_pres[MASK_SO], ls='solid',   c=color, lw=lw)
        axs[3].plot( yy[MASK_MI], xx_pres[MASK_MI], ls='dashed',  c=color, lw=lw)
        axs[3].plot( yy[MASK_ME], xx_pres[MASK_ME], ls='dotted',  c=color, lw=lw)
        flux_min = min(flux_min, np.amin(yy))
        flux_max = max(flux_max, np.amax(yy))

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
    if flux_max > 100.0*flux_min:
        axs[3].set_xscale("log")

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


def plot_interior_entry(handler: Proteus):
    module = handler.config.interior.module
    if module == "spider":
        extension = ".json"
    elif module == "aragog":
        extension = "_int.nc"
    plot_times,_ = sample_output(handler, extension=extension, tmin=1e3)
    print("Snapshots:", plot_times)

    data = read_interior_data(handler.directories["output"], module, plot_times)

    plot_interior(
        output_dir=handler.directories['output'],
        times=plot_times,
        data=data,
        module=module,
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_interior_entry(handler)
