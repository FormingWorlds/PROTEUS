from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import numpy as np
from cmcrameri import cm

from proteus.interior.spider import MyJSON
from proteus.utils.plot import latex_float, MyFuncFormatter, sample_output

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_interior(output_dir: str, times: list | np.ndarray, plot_format: str="pdf"):

    log.info("Plot interior")

    # Init figure
    scale = 1.0
    fig,axs = plt.subplots(1,4, figsize=(10*scale,6*scale), sharey=True)

    # Create colormapping stuff
    norm = mpl.colors.LogNorm(vmin=max(1,times[0]), vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    visc_min, visc_max = 0.1, 1.0

    # loop over times
    for time in times:

        # Get interior data for this time
        myjson_o = MyJSON(os.path.join(output_dir, "data", "%d.json"%time))

        # Pressure grid
        xx_pres = myjson_o.get_dict_values(['data','pressure_b'])
        xx_pres *= 1.0E-9
        xx_pres_s = myjson_o.get_dict_values(['data','pressure_s'])
        xx_pres_s *= 1.0E-9

        # Depth grid
        xx_radius = myjson_o.get_dict_values(['data','radius_b'])
        xx_radius *= 1.0E-3
        xx_depth = xx_radius[0] - xx_radius
        xx_radius_s = myjson_o.get_dict_values(['data','radius_s'])
        xx_radius_s *= 1.0E-3

        # use melt fraction to determine mixed region
        MASK_MI = myjson_o.get_mixed_phase_boolean_array( 'basic' )
        MASK_ME = myjson_o.get_melt_phase_boolean_array(  'basic' )
        MASK_SO = myjson_o.get_solid_phase_boolean_array( 'basic' )

        # overlap lines by 1 node
        for m in (MASK_MI, MASK_ME, MASK_SO):
            m_new = m[:]
            for i in range(2,len(MASK_MI)-2,1):
                m_new[i] = m[i] or m[i-1] or m[i+1]
            m = m_new[:]

        # Get decorators
        label = latex_float(time)+" yr"
        color = sm.to_rgba(time)

        # Plot temperature
        yy = myjson_o.get_dict_values(['data','temp_b'])
        axs[0].plot( yy[MASK_SO], xx_pres[MASK_SO], ls='solid',  c=color, lw=1.5, label=label )
        axs[0].plot( yy[MASK_MI], xx_pres[MASK_MI], ls='dashed', c=color, lw=1.5)
        axs[0].plot( yy[MASK_ME], xx_pres[MASK_ME], ls='dotted', c=color, lw=1.5)

        # Plot melt fraction
        yy = myjson_o.get_dict_values(['data','phi_b'])
        axs[1].plot( yy[MASK_SO], xx_pres[MASK_SO], ls='solid',  c=color, lw=1.5)
        axs[1].plot( yy[MASK_MI], xx_pres[MASK_MI], ls='dashed', c=color, lw=1.5)
        axs[1].plot( yy[MASK_ME], xx_pres[MASK_ME], ls='dotted', c=color, lw=1.5)

        # Plot viscosity
        visc_const = 1 # this is used for the arcsinh scaling
        visc_fmt = MyFuncFormatter( visc_const )
        yy = myjson_o.get_dict_values(['data','visc_b'], visc_fmt)
        axs[2].plot( yy[MASK_SO], xx_pres[MASK_SO], ls='solid',  c=color, lw=1.5)
        axs[2].plot( yy[MASK_MI], xx_pres[MASK_MI], ls='dashed', c=color, lw=1.5)
        axs[2].plot( yy[MASK_ME], xx_pres[MASK_ME], ls='dotted', c=color, lw=1.5)
        visc_min = min(visc_min, np.amin(yy))
        visc_max = max(visc_max, np.amax(yy))

        # Plot entropy (at cell centres, rather than edges)
        yy = myjson_o.get_dict_values(['data','S_s'])
        axs[3].plot( yy[MASK_SO[:-1]], xx_pres_s[MASK_SO[:-1]], ls='solid',  c=color, lw=1.5)
        axs[3].plot( yy[MASK_MI[:-1]], xx_pres_s[MASK_MI[:-1]], ls='dashed', c=color, lw=1.5)
        axs[3].plot( yy[MASK_ME[:-1]], xx_pres_s[MASK_ME[:-1]], ls='dotted', c=color, lw=1.5)


    # Decorate figure
    title = '(a) Temperature' #'(a) Temperature, {}'.format(units)
    axs[0].set( title=title, xlabel='$T$ [K]', ylabel='$P$ [GPa]')
    axs[0].set_ylim(top=np.amin(xx_pres), bottom=np.amax(xx_pres))
    axs[0].yaxis.set_minor_locator(MultipleLocator(10.0))
    axs[0].xaxis.set_minor_locator(MultipleLocator(500.0))
    axs[0].legend( fontsize=8, fancybox=True, framealpha=0.5, loc=3 )

    title = '(b) Melt fraction'
    xticks = [0,0.2,0.4,0.6,0.8,1.0]
    axs[1].set(title=title, xlabel=r'$\phi$', xticks=xticks )
    axs[1].set_xlim(left=-0.05, right=1.05)

    title = '(c) Viscosity'
    axs[2].set( title=title, xlabel=r'$\eta$ [Pa s]')
    if visc_max > 100.0*visc_min:
        axs[2].set_xscale("log")

    title = '(d) Specific entropy'
    xticks = [400,800,1600,2400,3200]
    axs[3].set( title=title, xlabel=r'$S$ [J K$^{-1} $kg$^{-1}$]', xticks=xticks)

    # Pressure-depth conversion for y-axis
    ax3b = axs[3].twinx()
    yy = myjson_o.get_dict_values(['data','S_b'])
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
    plot_times,_ = sample_output(handler, ftype='json')
    print("Snapshots:", plot_times)

    plot_interior(
        output_dir=handler.directories['output'],
        times=plot_times,
        plot_format=handler.config["plot_format"],
    )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_interior_entry(handler)
