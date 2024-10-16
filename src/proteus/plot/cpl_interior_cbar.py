from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from proteus.interior.spider import MyJSON

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)


def plot_interior_cbar(output_dir, plot_format="pdf"):

    log.info("Plot interior colourbar")

    # Gather data
    output_files = glob.glob(output_dir+"/data/*.json")
    output_times = [ int(str(f).split('/')[-1].split('.')[0]) for f in output_files]
    sort_mask = np.argsort(output_times)
    sorted_files = np.array(output_files)[sort_mask]
    sorted_times = np.array(output_times)[sort_mask]

    if len(sorted_times) < 3:
        log.warning("Too few samples to make interior_cbar plot")
        return

    stride = int(50)
    sorted_files = sorted_files[::stride]
    sorted_times = sorted_times[::stride]
    nfiles = len(sorted_times)

    # Initialise plot
    fig,(ax1,ax2,ax3,ax4) = plt.subplots(1,4, sharey=True, figsize=(10,5))

    ax1.set_ylabel("Interior pressure, $P$ [GPa]")
    ax1.invert_yaxis()
    ax1.set_xlabel("$T$ [K]")
    ax1.set_title("(a) Temperature")

    ax2.set_xlabel(r"$\phi$")
    ax2.set_title("(b) Melt fraction")

    ax3.set_xlabel(r"$\eta$ [Pa s]")
    ax3.set_title("(c) Viscosity")
    ax3.set_xscale("log")

    ax4.set_xlabel("$S$ [J K$^{-1}$ kg$^{-1}$]")
    ax4.set_title("(d) Specific entropy")

    # Colour mapping
    norm = mpl.colors.Normalize(vmin=1.0, vmax=np.amax(output_times))
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm) #
    sm.set_array([])

    # Loop through years
    for i in range(nfiles):

        # Get colour
        a = 0.6
        lw = 2.0
        c = sm.to_rgba(sorted_times[i])

        # Get data
        myjson_o = MyJSON( output_dir+"/data/%d.json"%sorted_times[i])

        # Process data
        xx_pres = myjson_o.get_dict_values(['data','pressure_b']) * 1.0E-9
        xx_pres_s = myjson_o.get_dict_values(['data','pressure_s']) * 1.0E-9

        MASK_ME = myjson_o.get_melt_phase_boolean_array(  'basic' )
        MASK_MI = myjson_o.get_mixed_phase_boolean_array( 'basic' )
        MASK_SO = myjson_o.get_solid_phase_boolean_array( 'basic' )

        y_tmp = myjson_o.get_dict_values(['data','temp_b'])
        y_phi = myjson_o.get_dict_values(['data','phi_b'])
        y_vis = myjson_o.get_dict_values(['data','visc_b'])
        y_ent = myjson_o.get_dict_values(['data','S_s'])

        # Plot data for this year
        ax1.plot( y_tmp[MASK_SO], xx_pres[MASK_SO], linestyle='solid',  color=c, lw=lw, alpha=a )
        ax1.plot( y_tmp[MASK_MI], xx_pres[MASK_MI], linestyle='dashed', color=c, lw=lw, alpha=a )
        ax1.plot( y_tmp[MASK_ME], xx_pres[MASK_ME], linestyle='dotted', color=c, lw=lw, alpha=a )
        ax2.plot( y_phi, xx_pres,   color=c, lw=lw, alpha=a)
        ax3.plot( y_vis, xx_pres,   color=c, lw=lw, alpha=a)
        ax4.plot( y_ent, xx_pres_s, color=c, lw=lw, alpha=a)

    # Adjust axes
    ax1.set_ylim([np.amax(xx_pres), np.amin(xx_pres)])


    # Plot colourbar
    cax = inset_axes(ax1, width="10%", height="40%", loc='lower left', borderpad=1.0)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label("Time [yr]")

    # Save plot
    fname = os.path.join(output_dir,"plot_interior_cbar.%s"%plot_format)
    fig.subplots_adjust(top=0.94, bottom=0.11, right=0.99, left=0.07, wspace=0.1)
    fig.savefig(fname, dpi=200)

def plot_interior_cbar_entry(handler: Proteus):
    # Plot fixed set from above
    plot_interior_cbar(
        output_dir=handler.directories["output"],
        plot_format=handler.config.params.out.plot_fmt,
    )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_interior_cbar_entry(handler)
