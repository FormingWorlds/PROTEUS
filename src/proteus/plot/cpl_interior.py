from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm

from proteus.utils.plot import FigureData, MyFuncFormatter, latex_float
from proteus.utils.spider import MyJSON, get_all_output_times

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("PROTEUS")


def plot_interior(output_dir: str, times: list | np.ndarray, plot_format: str="pdf"):

    log.info("Plot interior")

    width = 12.00 #* 3.0/2.0
    height = 6.0
    fig_o = FigureData( 1, 4, width, height, output_dir+'/'+'plot_interior', units='kyr' ) #, times
    fig_o.fig.subplots_adjust(wspace=0.15,hspace=0.2)
    fig_o.time = times

    ax0 = fig_o.ax[0]
    ax1 = fig_o.ax[1]
    ax2 = fig_o.ax[2]
    ax3 = fig_o.ax[3]

    time = fig_o.time[0] # first timestep since liquidus and solidus
                         # are time-independent

    myjson_o = MyJSON( output_dir+'/data/{}.json'.format(time) )

    xx_pres = myjson_o.get_dict_values(['data','pressure_b'])
    xx_pres *= 1.0E-9
    xx_pres_s = myjson_o.get_dict_values(['data','pressure_s'])
    xx_pres_s *= 1.0E-9

    xx_radius = myjson_o.get_dict_values(['data','radius_b'])
    xx_radius *= 1.0E-3
    xx_depth = xx_radius[0] - xx_radius
    xx_radius_s = myjson_o.get_dict_values(['data','radius_s'])
    xx_radius_s *= 1.0E-3

    handle_l = []

    fig_o.set_cmap(cm.batlowK_r)

    for nn, time in enumerate( fig_o.time ):

        # read json
        myjson_o = MyJSON( output_dir+'/data/{}.json'.format(time) )

        color = fig_o.get_color( 1.0*nn/len(fig_o.time) )

        # use melt fraction to determine mixed region
        MASK_MI = myjson_o.get_mixed_phase_boolean_array( 'basic' )
        MASK_ME = myjson_o.get_melt_phase_boolean_array(  'basic' )
        MASK_SO = myjson_o.get_solid_phase_boolean_array( 'basic' )

        # label = fig_o.get_legend_label( time )
        label = latex_float(time)+" yr"

        # temperature
        yy = myjson_o.get_dict_values(['data','temp_b'])
        handle, = ax0.plot( yy[MASK_SO], xx_pres[MASK_SO], linestyle='solid',  color=color, lw=1.5, label=label )
        ax0.plot(           yy[MASK_MI], xx_pres[MASK_MI], linestyle='dashed', color=color, lw=1.5 )
        ax0.plot(           yy[MASK_ME], xx_pres[MASK_ME], linestyle='dotted', color=color, lw=1.5 )

        # melt fraction
        yy = myjson_o.get_dict_values(['data','phi_b'])
        ax1.plot( yy, xx_pres, '-', color=color, lw=1.5 )
        # ax1.fill_betweenx( yy, xx_pres, color=color , alpha=0.4)

        # viscosity
        visc_const = 1 # this is used for the arcsinh scaling
        visc_fmt = MyFuncFormatter( visc_const )
        yy = myjson_o.get_dict_values(['data','visc_b'], visc_fmt)
        ax2.plot( yy, xx_pres, '-', color=color, lw=1.5 )

        # entropy
        # plot staggered, since this is where entropy is defined
        # cell-wise and we can easily see the CMB boundary condition
        yy = myjson_o.get_dict_values(['data','S_s'])
        ax3.plot( yy, xx_pres_s, '-', color=color, lw=1.5 )

        # legend
        handle_l.append( handle )

    yticks = [0,20,40,60,80,100,120,int(xx_pres_s[-1])]
    ymax = int(xx_pres_s[-1])

    # titles and axes labels, legends, etc

    units = myjson_o.get_dict_units(['data','temp_b'])
    title = '(a) Temperature' #'(a) Temperature, {}'.format(units)
    xticks = [200, 1000, 2000, 3000, 4000, 5000]
    fig_o.set_myaxes( ax0, title=title, xlabel='$T$ ['+str(units).strip()+']', ylabel='$P$\n[GPa]', xticks=xticks, ymax=ymax, yticks=yticks ) # , xmin=300, xmax=4200
    ax0.set_xlim( 200, 5000 )
    ax0.yaxis.set_label_coords(-0.25,0.5)
    ax0.invert_yaxis()
    # fig_o.set_mylegend( ax0, handle_l, loc=3, ncol=1 )
    ax0.legend( fontsize=8, fancybox=True, framealpha=0.5, loc=3 )

    title = '(b) Melt fraction'
    xticks = [0,0.2,0.4,0.6,0.8,1.0]
    fig_o.set_myaxes( ax1, title=title, xlabel=r'$\phi$', xticks=xticks, ymax=ymax, yticks=yticks )
    # ax1.yaxis.set_label_coords(-0.075,0.475)
    ax1.set_xlim( [-0.05, 1.05] )
    ax1.set_yticklabels([])
    ax1.invert_yaxis()

    units = myjson_o.get_dict_units(['data','visc_b'])
    title = '(c) Viscosity'#'(c) Viscosity, ' + units
    fig_o.set_myaxes( ax2, title=title, xticks=[1.0E1, 1.0E5, 1.0E10, 1.0E15, 1.0E19, 1.0E22], xlabel=r'$\eta$ ['+str(units).strip()+']', yticks=[0,20,40,60,80,100,120,140], xfmt=visc_fmt )
    ax2.set_yticklabels([])
    ax2.invert_yaxis()

    units = myjson_o.get_dict_units(['data','S_b'])
    title = '(d) Specific entropy'#'(d) Entropy, {}'.format(units)
    xticks = [400,800,1600,2400,3200]
    fig_o.set_myaxes( ax3, title=title, xlabel='$S$ ['+str(units).strip()+']', xticks=xticks, ymax=ymax, yticks=yticks ) # ' $(J \; kg^\mathrm{-1} \; K^\mathrm{-1})$'
    ax3.set_yticklabels([])
    ax3.invert_yaxis()

    # Pressure-depth conversion for y-axis
    ax3b = ax3.twinx()
    yy = myjson_o.get_dict_values(['data','S_b'])
    ax3b.plot( yy, xx_depth, alpha=0.0)
    ax3b.set_ylim(top=xx_depth[-1], bottom=xx_depth[0])
    ax3b.set_xlim(right=np.max(xticks), left=np.min(xticks))
    ax3b.set_yticks([0, 500, 1000, 1500, 2000, 2500, int(xx_depth[-1])])
    ax3b.set_ylabel( '$d$\n[km]', rotation=0 )
    ax3b.yaxis.set_label_coords(1.30,0.55)
    ax3b.invert_yaxis()

    plt.close()
    plt.ioff()
    fig_o.savefig(1, plot_format)


def plot_interior_entry(handler: Proteus):
    output_list = get_all_output_times(handler.directories['output'])

    if len(output_list) <= 8:
        times = output_list
    else:
        times = [ output_list[0] ]
        for f in [15,25,33,50,66,75]:
            times.append(output_list[int(round(len(output_list)*(float(f)/100.)))])
        times.append(output_list[-1])

        print("Snapshots:", times)

    plot_interior(
        output_dir=handler.directories['output'],
        times=times,
        plot_format=handler.config["plot_format"],
    )


if __name__ == "__main__":
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_interior_entry(handler)
