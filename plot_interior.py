#!/usr/bin/env python

import spider_coupler_utils as su
import coupler_utils
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import sys
import glob
from natsort import natsorted #https://pypi.python.org/pypi/natsort
from decimal import Decimal
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

# Define Crameri colormaps
folder = "/Users/tim/Dropbox/work/Projects/20_greenedge/colormaps/ScientificColourMaps5/"
__all__ = {'acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc', 'buda',
           'cork', 'davos', 'devon', 'grayC', 'hawaii', 'imola', 'lajolla',
           'lapaz', 'lisbon', 'nuuk', 'oleron', 'oslo', 'roma', 'tofino',
           'tokyo', 'turku', 'vik'}
for name in __all__:
    file = os.path.join(folder, name, name + '.txt')
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[name+"_r"] = LinearSegmentedColormap.from_list(name, cm_data[::-1])

logger = su.get_my_logger(__name__)

#====================================================================
def mantle_evolution( times ):

    # article class text width is 4.7747 inches
    # http://tex.stackexchange.com/questions/39383/determine-text-width

    logger.info( 'building mantle_evolution' )

    width = 12.00 #* 3.0/2.0
    height = 6.0
    fig_o = su.FigureData( 1, 4, width, height, 'output/'+'interior_evolution', units='kyr' ) #, times
    fig_o.fig.subplots_adjust(wspace=0.15,hspace=0.2)
    fig_o.time = times

    ax0 = fig_o.ax[0]
    ax1 = fig_o.ax[1]
    ax2 = fig_o.ax[2]
    ax3 = fig_o.ax[3]

    time = fig_o.time[0] # first timestep since liquidus and solidus
                         # are time-independent

    myjson_o = su.MyJSON( 'output/{}.json'.format(time) )

    # print(myjson_o.data_d)
    # TIMEYRS = myjson_o.data_d['nstep']

    # hack to compute some average properties for Bower et al. (2018)
    #xx_liq, yy_liq = fig_o.get_xy_data( 'liquidus_rho', time )
    #xx_sol, yy_sol = fig_o.get_xy_data( 'solidus_rho', time )
    #diff = (yy_liq - yy_sol) / yy_sol * 100.0
    #print diff[40:]
    #print np.mean(diff[40:])
    #sys.exit(1)

    xx_pres = myjson_o.get_dict_values_internal(['data','pressure_b'])
    xx_pres *= 1.0E-9

    xx_pres_s = myjson_o.get_dict_values(['data','pressure_s'])
    xx_pres_s *= 1.0E-9

    # shade grey between liquidus and solidus
    yy_liq = myjson_o.get_dict_values_internal(['data','liquidus_b'])
    yy_sol = myjson_o.get_dict_values_internal(['data','solidus_b'])
    #ax0.fill_between( xx_pres, yy_liq, yy_sol, facecolor='grey', alpha=0.35, linewidth=0 )
    yy_liqt = myjson_o.get_dict_values_internal(['data','liquidus_temp_b'])
    yy_solt = myjson_o.get_dict_values_internal(['data','solidus_temp_b'])
    #ax1.fill_between( xx_pres, yy_liqt, yy_solt, facecolor='grey', alpha=0.35, linewidth=0 )
    # hack to compute some average properties for Bower et al. (2018)
    #print xx_sol
    #print np.mean(yy_liq[20:]-yy_sol[20:])
    #sys.exit(1)

    # dotted lines of constant melt fraction
    #for xx in range( 0, 11, 2 ):
    #    yy_b = xx/10.0 * (yy_liq - yy_sol) + yy_sol
    #    if xx == 0:
    #        # solidus
    #        ax0.plot( xx_pres, yy_b, '-', linewidth=0.5, color='black' )
    #    elif xx == 3:
    #        # typically, the approximate location of the rheological transition
    #        ax0.plot( xx_pres, yy_b, '-', linewidth=1.0, color='white')
    #    elif xx == 10:
    #        # liquidus
    #        ax0.plot( xx_pres, yy_b, '-', linewidth=0.5, color='black' )
    #    else:
    #        # dashed constant melt fraction lines
    #        ax0.plot( xx_pres, yy_b, '--', linewidth=1.0, color='white' )
    ax0.plot( xx_pres, yy_sol, '-', linewidth=1.0, color='black' )
    ax1.plot( xx_pres, yy_solt, '-', linewidth=1.0, color='black' )

    handle_l = [] # handles for legend

    # folder = "/Users/tim/Dropbox/work/Projects/20_greenedge/colormaps/ScientificColourMaps5/"
    #
    # __all__ = {'acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc', 'buda',
    #            'cork', 'davos', 'devon', 'grayC', 'hawaii', 'imola', 'lajolla',
    #            'lapaz', 'lisbon', 'nuuk', 'oleron', 'oslo', 'roma', 'tofino',
    #            'tokyo', 'turku', 'vik'}
    #
    # file = os.path.join("/Users/tim/Dropbox/work/Projects/20_greenedge/colormaps/ScientificColourMaps5/", 'roma', 'roma' + '.txt')
    # cm_data = np.loadtxt(file)
    # roma = LinearSegmentedColormap.from_list('roma', cm_data)

    fig_o.set_colors(cmap=vik_r) # "magma_r"

    for nn, time in enumerate( fig_o.time ):

        # read json
        myjson_o = su.MyJSON( 'output/{}.json'.format(time) )

        color = fig_o.get_color( nn )
        # use melt fraction to determine mixed region
        MIX = myjson_o.get_mixed_phase_boolean_array( 'basic_internal' )
        MIX_s = myjson_o.get_mixed_phase_boolean_array( 'staggered' )

        # label = fig_o.get_legend_label( time )
        # label = "{:.1e}".format(Decimal(time))+" yr"
        label = coupler_utils.latex_float(time)+" yr"

        # temperature
        yy = myjson_o.get_dict_values_internal(['data','temp_b'])
        ax0.plot( yy, xx_pres, '--', color=color )
        ax0.plot( yy*MIX, xx_pres*MIX, '-', color=color )

        # melt fraction
        yy = myjson_o.get_dict_values_internal(['data','phi_b'])
        ax1.plot( yy, xx_pres, '-', color=color )

        # viscosity
        visc_const = 1 # this is used for the arcsinh scaling
        visc_fmt = su.MyFuncFormatter( visc_const )
        yy = myjson_o.get_dict_values_internal(['data','visc_b'], visc_fmt)
        ax2.plot( yy, xx_pres, '-', color=color )
        # visc_const = 1 # this is used for the arcsinh scaling
        # visc_fmt = su.MyFuncFormatter( visc_const )
        # yy = myjson_o.get_dict_values_internal(['data','visc_b'], visc_fmt)
        # ax2.plot( xx_pres, yy, '-', color=color )

        # entropy
        # plot staggered, since this is where entropy is defined
        # cell-wise and we can easily see the CMB boundary condition
        yy = myjson_o.get_dict_values(['data','S_s'])
        #yy = myjson_o.get_dict_values_internal('S_b')
        ax3.plot( yy, xx_pres_s, '-', color=color )
        handle, = ax3.plot( yy*MIX_s, xx_pres_s*MIX_s, '-', color=color, label=label )
        handle_l.append( handle )

    yticks = [0,20,40,60,80,100,120,135]
    ymax = 135

    # titles and axes labels, legends, etc

    units = myjson_o.get_dict_units(['data','temp_b'])
    title = '(a) Temperature' #'(a) Temperature, {}'.format(units)
    xticks = [200, 1000, 2000, 3000, 4000, 5000]
    fig_o.set_myaxes( ax0, title=title, xlabel='$T$, '+units, ylabel='$P$\n(GPa)', xticks=xticks, ymax=ymax, yticks=yticks ) # , xmin=300, xmax=4200
    ax0.set_xlim( 200, 5000 )
    ax0.yaxis.set_label_coords(-0.255,0.5)
    ax0.invert_yaxis()
    fig_o.set_mylegend( ax0, handle_l, loc=3, ncol=1 )

    # fig_o.set_mylegend( ax2, handle_l, loc=4, ncol=2 )
    #fig_o.set_mylegend( ax0, handle_l, loc=2, ncol=2 )
    title = '(b) Melt fraction'
    xticks = [0,0.2,0.4,0.6,0.8,1.0]
    fig_o.set_myaxes( ax1, title=title, xlabel='$\phi$', xticks=xticks, ymax=ymax, yticks=yticks )
    # ax1.yaxis.set_label_coords(-0.075,0.475)
    ax1.set_xlim( [0, 1] )
    ax1.set_yticklabels([])
    ax1.invert_yaxis()

    # units = myjson_o.get_dict_units(['data','visc_b'])
    # title = '(d) Viscosity, ' + units
    # yticks = [1.0E1, 1.0E4, 1.0E8, 1.0E12, 1.0E16, 1.0E20, 1.0E22]
    # fig_o.set_myaxes( ax2, title=title, xlabel='$P$ (GPa)', yticks=yticks,
    #     ylabel='$\eta$', xticks=[0,20,40,60,80,100,120,140], xmax=140, fmt=visc_fmt )
    # ax2.yaxis.set_label_coords(-0.075,0.67)

    units = myjson_o.get_dict_units(['data','visc_b'])
    title = '(c) Viscosity'#'(c) Viscosity, ' + units
    # ax2.yticks=[0,20,40,60,80,100,120,140]
    # ax2.xticks=[1.0E1, 1.0E4, 1.0E8, 1.0E12, 1.0E16, 1.0E20, 1.0E22]
    fig_o.set_myaxes( ax2, title=title, xticks=[1.0E1, 1.0E5, 1.0E10, 1.0E15, 1.0E19, 1.0E22], xlabel='$\eta$, '+ units, yticks=[0,20,40,60,80,100,120,140], xfmt=visc_fmt )
    ax2.set_yticklabels([])
    ax2.invert_yaxis()

    units = myjson_o.get_dict_units(['data','S_b'])
    title = '(d) Entropy'#'(d) Entropy, {}'.format(units)
    xticks = [400,800,1600,2400,3200]
    fig_o.set_myaxes( ax3, title=title, xlabel='$S$, '+ units, xticks=xticks, ymax=ymax, yticks=yticks ) # ' $(J \; kg^\mathrm{-1} \; K^\mathrm{-1})$'
    ax3.set_yticklabels([])
    ax3.invert_yaxis()

    # if 0:
    #     # add sol and liq text boxes to mark solidus and liquidus
    #     # these have to manually adjusted according to the solidus
    #     # and liquidus used for the model
    #     prop_d = {'boxstyle':'round', 'facecolor':'white', 'linewidth': 0}
    #     ax0.text(120, 2925, r'liq', fontsize=8, bbox=prop_d )
    #     ax0.text(120, 2075, r'sol', fontsize=8, bbox=prop_d )
    #     # DJB commented out for now, below labels 0.2 melt fraction
    #     # so the reader knows that dashed white lines denote melt
    #     # fraction contours
    #     ax0.text(110, 2275, r'$\phi=0.2$', fontsize=6 )

    fig_o.savefig(1)

#====================================================================
def main():

    # output_list = [os.path.basename(x) for x in natsorted(glob.glob("output/"+"*.json"))]
    # output_list = [ int(x[0:-5]) for x in output_list ]
    # print(output_list)
    # plot_list = [ output_list[0], output_list[int(round(len(output_list)*(1./5.)))], output_list[int(round(len(output_list)*(2./5.)))], output_list[int(round(len(output_list)*(3./5.)))], output_list[int(round(len(output_list)*(4./5.)))], output_list[-1] ]

    output_list = su.get_all_output_times()
    # plot_list = str(output_list[0])+","+str(output_list[int(round(len(output_list)*(1./5.)))])+","+str( output_list[int(round(len(output_list)*(2./5.)))])+","+str(output_list[int(round(len(output_list)*(3./5.)))])+","+str(output_list[int(round(len(output_list)*(4./5.)))])+","+str(output_list[-1])
    plot_list = [ output_list[0], output_list[int(round(len(output_list)*(2./100.)))], output_list[int(round(len(output_list)*(15./100.)))], output_list[int(round(len(output_list)*(22./100.)))], output_list[int(round(len(output_list)*(30./100.)))], output_list[int(round(len(output_list)*(45./100.)))], output_list[int(round(len(output_list)*(66./100.)))], output_list[-1] ]
    print("Snapshot:", plot_list)

    # arguments (run with -h to summarize)
    # parser = argparse.ArgumentParser(description='SPIDER plotting script')
    # parser.add_argument('-t', '--times', type=str, help='Comma-separated (no spaces) list of times');
    # parser.add_argument('-f3', '--fig3', help='Plot figure 3', action="store_true")
    # args = parser.parse_args()
    #
    # print(args.times)

    # # if nothing specified, choose a default set
    # if not args.fig3:
    #     args.fig3 = True;
    #
    # if args.fig3 :
    #     if not args.times:
    #         logger.critical( 'You must specify times in a comma-separated list (no spaces) using -t' )
    #         sys.exit(0)

    # # reproduce staple figures in Bower et al. (2018)
    # # i.e., figs 3,4,5,6,7,8
    # if args.fig3 :
    #     solid_evolution_fig3( times=args.times )

    # output_dir = parser.parse_args().dir

    mantle_evolution( times=plot_list )

    # plt.show()

#====================================================================

if __name__ == "__main__":

    main()
