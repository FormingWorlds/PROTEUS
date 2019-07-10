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

# Define Crameri colormaps (+ recursive)
from matplotlib.colors import LinearSegmentedColormap
for name in [ 'acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc', 'buda',
           'cork', 'davos', 'devon', 'grayC', 'hawaii', 'imola', 'lajolla',
           'lapaz', 'lisbon', 'nuuk', 'oleron', 'oslo', 'roma', 'tofino',
           'tokyo', 'turku', 'vik' ]:
    file = os.path.join("plotting_tools/ScientificColourMaps5/", name + '.txt')
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[name+"_r"] = LinearSegmentedColormap.from_list(name, cm_data[::-1])

logger = su.get_my_logger(__name__)

#====================================================================
def stacked_evolution( times ):

    # article class text width is 4.7747 inches
    # http://tex.stackexchange.com/questions/39383/determine-text-width

    logger.info( 'building stacked interior atmosphere' )

    width = 4.00 #* 3.0/2.0
    height = 8.0
    fig_o = su.FigureData( 2, 1, width, height, 'output/'+'stacked_interior_atmosphere', units='kyr' ) #, times
    fig_o.fig.subplots_adjust(wspace=0.0,hspace=0.0)
    fig_o.time = times

    ax0 = fig_o.ax[0]
    ax1 = fig_o.ax[1]
    # ax2 = fig_o.ax[2]
    # ax3 = fig_o.ax[3]

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

    xx_radius = myjson_o.get_dict_values_internal(['data','radius_b'])
    xx_radius *= 1.0E-3
    xx_depth = xx_radius[0] - xx_radius
    xx_radius_s = myjson_o.get_dict_values(['data','radius_s'])
    xx_radius_s *= 1.0E-3
    xx_depth_s = xx_radius_s[0] - xx_radius_s

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

    # # dotted lines of constant melt fraction
    # for xx in range( 0, 11, 2 ):
    #    yy_b = xx/10.0 * (yy_liq - yy_sol) + yy_sol
    #    if xx == 0:
    #        # solidus
    #        ax1.plot( yy_b, xx_pres, ':', linewidth=0.5, color='black' )
    #    elif xx == 3:
    #        # typically, the approximate location of the rheological transition
    #        ax1.plot( yy_b, xx_pres, ':', linewidth=1.0, color='white')
    #    elif xx == 10:
    #        # liquidus
    #        ax1.plot( yy_b, xx_pres, ':', linewidth=0.5, color='black' )
    #    else:
    #        # dashed constant melt fraction lines
    #        ax1.plot( yy_b, xx_pres, ':', linewidth=1.0, color='white' )
    # ax1.plot( yy_sol, xx_pres, ':', linewidth=1.0, color='black' )
    # # ax0.plot( yy_solt, xx_pres, ':', linewidth=1.0, color='black' )
    # yy_b = xx/10.0 * (yy_liq - yy_sol) + yy_sol
    # ax1.plot( yy_b, xx_pres, ':', linewidth=1.5, color='black')

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

    ymax_atm = 0

    for nn, time in enumerate( fig_o.time ):

        # Read atmosphere properties
        atm_TP_profile  = np.loadtxt('output/'+str(int(time))+"_atm_TP_profile.dat")
        atm_spectral_flux = np.loadtxt('output/'+str(int(time))+"_atm_spectral_flux.dat")

        temperature_profile = []
        pressure_profile    = []
        band_centres        = []
        spectral_flux       = []
        for i in range(0, len(atm_TP_profile)):
            temperature_profile.append(atm_TP_profile[i][0])
            pressure_profile.append(atm_TP_profile[i][1])
            band_centres.append(atm_spectral_flux[i][0])
            spectral_flux.append(atm_spectral_flux[i][1])

        # read json
        myjson_o = su.MyJSON( 'output/{}.json'.format(time) )

        color = fig_o.get_color( nn )
        # use melt fraction to determine mixed region
        MIX = myjson_o.get_mixed_phase_boolean_array( 'basic_internal' )
        MIX_s = myjson_o.get_mixed_phase_boolean_array( 'staggered' )

        label = coupler_utils.latex_float(time)+" yr"

        # atmosphere
        ax0.plot( temperature_profile, pressure_profile, '-', color=color )

        # interior
        yy = myjson_o.get_dict_values_internal(['data','temp_b'])
        ax1.plot( yy, xx_pres, '--', color=color )
        ax1.plot( yy*MIX, xx_pres*MIX, '-', color=color )

        ymax_atm = np.max([ymax_atm, np.max(pressure_profile)])

    yticks = [0,20,40,60,80,100,120,int(xx_pres_s[-1])]
    ymax = int(xx_pres_s[-1])

    units = myjson_o.get_dict_units(['data','temp_b'])
    title = 'Atmosphere + mantle temperature profile' #'(a) Temperature, {}'.format(units)
    xticks = [200, 1000, 2000, 3000, 4000, 5000]
    fig_o.set_myaxes( ax0, xlabel='$T$, '+units, ylabel='$p$\n(bar)', xticks=xticks, ymin=0, ymax=ymax_atm ) # , xmin=300, xmax=4200, , yticks=yticks, , title=title
    ax0.set_xlim( 200, 5000 )
    ax0.set_ylim( 0, ymax_atm )
    ax0.yaxis.set_label_coords(-0.13,0.5)
    ax0.invert_yaxis()
    # ax0.set_xticklabels([])
    # ax0.set_xticks([])
    # fig_o.set_mylegend( ax0, handle_l, loc=3, ncol=1 )
    ax0.xaxis.set_label_position('top')
    ax0.xaxis.tick_top()

    xticks = [200, 1000, 2000, 3000, 4000, 5000]
    fig_o.set_myaxes( ax1, xlabel='$T$, '+units, ylabel='$d$\n(km)', xticks=xticks, ymax=ymax, yticks=yticks )
    ax0.set_xlim( 200, 5000 )
    # ax1.set_yticklabels([])
    ax1.yaxis.set_label_coords(-0.13,0.5)
    ax1.invert_yaxis()

    fig_o.savefig(1)

#====================================================================
def main():

    output_list = su.get_all_output_times()

    if len(output_list) <= 8:
        plot_list = output_list
    else:
        plot_list = [ output_list[0], output_list[int(round(len(output_list)*(2./100.)))], output_list[int(round(len(output_list)*(15./100.)))], output_list[int(round(len(output_list)*(22./100.)))], output_list[int(round(len(output_list)*(30./100.)))], output_list[int(round(len(output_list)*(45./100.)))], output_list[int(round(len(output_list)*(66./100.)))], output_list[-1] ]
    print("snapshots:", plot_list)

    stacked_evolution( times=plot_list )

#====================================================================

if __name__ == "__main__":

    main()
