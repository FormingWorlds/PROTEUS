#!/usr/bin/env python

import spider_coupler_utils as su
import coupler_utils
import argparse
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import sys
import glob
# from natsort import natsorted #https://pypi.python.org/pypi/natsort
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

    width = 5.00 #* 3.0/2.0
    height = 10.0
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

    xx_pres = myjson_o.get_dict_values(['data','pressure_b'])
    # xx_pres = myjson_o.get_dict_values_internal(['data','pressure_b'])
    xx_pres *= 1.0E-9
    xx_pres_s = myjson_o.get_dict_values(['data','pressure_s'])
    xx_pres_s *= 1.0E-9

    xx_radius = myjson_o.get_dict_values(['data','radius_b'])
    # xx_radius = myjson_o.get_dict_values_internal(['data','radius_b'])
    xx_radius *= 1.0E-3
    xx_depth = xx_radius[0] - xx_radius
    xx_radius_s = myjson_o.get_dict_values(['data','radius_s'])
    xx_radius_s *= 1.0E-3
    xx_depth_s = xx_radius_s[0] - xx_radius_s


    handle_l = [] # handles for legend


    fig_o.set_colors(cmap=vik_r) # "magma_r"

    ymax_atm = 0
    ymin_atm = 0

    for nn, time in enumerate( fig_o.time ):

        if os.path.exists('output/'+str(int(time))+"_atm_TP_profile.dat"):

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
            MIX = myjson_o.get_mixed_phase_boolean_array( 'basic' )
            # MIX = myjson_o.get_mixed_phase_boolean_array( 'basic_internal' )
            # MIX_s = myjson_o.get_mixed_phase_boolean_array( 'staggered' )

            label = coupler_utils.latex_float(time)+" yr"

            # atmosphere
            ax0.semilogy( temperature_profile, pressure_profile, '-', color=color, label=label, lw=1.5)
            # handle_l.append( handle )

            # interior
            yy = myjson_o.get_dict_values(['data','temp_b'])
            # yy_s = myjson_o.get_dict_values_internal(['data','temp_s'])
            ax1.plot( yy, xx_pres, '--', color=color, lw=1.5 )
            ax1.plot( yy*MIX, xx_pres*MIX, '-', color=color, label=label, lw=1.5 )

            # print(time, 'temperature_profile', temperature_profile[-1], temperature_profile[-2])
            # print(time, 'temp_b', yy[0], yy[1])
            # print(time, 'temp_s', yy_s[0], yy_s[1])

            if np.min(pressure_profile) > 0:
                ymax_atm = np.max([ymax_atm, np.max(pressure_profile)])
                ymin_atm = np.min([ymin_atm, np.min(pressure_profile)])



    xticks = [100, 1000, 2000, 3000, 4000, 5000]
    xmin = 100
    xmax = 5000

    units = myjson_o.get_dict_units(['data','temp_b'])
    # title = 'Atmosphere + mantle temperature profile' #'(a) Temperature, {}'.format(units)

    # Atmosphere part
    fig_o.set_myaxes( ax0, ylabel='$P_\mathrm{atm}$\n(bar)', xmin=xmin, xmax=xmax, xticks=xticks, ymin=0, ymax=ymax_atm ) # , xmin=300, xmax=4200, , yticks=yticks, , title=title
    # ax0.set_xlim( 100, 5000 )

    # Ensure sensible axes limits
    ymax_atm = np.min([ymax_atm, 1e+30])
    ymin_atm = np.max([ymin_atm, 1e-30])
    print(ymin_atm, ymax_atm)

    # ax0.set_ylim( top=ymax_atm, bottom=ymin_atm )
    # ax0.set_yticks([ymin_atm, 1e0, 1e1, 1e2, ymax_atm])
    # ax0.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax0.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax0.yaxis.set_label_coords(-0.13,0.5)
    ax0.invert_yaxis()
    ax0.set_xticklabels([])

    # Interior part
    yticks = [0, 20,40,60,80,100,120,int(xx_pres[-1])]
    ymax = int(xx_pres[-1])
    fig_o.set_myaxes( ax1, xlabel='$T$, '+units, ylabel='$P_\mathrm{mantle}$\n(GPa)', xmin=xmin, xmax=xmax, xticks=xticks, ymin=0, ymax=ymax, yticks=yticks )
    ax1.set_yticklabels(["", "20","40","60","80","100","120",str(int(xx_pres[-1]))])
    ax1.yaxis.set_label_coords(-0.13,0.5)
    ax1.invert_yaxis()
    ax1.legend( fontsize=8, fancybox=True, framealpha=0.5, loc=3 )

    # Pressure-depth conversion for y-axis
    ax1b = ax1.twinx()
    yy = myjson_o.get_dict_values(['data','temp_b'])
    ax1b.plot( yy, xx_depth, alpha=0.0)
    ax1b.set_xlim( right=xmax, left=xmin )
    ax1b.set_ylim(top=xx_depth[-1], bottom=xx_depth[0])
    ax1b.set_yticks([0, 500, 1000, 1500, 2000, 2500, int(xx_depth[-1])])
    ax1b.set_ylabel( '$d$\n(km)', rotation=0 )
    ax1b.yaxis.set_label_coords(1.15,0.55)
    ax1b.invert_yaxis()

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
