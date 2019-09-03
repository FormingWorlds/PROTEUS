#!/usr/bin/env python

import spider_coupler_utils as su
import coupler_utils
import argparse
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import sys
import glob
# from natsort import natsorted #https://pypi.python.org/pypi/natsort
from decimal import Decimal
import matplotlib.ticker as ticker

# Font settings
# https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
# https://olgabotvinnik.com/blog/how-to-set-helvetica-as-the-default-sans-serif-font-in/
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['font.sans-serif'] = "Helvetica"
# matplotlib.rcParams['font.family'] = "sans-serif"
# params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
# matplotlib.rcParams.update(params)
# matplotlib.rcParams['ps.useafm'] = True
# matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True

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
def plot_stacked( times ):

    # article class text width is 4.7747 inches
    # http://tex.stackexchange.com/questions/39383/determine-text-width

    logger.info( 'building stacked interior atmosphere' )

    width = 5.00 #* 3.0/2.0
    height = 10.0
    fig_o = su.FigureData( 2, 1, width, height, 'output/'+'plot_stacked', units='kyr' ) #, times
    fig_o.fig.subplots_adjust(wspace=0.0,hspace=0.04)
    fig_o.time = times

    ax0 = fig_o.ax[0]
    ax1 = fig_o.ax[1]
    # ax2 = fig_o.ax[2]
    # ax3 = fig_o.ax[3]

    time = fig_o.time[0] # first timestep since liquidus and solidus
                         # are time-independent

    myjson_o = su.MyJSON( 'output/{}.json'.format(time) )

    pressure_interior = myjson_o.get_dict_values(['data','pressure_b'])
    # pressure_interior = myjson_o.get_dict_values_internal(['data','pressure_b'])
    pressure_interior *= 1.0E-9
    pressure_interior_s = myjson_o.get_dict_values(['data','pressure_s'])
    pressure_interior_s *= 1.0E-9

    xx_radius = myjson_o.get_dict_values(['data','radius_b'])
    # xx_radius = myjson_o.get_dict_values_internal(['data','radius_b'])
    xx_radius *= 1.0E-3
    xx_depth = xx_radius[0] - xx_radius
    xx_radius_s = myjson_o.get_dict_values(['data','radius_s'])
    xx_radius_s *= 1.0E-3
    xx_depth_s = xx_radius_s[0] - xx_radius_s
    r_planet = np.max(xx_radius*1e3) # m


    handle_l = [] # handles for legend


    fig_o.set_colors(cmap=vik_r) # "magma_r"

    ymax_atm_pressure = 0
    ymin_atm_pressure = 1000
    ymax_atm_z = 0
    ymin_atm_z = 0

    for nn, time in enumerate( fig_o.time ):

        if os.path.exists('output/'+str(int(time))+"_atm_TP_profile.dat"):

            # Read atmosphere properties
            atm_TP_profile  = np.loadtxt('output/'+str(int(time))+"_atm_TP_profile.dat")

            temperature_atmosphere  = []
            pressure_atmosphere     = []
            for i in range(0, len(atm_TP_profile)):
                temperature_atmosphere.append(atm_TP_profile[i][0])
                pressure_atmosphere.append(atm_TP_profile[i][1])

            # read json
            myjson_o = su.MyJSON( 'output/{}.json'.format(time) )

            color = fig_o.get_color( nn )
            # use melt fraction to determine mixed region
            MIX = myjson_o.get_mixed_phase_boolean_array( 'basic' )
            # MIX = myjson_o.get_mixed_phase_boolean_array( 'basic_internal' )
            # MIX_s = myjson_o.get_mixed_phase_boolean_array( 'staggered' )

            label = coupler_utils.latex_float(time)+" yr"

            # Pressure-height conversion for y-axis
            
            # Plot height instead of pressure as y-axis
            ### CAREFUL: HARDCODED TO M_EARTH !!!
            z_profile = coupler_utils.AtmosphericHeight(temperature_atmosphere, pressure_atmosphere, 5.972E24, r_planet) ## WRONG UNITS!
            z_profile = z_profile*1e-3 # km
            # ax0.plot( temperature_atmosphere, z_profile, '-', color=color, label=label, lw=1.5)

            # Atmospphere T-P
            ax0.semilogy( temperature_atmosphere, pressure_atmosphere, '-', color=color, label=label, lw=1.5)
            # ax0.plot( temperature_atmosphere, pressure_atmosphere, '-', color=color, label=label, lw=1.5)

            # interior
            temperature_interior = myjson_o.get_dict_values(['data','temp_b'])
            # yy_s = myjson_o.get_dict_values_internal(['data','temp_s'])
            ax1.plot( temperature_interior, pressure_interior, '--', color=color, lw=1.5 )
            ax1.plot( temperature_interior*MIX, pressure_interior*MIX, '-', color=color, label=label, lw=1.5 )

            # # connect atmosphere and interior lines at interface
            # connecting_line_T = [ temperature_atmosphere[-1], temperature_interior[0] ]
            # connecting_line_P = [ pressure_atmosphere[-1], pressure_atmosphere[-1]*1.1 ]
            # ax0.semilogy( connecting_line_T, connecting_line_P, ':', color=color, lw=1.5)

            if np.min(pressure_atmosphere) > 0:
                ymax_atm_pressure = np.max([ymax_atm_pressure, np.max(pressure_atmosphere)])
                ymin_atm_pressure = np.min([ymin_atm_pressure, np.min(pressure_atmosphere)])
                ymax_atm_z = np.max([ymax_atm_z, np.max(z_profile)])
                ymin_atm_z = np.min([ymin_atm_z, np.min(z_profile)])

            # ymax_atm = pressure_atmosphere[-1]*1.1

            print(time)

    xticks = [50, 1000, 2000, 3000, 4000, 5000]
    xmin = 50
    xmax = 5000

    units = myjson_o.get_dict_units(['data','temp_b'])
    # title = 'Atmosphere + mantle temperature profile' #'(a) Temperature, {}'.format(units)

    ##### Atmosphere part

    # Y-axis settings
    fig_o.set_myaxes( ax0, ylabel='$P_\mathrm{atm}$\n(mbar)', xmin=xmin, xmax=xmax, xticks=xticks )
    ax0.set_ylim( top=ymax_atm_pressure, bottom=ymin_atm_pressure )
    ax0.set_yticks([ymin_atm_pressure, 1e1, 1e2, 1e3, 1e4, ymax_atm_pressure])
    ax0.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax0.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax0.xaxis.tick_top()
    ax0.tick_params(direction='in')
    ax0.yaxis.set_label_coords(-0.13,0.5)
    ax0.invert_yaxis()
    ax0.set_xticklabels([])

    # # Pressure on Y-axis from min/max values
    # ax0b = ax0.twinx()
    # ax0b.plot( temperature_atmosphere, pressure_atmosphere, alpha=0.0)
    # ax0b.set_xlim( right=xmax, left=xmin )
    # ax0b.set_ylim( top=ymin_atm_pressure, bottom=ymax_atm_pressure )
    # yticks_pressure = [ymin_atm_pressure, ymax_atm_pressure*(1./6.), ymax_atm_pressure*(2./6.), ymax_atm_pressure*(1./2.), ymax_atm_pressure*(4./6.), ymax_atm_pressure*(5./6.), ymax_atm_pressure]
    # ax0b.set_yticks( yticks_pressure )
    # ax0b.set_ylabel( '$P_\mathrm{atm}$\n(bar)', rotation=0 )
    # ax0b.yaxis.set_label_coords(1.15,0.55)

    # Legend
    ax0.legend( fontsize=8, fancybox=True, framealpha=0.5 )

    #####  Interior part
    yticks = [0, 20,40,60,80,100,120,int(pressure_interior[-1])]
    ymax = int(pressure_interior[-1])
    fig_o.set_myaxes( ax1, xlabel='$T$, '+units, ylabel='$d_\mathrm{mantle}$\n(km)', xmin=xmin, xmax=xmax, xticks=xticks, ymin=0, ymax=ymax, yticks=yticks )
    ax1.set_yticklabels(["0", "20","40","60","80","100","120",str(int(pressure_interior[-1]))])
    ax1.invert_yaxis()
    ax1.yaxis.set_label_coords(1.15,0.5)

    # Pressure-depth conversion for interior y-axis
    ax1b = ax1.twinx()
    yy = myjson_o.get_dict_values(['data','temp_b'])
    ax1b.plot( yy, xx_depth, alpha=0.0)
    ax1b.set_xlim( right=xmax, left=xmin )
    ax1b.set_ylim(top=xx_depth[-1], bottom=xx_depth[0])
    ax1b.set_yticks([100, 500, 1000, 1500, 2000, 2500, int(xx_depth[-1])])
    ax1b.set_ylabel( '$P_\mathrm{mantle}$\n(GPa)', rotation=0 )
    ax1b.yaxis.set_label_coords(-0.13,0.55)
    ax1b.invert_yaxis()

    # # Pressure-height conversion for y-axis
    # ax0b = ax0.twinx()
    # r_planet = np.max(xx_radius*1e3) # m
    # ### CAREFUL: HARDCODED TO M_EARTH !!!
    # z_profile = coupler_utils.AtmosphericHeight(temperature_atmosphere, pressure_atmosphere, 5.972E24, r_planet)
    # z_profile = z_profile*1e-3 # km
    # ax0b.plot( temperature_atmosphere, z_profile, alpha=0.0)
    # ax0b.set_xlim( right=xmax, left=xmin )
    # ax0b.set_ylim(top=np.max(z_profile), bottom=np.min(z_profile))
    # ax0b.set_yticks([0, np.max(z_profile)/2., np.max(z_profile)])
    # ax0b.set_ylabel( '$z$\n(km)', rotation=0 )
    # ax0b.yaxis.set_label_coords(1.15,0.55)
    # # ax0b.invert_yaxis()

    fig_o.savefig(1)
    plt.close()

#====================================================================
def main():

    output_list = su.get_all_output_times()

    if len(output_list) <= 8:
        plot_list = output_list
    else:
        plot_list = [ output_list[0], output_list[int(round(len(output_list)*(2./100.)))], output_list[int(round(len(output_list)*(15./100.)))], output_list[int(round(len(output_list)*(22./100.)))], output_list[int(round(len(output_list)*(33./100.)))], output_list[int(round(len(output_list)*(50./100.)))], output_list[int(round(len(output_list)*(66./100.)))], output_list[-1] ]
    print("snapshots:", plot_list)

    plot_stacked( times=plot_list )

#====================================================================

if __name__ == "__main__":

    main()
