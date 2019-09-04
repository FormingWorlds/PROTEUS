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
def plot_atmosphere( times ):

    # article class text width is 4.7747 inches
    # http://tex.stackexchange.com/questions/39383/determine-text-width

    logger.info( 'building stacked interior atmosphere' )

    width = 12.00 #* 3.0/2.0
    height = 6.0
    fig_o = su.FigureData( 2, 2, width, height, 'output/'+'plot_atmosphere', units='kyr' ) #, times
    fig_o.fig.subplots_adjust(wspace=0.07,hspace=0.25)
    fig_o.time = times

    ax0 = fig_o.ax[0][0]
    ax1 = fig_o.ax[1][0]
    ax2 = fig_o.ax[0][1]
    ax3 = fig_o.ax[1][1]

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

    fig_o.set_colors(cmap="viridis_r") # "magma_r"

    ymax_atm_pressure = 0
    ymin_atm_pressure = 1000
    ymax_atm_z = 0
    ymin_atm_z = 0

    # Find planet mass
    core_mass   = myjson_o.get_dict_values(['atmosphere','mass_core'])
    mantle_mass = myjson_o.get_dict_values(['atmosphere','mass_mantle'])
    planet_mass = core_mass + mantle_mass

    for nn, time in enumerate( fig_o.time ):

        if os.path.exists('output/'+str(int(time))+"_atm_TP_profile.dat"):

            # Read atmosphere properties
            atm_TP_profile      = np.loadtxt('output/'+str(int(time))+"_atm_TP_profile.dat")
            atm_spectral_flux   = np.loadtxt('output/'+str(int(time))+"_atm_spectral_flux.dat")

            temperature_atmosphere  = []
            pressure_atmosphere     = []
            band_centres            = []
            spectral_flux           = []
            for i in range(0, len(atm_TP_profile)):
                temperature_atmosphere.append(atm_TP_profile[i][0])
                pressure_atmosphere.append(atm_TP_profile[i][1])
            for i in range(0, len(atm_spectral_flux)):
                band_centres.append(atm_spectral_flux[i][0])
                spectral_flux.append(atm_spectral_flux[i][1])

                # print(time, atm_spectral_flux[i])

            # read json
            myjson_o = su.MyJSON( 'output/{}.json'.format(time) )

            color = fig_o.get_color( nn )
            # use melt fraction to determine mixed region
            MIX = myjson_o.get_mixed_phase_boolean_array( 'basic' )

            label = coupler_utils.latex_float(time)+" yr"
            
            # Atmosphere T-Z
            pressure_atmosphere_Pa = [ n*100. for n in pressure_atmosphere]     # Pa
            z_profile = coupler_utils.AtmosphericHeight(temperature_atmosphere, pressure_atmosphere_Pa, planet_mass, r_planet) # m
            z_profile = z_profile*1e-3 # km
            ax0.plot( temperature_atmosphere, z_profile, '-', color=color, label=label, lw=1.5)

            # Atmosphere T-P
            pressure_atmosphere_bar = [ n/1000. for n in pressure_atmosphere]    # bar
            ax1.semilogy( temperature_atmosphere, pressure_atmosphere_bar, '-', color=color, label=label, lw=1.5)

            # Atmospheric mixing ratios
            h2o_kg = myjson_o.get_dict_values( ['atmosphere','H2O','atmosphere_kg'] )
            co2_kg = myjson_o.get_dict_values( ['atmosphere','CO2','atmosphere_kg'] )
            h2_kg  = 0  # kg
            ch4_kg = 0  # kg
            co_kg  = 0  # kg
            n2_kg  = 0  # kg
            o2_kg  = 0  # kg
            he_kg  = 0  # kg
            h2o_ratio, co2_ratio, h2_ratio, ch4_ratio, co_ratio, n2_ratio, o2_ratio, he_ratio = coupler_utils.CalcMolRatios(h2o_kg, co2_kg, h2_kg, ch4_kg, co_kg, n2_kg, o2_kg, he_kg)
            h2o_ratio = h2o_ratio*np.ones(len(z_profile))
            co2_ratio = co2_ratio*np.ones(len(z_profile))
            ax2.plot( h2o_ratio, z_profile, '-', color=color, label=label, lw=1.5)

            # Spectral OLR
            ax3.plot( band_centres, spectral_flux, '-', color=color, label=label, lw=1.5)

            # Reset y-axis boundaries
            if np.min(pressure_atmosphere_bar) > 0:
                ymax_atm_pressure = np.max([ymax_atm_pressure, np.max(pressure_atmosphere_bar)])
                ymin_atm_pressure = np.min([ymin_atm_pressure, np.min(pressure_atmosphere_bar)])
                ymax_atm_z = np.max([ymax_atm_z, np.max(z_profile)])
                ymin_atm_z = np.min([ymin_atm_z, np.min(z_profile)])

    ### Figure settings
    xticks = [10, 500, 1000, 1500, 2000, 2500, 3000]
    xmin = 10
    xmax = 3000

    #####  T-Z
    fig_o.set_myaxes( ax0, xlabel='$T$ (K)', ylabel='$z_\mathrm{atm}$\n(km)', xmin=xmin, xmax=xmax, ymin=0, ymax=ymax_atm_z, xticks=xticks )
    ax0.set_ylim( top=ymax_atm_z, bottom=ymin_atm_z )
    # ax0.set_yticks([ymin_atm_pressure, 1e-2, 1e-1, 1e0, 1e1, ymax_atm_pressure])
    ax0.yaxis.set_label_coords(-0.13,0.5)
    # ax0.tick_params(direction='in')
    # ax0.set_xticklabels([])
    # ax0.xaxis.tick_top()
    # ax0.tick_params(direction='in')

    #####  T-P
    fig_o.set_myaxes( ax1, xlabel='$T$ (K)', ylabel='$P_\mathrm{atm}$\n(bar)', xmin=xmin, xmax=xmax, xticks=xticks )
    ax1.set_ylim( top=ymax_atm_pressure, bottom=ymin_atm_pressure )
    ax1.set_yticks([ymin_atm_pressure, 1e-2, 1e-1, 1e0, 1e1, ymax_atm_pressure])
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax1.yaxis.set_label_coords(-0.13,0.5)
    ax1.invert_yaxis()

    #####  Volatile mixing ratios
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_xlim( left=0, right=1 )
    ax2.set_ylabel( "$z_\mathrm{atm}$\n(km)", rotation=0)
    ax2.yaxis.set_label_coords(1.10,0.5)
    ax2.set_xlabel( '$X_{H_2O}$ (mol$_i$/mol)')

    #####  Spectral OLR
    fig_o.set_myaxes( ax3, xlabel='Wavenumber (cm$^{-1}$)', ylabel='Spectral flux' )
    ax3.set_xlim( left=0, right=10000 )
    ax3.set_ylabel( 'Spectral flux', rotation=90)
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    # ax3.set_ylim( bottom=0 )
    # ax0.yaxis.set_label_coords(-0.13,0.5)

    # Legend
    ax0.legend( fontsize=8, fancybox=True, framealpha=0.5 )

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

    plot_atmosphere( times=plot_list )

#====================================================================

if __name__ == "__main__":

    main()
