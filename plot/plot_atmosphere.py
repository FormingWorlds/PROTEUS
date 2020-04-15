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
import pandas as pd

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
    file = os.path.join("plot/ScientificColourMaps5/", name + '.txt')
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[name+"_r"] = LinearSegmentedColormap.from_list(name, cm_data[::-1])

logger = su.get_my_logger(__name__)

vol_colors = {
    "black_1" : "#000000",
    "black_2" : "#323232",
    "black_3" : "#7f7f7f",
    "H2O_1"   : "#8db4cb",
    "H2O_2"   : "#4283A9",
    "H2O_3"   : "#274e65",
    "CO2_1"   : "#811111",
    "CO2_2"   : "#B91919",
    "CO2_3"   : "#ce5e5e",
    "H2_1"    : "#a0d2cb",
    "H2_2"    : "#62B4A9",
    "H2_3"    : "#3a6c65",
    "CH4_1"   : "#eb9194",
    "CH4_2"   : "#E6767A",
    "CH4_3"   : "#b85e61",
    "CO_1"    : "#eab597",
    "CO_2"    : "#DD8452",
    "CO_3"    : "#844f31",
    "N2_1"    : "#c29fb2",
    "N2_2"    : "#9A607F",
    "N2_3"    : "#4d303f",  
    "S_1"     : "#f1ca70",
    "S_2"     : "#EBB434",
    "S_3"     : "#a47d24",    
    "O2_1"    : "#57ccda",
    "O2_2"    : "#2EC0D1",
    "O2_3"    : "#2499a7",
    "He_1"    : "#acbbbf",
    "He_2"    : "#768E95",
    "He_3"    : "#465559"
}

#====================================================================
def plot_atmosphere( output_dir, times ):

    # article class text width is 4.7747 inches
    # http://tex.stackexchange.com/questions/39383/determine-text-width

    # logger.info( 'building stacked interior atmosphere' )

    width = 6.00 #* 3.0/2.0
    height = 12.0
    fig_o = su.FigureData( 3, 1, width, height, output_dir+'plot_atmosphere', units='kyr' ) #, times
    # fig1 = plt.figure()
    fig_o.fig.subplots_adjust(wspace=0.07,hspace=0.2)
    fig_o.time = times

    ax0 = fig_o.ax[0]#[0]
    ax1 = fig_o.ax[1]#[0]
    ax2 = fig_o.ax[2]#[0]
    # ax3 = fig_o.ax[1][1]

    time = fig_o.time[0] # first timestep since liquidus and solidus
                         # are time-independent

    myjson_o = su.MyJSON( output_dir+'{}.json'.format(time) )

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

    # # Load runtime helpfile
    # runtime_helpfile = pd.read_csv(output_dir+"runtime_helpfile.csv")

    for nn, time in enumerate( fig_o.time ):

        if os.path.exists(output_dir+str(int(time))+"_atm_TP_profile.dat"):

            # Read atmosphere properties
            atm_TP_profile      = np.loadtxt(output_dir+str(int(time))+"_atm_TP_profile.dat")
            atm_spectral_flux   = np.loadtxt(output_dir+str(int(time))+"_atm_spectral_flux.dat")

            # # Read atmospheric chemistry
            # atm_chemistry       = pd.read_csv(output_dir+"runtime_helpfile.csv")

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
            myjson_o = su.MyJSON( output_dir+'{}.json'.format(time) )

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
            # pressure_atmosphere_bar = [ n/1000. for n in pressure_atmosphere]    # bar
            ax1.semilogy( temperature_atmosphere, pressure_atmosphere, '-', color=color, label=label, lw=1.5)

            # # Atmospheric mixing ratios
            # h2o_kg = myjson_o.get_dict_values( ['atmosphere','H2O','atmosphere_kg'] )
            # co2_kg = myjson_o.get_dict_values( ['atmosphere','CO2','atmosphere_kg'] )
            # h2_kg  = 0  # kg
            # ch4_kg = 0  # kg
            # co_kg  = 0  # kg
            # n2_kg  = 0  # kg
            # o2_kg  = 0  # kg
            # he_kg  = 0  # kg
            # h2o_ratio, co2_ratio, h2_ratio, ch4_ratio, co_ratio, n2_ratio, o2_ratio, he_ratio = coupler_utils.CalcMolRatios(h2o_kg, co2_kg, h2_kg, ch4_kg, co_kg, n2_kg, o2_kg, he_kg)
            # h2o_ratio = h2o_ratio*np.ones(len(z_profile))
            # co2_ratio = co2_ratio*np.ones(len(z_profile))
            # ax2.plot( h2o_ratio, z_profile, '-', color=color, label=label, lw=1.5)

            # Spectral OLR
            ax2.plot( band_centres, spectral_flux, '-', color=color, label=label, lw=1.5)

            # Reset y-axis boundaries
            if np.min(pressure_atmosphere) > 0:
                ymax_atm_pressure = np.max([ymax_atm_pressure, np.max(pressure_atmosphere)])
                ymin_atm_pressure = np.min([ymin_atm_pressure, np.min(pressure_atmosphere)])
                ymax_atm_z = np.max([ymax_atm_z, np.max(z_profile)])
                ymin_atm_z = np.min([ymin_atm_z, np.min(z_profile)])

    ### Figure settings
    xticks = [10, 500, 1000, 1500, 2000, 2500, 3000]
    xmin = 10
    xmax = 3000
    title_xcoord = -0.09
    title_ycoord = 0.5

    #####  T-Z
    # fig_o.set_myaxes( ax0, xlabel='$T$ (K)', ylabel='$z_\mathrm{atm}$\n(km)', xmin=xmin, xmax=xmax, ymin=0, ymax=ymax_atm_z, xticks=xticks )
    ax0.set_xlabel("Temperature, $T$ (K)")
    ax0.set_ylabel("Height, $z_\mathrm{atm}$ (km)")
    ax0.set_ylim( top=ymax_atm_z, bottom=ymin_atm_z )
    # ax0.set_yticks([ymin_atm_pressure, 1e-2, 1e-1, 1e0, 1e1, ymax_atm_pressure])
    ax0.yaxis.set_label_coords(title_xcoord,title_ycoord)
    # ax0.tick_params(direction='in')
    # ax0.set_xticklabels([])
    # ax0.xaxis.tick_top()
    # ax0.tick_params(direction='in')

    #####  T-P
    # fig_o.set_myaxes( ax1, xlabel='$T$ (K)', ylabel='$P_\mathrm{atm}$\n(bar)', xmin=xmin, xmax=xmax, xticks=xticks )
    ax1.set_xlabel("Temperature, $T$ (K)")
    ax1.set_ylabel("Pressure, $P_\mathrm{tot}$ (bar)")
    ax1.set_ylim( top=ymax_atm_pressure, bottom=ymin_atm_pressure )
    ax1.set_ylim( top=ymax_atm_pressure, bottom=ymin_atm_pressure )
    # ax1.set_yticks([ymin_atm_pressure, 1e-2, 1e-1, 1e0, 1e1, ymax_atm_pressure])
    # ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax1.yaxis.set_label_coords(title_xcoord,title_ycoord)
    ax1.invert_yaxis()

    # #####  Volatile mixing ratios
    # ax2.yaxis.set_label_position("right")
    # ax2.yaxis.tick_right()
    # ax2.set_xlim( left=0, right=1 )
    # ax2.set_ylabel( "$z_\mathrm{atm}$\n(km)", rotation=0)
    # ax2.yaxis.set_label_coords(1.10,0.5)
    # ax2.set_xlabel( '$X_{H_2O}$ (mol$_i$/mol)')

    #####  Spectral OLR
    fig_o.set_myaxes( ax2, xlabel='Wavenumber (cm$^{-1}$)', ylabel='Spectral flux' )
    
    ax2.set_ylabel( 'Spectral flux density (Jy)', rotation=90)
    # ax2.yaxis.set_label_position("right")
    # ax2.yaxis.tick_right()
    ax2.set_xlim( left=0, right=20000 )
    ax2.set_ylim( bottom=0 ) #, top=10
    ax2.yaxis.set_label_coords(title_xcoord,title_ycoord)

    # Legend
    ax1.legend( fontsize=8, fancybox=True, framealpha=0.5 )

    fig_o.savefig(1)
    plt.close()

# Plot settings
lw=2
title_fs   = 10
title_xy   = (0.02, 0.01)
title_x    = title_xy[0]
title_y    = title_xy[1]
title_xycoords = 'axes fraction'
title_ha   = "left"
title_va   = "bottom"
title_font = 'Arial'

volatile_species = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2", "S", "He" ]

def plot_current_mixing_ratio( output_dir, times, vulcan_setting ):

    fig, ax1 = plt.subplots()

    time = str(times)

    # Read atmospheric chemistry
    atm_chemistry       = pd.read_csv(output_dir+time+"_atm_chemistry_volume.dat", skiprows=1, delim_whitespace=True) # skipfooter=1

    if vulcan_setting == 0 or vulcan_setting == 1:
        runtime_helpfile = pd.read_csv(output_dir+"runtime_helpfile.csv", delim_whitespace=True)

        for vol in volatile_species:
            if runtime_helpfile.iloc[-1][vol+"_mr"] > 0.:
                ax1.axvline(x=runtime_helpfile.iloc[-1][vol+"_mr"], linewidth=lw+0.5, color=vol_colors[vol+"_2"], ls=":")

    ax1.plot(atm_chemistry["H2"], atm_chemistry["Pressure"], color=vol_colors["H2_2"], ls="-", lw=lw, label=r"H$_2$")
    ax1.plot(atm_chemistry["H2O"], atm_chemistry["Pressure"], color=vol_colors["H2O_2"], ls="-", lw=lw, label=r"H$_2$O")

    ax1.plot(atm_chemistry["CO2"], atm_chemistry["Pressure"], color=vol_colors["CO2_2"], ls="-", lw=lw, label=r"CO$_2$")
    ax1.plot(atm_chemistry["CO"], atm_chemistry["Pressure"], color=vol_colors["CO_2"], ls="-", lw=lw, label=r"CO")
    ax1.plot(atm_chemistry["CH4"], atm_chemistry["Pressure"], color=vol_colors["CH4_2"], ls="--", lw=lw, label=r"CH$_4$")
    ax1.plot(atm_chemistry["N2"], atm_chemistry["Pressure"], color=vol_colors["N2_2"], ls="-", lw=lw, label=r"N$_2$")
    ax1.plot(atm_chemistry["O2"], atm_chemistry["Pressure"], color=vol_colors["O2_2"], ls="--", lw=lw, label=r"O$_2$")
    ax1.plot(atm_chemistry["S"], atm_chemistry["Pressure"], color=vol_colors["S_2"], ls="--", lw=lw, label=r"S")
    ax1.plot(atm_chemistry["He"], atm_chemistry["Pressure"], color=vol_colors["He_2"], ls="--", lw=lw, label=r"He")
    
    ax1.set_xlabel("Volume mixing ratio, log ($n^{\mathrm{i}}$/$n_{\mathrm{tot}}$)")
    ax1.set_ylabel("Pressure, $P_{\mathrm{tot}}$ (bar)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(left=1e-16, right=2e+0)
    ax1.set_ylim(np.max(atm_chemistry["Pressure"]), np.min(atm_chemistry["Pressure"]))

    # Temperature y-axis on the right
    ax1b = ax1.twinx()
    ax1b.plot( atm_chemistry["H2O"], atm_chemistry["Temp"], alpha=0.0)
    ax1b.set_ylim(np.max(atm_chemistry["Temp"]), np.min(atm_chemistry["Temp"]))
    if not np.min(atm_chemistry["Temp"]) == np.max(atm_chemistry["Temp"]):
        ax1b.set_yticks(
         [ 
            int(np.max(atm_chemistry["Temp"])), 
            int(np.min(atm_chemistry["Temp"])+0.75*(np.max(atm_chemistry["Temp"])-np.min(atm_chemistry["Temp"]))), 
            int(np.min(atm_chemistry["Temp"])+0.5*(np.max(atm_chemistry["Temp"])-np.min(atm_chemistry["Temp"]))), 
            int(np.min(atm_chemistry["Temp"])+0.25*(np.max(atm_chemistry["Temp"])-np.min(atm_chemistry["Temp"]))), 
            int(np.min(atm_chemistry["Temp"]))
         ])
    ax1b.yaxis.tick_right()
    ax1b.set_ylabel( 'Temperature, $T$ (K)' )

    ax1.set_title("Time = "+coupler_utils.latex_float(float(time))+" yr", fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va)


    ax1.legend(loc=2, fancybox=True, framealpha=0.9)
    plt.xticks([1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0], ["-16", "-14", "-12", "-10", "-8", "-6", "-4", "-2", "0"])

    plt.savefig(output_dir+"plot_atm_chemistry_"+str(time)+".pdf", bbox_inches = 'tight',
    pad_inches = 0.1)
    plt.close()


#====================================================================
def main():

    # Optional command line arguments for running from the terminal
    # Usage: $ python plot_atmosphere.py -t 0,718259
    parser = argparse.ArgumentParser(description='COUPLER plotting script')
    parser.add_argument('-t', '--times', type=str, help='Comma-separated (no spaces) list of times');
    args = parser.parse_args()

    if args.times:
        print("Snapshots:", args.times)
        plot_list = [ int(time) for time in args.times.split(',') ]
    else:
        output_list = su.get_all_output_times()

        if len(output_list) <= 8:
            plot_list = output_list
        else:
            plot_list = [ output_list[0], output_list[int(round(len(output_list)*(2./100.)))], output_list[int(round(len(output_list)*(15./100.)))], output_list[int(round(len(output_list)*(22./100.)))], output_list[int(round(len(output_list)*(33./100.)))], output_list[int(round(len(output_list)*(50./100.)))], output_list[int(round(len(output_list)*(66./100.)))], output_list[-1] ]
        print("Snapshots:", plot_list)

    # Plot only last snapshot
    plot_current_mixing_ratio( output_dir="output/", times=plot_list[-1], vulcan_setting=0 )

    # Plot fixed set from above
    plot_atmosphere( output_dir="output/", times=plot_list )

#====================================================================

if __name__ == "__main__":

    main()
