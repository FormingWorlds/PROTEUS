#!/usr/bin/env python

import logging
import spider_coupler_utils as su
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker
import argparse
import matplotlib
import pandas as pd
# import seaborn as sns
# # https://seaborn.pydata.org/tutorial/aesthetics.html
# sns.set_style("white")

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

# Color definitions, https://chrisalbon.com/python/seaborn_color_palettes.html
qgray       = "#768E95"
qblue       = "#4283A9" # http://www.color-hex.com/color/4283a9
qgreen      = "#62B4A9" # http://www.color-hex.com/color/62b4a9
qred        = "#E6767A"
qturq       = "#2EC0D1"
qmagenta    = "#9A607F"
qyellow     = "#EBB434"
qgray_dark  = "#465559"
qblue_dark  = "#274e65"
qgreen_dark = "#3a6c65"
qred_dark   = "#b85e61"
qturq_dark  = "#2499a7"
qmagenta_dark = "#4d303f"
qyellow_dark  = "#a47d24"
qgray_light  = "#acbbbf"
qblue_light  = "#8db4cb"
qgreen_light = "#a0d2cb"
qred_light   = "#eb9194"
qturq_light  = "#57ccda"
qmagenta_light = "#c29fb2"
qyellow_light = "#f1ca70"

# Plot settings
lw      = 2.5
fscale  = 1.1
fsize   = 18
fs_title=18

# Output dir, optional argument: https://towardsdatascience.com/learn-enough-python-to-be-useful-argparse-e482e1764e05
parser = argparse.ArgumentParser(description='Define file output directory.')
parser.add_argument('--dir', default="output", help='Provide path to output directory.' )

#====================================================================
def plot_global( output_dir ):

    logger.info( 'building atmosphere' )

    width = 12.00 #* 3.0/2.0
    height = 8.0 #/ 2.0
    fig_o = su.FigureData( 3, 2, width, height, output_dir+'plot_global', units='yr' )
    fig_o.fig.subplots_adjust(wspace=0.05,hspace=0.3)

    # Subplot titles
    title_fs = 8
    title_xy=(0.98, 0.5)
    title_xycoords='axes fraction'
    title_ha="right"
    title_va="center"

    fs_legend = 9

    lw = 1.5

    ax0 = fig_o.ax[0][0]
    ax1 = fig_o.ax[1][0]
    ax2 = fig_o.ax[2][0]
    ax3 = fig_o.ax[0][1]
    ax4 = fig_o.ax[1][1]
    ax5 = fig_o.ax[2][1]

    # runtime_helpfile = pd.read_csv(output_dir+"runtime_helpfile.csv", sep=" ")
    # print(runtime_helpfile)

    fig_o.time = su.get_all_output_times()
    print("Times", fig_o.time)

    keys_t = ( ('atmosphere','mass_liquid'),
               ('atmosphere','mass_solid'),
               ('atmosphere','mass_mantle'),
               ('atmosphere','temperature_surface'),
               ('atmosphere','emissivity'),
               ('rheological_front_phi','phi_global'),
               ('atmosphere','Fatm'),
               ('atmosphere','mass_core'),
               ('atmosphere','pressure_surface'),
               ('atmosphere','H2O','liquid_kg'),
               ('atmosphere','H2O','solid_kg'),
               ('atmosphere','H2O','initial_kg'),
               ('atmosphere','H2O','atmosphere_kg'),
               ('atmosphere','H2O','atmosphere_bar'),
               ('atmosphere','CO2','liquid_kg'),
               ('atmosphere','CO2','solid_kg'),
               ('atmosphere','CO2','initial_kg'),
               ('atmosphere','CO2','atmosphere_kg'),
               ('atmosphere','CO2','atmosphere_bar')
               )

    data_a = su.get_dict_surface_values_for_times( keys_t, fig_o.time )

    # out_a = np.column_stack( (timeMyr_a,data_a[15,:],data_a[12,:]) )
    # np.savetxt( 'out.dat', out_a )
    mass_liquid         = data_a[0,:]
    mass_solid          = data_a[1,:]
    mass_mantle         = data_a[2,:]
    T_surf              = data_a[3,:]
    emissivity          = data_a[4,:]
    phi_global          = data_a[5,:]
    Fatm                = data_a[6,:]
    mass_core           = data_a[7,:]
    P_surf              = data_a[8,:]

    H2O_liquid_kg       = data_a[9,:]
    H2O_solid_kg        = data_a[10,:]
    H2O_initial_kg      = data_a[11,:]
    H2O_atm_kg          = data_a[12,:]
    H2O_atm_pressure    = data_a[13,:]

    CO2_liquid_kg       = data_a[14,:]
    CO2_solid_kg        = data_a[15,:]
    CO2_initial_kg      = data_a[16,:]
    CO2_atm_kg          = data_a[17,:]
    CO2_atm_pressure    = data_a[18,:]

    H2O_interior_kg     = H2O_liquid_kg   + H2O_solid_kg
    CO2_interior_kg     = CO2_liquid_kg   + CO2_solid_kg

    H2O_total_kg        = H2O_interior_kg + H2O_atm_kg
    CO2_total_kg        = CO2_interior_kg + CO2_atm_kg

    vol_mass_atm        = H2O_atm_kg      + CO2_atm_kg
    vol_mass_interior   = H2O_interior_kg + CO2_interior_kg
    vol_mass_total      = H2O_total_kg    + CO2_total_kg
    
    planet_mass_total   = vol_mass_total  + mass_mantle     + mass_core

    #xticks = [1E-5,1E-4,1E-3,1E-2,1E-1]#,1]
    #xticks = [1.0E-2, 1.0E-1, 1.0E0, 1.0E1, 1.0E2,1.0E3] #[1E-6,1E-4,1E-2,1E0,1E2,1E4,1E6]#,1]
    #xticks = (1E-6,1E-5,1E-4,1E-3,1E-2,1E-1,1E0)
    xlabel = 'Time (yr)'
    #xlim = (1.0E-2, 4550)
    xlim = (1e1,1e7)

    red = (0.5,0.1,0.1)
    blue = (0.1,0.1,0.5)
    black = 'black'

    # fig_o.set_colors(cmap=vik_r)
    # volatiles = [ "H2O", "CO2", "H2", "CO", "N2", "O2", "CH4", "He", "S" ]
    # color_H2O = fig_o.get_color( 0 )
    # color_CO2 = fig_o.get_color( 1 )
    # color_H2  = fig_o.get_color( 2 )
    # color_CO  = fig_o.get_color( 3 )
    # color_N2  = fig_o.get_color( 4 )
    # color_O2  = fig_o.get_color( 5 )
    # color_CH4 = fig_o.get_color( 6 )
    # color_He  = fig_o.get_color( 7 )
    # color_S   = fig_o.get_color( 8 )

    # to plot melt fraction contours on figure (a)
    # compute time at which desired melt fraction is reached
    #phi_a = mass_liquid_a / mass_mantle
    #phi_cont_l = [0.75, 0.5,0.25,0.1,0.01]
    #phi_time_l = [] # contains the times for each contour
    #for cont in phi_cont_l:
    #    time_temp_l = su.find_xx_for_yy( timeMyr_a, phi_global, cont )
    #    index = su.get_first_non_zero_index( time_temp_l )
    #    if index is None:
    #        out = 0.0
    #    else:
    #        out = timeMyr_a[index]
    #    phi_time_l.append( out )

    xcoord_l = -0.10
    ycoord_l = 0.5
    xcoord_r = 1.09
    ycoord_r = 0.5

    ##########
    # figure a
    ##########
    title = r'(a) Heat flux to space'
    ylabel = '$F_\mathrm{atm}$ (W/m$^2$)'
    # yticks = (1.0E2,1.0E3,1.0E4,1.0E5,1.0E6,1.0E7)
    h1, = ax0.loglog( fig_o.time, Fatm,'k', lw=lw )
    fig_o.set_myaxes( ax0, title=title)#, yticks=yticks, xlabel=xlabel, xticks=xticks )
    # ax0.yaxis.tick_right()
    # ax0.yaxis.set_label_position("right")
    ax0.set_ylabel(ylabel)
    ax0.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax0.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax0.set_xlim( *xlim )
    ax0.yaxis.set_label_coords(xcoord_l,ycoord_l)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, loc='upper right', ncol=1, frameon=0, fontsize=fs_legend)

    ##########
    # figure b
    ##########
    title = r'(b) Surface temperature'
    ylabel = '$T_\mathrm{surf}$ (K)'
    if np.max(fig_o.time) >= 1e3: 
        ymin = np.min(T_surf)
        ymax = np.max(T_surf)
    else: 
        ymin = 200
        ymax = 3000
    yticks = [ymin, ymin+0.2*(ymax-ymin), ymin+0.4*(ymax-ymin), ymin+0.6*(ymax-ymin), ymin+0.8*(ymax-ymin), ymax]
    h1, = ax1.semilogx( fig_o.time, T_surf, 'k-', lw=lw, label=r'Surface temp, $T_s$' )
    #ax2b = ax2.twinx()
    #h2, = ax2b.loglog( timeMyr_a, emissivity_a, 'k--', label=r'Emissivity, $\epsilon$' )
    fig_o.set_myaxes( ax1, title=title, yticks=yticks)#, xlabel=xlabel )
    ax1.set_ylabel(ylabel)
    ax1.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax1.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.set_xlim( *xlim )
    ax1.yaxis.set_label_coords(xcoord_l,ycoord_l)
    ax1.set_ylim(ymin,ymax)
    # ax1.set_title('(a) Surface temperature', fontname='Arial', fontsize=fs_title)

    ##########
    # figure c
    ##########
    title = r'(c) Global mantle melt fraction'
    ylabel = '$\phi_{\mathrm{mantle}}$'
    # trans = transforms.blended_transform_factory(
    #     ax2.transData, ax2.transAxes)
    h1, = ax2.semilogx( fig_o.time, phi_global, color=black, linestyle='-', lw=lw, label=r'Melt, $\phi_{\mathrm{mantle}}$')
    # h2, = ax2.semilogx( timeMyr_a, mass_liquid_a / mass_mantle, 'k--', label='melt' )
    fig_o.set_myaxes( ax2, title=title, xlabel=xlabel )#, xlabel=xlabel, xticks=xticks )
    ax2.set_ylabel(ylabel)
    ax2.set_xlim( *xlim )
    ax2.set_ylim( 0, 1 )
    ax2.set_ylabel(ylabel)
    ax2.yaxis.set_label_coords(xcoord_l,ycoord_l)
    # handles, labels = ax2.get_legend_handles_labels()
    # ax2.legend(handles, labels, loc='center left', ncol=1, frameon=0)

    # Set up species dependent plots

    ##########
    # figure d
    ##########
    # if 1:
    title = r'(d) Atmospheric volatile partial pressure'
    trans = transforms.blended_transform_factory(
        ax3.transData, ax3.transAxes)
    ax3.semilogx( fig_o.time, P_surf, color="gray", linestyle='-', lw=lw, label=r'Total')
    ax3.semilogx( fig_o.time, H2O_atm_pressure, color=blue, linestyle='-', lw=lw, label=r'H$_2$O')
    ax3.semilogx( fig_o.time, CO2_atm_pressure, color=red, linestyle='-', lw=lw, label=r'CO$_2$')
    
    fig_o.set_myaxes( ax3, title=title)#, xlabel=xlabel, xticks=xticks )
    # ax3.set_title(title, y=0.8)
    ax3.set_ylabel('$p^{\mathrm{i}}$ (bar)')
    ax3.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax3.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax3.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax3.set_xlim( *xlim )
    # ax3.set_ylim( 0, 350 )
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.set_label_coords(xcoord_r,ycoord_r)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels, loc='center left', ncol=1, frameon=1, fancybox=True, framealpha=0.9, fontsize=fs_legend)

    ##########
    # figure e
    ##########
    title = r'(e) Atmospheric volatile mass fraction'
    ax4.semilogx( fig_o.time, vol_mass_atm/vol_mass_total, lw=lw, color="gray", linestyle='-', label=r'Total')
    ax4.semilogx( fig_o.time, H2O_atm_kg/H2O_total_kg, lw=lw, color=blue, linestyle='-', label=r'H$_2$O')
    ax4.semilogx( fig_o.time, CO2_atm_kg/CO2_total_kg, lw=lw, color=red, linestyle='-', label=r'CO$_2$')
    fig_o.set_myaxes( ax4, title=title) #, xlabel=xlabel,xticks=xticks )
    # ax4.set_title(title, y=0.8)
    ax4.set_ylabel('$M_{\mathrm{atm}}^{\mathrm{i}}/M_{\mathrm{tot}}^{\mathrm{i}}$')
    ax4.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax4.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax4.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.set_label_coords(xcoord_r,ycoord_r)
    ax4.set_xlim( *xlim )
    ax4.set_ylim( 0, 1 )
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles, labels, loc='center left', ncol=1, frameon=1, fancybox=True, framealpha=0.9, fontsize=fs_legend)

    ##########
    # figure f
    ##########
    title = r'(f) Interior volatile mass fraction'
    ax5.semilogx( fig_o.time, vol_mass_interior/vol_mass_total, lw=lw, color="gray", linestyle='-', label=r'Total')
    ax5.semilogx( fig_o.time, H2O_interior_kg/H2O_total_kg, lw=lw, color=blue, linestyle='-', label=r'H$_2$O' )
    ax5.semilogx( fig_o.time, CO2_interior_kg/CO2_total_kg, lw=lw, color=red, linestyle='-', label=r'CO$_2$' )
    fig_o.set_myaxes( ax5, title=title)
    # ax5.set_title(title, y=0.8)
    ax5.set_ylabel('$M_{\mathrm{int}}^{\mathrm{i}}/M_{\mathrm{tot}}^{\mathrm{i}}$')
    ax5.set_xlabel(xlabel)
    ax5.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax5.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax5.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax5.set_xlim( *xlim )
    ax5.set_ylim( 0, 1 )
    ax5.yaxis.tick_right()
    ax5.yaxis.set_label_coords(xcoord_r,ycoord_r)
    ax5.yaxis.set_label_position("right")
    handles, labels = ax5.get_legend_handles_labels()
    ax5.legend(handles, labels, loc='center left', ncol=1, frameon=1, fancybox=True, framealpha=0.9, fontsize=fs_legend)


    fig_o.savefig(6)
    plt.close()

#====================================================================
def main():

    # Read optional argument from console to provide output dir
    output_dir = parser.parse_args().dir

    plot_global( output_dir="output/")
    # plt.show()

#====================================================================

if __name__ == "__main__":

    main()
