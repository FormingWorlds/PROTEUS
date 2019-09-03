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
def plot_global( output_dir='output' ):

    logger.info( 'building atmosphere' )

    width = 12.00 #* 3.0/2.0
    height = 8.0 #/ 2.0
    fig_o = su.FigureData( 3, 2, width, height, output_dir+'/plot_global', units='yr' )
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

    fig_o.time = su.get_all_output_times()
    timeMyr_a = fig_o.time #* 1.0E-6 # Myrs

    keys_t = ( ('atmosphere','mass_liquid'),
               ('atmosphere','mass_solid'),
               ('atmosphere','mass_mantle'),
               ('atmosphere','CO2','liquid_kg'),
               ('atmosphere','CO2','solid_kg'),
               ('atmosphere','CO2','initial_kg'),
               ('atmosphere','CO2','atmosphere_kg'),
               ('atmosphere','CO2','atmosphere_bar'),
               ('atmosphere','H2O','liquid_kg'),
               ('atmosphere','H2O','solid_kg'),
               ('atmosphere','H2O','initial_kg'),
               ('atmosphere','H2O','atmosphere_kg'),
               ('atmosphere','H2O','atmosphere_bar'),
               ('atmosphere','temperature_surface'),
               ('atmosphere','emissivity'),
               ('rheological_front_phi','phi_global'),
               ('atmosphere','Fatm'))

    data_a = su.get_dict_surface_values_for_times( keys_t, fig_o.time )

    out_a = np.column_stack( (timeMyr_a,data_a[15,:],data_a[12,:]) )
    np.savetxt( 'out.dat', out_a )

    mass_liquid_a = data_a[0,:]
    mass_solid_a = data_a[1,:]
    mass_mantle_a = data_a[2,:]
    mass_mantle = mass_mantle_a[0] # time independent

    # compute total mass (kg) in each reservoir
    CO2_liquid_kg_a = data_a[3,:]
    CO2_solid_kg_a = data_a[4,:]
    CO2_total_kg_a = data_a[5,:]
    CO2_total_kg = CO2_total_kg_a[0] # time-independent
    CO2_atmos_kg_a = data_a[6,:]
    CO2_atmos_a = data_a[7,:]
    CO2_escape_kg_a = CO2_total_kg - CO2_liquid_kg_a - CO2_solid_kg_a - CO2_atmos_kg_a

    H2O_liquid_kg_a = data_a[8,:]
    H2O_solid_kg_a = data_a[9,:]
    H2O_total_kg_a = data_a[10,:]
    H2O_total_kg = H2O_total_kg_a[0] # time-independent
    H2O_atmos_kg_a = data_a[11,:]
    H2O_atmos_a = data_a[12,:]
    H2O_escape_kg_a = H2O_total_kg - H2O_liquid_kg_a - H2O_solid_kg_a - H2O_atmos_kg_a

    temperature_surface_a = data_a[13,:]
    emissivity_a = data_a[14,:]
    phi_global = data_a[15,:]
    Fatm = data_a[16,:]

    #xticks = [1E-5,1E-4,1E-3,1E-2,1E-1]#,1]
    #xticks = [1.0E-2, 1.0E-1, 1.0E0, 1.0E1, 1.0E2,1.0E3] #[1E-6,1E-4,1E-2,1E0,1E2,1E4,1E6]#,1]
    #xticks = (1E-6,1E-5,1E-4,1E-3,1E-2,1E-1,1E0)
    xlabel = 'Time (yr)'
    #xlim = (1.0E-2, 4550)
    xlim = (1e1,1e7)

    red = (0.5,0.1,0.1)
    blue = (0.1,0.1,0.5)
    black = 'black'

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

    ##########
    # figure a
    ##########
    title = r'(a) Heat flux to space'
    ylabel = '$F_\mathrm{atm}$\n$(W/m^2)$'
    yticks = (1.0E2,1.0E3,1.0E4,1.0E5,1.0E6,1.0E7)
    h1, = ax0.loglog( timeMyr_a, Fatm,'k', lw=lw )
    fig_o.set_myaxes( ax0, title=title, ylabel=ylabel, yticks=yticks)#, xlabel=xlabel, xticks=xticks )
    # ax0.yaxis.tick_right()
    # ax0.yaxis.set_label_position("right")
    ax0.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax0.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax0.set_xlim( *xlim )
    ax0.yaxis.set_label_coords(-0.13,0.5)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, loc='upper right', ncol=1, frameon=0, fontsize=fs_legend)

    ##########
    # figure b
    ##########
    title = r'(b) Surface temperature'
    ylabel = '$T_\mathrm{s}$\n$(K)$'
    yticks = [200, 500, 1000, 1500, 2000, 2500, 3000]
    h1, = ax1.semilogx( timeMyr_a, temperature_surface_a, 'k-', lw=lw, label=r'Surface temp, $T_s$' )
    #ax2b = ax2.twinx()
    #h2, = ax2b.loglog( timeMyr_a, emissivity_a, 'k--', label=r'Emissivity, $\epsilon$' )
    fig_o.set_myaxes( ax1, title=title, ylabel=ylabel, yticks=yticks)#, xlabel=xlabel )
    ax1.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax1.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.set_xlim( *xlim )
    ax1.yaxis.set_label_coords(-0.13,0.5)
    #ax1.set_ylim( 1050, 1850 )
    #ax1.set_xlim( 1E-5 , 1 )
    ax1.set_ylim(200,np.max(yticks))
    # ax1.set_title('(a) Surface temperature', fontname='Arial', fontsize=fs_title)

    ##########
    # figure c
    ##########
    title = r'(c) Global mantle melt fraction'
    ylabel = '$\phi_\mathrm{g}$'
    trans = transforms.blended_transform_factory(
        ax2.transData, ax2.transAxes)
    h1, = ax2.semilogx( timeMyr_a, phi_global, color=black, linestyle='-', lw=lw, label=r'Melt, $\phi_g$')
    # h2, = ax2.semilogx( timeMyr_a, mass_liquid_a / mass_mantle, 'k--', label='melt' )
    fig_o.set_myaxes( ax2, title=title, xlabel=xlabel, ylabel=ylabel )#, xlabel=xlabel, xticks=xticks )
    ax2.set_xlim( *xlim )
    ax2.set_ylim( 0, 1 )
    # ax0.annotate(title, xy=title_xy, xycoords=title_xycoords, ha=title_ha, va=title_va, fontsize=title_fs)
    ax2.yaxis.set_label_coords(-0.13,0.5)
    # handles, labels = ax2.get_legend_handles_labels()
    # ax2.legend(handles, labels, loc='center left', ncol=1, frameon=0)

    ##########
    # figure d
    ##########
    # if 1:
    title = r'(d) Atmospheric volatile partial pressure'
    ylabel = '$p_{\mathrm{vol}}$\n$(\mathrm{bar})$'
    trans = transforms.blended_transform_factory(
        ax3.transData, ax3.transAxes)
    #for cc, cont in enumerate(phi_cont_l):
    #    ax0.axvline( phi_time_l[cc], ymin=0.05, ymax=0.7, color='0.25', linestyle=':' )
    #    label = cont #int(cont*100) # as percent
    #    ax0.text( phi_time_l[cc], 0.40, '{:2d}'.format(label), va='bottom', ha='center', rotation=90, bbox=dict(facecolor='white'), transform=trans )
    #ax0.text( 0.1, 0.9, '$\phi (\%)$', ha='center', va='bottom', transform=ax0.transAxes )
    h1, = ax3.semilogx( timeMyr_a, H2O_atmos_a, color=blue, linestyle='-', lw=lw, label=r'H$_2$O')
    h2, = ax3.semilogx( timeMyr_a, CO2_atmos_a, color=red, linestyle='-', lw=lw, label=r'CO$_2$')
    fig_o.set_myaxes( ax3, title=title, ylabel=ylabel )#, xlabel=xlabel, xticks=xticks )
    ax3.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax3.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax3.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax3.set_xlim( *xlim )
    ax3.set_ylim( 0, 350 )
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.set_label_coords(1.12,0.5)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels, loc='center left', ncol=1, frameon=0, fontsize=fs_legend)

    ##########
    # figure e
    ##########
    title = r'(e) Atmospheric volatile mass fraction'
    h1, = ax4.semilogx( timeMyr_a, H2O_atmos_kg_a / H2O_total_kg, lw=lw, color=blue, linestyle='-', label=r'H$_2$O')
    h2, = ax4.semilogx( timeMyr_a, CO2_atmos_kg_a / CO2_total_kg, lw=lw, color=red, linestyle='-', label=r'CO$_2$' )
    fig_o.set_myaxes( ax4, title=title, ylabel='$X_\mathrm{atm}$') #, xlabel=xlabel,xticks=xticks )
    ax4.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax4.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax4.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.set_label_coords(1.12,0.5)
    ax4.set_xlim( *xlim )
    ax4.set_ylim( 0, 1 )
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles, labels, loc='center left', ncol=1, frameon=0, fontsize=fs_legend)

    ##########
    # figure f
    ##########
    title = r'(f) Interior volatile mass fraction'
    #h5, = ax1.semilogx( timeMyr_a, mass_liquid_a / mass_mantle, 'k--', label='melt' )
    h1, = ax5.semilogx( timeMyr_a, (H2O_liquid_kg_a+H2O_solid_kg_a) / H2O_total_kg, lw=lw, color=blue, linestyle='-', label=r'H$_2$O' )
    h2, = ax5.semilogx( timeMyr_a, (CO2_liquid_kg_a+CO2_solid_kg_a) / CO2_total_kg, lw=lw, color=red, linestyle='-', label=r'CO$_2$' )
    fig_o.set_myaxes( ax5, title=title, ylabel='$X_{\mathrm{int}}$', xlabel=xlabel)#,xticks=xticks )
    ax5.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax5.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax5.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax5.set_xlim( *xlim )
    ax5.set_ylim( 0, 1 )
    ax5.yaxis.tick_right()
    ax5.yaxis.set_label_coords(1.12,0.5)
    ax5.yaxis.set_label_position("right")
    handles, labels = ax5.get_legend_handles_labels()
    ax5.legend(handles, labels, loc='center left', ncol=1, frameon=0, fontsize=fs_legend)


    fig_o.savefig(6)
    plt.close()

#====================================================================
def main():

    # Read optional argument from console to provide output dir
    output_dir = parser.parse_args().dir

    plot_global( output_dir )
    # plt.show()

#====================================================================

if __name__ == "__main__":

    main()
