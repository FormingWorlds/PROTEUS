#!/usr/bin/env python

# Import utils- and plot-specific modules
from utils.modules_plot import *

# import logging
# import spider_coupler_utils as su
# import matplotlib.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import matplotlib.ticker as ticker
# import argparse
# import matplotlib
# import pandas as pd
# import mmap
# import seaborn as sns
# # https://seaborn.pydata.org/tutorial/aesthetics.html
# sns.set_style("white")

# # Output dir, optional argument: https://towardsdatascience.com/learn-enough-python-to-be-useful-argparse-e482e1764e05
# parser = argparse.ArgumentParser(description='Define file output directory.')
# parser.add_argument('--dir', default="output/", help='Provide path to output directory.' )

# # Specify varying volatile abundances: $ python plot_atmosphere.py -H2 0,10
# parser = argparse.ArgumentParser(description='COUPLER optional command line arguments')
# parser.add_argument('--dir', default="output/", help='Provide path to output directory.' )
# parser.add_argument('-H2O', type=float, help='H2O initial abundance (ppm wt)')
# # args = parser.parse_args()

# # Define Crameri colormaps (+ recursive)
# from matplotlib.colors import LinearSegmentedColormap
# for name in [ 'acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc', 'buda',
#            'cork', 'davos', 'devon', 'grayC', 'hawaii', 'imola', 'lajolla',
#            'lapaz', 'lisbon', 'nuuk', 'oleron', 'oslo', 'roma', 'tofino',
#            'tokyo', 'turku', 'vik' ]:
#     file = os.path.join("plot/ScientificColourMaps5/", name + '.txt')
#     cm_data = np.loadtxt(file)
#     vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
#     vars()[name+"_r"] = LinearSegmentedColormap.from_list(name, cm_data[::-1])

# logger = su.get_my_logger(__name__)

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

# color_cycle2 = [ "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c", "#7f7f7f", "#bcbd22", "#17becf" ]

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

vol_latex = {
    "H2O": r"H$_2$O",
    "CO2": r"CO$_2$",
    "H2": r"H$_2$",
    "CH4": r"CH$_4$",
    "CO": r"CO",
    "O2": r"O$_2$",
    "N2": r"N$_2$",
    "S": r"S",
    "He": r"S"
}

# Plot settings
lw       = 1.5
fscale   = 1.1
fsize    = 18
fs_title = 18
width = 12.00 #* 3.0/2.0
height = 8.0 #/ 2.0

volatile_species = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2", "S", "He" ]

#====================================================================
def plot_global( output_dir ):

    # logger.info( 'building atmosphere' )

    fig_o = su.FigureData( 3, 2, width, height, output_dir+'/plot_global', units='yr' )
    fig_o.fig.subplots_adjust(wspace=0.05,hspace=0.1)

    # Subplot titles
    title_fs   = 10
    title_xy   = (0.5, 0.75)
    title_x    = title_xy[0]
    title_y    = title_xy[1]
    title_xycoords = 'axes fraction'
    title_ha   = "center"
    title_va   = "center"
    title_font = 'Arial'

    fs_legend = 9

    ax0 = fig_o.ax[0][0]
    ax1 = fig_o.ax[1][0]
    ax2 = fig_o.ax[2][0]
    ax3 = fig_o.ax[0][1]
    ax4 = fig_o.ax[1][1]
    ax5 = fig_o.ax[2][1]

    # runtime_helpfile = pd.read_csv(output_dir+"runtime_helpfile.csv", sep=" ")
    # print(runtime_helpfile)

    fig_o.time = su.get_all_output_times(output_dir)
    print("Times", fig_o.time)

    ########## Global properties
    keys_t = ( ('atmosphere','mass_liquid'),
               ('atmosphere','mass_solid'),
               ('atmosphere','mass_mantle'),
               ('atmosphere','mass_core'),
               ('atmosphere','temperature_surface'),
               ('atmosphere','emissivity'),
               ('rheological_front_phi','phi_global'),
               ('atmosphere','Fatm'),
               ('atmosphere','pressure_surface')
               )
    data_a = su.get_dict_surface_values_for_times( keys_t, fig_o.time, output_dir )
    mass_liquid         = data_a[0,:]
    mass_solid          = data_a[1,:]
    mass_mantle         = data_a[2,:]
    mass_core           = data_a[3,:]
    T_surf              = data_a[4,:]
    emissivity          = data_a[5,:]
    phi_global          = data_a[6,:]
    Fatm                = data_a[7,:]
    P_surf              = data_a[8,:]


    xlabel = 'Time (yr)'
    xlim = (1e1,1e7)

    red = (0.5,0.1,0.1)
    blue = (0.1,0.1,0.5)
    black = 'black'

    xcoord_l = -0.10
    ycoord_l = 0.5
    xcoord_r = 1.09
    ycoord_r = 0.5

    rolling_mean = 0

    ##########
    # figure a
    ##########
    title = r'(a) Heat flux to space'
    ylabel = '$F_\mathrm{atm}$ (W/m$^2$)'
    
    if rolling_mean == 1:
        ax0.loglog( fig_o.time[:10], Fatm[:10], 'k', lw=lw, alpha=1.0 )
        nsteps = 6
        Fatm_rolling = np.convolve(Fatm, np.ones((nsteps,))/nsteps, mode='valid')
        Time_rolling = np.convolve(fig_o.time, np.ones((nsteps,))/nsteps, mode='valid')
        ax0.loglog( Time_rolling, Fatm_rolling,'k', lw=lw )
    else:
        ax0.loglog( fig_o.time, Fatm, 'k', lw=lw, alpha=1.0 )
        
    fig_o.set_myaxes( ax0)#, title=title)#, yticks=yticks, xlabel=xlabel, xticks=xticks )
    # ax0.yaxis.tick_right()
    # ax0.yaxis.set_label_position("right")
    ax0.set_ylabel(ylabel)
    ax0.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax0.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax0.set_xlim( *xlim )
    ax0.set_xticklabels([])
    ax0.yaxis.set_label_coords(xcoord_l,ycoord_l)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, loc='upper right', ncol=1, frameon=0, fontsize=fs_legend)
    ax0.set_title(title, fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va)

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
    h1, = ax1.semilogx( fig_o.time, T_surf, ls="-", lw=lw, color=vol_colors["black_2"], label=r'Surface temp, $T_s$' )
    #ax2b = ax2.twinx()
    #h2, = ax2b.loglog( timeMyr_a, emissivity_a, 'k--', label=r'Emissivity, $\epsilon$' )
    fig_o.set_myaxes( ax1, title=title, yticks=yticks)#, xlabel=xlabel )
    ax1.set_ylabel(ylabel)
    ax1.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax1.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.set_xlim( *xlim )
    ax1.set_xticklabels([])
    ax1.yaxis.set_label_coords(xcoord_l,ycoord_l)
    ax1.set_ylim(ymin,ymax)
    ax1.set_title(title, fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va)
    ##########
    # figure c
    ##########
    title = r'(c) Global mantle melt fraction'
    ylabel = '$\phi_{\mathrm{mantle}}$'
    ax2.semilogx( fig_o.time, phi_global, color=vol_colors["black_2"], linestyle='-', lw=lw, label=r'Melt, $\phi_{\mathrm{mantle}}$')
    # h2, = ax2.semilogx( timeMyr_a, mass_liquid_a / mass_mantle, 'k--', label='melt' )
    fig_o.set_myaxes( ax2, xlabel=xlabel )#, xlabel=xlabel, xticks=xticks, title=title )
    ax2.set_ylabel(ylabel)
    ax2.set_xlim( *xlim )
    # ax2.set_ylim( 0, 1. )
    ax2.set_ylabel(ylabel)
    ax2.yaxis.set_label_coords(xcoord_l,ycoord_l)
    ax2.set_title(title, fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va)

    ### Plot axes setup
    ##########
    # figure d
    ##########
    title_ax3 = r'(d) Surface volatile partial pressure'
    # Total pressure
    ax3.semilogx( fig_o.time, P_surf, color=vol_colors["black_2"], linestyle='-', lw=lw, label=r'Total')
    ##########
    # figure e
    ##########
    title_ax4 = r'(e) Atmospheric volatile mass fraction'
    ##########
    # figure f
    ##########
    title_ax5 = r'(f) Interior volatile mass fraction'

    ########## Volatile species-specific plots
    
    # Check for times when volatile is present in data dumps
    for vol in volatile_species:

        vol_times   = []
        # M_atm_total = np.zeros(len(fig_o.time))

        print(vol+":", end=" ")

        # For all times
        for sim_time in fig_o.time:

            # Define file name
            json_file = output_dir+"/"+str(int(sim_time))+".json"

            # For string check
            vol_str = '"'+vol+'"'

            # # Find the times with the volatile in the JSON file
            # # https://stackoverflow.com/questions/4940032/how-to-search-for-a-string-in-text-files
            with open(json_file, 'rb', 0) as file, \
                mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
                # if s.find(vol.encode()) != -1:
                if s.find(vol_str.encode()) != -1:
                    print(sim_time, end=" ")
                    keys_t = ( ('atmosphere', vol, 'liquid_kg'),
                               ('atmosphere', vol, 'solid_kg'),
                               ('atmosphere', vol, 'initial_kg'),
                               ('atmosphere', vol, 'atmosphere_kg'),
                               ('atmosphere', vol, 'atmosphere_bar')
                               )

                    vol_times.append(sim_time)

        print()

        # # Find the times the volatile is *not* present
        # np.setdiff1d(fig_o.time,vol_times)

        # Only for volatiles that are present at some point 
        if vol_times:

            # Get the data for these files
            data_vol = su.get_dict_surface_values_for_times( keys_t, vol_times, output_dir )
            vol_liquid_kg       = data_vol[0,:]
            vol_solid_kg        = data_vol[1,:]
            vol_initial_kg      = data_vol[2,:]
            vol_atm_kg          = data_vol[3,:]
            vol_atm_pressure    = data_vol[4,:]
            vol_interior_kg     = vol_liquid_kg   + vol_solid_kg
            vol_total_kg        = vol_interior_kg + vol_atm_kg

            ##########
            # figure d
            ##########
            ax3.semilogx( vol_times, vol_atm_pressure, color=vol_colors[vol+"_2"], linestyle='-', lw=lw, label=vol_latex[vol])
            ##########
            # figure e
            ##########
            ax4.semilogx( vol_times, vol_atm_kg/vol_total_kg, lw=lw, color=vol_colors[vol+"_2"], linestyle='-', label=vol_latex[vol])
            ##########
            # figure f
            ##########
            # ax5.semilogx( fig_o.time, vol_mass_interior/vol_mass_total, lw=lw, color="gray", linestyle='-', label=r'Total')
            ax5.semilogx( vol_times, vol_interior_kg/vol_total_kg, lw=lw, color=vol_colors[vol+"_2"], linestyle='-', label=vol_latex[vol] )

    ##########
    # figure d
    ##########
    fig_o.set_myaxes( ax3)#, xlabel=xlabel, xticks=xticks, title=title_ax3 )
    # ax3.set_title(title, y=0.8)
    ax3.set_ylabel('$p^{\mathrm{i}}$ (bar)')
    ax3.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax3.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax3.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax3.set_xlim( *xlim )
    # ax3.set_ylim( bottom=0 )
    # https://brohrer.github.io/matplotlib_ticks.html#tick_style
    # ax3.tick_params(axis="x", direction="in", length=3, width=1)
    # ax3.tick_params(axis="y", direction="in", length=10, width=1)
    ax3.set_xticklabels([])
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.set_label_coords(xcoord_r,ycoord_r)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels, ncol=1, frameon=1, fancybox=True, framealpha=0.9, fontsize=fs_legend) # , loc='center left'
    ax3.set_title(title_ax3, fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va)
    ##########
    # figure e
    ##########
    # Check atmospheric comparisons for mass conservation
    runtime_helpfile = pd.read_csv(output_dir+"/"+"runtime_helpfile.csv", delim_whitespace=True)
    ax4.semilogx( runtime_helpfile["Time"], runtime_helpfile["M_atm"]/runtime_helpfile.iloc[0]["M_atm"], lw=lw, color=vol_colors["black_2"], linestyle='-')
    # ax4.semilogx( runtime_helpfile.loc[runtime_helpfile['Input'] == "Interior"]["Time"], runtime_helpfile.loc[runtime_helpfile['Input'] == "Interior"]["H_mol_total"]/runtime_helpfile.iloc[0]["H_mol_total"], lw=lw, color=vol_colors["black_3"], linestyle='--')
    # ax4.semilogx( runtime_helpfile.loc[runtime_helpfile['Input'] == "Atmosphere"]["Time"], runtime_helpfile.loc[runtime_helpfile['Input'] == "Atmosphere"]["C/H_atm"]/runtime_helpfile.iloc[0]["C/H_atm"], lw=lw, color=vol_colors["black_1"], linestyle=':')
    print(runtime_helpfile[["Time", "Input", "M_atm", "M_atm_kgmol", "H_mol_atm", "H_mol_solid", "H_mol_liquid", "H_mol_total", "O_mol_total", "O/H_atm"]])

    fig_o.set_myaxes( ax4) #, xlabel=xlabel,xticks=xticks, title=title_ax4 )
    # ax4.set_title(title, y=0.8)
    ax4.set_ylabel('$M_{\mathrm{atm}}^{\mathrm{i}}/M_{\mathrm{total}}^{\mathrm{i}}$')
    ax4.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax4.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax4.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax4.yaxis.tick_right()
    ax4.set_xticklabels([])
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.set_label_coords(xcoord_r,ycoord_r)
    ax4.set_xlim( *xlim )
    # ax4.set_ylim( 0, 1 )
    handles, labels = ax4.get_legend_handles_labels()
    # ax4.legend(handles, labels, ncol=1, frameon=1, fancybox=True, framealpha=0.9, fontsize=fs_legend) # , loc='center left'
    ax4.set_title(title_ax4, fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va)


    ##########
    # figure f
    ##########
    fig_o.set_myaxes( ax5, title=title_ax5)
    # ax5.set_title(title, y=0.8)
    ax5.set_ylabel('$M_{\mathrm{int}}^{\mathrm{i}}/M_{\mathrm{total}}^{\mathrm{i}}$')
    ax5.set_xlabel(xlabel)
    ax5.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax5.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax5.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax5.set_xlim( *xlim )
    # ax5.set_ylim( 0, 1 )
    ax5.yaxis.tick_right()
    ax5.yaxis.set_label_coords(xcoord_r,ycoord_r)
    ax5.yaxis.set_label_position("right")
    handles, labels = ax5.get_legend_handles_labels()
    # ax5.legend(handles, labels, ncol=1, frameon=1, fancybox=True, framealpha=0.9, fontsize=fs_legend) # , loc='center left'
    ax5.set_title(title_ax5, fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va)


    fig_o.savefig(6)
    plt.close()

#====================================================================
def main():

    # Read optional argument from console to provide output dir
    output_dir_read = parser.parse_args().dir

    # output_dir_read = os.getcwd()+"/output/save/2019-11-01_11-16-35/"
    # print(output_dir_read)

    # plot_global( output_dir="output/")
    # plt.show()
    plot_global( output_dir=output_dir_read)

#====================================================================

if __name__ == "__main__":

    main()
