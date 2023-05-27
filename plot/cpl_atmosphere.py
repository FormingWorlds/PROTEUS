#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_ext import *
from utils.plot import *
from utils.spider import MyJSON, get_all_output_times

from AEOLUS.modules.spectral_planck_surface import surf_Planck_nu

#====================================================================
def plot_atmosphere( output_dir, times ):

    # article class text width is 4.7747 inches
    # http://tex.stackexchange.com/questions/39383/determine-text-width

    # logger.info( 'building stacked interior atmosphere' )

    width = 6.00 #* 3.0/2.0
    height = 12.0
    fig_o = FigureData( 3, 1, width, height, output_dir+'/plot_atmosphere', units='kyr', times=times ) 
    # fig1 = plt.figure()
    fig_o.fig.subplots_adjust(wspace=0.07,hspace=0.2)
    fig_o.time = times

    # sns.set_style("ticks")
    # sns.despine()

    ax0 = fig_o.ax[0]#[0]
    ax1 = fig_o.ax[1]#[0]
    ax2 = fig_o.ax[2]#[0]
    # ax3 = fig_o.ax[1][1]

    time = fig_o.time[0] # first timestep since liquidus and solidus
                         # are time-independent

    myjson_o = MyJSON( output_dir+'/{}.json'.format(time) )

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

    fig_o.set_cmap(sci_colormaps["vik_r"])

    ymax_atm_pressure = 0
    ymin_atm_pressure = 1000
    ymax_atm_z = 0
    ymin_atm_z = 0
    ymax_sp_flux = 0

    # Find planet mass
    core_mass   = myjson_o.get_dict_values(['atmosphere','mass_core'])
    mantle_mass = myjson_o.get_dict_values(['atmosphere','mass_mantle'])
    planet_mass = core_mass + mantle_mass

    # # Load runtime helpfile
    # runtime_helpfile = pd.read_csv(output_dir+"runtime_helpfile.csv")

    for nn, time in enumerate( fig_o.time ):

        atm_file = output_dir+"/"+str(int(time))+"_atm.pkl"

        if os.path.exists(atm_file):

            # Read pickle file
            atm_file_stream = open(atm_file,'rb')
            atm = pkl.load(atm_file_stream)
            atm_file_stream.close()

            # read json
            myjson_o = MyJSON( output_dir+"/"+'{}.json'.format(time) )

            color = fig_o.get_color( 1.0*nn/len(fig_o.time) )
            # use melt fraction to determine mixed region
            MIX = myjson_o.get_mixed_phase_boolean_array( 'basic' )

            label = latex_float(time)+" yr"
            
            # Atmosphere T-Z
            z_profile = AtmosphericHeight(atm, planet_mass, r_planet) # m
            z_profile = z_profile*1e-3 # km
            ax0.plot( atm.tmp, z_profile, '-', color=color, label=label, lw=1.5)

            # print(time,atm.tmp, atm.p)

            # Atmosphere T-P
            ax1.semilogy( atm.tmp, atm.p/1e5, '-', color=color, label=label, lw=1.5)

            # Spectral flux density per wavenumber
            # ax2.plot( atm.band_centres, atm.LW_spectral_flux_up[:,0]/atm.band_widths, '-', color=color, label=label, lw=1.5)
            ax2.plot( atm.band_centres, atm.net_spectral_flux[:,0]/atm.band_widths, '-', color=color, label=label, lw=1.5)
            ymax_sp_flux = np.max( [ ymax_sp_flux, np.max(atm.net_spectral_flux[:,0]/atm.band_widths)] )

            # Blackbody curves
            ax2.plot(atm.band_centres,surf_Planck_nu(atm)/atm.band_widths, color=color, ls='--', lw=1.0, alpha=0.5, label=str(round(atm.ts))+' K blackbody')

            # Reset y-axis boundaries
            if np.min(atm.p) > 0:
                ymax_atm_pressure = np.max([ymax_atm_pressure, np.max(atm.p/1e5)])
                ymin_atm_pressure = np.min([ymin_atm_pressure, np.min(atm.p/1e5)])
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
    ax0.set_ylabel("Atmosphere height, $z_\mathrm{atm}$ (km)")
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
    ax1.set_ylabel("Atmosphere pressure, $P$ (bar)")
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
    
    ax2.set_ylabel( 'Spectral flux density (W m$^{-2}$ cm$^{-1}$)', rotation=90)
    # ax2.yaxis.set_label_position("right")
    # ax2.yaxis.tick_right()
    ax2.set_xlim( left=0, right=20000 )
    ax2.set_ylim( bottom=0, top=ymax_sp_flux )
    ax2.yaxis.set_label_coords(title_xcoord,title_ycoord)

    # Legend
    ax1.legend( fontsize=8, fancybox=True, framealpha=0.5 )
    # ax2.legend( fontsize=8, fancybox=True, framealpha=0.5 )

    plt.close()
    plt.ioff()
    fig_o.savefig(1)

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

# def plot_current_mixing_ratio( output_dir, times, vulcan_setting ):

#     fig, ax1 = plt.subplots()

#     time = str(times)

#     # # Read atmospheric chemistry
#     # atm_chemistry       = pd.read_csv(output_dir+time+"_atm_chemistry_volume.dat", skiprows=1, delim_whitespace=True) # skipfooter=1

#     if vulcan_setting == 0 or vulcan_setting == 1:
#         runtime_helpfile = pd.read_csv(output_dir+"/"+"runtime_helpfile.csv", delim_whitespace=True)

#         for vol in volatile_species:
#             if runtime_helpfile.iloc[-1][vol+"_mr"] > 0.:
#                 ax1.axvline(x=runtime_helpfile.iloc[-1][vol+"_mr"], linewidth=lw+0.5, color=dict_colors[vol+"_2"], ls=":")

#     atm_file = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
    
#     ax1.set_xlabel("Mole fraction, log ($n^{\mathrm{i}}$/$n_{\mathrm{tot}}$)")
#     ax1.set_ylabel("Pressure, $P_{\mathrm{tot}}$ (bar)")
#     ax1.set_xscale("log")
#     ax1.set_yscale("log")
#     ax1.set_xlim(left=1e-16, right=2e+0)
#     ax1.set_ylim(np.max(atm.p), np.min(atm.p))

#     # Temperature y-axis on the right
#     ax1b = ax1.twinx()
#     ax1b.plot( atm_chemistry["H2O"], atm_chemistry["Temp"], alpha=0.0)
#     ax1b.set_ylim(np.max(atm_chemistry["Temp"]), np.min(atm_chemistry["Temp"]))
#     if not np.min(atm_chemistry["Temp"]) == np.max(atm_chemistry["Temp"]):
#         ax1b.set_yticks(
#          [ 
#             int(np.max(atm_chemistry["Temp"])), 
#             int(np.min(atm_chemistry["Temp"])+0.75*(np.max(atm_chemistry["Temp"])-np.min(atm_chemistry["Temp"]))), 
#             int(np.min(atm_chemistry["Temp"])+0.5*(np.max(atm_chemistry["Temp"])-np.min(atm_chemistry["Temp"]))), 
#             int(np.min(atm_chemistry["Temp"])+0.25*(np.max(atm_chemistry["Temp"])-np.min(atm_chemistry["Temp"]))), 
#             int(np.min(atm_chemistry["Temp"]))
#          ])
#     ax1b.yaxis.tick_right()
#     ax1b.set_ylabel( 'Temperature, $T$ (K)' )

#     ax1.set_title("Time = "+latex_float(float(time))+" yr", fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va)


#     ax1.legend(loc=2, fancybox=True, framealpha=0.9)
#     plt.xticks([1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0], ["-16", "-14", "-12", "-10", "-8", "-6", "-4", "-2", "0"])

#     plt.savefig(output_dir+"plot_atm_chemistry_"+str(time)+".pdf", bbox_inches = 'tight',
#     pad_inches = 0.1)
#     plt.close()


#====================================================================
def main():

    # Optional command line arguments for running from the terminal
    # Usage: $ python plot_atmosphere.py -t 0,718259
    parser = argparse.ArgumentParser(description='COUPLER plotting script')
    parser.add_argument('-odir', '--output_dir', type=str, help='Full path to output directory');
    parser.add_argument('-t', '--times', type=str, help='Comma-separated (no spaces) list of times');
    args = parser.parse_args()

    # Define output directory for plots
    if args.output_dir:
        output_dir = args.output_dir
        print("Output directory:", output_dir)
        
    else:
        output_dir = os.getcwd() + "/output/"
        print("Output directory:", output_dir)

    # Define which times are plotted
    if args.times:
        plot_list = [ int(time) for time in args.times.split(',') ]
        print("Snapshots:", plot_list)
    else:
        output_list = get_all_output_times(output_dir)

        if len(output_list) <= 8:
            plot_list = output_list
        else:
            plot_list = [ output_list[0], output_list[int(round(len(output_list)*(2./100.)))], output_list[int(round(len(output_list)*(15./100.)))], output_list[int(round(len(output_list)*(22./100.)))], output_list[int(round(len(output_list)*(33./100.)))], output_list[int(round(len(output_list)*(50./100.)))], output_list[int(round(len(output_list)*(66./100.)))], output_list[-1] ]
        print("Snapshots:", plot_list)

    # # Plot only last snapshot
    # plot_current_mixing_ratio( output_dir=output_dir, times=plot_list[-1], vulcan_setting=0 )

    # Plot fixed set from above
    plot_atmosphere( output_dir=output_dir, times=plot_list )

#====================================================================

if __name__ == "__main__":

    main()
