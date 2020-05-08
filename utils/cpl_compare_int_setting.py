#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_plot import *

import matplotlib.gridspec as gridspec

def find_nearest(array, value):
    array   = np.asarray(array)
    idx     = (np.abs(array - value)).argmin()
    return array[idx]

#====================================================================
def plot_atmosphere( output_dir, sub_dirs ):

    # article class text width is 4.7747 inches
    # http://tex.stackexchange.com/questions/39383/determine-text-width
    # logger.info( 'building stacked interior atmosphere' )

    width   = 12.00 #* 3.0/2.0
    height  = 6.0
    
    # fig_o   = su.FigureData( 1, 3, width, height, output_dir+'/compare_sfd+melt', units='kyr' )
    # fig_o.fig.subplots_adjust(wspace=0.15, hspace=0.0)

    

    # ax0 = fig_o.ax[0]#[0]
    # ax1 = fig_o.ax[1]#[1]
    # ax2 = fig_o.ax[2]#[0]
    # # ax3 = fig_o.ax[1][1]

    # Define figure with gridspec
    # https://matplotlib.org/3.2.1/tutorials/intermediate/gridspec.html
    fig = plt.figure(tight_layout=True, constrained_layout=False, figsize=[width, height])

    gs = fig.add_gridspec(nrows=2, ncols=4, wspace=0.1, hspace=0.25, left=0.055, right=0.98, top=0.98, bottom=0.08)


    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    ax0b = fig.add_subplot(gs[0, 2:4])
    ax1b = fig.add_subplot(gs[1, 2])
    ax2b = fig.add_subplot(gs[1, 3])

    # sns.set_style("ticks")
    # sns.despine()

    
    handle_l = [] # handles for legend

    ymax_atm_pressure   = 0
    ymin_atm_pressure   = 1000
    ymax_atm_z          = 0
    ymin_atm_z          = 0
    ymax_sp_flux        = 0
    xmin                = 0
    xmax                = 0.1

    fs_legend   = 11
    fs_label    = 11

    # Initiate legends
    legend_ax0_handles = []
    legend_ax0b_handles = []


    # Show wavelenght or wavenumber
    print_wavelength = True

    # # Load runtime helpfile
    # runtime_helpfile = pd.read_csv(output_dir+"runtime_helpfile.csv")

    for subdir in sub_dirs:

        settings = [ "", "_mix", "_sep", "_utbl" ]

        color_idx   = 5

        for nn, setting in enumerate( settings ):

            if nn == 0:
                lw = 1.5
                ls = "-"
            if nn == 1:
                lw = 1.5
                ls = "--"
            if nn == 2:
                lw = 1.5
                ls = "-."
            if nn == 3:
                lw = 1.5
                ls = ":"
            if nn == 4:
                lw = 1.5
                ls = (0, (2, 1))

            if setting == "":
                setting_name = "(a) Fiducial case"
            if setting == "_mix":
                setting_name = r"(b) Variable $\mathcal{L}$"
            if setting == "_sep":
                setting_name = r"(c) Solid–melt sep."
            if setting == "_utbl":
                setting_name = r"(d) Boundary layer"
            if setting == "_sep_mix":
                setting_name = r"Variable $\mathcal{L}$"+"\n"+"+ solid–melt sep."
            if setting == "_sep_mix_utbl":
                setting_name = r"Variable $\mathcal{L}$"+"\n"+"+ solid–melt sep."+"\n"+"+ UTBL"

            data_dir = output_dir+"/"+subdir+setting

            print(data_dir)

            # Read in time sequences
            data_times = su.get_all_output_times(data_dir)
            keys = ( ('atmosphere','mass_liquid'),
                       ('atmosphere','mass_solid'),
                       ('atmosphere','mass_mantle'),
                       ('atmosphere','mass_core'),
                       ('atmosphere','temperature_surface'),
                       ('atmosphere','emissivity'),
                       ('rheological_front_phi','phi_global'),
                       ('atmosphere','Fatm'),
                       ('atmosphere','pressure_surface'),
                       ('rheological_front_dynamic','depth'),
                       ('rheological_front_dynamic','mesh_index')
                       )
            data = su.get_dict_surface_values_for_times( keys, data_times, data_dir )
            mass_liquid         = data[0,:]
            mass_solid          = data[1,:]
            mass_mantle         = data[2,:]
            mass_core           = data[3,:]
            T_surf              = data[4,:]
            emissivity          = data[5,:]
            phi_global          = data[6,:]
            Fatm                = data[7,:]
            P_surf              = data[8,:]
            rheol_front         = data[9,:]
            rheol_front_idx     = data[10,:]

            # FIND SPECIFIC TIMESTAMP

            # Find critical parameter
            RF_depth_crit = 0.01
            # print(rheol_front/np.max(rheol_front))
            RF_depth_crit_num, RF_depth_crit_idx = find_nearest(rheol_front/np.max(rheol_front), RF_depth_crit)
            RF_depth_crit_time   = data_times[RF_depth_crit_idx]
            
            # RF_half_depth, RF_half_depth_idx = find_nearest(rheol_front, 0.5)
            # RF_half_depth_time   = data_times[RF_half_depth_idx]
            # Phi_global_intersect = phi_global[RF_half_depth_idx]
            # print("RF:", RF_half_depth_idx, RF_half_depth, RF_half_depth_time, Phi_global_intersect)

            phi_crit = 0.2
            phi_crit_num, phi_crit_idx = find_nearest(phi_global, phi_crit)
            phi_crit_time = data_times[phi_crit_idx]
            # print("RF:", phi_crit_idx, RF_half_depth, RF_half_depth_time, Phi_global_intersect)

            Fatm_crit = 1e+3
            Fatm_crit_num, Fatm_crit_idx = find_nearest(Fatm, Fatm_crit)
            Fatm_crit_time = data_times[Fatm_crit_idx]

            # Define output time based on which criterion
            if setting == "":
                output_time =  RF_depth_crit_time

            # print(RF_depth_crit_time)

            # # Fix output time
            # output_time = 1e+5

            # Find data_time closest to output_time wanted
            atm_data_times = su.get_all_output_pkl_times(data_dir)
            # print(atm_data_times)
            time, time_idx = find_nearest(atm_data_times, output_time)

            # Find planet mass and radius
            myjson_o  = su.MyJSON( data_dir+'/{}.json'.format(data_times[0]) )
            xx_radius = myjson_o.get_dict_values(['data','radius_b'])*1.0E-3
            r_planet  = np.max(xx_radius*1e3) # m
            core_mass   = myjson_o.get_dict_values(['atmosphere','mass_core'])
            mantle_mass = myjson_o.get_dict_values(['atmosphere','mass_mantle'])
            planet_mass = core_mass + mantle_mass

            print(subdir, setting, output_time)

            label = setting_name

            atm_file = data_dir+"/"+str(int(time))+"_atm.pkl"

            # Read pickle file
            atm_file_stream = open(atm_file,'rb')
            atm = pkl.load(atm_file_stream)
            atm_file_stream.close()

            # color = fig_o.get_color( nn )
            color   = vol_colors[subdir][color_idx]
          
            # # Atmosphere T-Z
            # z_profile = AtmosphericHeight(atm, planet_mass, r_planet) # m
            # z_profile = z_profile*1e-3 # km

            # # Connect interior and atmosphere lines
            # atm.tmp[-1] = atm.ts
            
            

            
            ### INTERIOR

            myjson_o = su.MyJSON( data_dir+"/"+'{}.json'.format(time) )

            # Melt fraction
            mantle_prs = myjson_o.get_dict_values(['data','pressure_b'])*1.0E-9
            mantle_phi = myjson_o.get_dict_values(['data','phi_b'])
            

            # Temperature
            mantle_tmp = myjson_o.get_dict_values(['data','temp_b'])


            # Plots
            if subdir == sub_dirs[0]:

                output_time_print_a = output_time
                
                if print_wavelength == True:
                    # Wavelength / microns
                    wavelength_micron  = [ 1e+4/i for i in atm.band_centres ]
                    flux_micron = [ 1e+4*i for i in atm.net_spectral_flux[:,0]/atm.band_widths ]
                    l1, = ax0.plot( wavelength_micron, flux_micron, ls=ls, color=color, label=label, lw=lw)
                else:               
                    l1, = ax0.plot( atm.band_centres, atm.net_spectral_flux[:,0]/atm.band_widths, ls=ls, color=color, label=label, lw=lw)

                    # Blackbody curves
                    ax0.plot(atm.band_centres,surf_Planck_nu(atm)/atm.band_widths, color=color, ls=ls, lw=0.5, alpha=0.5, label=str(round(atm.ts))+' K blackbody')

                
                ymax_sp_flux = np.max( [ ymax_sp_flux, np.max(atm.net_spectral_flux[:,0]/atm.band_widths)] )
                ax1.plot( mantle_phi, mantle_prs, ls=ls, lw=lw, color=color )
                ax2.plot( mantle_tmp, mantle_prs, ls=ls, lw=lw, color=color )
                legend_ax0_handles.append(l1)
            if subdir == sub_dirs[1]:

                output_time_print_b = output_time

                if print_wavelength == True:
                    # Wavelength / microns
                    wavelength_micron  = [ 1e+4/i for i in atm.band_centres ]
                    flux_micron = [ 1e+4*i for i in atm.net_spectral_flux[:,0]/atm.band_widths ]
                    l1, = ax0b.plot( wavelength_micron, flux_micron, ls=ls, color=color, label=label, lw=lw)
                else:               
                    l1, = ax0b.plot( atm.band_centres, atm.net_spectral_flux[:,0]/atm.band_widths, ls=ls, color=color, label=label, lw=lw)
                    # Blackbody curves
                    ax0b.plot(atm.band_centres,surf_Planck_nu(atm)/atm.band_widths, color=color, ls=ls, lw=0.5, alpha=0.5, label=str(round(atm.ts))+' K blackbody')

                ymax_sp_flux = np.max( [ ymax_sp_flux, np.max(atm.net_spectral_flux[:,0]/atm.band_widths)] )
                ax1b.plot( mantle_phi, mantle_prs, ls=ls, lw=lw, color=color )
                ax2b.plot( mantle_tmp, mantle_prs, ls=ls, lw=lw, color=color )
                legend_ax0b_handles.append(l1)

            # # Viscosity
            # mantle_visc = myjson_o.get_dict_values(['data','visc_b'])
            # # print(mantle_visc)
            # ax3.plot( mantle_visc, mantle_prs, ls=ls, lw=lw, color=color )


            # Fill color and P_surf legends
            # if nn == 0:
            

            # if subdir == sub_dirs[0]:
            #     l2, = ax0.plot( 0, 0, lw=lw, color=qgray, label=latex_float(time)+" yr", ls=ls)
            #     legend_ax0_2_handles.append(l2)

            
            # Reset x-axis boundaries
            xmin = np.min([ xmin, np.min(mantle_phi) ])
            xmax = np.max([ xmax, np.max(mantle_phi) ])

            # color_idx   += 3

    # Print liquidus and solidus
    xx_pres = myjson_o.get_dict_values(['data','pressure_b'])*1.0E-9
    yy_liqt = myjson_o.get_dict_values(['data','liquidus_temp_b'])
    yy_solt = myjson_o.get_dict_values(['data','solidus_temp_b'])
    ax2.fill_betweenx( xx_pres, yy_liqt, yy_solt, facecolor=qmagenta_light, alpha=0.3, linewidth=0 )
    ax2b.fill_betweenx( xx_pres, yy_liqt, yy_solt, facecolor=qmagenta_light, alpha=0.3, linewidth=0 )

    ### Figure settings
    
    # xmin    = 0
    xticks  = np.linspace(xmin, xmax, 10)
    xticks  = [ round(x,2) for x in xticks ]
    title_xcoord = -0.09
    title_ycoord = 0.5

    linthreshy_mantle   = 3
    linthreshy_atm      = 100

    yticks_mantle = [ 0, 1, 2, 3, 10, 30, 70, np.max(mantle_prs) ]
    xticks_melt = [ 0, 0.2, 0.4, 0.6, 0.8, 1 ]
    xticks_tmp = [ 1000, 2000, 3000, 4000 ]

    if print_wavelength == True:
        ax0.set_ylabel(r'Spectral flux density (W m$^{-2}$ $\mu$m$^{-1}$)', fontsize=fs_label)
        ax0.set_xlabel(r'Wavelength $\lambda$ ($\mu$m)', fontsize=fs_label)
        ax0.set_xscale("log")
        ax0.set_yscale("log") 
        # ax0.set_yscale("symlog")
        # ax0.set_xlim(left=0.1, right=100)
        ax0.set_ylim(bottom=1e+2)
        # ax0.set_yticks([1e-10, 1e-5, 1e0, 1e5])
        # ax0.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100])
        # ax0.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30", "100"])
    else:
        ax0.set_ylabel( r'Spectral flux density (W m$^{-2}$ cm$^{-1}$)', fontsize=fs_label)
        ax0.set_xlabel( r'Wavenumber (cm$^{-1}$)', fontsize=fs_label )
        # ax0.set_xlim( left=0, right=7500 )
        # ax0.set_ylim( bottom=0 ) # bottom=0, , top=ymax_sp_flux
        # ax0.set_yscale("symlog", linthreshy=linthreshy_atm)
        ax0.set_yscale("log")

    ax1.set_ylim([0, int(mantle_prs[-1])])
    ax1.set_xlim( np.min(xticks_melt), np.max(xticks_melt) )
    ax1.set_xticks(xticks_melt)
    ax1.set_xlabel( r'Mantle melt fraction, $\phi$ (wt)', fontsize=fs_label )
    ax1.invert_yaxis()
    ax1.set_ylabel( 'Mantle pressure, $P_\mathrm{mantle}$ (GPa)', fontsize=fs_label )
    ax1.set_yscale("symlog", linthreshy=linthreshy_mantle)
    ax1.set_yticks(yticks_mantle)
    ax1.set_yticklabels([ str(int(round(y,0))) for y in yticks_mantle])

    ax2.set_ylim([0, int(mantle_prs[-1])])
    ax2.set_yticklabels([])
    ax2.set_xlim( np.min(xticks_tmp)-500, np.max(xticks_tmp)+500 )
    ax2.set_xticks(xticks_tmp)
    ax2.set_xlabel( r'Mantle temperature, $T$ (K)', fontsize=fs_label )
    ax2.invert_yaxis()
    ax2.set_yscale("symlog", linthreshy=linthreshy_mantle)
    ax2.set_yticks(yticks_mantle)
    ax2.set_yticklabels([])
    
    if print_wavelength == True:
        ax0b.set_xlabel(r'Wavelength $\lambda$ ($\mu$m)', fontsize=fs_label)
        ax0b.set_xscale("log")
        ax0b.set_yscale("log") 
        # ax0b.set_yscale("symlog")
        # ax0b.set_xlim(left=0.1, right=100)
        ax0b.set_ylim(bottom=5e+1)
        # ax0b.set_yticks([1e-10, 1e-5, 1e0, 1e5])
        # ax0b.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100])
        # ax0b.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30", "100"])
    else:
        # ax0b.set_ylabel( r'Spectral flux density (W m$^{-2}$ cm$^{-1}$)', fontsize=fs_label)
        ax0b.set_xlabel( r'Wavenumber (cm$^{-1}$)', fontsize=fs_label )
        # ax0b.set_xlim( left=0, right=7500 )
        # ax0b.set_ylim( bottom=0 ) #, top=ymax_sp_flux
        # ax0b.set_yscale("symlog", linthreshy=linthreshy_atm)
        ax0b.set_yscale("log")

    ax1b.set_ylim([0, int(mantle_prs[-1])])
    # ax1b.set_yticks([])
    # ax1b.yaxis.tick_right()
    ax1b.set_xlim( np.min(xticks_melt), np.max(xticks_melt) )
    ax1b.set_xticks(xticks_melt)
    ax1b.set_xlabel( r'Mantle melt fraction, $\phi$ (wt)', fontsize=fs_label )
    ax1b.invert_yaxis()
    ax1b.set_yscale("symlog", linthreshy=linthreshy_mantle)
    ax1b.set_yticks(yticks_mantle)
    ax1b.set_yticklabels([])

    ax2b.set_ylim([0, int(mantle_prs[-1])])
    # ax2b.yaxis.tick_right()
    # ax2b.yaxis.set_label_position("right")
    ax2b.set_xlim( np.min(xticks_tmp)-500, np.max(xticks_tmp)+500 )
    ax2b.set_xticks(xticks_tmp)
    ax2b.set_xlabel( r'Mantle temperature, $T$ (K)', fontsize=fs_label )
    ax2b.invert_yaxis()
    ax2b.set_yscale("symlog", linthreshy=linthreshy_mantle)
    ax2b.set_yticks(yticks_mantle)
    ax2b.set_yticklabels([])

    try:
        sns.set_style("ticks")
        sns.despine()
    except:
        print("No seaborn.")

    # Legend(s)
    legend_ax0 = ax0.legend(handles=legend_ax0_handles, loc=1, ncol=1, fontsize=fs_legend, framealpha=0.3, title=r"H$_\mathrm{2}$O"+", $t = $"+latex_float(output_time_print_a)+" yr")
    ax0.add_artist(legend_ax0)
    
    # for text in legend_ax0.get_texts():
    #     text.set_color("red")

    legend_ax0b = ax0b.legend(handles=legend_ax0b_handles, loc=1, ncol=1, fontsize=fs_legend, framealpha=0.3, title=r"H$_\mathrm{2}$"+", $t = $"+latex_float(output_time_print_b)+" yr")
    ax0b.add_artist(legend_ax0b)

    ax2.text(0.6, 0.28, 'Mush', color=qmagenta_light, rotation=0, ha="left", va="top", fontsize=fs_label, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax2b.text(0.6, 0.28, 'Mush', color=qmagenta_light, rotation=0, ha="left", va="top", fontsize=fs_label, transform=ax2b.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    
    ax0.text(0.02, 0.015, 'A1', color="k", rotation=0, ha="left", va="bottom", fontsize=fs_label+3, transform=ax0.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax1.text(0.02, 0.015, 'A2', color="k", rotation=0, ha="left", va="bottom", fontsize=fs_label+3, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax2.text(0.02, 0.015, 'A3', color="k", rotation=0, ha="left", va="bottom", fontsize=fs_label+3, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax0b.text(0.02, 0.015, 'B1', color="k", rotation=0, ha="left", va="bottom", fontsize=fs_label+3, transform=ax0b.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax1b.text(0.02, 0.015, 'B2', color="k", rotation=0, ha="left", va="bottom", fontsize=fs_label+3, transform=ax1b.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax2b.text(0.02, 0.015, 'B3', color="k", rotation=0, ha="left", va="bottom", fontsize=fs_label+3, transform=ax2b.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

    # ax0.legend( fancybox=True, framealpha=0.5, ncol=1, fontsize=fs_legend)
    # ax2.legend( fontsize=8, fancybox=True, framealpha=0.5 )
    plt.savefig(output_dir+'/compare_int_setting.pdf', tight_layout=True)

    # fig_o.savefig(1)
    plt.close()

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
        output_dir = os.getcwd()
        print("Output directory:", output_dir)

    # # Define which times are plotted
    # if args.times:
    #     data_times = [ int(time) for time in args.times.split(',') ]
    #     print("Snapshots:", output_times)
    # else:
    #     data_times = su.get_all_output_times(output_dir)
    #     print("Snapshots:", output_times)

    vols    = [ "H2O", "H2" ] #, "CO", "O2"

    output_dir  = "/Users/tim/runs/coupler_tests/set_260bar"
    print("Host directory:", output_dir)

    # Plot fixed set from above
    plot_atmosphere( output_dir=output_dir, sub_dirs=vols )

#====================================================================

if __name__ == "__main__":

    # Import utils- and plot-specific modules
    from modules_utils import *
    from modules_plot import *
    import utils_coupler as cu
    import utils_spider as su

    main()
