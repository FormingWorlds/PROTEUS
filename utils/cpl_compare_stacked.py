#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_plot import *

def find_nearest(array, value):
    array   = np.asarray(array)
    idx     = (np.abs(array - value)).argmin()
    return array[idx]

#====================================================================
def plot_atmosphere( output_dir, sub_dirs, output_times ):

    # article class text width is 4.7747 inches
    # http://tex.stackexchange.com/questions/39383/determine-text-width
    # logger.info( 'building stacked interior atmosphere' )

    width   = 6.00 #* 3.0/2.0
    height  = 12.0
    fig_o   = su.FigureData( 2, 1, width, height, output_dir+'/compare_stacked', units='kyr' )
    fig_o.fig.subplots_adjust(wspace=0.15, hspace=0.0)

    # sns.set_style("ticks")
    # sns.despine()

    ax0 = fig_o.ax[0]#[0]
    ax1 = fig_o.ax[1]#[1]
    # ax2 = fig_o.ax[1][0]
    # ax3 = fig_o.ax[1][1]

    handle_l = [] # handles for legend

    ymax_atm_pressure   = 0
    ymin_atm_pressure   = 1000
    ymax_atm_z          = 0
    ymin_atm_z          = 0
    ymax_sp_flux        = 0
    xmin                = 100
    xmax                = 3000

    fs_legend   = 11
    fs_label    = 11

    # Initiate legends
    legend_ax0_1_handles = []
    legend_ax0_2_handles = []

    # # Load runtime helpfile
    # runtime_helpfile = pd.read_csv(output_dir+"runtime_helpfile.csv")

    for subdir in sub_dirs:

        # if subdir in [ "H2O", "H2", "CO2", "CH4", "N2" ]:
        # if output_times
        #     lw = 2.0
        #     ls = "-"
        # if subdir in [ "O2", "CO" ]:
        #     lw = 1.5
        #     ls = "--"

        data_dir = output_dir+"/"+subdir

        # fig_o.set_colors(cmap=vol_colors[subdir]) # "magma_r"

        # Get data times for subdir
        data_times = su.get_all_output_pkl_times(data_dir)

        # print(data_times)

        # Find solidus and liquidus lines
        # Use first timestep since liquidus and solidus are time-independent
        time        = data_times[0]                    
        myjson_o    = su.MyJSON( data_dir+'/{}.json'.format(time) )
        xx_pres     = myjson_o.get_dict_values(['data','pressure_b'])*1.0E-9
        xx_pres_s   = myjson_o.get_dict_values(['data','pressure_s'])*1.0E-9
        xx_radius   = myjson_o.get_dict_values(['data','radius_b'])*1.0E-3
        xx_depth    = xx_radius[0] - xx_radius
        xx_radius_s = myjson_o.get_dict_values(['data','radius_s'])*1.0E-3
        xx_depth_s  = xx_radius_s[0] - xx_radius_s
        r_planet    = np.max(xx_radius*1e3) # m

        # Find planet mass
        core_mass   = myjson_o.get_dict_values(['atmosphere','mass_core'])
        mantle_mass = myjson_o.get_dict_values(['atmosphere','mass_mantle'])
        planet_mass = core_mass + mantle_mass

        color_idx   = 5

        for nn, output_time in enumerate( output_times ):

            if nn == 0:
                lw = 2.0
                ls = "-"
            if nn == 1:
                lw = 2.0
                ls = "--"
            if nn == 3:
                lw = 2.0
                ls = ":"

            print(subdir, output_time)

            # Find data_time closest to output_time wanted
            time = find_nearest(data_times, output_time)

            label = vol_latex[subdir]#+", "+latex_float(time)+" yr"

            atm_file = data_dir+"/"+str(int(time))+"_atm.pkl"

            # Read pickle file
            atm_file_stream = open(atm_file,'rb')
            atm = pkl.load(atm_file_stream)
            atm_file_stream.close()

            # # read json
            # myjson_o = su.MyJSON( data_dir+"/"+'{}.json'.format(time) )

            # color = fig_o.get_color( nn )
            color   = vol_colors[subdir][color_idx]

            # # use melt fraction to determine mixed region
            # MIX     = myjson_o.get_mixed_phase_boolean_array( 'basic' )
          
            # Atmosphere T-Z
            z_profile = AtmosphericHeight(atm, planet_mass, r_planet) # m
            z_profile = z_profile*1e-3 # km

            # Connect interior and atmosphere lines
            atm.tmp[-1] = atm.ts

            l1, = ax0.plot( atm.tmp, z_profile, lw=lw, color=color, label=label, ls=ls)

            # Fill color and P_surf legends
            if nn == 0:
                legend_ax0_1_handles.append(l1)

            if subdir == sub_dirs[0]:
                l2, = ax0.plot( 0, 0, lw=lw, color=qgray, label=latex_float(time)+" yr", ls=ls)
                legend_ax0_2_handles.append(l2)

            # print(time,atm.tmp, atm.p)

            # # Atmosphere T-P
            # ax1.semilogy( atm.tmp, atm.p/1e5, '-', color=color, label=label, lw=1.5)

            # # Spectral flux density per wavenumber
            # ax2.plot( atm.band_centres, atm.net_spectral_flux[:,0]/atm.band_widths, '-', color=color, label=label, lw=1.5)
            # ymax_sp_flux = np.max( [ ymax_sp_flux, np.max(atm.net_spectral_flux[:,0]/atm.band_widths)] )
            # # Blackbody curves
            # ax2.plot(atm.band_centres,surf_Planck_nu(atm)/atm.band_widths, color=color, ls='--', lw=1.0, alpha=0.5, label=str(round(atm.ts))+' K blackbody')


            ## INTERIOR

            myjson_o = su.MyJSON( data_dir+"/"+'{}.json'.format(time) )

            # T-P mantle
            mantle_prs  = myjson_o.get_dict_values(['data','pressure_b'])*1.0E-9
            mantle_tmp  = myjson_o.get_dict_values(['data','temp_b'])

            ax1.plot( mantle_tmp, mantle_prs, ls=ls, lw=lw, color=color )

            # print(np.max(atm.ts), np.min(mantle_tmp))


            # # Plot mixed and pure phases with dash and dash-dotted lines
            # use melt fraction to determine mixed region
            # MIX     = myjson_o.get_mixed_phase_boolean_array( 'basic' ) # basic_internal
            # MIX_s   = myjson_o.get_mixed_phase_boolean_array( 'staggered' )
            # ax1.plot( mantle_tmp, xx_pres, '--', lw=lw, color=color )
            # handle, = ax1.plot( mantle_tmp*MIX, xx_pres*MIX, '-', lw=lw, color=color, label=label )


            # Reset y-axis boundaries
            if np.min(atm.p) > 0:
                ymax_atm_pressure = np.max([ymax_atm_pressure, np.max(atm.p/1e5)])
                ymin_atm_pressure = np.min([ymin_atm_pressure, np.min(atm.p/1e5)])
                ymax_atm_z = np.max([ymax_atm_z, np.max(z_profile)])
                ymin_atm_z = np.min([ymin_atm_z, np.min(z_profile)])


            # Reset x-axis boundaries
            xmin = np.min([ xmin, np.min(atm.tmp) ])
            xmax = np.max([ xmax, np.max(mantle_tmp) ])

            # color_idx   += 3

    # Print liquidus and solidus
    xx_pres = myjson_o.get_dict_values(['data','pressure_b'])*1.0E-9
    yy_liqt = myjson_o.get_dict_values(['data','liquidus_temp_b'])
    yy_solt = myjson_o.get_dict_values(['data','solidus_temp_b'])
    ax1.fill_betweenx( xx_pres, yy_liqt, yy_solt, facecolor=qmagenta_light, alpha=0.3, linewidth=0 )
    # ax1.fill_betweenx( mantle_tmp, yy_liq, yy_sol, facecolor='grey', alpha=0.35, linewidth=0 )
    # yy_liqt = myjson_o.get_dict_values_internal(['data','liquidus_temp_b'])
    # yy_solt = myjson_o.get_dict_values_internal(['data','solidus_temp_b'])
    # ax1.fill_between( xx_pres, yy_liqt, yy_solt, facecolor='grey', alpha=0.35, linewidth=0 )

    # ax1.fill_between(mantle_tmp, 1000, y2)

    ### Figure settings
    
    # xmin    = 0
    xticks  = [xmin, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, xmax]
    title_xcoord = -0.09
    title_ycoord = 0.5

    #####  T-Z
    # fig_o.set_myaxes( ax0, xlabel='$T$ (K)', ylabel='$z_\mathrm{atm}$\n(km)', xmin=xmin, xmax=xmax, ymin=0, ymax=ymax_atm_z, xticks=xticks )
    # ax0.set_xlabel("Temperature, $T$ (K)")
    ax0.set_ylabel("Atmosphere height, $z_\mathrm{atm}$ (km)", fontsize=fs_label)
    # ax0.set_yticks([ymin_atm_pressure, 1e-2, 1e-1, 1e0, 1e1, ymax_atm_pressure])
    ax0.yaxis.set_label_coords(title_xcoord,title_ycoord)
    # ax0.tick_params(direction='in')
    # ax0.set_xticklabels([])
    # ax0.xaxis.tick_top()
    # ax0.tick_params(direction='in')
    ax0.set_xlim( xmin, xmax )
    ax0.set_xticks(xticks)
    ax0.set_xticklabels([])
    ax0.set_yscale("symlog", linthreshy=50)
    ax0.set_ylim( 0, ymax_atm_z )
    ax0.set_yticks( [ 10, 20, 30, 60, 100, 300, 1000, 3000, ymax_atm_z] )
    ax0.set_yticklabels( [ "10", "20", "30", "60", "100", "300", "1000", "3000", str(int(ymax_atm_z))] )
    

    # #####  T-P
    # # fig_o.set_myaxes( ax1, xlabel='$T$ (K)', ylabel='$P_\mathrm{atm}$\n(bar)', xmin=xmin, xmax=xmax, xticks=xticks )
    # ax1.set_xlabel("Temperature, $T$ (K)")
    # ax1.set_ylabel("Atmosphere pressure, $P$ (bar)")
    # ax1.set_ylim( top=ymax_atm_pressure, bottom=ymin_atm_pressure )
    # ax1.set_ylim( top=ymax_atm_pressure, bottom=ymin_atm_pressure )
    # # ax1.set_yticks([ymin_atm_pressure, 1e-2, 1e-1, 1e0, 1e1, ymax_atm_pressure])
    # # ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax1.get_yaxis().get_major_formatter().labelOnlyBase = False
    # ax1.yaxis.set_label_coords(title_xcoord,title_ycoord)
    # ax1.invert_yaxis()

    ax1.set_ylim([0, int(mantle_prs[-1])])
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.set_xlim( xmin, xmax )
    ax1.set_xticks(xticks)
    # ax2.yaxis.set_label_coords(-0.25,0.5)
    ax1.set_xlabel( r'Temperature, $T$ (K)', fontsize=fs_label )
    ax1.invert_yaxis()

    ax1b = ax1.twinx()
    ax1b.plot( mantle_tmp, xx_depth, alpha=0.0)
    ax1b.set_xlim( right=xmax, left=xmin )
    ax1b.set_xticks(xticks)
    ax1b.set_ylim(top=xx_depth[-1], bottom=xx_depth[0])
    # ax1b.set_yticks([0, int(0.2*xx_depth[-1]), int(0.4*xx_depth[-1]), int(0.6*xx_depth[-1]), int(0.8*xx_depth[-1]), int(xx_depth[-1])])
    ax1b.invert_yaxis()
    ax1b.yaxis.tick_left()
    ax1.set_ylabel( 'Mantle depth, $d_\mathrm{mantle}$ (km)', fontsize=fs_label )
    ax1.yaxis.set_label_coords(title_xcoord,title_ycoord)
    # ax1.set_yscale("symlog", linthreshy=20)


    ax1.text(0.05, 0.05, 'Pure solid', color=qblue_dark, rotation=0, ha="left", va="bottom", fontsize=fs_label, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax1.text(0.6, 0.55, 'Mush', color=qmagenta_dark, rotation=0, ha="right", va="top", fontsize=fs_label, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax1.arrow(0.6, 0.55, 0.07, 0.03, head_width=0.0, head_length=0.0, fc=qmagenta_dark, ec=qmagenta_dark, transform=ax1.transAxes)
    ax1.text(0.95, 0.94, 'Pure melt', color=qred_dark, rotation=0, ha="right", va="top", fontsize=fs_label, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))


    # # title = '(b) Melt fraction'
    # # xticks = [0,0.2,0.4,0.6,0.8,1.0]
    # # fig_o.set_myaxes( ax1, title=title, xlabel='$\phi$', xticks=xticks, ymax=ymax, yticks=yticks )
    # # ax1.yaxis.set_label_coords(-0.075,0.475)
    # ax3.set_xlim([0, 1])
    # ax3.set_yticks([0,20,40,60,80,100,120,int(xx_pres_s[-1])])
    # ax3.set_yticklabels([])
    # ax3.invert_yaxis()

    # #####  Volatile mixing ratios
    # ax2.yaxis.set_label_position("right")
    # ax2.yaxis.tick_right()
    # ax2.set_xlim( left=0, right=1 )
    # ax2.set_ylabel( "$z_\mathrm{atm}$\n(km)", rotation=0)
    # ax2.yaxis.set_label_coords(1.10,0.5)
    # ax2.set_xlabel( '$X_{H_2O}$ (mol$_i$/mol)')

    # #####  Spectral OLR
    # fig_o.set_myaxes( ax2, xlabel='Wavenumber (cm$^{-1}$)', ylabel='Spectral flux' )
    # ax2.set_ylabel( 'Spectral flux density (W m$^{-2}$ cm$^{-1}$)', rotation=90)
    # # ax2.yaxis.set_label_position("right")
    # # ax2.yaxis.tick_right()
    # ax2.set_xlim( left=0, right=20000 )
    # ax2.set_ylim( bottom=0, top=ymax_sp_flux )
    # ax2.yaxis.set_label_coords(title_xcoord,title_ycoord)

    try:
        sns.set_style("ticks")
        sns.despine()
    except:
        print("No seaborn.")

    # Legend(s)
    legend_ax0_1 = ax0.legend(handles=legend_ax0_1_handles, loc=1, ncol=1, fontsize=fs_legend, framealpha=0.3, title="Volatiles")
    ax0.add_artist(legend_ax0_1)
    legend_ax0_2 = ax0.legend(handles=legend_ax0_2_handles, loc=4, ncol=1, fontsize=fs_legend, framealpha=0.3, title="Times")
    ax0.add_artist(legend_ax0_2)
    # ax0.legend( fancybox=True, framealpha=0.5, ncol=1, fontsize=fs_legend)
    # ax2.legend( fontsize=8, fancybox=True, framealpha=0.5 )

    fig_o.savefig(1)
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

    sub_dirs    = [ "H2", "CH4", "H2O", "CO2", "N2", "CO", "O2" ] #, "CO", "O2"

    # Times at which to plot
    output_times = [ 1e+2, 1e+7 ]

    output_dir  = "/Users/tim/runs/coupler_tests/set_260bar"
    print("Host directory:", output_dir)

    # Plot fixed set from above
    plot_atmosphere( output_dir=output_dir, sub_dirs=sub_dirs, output_times=output_times )

#====================================================================

if __name__ == "__main__":

    # Import utils- and plot-specific modules
    from modules_utils import *
    from modules_plot import *
    import utils_coupler as cu
    import utils_spider as su

    main()
