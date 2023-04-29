#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_plot import *
from utils.modules_plot import MyFuncFormatter

import matplotlib.gridspec as gridspec

def find_nearest(array, value):
    array   = np.asarray(array)
    idx     = (np.abs(array - value)).argmin()
    return array[idx]

#====================================================================
def plot_atmosphere( output_dir, sub_dirs, output_times ):

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

    gs = fig.add_gridspec(nrows=2, ncols=3, wspace=0.03, hspace=0.2, left=0.05, right=0.98, top=0.98, bottom=0.08)


    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])

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
    legend_ax0_1_handles = []
    legend_ax0_2_handles = []

    # # Load runtime helpfile
    # runtime_helpfile = pd.read_csv(output_dir+"runtime_helpfile.csv")

    for subdir in sub_dirs:

        if subdir in [ "H2" ]: 
        # RF: 86 28959.090909091097 1045637 0.213690957428444
            output_times = [ 1045637 ]
        if subdir in [ "H2O" ]:
        # RF: 266 28959.090909091097 11422 0.267570613371205
            output_times = [ 11422 ]
        if subdir in [ "CH4" ]:
        # RF: 55 28959.090909091097 11914 0.2676858221410769
            output_times = [ 11914 ]
        if subdir in [ "CO2" ]:
        # RF: 55 28959.090909091097 7579 0.2695751505696307
            output_times = [ 7579 ]
        if subdir in [ "N2" ]:
        # RF: 47 28959.090909091097 857 0.2977654364427622
            output_times = [ 857 ]
        if subdir in [ "O2" ]:
        # RF: 47 28959.090909091097 881 0.2997064798750272
            output_times = [ 881 ]
        if subdir in [ "CO" ]:
        # RF: 47 28959.090909091097 883 0.2987908045637139
            output_times = [ 883 ]
        if subdir in [ "N2_reduced" ]:
        # RF: 561 28959.090909091097 716 0.3009983998102829
            output_times = [ 716 ]

        data_dir = output_dir+"/"+subdir

        # fig_o.set_colors(cmap=vol_colors[subdir]) # "magma_r"

        # Get data times for subdir
        data_times = su.get_all_output_pkl_times(data_dir)

        # print(data_times)

        # Find solidus and liquidus lines
        # Use first timestep since liquidus and solidus are time-independent
        time        = data_times[0]                    
        myjson_o    = MyJSON( data_dir+'/{}.json'.format(time) )
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
                lw = 1.5
                ls = "-"
            if nn == 1:
                lw = 1.5
                ls = "--"
            if nn == 3:
                lw = 1.5
                ls = ":"

            print(subdir, output_time)

            # Find data_time closest to output_time wanted
            time = find_nearest(data_times, output_time)

            label = vol_latex[subdir]+", "+latex_float(time)+" yr"

            atm_file = data_dir+"/"+str(int(time))+"_atm.pkl"

            # Read pickle file
            atm_file_stream = open(atm_file,'rb')
            atm = pkl.load(atm_file_stream)
            atm_file_stream.close()

            # color = fig_o.get_color( nn )
            color   = vol_colors[subdir][color_idx]
          
            # Atmosphere T-Z
            z_profile = AtmosphericHeight(atm, planet_mass, r_planet) # m
            z_profile = z_profile*1e-3 # km

            # Connect interior and atmosphere lines
            atm.tmp[-1] = atm.ts

            # Spectral flux density per wavenumber
            l1, = ax0.plot( atm.band_centres, atm.net_spectral_flux[:,0]/atm.band_widths, '-', color=color, label=label, lw=1.5)
            ymax_sp_flux = np.max( [ ymax_sp_flux, np.max(atm.net_spectral_flux[:,0]/atm.band_widths)] )
            # Blackbody curves
            ax0.plot(atm.band_centres,surf_Planck_nu(atm)/atm.band_widths, color=color, ls='--', lw=1.0, alpha=0.5, label=str(round(atm.ts))+' K blackbody')

            # Fill color and P_surf legends
            # if nn == 0:
            legend_ax0_1_handles.append(l1)

            if subdir == sub_dirs[0]:
                l2, = ax0.plot( 0, 0, lw=lw, color=qgray, label=latex_float(time)+" yr", ls=ls)
                legend_ax0_2_handles.append(l2)

            ### INTERIOR

            myjson_o = MyJSON( data_dir+"/"+'{}.json'.format(time) )

            # Melt fraction
            mantle_prs = myjson_o.get_dict_values(['data','pressure_b'])*1.0E-9
            mantle_phi = myjson_o.get_dict_values(['data','phi_b'])
            ax1.plot( mantle_phi, mantle_prs, ls=ls, lw=lw, color=color )

            # Temperature
            mantle_tmp = myjson_o.get_dict_values(['data','temp_b'])
            ax2.plot( mantle_tmp, mantle_prs, ls=ls, lw=lw, color=color )

            # Viscosity
            visc_const  = 1 # this is used for the arcsinh scaling
            visc_fmt    = MyFuncFormatter( visc_const )
            mantle_visc = myjson_o.get_dict_values(['data','visc_b'], visc_fmt)
            ax3.plot( mantle_visc, mantle_prs, ls=ls, lw=lw, color=color )
            
            # Reset x-axis boundaries
            xmin = np.min([ xmin, np.min(mantle_phi) ])
            xmax = np.max([ xmax, np.max(mantle_phi) ])

            # color_idx   += 3

    # Print liquidus and solidus
    xx_pres = myjson_o.get_dict_values(['data','pressure_b'])*1.0E-9
    yy_liqt = myjson_o.get_dict_values(['data','liquidus_temp_b'])
    yy_solt = myjson_o.get_dict_values(['data','solidus_temp_b'])
    ax2.fill_betweenx( xx_pres, yy_liqt, yy_solt, facecolor=qmagenta_light, alpha=0.3, linewidth=0 )
    # ax1.fill_betweenx( mantle_tmp, yy_liq, yy_sol, facecolor='grey', alpha=0.35, linewidth=0 )
    # yy_liqt = myjson_o.get_dict_values_internal(['data','liquidus_temp_b'])
    # yy_solt = myjson_o.get_dict_values_internal(['data','solidus_temp_b'])
    # ax1.fill_between( xx_pres, yy_liqt, yy_solt, facecolor='grey', alpha=0.35, linewidth=0 )

    # ax1.fill_between(mantle_tmp, 1000, y2)

    ### Figure settings
    
    # xmin    = 0
    xticks  = np.linspace(xmin, xmax, 10)
    xticks  = [ round(x,2) for x in xticks ]
    title_xcoord = -0.09
    title_ycoord = 0.5

    #####  T-Z
    # fig_o.set_myaxes( ax0, xlabel='$T$ (K)', ylabel='$z_\mathrm{atm}$\n(km)', xmin=xmin, xmax=xmax, ymin=0, ymax=ymax_atm_z, xticks=xticks )
    # ax0.set_xlabel("Temperature, $T$ (K)")
    ax0.set_ylabel("Atmosphere height, $z_\mathrm{atm}$ (km)", fontsize=fs_label)
    # ax0.set_yticks([ymin_atm_pressure, 1e-2, 1e-1, 1e0, 1e1, ymax_atm_pressure])
    # ax0.yaxis.set_label_coords(title_xcoord,title_ycoord)
    ax0.set_xlabel( r'Wavenumber (cm$^{-1}$)', fontsize=fs_label )
    ax0.set_ylabel( r'Spectral flux density (W m$^{-2}$ cm$^{-1}$)', fontsize=fs_label)
    ax0.set_xlim( left=0, right=20000 )
    ax0.set_ylim( bottom=0, top=ymax_sp_flux ) # bottom=0, 
    ax0.set_yscale("symlog", linthreshy=1)
    
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
    # ax1.set_yticks([])
    # ax1.set_yticklabels([])
    ax1.set_xlim( xmin, xmax )
    ax1.set_xticks(xticks)
    # ax2.yaxis.set_label_coords(-0.25,0.5)
    ax1.set_xlabel( r'Mantle melt fraction, $\phi$ (wt)', fontsize=fs_label )
    ax1.invert_yaxis()
    ax1.set_ylabel( 'Mantle pressure, $P_\mathrm{mantle}$ (GPa)', fontsize=fs_label )

    # ax1b = ax1.twinx()
    # ax1b.plot( mantle_phi, xx_depth, alpha=0.0)
    # ax1b.set_xlim( right=xmax, left=xmin )
    # ax1b.set_xticks(xticks)
    # ax1b.set_ylim(top=xx_depth[-1], bottom=xx_depth[0])
    # # ax1b.set_yticks([0, int(0.2*xx_depth[-1]), int(0.4*xx_depth[-1]), int(0.6*xx_depth[-1]), int(0.8*xx_depth[-1]), int(xx_depth[-1])])
    # ax1b.invert_yaxis()
    # ax1b.yaxis.tick_left()
    # ax1.set_ylabel( 'Mantle depth, $d_\mathrm{mantle}$ (km)', fontsize=fs_label )
    # ax1.yaxis.set_label_coords(title_xcoord,title_ycoord)
    # ax1.set_yscale("symlog", linthreshy=10)


    ax2.set_ylim([0, int(mantle_prs[-1])])
    # ax2.set_yticks([])
    ax2.set_yticklabels([])
    # ax2.set_xlim( xmin, xmax )
    # ax2.set_xticks(xticks)
    ax2.set_xlabel( r'Mantle temperature, $T$ (K)', fontsize=fs_label )
    ax2.invert_yaxis()
    # ax2b = ax1.twinx()
    # ax2b.plot( mantle_tmp, xx_depth, alpha=0.0)
    # ax2b.set_xlim( right=xmax, left=xmin )
    # ax2b.set_xticks(xticks)
    # ax2b.set_ylim(top=xx_depth[-1], bottom=xx_depth[0])
    # yticks = [ 0, 250, 500, 1000, 1500, 2000, 2500, xx_depth[-1] ]
    # ax2b.set_yticks( yticks )
    # ax2b.set_yticklabels( [ str(int(i)) for i in yticks ] )
    # ax2b.invert_yaxis()
    # ax2b.yaxis.tick_left()
    # ax2.set_ylabel( 'Mantle depth, $d_\mathrm{mantle}$ (km)', fontsize=fs_label )
    # ax2.yaxis.set_label_coords(title_xcoord,title_ycoord)
    # # ax1.set_yscale("symlog", linthreshy=20)

    ax3.set_ylim([0, int(mantle_prs[-1])])
    ax3.set_yticklabels([])
    ax3.set_xlabel( r'Mantle viscosity, $\eta$ (Pa s)', fontsize=fs_label )
    ax3.invert_yaxis()


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

    # Legend(s)
    legend_ax0_1 = ax0.legend(handles=legend_ax0_1_handles, loc=1, ncol=1, fontsize=fs_legend, framealpha=0.3, title="Volatiles")
    ax0.add_artist(legend_ax0_1)
    # legend_ax0_2 = ax0.legend(handles=legend_ax0_2_handles, loc=4, ncol=1, fontsize=fs_legend, framealpha=0.3, title="Times")
    # ax0.add_artist(legend_ax0_2)

    # ax0.legend( fancybox=True, framealpha=0.5, ncol=1, fontsize=fs_legend)
    # ax2.legend( fontsize=8, fancybox=True, framealpha=0.5 )
    plt.savefig(output_dir+'/compare_sfd+melt.pdf', tight_layout=True)

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

    sub_dirs    = [ "H2", "CH4", "H2O", "CO2", "N2", "CO", "O2" ] #, "CO", "O2"

    # Times at which to plot
    output_times = [ 1e+6 ]

    output_dir  = "/Users/tim/runs/coupler_tests/set2_260bar"
    print("Host directory:", output_dir)

    # Plot fixed set from above
    plot_atmosphere( output_dir=output_dir, sub_dirs=sub_dirs, output_times=output_times )

#====================================================================

if __name__ == "__main__":

    # Import utils- and plot-specific modules
    from utils.modules_utils import *
    from utils.modules_plot import *
    import utils_coupler as cu
    import utils_spider as su

    main()
