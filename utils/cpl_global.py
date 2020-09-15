#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_plot import *

#====================================================================
def plot_global( output_dir ):

    # Plot settings
    lw         = 1.5
    fscale     = 1.1
    fsize      = 18
    fs_title   = 18
    fs_legend  = 11
    width      = 12.00 #* 3.0/2.0
    height     = 8.0 #/ 2.0
    # Subplot titles
    title_fs   = 12
    title_xy   = (0.07, 0.02)
    title_x    = title_xy[0]
    title_y    = title_xy[1]
    title_xycoords = 'axes fraction'
    title_ha   = "left"
    title_va   = "bottom"
    title_font = 'Arial'
    txt_alpha  = 0.5
    txt_pad    = 0.1
    label_fs   = 11

    fig_o = su.FigureData( 3, 2, width, height, output_dir+'/plot_global', units='yr' )
    fig_o.fig.subplots_adjust(wspace=0.05,hspace=0.1)

    ax0 = fig_o.ax[0][0]
    ax1 = fig_o.ax[1][0]
    ax2 = fig_o.ax[2][0]
    ax3 = fig_o.ax[0][1]
    ax4 = fig_o.ax[1][1]
    ax5 = fig_o.ax[2][1]

    title_xNumbering = 0.02
    title_yNumbering = 0.05
    fsplus = 5
    ax0.text(title_xNumbering, title_yNumbering, 'A', color="k", rotation=0, ha="left", va="bottom", fontsize=label_fs+fsplus, transform=ax0.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax1.text(title_xNumbering, title_yNumbering, 'B', color="k", rotation=0, ha="left", va="bottom", fontsize=label_fs+fsplus, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax2.text(title_xNumbering, title_yNumbering, 'C', color="k", rotation=0, ha="left", va="bottom", fontsize=label_fs+fsplus, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax3.text(title_xNumbering, title_yNumbering, 'D', color="k", rotation=0, ha="left", va="bottom", fontsize=label_fs+fsplus, transform=ax3.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax4.text(title_xNumbering, title_yNumbering, 'E', color="k", rotation=0, ha="left", va="bottom", fontsize=label_fs+fsplus, transform=ax4.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax5.text(title_xNumbering, title_yNumbering, 'F', color="k", rotation=0, ha="left", va="bottom", fontsize=label_fs+fsplus, transform=ax5.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

    # runtime_helpfile = pd.read_csv(output_dir+"runtime_helpfile.csv", sep=" ")
    # print(runtime_helpfile)

    fig_o.time = su.get_all_output_times(output_dir)
    print("output_dir", output_dir)
    print("Times", fig_o.time)

    # Read in runtime helpfile and separate in atmosphere and interior params
    df = pd.read_csv(output_dir+"/runtime_helpfile.csv", sep=" ")
    df_int = df.loc[df['Input']=='Interior']
    df_atm = df.loc[df['Input']=='Atmosphere']

    ########## Global properties
    keys_t = ( ('atmosphere','mass_liquid'),
               ('atmosphere','mass_solid'),
               ('atmosphere','mass_mantle'),
               ('atmosphere','mass_core'),
               ('atmosphere','temperature_surface'),
               ('atmosphere','emissivity'),
               ('rheological_front_phi','phi_global'),
               ('atmosphere','Fatm'),
               ('atmosphere','pressure_surface'),
               ('rheological_front_dynamic','depth')
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
    rheol_front         = data_a[9,:]


    xlabel = r'Time, $t$ (yr)'
    xlim = (1e1,1e7)

    red = (0.5,0.1,0.1)
    blue = (0.1,0.1,0.5)
    black = 'black'

    xcoord_l = -0.10
    ycoord_l = 0.5
    xcoord_r = 1.09
    ycoord_r = 0.5

    rolling_mean = 0
    nsteps       = 4

    # Replace NaNs
    for idx, val in enumerate(T_surf):
        # print(idx, val)
        if np.isnan(val):
            json_file_time = su.MyJSON( output_dir+'/{}.json'.format(fig_o.time[idx]) )
            int_tmp   = json_file_time.get_dict_values(['data','temp_b'])
            print("T_surf:", idx, val, "-->", round(int_tmp[0],3), "K")
            T_surf[idx] = int_tmp[0]
            

    ##########
    # figure a
    ##########
    title = r'Heat flux to space'  
    # Use helpfile information
    time = df_atm["Time"].tolist()
    Fatm1 = df_atm["Heat_flux"].tolist()
    if rolling_mean == 1:
        ax0.loglog( time[:nsteps+4], Fatm[:nsteps+4], qgray_dark, lw=lw, alpha=1.0 )
        
        Fatm_rolling = np.convolve(Fatm, np.ones((nsteps,))/nsteps, mode='valid')
        Time_rolling = np.convolve(time, np.ones((nsteps,))/nsteps, mode='valid')
        ax0.loglog( Time_rolling, Fatm_rolling, qgray_dark, lw=lw )
    else:
        ax0.loglog( fig_o.time, Fatm, "red", lw=lw, alpha=1.0 )
        ax0.loglog( time, Fatm1, qgray_dark, lw=lw, alpha=1.0 )
      
    # fig_o.set_myaxes(ax0)
    ax0.set_ylabel(r'$F_\mathrm{atm}^{\uparrow}$ (W m$^{-2}$)', fontsize=label_fs)
    ax0.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax0.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax0.set_xlim( *xlim )
    # ax0.set_ylim(top=np.amax(df_atm["Heat_flux"])*1.1, bottom=np.amin(df_atm["Heat_flux"])*0.9)
    ax0.set_yscale('symlog')
    ax0.set_xticklabels([])
    ax0.yaxis.set_label_coords(xcoord_l,ycoord_l)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, loc='upper right', ncol=1, frameon=0, fontsize=fs_legend)
    ax0.set_title(title, fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va, bbox=dict(fc='white', ec="white", alpha=txt_alpha, pad=txt_pad))

    # print(T_surf)
    ##########
    # figure b
    ##########
    title = r'Surface temperature'
    # h1, = ax1.semilogx( fig_o.time, T_surf, ls="-", lw=lw, color=qgray_dark, label=r'Surface temp, $T_s$' )
    h1, = ax1.semilogx(df_atm["Time"], df_atm["T_surf"], ls="-", lw=lw, color=qgray_dark, label=r'Surface temp, $T_s$') # , color="blue"
    h2, = ax1.semilogx(df_int["Time"], df_int["T_surf"], color="red", label="Interior")
    if np.max(fig_o.time) >= 1e3: 
        ymin = np.min(df_atm["T_surf"])*0.9
        ymax = np.max(df_atm["T_surf"])*1.1
    else: 
        ymin = 200
        ymax = 3000
    yticks = [ymin, ymin+0.2*(ymax-ymin), ymin+0.4*(ymax-ymin), ymin+0.6*(ymax-ymin), ymin+0.8*(ymax-ymin), ymax]
    # fig_o.set_myaxes( ax1, title=title, yticks=yticks)
    ax1.set_ylabel(r'$T_\mathrm{s}$ (K)', fontsize=label_fs)
    ax1.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax1.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.set_xlim( *xlim )
    ax1.set_xticklabels([])
    ax1.yaxis.set_label_coords(xcoord_l,ycoord_l)
    ax1.set_ylim(ymin, ymax)
    ax1.set_title(title, fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va, bbox=dict(fc='white', ec="white", alpha=txt_alpha, pad=txt_pad))

    ##########
    # figure c
    ##########

    # Plot rheological front depth
    ax2.semilogx( fig_o.time, rheol_front/np.max(rheol_front), ls="-", lw=lw, color=qgray_light, label=r'Rheol. front, $d_{\mathrm{front}}$')
    
    # Mante melt + solid fraction
    ax2.semilogx( fig_o.time, phi_global, color=qgray_dark, linestyle=':', lw=lw, label=r'Melt, $\phi_{\mathrm{mantle}}$')
    ax2.semilogx( fig_o.time, mass_solid/(mass_liquid+mass_solid), color=qgray_dark, linestyle='--', lw=lw, label=r'Solid, $1-\phi_{\mathrm{mantle}}$')

    ax2.set_xlim( *xlim )
    ax2.set_xlabel(xlabel, fontsize=label_fs)
    ax2.set_ylabel(r'Mantle fraction', fontsize=label_fs)
    ax2.yaxis.set_label_coords(xcoord_l,ycoord_l)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, ncol=1, loc=6, frameon=1, fancybox=True, framealpha=0.9, fontsize=fs_legend-1)

    title = r'Mantle evolution'
    ax2.set_title(title, fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va, bbox=dict(fc='white', ec="white", alpha=txt_alpha, pad=txt_pad))

    ### Plot axes setup
    ##########
    # figure d
    ##########
    title_ax3 = r'Surface volatile partial pressure'
    # Total pressure
    ax3.semilogx( fig_o.time, P_surf, color=qgray_dark, linestyle='-', lw=lw, label=r'Total')
    ##########
    # figure e
    ##########
    title_ax4 = r'Atmosphere volatile mass fraction'
    ##########
    # figure f
    ##########
    title_ax5 = r'Interior volatile mass fraction'

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
    fig_o.set_myaxes( ax3)
    ax3.set_ylabel('$p^{\mathrm{i}}$ (bar)', fontsize=label_fs)
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
    ax3.legend(handles, labels, ncol=2, loc=2, frameon=1, fancybox=True, framealpha=0.9, fontsize=fs_legend) # , loc='center left'
    ax3.set_title(title_ax3, fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va, bbox=dict(fc='white', ec="white", alpha=txt_alpha, pad=txt_pad))
    ##########
    # figure e
    ##########
    # # Check atmospheric comparisons for mass conservation
    # runtime_helpfile = pd.read_csv(output_dir+"/"+"runtime_helpfile.csv", delim_whitespace=True)
    # ax4.semilogx( runtime_helpfile["Time"], runtime_helpfile["M_atm"]/runtime_helpfile.iloc[0]["M_atm"], lw=lw, color=vol_colors["black_2"], linestyle='-')
    # ax4.semilogx( runtime_helpfile.loc[runtime_helpfile['Input'] == "Interior"]["Time"], runtime_helpfile.loc[runtime_helpfile['Input'] == "Interior"]["H_mol_total"]/runtime_helpfile.iloc[0]["H_mol_total"], lw=lw, color=vol_colors["black_3"], linestyle='--')
    # ax4.semilogx( runtime_helpfile.loc[runtime_helpfile['Input'] == "Atmosphere"]["Time"], runtime_helpfile.loc[runtime_helpfile['Input'] == "Atmosphere"]["C/H_atm"]/runtime_helpfile.iloc[0]["C/H_atm"], lw=lw, color=vol_colors["black_1"], linestyle=':')
    # print(runtime_helpfile[["Time", "Input", "M_atm", "M_atm_kgmol", "H_mol_atm", "H_mol_solid", "H_mol_liquid", "H_mol_total", "O_mol_total", "O/H_atm"]])

    fig_o.set_myaxes( ax4)
    ax4.set_ylabel(r'$X_{\mathrm{atm}}^{\mathrm{i}}/X_{\mathrm{tot}}^{\mathrm{i}}$', fontsize=label_fs)
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
    ax4.set_title(title_ax4, fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va, bbox=dict(fc='white', ec="white", alpha=txt_alpha, pad=txt_pad))


    ##########
    # figure f
    ##########
    fig_o.set_myaxes( ax5, title=title_ax5)
    ax5.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20) )
    ax5.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=20))
    ax5.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax5.set_xlim( *xlim )
    # ax5.set_ylim( 0, 1 )
    ax5.yaxis.tick_right()
    ax5.yaxis.set_label_coords(xcoord_r,ycoord_r)
    ax5.yaxis.set_label_position("right")
    handles, labels = ax5.get_legend_handles_labels()
    ax5.set_xlabel(xlabel, fontsize=label_fs)
    ax5.set_ylabel(r'$X_{\mathrm{mantle}}^{\mathrm{i}}/X_{\mathrm{tot}}^{\mathrm{i}}$', fontsize=label_fs)
    ax5.set_title(title_ax5, fontname=title_font, fontsize=title_fs, x=title_x, y=title_y, ha=title_ha, va=title_va, bbox=dict(fc='white', ec="white", alpha=txt_alpha, pad=txt_pad))


    fig_o.savefig(6)
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
    else:
        output_dir = os.getcwd()

    print("Output directory:", output_dir)

    # Define which times are plotted
    if args.times:
        plot_list = [ int(time) for time in args.times.split(',') ]
    else:
        output_list = su.get_all_output_times(output_dir)

        if len(output_list) <= 8:
            plot_list = output_list
        else:
            plot_list = [ output_list[0], output_list[int(round(len(output_list)*(2./100.)))], output_list[int(round(len(output_list)*(15./100.)))], output_list[int(round(len(output_list)*(22./100.)))], output_list[int(round(len(output_list)*(33./100.)))], output_list[int(round(len(output_list)*(50./100.)))], output_list[int(round(len(output_list)*(66./100.)))], output_list[-1] ]
    print("Snapshots:", plot_list)

    plot_global(output_dir=output_dir)

#====================================================================

if __name__ == "__main__":

    # Import utils- and plot-specific modules
    from modules_utils import *
    from modules_plot import *
    import utils_coupler as cu
    import utils_spider as su

    main()
