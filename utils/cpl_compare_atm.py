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

    gs = fig.add_gridspec(nrows=2, ncols=4, wspace=0.2, hspace=0.2, left=0.055, right=0.98, top=0.98, bottom=0.08)


    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    ax7 = fig.add_subplot(gs[1, 3])


    fs_legend   = 11
    fs_label    = 11

    # Initiate legends
    legend_ax7_handles = []


    output_times = [ 1e+3, 1e+4, 1e+6 ]

    for subdir in sub_dirs:

        color_idx   = 2

        if subdir == "H2":
            ax = ax0
        if subdir == "H2O":
            ax = ax1
        if subdir == "CO2":
            ax = ax2
        if subdir == "CH4":
            ax = ax3
        if subdir == "O2":
            ax = ax4
        if subdir == "CO":
            ax = ax5
        if subdir == "N2":
            ax = ax6
        # if subdir == "N2_reduced":
        #     ax = ax6

        ymax_sp_flux = 0
        xmax_wavenumber = 0

        legend_ax_handles = []

        for nn, output_time in enumerate( output_times ):

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
            ls = "-"

            data_dir = output_dir+"/"+subdir

            # print(data_dir)

            # Find data_time closest to output_time wanted
            atm_data_times = su.get_all_output_pkl_times(data_dir)
            time, time_idx = find_nearest(atm_data_times, output_time)

            print(subdir, output_time)

            time_label = latex_float(output_time)


            atm_file = data_dir+"/"+str(int(time))+"_atm.pkl"

            # Read pickle file
            atm_file_stream = open(atm_file,'rb')
            atm = pkl.load(atm_file_stream)
            atm_file_stream.close()

            Ts_label = str(int(round(atm.ts)))

            # color = fig_o.get_color( nn )
            color   = vol_colors[subdir][color_idx]

            sp_flux = atm.net_spectral_flux[:,0]/atm.band_widths
          
            l1, = ax.plot( atm.band_centres, sp_flux, ls=ls, color=color, label=Ts_label, lw=lw)

            legend_ax_handles.append(l1)
            
            # Blackbody curves
            ax.plot(atm.band_centres,surf_Planck_nu(atm)/atm.band_widths, color=color, ls="--", lw=0.5, alpha=0.5, label=str(round(atm.ts))+' K blackbody')

            ymax_sp_flux = np.max( [ ymax_sp_flux, np.max(sp_flux)] )
            xmax_wavenumber = np.max( [ xmax_wavenumber, np.max(np.where(sp_flux > ymax_sp_flux/100., atm.band_centres, 0.)) ] )
            
            # Time legend    
            if ax == ax0:
                l2, = ax7.plot( 0, 0, ls=ls, color=vol_colors["greys"][color_idx], label=time_label, lw=lw)
                legend_ax7_handles.append(l2)
     
            color_idx   += 3

        ax.set_xlim( left=0, right=xmax_wavenumber )
        ax.set_ylim( bottom=0, top=ymax_sp_flux )
        # ax.set_yscale("symlog", linthreshy=ymax_sp_flux/100.)

        if ax == ax0:
            ax.set_ylabel( 'Spectral flux density (W m$^{-2}$ cm)', fontsize=fs_label)
            ax.yaxis.set_label_coords(-0.18, -0.12)

        if ax == ax5:
            ax.set_xlabel( 'Wavenumber (cm$^{-1}$)', fontsize=fs_label)

        # Individual legends
        ax.legend(handles=legend_ax_handles, loc=1, ncol=1, fontsize=fs_legend, framealpha=0.3, title="$T_\mathrm{surf}$ (K)")
        #vol_latex[subdir]+", "+
        ax.text(0.02, 0.985, vol_latex[subdir], color=vol_colors[subdir][5], rotation=0, ha="left", va="top", fontsize=fs_label+3, transform=ax.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

    sns.set_style("ticks")
    sns.despine()
    sns.despine(ax=ax7, top=True, right=True, left=True, bottom=True)
    ax7.set_yticklabels([])
    ax7.set_xticklabels([])

    ax0.text(0.32, 0.35, 'Blackbody', color=qgray_dark, rotation=0, ha="right", va="top", fontsize=fs_label, transform=ax0.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    ax0.arrow(0.32, 0.35, 0.06, 0.03, head_width=0.0, head_length=0.0, fc=qgray_dark, ec=qgray_dark, transform=ax0.transAxes, ls="--", lw=0.5)

    # # Legend(s)
    # legend_ax0 = ax0.legend(handles=legend_ax0_handles, loc=1, ncol=1, fontsize=fs_legend, framealpha=0.3, title=r"H$_\mathrm{2}$O"+", $t = $"+latex_float(output_time_print_a)+" yr")
    # ax0.add_artist(legend_ax0)
    
    # # for text in legend_ax0.get_texts():
    # #     text.set_color("red")

    legend_ax7 = ax7.legend(handles=legend_ax7_handles, loc="center", ncol=1, fontsize=fs_legend, framealpha=0.3, title=r"Time (yr)")
    ax7.add_artist(legend_ax7)

    # ax2.text(0.6, 0.28, 'Mush', color=qmagenta_light, rotation=0, ha="left", va="top", fontsize=fs_label, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    # ax2b.text(0.6, 0.28, 'Mush', color=qmagenta_light, rotation=0, ha="left", va="top", fontsize=fs_label, transform=ax2b.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    
    # ax0.text(0.02, 0.015, 'A1', color="k", rotation=0, ha="left", va="bottom", fontsize=fs_label+3, transform=ax0.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    # ax1.text(0.02, 0.015, 'A2', color="k", rotation=0, ha="left", va="bottom", fontsize=fs_label+3, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    # ax2.text(0.02, 0.015, 'A3', color="k", rotation=0, ha="left", va="bottom", fontsize=fs_label+3, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    # ax0b.text(0.02, 0.015, 'B1', color="k", rotation=0, ha="left", va="bottom", fontsize=fs_label+3, transform=ax0b.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    # ax1b.text(0.02, 0.015, 'B2', color="k", rotation=0, ha="left", va="bottom", fontsize=fs_label+3, transform=ax1b.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
    # ax2b.text(0.02, 0.015, 'B3', color="k", rotation=0, ha="left", va="bottom", fontsize=fs_label+3, transform=ax2b.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

    # ax0.legend( fancybox=True, framealpha=0.5, ncol=1, fontsize=fs_legend)
    # ax2.legend( fontsize=8, fancybox=True, framealpha=0.5 )
    plt.savefig(output_dir+'/compare_atm.pdf', tight_layout=True)

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

    vols    = [ "H2", "H2O", "CO2", "CH4" ]
    vols    = [ "H2", "H2O", "CO2", "CH4", "O2", "N2", "CO" ]

    output_dir  = "/Users/tim/runs/coupler_tests/set2_260bar"
    print("Host directory:", output_dir)

    # Plot fixed set from above
    plot_atmosphere( output_dir=output_dir, sub_dirs=vols )

#====================================================================

if __name__ == "__main__":

    # Import utils- and plot-specific modules
    from utils.modules_utils import *
    from utils.modules_plot import *
    import utils_coupler as cu
    import utils_spider as su

    main()
