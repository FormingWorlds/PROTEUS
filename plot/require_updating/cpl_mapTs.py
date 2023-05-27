#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_plot import *

#====================================================================
def plot_mapTs( output_dir, subdirs ):

    # article class text width is 4.7747 inches

    # width = 6.00 #* 3.0/2.0
    # height = 12.0
    # fig_o = su.FigureData( 3, 1, width, height, output_dir+'/plot_atmosphere', units='kyr' )
    # fig_o.fig.subplots_adjust(wspace=0.07,hspace=0.2)

    fig, ax1 = plt.subplots()

    Sfrac_list  = []
    CO2_list    = []
    H2O_list    = []
    Tsurf_list  = []

    for subdir in subdirs:

        # # if os.path.exists(subdir+"*.json"):

        # output_list = su.get_all_output_times(subdir)
        # if not list(output_list):
        #     print(subdir, "No outputs")
        # else:
        #     print(subdir, np.max(output_list))
        #     # last_timestep = output_list[-1]

        #     # print(subdir, last_timestep)


        if os.path.exists(subdir+"runtime_helpfile.csv"):

            output_list = su.get_all_output_times(subdir)
            last_timestep = output_list[-1]

            print(subdir, last_timestep, end=" | ")

            # Read in runtime helpfile and separate in atmosphere and interior params
            df = pd.read_csv(subdir+"runtime_helpfile.csv", sep=" ")
            df_int = df.loc[df['Input']=='Interior']
            df_atm = df.loc[df['Input']=='Atmosphere']

            # Read out the parameters
            Sfrac       = float(subdir[-20:-17])/100
            CO2_mass    = float(subdir[-12:-9])
            H2O_mass    = float(subdir[-4:-1])
            Tsurf       = float(round(df_atm["T_surf"].iloc[-1],2))
            
            print(Sfrac, CO2_mass, H2O_mass, round(df_atm["T_surf"].iloc[-1],2))

            # Add to list
            Sfrac_list.append(Sfrac)
            CO2_list.append(CO2_mass)
            H2O_list.append(H2O_mass)
            Tsurf_list.append(Tsurf)

        else:

            print(subdir, "No outputs")

    Sfrac_array = np.array(Sfrac_list)
    CO2_array   = np.array(CO2_list)
    H2O_array   = np.array(H2O_list)
    Tsurf_array = np.array(Tsurf_list)

    # https://matplotlib.org/stable/gallery/images_contours_and_fields/irregulardatagrid.html#sphx-glr-gallery-images-contours-and-fields-irregulardatagrid-py
    clevels = np.linspace(150,250,10)
    clevels = [ round(i) for i in clevels ]
    ax1.tricontour(Sfrac_array, CO2_array, Tsurf_array, levels=clevels,
                   extend='both', linewidths=0.5, colors='k')
    cntr2 = ax1.tricontourf(Sfrac_array, CO2_array, Tsurf_array, levels=clevels,
                   extend='both', cmap="plasma")

    fig.colorbar(cntr2, ax=ax1, label=r"Surface temperature $T_\mathrm{surf}$ (K)")
    ax1.plot(Sfrac_array, CO2_array, 'ko', ms=1)
    # ax1.set(xlim=(-2, 2), ylim=(-2, 2))
    # ax1.set_title('tricontour (%d points)' % npts)
    # plt.subplots_adjust(hspace=0.5)

    ax1.set_xlabel(r"Instellation $S_\mathrm{eff}$ ($S_\mathrm{Earth}$)")
    ax1.set_ylabel(r"CO$_2$ abundance (ppm wt)")


    plt.savefig("cpl_mapTs.pdf", bbox_inches='tight')


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

    # Find all subdirectories
    subdirs = natsorted(glob.glob(f'{output_dir}/*/'))
    print("Subdirectories:", subdirs)

    # # Plot only last snapshot
    # plot_current_mixing_ratio( output_dir=output_dir, times=plot_list[-1], vulcan_setting=0 )

    # Plot fixed set from above
    plot_mapTs( output_dir=output_dir, subdirs=subdirs )

#====================================================================

if __name__ == "__main__":

    # Import utils- and plot-specific modules
    from utils.modules_ext import *
    from utils.modules_plot import *
    import utils_coupler as cu
    import utils_spider as su

    main()
