#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_ext import *
from utils.spider import *
from utils.plot import *


#====================================================================
def plot_stacked( output_dir, times ):

    # article class text width is 4.7747 inches
    # http://tex.stackexchange.com/questions/39383/determine-text-width

    width = 5.00 #* 3.0/2.0
    height = 10.0
    fig_o = FigureData( 2, 1, width, height, output_dir+'/'+'plot_stacked', units='kyr' ) #, times
    fig_o.fig.subplots_adjust(wspace=0.0,hspace=0.0)
    fig_o.time = times

    # sns.set_style("ticks")
    # sns.despine()

    ax0 = fig_o.ax[0]
    ax1 = fig_o.ax[1]
    # ax2 = fig_o.ax[2]
    # ax3 = fig_o.ax[3]

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

    fig_o.set_cmap(sci_colormaps['vik_r']) # "magma_r"

    ymax_atm_pressure = 0
    ymin_atm_pressure = 1000
    ymax_atm_z = 0
    ymin_atm_z = 0

    # Find planet mass
    core_mass   = myjson_o.get_dict_values(['atmosphere','mass_core'])
    mantle_mass = myjson_o.get_dict_values(['atmosphere','mass_mantle'])
    planet_mass = core_mass + mantle_mass

    for nn, time in enumerate( fig_o.time ):

        atm_file = output_dir+"/"+str(int(time))+"_atm.pkl"

        if os.path.exists(atm_file):

            # Read pickle file
            atm_file_stream = open(atm_file,'rb')
            atm = pkl.load(atm_file_stream)
            atm_file_stream.close()

            # read json
            myjson_o = MyJSON( output_dir+'/{}.json'.format(time) )

            color = fig_o.get_color( 1.0*nn/len(fig_o.time) )
            # use melt fraction to determine mixed region
            MIX = myjson_o.get_mixed_phase_boolean_array( 'basic' )
            # MIX = myjson_o.get_mixed_phase_boolean_array( 'basic_internal' )
            # MIX_s = myjson_o.get_mixed_phase_boolean_array( 'staggered' )

            label = latex_float(time)+" yr"

            # Pressure-height conversion for y-axis
            z_profile = AtmosphericHeight(atm, planet_mass, r_planet) # m
            z_profile = z_profile*1e-3 # km
            ax0.plot( atm.tmp, z_profile, '-', color=color, label=label, lw=1.5)

            # # Atmosphere T-P
            # ax0.semilogy( temperature_atmosphere, pressure_atmosphere, '-', color=color, label=label, lw=1.5)

            # interior
            temperature_interior = myjson_o.get_dict_values(['data','temp_b'])
            # yy_s = myjson_o.get_dict_values_internal(['data','temp_s'])
            ax1.plot( temperature_interior, pressure_interior, '--', color=color, lw=1.5 )
            ax1.plot( temperature_interior*MIX, pressure_interior*MIX, '-', color=color, label=label, lw=1.5 )

            # # connect atmosphere and interior lines at interface
            # connecting_line_T = [ temperature_atmosphere[-1], temperature_interior[0] ]
            # connecting_line_P = [ pressure_atmosphere[-1], pressure_atmosphere[-1]*1.1 ]
            # ax0.semilogy( connecting_line_T, connecting_line_P, ':', color=color, lw=1.5)

            if np.min(atm.p) > 0:
                ymax_atm_pressure = np.max([ymax_atm_pressure, np.max(atm.p)])
                ymin_atm_pressure = np.min([ymin_atm_pressure, np.min(atm.p)])
                ymax_atm_z = np.max([ymax_atm_z, np.max(z_profile)])
                ymin_atm_z = np.min([ymin_atm_z, np.min(z_profile)])

            # ymax_atm = pressure_atmosphere[-1]*1.1

    xmin         = 100
    xmax         = 5000
    xticks       = [ xmin, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, xmax ]
    title_xcoord = -0.11
    title_ycoord = 0.5

    units = myjson_o.get_dict_units(['data','temp_b'])
    # title = 'Atmosphere + mantle temperature profile' #'(a) Temperature, {}'.format(units)

    ##### Atmosphere part

    # ax0.set_xlabel("Temperature, $T$ (K)")
    ax0.set_ylabel("Atmosphere height, $z_\mathrm{atm}$ (km)")
    ax0.set_xlim( left=xmin, right=xmax )
    ax0.set_ylim( top=ymax_atm_z, bottom=ymin_atm_z )
    # ax0.set_yticks([ymin_atm_pressure, 1e-2, 1e-1, 1e0, 1e1, ymax_atm_pressure])
    ax0.set_yticks([round(ymin_atm_z,1), round(0.2*ymax_atm_z,1), round(0.4*ymax_atm_z,1), round(0.6*ymax_atm_z,1), round(0.8*ymax_atm_z,1), round(ymax_atm_z,1)])
    # ax0.set_yticklabels(["", str(round(0.25*ymax_atm_z,1)), str(round(0.5*ymax_atm_z,1)), str(round(0.75*ymax_atm_z,1)), str(round(ymax_atm_z,1))])
    ax0.yaxis.set_label_coords(title_xcoord,title_ycoord)
    # ax0.xaxis.tick_top()
    ax0.set_xticks(xticks)
    ax0.set_xticklabels([])
    # ax0.tick_params(direction='in')

    # # Y-axis settings
    # fig_o.set_myaxes( ax0, ylabel='$P_\mathrm{atm}$\n(mbar)', xmin=xmin, xmax=xmax, xticks=xticks )
    # ax0.set_ylim( top=ymax_atm_pressure, bottom=ymin_atm_pressure )
    # ax0.set_yticks([ymin_atm_pressure, 1e1, 1e2, 1e3, 1e4, ymax_atm_pressure])
    # ax0.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax0.get_yaxis().get_major_formatter().labelOnlyBase = False
    # ax0.xaxis.tick_top()
    # ax0.tick_params(direction='in')
    # ax0.yaxis.set_label_coords(-0.13,0.5)
    # ax0.invert_yaxis()
    # ax0.set_xticklabels([])

    # # Pressure on Y-axis from min/max values
    # ax0b = ax0.twinx()
    # ax0b.plot( temperature_atmosphere, pressure_atmosphere, alpha=0.0)
    # ax0b.set_xlim( right=xmax, left=xmin )
    # ax0b.set_ylim( top=ymin_atm_pressure, bottom=ymax_atm_pressure )
    # yticks_pressure = [ymin_atm_pressure, ymax_atm_pressure*(1./6.), ymax_atm_pressure*(2./6.), ymax_atm_pressure*(1./2.), ymax_atm_pressure*(4./6.), ymax_atm_pressure*(5./6.), ymax_atm_pressure]
    # ax0b.set_yticks( yticks_pressure )
    # ax0b.set_ylabel( '$P_\mathrm{atm}$\n(bar)', rotation=0 )
    # ax0b.yaxis.set_label_coords(1.15,0.55)

    # sns.despine(top=False, bottom=True)

    # Legend
    ax0.legend( fontsize=8, fancybox=True, framealpha=0.5 )

    #####  Interior part
    yticks = [0,20,40,60,80,100,120,int(pressure_interior[-1])]
    ymax = int(pressure_interior[-1])
    fig_o.set_myaxes( ax1, xlabel='Temperature, $T$ (K)', xmin=xmin, xmax=xmax, xticks=xticks, ymin=0, ymax=ymax, yticks=yticks )
    ax1.invert_yaxis()
    # ax1.set_ylabel( '$P_\mathrm{mantle}$ (GPa)' ) # , rotation=0
    # ax1.yaxis.set_label_coords(title_xcoord,title_ycoord)
    # ax1.yaxis.tick_right()
    # Do not show pressure labels for this plot
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    # ax1.tick_params(direction='in')
    
    # Pressure-depth conversion for interior y-axis
    ax1b = ax1.twinx()
    yy = myjson_o.get_dict_values(['data','temp_b'])
    ax1b.plot( yy, xx_depth, alpha=0.0)
    ax1b.set_xlim( right=xmax, left=xmin )
    ax1b.set_ylim(top=xx_depth[-1], bottom=xx_depth[0])
    # ax1b.set_yticks([0, 100, 500, 1000, 1500, 2000, 2500, int(xx_depth[-1])])
    ax1b.set_yticks([0, int(0.2*xx_depth[-1]), int(0.4*xx_depth[-1]), int(0.6*xx_depth[-1]), int(0.8*xx_depth[-1]), int(xx_depth[-1])])
    ax1b.invert_yaxis()
    ax1b.yaxis.tick_left()
    # ax1b.tick_params(direction='in')
    ax1.set_ylabel( 'Mantle depth, $d_\mathrm{mantle}$ (km)' )
    ax1.yaxis.set_label_coords(title_xcoord,title_ycoord)

    # # Pressure-height conversion for y-axis
    # ax0b = ax0.twinx()
    # r_planet = np.max(xx_radius*1e3) # m
    # ### CAREFUL: HARDCODED TO M_EARTH !!!
    # z_profile = coupler_AtmosphericHeight(temperature_atmosphere, pressure_atmosphere, 5.972E24, r_planet)
    # z_profile = z_profile*1e-3 # km
    # ax0b.plot( temperature_atmosphere, z_profile, alpha=0.0)
    # ax0b.set_xlim( right=xmax, left=xmin )
    # ax0b.set_ylim(top=np.max(z_profile), bottom=np.min(z_profile))
    # ax0b.set_yticks([0, np.max(z_profile)/2., np.max(z_profile)])
    # ax0b.set_ylabel( '$z$\n(km)', rotation=0 )
    # ax0b.yaxis.set_label_coords(1.15,0.55)
    # # ax0b.invert_yaxis()

    plt.close()
    plt.ioff()
    fig_o.savefig(1)

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
        output_list = get_all_output_pkl_times(output_dir)

        if len(output_list) <= 8:
            plot_list = output_list
        else:
            plot_list = [ output_list[0], output_list[int(round(len(output_list)*(2./100.)))], output_list[int(round(len(output_list)*(15./100.)))], output_list[int(round(len(output_list)*(22./100.)))], output_list[int(round(len(output_list)*(33./100.)))], output_list[int(round(len(output_list)*(50./100.)))], output_list[int(round(len(output_list)*(66./100.)))], output_list[-1] ]
        print("Snapshots:", plot_list)

    # # Read optional argument from console to provide output dir
    # output_dir_read = parser.parse_args().dir

    # output_list = su.get_all_output_times(output_dir_read)

    # if len(output_list) <= 8:
    #     plot_list = output_list
    # else:
    #     plot_list = [ output_list[0], output_list[int(round(len(output_list)*(2./100.)))], output_list[int(round(len(output_list)*(15./100.)))], output_list[int(round(len(output_list)*(22./100.)))], output_list[int(round(len(output_list)*(33./100.)))], output_list[int(round(len(output_list)*(50./100.)))], output_list[int(round(len(output_list)*(66./100.)))], output_list[-1] ]
    # print("snapshots:", plot_list)

    plot_stacked( output_dir=output_dir, times=plot_list )

#====================================================================

if __name__ == "__main__":

    main()
