#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_plot import *

#====================================================================
def plot_interior( output_dir, times ):

    # article class text width is 4.7747 inches
    # http://tex.stackexchange.com/questions/39383/determine-text-width

    # logger.info( 'building mantle_evolution' )

    width = 12.00 #* 3.0/2.0
    height = 6.0
    fig_o = su.FigureData( 1, 4, width, height, output_dir+'/'+'plot_interior', units='kyr' ) #, times
    fig_o.fig.subplots_adjust(wspace=0.15,hspace=0.2)
    fig_o.time = times

    ax0 = fig_o.ax[0]
    ax1 = fig_o.ax[1]
    ax2 = fig_o.ax[2]
    ax3 = fig_o.ax[3]

    time = fig_o.time[0] # first timestep since liquidus and solidus
                         # are time-independent

    myjson_o = su.MyJSON( output_dir+'/{}.json'.format(time) )

    # print(myjson_o.data_d)
    # TIMEYRS = myjson_o.data_d['nstep']

    # hack to compute some average properties for Bower et al. (2018)
    #xx_liq, yy_liq = fig_o.get_xy_data( 'liquidus_rho', time )
    #xx_sol, yy_sol = fig_o.get_xy_data( 'solidus_rho', time )
    #diff = (yy_liq - yy_sol) / yy_sol * 100.0
    #print diff[40:]
    #print np.mean(diff[40:])
    #sys.exit(1)

    xx_pres = myjson_o.get_dict_values(['data','pressure_b'])
    xx_pres *= 1.0E-9
    xx_pres_s = myjson_o.get_dict_values(['data','pressure_s'])
    xx_pres_s *= 1.0E-9

    xx_radius = myjson_o.get_dict_values(['data','radius_b'])
    xx_radius *= 1.0E-3
    xx_depth = xx_radius[0] - xx_radius
    xx_radius_s = myjson_o.get_dict_values(['data','radius_s'])
    xx_radius_s *= 1.0E-3
    xx_depth_s = xx_radius_s[0] - xx_radius_s

    handle_l = [] # handles for legend

    fig_o.set_colors(cmap=vik_r) # "magma_r"

    for nn, time in enumerate( fig_o.time ):

        # read json
        myjson_o = su.MyJSON( output_dir+'/{}.json'.format(time) )

        color = fig_o.get_color( nn )

        # use melt fraction to determine mixed region
        MIX = myjson_o.get_mixed_phase_boolean_array( 'basic' ) # basic_internal
        MIX_s = myjson_o.get_mixed_phase_boolean_array( 'staggered' )

        # label = fig_o.get_legend_label( time )
        # label = "{:.1e}".format(Decimal(time))+" yr"
        label = latex_float(time)+" yr"

        print(time, )

        # temperature
        yy = myjson_o.get_dict_values(['data','temp_b'])
        ax0.plot( yy, xx_pres, '--', color=color, lw=1.5 )
        handle, = ax0.plot( yy*MIX, xx_pres*MIX, '-', color=color, lw=1.5, label=label )

        # melt fraction
        yy = myjson_o.get_dict_values(['data','phi_b'])
        ax1.plot( yy, xx_pres, '-', color=color, lw=1.5 )

        # viscosity
        visc_const = 1 # this is used for the arcsinh scaling
        visc_fmt = su.MyFuncFormatter( visc_const )
        yy = myjson_o.get_dict_values(['data','visc_b'], visc_fmt)
        ax2.plot( yy, xx_pres, '-', color=color, lw=1.5 )

        # entropy
        # plot staggered, since this is where entropy is defined
        # cell-wise and we can easily see the CMB boundary condition
        yy = myjson_o.get_dict_values(['data','S_s'])
        # yy = myjson_o.get_dict_values(['data','S_b'])
        ax3.plot( yy, xx_pres_s, '-', color=color, lw=1.5 )
        # print(yy, xx_pres)
        # ax3.plot( yy*MIX_s, xx_pres_s*MIX_s, '-', color=color, label=label )

        # legend
        handle_l.append( handle )

        print(time)

    yticks = [0,20,40,60,80,100,120,int(xx_pres_s[-1])]
    ymax = int(xx_pres_s[-1])

    # titles and axes labels, legends, etc

    units = myjson_o.get_dict_units(['data','temp_b'])
    title = '(a) Temperature' #'(a) Temperature, {}'.format(units)
    xticks = [200, 1000, 2000, 3000, 4000, 5000]
    fig_o.set_myaxes( ax0, title=title, xlabel='$T$, '+units, ylabel='$P$\n(GPa)', xticks=xticks, ymax=ymax, yticks=yticks ) # , xmin=300, xmax=4200
    ax0.set_xlim( 200, 5000 )
    ax0.yaxis.set_label_coords(-0.25,0.5)
    ax0.invert_yaxis()
    # fig_o.set_mylegend( ax0, handle_l, loc=3, ncol=1 )
    ax0.legend( fontsize=8, fancybox=True, framealpha=0.5, loc=3 )

    title = '(b) Melt fraction'
    xticks = [0,0.2,0.4,0.6,0.8,1.0]
    fig_o.set_myaxes( ax1, title=title, xlabel='$\phi$', xticks=xticks, ymax=ymax, yticks=yticks )
    # ax1.yaxis.set_label_coords(-0.075,0.475)
    ax1.set_xlim( [0, 1] )
    ax1.set_yticklabels([])
    ax1.invert_yaxis()

    units = myjson_o.get_dict_units(['data','visc_b'])
    title = '(c) Viscosity'#'(c) Viscosity, ' + units
    fig_o.set_myaxes( ax2, title=title, xticks=[1.0E1, 1.0E5, 1.0E10, 1.0E15, 1.0E19, 1.0E22], xlabel='$\eta$, '+ units, yticks=[0,20,40,60,80,100,120,140], xfmt=visc_fmt )
    ax2.set_yticklabels([])
    ax2.invert_yaxis()

    units = myjson_o.get_dict_units(['data','S_b'])
    title = '(d) Entropy'#'(d) Entropy, {}'.format(units)
    xticks = [400,800,1600,2400,3200]
    fig_o.set_myaxes( ax3, title=title, xlabel='$S$, '+ units, xticks=xticks, ymax=ymax, yticks=yticks ) # ' $(J \; kg^\mathrm{-1} \; K^\mathrm{-1})$'
    ax3.set_yticklabels([])
    ax3.invert_yaxis()

    # Pressure-depth conversion for y-axis
    ax3b = ax3.twinx()
    yy = myjson_o.get_dict_values(['data','S_b'])
    ax3b.plot( yy, xx_depth, alpha=0.0)
    ax3b.set_ylim(top=xx_depth[-1], bottom=xx_depth[0])
    ax3b.set_xlim(right=np.max(xticks), left=np.min(xticks))
    ax3b.set_yticks([0, 500, 1000, 1500, 2000, 2500, int(xx_depth[-1])])
    ax3b.set_ylabel( '$d$\n(km)', rotation=0 )
    ax3b.yaxis.set_label_coords(1.30,0.55)
    ax3b.invert_yaxis()

    # if 0:
    #     # add sol and liq text boxes to mark solidus and liquidus
    #     # these have to manually adjusted according to the solidus
    #     # and liquidus used for the model
    #     prop_d = {'boxstyle':'round', 'facecolor':'white', 'linewidth': 0}
    #     ax0.text(120, 2925, r'liq', fontsize=8, bbox=prop_d )
    #     ax0.text(120, 2075, r'sol', fontsize=8, bbox=prop_d )
    #     # DJB commented out for now, below labels 0.2 melt fraction
    #     # so the reader knows that dashed white lines denote melt
    #     # fraction contours
    #     ax0.text(110, 2275, r'$\phi=0.2$', fontsize=6 )

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

    # Define which times are plotted
    if args.times:
        plot_list = [ int(time) for time in args.times.split(',') ]
        print("Snapshots:", plot_list)
    else:
        output_list = su.get_all_output_times(output_dir)

        if len(output_list) <= 8:
            plot_list = output_list
        else:
            plot_list = [ output_list[0], output_list[int(round(len(output_list)*(2./100.)))], output_list[int(round(len(output_list)*(15./100.)))], output_list[int(round(len(output_list)*(22./100.)))], output_list[int(round(len(output_list)*(33./100.)))], output_list[int(round(len(output_list)*(50./100.)))], output_list[int(round(len(output_list)*(66./100.)))], output_list[-1] ]
        print("Snapshots:", plot_list)

    plot_interior( output_dir=output_dir, times=plot_list )

#====================================================================

if __name__ == "__main__":

    # Import utils- and plot-specific modules
    from modules_utils import *
    from modules_plot import *
    import utils_coupler as cu
    import utils_spider as su

    main()
