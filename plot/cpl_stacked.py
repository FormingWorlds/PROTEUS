#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_ext import *
from utils.spider import *
from utils.plot import *

log = logging.getLogger("PROTEUS")

#====================================================================
def plot_stacked( output_dir, times, plot_format="pdf" ):

    log.info("Plot stacked")

    scale = 1.1
    fig,(axt,axb) = plt.subplots(2,1, figsize=(5*scale,10*scale), sharex=True)

    norm = mpl.colors.Normalize(vmin=times[0], vmax=times[-1])
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)  
    sm.set_array([])

    # limits
    y_depth, y_height = 100, 100

    # loop over times
    for time in times:

        # Get atmosphere data for this time 
        atm_file = os.path.join(output_dir, "data", "%d_atm.nc"%time)
        ds = nc.Dataset(atm_file)
        tmp =   np.array(ds.variables["tmpl"][:])
        z   =   np.array(ds.variables["zl"][:]) * 1e-3  # convert to km
        ds.close()
        
        # Get interior data for this time
        int_file = os.path.join(output_dir, "data", "%d.json"%time)
        myjson_o = MyJSON(int_file)
        temperature_interior = myjson_o.get_dict_values(['data','temp_b'])
        xx_radius = myjson_o.get_dict_values(['data','radius_b'])
        xx_radius *= 1.0E-3
        xx_depth = xx_radius[0] - xx_radius

        # use melt fraction to determine mixed region
        MASK_MI = myjson_o.get_mixed_phase_boolean_array( 'basic' ) 
        MASK_ME = myjson_o.get_melt_phase_boolean_array(  'basic' ) 
        MASK_SO = myjson_o.get_solid_phase_boolean_array( 'basic' ) 

        # overlap lines by 1 node 
        for m in (MASK_MI, MASK_ME, MASK_SO):
            m_new = m[:]
            for i in range(2,len(MASK_MI)-2,1):
                m_new[i] = m[i] or m[i-1] or m[i+1]
            m = m_new[:]

        # Plot decorators
        label = latex_float(time)+" yr"
        color = sm.to_rgba(time)

        # Plot atmosphere
        axt.plot( tmp, z, '-', color=color, label=label, lw=1.5)

        # Plot interior
        axb.plot( temperature_interior[MASK_SO], xx_depth[MASK_SO], linestyle='solid',  color=color, lw=1.5 )
        axb.plot( temperature_interior[MASK_MI], xx_depth[MASK_MI], linestyle='dashed', color=color, lw=1.5 )
        axb.plot( temperature_interior[MASK_ME], xx_depth[MASK_ME], linestyle='dotted', color=color, lw=1.5 )

        # update limits 
        y_height = max(y_height, np.amax(z))
        y_depth  = max(y_depth,  np.amax(xx_depth))


    ytick_spacing = 100.0 # km

    # Decorate top plot 
    axt.set(ylabel="Atmosphere height [km]")
    axt.set_ylim(bottom=0.0, top=y_height)
    axt.legend()
    axt.yaxis.set_minor_locator(MultipleLocator(ytick_spacing))
    axt.set_facecolor(dict_colors["atm_bkg"] )

    # Decorate bottom plot 
    axb.set(ylabel="Interior depth [km]", xlabel="Temperature [K]")
    axb.invert_yaxis()
    axb.set_ylim(top=0.0, bottom=y_depth)
    axb.yaxis.set_minor_locator(MultipleLocator(ytick_spacing))
    axb.set_facecolor(dict_colors["int_bkg"] )
    axb.xaxis.set_major_locator(MultipleLocator(1000))
    axb.xaxis.set_minor_locator(MultipleLocator(250))

    fig.subplots_adjust(hspace=0)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plot_stacked.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')


#====================================================================
def main():

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg' 

    # Read in COUPLER input file
    log.info("Read cfg file")
    from utils.coupler import ReadInitFile, SetDirectories
    COUPLER_options, time_dict = ReadInitFile( cfg )

    # Set directories dictionary
    dirs = SetDirectories(COUPLER_options)

    files = glob.glob(os.path.join(dirs["output"], "data", "*_atm.nc"))
    times = [int(f.split("/")[-1].split("_")[0]) for f in files]

    if len(times) <= 8:
        plot_list = times
    else:
        plot_list = [ times[0] ]
        for f in [15,25,33,50,66,75]:
            plot_list.append(times[int(round(len(times)*(float(f)/100.)))])
        plot_list.append(times[-1])
    print("Snapshots:", plot_list)

    plot_stacked( output_dir=dirs["output"], times=plot_list, 
                 plot_format=COUPLER_options["plot_format"] )

#====================================================================

if __name__ == "__main__":

    main()
