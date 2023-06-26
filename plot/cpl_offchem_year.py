#!/usr/bin/env python3

# Plot evolution of mixing ratios over time, for a single species, versus pressure

# Import things
import matplotlib as mpl
mpl.use("Agg")
from utils.plot import *


def plot_offchem_year(output_dir, year_dict, species, plot_init_mx=False):
    """Plot a set of species in the atmosphere, for a single year only
    
    Reads-in the data from output_dir for a single year. Can also
    include AEOLUS/SPIDER mixing ratios as dashed lines, which were used to 
    initialise each VULCAN run.

    Parameters
    ----------
        output_dir : str
            Output directory that was specified in the PROTEUS cfg file
        sp : list
            Which species to plot? (e.g. H2O)
        tmin : float
            Initial year to include [yr]
        tmax : float 
            Final yeat to include [yr]
        plot_init_mx : bool
            Include initial mixing ratios for each VULCAN run in plot?
    """

    lw = 2

    mpl.use("Agg")

    fig, (ax0,ax1) = plt.subplots(1,2,sharey=True,figsize=(8,4))

    # Temperature profile
    ax0.set_title("Time = %g kyr" % (year_dict["year"]/1.e3))
    ax0.set_ylabel("Pressure [bar]")
    ax0.invert_yaxis()
    ax0.set_yscale("log")
    ax0.set_xlabel("Temperature [K]")
   
    ax0.plot(year_dict["temperature"],year_dict["pressure"],color='black',lw=lw)

    # Mixing ratios
    ax1.set_xlabel("Mixing ratio")
    ax1.set_xscale("log")


    min_mix = 5e-1
    for i,s in enumerate(species):

        color_key = s+"_1"
        if color_key in dict_colors:
            color = dict_colors[color_key]
        else:
            color = 'black'
            print("Warning: could not find a defined colour for species '%s' " % s)


        # VULCAN result
        key = str("mx_"+s)
        if key in year_dict.keys():
            ax1.plot(year_dict[key],year_dict["pressure"],lw=lw+0.4,color='black')
            ax1.plot(year_dict[key],year_dict["pressure"],label=s,lw=lw,color=color)
            min_mix = min(min_mix,np.amin(year_dict[key]))
            
        # AEOLUS result
        if plot_init_mx:
            key = str("ae_"+s)
            if key in year_dict.keys():
                ax1.plot(np.ones((len(year_dict["pressure"]))) * year_dict[key],year_dict["pressure"],linestyle='--',lw=lw*0.5,color=color)
            
    ax1.set_xlim([min_mix,1])
    ax1.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(output_dir+"plot_offchem_year_%d.pdf"%year_dict["year"])
    plt.close(fig)
    
    
# If executed directly
if __name__ == '__main__':
    print("Plotting offline chemistry for each year (mixing ratios vs pressure)...")

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg' 


    plot_aeolus = False


    # Read in COUPLER input file
    from utils.coupler import ReadInitFile, SetDirectories
    COUPLER_options, time_dict = ReadInitFile( cfg )

    # Species to make plots for
    species = ["H2", "H2O", "H", "OH", "CO2", "CO", "CH4","HCN", "NH3", "N2", "NO"]

    # Set directories dictionary
    dirs = SetDirectories(COUPLER_options)

    # Load data
    ls = glob.glob(dirs["output"]+"offchem/*/output.vul")
    years = [int(f.split("/")[-2]) for f in ls]
    years_data = [offchem_read_year(dirs["output"],y,read_const=plot_aeolus) for y in years]

    if len(years) == 0:
        raise Exception('No VULCAN output files found')

    # Call plotting function
    for yd in years_data:
        print("Year = %d" % yd["year"])
        plot_offchem_year(dirs["output"],yd,species,plot_init_mx=plot_aeolus)

    print("Done!")
