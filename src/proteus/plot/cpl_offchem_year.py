#!/usr/bin/env python3

# Plot evolution of mixing ratios over time, for a single species, versus pressure

# Import things
import matplotlib as mpl
mpl.use("Agg")
from proteus.utils.plot_offchem import *


def plot_offchem_year(output_dir, year_dict, species, plot_init_mx=False):
    """Plot a set of species in the atmosphere, for a single year only
    
    Reads-in the data from output_dir for a single year. Can also
    include JANUS/SPIDER mixing ratios as dashed lines, which were used to 
    initialise each VULCAN run.

    Parameters
    ----------
        output_dir : str
            Output directory that was specified in the PROTEUS cfg file
        species : list
            Which species to plot? (e.g. H2O) If none, then plot all.
        tmin : float
            Initial year to include [yr]
        tmax : float 
            Final yeat to include [yr]
        plot_init_mx : bool
            Include initial mixing ratios for each VULCAN run in plot?
    """

    if species == None:
        species = []
        for k in year_dict.keys():
            if "_" not in k:
                continue 
            species.append(k.split("_")[1])

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
    ax1.set_xlabel("Mole fraction")
    ax1.set_xscale("log")


    min_mix = 5e-1
    for i,s in enumerate(species):

        if s in dict_colors.keys():
            color = dict_colors[s]
            ls = 'solid'
        else:
            color = None
            ls = 'dotted'
            print("Warning: could not find a defined colour for species '%s' " % s)

        pretty = s
        if s in vol_latex.keys():
            pretty = vol_latex[s]

        # VULCAN result
        key = str("mx_"+s)
        if key in year_dict.keys():
            l = ax1.plot(year_dict[key],year_dict["pressure"],label=pretty,ls=ls,lw=lw,color=color)[0]
            color = l.get_color()
            min_mix = min(min_mix,np.amin(year_dict[key]))
            
        # JANUS result
        if plot_init_mx:
            key = str("mv_"+s)
            if key in year_dict.keys():
                ax1.scatter([year_dict[key]],[np.amax(year_dict["pressure"])],color=color)
            
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


    plot_janus = True


    # Read in COUPLER input file
    from utils.coupler import ReadInitFile, SetDirectories
    OPTIONS = ReadInitFile( cfg )

    # Species to make plots for
    species = ["H2", "H2O", "H", "OH", "CO2", "CO", "CH4","HCN", "NH3", "N2", "NO"]
    # species = None

    # Set directories dictionary
    dirs = SetDirectories(OPTIONS)

    # Load data
    ls = glob.glob(dirs["output"]+"offchem/*/output.vul")
    years = [int(f.split("/")[-2]) for f in ls]
    years_data = [offchem_read_year(dirs["output"],y,read_const=plot_janus) for y in years]

    if len(years) == 0:
        raise Exception('No VULCAN output files found')

    # Call plotting function
    for yd in years_data:
        print("Year = %d" % yd["year"])
        plot_offchem_year(dirs["output"],yd,species,plot_init_mx=plot_janus)

    print("Done!")
