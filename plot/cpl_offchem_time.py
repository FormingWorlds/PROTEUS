#!/usr/bin/env python3

# Plot evolution of offline mixing ratios over time, for a set of species

# Import things
import matplotlib as mpl
from utils.plot import *
import sys


def plot_offchem_time(output_dir, species, plot_init_mx=False, tmin=-1):
    """Plot evolution of chemistry according to offline VULCAN output
    
    Reads-in the data from output_dir for a set of provided species. Can also
    include AEOLUS/SPIDER mixing ratios as dashed lines, which were used to 
    initialise each VULCAN run.

    Parameters
    ----------
        output_dir : str
            Output directory that was specified in the PROTEUS cfg file
        species : list
            List of species to plot
        plot_init_mx : bool
            Include initial mixing ratios for each VULCAN run in plot?
        tmin : float
            Initial plotting time (-1 for start of data)
    """

    mpl.use("Agg")

    ls = glob.glob(output_dir+"/offchem/*/output.vul")
    years = [int(f.split("/")[-2]) for f in ls]
    years_data = [offchem_read_year(output_dir,y,read_const=plot_init_mx) for y in years]

    if len(years) == 0:
        raise Exception('No VULCAN output files found')

    lw = 2

    fig, ax = plt.subplots(1,1,figsize=(8,4))

    ax.set_ylabel("Mean mixing ratio")
    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_title("Evolution of mean mixing ratio over time")
    ax.set_xlabel("Time [yr]")
    ax.set_xscale("log")

    for i,sp in enumerate(species):

        color_key = sp+"_1"
        if color_key in dict_colors:
            color = dict_colors[color_key]
        else:
            color = 'black'
            print("Warning: could not find a defined colour for species '%s' " % sp)

        times =     []
        mx_vul =    []
        mx_aeo =    []

        for yd in years_data:
            times.append(yd["year"])

            clean_mx = np.nan_to_num(yd["mx_"+sp])
            mean_mx = np.mean(clean_mx)              # Need to think about a better way of doing this
            mx_vul.append(float(mean_mx))

        ax.plot(times,mx_vul,color='black',lw=lw+0.4)
        ax.plot(times,mx_vul,color=color,label=sp,lw=lw)

        if plot_init_mx:
            if str("ae_"+sp) in years_data[0].keys():
                for yd in years_data:
                    clean_mx = np.nan_to_num(yd["ae_"+sp])
                    mean_mx = np.mean(clean_mx)   
                    mx_aeo.append(float(mean_mx))

                ax.plot(times,mx_aeo,color=color,linestyle='--',lw=lw)


    ax.legend(loc="lower left",ncol=6)
    ax.set_ylim([1e-10,1.5])

    if tmin == -1:
        vmin = times[0]
    else:
        vmin = tmin
    ax.set_xlim([max(vmin,1.0),times[-1]*1.2])

    ax.axvline(x=times[-1],color='black',lw=0.7)

    fig.tight_layout()
    fig.savefig(output_dir+"/plot_offchem_time.pdf")
    plt.close('all')


# If executed directly
if __name__ == '__main__':
    print("Plotting offline chemistry (mixing ratios vs time)...")

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg' 

    # Read in COUPLER input file
    from utils.coupler import ReadInitFile, SetDirectories
    COUPLER_options, time_dict = ReadInitFile( cfg )

    # Species to plot
    species = ["H2", "H2O", "H", "OH", "CO2", "CO", "CH4", "HCN", "NH3", "N2", "NO"]

    # Set directories dictionary
    dirs = SetDirectories(COUPLER_options)

    # Call plotting function
    plot_offchem_time(dirs["output"],species,tmin=1e3)

    print("Done!")

