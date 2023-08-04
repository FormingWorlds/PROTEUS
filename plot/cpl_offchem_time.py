#!/usr/bin/env python3

# Plot evolution of offline mixing ratios over time, for a set of species

# Import things
import matplotlib as mpl
from utils.plot import *
from utils.helper import find_nearest
import sys


def plot_offchem_time(output_dir, species, plot_init_mx=False, tmin=-1, prange=None):
    """Plot evolution of chemistry according to offline VULCAN output.
    
    Reads-in the data from output_dir for a set of provided species. Can also
    include AEOLUS/SPIDER mixing ratios as dashed lines, which were used to 
    initialise each VULCAN run. Mixing ratios are averaged over the provided 
    pressure range.

    Parameters
    ----------
        output_dir : str
            Output directory that was specified in the PROTEUS cfg file
        species : list
            List of species to plot
        prange : list(2) or float
            Minimum and maximum pressures (bar) to average over. If None, then whole column is used. If float, then single layer is used.
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
    ax.set_xlabel("Time [yr]")
    ax.set_xscale("log")
    title_str = "Mixing ratio versus time "

    prange_mode = -1
    if prange == None:
        prange_mode = 0
        title_str += "(whole column)"
    elif type(prange) == float:
        prange_mode = 1
        title_str += str("(p ≈ %1.1e bar)" % prange)
    elif len(prange) == 2:
        prange_mode = 2
        title_str += str("(p ≈ [%1.1e,%1.1e] bar)" % (prange[0],prange[1]))
    else:
        raise Exception("Error parsing value for parameter 'prange': " + str(prange))


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

            # Crop to pressure range
            if prange_mode == 0:
                mean_mx = np.mean(clean_mx)
            elif prange_mode == 1:
                _,vi = find_nearest(yd["pressure"],prange)
                mean_mx = clean_mx[vi]
            elif prange_mode == 2:
                vmin,vimin = find_nearest(yd["pressure"],prange[0])
                vmax,vimax = find_nearest(yd["pressure"],prange[1])
                if vimin == vimax:
                    raise Exception("Upper and lower bound of 'prange' are too close!")
                if vimin > vimax:
                    vimin,vimax = vimax,vimin
                    vmin,vmax = vmax,vmin
                mean_mx = np.mean(clean_mx[vimin:vimax])

            mean_mx = np.mean(clean_mx)           
            mx_vul.append(float(mean_mx))

        ax.plot(times,mx_vul,color='black',lw=lw+0.4,zorder=3)
        ax.plot(times,mx_vul,color=color,label=sp,lw=lw,zorder=4)

        if plot_init_mx:
            if str("ae_"+sp) in years_data[0].keys():
                for yd in years_data:
                    clean_mx = np.nan_to_num(yd["ae_"+sp])
                    mean_mx = np.mean(clean_mx)   
                    mx_aeo.append(float(mean_mx))

                ax.plot(times,mx_aeo,color=color,linestyle='--',lw=lw,zorder=2)

    title_str += str(", %d samples" % len(years_data))

    ax.set_title(title_str)
    ax.legend(loc="lower left",ncol=6)
    ax.set_ylim([1e-10,1.5])

    if tmin == -1:
        vmin = times[0]
    else:
        vmin = tmin
    ax.set_xlim([max(vmin,1.0),times[-1]*1.2])

    ax.axvline(x=times[-1],color='black',lw=0.7,zorder=1)

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
    prange = [1e0,1e-4]
    plot_offchem_time(dirs["output"],species,tmin=1e3,prange=prange)

    print("Done!")

