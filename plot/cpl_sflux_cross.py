#!/usr/bin/env python3

# Plots stellar flux from output directory for a set of wavelength bins

from utils.modules_ext import *
from utils.constants import *
from utils.plot import *
from utils.helper import natural_sort
from matplotlib.colors import LinearSegmentedColormap

star_cmap = sci_colormaps['oleron']

# Planck function value at stellar surface
# lam in nm
# erg s-1 cm-2 nm-1
def planck_function(lam, T):

    x = lam * 1.0e-9   # convert nm -> m
    hc_by_kT = const_h*const_c / (const_k*T) 

    planck_func = 1.0/( (x ** 5.0) * ( np.exp( hc_by_kT/ x) - 1.0 ) )  
    planck_func *= 2 * const_h * const_c * const_c #  w m-2 sr-1 s-1 m-1
    planck_func *= np.pi * 1.0e3 * 1.0e-9  # erg s-1 cm-2 nm-1

    return planck_func

def plot_sflux_cross(output_dir, wl_targets:list=[], surface:bool=False):
    """Plots stellar flux vs time, for a set of wavelengths.

    Note that this function will plot the flux from EVERY file it finds.
    Saves plot as 'cpl_sflux_cross.pdf' in  the output directory.

    Parameters
    ----------
        output_dir : str
            Directory for both reading from and saving to.

        wl_targets : list
            List of wavelengths to plot [nm]
        surface : bool
            Use fluxes at surface? If not, will use fluxes at 1 AU.
    """ 

    mpl.use('Agg')

    # Wavelength targets default value
    if len(wl_targets) < 1:
        wl_targets = [1.0, 12.0, 50.0, 121.0, 200.0, 400.0, 500.0, 2000.0]

    # Find and sort files
    if surface:
        suffix = 'sfluxsurf'
    else:
        suffix = 'sflux'

    files = glob.glob(output_dir+"/data/*."+suffix)
    files = natural_sort(files)

    if (len(files) == 0):
        print("No files found!")
        exit(1)

    # Arrays for storing data over time
    time_t = []
    wave_t = []
    flux_t = []
    for f in files:
        # Load data
        X = np.loadtxt(f,skiprows=1,delimiter='\t').T
        
        # Parse data
        time = int(f.split('/')[-1].split('.')[0])
        if (time < 0): continue
        wave = X[0]
        flux = X[1]

        # Save data
        time_t.append(time)
        wave_t.append(wave)
        flux_t.append(flux)

    time_t = np.array(time_t)
    wave_t = np.array(wave_t)
    flux_t = np.array(flux_t)

    # Create figure
    fig,ax = plt.subplots(1,1)

    ax.set_yscale("log")
    ax.set_ylabel("Flux [erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$]")

    ax.set_xscale("log")
    ax.set_xlabel("Time [yr]")
    if surface:
        ax.set_title("Surface flux versus time")
    else:
        ax.set_title("1 AU flux versus time")

    vmin = max(time_t[0],1.0)
    vmax = time_t[-1]
    ax.set_xlim([vmin,vmax])

    # Find indices for wavelength bins
    wl_iarr = []
    wl_varr = []
    for w in wl_targets:
        wl_v, wl_i = find_nearest(wave,w)
        wl_iarr.append(wl_i)
        wl_varr.append(wl_v)
    N = len(wl_iarr)

    # Load modern spectrum
    if not surface:
        X = np.loadtxt(output_dir+'/-1.sflux',skiprows=2).T

    # Plot spectra
    for i in range(N):

        fl = flux_t.T[wl_iarr[i]]
        c = star_cmap(1.0*i/N)
        lbl = "%d"%max(1,round(wl_varr[i]))

        ax.plot(time_t,fl,color='black',lw=3.5)
        ax.plot(time_t,fl,color=c      ,lw=2.8,label=lbl)

        # Plot modern values
        if not surface:
            ax.scatter(time_t[-1],X[1][wl_iarr[i]],marker='o',color='k',s=40, zorder=3)
            ax.scatter(time_t[-1],X[1][wl_iarr[i]],marker='o',color=c,  s=34, zorder=4)

    leg = ax.legend(title="$\lambda$ [nm]", loc='center left',bbox_to_anchor=(1.02, 0.5))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_sflux_cross.pdf",  bbox_inches="tight")


# Run directly
if __name__ == '__main__':

    print("Plotting stellar flux over time (bins)...")

    wl_bins = [1.0, 12.0, 50.0, 121.0, 200.0, 400.0, 500.0, 2000.0]

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg' 
   
    # Read in COUPLER input file
    from utils.coupler import ReadInitFile, SetDirectories
    COUPLER_options, time_dict = ReadInitFile( cfg )

    # Set directories dictionary
    dirs = SetDirectories(COUPLER_options)

    plot_sflux_cross(dirs['output'], wl_targets=wl_bins, surface=False)

    print("Done!")


# End of file
