#!/usr/bin/env python3

# Plots stellar flux from `output/` versus time (colorbar)

from utils.modules_ext import *
from utils.constants import *
from utils.helper import natural_sort
from mpl_toolkits.axes_grid1 import make_axes_locatable

log = logging.getLogger("PROTEUS")

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

def plot_sflux(output_dir, wl_max = 6000.0, plot_format="pdf"):
    """Plots stellar flux vs time for all wavelengths

    Note that this function will plot the flux from EVERY file it finds.
    Saves plot as 'cpl_sflux.pdf' in  the output directory.

    Parameters
    ----------
        output_dir : str
            Directory for both reading from and saving to.

        wl_max : float
            Upper limit of wavelength axis [nm]

    """ 

    mpl.use('Agg')
    star_cmap = plt.get_cmap('Spectral')

    # Find and sort files
    files_unsorted = glob.glob(output_dir+"/data/*.sflux")
    files = natural_sort(files_unsorted)

    if (len(files) == 0):
        log.warning("No files found when trying to plot stellar flux")
        return

    # Downsample data
    if (len(files) > 200):
        files = files[::2]
    if (len(files) > 500):
        files = files[::2]
    if (len(files) > 1000):
        files = files[::3]

    if (len(files) != len(files_unsorted)):
        log.debug("Downsampled data over time")

    # Arrays for storing data over time
    time_t = []
    wave_t = []
    flux_t = []
    for f in files:
        # Load data
        X = np.loadtxt(f,skiprows=1,delimiter='\t').T
        
        # Parse data
        time = int(f.split('/')[-1].split('.')[0])
        if time < 0: continue

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
    N = len(time_t)

    fig,ax = plt.subplots(1,1)

    # Colorbar
    justone = bool(N == 1)
    if not justone:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)

        vmin = max(time_t[0],1.0)
        vmax = time_t[-1]
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=star_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical') 
        cbar.set_label("Time [yr]") 
    else:
        print("Only one spectrum was found")

    ax.set_yscale("log")
    ax.set_ylabel("Flux [erg s-1 cm-2 nm-1]")

    ax.set_xscale("log")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_xlim([0.5,max(1.0,wl_max)])
    ax.set_title("TOA flux versus wavelength")

    # Plot historical spectra
    for i in range(N):
        if justone:
            c = 'tab:blue'
            l = "%.2e yr"%(time_t[i])
        else:
            c = sm.to_rgba(time_t[i])
            l = None
        ax.plot(wave_t[i],flux_t[i],color=c,lw=0.7,alpha=0.6, label=l)

    # Plot current spectrum (use the copy made in the output directory)
    X = np.loadtxt(output_dir+'/-1.sflux',skiprows=2).T
    ax.plot(X[0],X[1],color='black',label='Modern',lw=0.8,alpha=0.9)

    # Calculate planck function
    # Tstar = 3274.3578960897644
    # Rstar_cm = 36292459156.77782
    # AU_cm = 1.496e+13 
    # sf = Rstar_cm / AU_cm
    # planck_fl = [] 
    # for w in wave_t[4]:
    #     planck_fl.append(planck_function(w,Tstar) * sf * sf) 
    # ax.plot(wave_t[4],planck_fl,color='green',lw=1.5)

    ax.legend()

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_sflux.%s"%plot_format, 
                bbox_inches='tight', dpi=200)


# Run directly
if __name__ == '__main__':

    print("Plotting stellar flux over time (colorbar)...")

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg' 

    from utils.coupler import ReadInitFile, SetDirectories

    # Read in COUPLER input file
    COUPLER_options, time_dict = ReadInitFile( cfg )

    # Set directories dictionary
    dirs = SetDirectories(COUPLER_options)

    plot_sflux(dirs['output'], plot_format=COUPLER_options["plot_format"])

    print("Done!")


# End of file
