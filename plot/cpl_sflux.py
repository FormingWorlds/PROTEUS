#!/usr/bin/env python3

# Plots stellar flux from `output/`

from utils.modules_ext import *
from utils.constants import *
from utils.helper import natural_sort

star_cmap = plt.get_cmap('gnuplot2_r')

# Planck function value at stellar surface
# lam in nm
# erg s-1 cm-2 nm-1
def planck_function(lam, T):

    x = lam * 1.0e-9   # convert nm -> m
    hc_by_kT = phys.h*phys.c / (phys.k*T) 

    planck_func = 1.0/( (x ** 5.0) * ( np.exp( hc_by_kT/ x) - 1.0 ) )  
    planck_func *= 2 * phys.h * phys.c * phys.c #  w m-2 sr-1 s-1 m-1
    planck_func *= np.pi * 1.0e3 * 1.0e-9  # erg s-1 cm-2 nm-1

    return planck_func

def plot_sflux(output_dir):

    # Find and sort files
    files = glob.glob(output_dir+"/*.sflux")
    files = natural_sort(files)

    # Arrays for storing data over time
    time_t = []
    wave_t = []
    flux_t = []
    for f in files:
        # Load data
        X = np.loadtxt(f,skiprows=2,delimiter='\t').T
        
        # Parse data
        time = int(f.split('/')[-1].split('.')[0])
        wave = X[0]
        flux = X[1]

        # Save data
        time_t.append(time * 1.e-3)
        wave_t.append(wave)
        flux_t.append(flux)

    time_t = np.array(time_t)
    wave_t = np.array(wave_t)
    flux_t = np.array(flux_t)

    print("Found files for times: ", time_t)

    # Create figure
    N = len(time_t)

    fig,ax = plt.subplots(1,1)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)

    vmin = max(time_t[0],1.0)
    vmax = time_t[-1]
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=star_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical') 
    cbar.set_label("Time [kyr]") 

    ax.set_yscale("log")
    ax.set_ylabel("Flux [erg s-1 cm-2 nm-1]")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_xlim([0,1100.0])
    ax.set_title("Stellar flux (1 AU) scaled by F_band with age")

    # Plot spectra
    for i in range(N):
        c =  sm.to_rgba(time_t[i])
        ax.plot(wave_t[i],flux_t[i],color=c,alpha=0.6,lw=0.8)

    ymax = np.percentile(flux_t,99.5)
    ax.set_ylim([0,ymax])

    # Calculate planck function
    # Tstar = 3274.3578960897644
    # Rstar_cm = 36292459156.77782
    # AU_cm = 1.496e+13 
    # sf = Rstar_cm / AU_cm
    # planck_fl = [] 
    # for w in wave_t[4]:
    #     planck_fl.append(planck_function(w,Tstar) * sf * sf) 
    # ax.plot(wave_t[4],planck_fl,color='green',lw=1.5)

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_sflux.pdf")


# Run directly
if __name__ == '__main__':

    print("Plotting stellar flux over time...")

    plot_sflux(dirs['output'])

    print("Done!")


# End of file
