#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_ext import *
from utils.plot import *

log = logging.getLogger("PROTEUS")

def cpl_heatingrates(output_dir, atm, comp=[]):
    """Plot temperature structure and corresponding heating rates versus pressure
    
    Reads the temperature for each pressure level in atm, and plots it alongside 
    the heating rates on the same pressure y-axis. Can also plot the temperature
    structure from another atmos object for comparison.

    Parameters
    ----------
        output_dir : str
            Output directory that was specified in the PROTEUS cfg file
        atm : atmos
            Atmosphere object containing P, T, heatingrates

        comp : np.ndarray shape (2,n)
            Optional PT profile for comparison (p=comp[0] and T=comp[1])
    """

    log.info("Plotting heating rates at each level")

    mpl.use('Agg')

    fig,ax = plt.subplots(1,1)

    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylabel("Pressure [Bar]")

    ax2 = ax.twiny()
    c = 'tab:red'
    ax2.set_xlabel("Temperature [K]",color=c)
    ax2.tick_params(axis='x', colors=c)
    ax2.plot(atm.tmp,  atm.p  * 1.e-5, color=c, label="$T(p)$ centre", zorder=4)
    ax2.plot(atm.tmpl, atm.pl * 1.e-5, color=c, label="$T(p)$ edge",   zorder=3, alpha=0.5)

    if (len(comp) == 2):
        c = 'tab:purple'
        ax2.plot(comp[1],comp[0] * 1.e-5, color=c, label="$T(p)$ comparison",zorder=1)
        ax2.legend()

    c = 'black'
    ax.axvline(x=0,color=c,lw=0.4)
    ax.plot(atm.net_heating,atm.p * 1.e-5,color=c,zorder=2)
    ax.set_xlabel("Heating rate [K/day]")
    ax.set_xscale("symlog")
    ax.tick_params(axis='x', colors=c)

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_heatingrates.pdf")

if __name__ == '__main__':
    print("cpl_heatingrates does not support command-line execution.")
    exit(1)
