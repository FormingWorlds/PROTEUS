from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from proteus import Proteus

log = logging.getLogger("fwl."+__name__)

def plot_heatingrates(output_dir: str, atm, comp: list=[]):
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


def plot_heatingrates_entry(handler: Proteus):
    raise NotImplementedError(
        "`plot_heatingrates()` does not support command-line execution."
    )


if __name__ == '__main__':
    from proteus.plot._cpl_helpers import get_handler_from_argv
    handler = get_handler_from_argv()
    plot_heatingrates_entry(handler)
