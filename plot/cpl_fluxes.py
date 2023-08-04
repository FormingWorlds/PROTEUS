#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_ext import *
from utils.plot import *

# Plotting fluxes
def cpl_fluxes(output_dir, atm):

    print("Plotting fluxes at each level")

    mpl.use('Agg')

    fig,ax = plt.subplots()

    ax.axvline(0,color='black',lw=0.8)

    pl = atm.pl * 1.e-5

    ax.plot(atm.flux_up_total,pl,color='red',label='UP',lw=1)
    ax.plot(atm.SW_flux_up   ,pl,color='red',label='UP SW',linestyle='dotted',lw=2)
    ax.plot(atm.LW_flux_up   ,pl,color='red',label='UP LW',linestyle='dashed',lw=1)

    ax.plot(-1.0*atm.flux_down_total,pl,color='green',label='DN',lw=2)
    ax.plot(-1.0*atm.SW_flux_down   ,pl,color='green',label='DN SW',linestyle='dotted',lw=3)
    ax.plot(-1.0*atm.LW_flux_down   ,pl,color='green',label='DN LW',linestyle='dashed',lw=2)

    ax.plot(atm.net_flux ,pl,color='black',label='NET')

    ax.set_xscale("symlog")
    ax.set_xlabel("Upward-directed flux [W m-2]")
    ax.set_ylabel("Pressure [Bar]")
    ax.set_yscale("log")
    ax.set_ylim([pl[-1],pl[0]])

    ax.legend(loc='upper left')

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_fluxes.pdf")

if __name__ == '__main__':
    print("WARNING: cpl_fluxes does not support command-line execution.")
    exit(1)
