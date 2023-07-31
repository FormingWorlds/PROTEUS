#!/usr/bin/env python3

# Import utils- and plot-specific modules
from utils.modules_ext import *
from utils.plot import *

# Plotting heating rates
def cpl_heatingrates(output_dir, atm):

    print("Plotting heating rates at each level")

    mpl.use('Agg')

    fig,ax = plt.subplots(1,1)

    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylabel("Pressure [Bar]")

    ax2 = ax.twiny()
    ax2.set_xlabel("Temperature [K]",color='red')
    ax2.plot(atm.tmpl,atm.pl * 1.e-5,color='red')
    ax2.tick_params(axis='x', colors='red')

    ax.axvline(x=0,color='black',lw=0.4)
    ax.plot(atm.net_heating,atm.p * 1.e-5,color='black')
    ax.set_xlabel("Heating rate [K/day]")
    ax.set_xscale("symlog")
    ax.tick_params(axis='x', colors='black')

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_heatingrates.pdf")

if __name__ == '__main__':
    print("WARNING: cpl_heatingrates does not support command-line execution.")
    exit(1)
