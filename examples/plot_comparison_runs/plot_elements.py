#This routine needs to be called via python3 src/proteus/plot/compar_atmosphere_models.py.
# Functions used to help run PROTEUS which are mostly module agnostic.

# Import utils-specific modules
from __future__ import annotations

import glob
import logging
import os
import sys
import subprocess
from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import LogLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.use('Agg')  # noqa
from string import ascii_letters

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import vol_list, element_list
from proteus.utils.helper import find_nearest
from proteus.utils.plot import get_colour, latexify, sample_times

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)




def plot_atmosphere(output_dir):

    df = pd.read_csv(output_dir + 'runtime_helpfile.csv',sep='\t')
    log.info("Plot element inventroy during the evolution")


    # Initialise plot
    scale = 1.1
    alpha = 0.6
    fig,ax = plt.subplots(1,1,figsize=(5*scale,4*scale))
    ax.set_ylabel("element inventroy [kg]")
    ax.set_xlabel("time")
    ax.set_yscale("log")
    ax.set_xscale("log")
    

    for e in element_list:
        color = get_colour(e)
        ax.plot(df['Time'], df[e+'_kg_total'], color=color, linestyle='-',label=e, alpha=alpha, zorder=3)
        ax.plot(df['Time'], df[e+'_kg_atm'], color=color, linestyle='dotted',label=e, alpha=alpha, zorder=3)

    ax.set_ylim(10**4, max(df['O_kg_total'])*10)
    plt.legend(fontsize=6)
    # Save plot
    fname = os.path.join(output_dir,"plots","element_inventory.png")
    fig.savefig(fname, bbox_inches='tight', dpi=300)


def plot_fO2_shift(output_dir):

    df = pd.read_csv(output_dir + 'runtime_helpfile.csv',sep='\t')
    log.info("Plot element inventroy during the evolution")


    # Initialise plot
    scale = 1.1
    alpha = 0.6
    fig,ax = plt.subplots(1,1,figsize=(5*scale,4*scale))
    ax.set_ylabel("fO2 shift wrt IW buffer")
    ax.set_xlabel("time")
    ax.set_xscale("log")
 
    color = 'black'
    ax.plot(df['Time'], df['fO2_shift'], color=color, linestyle='-',label='fO2_shift dIW', alpha=alpha, zorder=3)

    ax.set_ylim(-10, max(df['fO2_shift'])*1.1)
    plt.legend(fontsize=6)
    # Save plot
    fname = os.path.join(output_dir,"plots","fO2_shift.png")
    fig.savefig(fname, bbox_inches='tight', dpi=300)




if __name__ == "__main__":

    output_dir=sys.argv[1]
    plot_atmosphere(output_dir)
    plot_fO2_shift(output_dir)