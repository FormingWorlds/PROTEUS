#!/usr/bin/env python3

# Import utils- and plot-specific modules
import argparse
import logging
import pathlib
import json
import subprocess
import os, sys, glob, shutil, re
from datetime import datetime
import copy
import warnings

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
from cmcrameri import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator, LinearLocator, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.font_manager as fm

import netCDF4 as nc
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from proteus.utils.plot import *

log = logging.getLogger("PROTEUS")

# Plotting fluxes
def plot_fluxes_atmosphere(output_dir, plot_format="pdf"):

    log.info("Plot atmosphere fluxes")

    files = glob.glob(os.path.join(output_dir, "data", "*_atm.nc"))
    nc_fpath = natural_sort(files)[-1]

    # Read netCDF
    ds = nc.Dataset(nc_fpath)
    atm_pl      = np.array(ds.variables['pl'     ][:])
    atm_fl_D_LW = np.array(ds.variables['fl_D_LW'][:])
    atm_fl_U_LW = np.array(ds.variables['fl_U_LW'][:])
    atm_fl_D_SW = np.array(ds.variables['fl_D_SW'][:])
    atm_fl_U_SW = np.array(ds.variables['fl_U_SW'][:])
    atm_fl_D    = np.array(ds.variables['fl_D'   ][:])
    atm_fl_U    = np.array(ds.variables['fl_U'   ][:])
    atm_fl_N    = np.array(ds.variables['fl_N'   ][:])
    ds.close()

    scale = 1.0
    fig,ax = plt.subplots(1,1, figsize=(7*scale,6*scale ))

    ax.axvline(0,color='black',lw=0.8)

    pl = atm_pl * 1.e-5

    ax.plot(atm_fl_U      ,pl,color='red',label='UP',lw=1)
    ax.plot(atm_fl_U_SW   ,pl,color='red',label='UP SW',linestyle='dotted',lw=2)
    ax.plot(atm_fl_U_LW   ,pl,color='red',label='UP LW',linestyle='dashed',lw=1)

    ax.plot(-1.0*atm_fl_D      ,pl,color='green',label='DN',lw=2)
    ax.plot(-1.0*atm_fl_D_SW   ,pl,color='green',label='DN SW',linestyle='dotted',lw=3)
    ax.plot(-1.0*atm_fl_D_LW   ,pl,color='green',label='DN LW',linestyle='dashed',lw=2)

    ax.plot(atm_fl_N ,pl,color='black',label='NET')

    ax.set_xscale("symlog")
    ax.set_xlabel("Upward-directed flux [W m-2]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale("log")
    ax.set_ylim([pl[-1],pl[0]])

    ax.legend(loc='upper left')

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_fluxes_atmosphere.%s"%plot_format,
                bbox_inches='tight', dpi=200)

if __name__ == '__main__':

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg'

    # Read in COUPLER input file
    from utils.coupler import ReadInitFile, SetDirectories
    OPTIONS = ReadInitFile( cfg )

    # Set directories dictionary
    dirs = SetDirectories(OPTIONS)

    plot_fluxes_atmosphere(dirs["output"], plot_format=OPTIONS["plot_format"])
