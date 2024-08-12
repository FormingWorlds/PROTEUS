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
def plot_fluxes_global(output_dir, OPTIONS, t0=100.0):

    log.info("Plot global fluxes")

    # Get values
    hf_all = pd.read_csv(output_dir+"/runtime_helpfile.csv", sep=r"\s+")
    hf_all = hf_all.loc[hf_all["Time"]>t0]

    time = np.array(hf_all["Time"] )

    F_net = np.array(hf_all["F_atm"])
    F_asf = np.array(hf_all["F_ins"]) * OPTIONS["asf_scalefactor"] * (1.0 - OPTIONS["albedo_pl"]) * np.cos(OPTIONS["zenith_angle"] * np.pi/180.0)
    F_olr = np.array(hf_all["F_olr"])
    F_upw = np.array(hf_all["F_olr"]) + np.array(hf_all["F_sct"])
    F_int = np.array(hf_all["F_int"])

    if len(time) < 3:
        log.warning("Cannot make plot with less than 3 samples")
        return

    # Create plot
    mpl.use('Agg')
    scale = 1.2
    fig,ax = plt.subplots(1,1,figsize=(6*scale,5*scale))
    lw = 2.0
    al = 0.96

    # Steam runaway line
    ax.axhline(y=280.0, color='black', lw=lw, linestyle='dashed', label="S-N limit", zorder=1)

    # Plot fluxes
    ax.plot(time, F_int, lw=lw, alpha=al, zorder=2, color=dict_colors["int"], label="Net (int.)")
    ax.plot(time, F_net, lw=lw, alpha=al, zorder=2, color=dict_colors["atm"], label="Net (atm.)")
    ax.plot(time, F_olr, lw=lw, alpha=al, zorder=2, color=dict_colors["OLR"], label="OLR")
    ax.plot(time, F_upw, lw=lw, alpha=al, zorder=3, color=dict_colors["sct"], label="OLR + Scat.")
    ax.plot(time, F_asf, lw=lw, alpha=al, zorder=3, color=dict_colors["ASF"], label="ASF", linestyle='dotted')

    # Configure plot
    ax.set_yscale("symlog")
    ax.set_ylabel("Unsigned flux [W m$^{-2}$]")
    ax.set_xscale("log")
    ax.set_xlabel("Time [yr]")
    ax.set_xlim(time[0], time[-1])
    legend = ax.legend(loc='lower left')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.99))
    ax.grid(color='black', alpha=0.05)

    plt.close()
    plt.ioff()
    fig.savefig(output_dir+"/plot_fluxes_global.%s"%OPTIONS["plot_format"],
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

    plot_fluxes_global(dirs["output"], OPTIONS)
