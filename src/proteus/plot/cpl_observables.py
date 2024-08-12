#!/usr/bin/env python3

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

def plot_observables( output_dir, plot_format="pdf", t0=100.0):

    log.info("Plot observables")

    hf_all = pd.read_csv(output_dir+"/runtime_helpfile.csv", sep=r"\s+")

    time = np.array(hf_all["Time"] )
    if len(time) < 3:
        log.warning("Cannot make plot with less than 3 samples")
        return

    # make plot
    lw = 1.2
    scale = 1.1
    fig,axl = plt.subplots(1,1, figsize=(7*scale,4*scale))

    # left axis
    axl.plot(time, hf_all["transit_depth"]*1e6, lw=lw, color='k')
    axl.set_ylabel("Transit depth [ppm]")

    # right axis
    axr = axl.twinx()
    color = "tab:red"
    axr.plot(time, hf_all["contrast_ratio"]*1e9, lw=lw, color=color)
    axr.set_ylabel("Contrast ratio [ppb]")
    axr.yaxis.label.set_color(color)
    axr.tick_params(axis='y', colors=color)

    # x-axis
    axl.set_xlabel("Time [yr]")
    axl.set_xscale("log")
    axl.set_xlim(left=t0, right=np.amax(time))
    axl.grid(alpha=0.2)

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plot_observables.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')

#====================================================================
def main():

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg'

    # Read in COUPLER input file
    log.info("Read cfg file")
    from utils.coupler import ReadInitFile, SetDirectories
    OPTIONS = ReadInitFile( cfg )

    # Set directories dictionary
    dirs = SetDirectories(OPTIONS)

    plot_observables( output_dir=dirs["output"], plot_format=OPTIONS["plot_format"] )

#====================================================================

if __name__ == "__main__":

    main()
