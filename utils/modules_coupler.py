#!/usr/bin/env python3

# https://realpython.com/python-modules-packages/#the-module-search-path
# https://docs.python.org/3/tutorial/modules.html

# Standard packages
import argparse
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os, sys, glob, shutil, re
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.transforms as transforms
import mmap
# import seaborn as sns
import pathlib
import errno
import json
import subprocess
import fileinput # https://kaijento.github.io/2017/05/28/python-replacing-lines-in-file/
import math
import importlib.util
import pickle as pkl

from datetime import datetime
from scipy import interpolate
# from natsort import natsorted # https://pypi.python.org/pypi/natsort
from decimal import Decimal

# Import coupler-specific package modules
import utils.utils_coupler as cu
import utils.utils_spider as su

import atm_rad_conv.SocRadConv
import atm_rad_conv.SocRadModel
import atm_rad_conv.GeneralAdiabat
from atm_rad_conv.atmosphere_column import atmos

# Coupler-specific paths
coupler_dir = str(pathlib.Path(__file__).parent.absolute())+"/.."
output_dir  = str(pathlib.Path().absolute())
vulcan_dir  = coupler_dir+"/vulcan_spider/"
radconv_dir = coupler_dir+"/atm_rad_conv/"
spider_dir  = coupler_dir+"/spider-dev/"

# If COUPLER executed from main repository dir, put output in subdirectory
if output_dir == coupler_dir: 
    output_dir  = coupler_dir+"/output/"
    
    # # Check if output directory exists, otherwise create
    # if not os.path.exists(dirs["output"]):
    #     os.makedirs(dirs["output"])
    #     print("--> Create data directory:", dirs["output"])

# Project main directories
dirs = { "output": output_dir, "coupler": coupler_dir, "rad_conv": radconv_dir, "vulcan": vulcan_dir, "spider": spider_dir}

# # Output dir, 
# parser = argparse.ArgumentParser(description='Define file output directory.')
# parser.add_argument('--dir', default="output/", help='Provide path to output directory.' )

# Handle optional command line arguments for volatiles
# Optional arguments: https://towardsdatascience.com/learn-enough-python-to-be-useful-argparse-e482e1764e05
parser = argparse.ArgumentParser(description='COUPLER optional command line arguments')
parser.add_argument('-init_file', type=str, default=dirs["output"]+"/init_coupler.opts", help='Specify init filename')
parser.add_argument('-dir', default=dirs["output"], help='Provide path to output directory.' )
parser.add_argument('-H2O_ppm', type=float, help='H2O initial abundance (ppm wt).')
parser.add_argument('-CO2_ppm', type=float, help='CO2 initial abundance (ppm wt).')
parser.add_argument('-H2_ppm', type=float, help='H2 initial abundance (ppm wt).')
parser.add_argument('-CO_ppm', type=float, help='CO initial abundance (ppm wt).')
parser.add_argument('-N2_ppm', type=float, help='N2 initial abundance (ppm wt).')
parser.add_argument('-CH4_ppm', type=float, help='CH4 initial abundance (ppm wt).')
parser.add_argument('-O2_ppm', type=float, help='O2 initial abundance (ppm wt).')
parser.add_argument('-H2O_bar', type=float, help='H2O initial abundance (bar).')
parser.add_argument('-CO2_bar', type=float, help='CO2 initial abundance (bar).')
parser.add_argument('-H2_bar', type=float, help='H2 initial abundance (bar).')
parser.add_argument('-CO_bar', type=float, help='CO initial abundance (bar).')
parser.add_argument('-N2_bar', type=float, help='N2 initial abundance (bar).')
parser.add_argument('-CH4_bar', type=float, help='CH4 initial abundance (bar).')
parser.add_argument('-O2_bar', type=float, help='O2 initial abundance (bar).')
args = parser.parse_args()

# Parse optional argument from console
if args.H2O_ppm:
    print("Set H2O_ppm from command line:", args.H2O_ppm)
    COUPLER_options["H2O_initial_total_abundance"] = float(args.H2O_ppm)
elif args.H2O_bar:
    print("Set H2O_bar from command line:", args.H2O_bar)
    COUPLER_options["H2O_initial_atmos_pressure"] = float(args.H2O_bar)
if args.CO2_ppm:
    print("Set CO2_ppm from command line:", args.CO2_ppm)
    COUPLER_options["CO2_initial_total_abundance"] = float(args.CO2_ppm)
elif args.CO2_bar:
    print("Set CO2_bar from command line:", args.CO2_bar)
    COUPLER_options["CO2_initial_atmos_pressure"] = float(args.CO2_bar)
if args.H2_ppm:
    print("Set H2_ppm from command line:", args.H2_ppm)
    COUPLER_options["H2_initial_total_abundance"] = float(args.H2_ppm)
elif args.H2_bar:
    print("Set H2_bar from command line:", args.H2_bar)
    COUPLER_options["H2_initial_atmos_pressure"] = float(args.H2_bar)
if args.CO_ppm:
    print("Set CO_ppm from command line:", args.CO_ppm)
    COUPLER_options["CO_initial_total_abundance"] = float(args.CO_ppm)
elif args.CO_bar:
    print("Set CO_bar from command line:", args.CO_bar)
    COUPLER_options["CO_initial_atmos_pressure"] = float(args.CO_bar)
if args.N2_ppm:
    print("Set N2_ppm from command line:", args.N2_ppm)
    COUPLER_options["N2_initial_total_abundance"] = float(args.N2_ppm)
elif args.N2_bar:
    print("Set N2_bar from command line:", args.N2_bar)
    COUPLER_options["N2_initial_atmos_pressure"] = float(args.N2_bar)
if args.CH4_ppm:
    print("Set CH4_ppm from command line:", args.CH4_ppm)
    COUPLER_options["CH4_initial_total_abundance"] = float(args.CH4_ppm)
elif args.CH4_bar:
    print("Set CH4_bar from command line:", args.CH4_bar)
    COUPLER_options["CH4_initial_atmos_pressure"] = float(args.CH4_bar)
if args.O2_ppm:
    print("Set O2_ppm from command line:", args.O2_ppm)
    COUPLER_options["O2_initial_total_abundance"] = float(args.O2_ppm)
elif args.O2_bar:
    print("Set O2_bar from command line:", args.O2_bar)
    COUPLER_options["O2_initial_atmos_pressure"] = float(args.O2_bar)