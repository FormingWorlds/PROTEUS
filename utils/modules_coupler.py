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
#import yaml
import pickle as pkl

from datetime import datetime
from scipy import interpolate
# from natsort import natsorted # https://pypi.python.org/pypi/natsort
from decimal import Decimal
from scipy.integrate import solve_ivp
from scipy import stats

# Import coupler-specific package modules
import utils.utils_coupler as cu
import utils.utils_spider as su

import AEOLUS.SocRadConv as SocRadConv

# Coupler-specific paths
coupler_dir = os.getenv('COUPLER_DIR')
output_dir  = coupler_dir+"/output/"
vulcan_dir  = coupler_dir+"/VULCAN/"
radconv_dir = coupler_dir+"/AEOLUS/"
spider_dir  = coupler_dir+"/SPIDER/"
utils_dir   = coupler_dir+"/utils/"

dirs = {"output": output_dir, 
        "coupler": coupler_dir, 
        "rad_conv": radconv_dir, 
        "vulcan": vulcan_dir, 
        "spider": spider_dir, 
        "utils": utils_dir
        }

# Check if output directory exists, otherwise create
if not os.path.exists(dirs["output"]):
    os.makedirs(dirs["output"])
    print("--> Create output directory:", dirs["output"])


# Handle optional command line arguments for volatiles
# Optional arguments: https://towardsdatascience.com/learn-enough-python-to-be-useful-argparse-e482e1764e05
def parse_console_arguments():
    
    parser = argparse.ArgumentParser(description='COUPLER optional command line arguments')
    parser.add_argument('-init_file', type=str, default=dirs["output"]+"/init_coupler.cfg", help='Specify init filename')
    parser.add_argument('-restart_file', type=str, default="0", help='Restart from specific .json file in folder. Specify only the number of the file.')
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
    # parser.add_argument('-rf', '--restart_file', type=str, help='Restart from specific .json file in folder. Specify only the number of the file.')
    parser.add_argument('-r', '--restart', action='store_true', help='Restart from last file in folder.')
    args = parser.parse_args()

    return args


