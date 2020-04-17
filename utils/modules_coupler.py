# https://realpython.com/python-modules-packages/#the-module-search-path
# https://docs.python.org/3/tutorial/modules.html

# Standard packages
import argparse
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os, sys, glob, shutil
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.transforms as transforms
import mmap
import seaborn as sns
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
from natsort import natsorted # https://pypi.python.org/pypi/natsort
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
if output_dir == coupler_dir: output_dir  = output_dir+"/output/"

# Project main directories
dirs = { "rad_conv": radconv_dir, "output": output_dir, "vulcan": vulcan_dir, "coupler": coupler_dir}

# Output dir, optional argument: https://towardsdatascience.com/learn-enough-python-to-be-useful-argparse-e482e1764e05
parser = argparse.ArgumentParser(description='Define file output directory.')
parser.add_argument('--dir', default="output/", help='Provide path to output directory.' )

# Handle optional command line arguments for volatiles
parser = argparse.ArgumentParser(description='COUPLER optional command line arguments')
parser.add_argument('-dir', default="output/", help='Provide path to output directory.' )
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