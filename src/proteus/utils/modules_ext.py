# External system modules (not submodules)

# https://realpython.com/python-modules-packages/#the-module-search-path
# https://docs.python.org/3/tutorial/modules.html

# System and file management
import argparse
import logging
import pathlib
import json
import subprocess
import os, sys, glob, shutil, re
from datetime import datetime
import copy
import warnings

# Plotting
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.size': 12.0})
logging.getLogger('matplotlib.font_manager').disabled = True  # Disable font fallback logging

import matplotlib.pyplot as plt
plt.ioff()

import matplotlib.ticker as ticker
from cmcrameri import cm 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator, LinearLocator, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.font_manager as fm

# Data and science
import netCDF4 as nc
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


