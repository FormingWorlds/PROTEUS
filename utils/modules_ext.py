# External system modules (not submodules)

# https://realpython.com/python-modules-packages/#the-module-search-path
# https://docs.python.org/3/tutorial/modules.html

# System and file management
import importlib.util
import argparse
import logging
import mmap
import pathlib
import errno
import json
import subprocess
import fileinput
import os, sys, glob, shutil, re
from datetime import datetime

# Plotting
import matplotlib as mpl
mpl.use('Agg')
logging.getLogger('matplotlib.font_manager').disabled = True  # Disable font fallback logging

import matplotlib.pyplot as plt
plt.ioff()

import matplotlib.ticker as ticker
from matplotlib import cm
import matplotlib.transforms as transforms
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm


# Data and science
import numpy as np
import pandas as pd
import math
import pickle as pkl
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy import interpolate
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.optimize import newton
from scipy.optimize import fsolve
from scipy import stats


