# https://realpython.com/python-modules-packages/#the-module-search-path

# print(f'Invoking __init__.py for Coupler {__name__} package.')

# Import standard and coupler-specific modules: 
# from utils.modules_coupler import *

# __all__ = [ 'coupler_modules', ]

# # Standard packages
# import argparse
# import logging
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick
# import numpy as np
# import os, sys, glob, shutil
# import matplotlib.ticker as ticker
# import pandas as pd
# import matplotlib.transforms as transforms
# import mmap
# import seaborn as sns
# import pathlib
# import errno
# import json
# import subprocess
# import fileinput # https://kaijento.github.io/2017/05/28/python-replacing-lines-in-file/
# import math
# import importlib.util

# from datetime import datetime
# from scipy import interpolate
# from natsort import natsorted #https://pypi.python.org/pypi/natsort
# from decimal import Decimal
# from decimal import Decimal

# # Define Crameri colormaps (+ recursive)
# from matplotlib.colors import LinearSegmentedColormap
# for name in [ 'acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc', 'buda',
#            'cork', 'davos', 'devon', 'grayC', 'hawaii', 'imola', 'lajolla',
#            'lapaz', 'lisbon', 'nuuk', 'oleron', 'oslo', 'roma', 'tofino',
#            'tokyo', 'turku', 'vik' ]:
#     file = os.path.join(str(pathlib.Path(__file__).parent.absolute())+"/ScientificColourMaps5/", name + '.txt')
#     cm_data = np.loadtxt(file)
#     vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
#     vars()[name+"_r"] = LinearSegmentedColormap.from_list(name, cm_data[::-1])

# # Import coupler-specific package modules
# import utils.plot_atmosphere
# import utils.plot_global
# import utils.plot_stacked
# import utils.plot_interior
# import utils.utils_coupler as cu
# import utils.utils_spider as su
# import atm_rad_conv.SocRadConv
# import atm_rad_conv.SocRadModel
# import atm_rad_conv.GeneralAdiabat
# from atm_rad_conv.atmosphere_column import atmos

# # Coupler-specific paths
# coupler_dir = str(pathlib.Path(__file__).parent.absolute())
# output_dir  = str(pathlib.Path().absolute())
# vulcan_dir  = coupler_dir+"/vulcan_spider/"
# radconv_dir = coupler_dir+"/atm_rad_conv/"
# if output_dir == coupler_dir: output_dir  = output_dir+"/output/"

