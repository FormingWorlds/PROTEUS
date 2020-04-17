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

from utils.cpl_atmosphere import *
from utils.cpl_global import *
from utils.cpl_stacked import *
from utils.cpl_interior import *

import atm_rad_conv.SocRadConv
import atm_rad_conv.SocRadModel
import atm_rad_conv.GeneralAdiabat
from atm_rad_conv.atmosphere_column import atmos
