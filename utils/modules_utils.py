#!/usr/bin/env python3

# https://realpython.com/python-modules-packages/#the-module-search-path
# https://docs.python.org/3/tutorial/modules.html

# Standard packages
import argparse
import logging

import matplotlib as mpl
mpl.use('Agg')

logging.getLogger('matplotlib.font_manager').disabled = True  # Disable font fallback logging

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os, sys, glob, shutil, re
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.transforms as transforms
import mmap
import pathlib
import errno
import json
import subprocess
import fileinput # https://kaijento.github.io/2017/05/28/python-replacing-lines-in-file/
import math
import importlib.util
import pickle as pkl
import requests
from astropy.io import fits
from astropy.table import Table

import Mors as mors

from datetime import datetime
from scipy import interpolate
# from natsort import natsorted # https://pypi.python.org/pypi/natsort
from decimal import Decimal
from scipy.integrate import solve_ivp
from scipy.integrate import fixed_quad
from scipy import stats

# Import coupler-specific package modules
#import utils_coupler as cu
import utils.utils_spider as su

import utils.cpl_atmosphere
import utils.cpl_global
import utils.cpl_stacked
import utils.cpl_interior

from AEOLUS import SocRadConv as SocRadConv
from AEOLUS.utils.atmosphere_column import atmos
from AEOLUS.utils import phys as phys

### Constants ###

debug = True

# Astronomical constants
L_sun           = 3.828e+26             # W, IAU definition
AU              = 1.495978707e+11       # m
R_gas           = 8.31446261815324      # J K−1 mol−1
M_earth         = 5.972E24              # kg
R_core_earth    = 3485000.0             # m
M_core_earth    = 1.94E24               # kg
mol             = 6.02214076e+23        # mol definition

# Elements
H_mol_mass      = 0.001008              # kg mol−1
C_mol_mass      = 0.012011              # kg mol−1
O_mol_mass      = 0.015999              # kg mol−1
N_mol_mass      = 0.014007              # kg mol−1
S_mol_mass      = 0.03206               # kg mol−1
He_mol_mass     = 0.0040026             # kg mol−1
Ar_mol_mass     = 0.039948              # kg mol−1
Ne_mol_mass     = 0.020180              # kg mol−1
Kr_mol_mass     = 0.083798              # kg mol−1
Xe_mol_mass     = 0.131293              # kg mol−1

# Volatile molar masses
H2O_mol_mass    = 0.01801528            # kg mol−1
CO2_mol_mass    = 0.04401               # kg mol−1
H2_mol_mass     = 0.00201588            # kg mol−1
CH4_mol_mass    = 0.01604               # kg mol−1
CO_mol_mass     = 0.02801               # kg mol−1
N2_mol_mass     = 0.028014              # kg mol−1
O2_mol_mass     = 0.031999              # kg mol−1
SO2_mol_mass    = 0.064066              # kg mol−1
H2S_mol_mass    = 0.0341                # kg mol−1

molar_mass      = {
          "H2O" : 0.01801528,           # kg mol−1
          "CO2" : 0.04401,              # kg mol−1
          "H2"  : 0.00201588,           # kg mol−1
          "CH4" : 0.01604,              # kg mol−1
          "CO"  : 0.02801,              # kg mol−1
          "N2"  : 0.028014,             # kg mol−1
          "O2"  : 0.031999,             # kg mol−1
          "SO2" : 0.064066,             # kg mol−1
          "H2S" : 0.0341,               # kg mol−1 
          "H"   : 0.001008,             # kg mol−1 
          "C"   : 0.012011,             # kg mol−1 
          "O"   : 0.015999,             # kg mol−1 
          "N"   : 0.014007,             # kg mol−1 
          "S"   : 0.03206,              # kg mol−1 
          "He"  : 0.0040026,            # kg mol−1 
          "NH3" : 0.017031,             # kg mol−1 
        }

volatile_species = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2", "S", "He" ]
element_list     = [ "H", "O", "C", "N", "S", "He" ] 

# volatile_species = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2"]
# element_list     = [ "H", "O", "C", "N" ] 
