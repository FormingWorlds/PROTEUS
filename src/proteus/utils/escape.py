# Functions used to handle escape

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

from proteus.utils.constants import *
from proteus.utils.helper import *

import mors
from proteus.utils.zephyrus import EL_escape

log = logging.getLogger("PROTEUS")


def RunDummyEsc(hf_row:dict, dt:float, phi_bulk:float):
    """Run dummy escape model.

    Parameters
    ----------
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        dt : float
            Time interval over which escape is occuring [yr]
        phi_bulk : float
            Bulk escape rate [kg s-1]

    Returns
    ----------
        esc_result : dict
            Dictionary of updated total elemental mass inventories [kg]

    """
    log.info("Running dummy escape...")

    # store value
    out = {}
    out["rate_bulk"] = phi_bulk

    # calculate total mass of volatiles (except oxygen, which is set by fO2)
    M_vols = 0.0
    for e in element_list:
        if e=='O': continue
        M_vols += hf_row[e+"_kg_total"]


    # for each elem, calculate new total inventory while
    # maintaining a constant mass mixing ratio
    for e in element_list:
        if e=='O': continue

        # current elemental mass ratio in total
        emr = hf_row[e+"_kg_total"]/M_vols

        log.debug("    %s mass ratio = %.2e "%(e,emr))

        # new total mass of element e, keeping a constant mixing ratio of that element
        out[e+"_kg_total"] = emr * (M_vols - phi_bulk * dt * secs_per_year)

    return out


def RunZEPHYRUS(hf_row, dt, M_star,Omega_star,tidal_contribution, semi_major_axis, eccentricity, M_planet, epsilon, R_earth, Rxuv):
    """Run energy-limited escape (for now) model.

    Parameters
    ----------
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        dt : float
            Time interval over which escape is occuring [yr]
        M_star : float
            Stellar mass in solar mass                  [M_sun, kg]
        Omega_star : float
            Stellar rotation rate                       [rad s-1]
        tidal_contribution  : float
            Tidal correction factor (0:None or 1:yes)   [dimensionless]
        semi_major_axis : float
            Planetary semi-major axis                   [m]
        eccentricity : float
            Planetary eccentricity                      [dimensionless]
        M_planet : float
            Planetary mass                              [kg]
        epsilon : float
            Escape efficiency factor                    [dimensionless]
        R_earth : float
            Planetary radius                            [m]
        Rxuv : float
            XUV planetary radius                        [m]

    Returns
    ----------
        out : dict
            Dictionary of updated total elemental mass inventories [kg]
    """

    log.info("Running EL escape (ZEPHYRUS) ...")

    ## Step 1 : Load stellar evolution track + compute EL escape
    log.info("Step 1 : Load stellar evolution track + compute EL escape ")

    # Get the age of the star at time t to compute XUV flux at that time
    age_star        = hf_row["age_star"]                                                # [years]
    star            = mors.Star(Mstar=M_star, Age=age_star/1e6, Omega=Omega_star)       # Load the stellar evolution track from MORS
    age_star_mors   = star.Tracks['Age']                                                # [Myrs]

    # Interpolating the XUV flux at the age of the star
    Fxuv_star_SI_full   = ((star.Tracks['Lx'] + star.Tracks['Leuv']) / (4 * np.pi * (semi_major_axis * 1e2)**2)) * ergcm2stoWm2
    Fxuv_star_SI        = np.interp(age_star, age_star_mors * 1e6, Fxuv_star_SI_full)                                            # Interpolate to get Fxuv at age_star
    log.info(f"Interpolated Fxuv_star_SI at age_star = {age_star} years is {Fxuv_star_SI}")

    # Compute energy-limited escape
    mlr                 = EL_escape(tidal_contribution, semi_major_axis, eccentricity, M_planet, M_star, epsilon, R_earth, Rxuv, Fxuv_star_SI)   # [kg s-1]

    ## Step 2 : Updated total elemental mass inventories
    log.info("Step 2 : Updated total elemental mass inventories")

    # store mass loss rate value
    out = {}
    out["rate_bulk"] = mlr

    # calculate total mass of volatiles (except oxygen, which is set by fO2)
    M_vols = 0.0
    for e in element_list:
        if e=='O': continue
        M_vols += hf_row[e+"_kg_total"]

    # for each elem, calculate new total inventory while
    # maintaining a constant mass mixing ratio

    for e in element_list:
        if e=='O': continue

        # current elemental mass ratio in total
        emr = hf_row[e+"_kg_total"]/M_vols

        log.debug("    %s mass ratio = %.2e "%(e,emr))

        # new total mass of element e, keeping a constant mixing ratio of that element
        out[e+"_kg_total"] = emr * (M_vols - mlr * dt * secs_per_year)

    log.info('Zephyrus escape computation done :)')

    return out
