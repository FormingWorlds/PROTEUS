# Function and classes used to run SPIDER
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import os

from proteus.utils.constants import gas_list

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

PLATON_DOWNSAMPLE=4
PLATON_METHOD="xsec"
PLATON_GASES=["H2", "H", "He", "H2O", "CH4", "CO", "CO2", "O", "C", "N", "NH3", "N2",
                "O2", "O3", "H2S", "HCN", "NO", "NO2", "OH", "PH3", "SiO", "SO2", "TiO",
                "VO", "Na", "K", "Ca", "Ti", "Fe", "Ni", "C2H2", "FeH"]

def _get_atm(outdir:str, time:int):
    from proteus.atmos_clim.common import read_atmosphere_data
    return read_atmosphere_data(outdir, [time])[-1]

def _get_abund(hf_row:dict):
    vmr_incl = []
    gas_incl = []
    # for all gases...
    for gas in gas_list:
        vmr = hf_row[gas+"_vmr"]
        # neglect unsupported and trace gases
        if (gas in PLATON_GASES) and (vmr > 1e-8):
            vmr_incl.append(vmr)
            gas_incl.append(gas)
    return gas_incl, vmr_incl

def _get_pt(atm:dict):
    prs = np.array(atm["p"]) # Pa
    tmp = np.array(atm["t"]) # K
    if prs[1] < prs[0]:
        prs = prs[::-1]
        tmp = tmp[::-1]
    return prs, tmp

def _get_prof(atm:dict, prs, tmp):
    from platon.TP_profile import Profile
    prof = Profile()
    prof.set_from_arrays(prs, tmp)
    return prof

def transit_depth(hf_row:dict, outdir:str):
    from platon.transit_depth_calculator import TransitDepthCalculator
    from proteus.observe.common import get_transit_fpath

    # All planet quantities in SI
    Rs = hf_row["R_star"]     # Radius of star [m]
    Mp = hf_row["M_tot"]      # Mass of planet [kg]
    Rp = hf_row["R_int"]      # Radius of planet [m]

    # Get profile
    atm = _get_atm(outdir, hf_row["Time"])
    prs, tmp = _get_pt(atm)
    prf = _get_prof(atm, prs, tmp)
    gas,vmr = _get_abund(hf_row)

    # create a TransitDepthCalculator object
    transcalc = TransitDepthCalculator(include_opacities=gas,
                                                    method=PLATON_METHOD,
                                                    downsample=PLATON_DOWNSAMPLE)

    # compute spectrum
    log.debug("Calculate transit depth spectrum")
    wl, de, _ = transcalc.compute_depths(Rs, Mp, Rp, None,
                                        logZ=None, CO_ratio=None,
                                        gases=gas, vmrs=vmr,
                                        custom_T_profile=tmp,
                                        custom_P_profile=prs,
                                        T_star=hf_row["T_star"])
    wl = np.array(wl) * 1e6 # convert to um
    de = np.array(de) * 1e6 # convert to ppm

    # write file
    log.debug("Writing transit depth spectrum")
    header = "Wavelength [um], transit depth [ppm]"
    np.savetxt(get_transit_fpath(outdir), np.array([wl, de]).T, fmt="%.6e", header=header)


def eclipse_depth(hf_row:dict, outdir:str):
    from platon.eclipse_depth_calculator import EclipseDepthCalculator
    from proteus.observe.common import get_eclipse_fpath


    # All planet quantities in SI
    Rs = hf_row["R_star"]     # Radius of star [m]
    Mp = hf_row["M_tot"]      # Mass of planet [kg]
    Rp = hf_row["R_int"]      # Radius of planet [m]
    Ts = hf_row["T_star"]     # Stellar temperature

    # Get profile
    atm = _get_atm(outdir, hf_row["Time"])
    prs, tmp = _get_pt(atm)
    prf = _get_prof(atm, prs, tmp)
    gas,vmr = _get_abund(hf_row)

    # create a EclipseDepthCalculator object
    eclipcalc = EclipseDepthCalculator(include_opacities=gas,
                                                    method=PLATON_METHOD,
                                                    downsample=PLATON_DOWNSAMPLE)

    # compute spectrum
    log.debug("Calculate eclipse depth spectrum")
    wl, de, _ = eclipcalc.compute_depths(prf, Rs, Mp, Rp, Ts,
                                                logZ=None, CO_ratio=None,
                                                gases=gas, vmrs=vmr,)
    wl = np.array(wl) * 1e6 # convert to um
    de = np.array(de) * 1e6 # convert to ppm

    # write file
    log.debug("Writing eclipse depth spectrum")
    header = "Wavelength [um], eclipse depth [ppm]"
    np.savetxt(get_eclipse_fpath(outdir), np.array([wl, de]).T, fmt="%.6e", header=header)
