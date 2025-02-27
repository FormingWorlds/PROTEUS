# Function and classes used to run SPIDER
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import gas_list

if TYPE_CHECKING:
    pass

log = logging.getLogger("fwl."+__name__)

WRITE_FMT="%.7e"
PLATON_DOWNSAMPLE=8
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

def _get_ptr(atm:dict):
    prs = np.array(atm["p"]) # Pa
    tmp = np.array(atm["t"]) # K
    rad = np.array(atm["r"]) # m
    if prs[1] < prs[0]:
        prs = prs[::-1]
        tmp = tmp[::-1]
        rad = rad[::-1]
    return prs, tmp, rad

def _get_prof(atm:dict):
    from platon.TP_profile import Profile

    prof = Profile()
    prs, tmp, rad = _get_ptr(atm)
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
    prs, tmp, rad = _get_ptr(atm)

    # Convert to platon format
    gases,vmrs = _get_abund(hf_row)

    # create a TransitDepthCalculator object
    log.debug("Compute transit depth spectra")
    transcalc = TransitDepthCalculator( include_opacities=gases,
                                        include_condensation=False,
                                        method=PLATON_METHOD,
                                        downsample=PLATON_DOWNSAMPLE)

    # arguments
    compute_args = {"logZ":None, "CO_ratio":None,
                    "gases":gases, "vmrs":vmrs,
                    "custom_T_profile":tmp,
                    "custom_P_profile":prs,
                    "T_star":hf_row["T_star"]}

    # compute full spectrum
    wl, de, _ = transcalc.compute_depths(Rs, Mp, Rp, None, **compute_args)
    wl = np.array(wl) * 1e6 # convert to um
    de = np.array(de) * 1e6 # convert to ppm
    X = [wl, de]
    header = "Wavelength/um,None/ppm"

    # loop over removing different gases
    for gas in gases:
        _, de, _ = transcalc.compute_depths(Rs, Mp, Rp, None,
                                            zero_opacities=[gas], **compute_args)
        X.append(np.array(de) * 1e6)
        header += ",%s/ppm"%gas

    # write file
    log.debug("Writing transit depth spectrum")
    np.savetxt(get_transit_fpath(outdir), np.array(X).T,
                    fmt=WRITE_FMT, header=header, delimiter=',',comments="")


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
    prs, tmp, rad = _get_ptr(atm)

    # Convert to platon format
    prf = _get_prof(atm)
    gases,vmrs = _get_abund(hf_row)

    # create a EclipseDepthCalculator object
    log.debug("Compute eclipse depth spectrum")
    eclipcalc = EclipseDepthCalculator( include_opacities=gases,
                                        include_condensation=False,
                                        method=PLATON_METHOD,
                                        downsample=PLATON_DOWNSAMPLE)

    # compute full spectrum
    compute_args = {
        "logZ":None, "CO_ratio":None,
        "gases":gases, "vmrs":vmrs
    }
    wl, de, _ = eclipcalc.compute_depths(prf, Rs, Mp, Rp, Ts, **compute_args)
    wl = np.array(wl) * 1e6 # convert to um
    de = np.array(de) * 1e6 # convert to ppm
    X = [wl, de]
    header = "Wavelength/um,None/ppm"

    # loop over removing different gases
    for gas in gases:
        _, de, _ = eclipcalc.compute_depths(prf, Rs, Mp, Rp, Ts,
                                            zero_opacities=[gas], **compute_args)
        X.append(np.array(de) * 1e6)
        header += ",%s/ppm"%gas

    # write file
    log.debug("Writing eclipse depth spectra")
    np.savetxt(get_eclipse_fpath(outdir), np.array(X).T,
                    fmt=WRITE_FMT, header=header, delimiter=',',comments="")
