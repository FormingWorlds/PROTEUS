# Function and classes used to run SPIDER
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

PLATON_METHOD="xsec"
PLATON_GASES=("H2", "H", "He", "H2O", "CH4", "CO", "CO2", "O", "C", "N", "NH3", "N2",
                "O2", "O3", "H2S", "HCN", "NO", "NO2", "OH", "PH3", "SiO", "SO2", "TiO",
                "VO", "Na", "K", "Ca", "Ti", "Fe", "Ni", "C2H2", "FeH")

def _get_atm_profile(outdir:str, hf_row:dict) -> dict:
    '''
    Reads the atmosphere data from the NetCDF file produced by the 'atmos_clim' module.
    '''
    from proteus.atmos_clim.common import read_atmosphere_data
    atm_arr = read_atmosphere_data(outdir, [hf_row["Time"]],
                                    extra_keys=["tmpl", "pl", "rl", "x_gas"])

    if (len(atm_arr) == 0) or (atm_arr[-1] is None):
        log.warning(f"Could not read atmosphere data from '{outdir}'")
        return None
    return atm_arr[-1]

def _get_atm_offchem(outdir:str, hf_row:dict, chem_module:str) -> dict:
    '''
    Reads the atmosphere data from the csv file produced by the 'atmos_chem' module.
    '''

    # Read file
    from proteus.atmos_chem.wrapper import read_result
    df = read_result(outdir, chem_module)

    # Check file exists
    if df is None:
        log.warning(f"Could not read offchem file from '{outdir}'")
        return None

    df.rename(columns={"tmp":"tmpl", "p":"pl", "z":"rl"}, inplace=True)
    df["rl"] = df["rl"] + hf_row["R_int"] # convert height to radius
    return df

def _get_mix(hf_row:dict, atm:dict, source:str, clip_vmr:float) -> tuple:
    '''
    Get the gas abundance profiles and names of included gases.

    Parameters
    ---------
    hf_row : dict
        The row of the helpfile for the current time-step.
    atm : dict
        The atmosphere data as a dictionary.
    source : str
        Method for setting the mixing ratios: "outgas", "profile", or "offchem".
    clip_vmr : float
        Minimum VMR for a species to be included in the radiative transfer.
    '''

    # Output arrays
    vmr_incl = []   # list of gas VMR arrays
    gas_incl = []   # list of gas names

    nlev_l = len(atm["pl"])

    # For all potentially supported gases...
    for gas in PLATON_GASES:

        key = gas+"_vmr"
        vmr = np.zeros(nlev_l) # default to zero abundance

        # Read VMR from the source requested
        if source == "outgas":
            # from helpfile -> constant VMR with height
            if key in hf_row:
                vmr = np.ones(nlev_l) * float(hf_row[key])

        elif source == "profile":
            # from NetCDF file
            if key in atm.keys():
                vmr = np.array(atm[key])

        elif source == "offchem":
            # from atmos_chem output file
            if gas in atm.keys():
                vmr = np.array(atm[gas])

        # neglect trace gases
        if np.amax(vmr) >= clip_vmr:
            vmr_incl.append(vmr)
            gas_incl.append(gas)

    return gas_incl, vmr_incl

def _construct_abundances(atm:dict, gas_incl:list, vmr_incl:list) -> dict:
    '''
    Constructs the abundance dictionary for PLATON from the gas names and VMR arrays.

    Parameters
    ----------
    atm : dict
        The atmosphere data as a dictionary.
    gas_incl : list
        The list of gas names.
    vmr_incl : list
        The list of VMR arrays.
    '''

    nlev_l = len(atm["pl"])

    abundances = {}
    for i, gas in enumerate(gas_incl):

        # 1D array of VMR versus height
        arr_1d = vmr_incl[i]

        # Projected onto 2D array of shape (T,P)
        arr_2d = np.zeros((nlev_l, nlev_l))
        for j in range(nlev_l):
            arr_2d[:, j] = arr_1d[:]

        abundances[gas] = arr_2d
    return abundances

def _get_ptr(atm:dict):
    '''
    Returns the pressure, temperature and radius from the atmosphere data.'''
    prs = np.array(atm["pl"]) # Pa
    tmp = np.array(atm["tmpl"]) # K
    rad = np.array(atm["rl"]) # m
    if prs[1] < prs[0]:
        prs = prs[::-1]
        tmp = tmp[::-1]
        rad = rad[::-1]
    return prs, tmp, rad

def _get_prof(atm:dict):
    '''
    Instantiate platon Profile object
    '''
    from platon.TP_profile import Profile

    prof = Profile()
    prs, tmp, rad = _get_ptr(atm)
    prof.set_from_arrays(prs, tmp)

    return prof

def transit_depth(hf_row:dict, outdir:str, config:Config, source:str):
    '''
    Computes the transit depth spectrum using PLATON.

    Parameters
    ----------
    hf_row : dict
        The row of the helpfile for the current time-step.
    outdir : str
        The output directory for the PROTEUS run.
    config : Config
        PROTEUS config object.
    source : str
        Method for setting the mixing ratios: "outgas", "profile", or "offchem".
    '''
    from platon.transit_depth_calculator import TransitDepthCalculator

    from proteus.observe.common import get_transit_fpath

    # All planet quantities in SI
    Rs = hf_row["R_star"]     # Radius of star [m]
    Mp = hf_row["M_tot"]      # Mass of planet [kg]
    Rp = hf_row["R_int"]      # Radius of planet [m]

    # Get profile from the required source
    if source == "offchem":
        atm = _get_atm_offchem(outdir, hf_row, config.atmos_chem.module)
    elif source in ("outgas", "profile"):
        atm = _get_atm_profile(outdir, hf_row)

    # Parse
    if atm is None:
        log.warning(f"Could not read atmosphere data for source '{source}'")
        return None
    prs, tmp, rad = _get_ptr(atm)

    # Get composition from requested source
    gases,vmrs = _get_mix(hf_row, atm, source, config.observe.platon.clip_vmr)

    # Construct the abundance dictionary
    abund = _construct_abundances(atm, gases, vmrs)

    # create a TransitDepthCalculator object
    log.debug("Compute transit depth spectra")
    transcalc = TransitDepthCalculator( include_opacities=gases,
                                        include_condensation=False,
                                        method=PLATON_METHOD,
                                        downsample=config.observe.platon.downsample)

    # arguments
    compute_args = {"logZ":None, "CO_ratio":None,
                    "gases":gases, "vmrs":None,
                    "custom_T_profile":tmp,
                    "custom_P_profile":prs,
                    "custom_abundances":abund,
                    "T_star":hf_row["T_star"]}

    # compute full spectrum
    wl, de, _ = transcalc.compute_depths(Rs, Mp, Rp, None, **compute_args)
    wl = np.array(wl) * 1e6 # convert to um
    de = np.array(de) * 1e6 # convert to ppm
    X = [wl, de]
    header = ""
    header += str("Wavelength/um").ljust(14, ' ') + "\t"
    header += str("None/ppm").ljust(14, ' ') + "\t"

    # loop over removing different gases
    for gas in gases:
        _, de, _ = transcalc.compute_depths(Rs, Mp, Rp, None,
                                            zero_opacities=[gas], **compute_args)
        X.append(np.array(de) * 1e6)
        header += str(f"{gas}/ppm").ljust(14, ' ') + "\t"  # compose header

    # write file
    X = np.array(X).T
    log.debug("Writing transit depth spectrum")
    np.savetxt(get_transit_fpath(outdir, source, "synthesis"), X,
                    delimiter='\t', fmt="%.8e", header=header, comments="")

    return X


def eclipse_depth(hf_row:dict, outdir:str, config:Config, source:str):
    '''
    Computes the eclipse depth spectrum using PLATON.

    Parameters
    ----------
    hf_row : dict
        The row of the helpfile for the current time-step.
    outdir : str
        The output directory for the PROTEUS run.
    config : Config
        PROTEUS config object.
    source : str
        Method for setting the mixing ratios: "outgas", "profile", or "offchem".
    '''
    from platon.eclipse_depth_calculator import EclipseDepthCalculator

    from proteus.observe.common import get_eclipse_fpath


    # All planet quantities in SI
    Rs = hf_row["R_star"]     # Radius of star [m]
    Mp = hf_row["M_tot"]      # Mass of planet [kg]
    Rp = hf_row["R_int"]      # Radius of planet [m]
    Ts = hf_row["T_star"]     # Stellar temperature

    # Get profile from the required source
    if source == "offchem":
        atm = _get_atm_offchem(outdir, hf_row, config.atmos_chem.module)
    elif source in ("outgas", "profile"):
        atm = _get_atm_profile(outdir, hf_row)

    # Parse
    if atm is None:
        log.warning(f"Could not read atmosphere data for source '{source}'")
        return None

    # Convert to platon format
    prf = _get_prof(atm)

    # Get composition from requested source
    gases,vmrs = _get_mix(hf_row, atm, source, config.observe.platon.clip_vmr)

    # Construct the abundance dictionary
    abund = _construct_abundances(atm, gases, vmrs)

    # create a EclipseDepthCalculator object
    log.debug("Compute eclipse depth spectrum")
    eclipcalc = EclipseDepthCalculator( include_opacities=gases,
                                        include_condensation=False,
                                        method=PLATON_METHOD,
                                        downsample=config.observe.platon.downsample)

    # compute full spectrum
    compute_args = {
        "logZ":None, "CO_ratio":None,
        "gases":gases, "vmrs":None,
        "custom_abundances":abund,
    }
    wl, de, _ = eclipcalc.compute_depths(prf, Rs, Mp, Rp, Ts, **compute_args)
    wl = np.array(wl) * 1e6 # convert to um
    de = np.array(de) * 1e6 # convert to ppm
    X = [wl, de]
    header = ""
    header += str("Wavelength/um").ljust(14, ' ') + "\t"
    header += str("None/ppm").ljust(14, ' ') + "\t"

    # loop over removing different gases
    for gas in gases:
        _, de, _ = eclipcalc.compute_depths(prf, Rs, Mp, Rp, Ts,
                                            zero_opacities=[gas], **compute_args)
        X.append(np.array(de) * 1e6)
        header += str(f"{gas}/ppm").ljust(14, ' ') + "\t"  # compose header

    # write file
    X = np.array(X).T
    log.debug("Writing eclipse depth spectra")
    np.savetxt(get_eclipse_fpath(outdir, source, "synthesis"), X,
                    delimiter='\t', fmt="%.8e", header=header, comments="")

    return X
