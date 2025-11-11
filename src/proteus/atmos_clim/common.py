# Common atmosphere climate model functions
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from proteus.utils.helper import find_nearest

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

# Atmosphere structure class
class Atmos_t():
    def __init__(self):
        # Atmosphere object internal to JANUS or AGNI
        self._atm = None

        # Albedo lookup object
        self.albedo_o:Albedo_t = None

def ncdf_flag_to_bool(var)->bool:
    '''Convert NetCDF flag (y/n) to Python bool (true/false)'''
    v = str(var[0].tobytes().decode()).lower()

    # check against expected
    if v == 'y':
        return True
    elif v == 'n':
        return False
    else:
        raise ValueError(f"Could not parse NetCDF atmos flag variable \n {var}")

def read_ncdf_profile(nc_fpath:str, extra_keys:list=[]):
    """Read data from atmosphere NetCDF output file.

    Automatically reads pressure (p), temperature (t), radius (z) arrays with
    cell-centre (N) and cell-edge (N+1) values interleaved into a single combined array of
    length (2*N+1).

    Extra keys can be read-in using the extra_keys parameter. These will be stored with
    the same dimensions as in the NetCDF file.

    Parameters
    ----------
        nc_fpath : str
            Path to NetCDF file.

        extra_keys : list
            List of extra keys (strings) to read from the file.

    Returns
    ----------
        out : dict
            Dictionary containing numpy arrays of data from the file.
    """

    # open file
    if not os.path.isfile(nc_fpath):
        log.error(f"Could not find NetCDF file '{nc_fpath}'")
        return None
    ds = nc.Dataset(nc_fpath)

    p = np.array(ds.variables["p"][:])
    pl = np.array(ds.variables["pl"][:])

    t = np.array(ds.variables["tmp"][:])
    tl = np.array(ds.variables["tmpl"][:])

    rp = float(ds.variables["planet_radius"][0])
    if "z" in ds.variables.keys():
        # probably from JANUS, which stores heights
        z  = np.array(ds.variables["z"][:])
        zl = np.array(ds.variables["zl"][:])
        r  = np.array(z) + rp
        rl = np.array(zl) + rp
    else:
        # probably from AGNI, which stores radii
        r  = np.array(ds.variables["r"][:])
        rl = np.array(ds.variables["rl"][:])
        z  = np.array(r) - rp
        zl = np.array(rl) - rp

    nlev_c = len(p)

    # read pressure, temperature, height data into dictionary values
    out = {}
    out["p"] = [pl[0]]
    out["t"] = [tl[0]]
    out["z"] = [zl[0]]
    out["r"] = [rl[0]]
    for i in range(nlev_c):
        out["p"].append(p[i])
        out["p"].append(pl[i+1])

        out["t"].append(t[i])
        out["t"].append(tl[i+1])

        out["z"].append(z[i])
        out["z"].append(zl[i+1])

        out["r"].append(r[i])
        out["r"].append(rl[i+1])

    # flags
    for fk in ("transparent", "solved", "converged"):
        if fk in ds.variables.keys():
            out[fk] = ncdf_flag_to_bool(ds.variables[fk])
        else:
            out[fk] = False # if not available

    # Read extra keys
    for key in extra_keys:

        # Check that key exists
        if key not in ds.variables.keys():
            log.error(f"Could not read '{key}' from NetCDF file")
            continue

        # Reading composition
        if key == "x_gas":

            gas_l = ds.variables["gases"][:] # names (bytes matrix)
            gas_x = ds.variables["x_gas"][:] # vmrs (float matrix)

            # get data for each gas
            for igas,gas in enumerate(gas_l):
                gas_lbl = "".join(  [c.decode(encoding="utf-8") for c in gas] ).strip()
                out[gas_lbl+"_vmr"] = np.array(gas_x[:,igas])

        else:
            out[key] = np.array(ds.variables[key][:])

    # close file
    ds.close()

    # convert to np arrays
    for key in out.keys():
        out[key] = np.array(out[key], dtype=float)

    return out

def read_atmosphere_data(output_dir:str, times:list, extra_keys=[]):
    """
    Read all p,t,z profiles from NetCDF files in a PROTEUS output folder.
    """
    profiles = [read_ncdf_profile(os.path.join(output_dir, "data", "%.0f_atm.nc"%t),
                                    extra_keys=extra_keys) for t in times]
    if None in profiles:
        log.warning("One or more NetCDF files could not be found")
        if os.path.exists(os.path.join(output_dir,"data","data.tar")):
            log.warning("You may need to extract archived data files")
        return

    return profiles

def get_spfile_name_and_bands(config:Config):
    """
    Get spectral file name and bands from config
    """

    # Get table corresponding to the right atmosphere module
    obj = getattr(config.atmos_clim, config.atmos_clim.module)

    # Get bands and group name (strings)
    bands = obj.spectral_bands
    group = obj.spectral_group

    return group, bands


def get_spfile_path(fwl_dir:str, config:Config):
    """
    Get path to spectral file, given name and bands.
    """

    # Get group and bands (strings) from config
    group, bands = get_spfile_name_and_bands(config)

    # Construct file path
    return os.path.join(fwl_dir,"spectral_files",group,bands,group)+".sf"

def get_radius_from_pressure(p_arr, r_arr, p_tgt):
    """
    Get the geometric radius corresponding to a given pressure.

    Parameters:
    ----------------
        p_arr: list
            Pressure array
        r_arr: list
            Radius array
        p_tgt: float
            Target pressure

    Returns:
    ----------------
        p_close: float
            Closest pressure in the array
        r_close: float
            Closest radius in the array
    """

    p_close, idx = find_nearest(p_arr, p_tgt)
    return float(p_close), float(r_arr[idx])


class Albedo_t():
    """
    Store and evaluate bond albedo as a function of other variables.
    """

    def __init__(self, csvfile:str):

        # Data table
        self._data:pd.DataFrame = None
        self.ok = False

        # Interpolator
        self._lims:dict = {}  # axis limits
        self._interp = None

        # Read data file
        #   tmp     ->   temperature [K]
        #   albedo  ->   bond albedo [-]
        if os.path.isfile(csvfile):
            try:
                self._data = pd.read_csv(csvfile, dtype=float)
            except Exception:
                log.error(f"Could not parse lookup data from file '{csvfile}'")
                self._data = None
                return
        else:
            log.error(f"Could not find file '{csvfile}'")
            return

        # Check that file has required keys
        for k in ("tmp", "albedo"):
            if k not in self._data.keys():
                log.error(f"Albedo lookup data does not have required key '{k}'")
                log.error(f"    File: {csvfile}")
                return

        # Store axis limits
        for k in self._data.keys():
            self._lims[k] = (np.amin(self._data[k]), np.amax(self._data[k]))

        # Process data by interpolation
        self.ok = True
        self._interp = PchipInterpolator(self._data["tmp"], self._data["albedo"])

    def evaluate(self, tmp:float) -> float:
        """
        Evalulate bond albedo at a given temperature [K]
        """
        if self._interp or not self.ok:

            # Ensure valid range on input parameters
            tmp = min(max(tmp, self._lims["tmp"][0]), self._lims["tmp"][1])

            # Evaluate albedo
            alb = float(self._interp(tmp))

            # Ensure valid range on output albedo
            if not (0 < alb < 1):
                log.warning(f"Interpolated `albedo_pl` is out of range: {alb}")
            return min(max(alb, 0.0), 1.0)

        else:
            log.error("Cannot evaluate bond albedo. Lookup data not loaded!")
            return None
