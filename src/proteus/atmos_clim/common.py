# Common atmosphere climate model functions
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import netCDF4 as nc
import numpy as np

from proteus.utils.helper import find_nearest

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def read_ncdf_profile(nc_fpath:str, extra_keys:list=[]):
    """Read data from atmosphere NetCDF output file.

    Automatically reads pressure (p), temperature (t), height (z) arrays with
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
    ds = nc.Dataset(nc_fpath)

    p = np.array(ds.variables["p"][:])
    pl = np.array(ds.variables["pl"][:])

    t = np.array(ds.variables["tmp"][:])
    tl = np.array(ds.variables["tmpl"][:])

    z = np.array(ds.variables["z"][:])
    zl = np.array(ds.variables["zl"][:])

    nlev_c = len(p)

    # read pressure, temperature, height data into dictionary values
    out = {}
    out["p"] = [pl[0]]
    out["t"] = [tl[0]]
    out["z"] = [zl[0]]
    for i in range(nlev_c):
        out["p"].append(p[i])
        out["p"].append(pl[i+1])

        out["t"].append(t[i])
        out["t"].append(tl[i+1])

        out["z"].append(z[i])
        out["z"].append(zl[i+1])

    # Read extra keys
    for key in extra_keys:
        if key in ds.variables.keys():
            out[key] = np.array(ds.variables[key][:])

    # close file
    ds.close()

    # convert to np arrays
    for key in out.keys():
        out[key] = np.array(out[key], dtype=float)

    return out

def read_atmosphere_data(output_dir:str, times:list):
    """
    Read all p,t,z profiles from NetCDF files in a PROTEUS output folder.
    """
    profiles = [read_ncdf_profile(os.path.join(output_dir, "data", "%d_atm.nc"%t)) for t in times]
    if len(profiles) < 1:
        log.warning("No NetCDF files found in output folder")
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

def get_height_from_pressure(p_arr, z_arr, p_tgt):
    """
    Get the geometric height [m] corresponding to a given pressure.

    Parameters:
    ----------------
        p_arr: list
            Pressure array
        z_arr: list
            Height array
        p_tgt: float
            Target pressure

    Returns:
    ----------------
        p_close: float
            Closest pressure in the array
        z_close: float
            Closest height in the array
    """

    p_close, idx = find_nearest(p_arr, p_tgt)
    z_close = z_arr[idx]

    return float(p_close), z_close
