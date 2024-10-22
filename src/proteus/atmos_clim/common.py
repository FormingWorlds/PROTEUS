# Common atmosphere climate model functions
from __future__ import annotations

import os
import logging
import netCDF4 as nc
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def read_ncdf_profile(nc_fpath:str):
    """
    Read temperature, pressure, height data from NetCDF file.
    """

    # open file
    ds = nc.Dataset(nc_fpath)

    p = np.array(ds.variables["p"][:])
    pl = np.array(ds.variables["pl"][:])

    t = np.array(ds.variables["tmp"][:])
    tl = np.array(ds.variables["tmpl"][:])

    z = np.array(ds.variables["z"][:])
    zl = np.array(ds.variables["zl"][:])

    nlev = len(p)

    # read pressure, temperature, height data into dictionary values
    out = {}
    out["p"] = [pl[0]]
    out["t"] = [tl[0]]
    out["z"] = [zl[0]]
    for i in range(nlev):
        out["p"].append(p[i])
        out["p"].append(pl[i+1])

        out["t"].append(t[i])
        out["t"].append(tl[i+1])

        out["z"].append(z[i])
        out["z"].append(zl[i+1])

    # close file
    ds.close()

    # convert to np arrays
    for k in out.keys():
        out[k] = np.array(out[k], dtype=float)

    return out

def read_ncdfs(output_dir:str, times:list):
    """
    Read all p,t,z profiles from NetCDF files in a PROTEUS output folder.
    """
    profiles = [read_ncdf_profile(os.path.join(output_dir, "data", "%d_atm.nc"%t)) for t in times]
    if len(profiles) < 1:
        log.warning("No NetCDF files found in output folder")
    return profiles

def get_spfile_path(dirs:dict, config:Config):
    """
    Get path to spectral file, given name and bands.
    """

    # Get table corresponding to the right atmosphere module
    obj = getattr(config.atmos_clim, config.atmos_clim.module)

    # Get bands and group name (strings)
    bands = obj.spectral_bands
    group = obj.spectral_group

    # Construct file path
    return os.path.join(dirs["fwl"],"spectral_files",group,bands,group)+".sf"

