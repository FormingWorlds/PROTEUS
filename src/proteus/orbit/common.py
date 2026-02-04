# Common tides model functions
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger("fwl."+__name__)

# Tides structure class
class Tides_t():
    def __init__(self):
        # Obliqua outputs
        self.sigma: Optional[NDArray[np.floating]] = None
        self.Imk2:  Optional[NDArray[np.floating]] = None


def read_ncdf_profile(nc_fpath:str):
    """Read data from tides NetCDF output file.

    Automatically reads forcing frequency (σ) and imaginary part of k2 Love number (Imk2) arrays.

    Parameters
    ----------
        nc_fpath : str
            Path to NetCDF file.

    Returns
    ----------
        out : dict
            Dictionary containing numpy arrays of data from the file.
    """

    import netCDF4 as nc

    # open file
    if not os.path.isfile(nc_fpath):
        log.error(f"Could not find NetCDF file '{nc_fpath}'")
        return None
    ds = nc.Dataset(nc_fpath)

    σ    = np.array(ds.variables["σ"][:])
    Imk2 = np.array(ds.variables["Imk2"][:])

    # read data into dictionary values
    out = {}
    out["sigma"] = σ
    out["Imk2"]  = Imk2

    # close file
    ds.close()

    # convert to np arrays
    for key in out.keys():
        out[key] = np.array(out[key], dtype=float)

    return out

def read_orbit_data(output_dir:str, times:list):
    """
    Read all Imk2 spectra from NetCDF files in a PROTEUS output folder.
    """
    profiles = [
        read_ncdf_profile(os.path.join(output_dir, "data", "%.0f_orb.nc"%t))
        for t in times
    ]
    if None in profiles:
        log.warning("One or more NetCDF files could not be found")
        if os.path.exists(os.path.join(output_dir,"data","data.tar")):
            log.warning("You may need to extract archived data files")
        return

    return profiles
