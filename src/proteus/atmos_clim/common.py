# Common atmosphere climate model functions
from __future__ import annotations

import os

import netCDF4 as nc
import numpy as np


def read_ncdf_profile(nc_fpath:str):
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
    return [read_ncdf_profile(os.path.join(output_dir, "data", "%d_atm.nc"%t)) for t in times]
