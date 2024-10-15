# Variables and functions to help with offline chemistry plotting functions
# These do not do the plotting themselves
from __future__ import annotations

import glob
import json
import pickle as pkl

import numpy as np

from proteus.config import read_config


def offchem_read_year(output_dir, year_int, mx_clip_min=1e-30, mx_clip_max=1.0, read_const=False):
    """Read PT profile, VULCAN output file, and optionally runtime mixing ratios.

    Given the path to an offline chemistry output folder, containing year
    subfolders, this function reads data from a given year. It optionally
    reads the mixing ratios calculated by SPIDER at runtime, and also applies
    a clip to the mixing ratios.

    Parameters
    ----------
        output_dir : str
            Path to offline chemistry output folder
        year_int : int
            Year to be read from `output_dir`

        mx_clip_min=1e-30 : float
            Mixing ratio floor
        mx_clip_max=1.0 : float
            Mixing ratio ceiling
        read_const=False : bool
            Read the mixing ratios that were used during the PROTEUS simulation?

    Returns
    ----------
        year_data : dict
            Dictionary of output arrays for the given year.
    """

    year_data = {}  # Dictionary of mixing ratios, pressure, temperature

    # Read VULCAN output file
    with open(output_dir+"offchem/%d/output.vul" % year_int, "rb") as handle:
        vul_data = pkl.load(handle)


    # Save year
    year_data["year"] = int(year_int)

    # Parse mixing ratios
    vul_var = vul_data['variable']
    for si,sp in enumerate(vul_var['species']):
        year_data["mx_"+sp] = np.clip(np.array(vul_var['ymix'])[:,si],mx_clip_min,mx_clip_max)

    # Parse PT profile
    vul_atm = vul_data['atm']
    year_data["pressure"] =     np.array(vul_atm["pco"]) / 1e6  # convert to bar
    year_data["temperature"] =  np.array(vul_atm["Tco"])

    # Read melt-vapour eqm mixing ratios which were used to initialise VULCAN
    if read_const:
        vol_data_str = None
        with open(output_dir+"offchem/%d/vulcan_cfg.py" % year_int, "r") as f:
            lines = f.read().splitlines()
            for ln in lines:
                if "const_mix" in ln:
                    vol_data_str = ln.split("=")[1].replace("'",'"')
                    break

        if vol_data_str is None:
            raise Exception("Could not parse vulcan cfg file!")

        vol_data = json.loads(vol_data_str)
        for vol in vol_data:
            mv_mx = float(vol_data[vol])
            if mv_mx > 1e-12:
                year_data["mv_"+vol] = mv_mx

    return year_data

def offchem_read_grid(grid_dir):
    """Read-in all information from a GridOfflineChemistry output folder.

    Given the path to a grid folder, containing case_* subfolders, this
    function reads all of the data required to process and plot the grid output.

    Parameters
    ----------
        grid_dir : str
            Path to grid output folder

    Returns
    ----------
        years : np.ndarray
            Array of years in the grid [gpoints, nyears]
        opts : np.ndarray
            Array of OPTIONS dicts in the grid [gpoints]
        data : np.ndarray
            Array of ReadOfflineChemistry output dicts [gpoints, nyears]

    """

    # Get folders for each grid point
    point_folders = glob.glob(grid_dir + "/case_*/")
    gpoints = len(point_folders)

    if gpoints < 1:
        raise Exception("Unable to read offchem grid because no grid points were found in the output folder.")

    # Read grid point-by-point
    grid_years  = []  # For each gp, get years
    grid_opts   = []  # For each gp, read-in OPTIONS
    grid_data   = []  # For each gp, for each year, get offchem data
    for i in range(gpoints):
        fol         = point_folders[i]        # folder
        years_read  = glob.glob(fol + "/offchem/*/")  # years as file paths

        # years as integers
        def ytoint(ypath):
            return int(str(ypath).split("/")[-2])
        gp_years = sorted([ ytoint(y) for y in years_read ])
        grid_years.append(gp_years)

        # OPTIONS dictionary
        options = read_config(fol + "/init_coupler.toml")

        gp_opts = dict(options[0])
        grid_opts.append(gp_opts)

        # offchem data
        gp_data = []  # for each year in this gp
        for i,y in enumerate(gp_years):
            year_data = offchem_read_year(fol, y, read_const = True )
            gp_data.append(year_data)
        grid_data.append(gp_data)

    # Convert to numpy arrays and assert types
    years = np.array(grid_years, dtype=int)
    opts  = np.array(grid_opts,  dtype=dict)
    data  = np.array(grid_data,  dtype=dict)

    return years, opts, data

def offchem_slice_grid(years, opts, data, cvar_filter):
    """Slice the output of GridOfflineChemistry according to control variables.

    Provided data from the output of GridOfflineChemistry, this function
    extracts a 'cross-section' of the data according to the provided control-
    variable filter. It returns a subset of the total grid, in which every
    grid point matches the required control variable arguments.

    For example, if `cvar_filter = {"semimajoraxis": 0.1}`, then it would only
    return grid points which ran with a `semimajoraxis` of `0.1` exactly. All
    other grid points are excluded from the return value.

    Parameters
    ----------
        years : np.ndarray
            Array of years in the grid [gpoints, nyears]
        opts : np.ndarray
            Array of OPTIONS dicts in the grid [gpoints]
        data : np.ndarray
            Array of `offchem_read_year` output dicts [gpoints, nyears]
        cvar_filter : dict
            Dictionary of control variables

    Returns
    ----------
        slice_years : np.ndarray
            Sliced array of years in the grid [sgpoints, nyears]
        slice_opts : np.ndarray
            Sliced array of OPTIONS dicts in the grid [sgpoints]
        slice_data : np.ndarray
            Sliced array of ReadOfflineChemistry output dicts [sgpoints, nyears]
    """

    # Subgrid slice
    slice_years  = []
    slice_opts   = []
    slice_data   = []

    gpoints = len(opts)

    # For each grid point, for each year...
    # check their options to see they each match cvar_filter or not
    for gi in range(gpoints):

        gp_opts = opts[gi]

        exclude = False  # exclude this grid point from slice?

        # For each filter key
        for k in cvar_filter.keys():

            # Mismatching keys (this can just be ignored I think?)
            if k not in gp_opts.keys():
                print("WARNING: filter key '%s' is not present in OPTIONS" % k)
                continue

            # Does not match
            if cvar_filter[k] != gp_opts[k]:
                exclude = True
                break

        if not exclude:
            slice_years.append(years[gi])
            slice_opts.append(gp_opts)
            slice_data.append(data[gi])

    # Convert to numpy arrays and assert types
    slice_years = np.array(slice_years, dtype=int)
    slice_opts  = np.array(slice_opts,  dtype=dict)
    slice_data  = np.array(slice_data,  dtype=dict)

    if np.shape(slice_opts)[0] == 0:
        print("WARNING: No grid points left after slicing")

    return slice_years, slice_opts, slice_data
