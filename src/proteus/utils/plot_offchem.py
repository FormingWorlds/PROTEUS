# Variables and functions to help with offline chemistry plotting functions
# These do not do the plotting themselves
from __future__ import annotations

import glob
import logging
import os

import numpy as np

from proteus.config import read_config
from proteus.utils.helper import list_snapshot_times, snapshot_path_for_time

log = logging.getLogger('fwl.' + __name__)


def offchem_read_year(output_dir, year, mx_clip_min=1e-30, mx_clip_max=1.0, read_const=False):
    """Read PT profile and VULCAN output for one snapshot.

    The VULCAN result of a snapshot is ``data/<year>_chem.nc``, with the year
    in the sub-year form of `format_subyear_time` (``12p320`` at 12.32 yr).

    Parameters
    ----------
        output_dir : str
            Path to run output folder.
        year : float
            Snapshot time [yr] to read.

        mx_clip_min=1e-30 : float
            Mixing ratio floor
        mx_clip_max=1.0 : float
            Mixing ratio ceiling
        read_const=False : bool
            Deprecated and unsupported. The per-year ``vulcan_cfg.py``
            dump is not emitted, so reading initial-composition mixing
            ratios from this output raises ``NotImplementedError``. Pull
            the ``<vol>_vmr`` columns from ``runtime_helpfile.csv`` instead.

    Returns
    ----------
        year_data : dict
            Dictionary of output arrays for the given year: 'year' [yr],
            'pressure' [bar], 'temperature' [K] and 'mx_<species>'.
    """

    if read_const:
        raise NotImplementedError(
            'read_const is not supported with the VULCAN NetCDF output; '
            'read <vol>_vmr columns from runtime_helpfile.csv instead'
        )

    fpath = snapshot_path_for_time(os.path.join(output_dir, 'data'), year, '_chem.nc')
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f'No VULCAN snapshot found at {fpath}')
    from proteus.atmos_chem.vulcan import read_vulcan_netcdf

    result = read_vulcan_netcdf(fpath)

    year_data = {'year': float(year)}

    # Parse mixing ratios
    for si, sp in enumerate(result['species']):
        year_data['mx_' + sp] = np.clip(result['x_gas'][:, si], mx_clip_min, mx_clip_max)

    # Parse PT profile
    year_data['pressure'] = result['p'] / 1e6  # dyn cm-2 to bar
    year_data['temperature'] = result['tmp']

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
    point_folders = glob.glob(grid_dir + '/case_*/')
    gpoints = len(point_folders)

    if gpoints < 1:
        raise Exception(
            'Unable to read offchem grid because no grid points were found in the output folder.'
        )

    # Read grid point-by-point
    grid_years = []  # For each gp, get years
    grid_opts = []  # For each gp, read-in OPTIONS
    grid_data = []  # For each gp, for each year, get offchem data
    for i in range(gpoints):
        fol = point_folders[i]  # folder
        # One VULCAN result per snapshot: data/<year>_chem.nc
        gp_years = list_snapshot_times(os.path.join(fol, 'data'), '_chem.nc')
        grid_years.append(gp_years)

        # OPTIONS dictionary
        options = read_config(fol + '/init_coupler.toml')

        gp_opts = dict(options[0])
        grid_opts.append(gp_opts)

        # offchem data
        gp_data = []  # for each year in this gp
        for y in gp_years:
            year_data = offchem_read_year(fol, y)
            gp_data.append(year_data)
        grid_data.append(gp_data)

    # Convert to numpy arrays and assert types
    years = np.array(grid_years, dtype=float)
    opts = np.array(grid_opts, dtype=dict)
    data = np.array(grid_data, dtype=dict)

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
    slice_years = []
    slice_opts = []
    slice_data = []

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
                log.warning("Filter key '%s' is not present in OPTIONS", k)
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
    slice_years = np.array(slice_years, dtype=float)
    slice_opts = np.array(slice_opts, dtype=dict)
    slice_data = np.array(slice_data, dtype=dict)

    if np.shape(slice_opts)[0] == 0:
        log.warning('No grid points left after slicing')

    return slice_years, slice_opts, slice_data
