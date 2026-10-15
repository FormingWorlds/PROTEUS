# Shared functions for atmospheric chemistry
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from proteus.atmos_clim.common import list_atmosphere_times
from proteus.utils.helper import find_nearest, format_subyear_time

log = logging.getLogger('fwl.' + __name__)


def chem_csv_name(module: str, time: float) -> str:
    """Name of the chemistry CSV for one snapshot, e.g. 'vulcan_12p320.csv' at 12.32 yr."""
    return f'{module}_{format_subyear_time(time)}.csv'


def resolve_chemistry_times(output_dir: str, hf_all: pd.DataFrame, times) -> list[float]:
    """Resolve `atmos_chem.times` into the snapshot times to run chemistry at.

    Snapshots are the ``data/*_atm.nc`` files. An atmosphere module that writes
    none (the dummy) leaves the helpfile times as the only snapshots.

    Parameters
    ----------
    output_dir : str
        Path to the output directory of the PROTEUS run.
    hf_all : pd.DataFrame
        Helpfile of the run, with its `Time` column [yr].
    times : str | list[float]
        'end' (the final helpfile time), 'all' (every snapshot), or a list of
        times [yr]. Each requested time maps to the nearest snapshot.

    Returns
    -------
    list of float
        Snapshot times [yr], ascending and without duplicates.
    """
    available = list_atmosphere_times(output_dir)
    if not available:
        log.warning('No atmosphere snapshots on disk; using the helpfile times')
        available = sorted({float(t) for t in hf_all['Time']})

    if times == 'all':
        return list(available)
    if times == 'end':
        times = [float(hf_all['Time'].iloc[-1])]

    resolved = set()
    for t in times:
        nearest = float(find_nearest(available, t)[0])
        if nearest != t:
            log.debug('Chemistry at t=%.3e yr uses the snapshot at t=%.3e yr' % (t, nearest))
        resolved.add(nearest)
    return sorted(resolved)


def helpfile_row_at(hf_all: pd.DataFrame, time: float) -> dict:
    """Return the helpfile row whose `Time` [yr] is nearest `time`, as a dict.

    The last such row wins, since the init stage keeps several rows at t = 0.
    """
    hf_times = hf_all['Time'].to_numpy(dtype=float)
    dist = np.abs(hf_times - time)
    idx = np.flatnonzero(dist == dist.min())[-1]
    return hf_all.iloc[idx].to_dict()


def read_vulcan_netcdf(fpath: str) -> dict:
    """Read the result of a VULCAN run from its NetCDF output file.

    Values are in the CGS units VULCAN writes: pressure in dyn cm-2, height in
    cm, Kzz in cm2 s-1. Arrays are ordered from the bottom of the column up.

    Parameters
    ----------
    fpath : str
        Path to a ``*_chem.nc`` file written by VULCAN.

    Returns
    -------
    dict
        'p' and 'tmp' at layer centres [nz], 'pl', 'zl' and 'Kzz' at the
        interfaces [nz+1] (Kzz is defined only on the nz-1 interior ones, and
        NaN at the bottom and top), 'species' (list of str) and 'x_gas',
        the volume mixing ratios [nz, number of species].
    """
    import netCDF4 as nc

    with nc.Dataset(fpath) as ds:
        out = {}
        for key in ('p', 'pl', 'tmp', 'zl', 'Kzz', 'x_gas'):
            out[key] = np.ma.filled(ds.variables[key][:].astype(float), np.nan)
        out['species'] = [str(sp) for sp in ds.variables['species'][:]]
    return out


def read_result(outdir: str, module: str, filename: str | None = None) -> pd.DataFrame:
    """
    Read chemistry model output file and return as DataFrame.

    Parameters
    ----------
    outdir : str
        Path to output directory of PROTEUS run.
    module : str
        Name of the atmospheric chemistry module used.
    filename : str | None
        Optional CSV filename override. If None, defaults to '{module}.csv'.
        Used to read the file of one snapshot (e.g. 'vulcan_12p320.csv').

    Returns
    ----------
    result : pd.DataFrame
        DataFrame containing the results of the chemistry model.
    """

    # Module valid?
    if (module is None) or (module == 'none'):
        log.warning("Cannot read chemistry output for `atmos_chem.module='none'`")
        return None

    # Path to CSV file
    if filename is None:
        filename = module + '.csv'
    csv_file = os.path.join(outdir, 'offchem', filename)
    if not os.path.exists(csv_file):
        log.warning(f"Could not read chemistry output: '{csv_file}'")
        return None

    # Read into DF and return
    return pd.read_csv(csv_file, delimiter=r'\s+')
