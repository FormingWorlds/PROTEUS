# Shared functions for atmospheric chemistry
from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd

from proteus.atmos_clim.common import list_atmosphere_times
from proteus.utils.helper import (
    SUBYEAR_TIME_RESOLUTION,
    format_subyear_time,
    latest_at_or_before,
    snapshot_path_for_time,
)

log = logging.getLogger('fwl.' + __name__)


def chem_csv_name(module: str, time: float) -> str:
    """Name of the chemistry CSV for one snapshot, e.g. 'vulcan_12p320.csv' at 12.32 yr."""
    return f'{module}_{format_subyear_time(time)}.csv'


def crossing_rows(values, target: float) -> np.ndarray:
    """Indices of the rows at which a helpfile column reaches or crosses a value.

    Row i counts when row i - 1 lies strictly on the other side of `target`, or when row i
    equals `target` and row i - 1 does not (row 0 counts when it equals `target`). A column
    that touches the value and turns back counts once; NaN rows never count.

    Parameters
    ----------
    values : array-like
        The column, in helpfile row order.
    target : float
        The value, in the unit of the column.

    Returns
    -------
    np.ndarray
        Row indices, ascending.
    """
    side = np.sign(np.asarray(values, dtype=float) - target)
    prev = np.concatenate(([np.nan], side[:-1]))
    reach = (side == 0) & (prev != 0)
    cross = (side != 0) & (prev == -side)
    return np.flatnonzero(reach | cross)


def resolve_chemistry_targets(hf_all: pd.DataFrame, times) -> list[float]:
    """Helpfile times [yr] that `atmos_chem.times` asks chemistry for.

    A table gives the time of every row at which each of its columns reaches or crosses
    each of its values (`crossing_rows`), with a warning for a value never reached, and the
    final time with `end = true`. A list of times is the table `{Time = [...]}`: each time
    gives the first row at or after it. 'end' gives the final time.

    Parameters
    ----------
    hf_all : pd.DataFrame
        Helpfile of the run, with its `Time` column [yr].
    times : str | list[float] | dict
        'end', a list of times [yr], or a table of helpfile keys and values.

    Returns
    -------
    list of float
        Target times [yr], ascending and without duplicates.
    """
    hf_times = hf_all['Time'].to_numpy(dtype=float)
    if times == 'end':
        return [float(hf_times[-1])]
    if isinstance(times, list):
        times = {'Time': times}

    targets = set()
    for key, values in times.items():
        if key == 'end':
            if values:
                targets.add(float(hf_times[-1]))
            continue
        if key not in hf_all.columns:
            log.warning(
                "Chemistry targets on '%s' skipped: the helpfile has no such column" % key
            )
            continue
        for value in values:
            rows = crossing_rows(hf_all[key].to_numpy(dtype=float), value)
            if len(rows) == 0:
                log.warning('Chemistry target %s = %g is never reached; skipped' % (key, value))
            targets.update(float(t) for t in hf_times[rows])
    return sorted(targets)


def chem_targets_crossed(hf_all: pd.DataFrame, times) -> bool:
    """Whether the latest helpfile row reaches a target of `atmos_chem.times`.

    For a list, a time in (t_prev, t_now] of the last two rows; for a table, a value that
    a column reaches or crosses at the last row, by the rule of `crossing_rows`, so the
    run writes a snapshot at the rows that `resolve_chemistry_targets` later selects.
    'end' and 'all' have no targets inside the run.
    """
    if not isinstance(times, (list, dict)) or len(hf_all) < 2:
        return False
    last = hf_all.iloc[-2:]
    if isinstance(times, list):
        t_prev, t_now = last['Time'].to_numpy(dtype=float)
        return any(t_prev < t <= t_now for t in times)
    for key, values in times.items():
        if key == 'end' or key not in last.columns:
            continue
        column = last[key].to_numpy(dtype=float)
        if any(1 in crossing_rows(column, value) for value in values):
            return True
    return False


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
    times : str | list[float] | dict
        'end' (the final helpfile time), 'all' (every snapshot), a list of
        times [yr], or a table of helpfile keys and values
        (`resolve_chemistry_targets`). Each target uses the latest snapshot at or before
        its row, which is the row's own snapshot when the run wrote one there.

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

    resolved = set()
    for t in resolve_chemistry_targets(hf_all, times):
        snap = latest_at_or_before(available, t)
        if snap is None:
            log.warning(
                'Chemistry target t=%.3e yr has no snapshot at or before it; skipped' % t
            )
            continue
        if t - snap > SUBYEAR_TIME_RESOLUTION:
            # a run writes a snapshot at each target row; an older run may not have
            log.warning(
                'Chemistry target t=%.3e yr has no snapshot of its own; '
                'using the latest before it, t=%.3e yr' % (t, snap)
            )
        resolved.add(snap)
    return sorted(resolved)


def snapshot_star_file(output_dir: str, time: float) -> str:
    """Stellar spectrum file (``data/<int>.sflux``) that the atmosphere snapshot at `time` used.

    The AGNI snapshot records its spectrum as `starfile`; the file of that name in the
    run's ``data/`` folder is used, so a moved run folder still works. Without one (another
    atmosphere module, a spectrum taken from the spectral-file cache, or a missing file) it
    is the latest spectrum of the evolution written at or before `time` [yr], as the
    atmosphere step reads it, or the earliest when every one is later. ``-1.sflux``, the
    copy of the modern spectrum, is not a spectrum of the evolution and is never used.

    Raises
    ------
    FileNotFoundError
        If the run has no ``.sflux`` file.
    """
    data_dir = os.path.join(output_dir, 'data')
    nc_path = snapshot_path_for_time(data_dir, time, '_atm.nc')
    if os.path.isfile(nc_path):
        import netCDF4 as nc

        with nc.Dataset(nc_path) as ds:
            recorded = str(ds.variables['starfile'][...]) if 'starfile' in ds.variables else ''
        candidate = os.path.join(data_dir, os.path.basename(recorded))
        if recorded.endswith('.sflux') and os.path.isfile(candidate):
            return candidate

    stamps = (int(os.path.basename(f).split('.')[0]) for f in glob.glob(data_dir + '/*.sflux'))
    written = sorted(t for t in stamps if t >= 0)
    if not written:
        raise FileNotFoundError(f'No stellar spectrum of the evolution (.sflux) in {data_dir}')
    latest = latest_at_or_before(written, time)
    return os.path.join(data_dir, '%d.sflux' % (written[0] if latest is None else latest))


def helpfile_row_at(hf_all: pd.DataFrame, time: float) -> dict:
    """Return the latest helpfile row at or before `time` [yr], as a dict.

    A snapshot carries the time of the row it was written at, rounded in its file name, so
    half of SUBYEAR_TIME_RESOLUTION is allowed after `time`. Of the init rows, which all
    hold t = 0, the last wins.

    Raises
    ------
    ValueError
        If every row is later than `time`.
    """
    hf_times = hf_all['Time'].to_numpy(dtype=float)
    idx = np.flatnonzero(hf_times <= time + 0.5 * SUBYEAR_TIME_RESOLUTION)
    if idx.size == 0:
        raise ValueError(f'The helpfile has no row at or before t={time:.3e} yr')
    return hf_all.iloc[idx[-1]].to_dict()


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
