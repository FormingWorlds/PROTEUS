# Shared functions for atmospheric chemistry
from __future__ import annotations

import logging
import os

import pandas as pd

log = logging.getLogger('fwl.' + __name__)


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
        Used by online mode to read per-snapshot files (e.g. 'vulcan_5000.csv').

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
