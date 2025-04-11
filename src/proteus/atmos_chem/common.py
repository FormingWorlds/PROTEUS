# Shared functions for atmospheric chemistry
from __future__ import annotations

import logging
import os

import pandas as pd

log = logging.getLogger("fwl."+__name__)

def read_result(outdir:str, module:str) -> pd.DataFrame:
    """
    Read offline chemistry model output file and return as DataFrame.

    Parameters
    ----------
    outdir : str
        Path to output directory of PROTEUS run.
    module : str
        Name of the atmospheric chemistry module used.

    Returns
    ----------
    result : pd.DataFrame
        DataFrame containing the results of the offline chemistry model.
    """

    # Module valid?
    if (module is None) or (module == "none"):
        log.warning("Cannot read chemistry output for `atmos_chem.module='none'`")
        return None

    # Path to CSV file
    csv_file = os.path.join(outdir, "offchem", module+".csv")
    if not os.path.exists(csv_file):
        log.warning(f"Could not read offline chemistry output: '{csv_file}'")
        return None

    # Read into DF and return
    return pd.read_csv(csv_file, delimiter=r"\s+")
