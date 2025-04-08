# Generic interior wrapper
from __future__ import annotations

import logging
import os
import pandas as pd

from typing import TYPE_CHECKING

log = logging.getLogger("fwl."+__name__)

if TYPE_CHECKING:
    from proteus.config import Config

def read_result(outdir:str) -> pd.DataFrame:
    """
    Read offline chemistry model output file and return as DataFrame.

    Parameters
    ----------
    outdir : str
        Path to output directory of PROTEUS run.

    Returns
    ----------
    result : pd.DataFrame
        DataFrame containing the results of the offline chemistry model.
    """

    # Path to CSV file
    csv_file = os.path.join(outdir, "offchem", "recent.csv")
    if not os.path.exists(csv_file):
        log.warning(f"Could not read offline chemistry output: '{csv_file}'")
        return None

    # Read into DF and return
    return pd.read_csv(csv_file, delimiter=r"\s+")

def run_offline(dirs:dict, config:Config, hf_row:dict) -> pd.DataFrame:
    """
    Run atmospheric chemistry model offline, to postprocess final PROTEUS iteration.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories.
    config : Config
        Configuration object.
    hf_row : dict
        Dictionary of current helpfile row.

    Returns
    ----------
        result : pd.DataFrame
            DataFrame containing the results of the offline chemistry model.
    """

    log.info("Running offline atmospheric chemistry...")

    if not config.atmos_chem.module:
        # no chemistry
        log.warning("Cannot run offline chemistry, no module specified")
        return None

    elif config.atmos_chem.module == 'vulcan':
        log.debug("Using VULCAN kinetics model")
        from proteus.atmos_chem.vulcan import run_vulcan_offline
        run_vulcan_offline(dirs, config, hf_row)

    else:
        raise ValueError(f"Invalid atmos_chem module: {config.atmos_chem.module}")

    return read_result(dirs["output"])
