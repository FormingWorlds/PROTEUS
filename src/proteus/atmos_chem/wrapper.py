# Generic atmospheric chemistry wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from proteus.atmos_chem.common import read_result

log = logging.getLogger('fwl.' + __name__)

if TYPE_CHECKING:
    from proteus.config import Config


def run_chemistry(dirs: dict, config: Config, hf_row: dict) -> pd.DataFrame:
    """
    Run atmospheric chemistry model offline, to postprocess final PROTEUS iteration.

    Results are saved to files on the disk, and returned as a DataFrame.

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

    log.info('Running atmospheric chemistry...')

    # Which chemistry solver to use (currently only 'vulcan' is supported)
    module = config.atmos_chem.module

    # When to run chemistry: 'manually' (skip), 'offline' (post-processing),
    # or 'online' (every snapshot during simulation). Defaults to 'manually'
    # for backwards compatibility with configs that lack the 'when' field.
    when = getattr(config.atmos_chem, 'when', 'manually')

    # Guard: no module configured — nothing to do
    if not module or module == 'none':
        log.warning('Cannot run atmospheric chemistry, no module specified')
        return None

    # Guard: only VULCAN is implemented as a chemistry solver
    if module != 'vulcan':
        raise ValueError(
            f"Invalid atmos_chem module: '{module}'. Currently only 'vulcan' is supported."
        )

    # Lazy import to avoid loading VULCAN (heavy dependency) unless needed
    from proteus.atmos_chem.vulcan import run_vulcan

    # Dispatch based on scheduling mode:
    #   'manually'  — user will invoke chemistry separately (e.g. via CLI)
    #   'offline'   — run once after simulation ends, on the final state
    #   'online'    — run at every snapshot during the main simulation loop
    filename = None  # default: read_result uses '{module}.csv'
    if when == 'manually':
        log.debug("Atmospheric chemistry set to 'manually'; skipping")
        return None
    elif when == 'offline':
        log.debug('Running atmospheric chemistry in OFFLINE mode')
        run_vulcan(dirs, config, hf_row)
    elif when == 'online':
        log.debug('Running atmospheric chemistry in ONLINE mode')
        run_vulcan(dirs, config, hf_row, online=True)
        # Online mode writes per-snapshot files (e.g. vulcan_5000.csv)
        filename = f'vulcan_{int(hf_row["Time"])}.csv'
    else:
        raise ValueError(f"Invalid atmos_chem.when value: '{when}'")

    # Read the CSV output written by VULCAN and return as a DataFrame
    return read_result(dirs['output'], module, filename=filename)
