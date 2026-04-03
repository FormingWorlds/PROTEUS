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
    Run atmospheric chemistry model (offline post-processing or online per-snapshot).

    Results are saved to CSV files on disk and returned as a DataFrame.

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

    module = config.atmos_chem.module

    # When to run chemistry: 'manually' (skip), 'offline' (post-processing),
    # or 'online' (every snapshot during simulation). Defaults to 'manually'
    # for backwards compatibility with configs that lack the 'when' field.
    when = getattr(config.atmos_chem, 'when', 'manually')

    # Guard: no module configured
    if not module or module == 'none':
        log.warning('Cannot run atmospheric chemistry, no module specified')
        return None

    # Guard: scheduling
    if when == 'manually':
        log.debug("Atmospheric chemistry set to 'manually'; skipping")
        return None

    # Resolve the runner function (lazy imports)
    if module == 'vulcan':
        from proteus.atmos_chem.vulcan import run_vulcan as _run
    elif module == 'dummy':
        from proteus.atmos_chem.dummy import run_dummy_chem as _run
    else:
        raise ValueError(f"Invalid atmos_chem module: '{module}'")

    # Dispatch based on scheduling mode
    filename = None
    if when == 'offline':
        log.debug('Running atmospheric chemistry in OFFLINE mode')
        _run(dirs, config, hf_row)
    elif when == 'online':
        log.debug('Running atmospheric chemistry in ONLINE mode')
        _run(dirs, config, hf_row, online=True)
        filename = f'{module}_{int(hf_row["Time"])}.csv'
    else:
        raise ValueError(f"Invalid atmos_chem.when value: '{when}'")

    # Read the CSV output and return as DataFrame
    return read_result(dirs['output'], module, filename=filename)
