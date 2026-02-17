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
    module = config.atmos_chem.module
    when = getattr(config.atmos_chem, 'when', 'manually')

    if not module or module == 'none':
        log.warning('Cannot run atmospheric chemistry, no module specified')
        return None

    if module != 'vulcan':
        raise ValueError(
            f"Invalid atmos_chem module: '{module}'. Currently only 'vulcan' is supported."
        )

    from proteus.atmos_chem.vulcan import (
        run_vulcan_offline,
        run_vulcan_online,
    )

    if when == 'manually':
        log.debug("Atmospheric chemistry set to 'manually'; skipping")
        return None

    elif when == 'offline':
        log.debug('Running atmospheric chemistry in OFFLINE mode')
    if not module:
        # no chemistry
        log.warning('Cannot run atmospheric chemistry, no module specified')
        return None

    elif module == 'vulcan':
        log.debug('Using VULCAN kinetics model')
        from proteus.atmos_chem.vulcan import run_vulcan_offline

        run_vulcan_offline(dirs, config, hf_row)
        return read_result(dirs['output'], module)

    elif when == 'online':
        log.debug('Running atmospheric chemistry in ONLINE mode')
        run_vulcan_online(dirs, config, hf_row)
        return read_result(dirs['output'], module)
    else:
        raise ValueError(f'Invalid atmos_chem module: {module}')
    return read_result(dirs['output'], module)
