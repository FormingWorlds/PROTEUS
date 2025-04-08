# Generic interior wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import pandas as pd
import os

log = logging.getLogger("fwl."+__name__)

if TYPE_CHECKING:
    from proteus.config import Config

def run_offline(dirs:dict, config:Config, hf_row:dict) -> pd.DataFrame:

    log.info("Running offline atmospheric chemistry...")

    if not config.atmos_chem.module:
        # no chemistry
        log.warning("Cannot run offline chemistry, no module specified")
        return None

    elif config.atmos_chem.module == 'vulcan':
        log.debug("Using VULCAN kinetics model")

        from proteus.atmos_chem.vulcan import run_vulcan_offline, read_result
        run_vulcan_offline(dirs, config, hf_row)

        # read result and return
        return read_result(dirs["output"])

    else:
        raise ValueError(f"Invalid atmos_chem module: {config.atmos_chem.module}")
