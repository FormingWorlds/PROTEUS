# Generic interior wrapper
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config

def run_offline(dirs:dict, config:Config, hf_row:dict):

    if not config.atmos_chem.module:
        # no chemistry
        pass

    elif config.atmos_chem.module == 'vulcan':
        from proteus.atmos_chem.vulcan import run_vulcan_offline
        run_vulcan_offline(dirs, config, hf_row)

    else:
        raise ValueError(f"Invalid atmos_chem module: {config.atmos_chem.module}")
