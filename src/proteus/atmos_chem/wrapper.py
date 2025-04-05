# Generic interior wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config

def run_chemistry(dirs:dict, config:Config):

    if not config.atmos_chem.module:
        # no chemistry
        pass

    elif config.atmos_chem.module == 'vulcan':
        from proteus.atmos_chem.vulcan import run_offline
        run_offline(dirs, config)

    else:
        raise ValueError(f"Invalid atmos_chem module: {config.atmos_chem.module}")
