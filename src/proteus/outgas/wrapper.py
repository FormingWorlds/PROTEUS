# Generic interior wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.utils.helper import PrintHalfSeparator

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

def run_outgas():
    '''
    Run volatile outgassing model
    '''

    PrintHalfSeparator()
