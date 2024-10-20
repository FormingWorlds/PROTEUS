# Generic stellar evolution wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)
