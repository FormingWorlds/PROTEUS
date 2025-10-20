# Check the status of a PROTEUS parameter grid's cases
from __future__ import annotations

import os


def pack(pgrid_dir:str):
    if (not os.path.exists(pgrid_dir)) or (not os.path.isdir(pgrid_dir)):
        raise Exception("Invalid path '%s'" % pgrid_dir)
