#!/usr/bin/env python3

"""
PROTEUS run script
"""
from __future__ import annotations

from proteus import Proteus

runner = Proteus(config_path='input/default.toml')
runner.start(
    resume=False,
)
