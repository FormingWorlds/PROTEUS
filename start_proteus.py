"""
PROTEUS Main file
"""
from __future__ import annotations

from proteus import Proteus

runner = Proteus(config_path='input/default.toml')
runner.start(
    resume=False,
)
