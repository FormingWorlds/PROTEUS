from __future__ import annotations

from itertools import chain
from pathlib import Path

import pytest

from proteus import Proteus

PROTEUS_ROOT = Path(__file__).parents[1]

PATHS = chain(
    (PROTEUS_ROOT / 'input').glob('*.toml'),
    (PROTEUS_ROOT / 'examples').glob('*/*.toml'),
)


@pytest.mark.parametrize('path', PATHS)
def test_proteus_init(path):
    runner = Proteus(config_path=path)
