from __future__ import annotations

from itertools import chain

import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus

PATHS = chain(
    (PROTEUS_ROOT / 'input').glob('*.toml'),
    (PROTEUS_ROOT / 'examples').glob('*/*.toml'),
)


@pytest.mark.parametrize('path', PATHS)
def test_proteus_init(path):
    runner = Proteus(config_path=path)
