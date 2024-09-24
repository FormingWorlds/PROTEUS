from __future__ import annotations

from helpers import PROTEUS_ROOT

from proteus.config import Config, read_config_object


def test_read_config_object():
    path = PROTEUS_ROOT / 'input' / 'new.toml'
    obj = read_config_object(path)

    assert isinstance(obj, Config)
