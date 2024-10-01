from __future__ import annotations

from helpers import PROTEUS_ROOT

from proteus.config import Config, read_config_object
from proteus.config._converters import none_if_none


def test_read_config_object():
    path = PROTEUS_ROOT / 'input_new' / 'dummy.toml'
    obj = read_config_object(path)

    assert isinstance(obj, Config)


def test_none_if_none():
    assert none_if_none('none') is None

    for val in ('', 'notnone', 'mors', 'None'):
        assert none_if_none(val) is not None
