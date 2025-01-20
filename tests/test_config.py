from __future__ import annotations

from itertools import chain

import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus
from proteus.config import Config, read_config_object
from proteus.config._converters import none_if_none
from proteus.config._delivery import Volatiles
from proteus.config._outgas import Calliope

PATHS = chain(
    (PROTEUS_ROOT / 'input').glob('*.toml'),
)


@pytest.fixture
def cfg():
    path = PROTEUS_ROOT / 'input' / 'demos'/ 'dummy.toml'
    obj = read_config_object(path)

    assert isinstance(obj, Config)

    return obj


def test_none_if_none():
    assert none_if_none('none') is None

    for val in ('', 'notnone', 'mors', 'None'):
        assert none_if_none(val) is not None


@pytest.mark.parametrize('path', PATHS)
def test_proteus_init(path):
    if "nogit" in str(path):
        return
    print(f"Testing config at {path}")
    runner = Proteus(config_path=path)
    assert isinstance(runner.config, Config)


def test_calliope_is_included():
    conf = Calliope(
        T_floor=0.1,
        include_CO=False,
        include_H2=True,
    )
    assert not conf.is_included('CO')
    assert conf.is_included('H2')
    with pytest.raises(AttributeError):
        conf.is_included('fails')


def test_delivery_get_pressure():
    conf = Volatiles(
        N2 = 123.0
    )
    assert conf.get_pressure('N2') == 123.0
    with pytest.raises(AttributeError):
        conf.get_pressure('fails')
