from __future__ import annotations

from itertools import chain

import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus
from proteus.config import Config, read_config_object
from proteus.config._converters import none_if_none

PATHS = chain(
    (PROTEUS_ROOT / 'input').glob('*.toml'),
)


@pytest.fixture
def cfg():
    path = PROTEUS_ROOT / 'input' / 'dummy.toml'
    obj = read_config_object(path)

    assert isinstance(obj, Config)

    return obj


def test_none_if_none():
    assert none_if_none('none') is None

    for val in ('', 'notnone', 'mors', 'None'):
        assert none_if_none(val) is not None


def test_dict_interface(cfg):
    # test dict interface for OPTIONS compatibility
    assert cfg['star_model'] == 'baraffe'

    with pytest.raises(ValueError):
        cfg['Phi_global']

    with pytest.raises(ValueError):
        cfg['shallow_ocean_layer']

    with pytest.raises(TypeError):
        cfg['star_model'] = 'spada'


@pytest.mark.parametrize('path', PATHS)
def test_proteus_init(path):
    runner = Proteus(config_path=path)

    assert isinstance(runner.config, Config)
