"""Bool config fields take only a TOML true or false.

cattrs would otherwise turn any value into a bool with ``bool()``, so a string
``"false"`` reads as true. These tests cover every bool field of the schema, the
file loader, and the global cattrs converter that other code shares. The grid,
inference and migration paths are tested in their own test files.
"""

from __future__ import annotations

import copy
import importlib
import pkgutil
import typing
from functools import reduce

import attrs
import cattrs
import pytest
import tomlkit
from helpers import PROTEUS_ROOT

import proteus.config
from proteus.config import Config, read_config, read_config_object, structure_config

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _bool_paths(cls=Config, prefix=''):
    """Dotted path of every field typed bool, walking the attrs sections from Config."""
    hints = typing.get_type_hints(cls)
    for f in attrs.fields(cls):
        kind = hints.get(f.name, f.type)
        for section in (kind, *typing.get_args(kind)):
            if attrs.has(section):
                yield from _bool_paths(section, f'{prefix}{f.name}.')
        if kind is bool:
            yield f'{prefix}{f.name}'


def _bool_fields_by_class():
    """Every (class, field) typed bool in the proteus.config modules, found without the walk."""
    found = set()
    for info in pkgutil.iter_modules(proteus.config.__path__):
        module = importlib.import_module(f'proteus.config.{info.name}')
        for obj in vars(module).values():
            if isinstance(obj, type) and attrs.has(obj) and obj.__module__ == module.__name__:
                hints = typing.get_type_hints(obj)
                found |= {
                    (obj, f.name)
                    for f in attrs.fields(obj)
                    if hints.get(f.name, f.type) is bool
                }
    return found


BOOL_PATHS = sorted(_bool_paths())
MINIMAL = read_config(PROTEUS_ROOT / 'input' / 'minimal.toml')


def _with(path, value):
    """minimal.toml as a raw dict with ``path`` set to ``value``."""
    raw = copy.deepcopy(MINIMAL)
    *sections, key = path.split('.')
    node = raw
    for name in sections:
        node = node.setdefault(name, {})
    node[key] = value
    return raw


def test_the_walk_finds_every_bool_field():
    """The parametrisation below covers each bool field once: the walk from Config finds
    as many paths as a scan of every attrs class in proteus.config finds bool fields."""
    assert len(BOOL_PATHS) == len(set(BOOL_PATHS)) == len(_bool_fields_by_class())
    assert {'params.out.remove_sf', 'interior_energetics.aragog.top_cell_dt_limit'} < set(
        BOOL_PATHS
    )


@pytest.mark.parametrize('path', BOOL_PATHS)
def test_a_bool_field_refuses_the_string_false(path):
    """``"false"`` in quotes is a string: refused with the field path and the value, never
    read as true."""
    with pytest.raises(ValueError, match='minimal.toml') as err:
        structure_config(_with(path, 'false'), 'minimal.toml')
    assert f"{path}: must be true or false, got 'false'" in str(err.value)


def test_every_bool_field_keeps_true_and_false():
    """A TOML true or false reaches each bool field as written, unless the field's own
    validator refuses that value with minimal.toml (an option not yet supported, or one
    another setting requires); the bool rule itself never refuses a boolean."""
    refused = set()
    for path in BOOL_PATHS:
        for value in (False, True):
            try:
                config = structure_config(_with(path, value), 'm.toml')
            except ValueError as err:
                assert 'must be true or false' not in str(err), path
                refused.add((path, value))
            else:
                assert reduce(getattr, path.split('.'), config) is value, path
    assert refused == {
        ('atmos_chem.vulcan.use_vm_mol', True),
        ('atmos_clim.agni.solve_energy', False),
        ('interior_energetics.heat_tidal', True),
        ('interior_struct.zalmoxis.global_miscibility', True),
        ('outgas.h2_binodal', True),
    }


def test_any_value_that_is_not_a_bool_is_refused():
    """Every TOML type other than a boolean (zero, one, nan, inf, the empty string, a
    list, a table, a date) is refused at any bool field, naming the field."""
    hypothesis = pytest.importorskip('hypothesis')
    st = hypothesis.strategies
    not_bool = st.one_of(
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(max_size=8),
        st.lists(st.integers(), max_size=3),
        st.dictionaries(st.text(max_size=4), st.integers(), max_size=2),
        st.dates(),
        st.datetimes(),
        st.times(),
    )

    @hypothesis.settings(max_examples=80, deadline=None, derandomize=True)
    @hypothesis.given(path=st.sampled_from(BOOL_PATHS), value=not_bool)
    def check(path, value):
        with pytest.raises(ValueError) as err:
            structure_config(_with(path, value), 'm.toml')
        assert f'{path}: must be true or false, got ' in str(err.value)

    check()


def test_a_config_file_with_a_quoted_false_is_refused(tmp_path):
    """The loader of config files refuses ``remove_sf = "false"``; the same file with
    ``false`` loads and holds False."""
    bad, good = tmp_path / 'bad.toml', tmp_path / 'good.toml'
    bad.write_text(tomlkit.dumps(_with('params.out.remove_sf', 'false')))
    good.write_text(tomlkit.dumps(_with('params.out.remove_sf', False)))
    with pytest.raises(ValueError, match='params.out.remove_sf: must be true or false'):
        read_config_object(bad)
    assert read_config_object(good).params.out.remove_sf is False


def test_importing_the_config_leaves_the_global_cattrs_converter_alone():
    """PROTEUS structures configs with a converter of its own, so other code that uses
    cattrs' global converter in the same process keeps the bool rules of a fresh one."""
    assert proteus.config.converter is not cattrs.global_converter
    for value in ('false', 0, 1.5):
        assert cattrs.structure(value, bool) is cattrs.Converter().structure(value, bool)
