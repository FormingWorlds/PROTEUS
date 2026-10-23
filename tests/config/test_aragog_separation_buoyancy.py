"""The aragog separation_buoyancy option over its input domain, against the aragog release.

Source: ``proteus.config._interior.Aragog.separation_buoyancy``. Contract: the choices and the
default equal those of the pinned aragog, a value outside them is refused at load with the
field named, and the TOML reader passes each choice through.
"""

from __future__ import annotations

import math
import re
import tomllib
from pathlib import Path

import attrs
import pytest
from packaging.version import Version

from proteus.config._interior import Aragog

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_MODES = ('mixture', 'solid')


def test_the_buoyancy_defaults_to_mixture_and_takes_solid():
    """'mixture' is the default; 'solid' is accepted at construction and by assignment."""
    assert Aragog().separation_buoyancy == 'mixture'
    assert Aragog(separation_buoyancy='solid').separation_buoyancy == 'solid'
    a = Aragog()
    a.separation_buoyancy = 'solid'
    assert a.separation_buoyancy == 'solid'


def test_another_buoyancy_value_is_refused():
    """Another word or spelling, a TOML true, a number, NaN, infinity or nothing is refused."""
    hypothesis = pytest.importorskip('hypothesis')
    st = hypothesis.strategies

    @hypothesis.given(
        value=st.one_of(
            st.text().filter(lambda v: v not in _MODES),
            st.booleans(),
            st.integers(),
            st.floats(allow_nan=True, allow_infinity=True),
            st.none(),
        )
    )
    def refused(value):
        with pytest.raises(ValueError) as err:
            Aragog(separation_buoyancy=value)
        # attrs puts its message first in args; str() of the error quotes it once more.
        assert err.value.args[0].startswith(
            "'separation_buoyancy' must be in ('mixture', 'solid')"
        )

    refused()


@pytest.mark.parametrize(
    'value',
    [True, 1, 1.0, 0.0, -1.0, math.nan, math.inf, None, '', 'Mixture', ' mixture', 'melt'],
)
def test_a_refused_assignment_keeps_the_value(value):
    """A refused assignment raises and leaves the configured choice in place."""
    a = Aragog(separation_buoyancy='solid')
    with pytest.raises(ValueError, match="'separation_buoyancy' must be in"):
        a.separation_buoyancy = value
    assert a.separation_buoyancy == 'solid'


@pytest.mark.parametrize('mode', [*_MODES, 'melt'])
def test_the_config_reader_takes_each_buoyancy(tmp_path, mode):
    """A TOML file carries each choice through the reader; another value stops the load."""
    import tomlkit
    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object

    with open(PROTEUS_ROOT / 'input' / 'dummy.toml', 'rb') as f:
        raw = tomllib.load(f)
    raw.setdefault('interior_energetics', {}).setdefault('aragog', {})[
        'separation_buoyancy'
    ] = mode
    path = tmp_path / 'buoyancy.toml'
    with open(path, 'w') as f:
        tomlkit.dump(raw, f)
    if mode == 'melt':
        with pytest.raises(ValueError, match='separation_buoyancy'):
            read_config_object(path)
    else:
        assert read_config_object(path).interior_energetics.aragog.separation_buoyancy == mode


def test_the_choices_and_default_equal_those_of_aragog():
    """The config offers exactly the aragog modes, with aragog's default."""
    phases = pytest.importorskip('aragog.config.phases')
    field = attrs.fields(Aragog).separation_buoyancy
    assert set(field.validator.options) == set(phases.SEPARATION_BUOYANCY_MODES)
    assert field.default == phases.SEPARATION_BUOYANCY_DEFAULT == 'mixture'


def test_the_pin_is_an_aragog_with_the_buoyancy_option():
    """aragog 26.10.22 is the first release that takes separation_buoyancy, and the installed
    aragog meets the floor."""
    from importlib.metadata import version

    deps = tomllib.loads((Path(__file__).parents[2] / 'pyproject.toml').read_text())
    pin = next(d for d in deps['project']['dependencies'] if d.startswith('fwl-aragog'))
    floor = Version(re.search(r'>=\s*([\w.]+)', pin).group(1))
    assert floor >= Version('26.10.22')
    assert Version(version('fwl-aragog')) >= floor
