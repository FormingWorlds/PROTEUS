"""The aragog top_node option over its input domain."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from packaging.version import Version

from proteus.config._interior import Aragog

pytestmark = pytest.mark.unit


def test_the_top_node_defaults_to_linear_and_takes_limited():
    assert Aragog().top_node == 'linear'
    assert Aragog(top_node='limited').top_node == 'limited'


@given(
    value=st.one_of(
        st.text().filter(lambda v: v not in ('linear', 'limited')),
        st.booleans(),
        st.integers(),
        st.floats(),
        st.none(),
    )
)
def test_another_top_node_value_is_refused(value):
    """Another word or spelling, a TOML true, a number or nothing is refused."""
    with pytest.raises(ValueError):
        Aragog(top_node=value)


def test_the_gradient_core_refuses_the_limited_top_node_at_load():
    """aragog refuses this pair; the config refuses it at load, before any structure solve."""
    assert Aragog(core_bc='gradient').top_node == 'linear'
    with pytest.raises(ValueError, match='top_node'):
        Aragog(core_bc='gradient', top_node='limited')


def test_a_later_gradient_core_refuses_the_limited_top_node():
    """The pair is refused whichever field is assigned last."""
    a = Aragog(top_node='limited')
    with pytest.raises(ValueError, match='top_node'):
        a.core_bc = 'gradient'


@given(value=st.sampled_from([True, 1, 1.0, None, 'Limited', 'minmod', float('nan')]))
def test_a_refused_value_names_the_field_and_the_choices(value):
    with pytest.raises(ValueError, match="'top_node' must be in"):
        Aragog(top_node=value)


@pytest.mark.parametrize('first', ['top_node', 'core_bc'])
def test_a_refused_assignment_leaves_the_config(first):
    """Either field set last refuses the pair and keeps the value it found."""
    a = Aragog(**{first: 'limited' if first == 'top_node' else 'gradient'})
    second, value = ('core_bc', 'gradient') if first == 'top_node' else ('top_node', 'limited')
    kept = getattr(a, second)
    with pytest.raises(ValueError, match='not available on core_bc'):
        setattr(a, second, value)
    assert getattr(a, second) == kept and getattr(a, first) != kept


@pytest.mark.parametrize('core_bc', ['energy_balance', 'gradient'])
def test_the_config_reader_takes_the_limited_top_node(tmp_path, core_bc):
    import tomllib

    import tomlkit
    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object

    with open(PROTEUS_ROOT / 'input' / 'dummy.toml', 'rb') as f:
        raw = tomllib.load(f)
    aragog = raw.setdefault('interior_energetics', {}).setdefault('aragog', {})
    aragog |= {'top_node': 'limited', 'core_bc': core_bc}
    path = tmp_path / 'limited.toml'
    with open(path, 'w') as f:
        tomlkit.dump(raw, f)
    if core_bc == 'gradient':
        with pytest.raises(ValueError, match='While structuring Aragog'):
            read_config_object(path)  # the energy_balance file differs only in core_bc
    else:
        assert read_config_object(path).interior_energetics.aragog.top_node == 'limited'


def _pinned_aragog():
    """Lowest fwl-aragog version that pyproject.toml allows."""
    import re
    import tomllib
    from pathlib import Path

    deps = tomllib.loads((Path(__file__).parents[2] / 'pyproject.toml').read_text())
    pin = next(d for d in deps['project']['dependencies'] if d.startswith('fwl-aragog'))
    return Version(re.search(r'>=\s*([\w.]+)', pin).group(1))


def test_the_pin_reader_finds_the_aragog_minimum():
    assert _pinned_aragog() >= Version('26.10.10')


@pytest.mark.parametrize('top_node', ['linear', 'limited', 'minmod'])
@pytest.mark.parametrize(
    'core_bc', ['quasi_steady', 'energy_balance', 'gradient', 'bower2018', 'core_module']
)
def test_the_config_and_aragog_refuse_the_same_pairs(top_node, core_bc):
    """The PROTEUS rule and the aragog rule accept and refuse the same (top_node, core_bc) pairs."""
    boundary = pytest.importorskip('aragog.config.boundary')
    if not hasattr(boundary, 'check_top_node'):
        if _pinned_aragog() > Version('26.10.10'):  # the releases up to 26.10.10 lack it
            pytest.fail('the pinned aragog must have check_top_node')
        pytest.skip('the pinned aragog release has no top_node option')
    try:
        boundary.check_top_node(top_node, core_bc)
        aragog_ok = True
    except ValueError:
        aragog_ok = False
    try:
        Aragog(top_node=top_node, core_bc=core_bc)
        proteus_ok = True
    except ValueError:
        proteus_ok = False
    assert proteus_ok is aragog_ok
