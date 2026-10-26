"""The accretion.remelt option: values, resolution per module and refusals.

Source: ``proteus.config._accretion.Accretion.remelt`` and ``remelt_mode``, and
``proteus.config._config.check_accretion_remelt_source``. Contract: 'none' (the
default) resolves to 'computed' with Morrigan and to 'full' otherwise, at the
time it is read; 'full' and 'computed' are kept; any other value is refused at
load; 'computed' needs a module that reports a melt fraction.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import attrs
import pytest

from proteus.config._accretion import Accretion, AccretionTimeline

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.parametrize(
    ('module', 'mode'),
    [(None, 'full'), ('dummy', 'full'), ('timeline', 'full'), ('morrigan', 'computed')],
)
def test_the_default_resolves_by_module(module, mode):
    """remelt = 'none' gives 'computed' with Morrigan and 'full' with every other module."""
    acc = Accretion(module=module, timeline=AccretionTimeline(timeline_path='x.csv'))
    assert acc.remelt is None
    assert acc.remelt_mode == mode


def test_the_resolution_follows_a_module_change_after_load():
    """The mode is resolved when read, so a module set after load is honoured."""
    acc = Accretion()
    assert acc.remelt_mode == 'full'
    acc.module = 'morrigan'
    assert acc.remelt_mode == 'computed'


@pytest.mark.parametrize('value', ['full', 'computed'])
def test_an_explicit_mode_is_kept_for_every_module(value):
    """An explicit 'full' or 'computed' is the mode, whatever the module."""
    for module in ('timeline', 'morrigan'):
        acc = Accretion(
            module=module, remelt=value, timeline=AccretionTimeline(timeline_path='x.csv')
        )
        assert acc.remelt == value
        assert acc.remelt_mode == value


def test_the_field_lists_its_choices():
    """The validator carries the 3 accepted values, which the config reference prints."""
    field = attrs.fields(Accretion).remelt
    assert set(field.validator.options) == {None, 'full', 'computed'}
    assert field.default == 'none'


def test_another_remelt_value_is_refused():
    """Another word or spelling, a TOML true, a number, NaN or infinity is refused at load."""
    hypothesis = pytest.importorskip('hypothesis')
    st = hypothesis.strategies

    @hypothesis.settings(derandomize=True, deadline=None, max_examples=200)
    @hypothesis.given(
        value=st.one_of(
            st.text(max_size=12).filter(lambda s: s not in ('full', 'computed', 'none')),
            st.sampled_from(['Full', 'COMPUTED', ' full', 'full ', 'None', 'partial']),
            st.booleans(),
            st.integers(),
            st.floats(allow_nan=True, allow_infinity=True),
            st.just(math.nan),
        )
    )
    def check(value):
        with pytest.raises(ValueError, match='remelt') as exc:
            Accretion(remelt=value)
        assert 'full' in str(exc.value) and 'computed' in str(exc.value)

    check()


def _instance(module, remelt):
    """Duck-typed config instance the remelt source check reads."""
    acc = Accretion(
        module=module, remelt=remelt, timeline=AccretionTimeline(timeline_path='x.csv')
    )
    return SimpleNamespace(accretion=acc)


@pytest.mark.parametrize('module', [None, 'dummy'])
def test_computed_without_a_melt_fraction_source_is_refused(module):
    """'computed' with no accretion module or the analytical one has no f to apply."""
    from proteus.config._config import check_accretion_remelt_source

    with pytest.raises(ValueError, match="remelt = 'computed' needs an accretion module"):
        check_accretion_remelt_source(_instance(module, 'computed'), None, None)
    assert check_accretion_remelt_source(_instance(module, 'full'), None, None) is None


@pytest.mark.parametrize('module', ['timeline', 'morrigan'])
def test_computed_with_a_melt_fraction_source_is_accepted(module):
    """Morrigan reports f per merger; a timeline can carry an f_melt column."""
    from proteus.config._config import check_accretion_remelt_source

    assert check_accretion_remelt_source(_instance(module, 'computed'), None, None) is None
    assert check_accretion_remelt_source(_instance(module, None), None, None) is None


def test_the_source_check_runs_when_the_config_is_built():
    """Building a Config with the analytical module and 'computed' fails; with 'full' it builds."""
    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object

    cfg = read_config_object(PROTEUS_ROOT / 'input' / 'all_options.toml')
    with pytest.raises(ValueError, match="remelt = 'computed'"):
        attrs.evolve(
            cfg, accretion=attrs.evolve(cfg.accretion, module='dummy', remelt='computed')
        )
    built = attrs.evolve(
        cfg, accretion=attrs.evolve(cfg.accretion, module='dummy', remelt='full')
    )
    assert built.accretion.remelt_mode == 'full'


def test_the_reference_config_ships_the_default():
    """input/all_options.toml carries remelt = 'none', which loads as the default."""
    import tomllib

    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object

    path = PROTEUS_ROOT / 'input' / 'all_options.toml'
    with open(path, 'rb') as f:
        assert tomllib.load(f)['accretion']['remelt'] == 'none'
    assert read_config_object(path).accretion.remelt is None
