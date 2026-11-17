"""Config of the core profile source of the Aragog core_module core (``profile_source``).

Module under test: ``proteus.config._interior.AragogCoreModule`` (``profile_source``,
``core_cp_override``, ``core_t_cmb_step``) and ``proteus.config._config``
(``check_core_profile_source_requirements``). Invariants: the Gaussian profile is the default;
the PALEOS profile reads a positive finite secular heat capacity or the string "table" and a
positive finite CMB temperature step, a TOML true is refused for both, and the PALEOS profile is
refused outside core_module, with a stratified core and with giant impacts.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip('hypothesis')

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from proteus.config import read_config_object  # noqa: E402
from proteus.config._interior import AragogCoreModule  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]


def _kept(name, value):
    """True when ``AragogCoreModule`` keeps ``value`` unchanged in field ``name``."""
    try:
        got = getattr(AragogCoreModule(**{name: value}), name)
    except (TypeError, ValueError):
        return False
    return got == value and type(got) is type(value)


def test_the_gaussian_profile_is_the_default():
    """The default profile is the Gaussian fit, with the PALEOS settings at their defaults:
    a secular c_p of 840 J/kg/K and a CMB temperature step of 10 K."""
    cm = AragogCoreModule()
    assert cm.profile_source == 'gaussian'
    assert cm.core_cp_override == pytest.approx(840.0)
    assert cm.core_t_cmb_step == pytest.approx(10.0)
    assert AragogCoreModule(profile_source='paleos').profile_source == 'paleos'
    for bad in ('PALEOS', 'tabulated', '', None, True):
        with pytest.raises((TypeError, ValueError)):
            AragogCoreModule(profile_source=bad)


NUMBERS = st.floats(allow_nan=True, allow_infinity=True) | st.integers(-10, 10**400)


@given(value=NUMBERS)
def test_the_c_p_override_is_a_positive_finite_number(value):
    """A number is kept exactly when it is finite and above 0."""
    try:
        want = math.isfinite(value) and value > 0
    except OverflowError:
        want = False
    assert _kept('core_cp_override', value) is want
    assert _kept('core_cp_override', 840) and not _kept('core_cp_override', 0.0)


@given(value=NUMBERS)
def test_the_grid_step_is_a_positive_finite_number(value):
    """A number is kept exactly when it is finite and above 0."""
    try:
        want = math.isfinite(value) and value > 0
    except OverflowError:
        want = False
    assert _kept('core_t_cmb_step', value) is want
    assert _kept('core_t_cmb_step', 1e-300) and not _kept('core_t_cmb_step', 0)


@pytest.mark.parametrize('value', [True, False, None, '840', 'Table', b'table', [840.0]])
def test_values_that_are_not_a_number_or_table_are_refused(value):
    """A bool, None, a numeric string, another string, bytes and a list are refused for the
    c_p override, and all but no string at all for the grid step; "table" is the one string."""
    assert _kept('core_cp_override', value) is False
    assert _kept('core_t_cmb_step', value) is False
    assert _kept('core_cp_override', 'table') is True
    assert _kept('core_t_cmb_step', 'table') is False


MINIMAL = 'input/minimal.toml'


def test_a_config_file_sets_the_three_fields(tmp_path):
    """Values in the config file reach the loaded config unchanged."""
    from helpers import PROTEUS_ROOT

    text = (PROTEUS_ROOT / MINIMAL).read_text() + (
        '\n[interior_energetics.aragog]\n'
        '    core_bc = "core_module"\n'
        '\n[interior_energetics.aragog.core_module]\n'
        '    profile_source = "paleos"\n'
        '    core_cp_override = "table"\n'
        '    core_t_cmb_step = 5\n'
    )
    path = tmp_path / 'user.toml'
    path.write_text(text)
    cm = read_config_object(path).interior_energetics.aragog.core_module
    assert cm.profile_source == 'paleos'
    assert cm.core_cp_override == 'table'
    assert cm.core_t_cmb_step == 5 and isinstance(cm.core_t_cmb_step, int)


@pytest.mark.parametrize(
    ('name', 'value', 'refusal'),
    [
        ('core_cp_override', 'true', 'must be a number or a string'),
        ('core_t_cmb_step', 'true', 'must be a number'),
        ('core_t_cmb_step', '"10"', 'must be a number'),
    ],
)
def test_a_config_file_value_of_the_wrong_type_is_refused(tmp_path, name, value, refusal):
    """A TOML true, and a string for the grid step, are refused with the field name."""
    from helpers import PROTEUS_ROOT

    path = tmp_path / 'user.toml'
    path.write_text(
        (PROTEUS_ROOT / MINIMAL).read_text()
        + f'\n[interior_energetics.aragog.core_module]\n    {name} = {value}\n'
    )
    with pytest.raises(ValueError, match=f'{name}.*{refusal}|{refusal}.*{name}'):
        read_config_object(path)
    path.write_text(path.read_text().replace(f'= {value}', '= 5.0'))
    cm = read_config_object(path).interior_energetics.aragog.core_module
    assert getattr(cm, name) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ('source', 'core_bc', 'stratified', 'accretion', 'refused'),
    [
        ('gaussian', 'energy_balance', False, 'dummy', None),
        ('gaussian', 'core_module', True, None, None),
        ('paleos', 'core_module', False, None, None),
        ('paleos', 'energy_balance', False, None, 'needs interior_energetics.aragog.core_bc'),
        ('paleos', 'core_module', True, None, 'stratification'),
        ('paleos', 'core_module', False, 'dummy', 'accretion'),
    ],
)
def test_the_paleos_profile_requirements_are_checked(
    source, core_bc, stratified, accretion, refused
):
    """The PALEOS profile is read only by core_module, which it needs; it has no stratified
    shell in Aragog and no refit at a giant impact, so both are refused with it. The Gaussian
    profile passes in each case."""
    from types import SimpleNamespace

    from proteus.config._config import check_core_profile_source_requirements

    instance = SimpleNamespace(
        accretion=SimpleNamespace(module=accretion),
        interior_energetics=SimpleNamespace(
            module='aragog',
            aragog=SimpleNamespace(
                core_bc=core_bc,
                core_module=SimpleNamespace(profile_source=source, stratification=stratified),
            ),
        ),
    )
    if refused:
        with pytest.raises(ValueError, match=refused):
            check_core_profile_source_requirements(instance, None, None)
    else:
        assert check_core_profile_source_requirements(instance, None, None) is None
    instance.interior_energetics.module = 'spider'
    assert check_core_profile_source_requirements(instance, None, None) is None


def test_the_paleos_profile_with_accretion_is_refused_when_the_config_is_built():
    """Building a configuration runs the check: core_module with the PALEOS profile builds,
    and adding a giant-impact module fails."""
    import attrs
    from helpers import PROTEUS_ROOT

    cfg = read_config_object(PROTEUS_ROOT / 'input' / 'all_options.toml')
    aragog = cfg.interior_energetics.aragog
    core = attrs.evolve(aragog.core_module, profile_source='paleos')
    interior = attrs.evolve(
        cfg.interior_energetics,
        module='aragog',
        aragog=attrs.evolve(aragog, core_bc='core_module', core_module=core),
    )
    built = attrs.evolve(cfg, interior_energetics=interior)
    assert built.interior_energetics.aragog.core_module.profile_source == 'paleos'
    with pytest.raises(ValueError, match='accretion'):
        attrs.evolve(built, accretion=attrs.evolve(cfg.accretion, module='dummy'))
