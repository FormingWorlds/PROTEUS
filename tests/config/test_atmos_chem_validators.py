"""Tests for the [atmos_chem] and [atmos_chem.vulcan] config validators.

Covers `proteus.config._atmos_chem`: each VULCAN solver option, the
scheduling keys `mode` and `times`, and the initial composition `ini_mix`
accept a valid value and refuse an invalid one at construction.
"""

from __future__ import annotations

import pytest

from proteus.config._atmos_chem import AtmosChem, Vulcan

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Options that must be strictly positive: tolerances, timesteps [s], step counts,
# particle radius [cm].
_POSITIVE = (
    'yconv_min',
    'conv_step',
    'runtime',
    'dttry',
    'dt_min',
    'atol',
    'mtol',
    'mtol_conv',
    'flux_atol',
    'rtol',
    'post_conden_rtol',
    'r_p',
)


@pytest.mark.unit
@pytest.mark.parametrize('name', _POSITIVE)
def test_vulcan_positive_options_refuse_zero_and_negative(name):
    """A tolerance, timestep, step count or particle radius of zero or below is
    refused, and a small positive value is kept unchanged."""
    small = 3 if name == 'conv_step' else 1e-30
    assert getattr(Vulcan(**{name: small}), name) == small
    for bad in (0, -small):
        with pytest.raises(ValueError, match=name):
            Vulcan(**{name: bad})


@pytest.mark.unit
def test_vulcan_st_factor_lies_strictly_between_zero_and_one():
    """The steady-state test period starts at st_factor times the elapsed
    model time, so st_factor is refused at 0, at 1 and beyond either end."""
    assert Vulcan(st_factor=0.25).st_factor == pytest.approx(0.25)
    assert Vulcan(st_factor=0.999).st_factor == pytest.approx(0.999)
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match='st_factor'):
            Vulcan(st_factor=bad)


@pytest.mark.unit
def test_vulcan_use_vm_mol_refuses_the_upwind_scheme():
    """VULCAN raises at setup when its upwind molecular-diffusion scheme is on,
    so the config refuses use_vm_mol = true and keeps false."""
    assert Vulcan(use_vm_mol=False).use_vm_mol is False
    with pytest.raises(ValueError, match='use_vm_mol'):
        Vulcan(use_vm_mol=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('module', 'mode', 'expected'),
    [('none', 'none', None), ('dummy', 'none', None), ('dummy', 'offline', 'offline')],
    ids=['no_module_no_chemistry', 'manual_only', 'automatic_after_run'],
)
def test_atmos_chem_mode_accepts_none_and_offline(module, mode, expected):
    """mode = 'none' (TOML string) becomes None, and 'offline' is kept when a
    chemistry module is selected."""
    chem = AtmosChem(module=module, mode=mode)
    assert chem.mode == expected
    assert chem.times == 'end'  # default snapshot set


@pytest.mark.unit
def test_atmos_chem_mode_refuses_online_as_not_implemented():
    """Coupled chemistry is not implemented, so mode = 'online' is refused with
    a message saying so, with or without a module."""
    for module in ('dummy', 'none'):
        with pytest.raises(ValueError, match='not yet implemented'):
            AtmosChem(module=module, mode='online')
    with pytest.raises(ValueError, match='mode'):
        AtmosChem(module='dummy', mode='manually')  # the retired `when` value


@pytest.mark.unit
def test_atmos_chem_offline_mode_requires_a_module():
    """Automatic chemistry with no module would do nothing, so it is refused;
    the same module with mode none is accepted."""
    with pytest.raises(ValueError, match="requires atmos_chem.module != 'none'"):
        AtmosChem(module='none', mode='offline')
    assert AtmosChem(module='none', mode='none').mode is None


@pytest.mark.unit
def test_atmos_chem_times_accepts_end_all_and_a_list_of_years():
    """'end' and 'all' are kept as strings; a list of times [yr] is kept in
    order and converted to floats, including t = 0 and an integer entry."""
    assert AtmosChem(times='end').times == 'end'
    assert AtmosChem(times='all').times == 'all'
    times = AtmosChem(times=[0, 12.5, 3]).times
    assert times == pytest.approx([0.0, 12.5, 3.0])
    assert all(type(t) is float for t in times)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('times', 'match'),
    [
        ('foo', "'end', 'all' or a list"),
        ([], 'empty list'),
        ([1.0, -1.0], 'finite and >= 0'),
        ([float('nan')], 'finite and >= 0'),
        ([float('inf')], 'finite and >= 0'),
    ],
    ids=['unknown_keyword', 'empty_list', 'negative_time', 'nan_time', 'infinite_time'],
)
def test_atmos_chem_times_refuses_invalid_values(times, match):
    """An unknown keyword, an empty list, or a time that is negative or not
    finite is refused, naming the problem."""
    with pytest.raises(ValueError, match=match):
        AtmosChem(times=times)
    # The default stays valid, so the refusal comes from the value given
    assert AtmosChem().times == 'end'


@pytest.mark.unit
def test_vulcan_ini_mix_previous_requires_a_fixed_surface():
    """ini_mix = 'previous' is accepted with fix_surf = true and refused with
    fix_surf = false; the other options are accepted with either."""
    assert Vulcan(ini_mix='previous', fix_surf=True).ini_mix == 'previous'
    with pytest.raises(ValueError, match='requires fix_surf = true'):
        Vulcan(ini_mix='previous', fix_surf=False)
    for option in ('profile', 'outgas'):
        assert Vulcan(ini_mix=option, fix_surf=False).ini_mix == option


@pytest.mark.unit
def test_vulcan_ini_mix_refuses_an_unknown_option():
    """An ini_mix outside profile, outgas and previous is refused."""
    with pytest.raises(ValueError, match='ini_mix'):
        Vulcan(ini_mix='bogus')
    assert Vulcan().ini_mix == 'profile'


@pytest.mark.unit
def test_vulcan_defaults_start_from_the_profile_with_a_free_surface():
    """By default VULCAN starts from the climate profile with free surface
    mixing ratios, so 'previous', which needs a fixed surface, has to be
    chosen together with fix_surf = true."""
    vul = Vulcan()
    assert vul.ini_mix == 'profile'
    assert vul.fix_surf is False
    with pytest.raises(ValueError, match='requires fix_surf = true'):
        Vulcan(ini_mix='previous')
