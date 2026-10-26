"""Unit tests for planet config validators (config/_planet.py).

Covers the initial entropy gradient rules: ini_dsdr must be finite in every
temperature mode, and a liquidus_super initial condition takes no positive
gradient.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import pytest

from proteus.config._planet import InteriorChem, Planet

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.parametrize('dsdr', [4.698e-6, 1e-12])
def test_positive_ini_dsdr_is_rejected_with_liquidus_super(dsdr):
    """A positive ini_dsdr, however small, lowers the deep entropy below the
    certified molten adiabat, so liquidus_super rejects it and the message
    names both fields."""
    with pytest.raises(ValueError) as exc:
        Planet(temperature_mode='liquidus_super', ini_dsdr=dsdr)

    msg = str(exc.value)
    assert 'planet.ini_dsdr' in msg
    assert 'liquidus_super' in msg


@pytest.mark.parametrize(
    ('mode', 'dsdr'),
    [
        ('liquidus_super', -4.698e-6),
        ('liquidus_super', 0.0),
        ('isentropic', 4.698e-6),
        ('adiabatic_from_cmb', 4.698e-6),
    ],
)
def test_ini_dsdr_accepted_where_it_cannot_erode_the_margin(mode, dsdr):
    """Zero and negative gradients keep liquidus_super valid (the boundary
    case 0 included), and other temperature modes keep any sign."""
    planet = Planet(temperature_mode=mode, ini_dsdr=dsdr)

    assert planet.ini_dsdr == pytest.approx(dsdr, abs=0.0)
    assert planet.temperature_mode == mode


@pytest.mark.parametrize('mode', ['liquidus_super', 'isentropic', 'adiabatic_from_cmb'])
@pytest.mark.parametrize('dsdr', [float('nan'), float('inf'), float('-inf')])
def test_non_finite_ini_dsdr_is_rejected_in_every_mode(mode, dsdr):
    """A non-finite ini_dsdr passes every comparison as False and gives a
    non-finite initial entropy profile in every mode, so it is rejected."""
    with pytest.raises(ValueError, match='is not finite') as exc:
        Planet(temperature_mode=mode, ini_dsdr=dsdr)
    assert 'planet.ini_dsdr' in str(exc.value)


@pytest.mark.parametrize('choice', ['mgsio3', 'bulk_BSE'])
def test_partition_coefficient_prescriptions_are_accepted(choice):
    """Both redox-tracker prescriptions are valid, and mgsio3 is the default."""
    assert InteriorChem(partition_coefficients=choice).partition_coefficients == choice
    assert InteriorChem().partition_coefficients == 'mgsio3'


@pytest.mark.parametrize('bad', ['bulk_bse', 'MgSiO3', 'schaefer', ''])
def test_unknown_partition_coefficient_prescription_is_rejected(bad):
    """A misspelt prescription fails at config load rather than at the first
    tracker call, deep into a run."""
    from proteus.interior_chem.redox import PARTITION_PRESCRIPTIONS

    with pytest.raises(ValueError) as exc:
        InteriorChem(partition_coefficients=bad)
    assert 'partition_coefficients' in str(exc.value)
    assert f"(got '{bad}')" in str(exc.value)
    # The config accepts exactly the sets the tracker implements.
    assert tuple(exc.value.args[2]) == PARTITION_PRESCRIPTIONS
    # Assignment after construction is validated too.
    chem = InteriorChem()
    with pytest.raises(ValueError):
        chem.partition_coefficients = bad
    assert chem.partition_coefficients == 'mgsio3'


@pytest.mark.parametrize('choice', ['deng2020', 'keane_table'])
def test_redox_melt_eos_choices_are_accepted(choice):
    """Both melt EOS of the redox pressure term are valid, and deng2020 is the
    default, so existing configurations keep their results."""
    assert InteriorChem(redox_dV_eos=choice).redox_dV_eos == choice
    assert InteriorChem().redox_dV_eos == 'deng2020'


@pytest.mark.parametrize('bad', ['Deng2020', 'keane', 'keane_deng', ''])
def test_unknown_redox_melt_eos_is_rejected(bad):
    """A misspelt melt EOS fails at config load, and the config accepts exactly
    the models eos_deng implements."""
    from proteus.interior_chem.eos_deng import MODELS

    with pytest.raises(ValueError) as exc:
        InteriorChem(redox_dV_eos=bad)
    assert 'redox_dV_eos' in str(exc.value)
    assert tuple(exc.value.args[2]) == MODELS
    chem = InteriorChem()
    with pytest.raises(ValueError):
        chem.redox_dV_eos = bad
    assert chem.redox_dV_eos == 'deng2020'


def test_redox_options_sit_in_the_interior_chem_section():
    """The tracker options are read from [planet.interior_chem], which a Planet
    builds with its defaults when the TOML omits the section."""
    planet = Planet(
        interior_chem=InteriorChem(metal_saturation=True, redox_dV_eos='keane_table')
    )
    assert planet.interior_chem.metal_saturation is True
    assert planet.interior_chem.redox_dV_eos == 'keane_table'
    default = Planet().interior_chem
    assert default == InteriorChem()
    assert default.ferric_fraction_initial == pytest.approx(0.1, abs=0.0)
    assert default.metal_saturation is False
    for key in ('ferric_fraction_initial', 'metal_saturation', 'partition_coefficients'):
        assert not hasattr(Planet(), key)


def test_redox_keys_left_directly_under_planet_are_unknown():
    """A config written before the [planet.interior_chem] section keeps the
    tracker options under [planet]; they are reported as unknown keys rather
    than silently replaced by the defaults, and the new location is accepted."""
    from proteus.config.orphans import find_key_problems

    old = {'planet': {'metal_saturation': True, 'redox_dV_eos': 'keane_table'}}
    orphans, mistyped = find_key_problems(old)
    assert orphans == ['planet.metal_saturation', 'planet.redox_dV_eos']
    assert mistyped == []
    new = {
        'planet': {'interior_chem': {'metal_saturation': True, 'redox_dV_eos': 'keane_table'}}
    }
    assert find_key_problems(new) == ([], [])
