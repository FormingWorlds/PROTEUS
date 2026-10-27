"""Tests for ``proteus.config._interior``: the mush rheology section ``Rheology``.

Covers the ``[interior_energetics.rheology]`` defaults (the Nur model's melt shear modulus
and critical melt volume fraction), a valid and an invalid input for every validator, the
'nur' profile requirements, and the block in ``input/all_options.toml``.

See docs/How-to/testing.md for the test standards.
"""

from __future__ import annotations

import attrs
import pytest

from proteus.config._interior import Rheology

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_rheology_defaults():
    """Defaults: G_inf = 1e10 Pa and no critical fraction.

    Only the Nur model is left, whose one elastic parameter is the critical melt volume fraction.
    """
    r = Rheology()
    assert r.melt_shear_modulus == pytest.approx(1.0e10)
    assert r.nur_critical_melt_volume_fraction is None
    assert set(attrs.fields_dict(Rheology)) == {
        'melt_shear_modulus',
        'nur_critical_melt_volume_fraction',
    }
    with pytest.raises(TypeError):
        Rheology(packing_fraction=0.55)


def test_melt_shear_modulus_validator_accepts_and_rejects():
    """G_inf: a low positive value is stored, zero and negative values raise."""
    assert Rheology(melt_shear_modulus=3.0e9).melt_shear_modulus == pytest.approx(3.0e9)
    assert Rheology(melt_shear_modulus=3.0e9) != Rheology()  # stored, not ignored
    for bad in (0.0, -1.0e10):
        with pytest.raises(ValueError):
            Rheology(melt_shear_modulus=bad)


def test_all_options_rheology_block_round_trips():
    """``all_options.toml`` carries the rheology block with the documented defaults."""
    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object

    cfg = read_config_object(PROTEUS_ROOT / 'input' / 'all_options.toml')
    r = cfg.interior_energetics.rheology
    assert isinstance(r, Rheology)
    assert attrs.asdict(r) == attrs.asdict(Rheology())


def test_nur_critical_fraction_accepts_value_or_none_and_rejects_range():
    """nur_critical_melt_volume_fraction: unset ('none' or default), or inside (0, 1)."""
    assert Rheology().nur_critical_melt_volume_fraction is None
    assert (
        Rheology(nur_critical_melt_volume_fraction='none').nur_critical_melt_volume_fraction
        is None
    )
    assert Rheology(
        nur_critical_melt_volume_fraction=0.45
    ).nur_critical_melt_volume_fraction == (pytest.approx(0.45))
    for bad in (0.0, 1.0, 1.4, -0.1):
        with pytest.raises(ValueError):
            Rheology(nur_critical_melt_volume_fraction=bad)


def test_nur_profile_needs_critical_fraction_aragog_and_paleos():
    """rheology_elastic_prf = 'nur' without the critical fraction, or without Aragog, fails.

    A valid configuration (aragog, PALEOS mantle EOS, v_c = 0.45) loads; the TOML literal
    "none" for the optional fraction loads as None.
    """
    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object

    cfg = read_config_object(PROTEUS_ROOT / 'input' / 'all_options.toml')
    assert cfg.interior_energetics.rheology.nur_critical_melt_volume_fraction is None
    ie = attrs.evolve(cfg.interior_energetics, rheology_elastic_prf='nur', module='aragog')
    with pytest.raises(ValueError, match='nur_critical_melt_volume_fraction'):
        attrs.evolve(cfg, interior_energetics=ie)
    rh = attrs.evolve(ie.rheology, nur_critical_melt_volume_fraction=0.45)
    ie_ok = attrs.evolve(ie, rheology=rh)
    assert (
        attrs.evolve(cfg, interior_energetics=ie_ok).interior_energetics.rheology_elastic_prf
        == 'nur'
    )
    with pytest.raises(ValueError, match='requires interior_energetics.module'):
        attrs.evolve(cfg, interior_energetics=attrs.evolve(ie_ok, module='dummy'))
