"""Tests for ``proteus.config._interior``: the mush rheology section ``Rheology``.

Covers the ``[interior_energetics.rheology]`` defaults, a valid and an invalid input for
every validator, the spherical-grain restriction on the crystal aspect ratio, and the
block in ``input/all_options.toml``.

See docs/How-to/testing.md for the test standards.
"""

from __future__ import annotations

import attrs
import pytest

from proteus.config._interior import Rheology

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_rheology_defaults_match_the_mush_construction():
    """Defaults: equant crystals, phi_m1 = 0.55, framework at 0.8 phi_m, G_inf = 1e10 Pa."""
    r = Rheology()
    assert r.aspect_ratio == pytest.approx(1.0)
    assert r.packing_fraction == pytest.approx(0.55)
    assert r.framework_fraction == pytest.approx(0.8)
    assert r.gompertz_width == pytest.approx(0.076)
    assert r.melt_shear_modulus == pytest.approx(1.0e10)
    assert r.melt_weakening == pytest.approx(26.0)
    # the framework onset lies at a larger melt fraction than the viscous onset
    assert 1.0 - r.framework_fraction * r.packing_fraction > 1.0 - r.packing_fraction


@pytest.mark.parametrize(
    'field, good, bad',
    [
        ('framework_fraction', 1.0, 0.0),
        ('framework_fraction', 0.5, 1.2),
        ('gompertz_width', 0.03, 0.0),
        ('gompertz_width', 0.5, 1.0),
        ('melt_shear_modulus', 3.0e9, 0.0),
        ('melt_weakening', 0.0, -1.0),
        ('packing_fraction', 0.64, 1.0),
        ('packing_fraction', 0.3, 0.0),
        ('packing_width', 1.13, 0.0),
    ],
    ids=[
        'framework_at_max_packing_vs_zero',
        'framework_half_vs_above_packing',
        'narrow_width_vs_zero',
        'wide_width_vs_whole_range',
        'low_G_inf_vs_zero',
        'no_weakening_vs_negative',
        'random_close_packing_vs_one',
        'loose_packing_vs_zero',
        'smooth_particle_width_vs_zero',
    ],
)
def test_rheology_validators_accept_and_reject(field, good, bad):
    """Each bounded field accepts a value inside its range and rejects one outside."""
    assert getattr(Rheology(**{field: good}), field) == pytest.approx(good)
    with pytest.raises(ValueError):
        Rheology(**{field: bad})


def test_aspect_ratio_restricted_to_spherical_grains():
    """Only r_p = 1 is accepted: the SCS tables assume spherical grains and inclusions."""
    assert Rheology(aspect_ratio=1.0).aspect_ratio == pytest.approx(1.0)
    with pytest.raises(ValueError, match='spherical grains'):
        Rheology(aspect_ratio=2.0)
    with pytest.raises(ValueError, match='spherical grains'):
        attrs.evolve(Rheology(), aspect_ratio=0.5)


def test_all_options_rheology_block_round_trips():
    """``all_options.toml`` carries the rheology block with the documented defaults."""
    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object

    cfg = read_config_object(PROTEUS_ROOT / 'input' / 'all_options.toml')
    r = cfg.interior_energetics.rheology
    assert isinstance(r, Rheology)
    assert attrs.asdict(r) == attrs.asdict(Rheology())
