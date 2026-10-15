"""Tests for the [atmos_chem.vulcan] config validators.

Covers `proteus.config._atmos_chem.Vulcan`: each VULCAN solver option
accepts a valid value and refuses an invalid one at construction.
"""

from __future__ import annotations

import pytest

from proteus.config._atmos_chem import Vulcan

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
