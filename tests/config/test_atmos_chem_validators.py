"""Tests for the [atmos_chem.vulcan] config validators.

Covers `proteus.config._atmos_chem.Vulcan`: each VULCAN solver option
accepts a valid value and refuses an invalid one at construction.
"""

from __future__ import annotations

import pytest

from proteus.config._atmos_chem import Vulcan

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Options that must be strictly positive: tolerances, timesteps [s], step counts.
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
)


@pytest.mark.unit
@pytest.mark.parametrize('name', _POSITIVE)
def test_vulcan_positive_options_refuse_zero_and_negative(name):
    """A tolerance, timestep or step count of zero or below is refused, and a
    small positive value is kept unchanged."""
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
def test_vulcan_solver_defaults_match_the_previous_hard_coded_values():
    """The defaults keep the values the wrapper set before the options were
    exposed, so an existing config runs VULCAN unchanged."""
    v = Vulcan()
    assert (v.st_factor, v.conv_step, v.yconv_min) == (0.5, 100, 0.5)
    assert (v.runtime, v.dttry, v.dt_min) == (1e22, 1e-8, 1e-10)
    assert (v.atol, v.mtol, v.mtol_conv) == (0.1, 1e-22, 1e-20)
    assert (v.flux_atol, v.rtol, v.post_conden_rtol) == (1.0, 0.9, 0.1)
    assert (v.use_vm_mol, v.use_vz, v.use_condense, v.use_settling) == (
        False,
        True,
        False,
        False,
    )
