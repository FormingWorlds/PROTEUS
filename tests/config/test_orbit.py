"""Tests for ``proteus.config._orbit``: the Obliqua melt bulk modulus and fluid1d_relax options.

Covers ``orbit.obliqua.solid.bulk_l``: a positive value in Pa, or 'none' for the PALEOS melt
value of the mush rheology; zero and negative values are rejected. Covers the whole-column
fluid model ``module_fluid = 'fluid1d_relax'`` and its options under ``orbit.obliqua.fluid``.

See docs/How-to/testing.md for the test standards.
"""

from __future__ import annotations

import pytest

from proteus.config._orbit import Obliqua, ObliquaFluid, ObliquaSolid

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_bulk_l_accepts_value_or_none_and_rejects_non_positive():
    """bulk_l: 1e9 Pa by default, 'none' converts to None, 0 and negative values raise."""
    assert ObliquaSolid().bulk_l == pytest.approx(1e9)
    assert ObliquaSolid(bulk_l='none').bulk_l is None
    assert ObliquaSolid(bulk_l=1.5e11).bulk_l == pytest.approx(1.5e11)
    with pytest.raises(ValueError):
        ObliquaSolid(bulk_l=0.0)
    with pytest.raises(ValueError):
        ObliquaSolid(bulk_l=-1e9)


def test_module_fluid_accepts_fluid1d_relax_and_rejects_unknown():
    """module_fluid: 'fluid1d_relax' is a valid choice next to the segment models; a typo raises."""
    assert Obliqua().module_fluid == 'fluid0d'
    assert Obliqua(module_fluid='fluid1d_relax').module_fluid == 'fluid1d_relax'
    with pytest.raises(ValueError):
        Obliqua(module_fluid='fluid1d-relax')


def test_fluid1d_relax_defaults():
    """Defaults: core switch 1e-5 rad/s, fluid layers below 1e8 Pa s, 10 sub-layers."""
    f = ObliquaFluid()
    assert f.omega_sw == pytest.approx(1e-5)
    assert f.visc_fluid == pytest.approx(1e8)
    assert f.n_sub == 10
    # the removed fluid overwrites are no longer configuration fields
    for name in (
        'mu_reg',
        'eta_eddy',
        'K_min',
        'K_max',
        'loss_fluid',
        'loss_tan',
        'loss_eta_max',
    ):
        assert not hasattr(f, name)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'omega_sw': 0.0},
        {'visc_fluid': 0.0},
        {'visc_fluid': -1e8},
        {'n_sub': 0},
    ],
    ids=[
        'omega_sw_zero',
        'visc_fluid_zero',
        'visc_fluid_negative',
        'n_sub_zero',
    ],
)
def test_fluid1d_relax_options_reject_invalid(kwargs):
    """A non-positive frequency or viscosity cutoff and zero sub-layers raise, naming the field."""
    with pytest.raises(ValueError) as exc:
        ObliquaFluid(**kwargs)
    assert next(iter(kwargs)) in str(exc.value)


def test_fluid1d_relax_options_accept_valid_values():
    """Positive values and one sub-layer are accepted as given."""
    f = ObliquaFluid(omega_sw=3e-5, visc_fluid=1e9, n_sub=1)
    assert (f.omega_sw, f.visc_fluid, f.n_sub) == (3e-5, 1e9, 1)
    assert f != ObliquaFluid()  # stored, not replaced by the defaults
