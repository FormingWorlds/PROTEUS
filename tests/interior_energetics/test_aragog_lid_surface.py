"""Coupling of the Aragog sub-grid lid: surface temperature, surface flux, conductance."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from proteus.atmos_clim.common import skin_conductance
from proteus.interior_energetics.aragog import _lid_surface_of_row, _surface_flux_of_step

pytestmark = pytest.mark.unit
YEAR = 3.15576e7


def _out(weight, dE=-1.0e24, r=6.4e6, flux_top=7.0):
    return SimpleNamespace(
        lid_weight=weight,
        step_dE_F_int_J=dE,
        r_basic=np.array([3.0e6, r]),
        heat_flux=np.array([1.0, flux_top]),
    )


def test_flux_mode_without_a_lid_books_the_flux_of_the_atmosphere():
    hf = {'F_atm': 42.0, 'Time': 100.0}
    out = _out(float('nan'))
    assert _surface_flux_of_step(out, hf, 'flux', 110.0) == 42.0
    assert _surface_flux_of_step(_out(0.0), hf, 'flux', 110.0) == 42.0


def test_grey_body_mode_books_the_surface_node():
    assert (
        _surface_flux_of_step(_out(1.0), {'F_atm': 42.0, 'Time': 0.0}, 'grey_body', 5.0) == 7.0
    )


@pytest.mark.parametrize('dt', [1.0, 10.0, 1000.0])
def test_a_live_lid_books_the_integrated_flux_of_the_step(dt):
    """``F_int A dt = -step_dE_F_int_J``: the energy of the step and the booked flux agree."""
    out = _out(0.7)
    F = _surface_flux_of_step(out, {'F_atm': 42.0, 'Time': 100.0}, 'flux', 100.0 + dt)
    area = 4.0 * np.pi * 6.4e6**2
    assert F * area * dt * YEAR == pytest.approx(-out.step_dE_F_int_J, rel=1e-12)


def test_a_state_with_no_step_books_the_flux_of_the_atmosphere():
    assert (
        _surface_flux_of_step(_out(1.0), {'F_atm': 42.0, 'Time': 100.0}, 'flux', 100.0) == 42.0
    )
    assert (
        _surface_flux_of_step(_out(1.0), {'F_atm': 42.0, 'Time': 100.0}, 'flux', None) == 42.0
    )


@pytest.mark.parametrize(
    'row, mode, on, want',
    [
        ({'T_surf': 300.0}, 'flux', True, 300.0),
        ({'T_surf': 300.0}, 'flux', False, None),
        ({'T_surf': 300.0}, 'grey_body', True, None),
        ({'T_surf': 0.0}, 'flux', True, None),
        ({}, 'flux', True, None),
    ],
)
def test_the_lid_takes_the_surface_of_the_atmosphere_only_in_flux_mode(row, mode, on, want):
    assert _lid_surface_of_row(row, mode, on) == want


def _cfg(k=2.0, d=0.01):
    return SimpleNamespace(atmos_clim=SimpleNamespace(surface_k=k, surface_d=d))


def test_conductance_is_the_skin_without_a_lid_and_a_series_sum_with_one():
    cfg = _cfg()
    assert skin_conductance(cfg, {}) == 200.0
    assert skin_conductance(cfg, {'lid_weight': 0.0, 'lid_conductance': 4.0}) == 200.0
    assert skin_conductance(cfg, {'lid_weight': 1.0, 'lid_conductance': 4.0}) == 4.0
    half = skin_conductance(cfg, {'lid_weight': 0.5, 'lid_conductance': 4.0})
    assert half == pytest.approx(1.0 / (0.5 / 4.0 + 0.5 / 200.0), rel=1e-14)
    assert skin_conductance(cfg, {'lid_weight': 1.0, 'lid_conductance': 0.0}) == 200.0
