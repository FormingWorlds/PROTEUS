"""Smoke test of ``proteus.outgas.lid_seal`` against the real CALLIOPE solver.

Invariants checked: the partial pressures, mixing ratios, mean molar mass and
surface pressure that the leak rebuilds from the species masses and moles equal
those of a solved CALLIOPE row; the dissolved equilibrium of the sealed
reservoir uses the solubility laws of the solve, with its arguments.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.outgas.calliope import calc_surface_pressures
from proteus.outgas.lid_seal import _refresh_atmosphere, equilibrium_fractions
from proteus.utils.constants import gas_list, soluble_gases

from ._calliope_helpers import _cold_start_config, _surface_pressure_hf_row
from ._from_o_budget_helpers import _earth_hf_row, _make_from_o_budget_config

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]


@pytest.mark.physics_invariant
def test_pressures_from_masses_match_a_solved_calliope_row():
    """On a CALLIOPE row, rebuilding the atmosphere from its own masses and
    moles gives back its partial pressures and mixing ratios to 1e-10, and a
    mass-weighted (not mole-weighted) mixing ratio misses by more than 1 %."""
    row = _surface_pressure_hf_row()
    np.random.set_state(np.random.RandomState(1).get_state())
    calc_surface_pressures({'output': '/tmp/test'}, _cold_start_config(), row)
    rebuilt = dict(row)
    _refresh_atmosphere(rebuilt)
    present = [s for s in gas_list if row.get(f'{s}_bar', 0.0) > 1e-8 * row['P_surf']]
    assert len(present) >= 3
    for s in present:
        assert rebuilt[f'{s}_bar'] == pytest.approx(row[f'{s}_bar'], rel=1e-10), s
        assert rebuilt[f'{s}_vmr'] == pytest.approx(row[f'{s}_vmr'], rel=1e-10), s
    assert rebuilt['P_surf'] == pytest.approx(row['P_surf'], rel=1e-10)
    assert rebuilt['atm_kg_per_mol'] == pytest.approx(row['atm_kg_per_mol'], rel=1e-10)
    m_atm = sum(row.get(f'{s}_kg_atm', 0.0) for s in gas_list)
    by_mass = {s: row[f'{s}_kg_atm'] / m_atm for s in present}
    assert max(abs(by_mass[s] / row[f'{s}_vmr'] - 1.0) for s in present) > 0.01


def _dissolved_parity(config, row):
    """Solve the row with CALLIOPE; compare X_eq of the seal with the solved melt.

    Returns the species with dissolved mass, each equal to relative 1e-12.
    """
    np.random.set_state(np.random.RandomState(1).get_state())
    calc_surface_pressures({'output': '/tmp/test'}, config, row)
    bar = {s: row.get(f'{s}_bar', 0.0) for s in gas_list}
    temp = max(row['T_magma'], config.outgas.T_floor)
    x = equilibrium_fractions(config, bar, row['P_surf'], temp, row['fO2_shift_IW_derived'])
    melt = row['M_mantle'] * row['Phi_global']
    for s in soluble_gases:
        assert x[s] == pytest.approx(row.get(f'{s}_kg_liquid', 0.0) / melt, rel=1e-12), s
    return {s for s in soluble_gases if x[s] > 0.0}


@pytest.mark.physics_invariant
@pytest.mark.parametrize(('shift', 't_magma'), [(-2.0, 1800.0), (2.0, 1000.0)])
def test_equilibrium_fractions_are_the_dissolved_fractions_of_a_solved_row(shift, t_magma):
    """At IW-2 and IW+2, with He and Ar on and T_magma above and below
    T_floor (1200 K), the seal's X_eq of each species equals the dissolved mass
    of the solve over M_mantle Phi_global, to 1e-12."""
    config = _cold_start_config({'He': True, 'Ar': True}, He_budget=3.0e16)
    config.outgas.fO2_shift_IW = shift
    config.planet.elements.Ar_budget = 1.0e16
    row = {**_surface_pressure_hf_row(), 'Phi_global': 0.4, 'T_magma': t_magma}
    solved = _dissolved_parity(config, row)
    assert solved == {'H2O', 'CO2', 'CO', 'CH4', 'N2', 'S2', 'He', 'Ar'}
    assert row['fO2_shift_IW_derived'] == shift


@pytest.mark.physics_invariant
def test_equilibrium_fractions_take_the_derived_fo2_under_from_o_budget():
    """Under from_O_budget the IW offset of the solve is fO2_shift_IW_derived;
    with it the seal's X_eq equals the solved melt, and with the configured
    offset the N2 and S2 fractions differ by more than 1 %."""
    legacy = _make_from_o_budget_config(fO2_shift_IW=1.0)
    legacy.planet.fO2_source = 'user_constant'
    row = _earth_hf_row()
    calc_surface_pressures({'output': '/tmp/test'}, legacy, row)
    config = _make_from_o_budget_config(fO2_shift_IW=3.0)
    row = {**_earth_hf_row(O_kg_total=row['O_kg_total']), 'Phi_global': 0.6}
    assert {'H2O', 'CO2', 'N2', 'S2'} <= _dissolved_parity(config, row)
    assert abs(row['fO2_shift_IW_derived'] - 3.0) > 1.0
    bar = {s: row.get(f'{s}_bar', 0.0) for s in gas_list}
    x = equilibrium_fractions(config, bar, row['P_surf'], 1800.0, 3.0)
    melt = row['M_mantle'] * row['Phi_global']
    for s in ('N2', 'S2'):
        assert abs(x[s] / (row[f'{s}_kg_liquid'] / melt) - 1.0) > 0.01, s
