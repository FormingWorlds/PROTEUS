"""Smoke test of ``proteus.outgas.lid_seal`` against the real CALLIOPE solver.

Invariant checked: the partial pressures, mixing ratios, mean molar mass and
surface pressure that the leak rebuilds from the species masses and moles equal
those of a solved CALLIOPE row.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.outgas.calliope import calc_surface_pressures
from proteus.outgas.lid_seal import _refresh_atmosphere
from proteus.utils.constants import gas_list

from ._calliope_helpers import _cold_start_config, _surface_pressure_hf_row

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]


@pytest.mark.physics_invariant
def test_pressures_from_masses_match_a_solved_calliope_row():
    """V5: on a CALLIOPE row, rebuilding the atmosphere from its own masses and
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
