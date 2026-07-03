"""End-to-end noble gas outgassing through the atmodeller backend.

Runs the real atmodeller solver through PROTEUS's atmodeller wrapper with a
helium budget and checks that the wrapper wires the noble gas into the species
network, the mass constraints, and the output, and that the tracked helium
mass is conserved. atmodeller has carried the Jambon et al. (1986) noble gas
solubility laws since before this branch, so this test runs on the released
atmodeller rather than being gated on a new capability.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip('atmodeller')
pytest.importorskip('calliope')  # the wrapper module imports calliope constants

from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller
from proteus.utils.constants import element_list, gas_list, noble_gases

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]


def _atmodeller_config():
    """MagicMock config exercising the atmodeller backend, C-H-O-N-S plus He."""
    config = MagicMock()
    ac = config.outgas.atmodeller
    for name in ('H2O', 'CO2', 'H2', 'N2', 'S2', 'CO', 'CH4'):
        setattr(ac, f'solubility_{name}', None)
    for name in ('H2O', 'CO2', 'H2', 'CH4', 'CO'):
        setattr(ac, f'eos_{name}', None)
    ac.include_condensates = False
    ac.solver_mode = 'robust'
    ac.solver_max_steps = 256
    ac.solver_multistart = 20
    config.outgas.mass_thresh = 1e10
    config.outgas.solver_atol = 1e-6
    config.outgas.solver_rtol = 1e-4
    config.outgas.fO2_shift_IW = 0.0
    config.outgas.T_floor = 1200.0
    config.planet.fO2_source = 'user_constant'
    config.interior_struct.core_frac = 0.325
    return config


def _hf_row(He_kg):
    hf = {
        'M_mantle': 4.0e24,
        'M_int': 4.5e24,
        'M_planet': 5.97e24,
        'gravity': 9.81,
        'R_int': 6.37e6,
        'T_magma': 1800.0,
        'Phi_global': 1.0,
    }
    for e in element_list:
        hf[f'{e}_kg_total'] = 0.0
    hf['H_kg_total'] = 1.5e20
    hf['C_kg_total'] = 3.0e19
    hf['He_kg_total'] = He_kg
    for s in list(gas_list) + list(noble_gases):
        hf[f'{s}_bar'] = 0.0
    return hf


@pytest.mark.physics_invariant
def test_atmodeller_backend_conserves_helium_mass():
    """The atmodeller wrapper partitions a helium budget between atmosphere
    and melt and writes back masses that close against the supplied budget.
    Mass conservation is the contract: total helium out equals helium in.
    """
    He_kg = 3.0e16
    hf = _hf_row(He_kg)
    calc_surface_pressures_atmodeller({}, _atmodeller_config(), hf)

    reservoir = hf['He_kg_atm'] + hf['He_kg_liquid']
    assert reservoir == pytest.approx(He_kg, rel=1e-6)
    assert hf['He_kg_total'] == pytest.approx(He_kg, rel=1e-6)
    assert hf['He_kg_solid'] == 0.0
    # Both reservoirs are physical.
    assert hf['He_kg_atm'] > 0.0
    assert hf['He_kg_liquid'] > 0.0
    assert hf['He_bar'] > 0.0
    # Discrimination guard: the pressure-only atmospheric-mass formula the
    # wrapper falls back to would misweight helium (its 4 g/mol molar mass is
    # far below the atmosphere's mean), inflating the total well above the
    # budget. The mass-conserving path keeps it at the budget.
    assert hf['He_kg_total'] < 1.5 * He_kg

    # The C-H-O-N-S background is still solved and present.
    assert hf['H2O_bar'] > 0.0
    assert hf['P_surf'] > 0.0


@pytest.mark.physics_invariant
def test_atmodeller_helium_partitioning_scales_with_budget():
    """A larger helium budget yields a larger helium partial pressure and a
    larger dissolved helium mass, both still closing against the budget.
    """
    out = []
    for He_kg in (3.0e15, 3.0e16, 3.0e17):
        hf = _hf_row(He_kg)
        calc_surface_pressures_atmodeller({}, _atmodeller_config(), hf)
        out.append(hf)
        assert hf['He_kg_atm'] + hf['He_kg_liquid'] == pytest.approx(He_kg, rel=1e-6)

    bars = [h['He_bar'] for h in out]
    diss = [h['He_kg_liquid'] for h in out]
    assert bars[0] < bars[1] < bars[2]
    assert diss[0] < diss[1] < diss[2]
    # A trace helium budget must not be dropped by the major-volatile
    # threshold: even 3e15 kg (well below outgas.mass_thresh) is partitioned.
    assert out[0]['He_bar'] > 0.0


def test_atmodeller_no_noble_budget_leaves_helium_absent():
    """With no helium budget, the atmodeller solve carries no helium: the
    tracked helium mass stays zero, so a run without a noble budget is
    unaffected by the noble gas wiring.
    """
    hf = _hf_row(0.0)
    calc_surface_pressures_atmodeller({}, _atmodeller_config(), hf)
    assert hf.get('He_kg_total', 0.0) == 0.0
    assert hf.get('He_bar', 0.0) == 0.0
    # The C-H-O-N-S solve still ran.
    assert hf['H2O_bar'] > 0.0
