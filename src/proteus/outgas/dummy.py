"""Dummy outgas module for fast development and wiring tests.

Parameterized volatile partitioning without thermodynamic solver.
Splits elemental budgets between atmosphere and melt using a simple
melt-fraction-dependent partition coefficient. No chemical equilibrium,
no solubility laws, no real gas EOS.

Physics (0th order):
- Dissolved fraction scales with melt fraction: f_dissolved = Phi_global
- Atmospheric fraction: f_atm = 1 - Phi_global
- Species mapping: H -> H2O, C -> CO2, N -> N2, S -> SO2 (fixed stoichiometry)
- Pressure: thin-atmosphere P = m*g / (4*pi*R^2)
- MMW: mass-weighted mean from partial pressures
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.outgas.common import expected_keys
from proteus.utils.constants import element_list, gas_list

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Molar masses [kg/mol]
_MMW = {
    'H2O': 18.015e-3, 'CO2': 44.009e-3, 'O2': 31.998e-3, 'H2': 2.016e-3,
    'CH4': 16.04e-3, 'CO': 28.010e-3, 'N2': 28.014e-3, 'NH3': 17.031e-3,
    'S2': 64.12e-3, 'SO2': 64.058e-3, 'H2S': 34.08e-3,
    'SiO': 44.08e-3, 'SiO2': 60.08e-3, 'MgO': 40.30e-3, 'FeO2': 87.84e-3,
}

# Element -> dominant gas species mapping (simplified, oxidizing conditions)
# Each entry: (species_name, kg_element_per_kg_species)
_ELEMENT_TO_SPECIES = {
    'H': ('H2O', 2 * 1.008 / 18.015),     # H2O: 2H per molecule
    'C': ('CO2', 12.011 / 44.009),         # CO2: 1C per molecule
    'N': ('N2', 2 * 14.007 / 28.014),      # N2: 2N per molecule
    'S': ('SO2', 32.060 / 64.058),         # SO2: 1S per molecule
}


def calc_surface_pressures_dummy(dirs: dict, config: Config, hf_row: dict):
    """Compute volatile partitioning with parameterized model.

    Parameters
    ----------
    dirs : dict
        Directory paths.
    config : Config
        PROTEUS configuration.
    hf_row : dict
        Helpfile row (modified in place).
    """
    Phi_global = float(hf_row.get('Phi_global', 1.0))
    gravity = float(hf_row.get('gravity', 9.81))
    R_int = float(hf_row.get('R_int', 6.371e6))
    area = 4.0 * np.pi * R_int**2

    # Initialize all expected keys to zero (M_atm is set by wrapper)
    for key in expected_keys():
        if key != 'M_atm':
            hf_row[key] = 0.0

    # Partition coefficient: fraction dissolved in melt
    # Phi=1 (fully molten) -> most volatiles dissolved
    # Phi=0 (fully solid) -> all volatiles outgassed
    f_dissolved = max(0.0, min(Phi_global, 1.0))
    f_atm = 1.0 - f_dissolved

    # Convert elemental budgets to species masses
    species_kg_total = {}
    for element, (species, mass_frac) in _ELEMENT_TO_SPECIES.items():
        e_kg = float(hf_row.get(f'{element}_kg_total', 0.0))
        if e_kg > 0 and mass_frac > 0:
            species_kg_total[species] = e_kg / mass_frac  # kg of species

    # Partition each species between atmosphere and melt
    P_total = 0.0
    for species, kg_total in species_kg_total.items():
        kg_atm = f_atm * kg_total
        kg_liquid = f_dissolved * kg_total
        kg_solid = 0.0
        mmw = _MMW.get(species, 0.028)

        hf_row[f'{species}_kg_total'] = kg_total
        hf_row[f'{species}_kg_atm'] = kg_atm
        hf_row[f'{species}_kg_liquid'] = kg_liquid
        hf_row[f'{species}_kg_solid'] = kg_solid

        mol_total = kg_total / mmw if mmw > 0 else 0.0
        hf_row[f'{species}_mol_total'] = mol_total
        hf_row[f'{species}_mol_atm'] = f_atm * mol_total
        hf_row[f'{species}_mol_liquid'] = f_dissolved * mol_total
        hf_row[f'{species}_mol_solid'] = 0.0

        # Partial pressure: P = m*g / A
        if gravity > 0 and area > 0:
            p_bar = kg_atm * gravity / area * 1e-5  # Pa -> bar
        else:
            p_bar = 0.0
        hf_row[f'{species}_bar'] = p_bar
        P_total += p_bar

    hf_row['P_surf'] = P_total

    # VMRs from partial pressures
    for s in gas_list:
        if P_total > 0:
            hf_row[f'{s}_vmr'] = float(hf_row.get(f'{s}_bar', 0.0)) / P_total
        else:
            hf_row[f'{s}_vmr'] = 0.0

    # Mean molecular weight
    if P_total > 0:
        mmw_sum = sum(
            hf_row.get(f'{s}_vmr', 0.0) * _MMW.get(s, 0.028) for s in gas_list
        )
        hf_row['atm_kg_per_mol'] = mmw_sum
    else:
        hf_row['atm_kg_per_mol'] = 0.028  # default ~N2

    # Element reservoir masses (from species)
    _species_elements = {
        'H2O': {'H': 2 * 1.008 / 18.015, 'O': 15.999 / 18.015},
        'CO2': {'C': 12.011 / 44.009, 'O': 2 * 15.999 / 44.009},
        'N2': {'N': 1.0},
        'SO2': {'S': 32.060 / 64.058, 'O': 2 * 15.999 / 64.058},
    }
    for e in element_list:
        hf_row[f'{e}_kg_atm'] = 0.0
        hf_row[f'{e}_kg_liquid'] = 0.0
        hf_row[f'{e}_kg_solid'] = 0.0

    for species, kg_total in species_kg_total.items():
        elem_fracs = _species_elements.get(species, {})
        for e, frac in elem_fracs.items():
            hf_row[f'{e}_kg_atm'] += f_atm * kg_total * frac
            hf_row[f'{e}_kg_liquid'] += f_dissolved * kg_total * frac

    # Oxygen total (owned by outgas, set by fO2 buffer approximation)
    hf_row['O_kg_total'] = hf_row.get('O_kg_atm', 0.0) + hf_row.get('O_kg_liquid', 0.0)

    log.info(
        'Dummy outgas: P_surf=%.2f bar, Phi=%.3f, f_atm=%.3f, %d species',
        P_total, Phi_global, f_atm, len(species_kg_total),
    )
