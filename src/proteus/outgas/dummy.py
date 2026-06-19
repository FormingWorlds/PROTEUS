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
from proteus.utils.helper import eval_gas_mmw

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Element -> dominant gas species mapping (simplified, oxidizing conditions)
# Each entry: (species_name, kg_element_per_kg_species)
_ELEMENT_TO_SPECIES = {
    'H': ('H2O', 2 * 1.008 / 18.015),  # H2O: 2H per molecule
    'C': ('CO2', 12.011 / 44.009),  # CO2: 1C per molecule
    'N': ('N2', 2 * 14.007 / 28.014),  # N2: 2N per molecule
    'S': ('SO2', 32.060 / 64.058),  # SO2: 1S per molecule
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
    Phi_global = float(hf_row['Phi_global'])
    gravity = float(hf_row['gravity'])
    R_int = float(hf_row['R_int'])
    if gravity <= 0 or R_int <= 0:
        raise ValueError(
            'Dummy outgassing needs a positive planet state: '
            f'gravity={gravity} m s-2, R_int={R_int} m'
        )
    area = 4.0 * np.pi * R_int**2

    # Save element totals before zeroing output keys (they are inputs)
    saved_element_kg = {}
    for element in element_list:
        saved_element_kg[element] = float(hf_row.get(f'{element}_kg_total', 0.0))

    # Initialize all expected keys to zero (M_atm is set by wrapper)
    for key in expected_keys(config):
        if key != 'M_atm':
            hf_row[key] = 0.0

    # Partition coefficient: fraction in atmosphere vs dissolved in melt.
    # Even at Phi=1 (fully molten), a fraction of volatiles is in the
    # atmosphere (solubility is finite). f_atm_floor ensures the
    # atmosphere is always non-empty so the coupling loop has a surface
    # pressure to work with.
    f_atm_floor = 0.1
    f_atm = f_atm_floor + (1.0 - f_atm_floor) * (1.0 - max(0.0, min(Phi_global, 1.0)))
    f_dissolved = 1.0 - f_atm

    # Convert elemental budgets to species masses
    species_kg_total = {}
    for element, (species, mass_frac) in _ELEMENT_TO_SPECIES.items():
        e_kg = saved_element_kg.get(element, 0.0)
        if e_kg > 0 and mass_frac > 0:
            species_kg_total[species] = e_kg / mass_frac  # kg of species

    # Partition each species between atmosphere and melt
    P_total = 0.0
    for species, kg_total in species_kg_total.items():
        kg_atm = f_atm * kg_total
        kg_liquid = f_dissolved * kg_total
        kg_solid = 0.0
        mmw = eval_gas_mmw(species)

        hf_row[f'{species}_kg_total'] = kg_total
        hf_row[f'{species}_kg_atm'] = kg_atm
        hf_row[f'{species}_kg_liquid'] = kg_liquid
        hf_row[f'{species}_kg_solid'] = kg_solid

        mol_total = kg_total / mmw
        hf_row[f'{species}_mol_total'] = mol_total
        hf_row[f'{species}_mol_atm'] = f_atm * mol_total
        hf_row[f'{species}_mol_liquid'] = f_dissolved * mol_total
        hf_row[f'{species}_mol_solid'] = 0.0

        # Partial pressure: P = m*g / A
        p_bar = kg_atm * gravity / area * 1e-5  # Pa -> bar
        hf_row[f'{species}_bar'] = p_bar
        P_total += p_bar

    hf_row['P_surf'] = P_total

    # VMRs from partial pressures
    for s in gas_list:
        if P_total > 0:
            hf_row[f'{s}_vmr'] = float(hf_row.get(f'{s}_bar', 0.0)) / P_total
        else:
            hf_row[f'{s}_vmr'] = 0.0

    # Mean molecular weight. An empty atmosphere (no volatile inventory)
    # has no meaningful MMW; store zero and let consumers check P_surf.
    if P_total > 0:
        mmw_sum = sum(hf_row.get(f'{s}_vmr', 0.0) * eval_gas_mmw(s) for s in gas_list)
        hf_row['atm_kg_per_mol'] = mmw_sum
    else:
        hf_row['atm_kg_per_mol'] = 0.0

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

    # Restore element totals (these are inputs, not outputs; the zeroing
    # above wiped them because expected_keys() includes _kg_total columns)
    for element, kg in saved_element_kg.items():
        if element != 'O':
            hf_row[f'{element}_kg_total'] = kg

    # Oxygen total. When the user supplied an O budget (O_mode != "ic_chemistry")
    # it is an input like the other elements and is restored verbatim; under
    # ic_chemistry there is no user O budget, so derive it from the
    # stoichiometric O in the outgassed species.
    saved_O = saved_element_kg.get('O', 0.0)
    if saved_O > 0.0:
        hf_row['O_kg_total'] = saved_O
    else:
        hf_row['O_kg_total'] = hf_row.get('O_kg_atm', 0.0) + hf_row.get('O_kg_liquid', 0.0)

    log.info(
        'Dummy outgas: P_surf=%.2f bar, Phi=%.3f, f_atm=%.3f, %d species',
        P_total,
        Phi_global,
        f_atm,
        len(species_kg_total),
    )
