"""Atmodeller outgassing module wrapper for PROTEUS.

Wraps the atmodeller package (Bower+2025, ApJ 995:59) to compute
volatile partitioning between atmosphere and magma ocean using
thermodynamically consistent equilibrium chemistry with real gas
EOS and non-ideal solubility laws.

Replaces CALLIOPE as an alternative outgassing module when
``config.outgas.module = 'atmodeller'``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def calc_surface_pressures_atmodeller(dirs: dict, config: Config, hf_row: dict):
    """Compute volatile partitioning using atmodeller.

    Solves for chemical equilibrium between the magma ocean and
    atmosphere, accounting for real gas EOS, solubility laws, and
    optionally condensation. Updates hf_row with partial pressures,
    VMRs, dissolved masses, and total surface pressure.

    Parameters
    ----------
    dirs : dict
        Directory paths.
    config : Config
        PROTEUS configuration object.
    hf_row : dict
        Current helpfile row (modified in place).
    """
    from atmodeller import (
        ChemicalSpecies,
        EquilibriumModel,
        Planet,
        SpeciesNetwork,
    )
    from atmodeller.solubility import get_solubility_models
    from atmodeller.thermodata import IronWustiteBuffer

    from proteus.utils.constants import M_earth, gas_list

    atm_config = config.outgas.atmodeller
    solubility_models = get_solubility_models()

    # Build species network
    species_list = []

    # Map PROTEUS gas species to atmodeller species with solubility
    _sol_map = {
        'H2O': atm_config.solubility_H2O,
        'CO2': atm_config.solubility_CO2,
        'H2': atm_config.solubility_H2,
        'N2': atm_config.solubility_N2,
        'S2': atm_config.solubility_S2,
    }

    # Species that atmodeller handles as gases
    _atm_gas_species = {
        'H2O': 'H2O',
        'H2': 'H2',
        'CO2': 'CO2',
        'CO': 'CO',
        'CH4': 'CH4',
        'N2': 'N2',
        'S2': 'S2',
        'SO2': 'SO2',
    }

    for proteus_name, atm_name in _atm_gas_species.items():
        sol_key = _sol_map.get(proteus_name)
        if sol_key and sol_key in solubility_models:
            species_list.append(
                ChemicalSpecies.create_gas(
                    atm_name, solubility=solubility_models[sol_key]
                )
            )
        else:
            species_list.append(ChemicalSpecies.create_gas(atm_name))

    # Add condensates if enabled
    if atm_config.include_condensates:
        try:
            species_list.append(ChemicalSpecies.create_condensed('C'))
        except Exception:
            pass  # Graphite not available in all versions

    species = SpeciesNetwork(tuple(species_list))
    model = EquilibriumModel(species)

    # Build planet state
    M_planet = float(hf_row.get('M_planet', 1.0 * M_earth))
    R_int = float(hf_row.get('R_int', 6.371e6))
    T_magma = float(hf_row.get('T_magma', 3000.0))
    Phi_global = float(hf_row.get('Phi_global', 1.0))

    # Core mass fraction from config
    if hasattr(config.struct, 'zalmoxis') and config.struct.zalmoxis:
        cmf = config.struct.zalmoxis.coremassfrac
    else:
        cmf = config.struct.corefrac

    planet = Planet(
        planet_mass=M_planet,
        core_mass_fraction=cmf,
        mantle_melt_fraction=Phi_global,
        surface_radius=R_int,
        temperature=T_magma,
        pressure=np.nan,  # Thin-atmosphere approximation
    )

    # Fugacity constraint: oxygen via IW buffer
    fugacity_constraints = {
        'O2_g': IronWustiteBuffer(config.outgas.fO2_shift_IW),
    }

    # Mass constraints from elemental budgets
    mass_constraints = {}
    for element in ('H', 'C', 'N', 'S'):
        key = f'{element}_kg_total'
        mass_kg = float(hf_row.get(key, 0.0))
        if mass_kg > config.outgas.mass_thresh:
            mass_constraints[element] = mass_kg

    if not mass_constraints:
        log.warning('No volatile element budgets above threshold; skipping atmodeller')
        return

    log.info(
        'Atmodeller solve: T=%.0f K, Phi=%.2f, elements=%s',
        T_magma,
        Phi_global,
        {k: f'{v:.2e}' for k, v in mass_constraints.items()},
    )

    # Solve equilibrium
    try:
        model.solve(
            state=planet,
            fugacity_constraints=fugacity_constraints,
            mass_constraints=mass_constraints,
            solver=atm_config.solver_mode,
        )
    except Exception as e:
        log.error('Atmodeller solve failed: %s', e)
        raise

    # Extract results
    output = model.output
    quick_look = output.quick_look()
    total_P = float(output.total_pressure())

    log.info('Atmodeller result: P_total=%.2f bar', total_P)

    # Map atmodeller output back to hf_row
    _reverse_map = {v: k for k, v in _atm_gas_species.items()}

    P_total = 0.0
    for atm_name, p_bar in quick_look.items():
        proteus_name = _reverse_map.get(atm_name.replace('_g', ''))
        if proteus_name and proteus_name in gas_list:
            p_val = float(p_bar)
            hf_row[f'{proteus_name}_bar'] = p_val
            P_total += p_val

            # Atmospheric mass: P = m*g / (4*pi*R^2)
            gravity = float(hf_row.get('gravity', 9.81))
            area = 4.0 * np.pi * R_int**2
            if gravity > 0 and area > 0:
                hf_row[f'{proteus_name}_kg_atm'] = p_val * 1e5 * area / gravity
            else:
                hf_row[f'{proteus_name}_kg_atm'] = 0.0

    # Total surface pressure
    hf_row['P_surf'] = P_total

    # VMRs
    if P_total > 0:
        for s in gas_list:
            hf_row[f'{s}_vmr'] = float(hf_row.get(f'{s}_bar', 0.0)) / P_total
    else:
        for s in gas_list:
            hf_row[f'{s}_vmr'] = 0.0

    # Dissolved masses: compute from total - atmospheric
    for s in gas_list:
        kg_total = float(hf_row.get(f'{s}_kg_total', 0.0))
        kg_atm = float(hf_row.get(f'{s}_kg_atm', 0.0))
        hf_row[f'{s}_kg_liquid'] = max(0.0, kg_total - kg_atm)
        hf_row[f'{s}_kg_solid'] = 0.0

    # Mean molecular weight (approximate from VMRs)
    _mmw = {
        'H2O': 18.015e-3, 'CO2': 44.01e-3, 'H2': 2.016e-3,
        'CO': 28.01e-3, 'CH4': 16.04e-3, 'N2': 28.01e-3,
        'S2': 64.13e-3, 'SO2': 64.07e-3, 'O2': 32.0e-3,
        'H2S': 34.08e-3, 'NH3': 17.03e-3,
    }
    mmw = sum(
        float(hf_row.get(f'{s}_vmr', 0.0)) * _mmw.get(s, 28.0e-3)
        for s in gas_list
    )
    if mmw > 0:
        hf_row['atm_kg_per_mol'] = mmw

    log.info(
        'Atmodeller: P_surf=%.2f bar, MMW=%.3f g/mol',
        P_total,
        mmw * 1e3,
    )
