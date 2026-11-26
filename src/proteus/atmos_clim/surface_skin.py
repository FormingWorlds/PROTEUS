# Conductive surface skin from saturated condensate layers.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.config._atmos_clim import surf_state_to_int
from proteus.utils.constants import (
    CONDENSATE_HELD_MASS_KG,
    CONDENSATE_SATURATED_ACTIVITY,
    SKIN_RESISTANCE_MIN,
    condensate_conductivity,
    condensate_density,
)

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def saturated_condensate_mass(hf_row: dict) -> dict[str, float]:
    """Mass of each condensate that forms a layer at the surface.

    A condensate forms a layer when its activity reaches
    ``CONDENSATE_SATURATED_ACTIVITY``, or when its mass is above
    ``CONDENSATE_HELD_MASS_KG``, which counts a layer held by
    ``outgas.prevent_desaturation`` at any activity.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row, with ``{c}_cd_kg`` [kg] and ``{c}_cd_activity`` [1].

    Returns
    -------
    dict[str, float]
        Condensed mass [kg] of each condensate with a layer, keyed by Hill formula.
    """
    masses = {}
    for c in condensate_density.keys():
        mass = float(hf_row[f'{c}_cd_kg'])
        activity = float(hf_row[f'{c}_cd_activity'])
        if not (np.isfinite(mass) and np.isfinite(activity)):
            log.warning(
                'Non-finite mass or activity for %s: mass = %s, activity = %s',
                c,
                mass,
                activity,
            )
            mass = 0.0
            activity = 0.0
        held = mass > CONDENSATE_HELD_MASS_KG
        saturated = activity >= CONDENSATE_SATURATED_ACTIVITY
        if mass > 0 and (held or saturated):
            masses[c] = mass
    return masses


def condensate_layer_thickness(hf_row: dict) -> dict[str, float]:
    """Thickness of each saturated condensate layer spread over the whole surface.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row, with ``R_int`` [m] and the condensate columns.

    Returns
    -------
    dict[str, float]
        Layer thickness ``m_c / (rho_c * 4 pi R_int**2)`` [m], keyed by Hill formula.
    """
    area = 4.0 * np.pi * float(hf_row['R_int']) ** 2
    return {
        c: mass / (condensate_density[c] * area)
        for c, mass in saturated_condensate_mass(hf_row).items()
    }


def update_surface_skin(config: Config, hf_row: dict) -> str:
    """Write the skin thickness and conductivity for this iteration into `hf_row`.

    This is the only writer of the two columns, which every atmosphere module reads.
    Numeric config values are written as they are. With 'saturation' the condensate
    layers are stacked in series, d = sum(d_i) and k = d / sum(d_i / k_i), and a stack
    with a resistance d / k below SKIN_RESISTANCE_MIN writes zeros (a fixed surface).

    Sets:
     - ``hf_row['boundary_layer_conductivity']`` [W m-1 K-1]
     - ``hf_row['boundary_layer_thickness']`` [m]

    Returns
    -------
    str
        The surface state used by the atmosphere module.
    """

    # is surface set by saturation?
    atm = config.atmos_clim
    if not atm.surface_d == 'saturation':
        state, d_tot, k_tot = atm.surf_state, float(atm.surface_d), float(atm.surface_k)

    else:
        # sum thickness and thermal resistance over the condensate stack
        thickness = condensate_layer_thickness(hf_row)
        d_sum = sum(thickness.values())
        resistance = sum(d_c / condensate_conductivity[c] for c, d_c in thickness.items())

        # a negligible stack (or none) leaves the surface fixed
        state, d_tot, k_tot = 'fixed', 0.0, 0.0
        if resistance >= SKIN_RESISTANCE_MIN:
            state, d_tot, k_tot = 'skin', d_sum, d_sum / resistance

        # inform
        log.info(
            'Surface skin from sat: %s, d = %.3e m, k = %.3e W m-1 K-1', state, d_tot, k_tot
        )

    # set values and return the skin state
    hf_row['boundary_layer_conductivity'] = k_tot
    hf_row['boundary_layer_thickness'] = d_tot
    return state


def effective_surf_state(config: Config, hf_row: dict) -> str:
    """Surface state for this iteration: 'fixed' while a saturation skin is absent."""
    if not (config.atmos_clim.surface_d == 'saturation'):
        return config.atmos_clim.surf_state
    return 'skin' if hf_row['boundary_layer_thickness'] > 0 else 'fixed'


def agni_sol_type(config: Config, hf_row: dict) -> int:
    """AGNI surface boundary condition code for this iteration."""
    if not (config.atmos_clim.surface_d == 'saturation'):
        return int(config.atmos_clim.surf_state_int)
    return surf_state_to_int(effective_surf_state(config, hf_row))
