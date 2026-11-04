"""Conductive surface skin from saturated condensate layers.

With ``atmos_clim.surface_d = surface_k = 'saturation'`` the conductive skin between
the magma and the atmosphere is the stack of condensate layers, such as graphite, that
saturate at the surface and float on it. Each layer of condensate c covers the whole
surface with the thickness ``d_c = m_c / (rho_c * 4 pi R_int**2)`` [m]. The layers
conduct heat in series, so the stack has the thermal resistance per unit area
``R = sum(d_c / k_c)`` [m2 K W-1] and acts as one skin of thickness ``d = sum(d_c)``
and conductivity ``k = d / R``: its flux ``k (T_magma - T_surf) / d`` equals
``(T_magma - T_surf) / R``. In steady conduction without internal heating the order
of the layers changes neither d nor k.

A stack with a resistance below ``SKIN_RESISTANCE_MIN`` is negligible. The surface is
then in the 'fixed' state (T_surf = T_magma), and the helpfile holds
``boundary_layer_thickness = boundary_layer_conductivity = 0``.

The densities (``condensate_density``) and conductivities (``condensate_conductivity``)
are constant. Sulfur above 388 K, water and sulfuric acid condense as liquids; a liquid
layer would convect, so treating it as purely conductive overestimates its insulation.
"""

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
    condensate_list,
    condensate_names,
)

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def skin_from_saturation(config: Config) -> bool:
    """Whether the conductive skin is derived from saturated condensate layers."""
    return config.atmos_clim.surface_d == 'saturation'


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
    for c in condensate_list:
        mass = float(hf_row[f'{c}_cd_kg'])
        activity = float(hf_row[f'{c}_cd_activity'])
        if not (np.isfinite(mass) and np.isfinite(activity)):
            raise ValueError(f'Non-finite condensate state for {c}: {mass} kg, a = {activity}')
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
    R_int = float(hf_row['R_int'])
    if not R_int > 0:
        raise ValueError(f'Surface radius must be positive to spread a layer, got {R_int} m')
    area = 4.0 * np.pi * R_int**2
    return {
        c: mass / (condensate_density[c] * area)
        for c, mass in saturated_condensate_mass(hf_row).items()
    }


def combine_layers(
    thickness: dict[str, float], conductivity: dict[str, float]
) -> tuple[float, float]:
    """Thickness and conductivity of one skin equivalent to stacked layers in series.

    Parameters
    ----------
    thickness : dict[str, float]
        Thickness of each layer [m].
    conductivity : dict[str, float]
        Thermal conductivity of each layer [W m-1 K-1], with the keys of `thickness`.

    Returns
    -------
    tuple[float, float]
        Total thickness ``d = sum(d_i)`` [m] and conductivity
        ``k = d / sum(d_i / k_i)`` [W m-1 K-1].
    """
    if not thickness:
        raise ValueError('Cannot combine an empty stack of layers')
    d_tot = 0.0
    resistance = 0.0
    for c, d_c in thickness.items():
        k_c = conductivity[c]
        if not (d_c > 0 and k_c > 0):
            raise ValueError(f'Layer {c} needs a positive thickness and conductivity')
        d_tot += d_c
        resistance += d_c / k_c
    return d_tot, d_tot / resistance


def resolve_surface_skin(config: Config, hf_row: dict) -> tuple[str, float, float]:
    """Surface state and conductive skin to give the atmosphere module.

    Parameters
    ----------
    config : Config
        Run configuration.
    hf_row : dict
        Current helpfile row, after the outgassing step.

    Returns
    -------
    tuple[str, float, float]
        Surface state, skin thickness [m] and skin conductivity [W m-1 K-1]. With
        numeric config values these are ``surf_state``, ``surface_d`` and
        ``surface_k``. With 'saturation' they are ('skin', d, k) of the condensate
        stack, or ('fixed', 0, 0) while the stack is negligible.
    """
    atm = config.atmos_clim
    if not skin_from_saturation(config):
        return atm.surf_state, float(atm.surface_d), float(atm.surface_k)
    thickness = condensate_layer_thickness(hf_row)
    if thickness:
        d, k = combine_layers(thickness, condensate_conductivity)
        if d / k >= SKIN_RESISTANCE_MIN:
            return 'skin', d, k
    return 'fixed', 0.0, 0.0


def update_surface_skin(config: Config, hf_row: dict) -> str:
    """Write the skin thickness and conductivity for this iteration into `hf_row`.

    Sets ``hf_row['boundary_layer_conductivity']`` [W m-1 K-1] and, with a skin from
    saturation, ``hf_row['boundary_layer_thickness']`` [m]. A numeric skin leaves the
    thickness to the interior module, which writes the configured value, or with the
    boundary interior its own thermal boundary layer.

    Returns
    -------
    str
        The surface state the atmosphere module solves with.
    """
    state, d, k = resolve_surface_skin(config, hf_row)
    hf_row['boundary_layer_conductivity'] = k
    if skin_from_saturation(config):
        hf_row['boundary_layer_thickness'] = d
        layers = condensate_layer_thickness(hf_row)
        listed = ', '.join(f'{condensate_names[c]} {d_c:.3e} m' for c, d_c in layers.items())
        log.info(
            'Surface skin from saturation: %s, d = %.3e m, k = %.3e W m-1 K-1 (%s)',
            state,
            d,
            k,
            listed or 'no layer',
        )
    return state


def effective_surf_state(config: Config, hf_row: dict) -> str:
    """Surface state for this iteration: 'fixed' while a saturation skin is absent."""
    if not skin_from_saturation(config):
        return config.atmos_clim.surf_state
    return 'skin' if hf_row['boundary_layer_thickness'] > 0 else 'fixed'


def skin_values(config: Config, hf_row: dict) -> tuple[float, float]:
    """Skin thickness [m] and conductivity [W m-1 K-1] in effect for this iteration."""
    if not skin_from_saturation(config):
        return float(config.atmos_clim.surface_d), float(config.atmos_clim.surface_k)
    return float(hf_row['boundary_layer_thickness']), float(
        hf_row['boundary_layer_conductivity']
    )


def agni_sol_type(config: Config, hf_row: dict) -> int:
    """AGNI surface boundary condition code for this iteration."""
    if not skin_from_saturation(config):
        return int(config.atmos_clim.surf_state_int)
    return surf_state_to_int(effective_surf_state(config, hf_row))
