from __future__ import annotations

import logging

import tomlkit
from attrs import asdict, define, field, validators

from ._atmos_chem import AtmosChem
from ._atmos_clim import AtmosClim
from ._converters import dict_replace_none
from ._delivery import Delivery
from ._escape import Escape
from ._interior import Interior
from ._observe import Observe
from ._orbit import Orbit
from ._outgas import Outgas
from ._params import Params
from ._planet import Planet
from ._star import Star
from ._struct import Struct

log = logging.getLogger('fwl.' + __name__)


def spada_zephyrus(instance, attribute, value):
    # using zephyrus
    #     zephyrus requires MORS + Spada
    if (instance.escape.module == 'zephyrus') and not (
        (instance.star.module == 'mors') and (instance.star.mors.tracks == 'spada')
    ):
        raise ValueError('ZEPHYRUS must be used with MORS and the Spada evolution tracks')


def instmethod_dummy(instance, attribute, value):
    # Instellation method 'inst' only support for dummy star module
    if (instance.orbit.instellation_method == 'inst') and not (instance.star.module == 'dummy'):
        raise ValueError("Instellation method can only be 'inst' when star.module=dummy ")


def instmethod_evolve(instance, attribute, value):
    # Orbital evolution not supported when installation_method is 'inst'
    if (instance.orbit.instellation_method == 'inst') and instance.orbit.evolve:
        raise ValueError(
            "Planet orbital evolution not supported for `instellation_method='inst'`"
        )


def satellite_evolve(instance, attribute, value):
    # Planetary orbital evolution not supported when also modelling satellite
    if instance.orbit.satellite and instance.orbit.evolve:
        raise ValueError(
            'Planet orbital evolution cannot be used simultaneously with a satellite'
        )


def tides_enabled_orbit(instance, attribute, value):
    # Tides in interior requires orbit module to not be None
    if (instance.interior_energetics.heat_tidal) and (instance.orbit.module is None):
        raise ValueError('Interior tidal heating requires an orbit module to be enabled')


def observe_resolved_atmosphere(instance, attribute, value):
    # Synthetic observations require a spatially resolved atmosphere profile
    if (instance.observe.synthesis is not None) and (instance.atmos_clim.module == 'dummy'):
        raise ValueError('Observational synthesis requires that atmos_clim != dummy')


def janus_escape_atmosphere(instance, attribute, value):
    # Using escape.zephyrus with JANUS requires params.stop.escape to be True
    if (
        (instance.escape.module == 'zephyrus')
        and (instance.atmos_clim.module == 'janus')
        and (instance.params.stop.escape is False)
    ):
        raise ValueError(
            'When using escape.zephyrus with JANUS, params.stop.escape must be True.'
        )


def mass_radius_valid(instance, attribute, value):
    """Cross-section validator: exactly one of planet_mass_tot or radius_int must be set."""
    from ._converters import none_if_none

    radius_int = none_if_none(instance.interior_struct.radius_int)
    mass_tot = none_if_none(instance.planet.planet_mass_tot)

    if (radius_int is None) and (mass_tot is None):
        raise ValueError('Must set one of `planet.planet_mass_tot` or `interior_struct.radius_int`')
    if (radius_int is not None) and (mass_tot is not None):
        raise ValueError(
            'Must set either `planet.planet_mass_tot` or `interior_struct.radius_int`, not both'
        )

    if mass_tot is not None:
        if mass_tot < 0:
            raise ValueError('The total planet mass must be > 0')
        if mass_tot > 20:
            raise ValueError('The total planet mass must be < 20 M_earth')

    if radius_int is not None:
        if radius_int < 0:
            raise ValueError('The interior radius must be > 0')
        if radius_int > 10:
            raise ValueError('The interior radius must be < 10 R_earth')


@define
class Config:
    """Root config parameters.

    Attributes
    ----------
    version: str
        Version of the configuration file.
    params: Params
        Parameters for code execution, output files, time-stepping, convergence.
    star: Star
        Stellar parameters, model selection.
    orbit: Orbit
        Orbital and star-system parameters.
    planet: Planet
        Bulk planet properties (mass).
    interior_struct: Struct
        Planetary structure calculation (radius, composition, Zalmoxis).
    interior_energetics: Interior
        Magma ocean / mantle energetics model parameters, model selection.
    atmos_clim: AtmosClim
        Planetary atmosphere climate parameters, model selection.
    atmos_chem: AtmosChem
        Planetary atmosphere chemistry parameters, model selection.
    escape: Escape
        Atmospheric escape parameters, model selection.
    outgas: Outgas
        Outgassing parameters (fO2, etc) and included volatiles.
    delivery: Delivery
        Initial volatile inventory, and delivery model selection.
    observe: Observe
        Synthetic observations.
    """

    version: str = field(validator=validators.in_(('2.0',)))

    params: Params
    star: Star
    orbit: Orbit = field(validator=(instmethod_dummy, instmethod_evolve, satellite_evolve))
    planet: Planet = field(validator=(mass_radius_valid,))
    interior_struct: Struct = field()
    interior_energetics: Interior = field(validator=(tides_enabled_orbit,))
    atmos_clim: AtmosClim
    atmos_chem: AtmosChem
    escape: Escape = field(validator=(spada_zephyrus,))
    outgas: Outgas
    delivery: Delivery
    observe: Observe

    def write(self, out: str):
        """
        Write configuration to a new TOML file.
        """

        # Convert to dictionary
        cfg = dict(asdict(self))

        # Replace None with "none"
        cfg = dict_replace_none(cfg)

        # Write to TOML file
        with open(out, 'w') as hdl:
            tomlkit.dump(cfg, hdl)
