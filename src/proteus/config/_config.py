from __future__ import annotations

import logging

import tomlkit
from attrs import asdict, define, field, validators

from ._accretion import Accretion
from ._atmos_chem import AtmosChem
from ._atmos_clim import AtmosClim
from ._converters import dict_replace_none
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


def planet_mass_valid(instance, attribute, value):
    """Validate that mass_tot is set and within range."""
    from ._converters import none_if_none

    mass_tot = none_if_none(instance.planet.mass_tot)

    if mass_tot is None:
        raise ValueError('`planet.mass_tot` must be set')
    if mass_tot <= 0:
        raise ValueError('The total planet mass must be > 0')
    if mass_tot > 20:
        raise ValueError('The total planet mass must be < 20 M_earth')


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
        Bulk planet properties (mass, initial volatile inventory).
    interior_struct: Struct
        Planetary structure calculation (radius, composition, Zalmoxis).
    interior_energetics: Interior
        Magma ocean / mantle energetics model parameters, model selection.
    outgas: Outgas
        Outgassing parameters (fO2, etc) and included volatiles.
    atmos_clim: AtmosClim
        Planetary atmosphere climate parameters, model selection.
    atmos_chem: AtmosChem
        Planetary atmosphere chemistry parameters, model selection.
    escape: Escape
        Atmospheric escape parameters, model selection.
    accretion: Accretion
        Late accretion / delivery model selection.
    observe: Observe
        Synthetic observations.
    """

    params: Params
    star: Star
    orbit: Orbit = field(validator=(instmethod_dummy, instmethod_evolve, satellite_evolve))
    planet: Planet = field(validator=(planet_mass_valid,))
    interior_struct: Struct = field()
    interior_energetics: Interior = field(validator=(tides_enabled_orbit,))
    outgas: Outgas
    atmos_clim: AtmosClim
    atmos_chem: AtmosChem
    escape: Escape = field(validator=(spada_zephyrus,))
    accretion: Accretion
    observe: Observe

    version: str = field(default='3.0', validator=validators.in_(('3.0',)))

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
