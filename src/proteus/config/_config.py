from __future__ import annotations

import logging

import tomlkit
from attrs import asdict, define, field

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


CURRENT_CONFIG_VERSION = '3.0'


def valid_config_version(instance, attribute, value):
    if value != CURRENT_CONFIG_VERSION:
        raise ValueError(
            f'Config file version "{value}" is not compatible with this version of PROTEUS '
            f'(requires config_version = "{CURRENT_CONFIG_VERSION}"). '
            f'Please update your configuration file to match the current format. '
            f'See input/all_options.toml for the full reference.'
        )


def check_module_dependencies(instance, attribute, value):
    """Check that required external packages are importable for the selected modules."""
    import importlib

    checks = {
        'calliope': (instance.outgas.module == 'calliope', 'calliope',
                     'outgas.module = "calliope" requires the calliope package. '
                     'Install with: git clone git@github.com:FormingWorlds/CALLIOPE && pip install -e CALLIOPE/.'),
        'atmodeller': (instance.outgas.module == 'atmodeller', 'atmodeller',
                       'outgas.module = "atmodeller" requires the atmodeller package. '
                       'Install with: pip install atmodeller'),
        'boreas': (instance.escape.module == 'boreas', 'boreas',
                   'escape.module = "boreas" requires the boreas package. '
                   'Install with: pip install boreas'),
    }

    for name, (needed, pkg, msg) in checks.items():
        if needed:
            try:
                importlib.import_module(pkg)
            except ImportError as e:
                raise ImportError(f'{msg}\n  Original error: {e}') from e


def boreas_requires_atmosphere(instance, attribute, value):
    """BOREAS escape requires a radiative atmosphere (not dummy)."""
    if (instance.escape.module == 'boreas') and (instance.atmos_clim.module == 'dummy'):
        raise ValueError(
            'escape.module = "boreas" requires a radiative atmosphere model (agni or janus), '
            'not atmos_clim.module = "dummy". BOREAS needs per-level T/P/composition profiles.'
        )


def observe_resolved_atmosphere(instance, attribute, value):
    # Synthetic observations require a spatially resolved atmosphere profile
    if (instance.observe.synthesis is not None) and (instance.atmos_clim.module == 'dummy'):
        raise ValueError('Observational synthesis requires that atmos_clim != dummy')


def janus_escape_atmosphere(instance, attribute, value):
    # Using escape.zephyrus with JANUS requires params.stop.escape to be True
    if (
        (instance.escape.module == 'zephyrus')
        and (instance.atmos_clim.module == 'janus')
        and (not instance.params.stop.escape.enabled)
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
    config_version: str
        Version of the configuration file format.
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

    params: Params = field(factory=Params)
    star: Star = field(factory=Star)
    orbit: Orbit = field(factory=Orbit, validator=(instmethod_dummy, instmethod_evolve, satellite_evolve))
    planet: Planet = field(factory=Planet, validator=(planet_mass_valid,))
    interior_struct: Struct = field(factory=Struct)
    interior_energetics: Interior = field(factory=Interior, validator=(tides_enabled_orbit,))
    outgas: Outgas = field(factory=Outgas)
    atmos_clim: AtmosClim = field(factory=AtmosClim)
    atmos_chem: AtmosChem = field(factory=AtmosChem)
    escape: Escape = field(
        factory=Escape,
        validator=(spada_zephyrus, janus_escape_atmosphere, boreas_requires_atmosphere),
    )
    accretion: Accretion = field(factory=Accretion)
    observe: Observe = field(factory=Observe, validator=(observe_resolved_atmosphere,))

    config_version: str = field(
        default='3.0',
        validator=(valid_config_version, check_module_dependencies),
    )

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
