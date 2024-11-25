from __future__ import annotations

import logging

import tomlkit
from attrs import asdict, define, field, validators

from ._atmos_clim import AtmosClim
from ._converters import dict_replace_none
from ._delivery import Delivery
from ._escape import Escape
from ._interior import Interior
from ._orbit import Orbit
from ._outgas import Outgas
from ._params import Params
from ._star import Star
from ._struct import Struct

log = logging.getLogger('fwl.' + __name__)

def spada_zephyrus(instance, attribute, value):
    # using zephyrus
    #     zephyrus requires MORS + Spada
    if (instance.escape.module == 'zephyrus') and \
        not ( (instance.star.module == 'mors') and (instance.star.mors.tracks == 'spada')):
        raise ValueError('ZEPHYRUS must be used with MORS and the Spada evolution tracks')

@define
class Config:
    """Root config parameters.

    Attributes
    ----------
    version: str
        Version of the configuration file.
    author: str
        Authors of the configuration file.
    params: Params
        Parameters for code execution, output files, time-stepping, convergence.
    star: Star
        Stellar parameters, model selection.
    orbit: Orbit
        Orbital and star-system parameters.
    struct: Struct
        Planetary structure calculation (mass, radius).
    atmos_clim: AtmosClim
        Planetary atmosphere parameters, model selection.
    escape: Escape
        Atmospheric escape parameters, model selection.
    interior: Interior
        Magma ocean / mantle model parameters, model selection.
    outgas: Outgas
        Outgassing parameters (fO2, etc) and included volatiles.
    delivery: Delivery
        Initial volatile inventory, and delivery model selection.
    """

    version: str = field(validator=validators.in_(('2.0',)))
    author: str

    params: Params
    star: Star
    orbit: Orbit
    struct: Struct
    atmos_clim: AtmosClim
    escape: Escape = field(validator=(spada_zephyrus,))
    interior: Interior
    outgas: Outgas
    delivery: Delivery

    def write(self, out:str):
        """
        Write configuration to a new TOML file.
        """

        # Convert to dictionary
        cfg = dict(asdict(self))

        # Replace None with "none"
        cfg = dict_replace_none(cfg)

        # Write to TOML file
        with open(out,'w') as hdl:
            tomlkit.dump(cfg, hdl)
