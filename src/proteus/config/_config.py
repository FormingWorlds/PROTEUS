from __future__ import annotations

from attrs import define, field, validators

from ._atmos import Atmos
from ._delivery import Delivery
from ._escape import Escape
from ._interior import Interior
from ._orbit import Orbit
from ._outgas import Outgas
from ._params import Params
from ._star import Star
from ._struct import Struct


@define
class Config:
    """Root config"""
    version: str = field(validator=validators.in_(('2.0',)))
    author: str

    params: Params
    star: Star
    orbit: Orbit
    struct: Struct
    atmos: Atmos
    escape: Escape
    interior: Interior
    outgas: Outgas
    delivery: Delivery
