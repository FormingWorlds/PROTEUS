from __future__ import annotations

from attrs import define, field, validators

from ._atmos_clim import AtmosClim
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
    atmos_clim: AtmosClim
    escape: Escape
    interior: Interior
    outgas: Outgas
    delivery: Delivery

    def __getitem__(self, key: str):
        """This method adds a compatibility layer with the old-style dict."""
        from ._compatibility import COMPAT_MAPPING
        conv = COMPAT_MAPPING[key]

        if callable(conv):
            val = conv(self)
        else:
            val = self
            for part in conv:
                val = getattr(val, part)

        return val
