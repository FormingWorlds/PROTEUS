from __future__ import annotations

import warnings

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
            hint = "`Proteus.config.xxx`."
        else:
            val = self
            for part in conv:
                val = getattr(val, part)
            new_key = '.'.join(conv)
            hint = f"`Proteus.config.{new_key}`"

        message = (
            f"Calling `OPTIONS['{key}']` via OPTIONS is deprecated, "
            f"please use the class-based config instead: {hint}."
            "See https://github.com/FormingWorlds/PROTEUS/issues/74 for more info."
        )
        warnings.warn(message, DeprecationWarning)

        return val
