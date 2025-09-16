from __future__ import annotations

from attrs import define, field
from attrs.validators import in_

from ._converters import none_if_none


def valid_zephyrus(instance, attribute, value):
    if instance.module != "zephyrus":
        return

    Pxuv = instance.zephyrus.Pxuv
    if (not Pxuv) or (Pxuv < 0) or (Pxuv > 10):
        raise ValueError("`zephyrus.Pxuv` must be >0 and < 10 bar")

    efficiency = instance.zephyrus.efficiency
    if (not efficiency) or (efficiency < 0) or (efficiency > 1):
        raise ValueError("`zephyrus.efficiency` must be >=0 and <=1")

@define
class Zephyrus:
    """Parameters for Zephyrus module.

    Attributes
    ----------
    Pxuv: float
        Pressure at which XUV radiation become opaque in the planetary atmosphere [bar]
    efficiency: float
        Escape efficiency factor
    tidal: bool
        Tidal contribution enabled
    """
    Pxuv: float       = field(default=5e-5)
    efficiency: float = field(default=0.1)
    tidal: bool       = field(default=False)

def valid_escapedummy(instance, attribute, value):
    if instance.module != "dummy":
        return

    rate = instance.dummy.rate
    if (not rate) or (rate < 0) :
        raise ValueError("`escape.dummy.rate` must be >0")

@define
class EscapeDummy:
    """Dummy module.

    Attributes
    ----------
    rate: float
        Bulk unfractionated escape rate [kg s-1]
    """
    rate = field(default=None)

@define
class Escape:
    """Escape parameters, model selection.

    Attributes
    ----------
    reservoir: str
        Element reservoir representing the escaping composition. Choices: bulk, outgas, pxuv
    module: str | None
        Escape module to use. Choices: "none", "dummy", "zephyrus".
    zephyrus: Zephyrus
        Parameters for zephyrus module.
    dummy: EscapeDummy
        Parameters for dummy escape module.
    """

    module: str | None = field(
        validator=in_((None, 'dummy', 'zephyrus')), converter=none_if_none
        )

    zephyrus: Zephyrus = field(factory=Zephyrus,    validator=valid_zephyrus)
    dummy: EscapeDummy = field(factory=EscapeDummy, validator=valid_escapedummy)

    reservoir: str = field(default='outgas', validator=in_(('bulk','outgas','pxuv')))
