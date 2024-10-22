from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Escape:
    """Escape parameters, model selection.

    Attributes
    ----------
    module: str | None
        Escape module to use. Choices: "none", "dummy", "zephyrus".
    zephyrus: Zephyrus
        Parameters for zephyrus module.
    dummy: EscapeDummy
        Parameters for dummy escape module.
    """
    module: str | None = field(
        validator=validators.in_((None, 'dummy', 'zephyrus')), converter=none_if_none
        )

    zephyrus: Zephyrus
    dummy: EscapeDummy


@define
class Zephyrus:
    """Parameters for Zephyrus module.

    Attributes
    ----------
    Pxuv: float
        Pressure related to the radius (Rxuv) at which the XUV radiation become opaque in the planetary atmosphere.
    """
    Pxuv: float


@define
class EscapeDummy:
    """Dummy module.

    Attributes
    ----------
    rate: float
        Bulk unfractionated escape rate [kg s-1]
    """
    rate: float
