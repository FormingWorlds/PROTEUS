from __future__ import annotations

from attrs import define, field
from attrs.validators import gt, in_

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
        validator=in_((None, 'dummy', 'zephyrus')), converter=none_if_none
        )

    zephyrus: Zephyrus
    dummy: EscapeDummy


@define
class Zephyrus:
    """Parameters for Zephyrus module.

    Attributes
    ----------
    some_parameter: str
        Not used.
    """
    some_parameter: str


@define
class EscapeDummy:
    """Dummy module.

    Attributes
    ----------
    rate: float
        Bulk unfractionated escape rate [kg s-1]
    """
    rate: float = field(validator=gt(0))
