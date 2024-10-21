from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


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
    rate: float


@define
class Escape:
    """Escape parameters, model selection.

    Attributes
    ----------
    module: str | None
        Select module, choice None, 'dummy', 'zephyrus'.
    zephyrus: Zephyrus
        Parameters for zephyrus module.
    dummy: EscapeDummy
        Parameters for dummy module.
    """
    module: str | None = field(
        validator=validators.in_((None, 'dummy', 'zephyrus')), converter=none_if_none
        )

    zephyrus: Zephyrus
    dummy: EscapeDummy
