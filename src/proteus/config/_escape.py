from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Zephyrus:
    some_parameter: str


@define
class EscapeDummy:
    rate: float


@define
class Escape:
    """Escape parameters, model selection"""
    module: str | None = field(
        validator=validators.in_((None, 'dummy', 'zephyrus')), converter=none_if_none
        )

    zephyrus: Zephyrus
    dummy: EscapeDummy
