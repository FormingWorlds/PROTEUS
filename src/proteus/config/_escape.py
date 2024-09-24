from __future__ import annotations

from attrs import define, field, validators


@define
class Zephyrus:
    some_parameter: str


@define
class EscapeDummy:
    rate: float


@define
class Escape:
    module: str = field(validator=validators.in_(('dummy', 'zephyrus')))

    zephyrus: Zephyrus
    dummy: EscapeDummy
