from __future__ import annotations

from attrs import define, field, validators


@define
class Struct:
    mass: float
    radius: float
    corefrac: float

    module: str = field(validator=validators.in_(("none", )))
