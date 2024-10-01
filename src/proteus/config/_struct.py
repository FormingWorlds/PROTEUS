from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Struct:
    """Planetary structure (mass, radius)"""
    mass: float
    radius: float
    corefrac: float

    module: str | None = field(validator=validators.in_((None,)), converter=none_if_none)
