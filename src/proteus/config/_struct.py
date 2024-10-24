from __future__ import annotations

from attrs import define, field
from attrs.validators import gt, in_, lt

from ._converters import none_if_none


@define
class Struct:
    """Planetary structure (mass, radius).

    mass: float
        M_earth
    radius: float
        R_earth
    corefrac: float
        Non-dimensional, radius fraction
    module: str | None
        Select internal structure module to use. Not used currently.
    """
    mass: float = field(validator=gt(0))
    radius: float  = field(validator=gt(0))
    corefrac: float = field(validator=(gt(0), lt(1)))

    module: str | None = field(validator=in_((None,)), converter=none_if_none)
