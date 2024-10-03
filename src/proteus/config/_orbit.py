from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Orbit:
    """Planetary orbital parameters"""
    semimajoraxis: float
    eccentricity: float = field(validator=[
        validators.ge(0),
        validators.le(1),
    ])
    zenith_angle: float
    s0_factor: float

    module: str | None = field(validator=validators.in_((None,)), converter=none_if_none)
