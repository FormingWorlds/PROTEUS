from __future__ import annotations

from attrs import define, field, validators


@define
class Orbit:
    semimajoraxis: float
    eccentricity: float
    zenith_angle: float
    s0_factor: float

    module: str  = field(validator=validators.in_(("none",)))
