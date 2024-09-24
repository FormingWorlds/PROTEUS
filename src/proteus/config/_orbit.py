from __future__ import annotations

from typing import Literal

from attrs import define


@define
class Orbit:
    semimajoraxis: float
    eccentricity: float
    zenith_angle: float
    s0_factor: float

    module: Literal["none"]
