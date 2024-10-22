from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Orbit:
    """Planetary orbital parameters.

    Attributes
    ----------
    semimajoraxis: float
        Semi-major axis of the planet's orbit [AU].
    eccentricity: float
        Eccentricity of the planet's orbit.
    zenith_angle: float
        Characteristic angle of incoming stellar radiation, relative to the zenith [deg].
    s0_factor: float
        Scale factor applies to incoming stellar radiation to represent planetary rotation and heat redistribution.
    module: str | None
        Select orbit module to use. Not used currently.
    """
    semimajoraxis: float
    eccentricity: float = field(validator=[
        validators.ge(0),
        validators.lt(1),
    ])
    zenith_angle: float
    s0_factor: float

    module: str | None = field(validator=validators.in_((None,)), converter=none_if_none)
