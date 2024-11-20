from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, lt

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
        Select orbit module to use. Choices: 'none', 'dummy'.
    """
    semimajoraxis: float = field(validator=gt(0))
    eccentricity: float = field(validator=(
        ge(0),
        lt(1),
    ))
    zenith_angle: float  = field(validator=(
        ge(0),
        lt(90),
    ))
    s0_factor: float = field(validator=gt(0))

    module: str | None = field(
        validator=in_((None, 'dummy')),
        converter=none_if_none,
    )

    dummy: OrbitDummy


@define
class OrbitDummy:
    """Dummy orbit module.

    Attributes
    ----------
    H_tide: float
        Fixed global heating rate from tides [W kg-1].
    """
    H_tide: float = field(validator=ge(0))
