from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, lt

from ._converters import none_if_none


def phi_tide_validator(instance, attribute, value):

    # direction of inequality
    if (value[0] not in ("<",">")) or (len(value) < 2):
        raise ValueError(f"Phi_tide must be an inequality (e.g. '<0.3') got {value}")

    # value of inequality
    try:
        number = float(value[1:])
    finally:
        if (number < 0.0) or (number > 1.0):
            raise ValueError(f"Phi_tide value must be between 0 and 1, got {number}")

@define
class OrbitDummy:
    """Dummy orbit module.

    Attributes
    ----------
    H_tide: float
        Fixed global heating rate from tides [W kg-1].
    Phi_tide: str
        Inequality which, if locally true, determines in which regions tides are applied.
    """
    H_tide: float   = field(default=0.0, validator=ge(0.0))
    Phi_tide: str   = field(default="<0.3", validator=phi_tide_validator)

@define
class Lovepy:
    """Lovepy tides module.

    Attributes
    ----------
    visc_thresh: float
        Minimum viscosity required for heating [Pa s].
    """
    visc_thresh: float = field(default=1e9, validator=gt(0))


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
        Select orbit module to use. Choices: 'none', 'dummy', 'lovepy'.
    """

    module: str | None = field(
        validator=in_((None, 'dummy', 'lovepy')),
        converter=none_if_none,
    )

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

    dummy:  OrbitDummy  = field(factory=OrbitDummy)
    lovepy: Lovepy      = field(factory=Lovepy)
