from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, le, lt

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
    """Dummy orbit/tidal heating module.

    Uses a fixed tidal heating power density and love number.

    Attributes
    ----------
    H_tide: float
        Fixed global heating rate from tides [W kg-1].
    Phi_tide: str
        Inequality which, if locally true, determines in which regions tides are applied.
    Imk2: float
        Imaginary part of k2 Love number, which is usually negative.
    """
    H_tide: float   = field(default=0.0, validator=ge(0.0))
    Phi_tide: str   = field(default="<0.3", validator=phi_tide_validator)
    Imk2: float     = field(default=0.0, validator=le(0.0))

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
    instellation_method: str
        Whether to use the semi-major axis ('sma') or instellation flux ('inst') to define the planet's orbit
    instellationflux: float
        Instellation flux received from the planet in Earth units.
    semimajoraxis: float
        Initial semi-major axis of the planet's orbit [AU].
    eccentricity: float
        Initial Eccentricity of the planet's orbit.
    zenith_angle: float
        Characteristic angle of incoming stellar radiation, relative to the zenith [deg].
    s0_factor: float
        Scale factor applies to incoming stellar radiation to represent planetary rotation and heat redistribution.
    module: str | None
        Select orbit module to use. Choices: 'none', 'dummy', 'lovepy'.
    """
    module: str | None = field(validator=in_((None, 'dummy', 'lovepy')),converter=none_if_none)
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
    instellation_method: str = field(default='sma',validator=in_(('sma','inst')))
    instellationflux: float = field(default=1.0,validator=gt(0))
    evolve: bool = field(default=False)

    dummy:  OrbitDummy  = field(factory=OrbitDummy)
    lovepy: Lovepy      = field(factory=Lovepy)

    satellite: bool     = field(default=False)
    semimajoraxis_sat: float = field(validator=gt(0))
    lod: float          = field(validator=gt(0))
    # system_am: float = field(validator=gt(0))
