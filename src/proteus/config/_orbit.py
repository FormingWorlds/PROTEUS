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
    ncalc: int
        Number of interpoltaed interior levels to use for solving tidal heating rates.
    """
    visc_thresh: float = field(default=1e9, validator=gt(0))
    ncalc: int = field(default=1000, validator=gt(100))

def ax_valid(instance, attribute, value):
    if value is None:
        return

    if float(value) <= 0:
        raise ValueError(f"Initial axial period must be >0 hours, got {value}")

@define
class Orbit:
    """Planetary and satellite orbital parameters.

    Includes initial conditions, and options for enabling dynamical evolution.

    Attributes
    ----------
    semimajoraxis: float
        Initial semi-major axis of the planet's orbit [AU].
    eccentricity: float
        Initial Eccentricity of the planet's orbit.
    instellation_method: str
        Whether to use the semi-major axis ('sma') or instellation flux ('inst') to define the planet's initial orbit
    instellationflux: float
        Instellation flux initially received by the planet in Earth units.

    zenith_angle: float
        Characteristic angle of incoming stellar radiation, relative to the zenith [deg].
    s0_factor: float
        Scale factor applies to incoming stellar radiation to represent planetary rotation.

    evolve: bool
        Allow the planet's orbit to evolve based on eccentricity tides?
    axial_period: float | None
        Planet initial day length [hours], will use orbital period if value is None.

    satellite: bool
        Model a satellite (moon) orbiting the planet and solve for its orbit?
    semimajoraxis_sat: float
        Satellit initial semi-major axis  [m]

    module: str | None
        Select orbit module to use. Choices: 'none', 'dummy', 'lovepy'.
    """

    # Tidal heating modules
    module: str | None   = field(validator=in_((None, 'dummy', 'lovepy')),
                                converter=none_if_none)

    # Planet initial orbital parameter
    semimajoraxis: float = field(validator=gt(0))
    eccentricity: float  = field(validator=(ge(0), lt(1),))

    # Climate parameters set by rotation of planet
    zenith_angle: float  = field(validator=(ge(0),lt(90),))
    s0_factor: float     = field(validator=gt(0))

    # Allow the planet's orbit to evolve based on eccentricity tides?
    evolve: bool = field(default=False)

    # Initial day length for planet [hours]
    # If none, assume 1:1 spin orbit resonance
    axial_period = field(default=None,
                            validator=ax_valid, converter=none_if_none)

    # Satellite orbit
    satellite: bool          = field(default=False)
    mass_sat: float          = field(default=7.347e22, validator=gt(0))
    semimajoraxis_sat: float = field(default=3e8, validator=gt(0))

    dummy:  OrbitDummy  = field(factory=OrbitDummy)
    lovepy: Lovepy      = field(factory=Lovepy)

    instellation_method: str = field(default='sma',validator=in_(('sma','inst')))
    instellationflux: float = field(default=1.0,validator=gt(0))
