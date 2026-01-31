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

@define
class ObliquaSolid:
    """Solid-tide configuration for Obliqua tides."""
    ncalc: int = field(default=1000, validator=gt(100))
    bulk_l: float = field(default=1e9, validator=gt(0))            # [Pa]
    permea: float = field(default=1e-7, validator=gt(0))           # [m^2]
    porosity_thresh: float = field(default=1e-5, validator=gt(0))  # [-]
    mush: bool = field(default=False)

@define
class ObliquaFluid:
    """Fluid-tide configuration for Obliqua tides."""
    sigma_R: float = field(default=1e-3, validator=gt(0))

@define
class Obliqua:
    """Obliqua tides module.

    Attributes
    ----------
    dim : int
        Dimensions used to evaluate solid tides (0D or 1D).
    min_frac : float
        Minimal segment radius fraction before smoothing.
    visc_l : float
        Pure liquid viscosity [Pa s].
    visc_l_lus : float
        Liquidus viscosity [Pa s].
    visc_s : float
        Pure solid viscosity [Pa s].
    visc_s_sus : float
        Solidus viscosity [Pa s].
    n : int
        Power of the radial factor (r/a)^n.
    m : int
        Tidal harmonic (m=2 semidiurnal, m=1 diurnal).
    N_sigma : int
        Number of probe frequencies for k2 evaluation.
    k_min, k_max : int
        Fourier index range in mean anomaly.
    p_min, p_max : int
        Log10 period range [kyr].
    material : str
        Rheology model ("andrade" or "maxwell").
    alpha : float
        Andrade power-law exponent.
    strain : bool
        Strategy for obtaining heating profile.
    """

    # global configuration
    dim: int = field(default=1, validator=in_((0, 1)))
    min_frac: float = field(default=0.02, validator=gt(0))

    visc_l: float = field(default=1e2, validator=gt(0))
    visc_l_lus: float = field(default=5e2, validator=gt(0))
    visc_s: float = field(default=1e21, validator=gt(0))
    visc_s_sus: float = field(default=5e13, validator=gt(0))

    n: int = field(default=2, validator=gt(0))
    m: int = field(default=2, validator=gt(0))

    N_sigma: int = field(default=10, validator=gt(0))
    k_min: int = field(default=-30)
    k_max: int = field(default=40)

    p_min: float = field(default=-20)
    p_max: float = field(default=6)

    material: str = field(
        default="andrade",
        validator=in_(("andrade", "maxwell"))
    )
    alpha: float = field(default=0.3, validator=gt(0))
    strain: bool = field(default=True)

    # module selection
    module_solid: str = field(
        default="solid0d",
        validator=in_(("solid0d", "solid1d", "solid1d-mush"))
    )
    module_fluid: str = field(
        default="fluid0d",
        validator=in_(("fluid0d"))
    )

    # submodules
    solid: ObliquaSolid = field(factory=ObliquaSolid)
    fluid: ObliquaFluid = field(factory=ObliquaFluid)

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
        Select orbit module to use. Choices: 'none', 'dummy', 'lovepy', 'obliqua'.
    """

    # Tidal heating modules
    module: str | None   = field(validator=in_((None, 'dummy', 'lovepy', 'obliqua')),
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
    obliqua: Obliqua    = field(factory=Obliqua)

    instellation_method: str = field(default='sma',validator=in_(('sma','inst')))
    instellationflux: float = field(default=1.0,validator=gt(0))
