from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, le, lt

from ._converters import none_if_none


def phi_tide_validator(instance, attribute, value):
    # direction of inequality
    if (value[0] not in ('<', '>')) or (len(value) < 2):
        raise ValueError(f"Phi_tide must be an inequality (e.g. '<0.3') got {value}")

    # value of inequality
    try:
        number = float(value[1:])
    except ValueError:
        raise ValueError(f'Phi_tide must contain a number (e.g. "<0.3"), got {value}')
    if (number < 0.0) or (number > 1.0):
        raise ValueError(f'Phi_tide value must be between 0 and 1, got {number}')


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

    H_tide: float = field(default=0.0, validator=ge(0.0))
    Phi_tide: str = field(default='<0.3', validator=phi_tide_validator)
    Imk2: float = field(default=0.0, validator=le(0.0))


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
    """Solid-tide configuration for Obliqua tides.

    Attributes
    ----------
    ncalc: int
        Number of interpolated interior levels to use for solving tidal heating rates (shooting method).
    dr_min: float
        Minimum radial grid spacing [m] (Henyey/relaxation method).
    dr_max: float
        Maximum radial grid spacing [m] (Henyey/relaxation method).
    core: str
        Core solution vector ("liquid", "solid", or "inertial").
    bulk_l: float
        Bulk modulus of the liquid phase [Pa].
    porosity_thresh: float
        Porosity threshold, hard cutoff below which melt fraction is set to zero [dimensionless].
    dbulk_power: int
        Drained bulk modulus powerlaw scaling exponent [dimensionless].
    """
    ncalc: int              = field(default=1000, validator=gt(100))
    dr_min: float           = field(default=300, validator=gt(0))
    dr_max: float           = field(default=3000, validator=gt(dr_min))
    core: str               = field(default="liquid", validator=in_(("liquid", "solid", "inertial")))
    bulk_l: float           = field(default=1e9, validator=gt(0))
    porosity_thresh: float  = field(default=3e-2, validator=gt(0))
    dbulk_power: int        = field(default=1, validator=gt(0))

@define
class ObliquaMushy:
    """Mushy-tide configuration for Obliqua tides.

    Attributes
    ----------
    b_width: float
        Scale width of the bottom heating decay profile [dimensionless].
    t_width: float
        Scale width of the top heating decay profile [dimensionless].
    """
    b_width: float      = field(default=5e-1)
    t_width: float      = field(default=3e-2)

@define
class ObliquaFluid:
    """Fluid-tide configuration for Obliqua tides.

    Attributes
    ----------
    sigma_R: float
        Rayleigh drag in the fluid-mush/solid boundary layers [1/s].
    sigma_R_inf: float
        Rayleigh drag in the pure fluid [1/s].
    sigma_R_prf: str
        Radial heating distribution profile [dimensionless].
    H_R: float
        Scale height to be used by heating profile [m].
    efficiency: float
        Rayleigh drag efficiency at core interface [dimensionless].
    """
    sigma_R: float      = field(default=1e-3, validator=gt(0))
    sigma_R_inf: float  = field(default=1e-7, validator=gt(0))
    sigma_R_prf: str    = field(default="exp", validator=in_(("uniform", "exp", "linear", "quadratic", "dynamic")))
    H_R: float          = field(default=1e4, validator=gt(0))
    efficiency: float   = field(default=0.3, validator=gt(0))

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
    visc_lus : float
        Liquidus viscosity [Pa s].
    visc_s : float
        Pure solid viscosity [Pa s].
    visc_sus : float
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
    min_frac: float = field(default=0.02, validator=gt(0))

    visc_l: float   = field(default=1e2, validator=gt(0))
    visc_lus: float = field(default=5e5, validator=gt(0))
    visc_s: float   = field(default=1e22, validator=gt(0))
    visc_sus: float = field(default=5e5, validator=gt(0))

    n: list = field(default=[2])
    m: list = field(default=[0, 2])

    spectrum: str = field(
        default="adaptive",
        validator=in_(("full", "adaptive"))
    )

    N_sigma: int = field(default=10, validator=gt(0))
    p_min: float = field(default=-20)
    p_max: float = field(default=6)

    k_min: int | str = field(default="none")
    k_max: int | str = field(default="none")

    material_mu: str = field(
        default="andrade",
        validator=in_(("andrade", "maxwell", "elastic"))
    )
    material_k: str = field(
        default="andrade",
        validator=in_(("andrade", "maxwell", "elastic"))
    )
    alpha: float = field(default=0.3, validator=gt(0))

    # module selection
    module_solid: str = field(
        default="solid0d",
        validator=in_(("none", "solid0d", "solid1d", "solid1d-relax", "solid1d-mush", "solid1d-mush-relax"))
    )
    module_mushy: str = field(
        default="none",
        validator=in_(("none", "interp"))
    )
    module_fluid: str = field(
        default="fluid0d",
        validator=in_(("none", "fluid0d", "fluid1d"))
    )

    # submodules
    solid: ObliquaSolid = field(factory=ObliquaSolid)
    mushy: ObliquaMushy = field(factory=ObliquaMushy)
    fluid: ObliquaFluid = field(factory=ObliquaFluid)

def ax_valid(instance, attribute, value):
    if value is None:
        return

    if float(value) <= 0:
        raise ValueError(f'Initial axial period must be >0 hours, got {value}')


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
        Whether to use the semi-major axis ('distance') or instellation flux ('inst') to define the planet's initial orbit
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
    mass_sat: float
        Satellite mass [kg]
    orbital_period_sat: float
        Satellite orbital period [s]
    semimajoraxis_sat: float
        Satellite initial semi-major axis  [m]
    eccentricity_sat: float
        Satellite initial orbital eccentricity [dimensionless]
    love_number_sat: str | None
        Satellite love number spectrum, provide absolute path to file.

    orbit_model: str | None
        Select orbit module to use. Choices: 'none', 'dummy', 'pla_sat_evec',

    module: str | None
        Select tides module to use. Choices: 'none', 'dummy', 'lovepy', 'obliqua'.
    """

    # Planet initial orbital parameter
    semimajoraxis: float = field(default=1.0, validator=gt(0))
    eccentricity: float = field(
        default=0.0,
        validator=(
            ge(0),
            lt(1),
        ),
    )
    # Climate parameters set by rotation of planet
    zenith_angle: float = field(
        default=48.19,
        validator=(
            ge(0),
            lt(90),
        ),
    )
    s0_factor: float = field(default=0.375, validator=gt(0))

    # Allow the planet's orbit to evolve based on tides?
    evolve: bool          = field(default=False)
    # Initial day length for planet [hours]
    # If none, assume 1:1 spin orbit synchronization and use orbital period as day length
    axial_period = field(default=None, validator=ax_valid, converter=none_if_none)

    # Perturber to induce tides on the planet. Options: 'star', 'satellite'.
    perturber: str | None = field(default=None, validator=in_((None, 'star', 'satellite')), converter=none_if_none)

    # Satellite orbit
    satellite: bool             = field(default=False)
    mass_sat: float             = field(default=7.347e22, validator=gt(0))
    orbital_period_sat: float   = field(default=2.36e6, validator=gt(0))
    semimajoraxis_sat: float    = field(default=3e8, validator=gt(0))
    eccentricity_sat: float     = field(default=0.0, validator=ge(0))
    love_number_sat: str | None = field(default=None, converter=none_if_none)

    # Tidal heating modules
    module: str | None = field(
        default='none', validator=in_((None, 'dummy', 'lovepy', 'obliqua')), converter=none_if_none
    )

    dummy:  OrbitDummy  = field(factory=OrbitDummy)
    lovepy: Lovepy      = field(factory=Lovepy)
    obliqua: Obliqua    = field(factory=Obliqua)

    instellation_method: str = field(default='distance', validator=in_(('distance', 'inst')))
    instellationflux: float  = field(default=1.0, validator=gt(0))
