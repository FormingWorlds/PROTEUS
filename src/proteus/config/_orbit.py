from __future__ import annotations

from typing import Literal, Union

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
class Dummy:
    """Dummy tidal heating module.

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
    dr_min: int             = field(default=300, validator=gt(0))
    dr_max: int             = field(default=3000, validator=gt(0))
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
    sigma_R_inf: float  = field(default=1e-6, validator=gt(0))
    sigma_R_prf: str    = field(default="exp", validator=in_(("uniform", "exp", "linear", "quadratic", "dynamic", "dynamic_interp")))
    H_R: float          = field(default=1e4, validator=gt(0))
    efficiency: float   = field(default=0.3, validator=gt(0))


@define
class Obliqua:
    """Obliqua tides module.

    Attributes
    ----------
    store_3D : bool
        Whether to store 3D information for solid tides.
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
    k_min, k_max : int
        Fourier index range in mean anomaly (adaptive spectrum).
    material_mu : str
        Rheology model for complex shear modulus ("andrade" or "maxwell").
    material_k : str
        Rheology model for complex bulk modulus ("andrade" or "maxwell").
    alpha : float
        Andrade power-law exponent.
    module_solid : str
        Solid-tide module to use ("none", "solid0d", "solid1d", "solid1d-relax", "solid1d-mush", "solid1d-mush-relax").
    module_mushy : str
        Mushy-tide module to use ("none", "interp").
    module_fluid : str
        Fluid-tide module to use ("none", "fluid0d", "fluid1d").
    solid : ObliquaSolid
        Solid-tide configuration.
    mushy : ObliquaMushy
        Mushy-tide configuration.
    fluid : ObliquaFluid
        Fluid-tide configuration.
    """

    # global configuration
    store_3D: bool = field(default=False)

    min_frac: float = field(default=0.02, validator=gt(0))

    visc_l: float   = field(default=1e2, validator=gt(0))
    visc_lus: float = field(default=5e5, validator=gt(0))
    visc_s: float   = field(default=1e22, validator=gt(0))
    visc_sus: float = field(default=5e5, validator=gt(0))

    n: list = field(default=[2])
    m: list = field(default=[0, 2])

    k_min: Union[int, Literal["none"]] = field(default="none")
    k_max: Union[int, Literal["none"]] = field(default="none")

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
class Satellite:
    """Satellite orbit configuration for planet-satellite systems.

    Attributes
    ----------
    include_satellite: bool
        Whether to model a satellite orbiting the planet.
    mass_sat: float
        Satellite mass [M_earth].
    radius_sat: float
        Satellite radius [R_earth].
    axial_period_sat: float | None
        Satellite initial day length [hours], will use orbital period if value is None.
    semimajoraxis_sat: float
        Satellite initial semi-major axis [AU].
    eccentricity_sat: float
        Satellite initial orbital eccentricity [dimensionless].
    aps_prec_angle: float
        Satellite apsidal precession angle [deg].
    c_factor_sat: float
        Satellite tidal dissipation factor (<= 0.4) [dimensionless].
    love_number_sat: str | None
        Satellite love number spectrum, provide absolute path to netCDF file containing forcing
        frequencies and complex Lovenumbers, and corresponding tidal degree in nmk format.
    """
    # Satellite orbit
    include_satellite: bool     = field(default=False)
    mass_sat: float             = field(default=0.012, validator=gt(0))
    radius_sat: float           = field(default=0.273, validator=gt(0))
    axial_period_sat            = field(default=None, validator=ax_valid, converter=none_if_none)
    semimajoraxis_sat: float    = field(default=0.133, validator=gt(0))
    eccentricity_sat: float     = field(default=0.0, validator=ge(0))
    aps_prec_angle: float       = field(default=0.0, validator=ge(0))
    c_factor_sat: float         = field(default=0.4, validator=(gt(0), le(0.4),))
    love_number_sat: str | None = field(default=None, converter=none_if_none)


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

    star_planet_model: str | None
        Select star-planet orbit module to use. Choices: 'none', 'sp0dae'.
    axial_period: float | None
        Planet initial day length [hours], will use orbital period if value is None.

    satellite: Satellite
        Satellite and orbit configuration for planet-satellite systems.

    planet_satellite_model: str | None
        Select planet-satellite orbit module to use. Choices: 'none', 'ps0d', 'ps1d_evec'.

    perturber: str | None
        Select perturber to induce tides on the planet. Options: 'none', 'star', 'satellite'.

    module: str | None
        Select tides module to use. Choices: 'none', 'dummy', 'lovepy', 'obliqua'.

    dummy: Dummy
        Dummy tidal heating module configuration.
    lovepy: Lovepy
        Lovepy tidal heating module configuration.
    obliqua: Obliqua
        Obliqua tidal heating module configuration.
    """

    # Planet initial orbital parameter
    semimajoraxis: float    = field(default=1.0, validator=gt(0))
    eccentricity: float     = field(
        default=0.0,
        validator=(
            ge(0),
            lt(1),
        ),
    )
    instellation_method: str = field(default='distance', validator=in_(('distance', 'inst')))
    instellationflux: float  = field(default=1.0, validator=gt(0))

    # Climate parameters set by rotation of planet
    zenith_angle: float = field(
        default=48.19,
        validator=(
            ge(0),
            lt(90),
        ),
    )
    s0_factor: float = field(default=0.375, validator=gt(0))

    # Orbital model to use for star-planet orbit evolution based on tides
    star_planet_model: str | None = field(default='none', validator=in_((None, 'none', 'sp0dae')), converter=none_if_none)
    # Initial day length for planet [hours]
    # If none, assume 1:1 spin orbit synchronization and use orbital period as day length
    axial_period                  = field(default=None, validator=ax_valid, converter=none_if_none)

    # Satellite orbit configuration
    satellite: Satellite = field(factory=Satellite)

    # Orbital model to use for planet-satellite orbit evolution based on tides
    planet_satellite_model: str | None = field(default='none', validator=in_((None, 'none', 'ps0d', 'ps1d_evec')), converter=none_if_none)

    # Perturber to induce tides on the planet. Options: 'none', 'star', 'satellite'.
    perturber: str | None = field(default=None, validator=in_((None, 'none', 'star', 'satellite')), converter=none_if_none)

    # Tidal heating modules
    module: str | None = field(
        default='none', validator=in_((None, 'dummy', 'lovepy', 'obliqua')), converter=none_if_none
    )

    dummy:  Dummy    = field(factory=Dummy)
    lovepy: Lovepy   = field(factory=Lovepy)
    obliqua: Obliqua = field(factory=Obliqua)
