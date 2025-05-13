from __future__ import annotations

from attrs import define, field
from attrs.validators import gt, lt, ge, in_

from ._converters import none_if_none


def mass_radius_valid(instance, attribute, value):

    radius_int = none_if_none(instance.radius_int)
    mass_tot = none_if_none(instance.mass_tot)

    if (radius_int is None) and (mass_tot is None):
        raise ValueError("Must set one of `radius_int` or `mass_tot`")
    if (radius_int is not None) and (mass_tot is not None):
        raise ValueError("Must set either `radius_int` or `mass_tot`, not both")

    if mass_tot is not None:
        if mass_tot < 0:
            raise ValueError("The total planet mass must be > 0")
        if mass_tot > 20:
            raise ValueError("The total planet mass must be < 20 M_earth")

    if radius_int is not None:
        if radius_int < 0:
            raise ValueError("The interior radius must be > 0")
        if radius_int > 10:
            raise ValueError("The interior radius must be < 10 R_earth")

def valid_zalmoxis(instance, attribute, value):
    if instance.module != "zalmoxis":
        return

    if attribute.name == "max_iterations_outer" and value < 3:
        raise ValueError(f"`interior.zalmoxis.{attribute.name}` must be > 2")
    if attribute.name == "max_iterations_inner" and value < 13:
        raise ValueError(f"`interior.zalmoxis.{attribute.name}` must be > 12")
    if attribute.name == "max_iterations_pressure" and value < 13:
        raise ValueError(f"`interior.zalmoxis.{attribute.name}` must be > 12")

@define
class Zalmoxis:
    """Parameters for Zalmoxis module.

    Attributes
    ----------
    EOSchoice: str
        EOS choice of Zalmoxis. Choices: "Tabulated".
    coremassfrac: float
        Fraction of the planet's interior mass corresponding to the core.
    weight_iron_frac: float
        Fraction of the planet's core mass that is iron.
    num_levels: int
        Number of Zalmoxis radius layers.
    max_iterations_outer: int
        Maximum number of iterations for the outer loop.
    tolerance_outer: float
        Convergence tolerance for the outer loop [kg].
    tolerance_radius: float
        Convergence tolerance for the cmb radius calculation in the outer loop [m].
    max_iterations_inner: int
        Maximum number of iterations for the inner loop.
    tolerance_inner: float
        Convergence tolerance for the inner loop [kg/m^3].
    relative_tolerance: float
        Relative tolerance for solve_ivp.
    absolute_tolerance: float
        Absolute tolerance for solve_ivp.
    target_surface_pressure: float
        Target surface pressure for the pressure adjustment [Pa].
    pressure_tolerance: float
        Convergence tolerance for the pressure adjustment [Pa].
    max_iterations_pressure: int
        Maximum number of iterations for the pressure adjustment.
    pressure_adjustment_factor: float
        Reduction factor for adjusting the pressure in the pressure adjustment.
    """

    EOSchoice: str                    = field(default="Tabulated", validator=in_(("Tabulated")))

    coremassfrac: float               = field(default=0.32, validator=(gt(0), lt(1)))
    weight_iron_frac: float           = field(default=0.35, validator=(gt(0), lt(1)))

    num_levels: int                   = field(default=100)

    max_iterations_outer: int         = field(default=20, validator=ge(1))
    tolerance_outer: float            = field(default=1e-3, validator=ge(0))
    max_iterations_inner: int         = field(default=100, validator=ge(1))
    tolerance_inner: float            = field(default=1e-4, validator=ge(0))
    relative_tolerance: float         = field(default=1e-5, validator=ge(0))
    absolute_tolerance: float         = field(default=1e-6, validator=ge(0))

    target_surface_pressure: float    = field(default=101325, validator=ge(0))
    pressure_tolerance: float         = field(default=1e11, validator=ge(0))
    max_iterations_pressure: int      = field(default=200, validator=ge(1))
    pressure_adjustment_factor: float = field(default=1.1, validator=ge(0))

@define
class Struct:
    """Planetary structure (mass, radius).

    mass_tot: float
        Total mass of the planet [M_earth]
    radius_int: float
        Radius of the atmosphere-mantle boundary [R_earth]
    corefrac: float
        Fraction of the planet's interior radius corresponding to the core.
    core_density: float
        Density of the planet's core [kg m-3]
    core_heatcap: float
        Specific heat capacity of the planet's core [J kg-1 K-1]
    module: str
        Module for solving the planet's interior structure. Choices: 'self', 'zalmoxis'.
    """

    corefrac: float         = field(validator=(gt(0), lt(1)))

    module: str             = field(validator=in_(('self', 'zalmoxis')))
    zalmoxis: Zalmoxis      = field(factory=Zalmoxis,      validator=valid_zalmoxis)

    mass_tot: float         = field(default='none', validator=mass_radius_valid, converter=none_if_none)
    radius_int: float       = field(default='none', validator=mass_radius_valid, converter=none_if_none)

    core_density: float     = field(default=10738.33, validator=gt(0))
    core_heatcap: float     = field(default=880.0,    validator=gt(0))

    @property
    def set_by(self) -> str:
        """How is the structure set?"""
        if self.mass_tot is not None:
            return 'mass_tot'
        if self.radius_int is not None:
            return 'radius_int'
        return None

