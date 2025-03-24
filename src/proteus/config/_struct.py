from __future__ import annotations

from attrs import define, field
from attrs.validators import gt, in_, lt
from ._converters import none_if_none

def mass_radius_valid(instance, attribute, value):

    # must set either mass_tot or radius_int
    if (instance.radius_int is None) and (instance.mass_tot is None):
        raise ValueError("Must set `radius_int` or `mass_tot` to define structure")
    if (instance.radius_int is not None) and (instance.mass_tot is not None):
        raise ValueError("Must set either `radius_int` OR `mass_tot` to define structure")

    if instance.mass_tot is not None:
        if instance.mass_tot < 0:
            raise ValueError("The total planet mass must be > 0")
        if instance.mass_tot > 20:
            raise ValueError("The total planet mass must be < 20 M_earth")

    if instance.radius_int is not None:
        if instance.radius_int < 0:
            raise ValueError("The interior radius must be > 0")
        if instance.radius_int > 10:
            raise ValueError("The interior radius must be < 10 R_earth")

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
    """

    corefrac: float     = field(validator=(gt(0), lt(1)))

    mass_tot: float | None    = field(default='none',
                                      validator=mass_radius_valid,
                                      converter=none_if_none)
    radius_int: float | None  = field(default='none',
                                      validator=mass_radius_valid,
                                      converter=none_if_none)

    core_density: float = field(default=10738.33, validator=gt(0))
    core_heatcap: float = field(default=880.0,    validator=gt(0))

    @property
    def set_by(self) -> str:
        """How is the structure set?"""
        if self.mass_tot is not None:
            return 'mass_tot'
        if self.radius_int is not None:
            return 'radius_int'
        return None
