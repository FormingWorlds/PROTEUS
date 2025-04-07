from __future__ import annotations

from attrs import define, field
from attrs.validators import gt, lt

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

    mass_tot            = field(default='none',
                                validator=mass_radius_valid,
                                converter=none_if_none)
    radius_int          = field(default='none',
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
