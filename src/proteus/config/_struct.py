from __future__ import annotations

from attrs import define, field
from attrs.validators import gt, in_, lt


@define
class Struct:
    """Planetary structure (mass, radius).

    set_by: str
        What defines the interior structure? Choices: "mass_tot", "radius_int".
    mass_tot: float
        Total mass of the planet [M_earth]
    radius_int: float
        Radius of the atmosphere-mantle boundary [R_earth]
    corefrac: float
        Fraction of the planet's interior radius corresponding to the core.
    core_density: float
        Density of the planet's core, optional [kg/m3]
    """

    set_by: str = field(validator=in_(('mass_tot','radius_int')))

    mass_tot: float = field(validator=(gt(0),lt(20)))
    radius_int: float = field(validator=(gt(0),lt(10)))

    corefrac: float = field(validator=(gt(0), lt(1)))
    core_density: float = field(default = 10738.33)
