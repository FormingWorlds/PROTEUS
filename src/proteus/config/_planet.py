from __future__ import annotations

from attrs import define, field

from ._converters import none_if_none


@define
class Planet:
    """Bulk planet properties.

    Attributes
    ----------
    planet_mass_tot: float or None
        Total planet mass (interior + atmosphere) in Earth masses [M_earth].
        Set to 'none' if using radius_int in [interior_struct] instead.
    """

    planet_mass_tot = field(
        default='none',
        converter=none_if_none,
    )
