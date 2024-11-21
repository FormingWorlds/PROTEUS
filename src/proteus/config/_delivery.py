from __future__ import annotations

from attr.validators import ge, gt, in_
from attrs import define, field

from ._converters import none_if_none


@define
class Elements:
    """Initial volatile inventory by planetary element abundances.

    Attributes
    ----------
    CH_ratio: float
        Volatile C/H nass ratio in combined mantle+atmosphere system.
    H_oceans: float
        Bulk hydrogen inventory in units of equivalent Earth oceans.
    N_ppmw: float
        Bulk nitrogen inventory in ppmw relative to mantle mass.
    S_ppmw: float
        Bulk sulfur inventory in ppmw relative to mantle mass.
    """
    CH_ratio: float = field(validator=gt(0))
    H_oceans: float = field(validator=gt(0))
    N_ppmw: float = field(validator=ge(0))
    S_ppmw: float = field(validator=ge(0))


@define
class Volatiles:
    """Initial volatile inventory set by partial pressures in atmosphere.

    Attributes
    ----------
    H2O: float
        Initial atmospheric partial surface pressure of H2O [bar].
    CO2: float
        Initial atmospheric partial surface pressure of CO2 [bar].
    N2: float
        Initial atmospheric partial surface pressure of N2 [bar].
    S2: float
        Initial atmospheric partial surface pressure of S2 [bar].
    SO2: float
        Initial atmospheric partial surface pressure of SO2 [bar].
    H2: float
        Initial atmospheric partial surface pressure of H2 [bar].
    CH4: float
        Initial atmospheric partial surface pressure of CH4 [bar].
    CO: float
        Initial atmospheric partial surface pressure of CO [bar].
    """
    H2O: float = field(default=0, validator=ge(0))
    CO2: float = field(default=0, validator=ge(0))
    N2: float = field(default=0, validator=ge(0))
    S2: float = field(default=0, validator=ge(0))
    SO2: float = field(default=0, validator=ge(0))
    H2: float = field(default=0, validator=ge(0))
    CH4: float = field(default=0, validator=ge(0))
    CO: float = field(default=0, validator=ge(0))

    def get_pressure(self, s: str) -> float:
        """Helper method for getting the pressure for `vol` by string."""
        return getattr(self, s)

@define
class Delivery:
    """Initial volatile inventory, radionuclide concentration, and delivery model selection.

    Attributes
    ----------
    initial: str
        Method by which to set the initial volatile inventory to use. Options: 'volatiles', 'elements'.
    module: str
        Delivery module to use (Not used as of yet).
    elements: Elements
        Parameters used when setting volatile inventory by element abundances.
    volatiles: Volatiles
        Parameters used when setting volatile inventory by partial pressures.
    radio_tref: float
        Reference age for setting radioactive decay [Gyr].
    radio_U: float
        Concentration (ppmw) of uranium at reference age of `t=radio_tref`
    radio_K: float
        Concentration (ppmw) of potassium at reference age of `t=radio_tref`
    radio_Th: float
        Concentration (ppmw) of thorium at reference age of `t=radio_tref`
    """
    initial: str = field(validator=in_(('elements', 'volatiles')))

    module: str | None = field(validator=in_((None,)), converter=none_if_none)

    elements: Elements
    volatiles: Volatiles

    radio_tref: float = field(validator=gt(0))
    radio_U: float    = field(validator=ge(0))
    radio_K: float    = field(validator=ge(0))
    radio_Th: float   = field(validator=ge(0))
