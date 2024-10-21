from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Elements:
    """Initial volatile inventory by planetary element abundances.

    Attributes
    ----------
    CH_ratio: float
        C/H ratio in mantle/atmosphere system.
    H_oceans: float
        Hydrogen inventory in units of equivalent Earth oceans, by mass.
    N_ppmw: float
        Nitrogen inventory in ppmw relative to mantle mass, by mass.
    S_ppmw: float
        Sulfur inventory in ppmw relative to mass of melt.
    """
    CH_ratio: float
    H_oceans: float
    N_ppmw: float
    S_ppmw: float


@define
class Volatiles:
    """Initial volatile inventory by partial pressures in atmosphere.

    Attributes
    ----------
    H2O: float
        Partial pressure of H2O.
    CO2: float
        Partial pressure of CO2.
    N2: float
       Partial pressure of N2.
    S2: float
       Partial pressure of S2.
    SO2: float
        Partial pressure of SO2.
    H2: float
       Partial pressure of H2.
    CH4: float
        Partial pressure of CH4.
    CO: float
       Partial pressure of CO.
    """
    H2O: float = field(default=0)
    CO2: float = field(default=0)
    N2: float = field(default=0)
    S2: float = field(default=0)
    SO2: float = field(default=0)
    H2: float = field(default=0)
    CH4: float = field(default=0)
    CO: float = field(default=0)


@define
class Delivery:
    """Initial volatile inventory, and delivery model selection

    Attributes
    ----------
    initial: str
        Set initial inventory to use, choice 'volatiles', 'elements'.
    module: str
        Delivery module to use (Not used as of yet).
    elements: Elements
        Set initial volatile inventory by planetary element abundances.
    volatiles: Volatiles
        Set initial volatile inventory by partial pressures in atmosphere.
    """
    initial: str = field(validator=validators.in_(('elements', 'volatiles')))

    module: str | None = field(validator=validators.in_((None,)), converter=none_if_none)

    elements: Elements
    volatiles: Volatiles
