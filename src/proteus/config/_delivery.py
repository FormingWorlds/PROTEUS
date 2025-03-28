from __future__ import annotations

from attr.validators import ge, gt, in_
from attrs import define, field

from ._converters import none_if_none


@define
class Elements:
    """Initial volatile inventory by planetary element abundances.

    For hydrogen: only H_oceans or H_ppmw should be used at any one time.
    For X in {C, N, S}: only XH_ratio or X_ppmw should be used at any one time.

    Attributes
    ----------
    H_oceans: float
        Absolute hydrogen inventory, units of equivalent Earth oceans.
    H_ppmw: float
        Relative hydrogen inventory, ppmw relative to mantle mass.

    CH_ratio: float
        Carbon metallicity. C/H mass ratio in combined mantle+atmosphere system.
    C_ppmw: float
        Relative carbon inventory, ppmw relative to mantle mass.

    NH_ratio: float
        Nitrogen metallicity. N/H mass ratio in combined mantle+atmosphere system.
    N_ppmw: float
        Relative nitrogen inventory, ppmw relative to mantle mass.

    SH_ratio: float
        Sulfur metallicity. C/H mass ratio in combined mantle+atmosphere system.
    S_ppmw: float
        Absolute sulfur inventory, ppmw relative to mantle mass.
    """
    H_oceans: float = field(default=0.0, validator=ge(0))
    H_ppmw: float   = field(default=0.0, validator=ge(0))

    CH_ratio: float = field(default=0.0, validator=ge(0))
    C_ppmw: float   = field(default=0.0, validator=ge(0))

    NH_ratio: float = field(default=0.0, validator=ge(0))
    N_ppmw: float   = field(default=0.0, validator=ge(0))

    SH_ratio: float = field(default=0.0, validator=ge(0))
    S_ppmw: float   = field(default=0.0, validator=ge(0))


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
    H2S: float
        Initial atmospheric partial surface pressure of H2S [bar].
    NH3: float
        Initial atmospheric partial surface pressure of NH3 [bar].
    H2: float
        Initial atmospheric partial surface pressure of H2 [bar].
    CH4: float
        Initial atmospheric partial surface pressure of CH4 [bar].
    CO: float
        Initial atmospheric partial surface pressure of CO [bar].
    """
    H2O: float  = field(default=0, validator=ge(0))
    CO2: float  = field(default=0, validator=ge(0))
    N2: float   = field(default=0, validator=ge(0))
    S2: float   = field(default=0, validator=ge(0))
    SO2: float  = field(default=0, validator=ge(0))
    H2S: float  = field(default=0, validator=ge(0))
    NH3: float  = field(default=0, validator=ge(0))
    H2: float   = field(default=0, validator=ge(0))
    CH4: float  = field(default=0, validator=ge(0))
    CO: float   = field(default=0, validator=ge(0))

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

    module: str | None = field(validator=in_((None,)), converter=none_if_none)

    elements: Elements   = field(factory=Elements)
    volatiles: Volatiles = field(factory=Volatiles)

    initial: str = field(default='elements', validator=in_(('elements', 'volatiles')))

    radio_tref: float = field(default=4.55,  validator=gt(0))
    radio_K: float    = field(default=310.0, validator=ge(0))
    radio_U: float    = field(default=0.031, validator=ge(0))
    radio_Th: float   = field(default=0.124, validator=ge(0))
