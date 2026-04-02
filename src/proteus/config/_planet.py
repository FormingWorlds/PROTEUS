from __future__ import annotations

from attr.validators import ge, gt, in_
from attrs import define, field

from ._converters import none_if_none


@define
class Elements:
    """Initial volatile inventory by planetary *bulk* element abundances.

    There are various ways to set these. You can specify a metallicity relative to solar
    alongside a total hydrogen abundance by providing `use_metallicity=True`.

    Instead of metallicity, provide the abundance of each element with either specific
    mass ratio relative to hydrogen or in terms of the concentration in the mantle.
    For X in {C, N, S}: only XH_ratio or X_ppmw should be used at any one time.

    Hydrogen abundance is set via *either* `H_oceans`, which is the number of oceans of
    hydrogen in the planet's mantle at initialisation (assumed to be fully molten). Or,
    you can set the hydrogen abundance in ppm relative to the mantle mass with `H_ppmw`.
    For hydrogen: only H_oceans or H_ppmw should be used at any one time.

    Attributes
    ----------
    H_oceans: float
        Absolute hydrogen inventory, units of equivalent Earth oceans.
    H_kg: float
        Absolute hydrogen inventory, kg.
    H_ppmw: float
        Relative hydrogen inventory, ppmw relative to mantle mass.

    use_metallicity: bool
        Whether or not to specify the elemental abundances in terms of solar metallicity
    metallicity: float
        Elemental metallicity relative to solar metallicity, by mass

    CH_ratio: float
        Carbon metallicity. C/H mass ratio in combined mantle+atmosphere system.
    C_kg: float
        Absolute carbon inventory, kg.
    C_ppmw: float
        Relative carbon inventory, ppmw relative to mantle mass.

    NH_ratio: float
        Nitrogen metallicity. N/H mass ratio in combined mantle+atmosphere system.
    N_kg: float
        Absolute nitrogen inventory, kg.
    N_ppmw: float
        Relative nitrogen inventory, ppmw relative to mantle mass.

    SH_ratio: float
        Sulfur metallicity. C/H mass ratio in combined mantle+atmosphere system.
    S_kg: float
        Absolute sulfur inventory, kg.
    S_ppmw: float
        Absolute sulfur inventory, ppmw relative to mantle mass.
    """

    use_metallicity: float = field(default=False)
    metallicity: float = field(default=1000.0, validator=ge(0))

    H_oceans: float = field(default=0.0, validator=ge(0))
    H_kg: float = field(default=0.0, validator=ge(0))
    H_ppmw: float = field(default=0.0, validator=ge(0))

    CH_ratio: float = field(default=0.0, validator=ge(0))
    C_kg: float = field(default=0.0, validator=ge(0))
    C_ppmw: float = field(default=0.0, validator=ge(0))

    NH_ratio: float = field(default=0.0, validator=ge(0))
    N_kg: float = field(default=0.0, validator=ge(0))
    N_ppmw: float = field(default=0.0, validator=ge(0))

    SH_ratio: float = field(default=0.0, validator=ge(0))
    S_kg: float = field(default=0.0, validator=ge(0))
    S_ppmw: float = field(default=0.0, validator=ge(0))


@define
class GasPrs:
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

    H2O: float = field(default=0, validator=ge(0))
    CO2: float = field(default=0, validator=ge(0))
    N2: float = field(default=0, validator=ge(0))
    S2: float = field(default=0, validator=ge(0))
    SO2: float = field(default=0, validator=ge(0))
    H2S: float = field(default=0, validator=ge(0))
    NH3: float = field(default=0, validator=ge(0))
    H2: float = field(default=0, validator=ge(0))
    CH4: float = field(default=0, validator=ge(0))
    CO: float = field(default=0, validator=ge(0))

    def get_pressure(self, s: str) -> float:
        """Helper method for getting the pressure for `vol` by string."""
        return getattr(self, s)


@define
class Planet:
    """Bulk planet properties, initial temperature profile, and volatile inventory.

    Attributes
    ----------
    mass_tot: float
        Total planet mass (interior + atmosphere) in Earth masses [M_earth].
    temperature_mode: str
        How to set the initial temperature profile.
        'isothermal': T = tsurf_init everywhere.
        'linear': T from tsurf_init (surface) to tcenter_init (center).
        'adiabatic': integrate dT/dP|_S downward from tsurf_init.
        'accretion': White & Li (2025) parameterization. Computes T from
        accretion and differentiation energy. Requires Zalmoxis.
    tsurf_init: float
        Initial magma surface temperature [K]. Used by isothermal, linear,
        and adiabatic modes. Ignored in accretion mode (computed by Zalmoxis).
    tcenter_init: float
        Center temperature [K]. Used by linear mode only (center endpoint).
        Ignored in isothermal, adiabatic, and accretion modes.
    f_accretion: float
        Heat retention efficiency for accretion energy [0-1].
        Only used in accretion mode. Default 0.04 (White & Li 2025).
    f_differentiation: float
        Heat retention efficiency for core-mantle differentiation energy [0-1].
        Only used in accretion mode. Default 0.50 (White & Li 2025).
    volatile_mode: str
        How to set the initial volatile inventory. Options: 'elements', 'gas_prs'.
    elements: Elements
        Parameters for setting volatile inventory by element abundances.
    gas_prs: GasPrs
        Parameters for setting volatile inventory by partial pressures.
    """

    mass_tot = field(
        default='none',
        converter=none_if_none,
    )

    # Initial temperature profile
    temperature_mode: str = field(
        default='adiabatic',
        validator=in_(('isothermal', 'linear', 'adiabatic', 'accretion')),
    )
    tsurf_init: float = field(default=4000.0, validator=gt(0))
    tcenter_init: float = field(default=6000.0, validator=gt(0))
    f_accretion: float = field(default=0.04, validator=ge(0))
    f_differentiation: float = field(default=0.50, validator=ge(0))

    # Initial volatile inventory
    volatile_mode: str = field(default='elements', validator=in_(('elements', 'gas_prs')))
    elements: Elements = field(factory=Elements)
    gas_prs: GasPrs = field(factory=GasPrs)
