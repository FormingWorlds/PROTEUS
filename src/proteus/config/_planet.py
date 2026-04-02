from __future__ import annotations

from attr.validators import ge, gt, in_
from attrs import define, field

from ._converters import none_if_none


@define
class Elements:
    """Initial volatile inventory by element abundances.

    Each element has a mode (how the budget is specified) and a budget (the value).

    H_mode options:
        'oceans': H_budget in Earth oceans (1 EO ~ 1.55e20 kg H).
        'ppmw': H_budget in ppmw relative to volatile_reservoir mass.
        'kg': H_budget in kg (absolute).

    C_mode / N_mode / S_mode options:
        'C/H' / 'N/H' / 'S/H': budget is mass ratio to H.
        'ppmw': budget in ppmw relative to volatile_reservoir mass.
        'kg': budget in kg (absolute).

    use_metallicity: when True, C/N/S are scaled from solar metallicity
    relative to H, overriding C_mode/N_mode/S_mode settings.

    Attributes
    ----------
    H_mode: str
        How H_budget is interpreted: 'oceans', 'ppmw', 'kg'.
    H_budget: float
        Hydrogen inventory value (units depend on H_mode).
    C_mode: str
        How C_budget is interpreted: 'C/H', 'ppmw', 'kg'.
    C_budget: float
        Carbon inventory value (units depend on C_mode).
    N_mode: str
        How N_budget is interpreted: 'N/H', 'ppmw', 'kg'.
    N_budget: float
        Nitrogen inventory value (units depend on N_mode).
    S_mode: str
        How S_budget is interpreted: 'S/H', 'ppmw', 'kg'.
    S_budget: float
        Sulfur inventory value (units depend on S_mode).
    use_metallicity: bool
        Scale C/N/S from solar metallicity (overrides C/N/S mode+budget).
    metallicity: float
        Metallicity relative to solar, by mass (only if use_metallicity=True).
    """

    H_mode: str = field(default='oceans', validator=in_(('oceans', 'ppmw', 'kg')))
    H_budget: float = field(default=0.0, validator=ge(0))

    C_mode: str = field(default='C/H', validator=in_(('C/H', 'ppmw', 'kg')))
    C_budget: float = field(default=0.0, validator=ge(0))

    N_mode: str = field(default='N/H', validator=in_(('N/H', 'ppmw', 'kg')))
    N_budget: float = field(default=0.0, validator=ge(0))

    S_mode: str = field(default='S/H', validator=in_(('S/H', 'ppmw', 'kg')))
    S_budget: float = field(default=0.0, validator=ge(0))

    use_metallicity: bool = field(default=False)
    metallicity: float = field(default=1000.0, validator=ge(0))


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
        'accretion': White & Li (2025) parameterization.
    tsurf_init: float
        Initial magma surface temperature [K] (isothermal, linear, adiabatic).
    tcenter_init: float
        Center temperature [K] (linear only).
    f_accretion: float
        Accretion heat retention [0-1] (accretion mode, White & Li 2025).
    f_differentiation: float
        Differentiation heat retention [0-1] (accretion mode).
    volatile_mode: str
        How to set the initial volatile inventory: 'elements' or 'gas_prs'.
    volatile_reservoir: str
        Interior mass reference for ppmw calculations.
        'mantle': M_mantle = M_int - M_core (default, backward compatible).
        'mantle+core': M_int = M_mantle + M_core (total dry interior).
    elements: Elements
        Element abundance parameters (used when volatile_mode = 'elements').
    gas_prs: GasPrs
        Partial pressure parameters (used when volatile_mode = 'gas_prs').
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
    volatile_reservoir: str = field(default='mantle', validator=in_(('mantle', 'mantle+core')))
    elements: Elements = field(factory=Elements)
    gas_prs: GasPrs = field(factory=GasPrs)
