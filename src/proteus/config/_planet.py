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
        How to set the initial mantle thermal state.
        'isothermal': T = tsurf_init everywhere.
        'linear': T from tsurf_init (surface) to tcenter_init (center).
        'adiabatic': integrate dT/dP|_S downward from tsurf_init.
        'adiabatic_from_cmb': anchor the adiabat at the core-mantle
            boundary (T = tcmb_init at P_cmb) and integrate upward to the
            surface. Use this when the surface-anchored adiabat under the
            current EOS would put the mantle into the mushy zone at IC
            and you want to force a fully molten initial state by pinning
            the CMB temperature instead. PROTEUS converts the (P_cmb,
            tcmb_init) anchor into a target entropy via PALEOS-2phase
            lookup, then hands that S to the interior solver as if it
            were the isentropic IC.
        'accretion': White & Li (2025) parameterization.
        'isentropic': set the initial specific entropy directly via
            ini_entropy + ini_dsdr (bypasses PALEOS lookup; matches the
            CHILI intercomparison protocol). The interior solver maps the
            entropy IC to T(P) via its own EOS table.
    tsurf_init: float
        Initial magma surface temperature [K] (isothermal, linear, adiabatic).
        Ignored when temperature_mode = 'isentropic' or 'adiabatic_from_cmb'.
    tcmb_init: float
        Initial core-mantle boundary temperature [K] (adiabatic_from_cmb only).
        The mantle adiabat is anchored at this temperature at P = P_cmb
        and integrated outward to the surface.
    tcenter_init: float
        Center temperature [K] (linear only).
    f_accretion: float
        Accretion heat retention [0-1] (accretion mode, White & Li 2025).
    f_differentiation: float
        Differentiation heat retention [0-1] (accretion mode).
    ini_entropy: float
        Initial specific entropy at the surface [J/kg/K] (isentropic mode).
        CHILI Earth-SPIDER reference: 3900.0.
    ini_dsdr: float
        Initial entropy gradient with radius [J/kg/K/m] (isentropic mode).
        CHILI Earth-SPIDER reference: -4.698e-6 (small numerical
        perturbation needed for SPIDER's BDF stability on a uniform IC).
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
    prevent_warming: bool
        When True, require the planet to monotonically cool over time.
        Enforced in all atmosphere modules and termination checks.
    """

    mass_tot = field(
        default='none',
        converter=none_if_none,
    )

    # Initial temperature profile
    # Default = 'adiabatic_from_cmb' (Stage 1a lock, 2026-04-20): CMB-anchored
    # adiabatic IC is the canonical starting state for the UnifyCoupling paper.
    # Deeper IC exploration (accretion-scaling, liquidus-offset, etc.) is
    # deferred to the dedicated Stage 4 (see roadmap).
    temperature_mode: str = field(
        default='adiabatic_from_cmb',
        validator=in_(
            ('isothermal', 'linear', 'adiabatic', 'adiabatic_from_cmb', 'accretion', 'isentropic')
        ),
    )
    tsurf_init: float = field(default=4000.0, validator=gt(0))
    tcmb_init: float = field(default=6000.0, validator=gt(0))
    tcenter_init: float = field(default=6000.0, validator=gt(0))
    f_accretion: float = field(default=0.04, validator=ge(0))
    f_differentiation: float = field(default=0.50, validator=ge(0))

    # Isentropic IC: set the initial specific entropy directly. Used when
    # temperature_mode = 'isentropic' (CHILI protocol). The interior solver
    # maps S -> T(P) via its own EOS table; tsurf_init is ignored.
    ini_entropy: float = field(default=3900.0, validator=gt(0))
    ini_dsdr: float = field(default=-4.698e-6)

    # Initial volatile inventory
    volatile_mode: str = field(default='elements', validator=in_(('elements', 'gas_prs')))
    volatile_reservoir: str = field(default='mantle', validator=in_(('mantle', 'mantle+core')))
    elements: Elements = field(factory=Elements)
    gas_prs: GasPrs = field(factory=GasPrs)

    # Structure override: bypass the root finder and use a fixed R_int.
    # Needed for SPIDER/Aragog parity runs where the two energetics
    # modules have different Adams-Williamson density implementations.
    # Set to the SPIDER run's R_int (in meters) to force both codes
    # onto the same mesh. Default None = use the root finder.
    R_int_override = field(default='none', converter=none_if_none)

    # Cooling constraint
    prevent_warming: bool = field(default=False)
