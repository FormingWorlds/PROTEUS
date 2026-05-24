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

    O_mode options (whole-planet oxygen accounting, see issue #677):
        'ppmw': O_budget in ppmw relative to volatile_reservoir mass.
        'kg': O_budget in kg (absolute).
        'FeO_mantle_wt_pct': O_budget interpreted as mantle FeO weight
            percent; converted to kg-of-O via M_O/M_FeO = 0.2227.
            Unit-of-convenience for petrologists. Does NOT change the
            mantle EOS density (PALEOS still assumes its built-in FeO
            content); the user-supplied wt% only sets the atmospheric
            and dissolved O budget that PROTEUS carries through the
            mass-balance bookkeeping.
        'ic_chemistry': do not pre-populate O_kg_total; let the first
            outgas call (CALLIOPE / atmodeller) populate it from the
            fO2-buffered equilibrium given the H/C/N/S budgets at IC.
            This mode preserves legacy behaviour for configs that
            predate the issue #677 fix.

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
    O_mode: str
        How O_budget is interpreted: 'ppmw', 'kg', 'FeO_mantle_wt_pct',
        'ic_chemistry'.
    O_budget: float
        Oxygen inventory value (units depend on O_mode). Ignored when
        O_mode = 'ic_chemistry'.
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

    O_mode: str = field(
        default='ic_chemistry',
        validator=in_(('ppmw', 'kg', 'FeO_mantle_wt_pct', 'ic_chemistry')),
    )
    O_budget: float = field(default=0.0, validator=ge(0))

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
        'liquidus_super': anchor the adiabat at T = T_liq(P_cmb) +
            delta_T_super at the core-mantle boundary, where T_liq is
            the Fei et al. (2021, PRL 127, 135701) MgSiO3 melting curve
            (piecewise Simon-Glatzel, with the Belonoshko et al. 2005
            low-pressure branch below 2.55 GPa, exactly matching the
            PALEOS internal liquidus). The adiabat is then integrated
            upward to the surface. Use this for EOS-agnostic IC
            comparison: the anchor is set by a third-party melting
            curve so it does not bake in either the WB17 or PALEOS
            entropy convention. delta_T_super (in K) is the
            user-controlled superliquidus offset.
    tsurf_init: float
        Initial magma surface temperature [K] (isothermal, linear, adiabatic).
        Ignored when temperature_mode = 'isentropic', 'adiabatic_from_cmb',
        or 'liquidus_super'.
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
    delta_T_super: float
        Superliquidus offset [K] at the core-mantle boundary
        (liquidus_super mode only). The CMB anchor temperature is
        T_cmb = T_liq_Fei2021(P_cmb) + delta_T_super. Default 500 K
        gives a fully molten initial state across the full mantle for
        Earth-mass and super-Earth planets. Setting delta_T_super = 0
        anchors the IC adiabat exactly at the liquidus.
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
    fO2_source: str
        How the chemistry solver treats atmospheric fO2.

        'user_constant' (default, legacy-compatible): fO2 is buffered to
            the iron-wustite offset set by ``outgas.fO2_shift_IW``;
            atmospheric and dissolved O are derived from the equilibrium
            chemistry at that fO2. This is the behaviour PROTEUS has
            shipped to date.
        'from_O_budget' (Path C): the user O budget (from
            ``planet.elements.O_mode``/``O_budget``) is authoritative;
            fO2 is *derived* by the chemistry solver as the IW-buffer
            offset that produces the supplied O inventory. Use this when
            you want whole-planet O accounting to drive the redox state
            instead of buffering to a fixed dIW. Requires
            ``O_mode != 'ic_chemistry'`` (the chemistry needs an O
            target to invert against).
        'from_mantle_redox' (reserved): fO2 is derived from a tracked
            Fe3+/Fe2+ ratio in the silicate melt (Schaefer et al. 2024
            / issue #653). NOT YET IMPLEMENTED; the config-level
            validator rejects this value until the radial fO2
            framework lands.
    prevent_warming: bool
        When True, require the planet to monotonically cool over time.
        Enforced in all atmosphere modules and termination checks.
    """

    mass_tot: float = field(default=1.0, validator=gt(0))

    # Initial temperature profile. Default 'adiabatic_from_cmb' anchors
    # the adiabat at T = tcmb_init at the core-mantle boundary and
    # integrates upward to the surface; this is the canonical starting
    # state across the codebase. The other six modes cover isothermal,
    # linear, surface-anchored adiabatic, accretion (White & Li 2025),
    # isentropic (CHILI protocol), and liquidus-anchored ICs.
    temperature_mode: str = field(
        default='adiabatic_from_cmb',
        validator=in_(
            (
                'isothermal',
                'linear',
                'adiabatic',
                'adiabatic_from_cmb',
                'accretion',
                'isentropic',
                'liquidus_super',
            )
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

    # Superliquidus offset at the CMB for temperature_mode = 'liquidus_super'.
    # T_cmb anchor is T_liq_Fei2021(P_cmb) + delta_T_super; the adiabat is
    # then integrated upward to the surface. Default 500 K guarantees a
    # fully molten state across Earth-mass and super-Earth mantles. Setting
    # delta_T_super = 0 places the IC adiabat exactly on the liquidus.
    delta_T_super: float = field(default=500.0, validator=ge(0))

    # Initial volatile inventory
    volatile_mode: str = field(default='elements', validator=in_(('elements', 'gas_prs')))
    volatile_reservoir: str = field(default='mantle', validator=in_(('mantle', 'mantle+core')))
    elements: Elements = field(factory=Elements)
    gas_prs: GasPrs = field(factory=GasPrs)

    # fO2 source. Default 'user_constant' preserves the legacy behaviour
    # where outgas.fO2_shift_IW buffers atmospheric fO2 and the chemistry
    # solver returns the implied O inventory. 'from_O_budget' inverts the
    # roles (Path C); 'from_mantle_redox' is reserved for issue #653 and
    # rejected by the config-level validator below until that work lands.
    fO2_source: str = field(
        default='user_constant',
        validator=in_(('user_constant', 'from_O_budget', 'from_mantle_redox')),
    )

    # Structure override: bypass the root finder and use a fixed R_int.
    # Needed for SPIDER/Aragog parity runs where the two energetics
    # modules have different Adams-Williamson density implementations.
    # Set to the SPIDER run's R_int (in meters) to force both codes
    # onto the same mesh. Default None = use the root finder.
    R_int_override = field(default='none', converter=none_if_none)

    # Cooling constraint
    prevent_warming: bool = field(default=False)
