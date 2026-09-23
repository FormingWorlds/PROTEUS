from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


def _reject_enabled_binodal(instance, attribute, value):
    """Reject `h2_binodal = true` until the feature is production ready.

    This gate is also what keeps the `dry_mantle = false` structure path
    water-only: `apply_binodal_h2` is the sole producer of a nonzero
    `H2_kg_liquid`, which `build_volatile_profile` would blend into the
    mantle EOS as `Chabrier:H`. The wet path's mass-conservation
    verification covers dissolved H2O only, and the per-shell binodal
    suppression in the mixed density removes dissolved H2 from the
    structure without returning it to the atmosphere, so lifting this
    gate requires extending the wet-path verification to H2 first
    (Zalmoxis tracker #64).
    """
    if value:
        raise ValueError(
            '`outgas.h2_binodal = true` is not yet supported: the H2-silicate '
            'binodal partitioning (Rogers et al. 2025) is not production '
            'ready. Keep `h2_binodal = false`.'
        )


@define
class Calliope:
    """Module parameters for Calliope.

    Attributes
    ----------
    include_H2O: bool
        If True, include H2O.
    include_CO2: bool
        If True, include CO2.
    include_N2: bool
        If True, include N2.
    include_S2: bool
        If True, include S2.
    include_SO2: bool
        If True, include SO2.
    include_H2S: bool
        If True, include H2S.
    include_NH3: bool
        If True, include NH3.
    include_H2: bool
        If True, include H2.
    include_CH4: bool
        If True, include CH4.
    include_CO: bool
        If True, include CO.
    include_He: bool
        If True, include He (noble gas; budget set in planet.elements).
    include_Ne: bool
        If True, include Ne (noble gas; budget set in planet.elements).
    include_Ar: bool
        If True, include Ar (noble gas; budget set in planet.elements).
    include_Kr: bool
        If True, include Kr (noble gas; budget set in planet.elements).
    include_Xe: bool
        If True, include Xe (noble gas; budget set in planet.elements).
    solubility: bool
        Enable solubility of volatiles into melt.
    nguess: int
        Maximum number of initial-guess samples for the CALLIOPE
        equilibrium solver. Default 1000.
    nsolve: int
        Maximum number of iterations of the CALLIOPE equilibrium
        solver per call. Default 3000.
    p_guess_max: float
        Upper bound [bar] of the CALLIOPE Monte-Carlo cold-start surface-
        pressure draw. Sets where the cold start samples, so raising it helps
        the solver find a high-pressure (e.g. sub-Neptune) basin faster. It
        does NOT raise the maximum pressure the solver can accept (CALLIOPE's
        fixed 1e7 bar box), so it is bounded to (0, 1e7]. Default 1e5.
    """

    include_H2O: bool = field(default=True)
    include_CO2: bool = field(default=True)
    include_N2: bool = field(default=True)
    include_S2: bool = field(default=True)
    include_SO2: bool = field(default=True)
    include_H2S: bool = field(default=True)
    include_NH3: bool = field(default=True)
    include_H2: bool = field(default=True)
    include_CH4: bool = field(default=True)
    include_CO: bool = field(default=True)
    # Noble gases are opt-in and default off, so a run with no noble budget is
    # unchanged. A noble gas contributes to the solve only when its flag is
    # true and its element budget in planet.elements is positive.
    include_He: bool = field(default=False)
    include_Ne: bool = field(default=False)
    include_Ar: bool = field(default=False)
    include_Kr: bool = field(default=False)
    include_Xe: bool = field(default=False)
    solubility: bool = field(default=True)
    nguess: int = field(default=int(1e3), validator=validators.gt(0))
    nsolve: int = field(default=int(3e3), validator=validators.gt(0))
    # Default 1e5 bar mirrors CALLIOPE's P_GUESS_MAX_BAR (so the default leaves
    # the cold-start draw unchanged); the 1e7 upper bound mirrors CALLIOPE's
    # P_CEILING_BAR solver box, above which a raised ceiling is unreachable
    # (the solver rejects roots beyond the box). The le() bound also rejects inf.
    p_guess_max: float = field(
        default=1.0e5, validator=validators.and_(validators.gt(0), validators.le(1.0e7))
    )

    def is_included(self, vol: str) -> bool:
        """Helper method for getting flag if `vol` is included in outgassing."""
        return getattr(self, f'include_{vol}')


@define
class Atmodeller:
    """Module parameters for Atmodeller (Bower+2025, ApJ 995:59).

    JAX-based volatile partitioning with real gas EOS, non-ideal
    solubility laws, and condensation. Replaces CALLIOPE for
    thermodynamically consistent magma-atmosphere equilibrium.

    Attributes
    ----------
    solver_mode : str
        Root-finding mode: 'robust' (slower compile, better convergence)
        or 'basic' (faster compile, less robust).
    solver_max_steps : int
        Maximum iterations for the root-finder.
    solver_multistart : int
        Number of random restarts for the root-finder.
    include_condensates : bool
        Enable condensate phases (graphite, etc.) in the equilibrium.
    solubility_H2O : str
        Solubility law for H2O. See atmodeller.solubility.library.
    solubility_CO2 : str
        Solubility law for CO2.
    solubility_H2 : str
        Solubility law for H2.
    solubility_N2 : str
        Solubility law for N2.
    solubility_S2 : str
        Solubility law for S2.
    solubility_CO : str
        Solubility law for CO. 'none' = no solubility.
    solubility_CH4 : str
        Solubility law for CH4. 'none' = no solubility.
    eos_H2O : str
        Real gas EOS for H2O. 'none' = ideal gas.
    eos_CO2 : str
        Real gas EOS for CO2. 'none' = ideal gas.
    eos_H2 : str
        Real gas EOS for H2. 'none' = ideal gas.
    eos_CH4 : str
        Real gas EOS for CH4. 'none' = ideal gas.
    eos_CO : str
        Real gas EOS for CO. 'none' = ideal gas.
    """

    solver_mode: str = field(
        default='robust',
        validator=validators.in_(('robust', 'basic')),
    )
    solver_max_steps: int = field(default=1024, validator=validators.gt(0))
    solver_multistart: int = field(default=10, validator=validators.gt(0))
    include_condensates: bool = field(default=True)
    solubility_H2O: str | None = field(default='H2O_peridotite_sossi23', converter=none_if_none)
    solubility_CO2: str | None = field(default='CO2_basalt_dixon95', converter=none_if_none)
    solubility_H2: str | None = field(default='H2_basalt_hirschmann12', converter=none_if_none)
    solubility_N2: str | None = field(default='N2_basalt_dasgupta22', converter=none_if_none)
    solubility_S2: str | None = field(
        default='S2_sulfide_basalt_boulliung23', converter=none_if_none
    )
    solubility_CO: str | None = field(default='CO_basalt_yoshioka19', converter=none_if_none)
    solubility_CH4: str | None = field(default='CH4_basalt_ardia13', converter=none_if_none)
    eos_H2O: str | None = field(default=None, converter=none_if_none)
    eos_CO2: str | None = field(default=None, converter=none_if_none)
    eos_H2: str | None = field(default=None, converter=none_if_none)
    eos_CH4: str | None = field(default=None, converter=none_if_none)
    eos_CO: str | None = field(default=None, converter=none_if_none)


@define
class Lavatmos:
    """Module parameters for LavAtmos rock vapourisation.

    Attributes
    ----------
    T_min: float
        Minimum surface temperature [K] used by LavAtmos.
    melt_comp_name: str
        Name of the melt composition file (without extension).
    P_melt: float
        Pressure used for melt activities [bar].
    xatol: float
        Absolute tolerance for LavAtmos fO2 solve.
    fO2_buffer_model: str
        IW buffer model used for LavAtmos fO2 solve. One of 'oneill', 'fischer'.
    """

    T_min: float = field(default=1500.0, validator=validators.gt(0.0))
    melt_comp_name: str = field(default='BSE_palm')
    P_melt: float = field(default=0.01, validator=validators.gt(0.0))
    xatol: float = field(default=1e-5, validator=validators.gt(0.0))
    fO2_buffer_model: str = field(
        default='oneill', validator=validators.in_(('oneill', 'fischer'))
    )


@define
class Outgas:
    """Outgassing parameters (fO2) and included volatiles.

    Attributes
    ----------
    module: str
        Outgassing module to be used. Choices: 'calliope', 'atmodeller', 'dummy'.
    fO2_shift_IW: float
        Oxygen fugacity relative to Iron-Wustite [log10 units].
    mass_thresh: float
        Minimum threshold for element mass [kg]. Inventories below this are set to zero.
    h2_binodal: bool
        Enable binodal-controlled H2 partitioning between atmosphere and
        magma ocean using the Rogers+2025 H2-MgSiO3 miscibility model.
    T_floor: float
        Temperature floor [K]. The outgassing temperature is clamped to
        this value from below before the chemistry solve.
    solver_rtol: float
        Relative tolerance for the volatile equilibrium solver.
    solver_atol: float
        Absolute tolerance for the volatile equilibrium solver.
    trap_mode: str
        Solid-phase volatile trapping during mantle crystallisation.
        Choices: 'none' (off), 'constant' (fixed F_tl), 'dynamic'
        (F_tl from the cooling rate, after Sim et al. 2024 Eq. 7).
    trap_F_tl: float
        Trapped melt fraction [1] used by `trap_mode = 'constant'`. The
        disaggregation melt fraction that bounds F_tl in `trap_mode =
        'dynamic'` is `interior_energetics.rfront_loc`, the melt fraction of
        the solver's own rheological transition, not a separate field.
    trap_tau: float
        Compaction time scale [yr] used by `trap_mode = 'dynamic'`.
    trap_tau_source: str
        Where the compaction time comes from in `trap_mode = 'dynamic'`.
        'fixed' uses `trap_tau` and `trap_delta_T`, the published linear
        law, and works with any interior module. 'aragog' replaces both
        by the drainage integral over the freezing front the interior
        solver resolves, which needs neither parameter; it falls back to
        'fixed' when the interior state is unavailable.
    trap_phi_min: float
        Porosity below which a node counts as solid [1]. Needed because
        the density-derived porosity never reaches exactly zero.
    trap_mush_log10visc: float
        Log10 viscosity of the mush near the solidus [log10(Pa s)],
        entering the matrix deformation time. Negative selects the
        default of 20, mid-range of the 18 to 22 the literature allows.
    trap_n_front_min: int
        Fewest nodes a freezing front must span before its drainage is
        integrated rather than bounded above.
    trap_max_front_fraction: float
        Largest fraction of the mantle thickness the front may occupy
        before its drainage is bounded above instead of integrated [1].
        The default of 1 sets no limit: where percolation controls the
        drainage, the retained fraction depends on the ratio of drainage
        speed to front speed and not on the front thickness, and a thicker
        front only lengthens the residence time.
    trap_delta_T: float
        Solidus to freezing-front temperature difference [K]. Negative
        derives it from the active melting curves as
        `interior_energetics.rfront_loc * (T_liquidus - T_solidus)`;
        positive overrides the derivation. Sim et al. fix 100 K.
    D_const_H2O: float
        Crystal/melt partition coefficient of H2O [1], D = w_solid / w_liquid.
    D_const_CO2: float
        Crystal/melt partition coefficient of CO2 [1], D = w_solid / w_liquid.
    D_const_O2: float
        Crystal/melt partition coefficient of O2 [1], D = w_solid / w_liquid.
    D_const_H2: float
        Crystal/melt partition coefficient of H2 [1], D = w_solid / w_liquid.
    D_const_CH4: float
        Crystal/melt partition coefficient of CH4 [1], D = w_solid / w_liquid.
    D_const_CO: float
        Crystal/melt partition coefficient of CO [1], D = w_solid / w_liquid.
    D_const_N2: float
        Crystal/melt partition coefficient of N2 [1], D = w_solid / w_liquid.
    D_const_NH3: float
        Crystal/melt partition coefficient of NH3 [1], D = w_solid / w_liquid.
    D_const_S2: float
        Crystal/melt partition coefficient of S2 [1], D = w_solid / w_liquid.
    D_const_SO2: float
        Crystal/melt partition coefficient of SO2 [1], D = w_solid / w_liquid.
    D_const_H2S: float
        Crystal/melt partition coefficient of H2S [1], D = w_solid / w_liquid.
    calliope: Calliope
        Parameters for CALLIOPE module.
    atmodeller: Atmodeller
        Parameters for atmodeller module.
    vapourise: bool
        Enable rock vapourisation via LavAtmos/ThermoEngineLite. Requires
        `LAVA_DIR` and `FC_DIR` to be set; see the optional modules
        installation guide. LavAtmos parameters are set in `outgas.lavatmos`.
    lavatmos: Lavatmos
        Parameters for the LavAtmos rock-vapour module.
    """

    module: str = field(
        default='calliope', validator=validators.in_(('calliope', 'atmodeller', 'dummy'))
    )
    fO2_shift_IW: float = field(
        default=4.0, validator=[validators.ge(-12.0), validators.le(12.0)]
    )

    mass_thresh: float = field(default=1e16, validator=validators.gt(0.0))
    # H2-silicate binodal partitioning (Rogers et al. 2025). Not yet
    # production ready: the H2 mass-fraction definition for multi-species
    # inventories and the coupling to the structure side are unsettled,
    # so enabling the flag is rejected at config load.
    h2_binodal: bool = field(
        default=False,
        validator=_reject_enabled_binodal,
    )

    # Shared solver parameters (calliope + atmodeller)
    T_floor: float = field(default=700.0, validator=validators.gt(0.0))
    solver_rtol: float = field(default=1e-4, validator=validators.gt(0.0))
    solver_atol: float = field(default=1e-6, validator=validators.gt(0.0))

    # Solid-phase volatile trapping. Backend-agnostic: the bracket runs before
    # and independently of the chemistry solve, so these sit beside T_floor
    # rather than under [outgas.calliope]. Default 'none' leaves every existing
    # run unchanged.
    trap_mode: str = field(
        default='none', validator=validators.in_(('none', 'constant', 'dynamic'))
    )
    trap_F_tl: float = field(default=0.01, validator=[validators.ge(0.0), validators.le(1.0)])
    trap_tau: float = field(default=1.0e6, validator=validators.gt(0.0))
    trap_delta_T: float = field(default=-1.0)

    # Drainage-integral parameters. Used only by trap_tau_source = 'aragog',
    # which computes the compaction time from the front the interior solver
    # resolves instead of taking it as an input.
    trap_tau_source: str = field(default='fixed', validator=validators.in_(('fixed', 'aragog')))
    trap_phi_min: float = field(
        default=0.01, validator=[validators.gt(0.0), validators.lt(1.0)]
    )
    trap_mush_log10visc: float = field(default=-1.0)
    trap_n_front_min: int = field(default=3, validator=validators.ge(2))
    trap_max_front_fraction: float = field(
        default=1.0, validator=[validators.gt(0.0), validators.le(1.0)]
    )

    # Crystal/melt partition coefficients, D = w_solid / w_liquid, matching the
    # per-phase mass fractions the structure side carries. Water is the only
    # species with appreciable lattice incorporation (Hauri, Gaetani & Green
    # 2006; Hirschmann et al. 2016); the rest are buried through the
    # interstitial-melt term alone, which is a physical statement rather than a
    # placeholder.
    D_const_H2O: float = field(default=0.0017, validator=validators.ge(0.0))
    D_const_CO2: float = field(default=0.0, validator=validators.ge(0.0))
    D_const_O2: float = field(default=0.0, validator=validators.ge(0.0))
    D_const_H2: float = field(default=0.0, validator=validators.ge(0.0))
    D_const_CH4: float = field(default=0.0, validator=validators.ge(0.0))
    D_const_CO: float = field(default=0.0, validator=validators.ge(0.0))
    D_const_N2: float = field(default=0.0, validator=validators.ge(0.0))
    D_const_NH3: float = field(default=0.0, validator=validators.ge(0.0))
    D_const_S2: float = field(default=0.0, validator=validators.ge(0.0))
    D_const_SO2: float = field(default=0.0, validator=validators.ge(0.0))
    D_const_H2S: float = field(default=0.0, validator=validators.ge(0.0))

    calliope: Calliope = field(factory=Calliope)
    atmodeller: Atmodeller = field(factory=Atmodeller)
    lavatmos: Lavatmos = field(factory=Lavatmos)

    # LavAtmos / silicate coupling is opt-in: default to disabled.
    vapourise: bool = field(default=False)


# Thematic grouping for the generated configuration reference. Each entry is
# ``(heading, qualifier, option names)``; a heading of ``None`` renders the
# section's opening table with no heading of its own. Order here is the order
# on the page, and is independent of the order the fields are declared in.
DOC_GROUPS = {
    'Calliope': (
        (
            'Species switches',
            'set to `false` to exclude a species from the equilibrium',
            (
                'include_H2O',
                'include_CO2',
                'include_N2',
                'include_S2',
                'include_SO2',
                'include_H2S',
                'include_NH3',
                'include_H2',
                'include_CH4',
                'include_CO',
                'include_He',
                'include_Ne',
                'include_Ar',
                'include_Kr',
                'include_Xe',
                'solubility',
            ),
        ),
        ('Solver', None, ('nguess', 'nsolve', 'p_guess_max')),
    ),
    'Atmodeller': (
        (
            None,
            None,
            (
                'solver_mode',
                'solver_max_steps',
                'solver_multistart',
                'include_condensates',
            ),
        ),
        (
            'Solubility laws',
            'set to `"none"` to disable dissolution for a species',
            (
                'solubility_H2O',
                'solubility_CO2',
                'solubility_H2',
                'solubility_N2',
                'solubility_S2',
                'solubility_CO',
                'solubility_CH4',
            ),
        ),
        (
            'Real gas EOS',
            'set to `"none"` for ideal gas',
            ('eos_H2O', 'eos_CO2', 'eos_H2', 'eos_CH4', 'eos_CO'),
        ),
    ),
}
