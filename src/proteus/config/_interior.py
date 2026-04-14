from __future__ import annotations

import logging
import warnings

from attrs import define, field
from attrs.validators import ge, gt, in_, lt

log = logging.getLogger('fwl.' + __name__)


def valid_spider(instance, attribute, value):
    if instance.module != 'spider':
        return

    # at least one energy term enabled
    if not (
        instance.trans_conduction
        or instance.trans_convection
        or instance.trans_mixing
        or instance.trans_grav_sep
    ):
        raise ValueError('Must enable at least one energy transport term in SPIDER')


@define
class Spider:
    """SPIDER-specific parameters.

    ``solver_type`` is the SUNDIALS integrator choice. The relative
    tolerance (``tolerance_rel``) was promoted to
    ``[interior_energetics].rtol`` in the Tier 4 refactor
    (2026-04-08) and is kept here only as a deprecated alias for
    backwards compat.

    ``matprop_smooth_width`` is the SPIDER-specific smoothing width
    for material-property blending across the solidus/liquidus. It
    was briefly top-level (Tier 4) for SPIDER-Aragog parity, but the
    2026-04-09 Aragog Jgrav fix replaced the tanh smoothing with a
    parameter-free cubic Hermite polynomial, so the knob is once
    again SPIDER-only.

    Attributes
    ----------
    solver_type: str
        SUNDIALS integrator choice. Choices: 'adams', 'bdf'.
    tolerance_rel: float
        Deprecated alias for ``Interior.rtol``. Set
        ``interior_energetics.rtol`` at the top level instead.
    matprop_smooth_width: float
        Melt-fraction window width for smoothing SPIDER's material
        properties across the solidus/liquidus. Passed to SPIDER as
        ``-matprop_smooth_width``. Unused by Aragog.
    """

    solver_type: str = field(default='bdf', validator=in_(('adams', 'bdf')))

    # Sentinel -1 means "not set"; old configs that set it will copy to
    # the top-level rtol in Interior.__attrs_post_init__ and emit a
    # DeprecationWarning. The validator allows -1 as a pass-through.
    tolerance_rel: float = field(default=-1.0)

    # Material-property smoothing width for the phase-boundary blend.
    # Passed to SPIDER as ``-matprop_smooth_width`` and to Aragog's
    # ``EntropyPhaseEvaluator`` via ``_PhaseMixedParameters``. Controls
    # the tanh transition between two-phase (Lever Rule) and single-
    # phase material properties near the solidus and liquidus.
    matprop_smooth_width: float = field(
        default=1.0e-2, validator=(gt(0), lt(1))
    )


def valid_aragog(instance, attribute, value):
    if instance.module != 'aragog':
        return

    # at least one energy term enabled (uses top-level interior fields)
    if not (
        instance.trans_conduction
        or instance.trans_convection
        or instance.trans_mixing
        or instance.trans_grav_sep
    ):
        raise ValueError('Must enable at least one energy transport term in Aragog')


@define
class Aragog:
    """Aragog-specific parameters.

    Attributes
    ----------
    dilatation: bool
        Whether to include dilatation (PdV) heating from gravitational
        separation. Default is False. SPIDER does not include this term,
        so enabling it breaks parity. The implementation is available for
        future physics work where both solvers should have it.
    mass_coordinates: bool
        Whether to use mass coordinates in the model. Default is True.
        Uses uniform spacing in mass coordinate space, giving larger cells
        at the surface where density is lower, matching SPIDER's mesh.
    jax: bool
        Use JAX/diffrax solver backend instead of scipy BDF. Default is False.
        When True, the entropy ODE is integrated with diffrax Tsit5 instead of
        scipy solve_ivp (BDF). Requires jax, equinox, and diffrax packages.
    atol_temperature_equivalent: float
        Effective temperature-scale absolute tolerance [K] for Aragog's
        scipy BDF integrator. Aragog's state variable is entropy (J/kg/K),
        but users think in Kelvin, so this is exposed as a temperature
        equivalent that Aragog converts internally via Cp/T. Default is
        0.01 K — tight enough that the solver resolves the ~0.3 K/yr
        cooling rate of a magma ocean.
    core_bc: str
        Core-mantle boundary condition mode. Default is 'quasi_steady'
        (v3 alpha-factor heat-flux partition). Valid values:
          - 'quasi_steady': legacy v3 BC, gives -19% T_core offset vs SPIDER
          - 'energy_balance': v5 Path A SPIDER bit-parity BC with dSdr_cmb as
                          a new state variable (mirrors SPIDER bc.c:76-131)
          - 'bower2018': EXPERIMENTAL tombstone, do not use for production
    """

    dilatation: bool = field(default=False)
    mass_coordinates: bool = field(default=True)
    jax: bool = field(default=False)
    atol_temperature_equivalent: float = field(default=1.0e-6, validator=gt(0))
    """Effective temperature-scale absolute tolerance [K] for Aragog's ODE integrator."""
    core_bc: str = field(default='energy_balance')
    phase_smoothing: str = field(default='tanh')
    """Phase-boundary smoothing for Jgrav and Jmix: 'tanh' (SPIDER parity) or 'cubic_hermite'."""
    solver_method: str = field(default='cvode')
    """ODE solver: 'cvode' (SUNDIALS, SPIDER parity), 'radau' (scipy), 'bdf' (scipy)."""


def valid_interiordummy(instance, attribute, value):
    if instance.module != 'dummy':
        return

    pass  # dummy uses planet.tsurf_init for initial temperature

    tliq = instance.dummy.mantle_tliq
    tsol = instance.dummy.mantle_tsol
    if tliq <= tsol:
        raise ValueError(f'Dummy liquidus ({tliq}K) must be greater than solidus ({tsol}K)')


@define
class InteriorDummy:
    """Parameters for Dummy interior module.

    Attributes
    ----------
    mantle_rho: float
        Mantle mass density [kg m-3].
    mantle_cp: float
        Mantle specific heat capacity [J kg-1 K-1]
    mantle_tliq: float
        Mantle liquidus temperature [K]
    mantle_tsol: float
        Mantle solidus temperature [K]
    heat_internal: float
        Internal heating rate (e.g., radiogenic, tidal) [W kg-1]
    """

    mantle_tliq: float = field(default=2700.0, validator=ge(0))
    mantle_tsol: float = field(default=1700.0, validator=ge(0))
    mantle_rho: float = field(default=4.55e3, validator=gt(0))
    mantle_cp: float = field(default=1792.0, validator=ge(0))
    heat_internal: float = field(default=0.0, validator=ge(0))


@define
class Interior:
    """Magma ocean model selection and parameters.

    Attributes
    ----------
    grain_size: float
        Crystal settling grain size [m].
    flux_guess: float
        Initial heat flux guess [W m-2]. When < 0 (default), computed
        automatically as sigma * T_magma^4. Set to a positive value to
        prescribe a specific initial flux. Set to 0 for zero initial flux.
    radio_tref: float
        Reference age for setting radioactive decay [Gyr].
    radio_K: float
        Concentration (ppmw) of potassium-40 at reference age t=radio_tref.
    radio_U: float
        Concentration (ppmw) of uranium at reference age t=radio_tref.
    radio_Th: float
        Concentration (ppmw) of thorium-232 at reference age t=radio_tref.
    heat_radiogenic: bool
        Include radiogenic heat production?
    heat_tidal: bool
        Include tidal heating?
    rfront_loc: float
        Centre of rheological transition in terms of melt fraction
    rfront_wid: float
        Width of rheological transition in terms of melt fraction

    module: str
        Module for simulating the magma ocean. Choices: 'spider', 'aragog', 'dummy'.
    spider: Spider
        Parameters for running the SPIDER module.
    aragog: Aragog
        Parameters for running the aragog module.
    dummy: Dummy
        Parameters for running the dummy module.
    melting_dir: str
        Melting curve set used by all interior modules (Zalmoxis, Aragog, SPIDER).
        Must correspond to a folder in FWL_DATA/interior_lookup_tables/Melting_curves/
        containing solidus_P-T.dat and liquidus_P-T.dat (T(P) format). SPIDER additionally
        requires pre-computed S(P) files in its lookup directory.
    eos_dir: str
        Equation of state used by SPIDER and Aragog. Must correspond to a
        folder under FWL_DATA/interior_lookup_tables/EOS/dynamic/ containing
        P-T/ (pressure-temperature format, used by Aragog) and P-S/
        (pressure-entropy format, used by SPIDER) subdirectories.
        Zalmoxis derives its EOS paths from struct.zalmoxis config instead.
    """

    module: str = field(default='aragog', validator=in_(('spider', 'aragog', 'dummy')))
    num_levels: int = field(default=80, validator=ge(40))

    # Tier 4 (2026-04-08): rtol was promoted from the Spider subclass
    # to the top level so both SPIDER and Aragog read numerical
    # tolerances from a single source of truth. num_tolerance is kept
    # as a deprecated alias for rtol — old configs still load but emit
    # a DeprecationWarning on the first Interior.__attrs_post_init__
    # call. Remove in a future release.
    #
    # matprop_smooth_width was also briefly promoted in Tier 4, but
    # the 2026-04-09 Aragog Jgrav fix made it a SPIDER-only knob again.
    # It now lives on the Spider subclass as a real field (no longer
    # a deprecation alias).
    rtol: float = field(default=1e-10, validator=gt(0))
    """Relative numerical tolerance for the interior ODE solver.
    SPIDER: -ts_sundials_rtol (used internally via atol_sf scaling).
    Aragog: scipy solve_ivp rtol. Replaces the legacy Spider.tolerance_rel
    and Interior.num_tolerance fields."""

    atol: float = field(default=1e-10, validator=gt(0))
    """Absolute numerical tolerance for the interior ODE solver.
    SPIDER: -ts_sundials_atol (scaled by atol_sf at runtime). Aragog
    uses [interior_energetics.aragog].atol_temperature_equivalent
    instead because its state variable is entropy, not temperature, and
    a direct entropy-scale atol would be unintuitive to tune."""

    # Deprecated alias for rtol. Kept for backwards compatibility with
    # existing configs. Emits DeprecationWarning when set to any non-
    # default value. Will be removed in a future release.
    num_tolerance: float = field(default=1e-10, validator=gt(0))

    trans_conduction: bool = field(default=True)
    trans_convection: bool = field(default=True)
    trans_grav_sep: bool = field(default=True)
    trans_mixing: bool = field(default=True)
    heat_radiogenic: bool = field(default=True)
    heat_tidal: bool = field(default=False)

    spider: Spider = field(factory=Spider, validator=valid_spider)
    aragog: Aragog = field(factory=Aragog, validator=valid_aragog)
    dummy: InteriorDummy = field(factory=InteriorDummy, validator=valid_interiordummy)

    mixing_length: str = field(default='nearest', validator=in_(('nearest', 'constant')))
    grain_size: float = field(default=0.1, validator=gt(0))
    flux_guess: float = field(default=-1)
    tmagma_atol: float = field(default=20.0, validator=ge(0))
    tmagma_rtol: float = field(default=0.02, validator=ge(0))

    radio_tref: float = field(default=4.567, validator=ge(0))
    radio_Al: float = field(default=0.0, validator=ge(0))
    radio_Fe: float = field(default=0.0, validator=ge(0))
    radio_K: float = field(default=310.0, validator=ge(0))
    radio_U: float = field(default=0.031, validator=ge(0))
    radio_Th: float = field(default=0.124, validator=ge(0))

    rfront_loc: float = field(default=0.5, validator=(gt(0), lt(1)))
    rfront_wid: float = field(default=0.2, validator=(gt(0), lt(1)))

    # Phase-dependent eddy diffusivity floor [m^2/s]. Default 0 = standard MLT.
    # When > 0, applies max(kh_MLT, floor * f(phi)) where f transitions from
    # 1 (liquid) to 0 (solid) at the rheological transition. Passed to both
    # SPIDER (-kappah_floor) and Aragog (kappah_floor in energy config).
    kappah_floor: float = field(default=10.0, validator=ge(0))

    # Ultra-thin thermal boundary layer parameterization (Bower et al. 2018, Eq. 18).
    # Corrects the surface temperature for the unresolved boundary layer:
    #   T_interior = T_surf + param_utbl_const * T_surf^3
    # Applies to both SPIDER and Aragog. Default off.
    param_utbl: bool = field(default=False)
    param_utbl_const: float = field(default=1e-7, validator=gt(0))

    # Surface boundary condition mode for SPIDER/Aragog.
    #
    # - 'flux' (default): prescribed heat flux from hf_row['F_atm']. SPIDER
    #   uses -SURFACE_BC 4, Aragog uses outer_boundary_condition=4. The Python
    #   atmosphere module (dummy, AGNI, JANUS) is responsible for computing
    #   F_atm, which the interior consumes unchanged for the full duration of
    #   the coupling step.
    #
    # - 'grey_body': native grey-body BC computed inside the interior solver
    #   per CVode substep from the current top-cell T. SPIDER uses -SURFACE_BC 1
    #   -emissivity0 1.0, Aragog uses outer_boundary_condition=1 with
    #   emissivity=1. Both compute F = sigma * (T_surf^4 - T_eqm^4) using the
    #   T_eqm value in hf_row['T_eqm']. This is the formulation used by
    #   SPIDER-Aragog parity tests: both solvers follow the identical physical
    #   law so their cooling trajectories can be compared directly. The
    #   Python-side atmosphere module still runs to populate diagnostic
    #   helpfile fields (F_olr, F_sct, R_obs, ...), but its F_atm output is
    #   not used by the interior.
    surface_bc_mode: str = field(
        default='flux',
        validator=in_(('flux', 'grey_body')),
    )

    # -----------------------------------------------------------------
    # Tier 3 parity fields (promoted from hardcoded values in
    # aragog.py::setup_solver and spider.py::build_call_sequence, 2026-04-08).
    # Defaults match the SPIDER-side values so existing SPIDER runs are
    # bit-exact unchanged. Aragog users get the SPIDER-matching values by
    # default for the first time — notably `solid_log10visc = 22.0` fixes
    # the 10x discrepancy with SPIDER that had been latent in Aragog
    # (it used hardcoded `1e21` = log10(21), SPIDER uses `-solid_log10visc 22.0`).
    # -----------------------------------------------------------------

    # Adams-Williamson mantle hydrostatic EOS parameters.
    adams_williamson_rhos: float = field(
        default=4078.95095544,
        validator=gt(0),
    )
    """Adams-Williamson surface density [kg/m^3]. Matches SPIDER
    -adams_williamson_rhos and Aragog _MeshParameters.surface_density.
    Previously hardcoded: SPIDER 4078.95095544, Aragog 4090 (0.27% off)."""

    adams_williamson_beta: float = field(
        default=1.1115348931000002e-07,
        validator=gt(0),
    )
    """Adams-Williamson density gradient [1/m]. Matches SPIDER
    -adams_williamson_beta. Aragog derives its own via bulk modulus so
    this value is SPIDER-only today."""

    adiabatic_bulk_modulus: float = field(
        default=260e9,
        validator=gt(0),
    )
    """Adiabatic bulk modulus [Pa] used by Aragog's Adams-Williamson EOS
    (_MeshParameters.adiabatic_bulk_modulus). SPIDER derives its own."""

    # Phase viscosities (log10 Pa s).
    melt_log10visc: float = field(default=2.0)
    """log10 viscosity of molten silicate [Pa s]. Matches SPIDER
    -melt_log10visc (2.0 = 1e2 Pa s)."""

    solid_log10visc: float = field(default=22.0)
    """log10 viscosity of solid silicate [Pa s]. Matches SPIDER
    -solid_log10visc (22.0 = 1e22 Pa s). Aragog previously hardcoded
    1e21 (log10=21), a 10x undervalue that made Aragog's solid-phase
    rheology an order of magnitude softer than SPIDER's. Fixed here."""

    # Phase thermal conductivity [W/m/K].
    melt_cond: float = field(default=4.0, validator=gt(0))
    """Thermal conductivity of molten silicate [W/m/K]. Matches SPIDER
    -melt_cond."""

    solid_cond: float = field(default=4.0, validator=gt(0))
    """Thermal conductivity of solid silicate [W/m/K]. Matches SPIDER
    -solid_cond."""

    # Eddy diffusivity scaling (dimensionless multiplier on MLT-derived kappa).
    eddy_diffusivity_thermal: float = field(default=1.0)
    """Multiplier on the internally-computed thermal eddy diffusivity.
    SPIDER: -eddy_diffusivity_thermal (1.0 default)."""

    eddy_diffusivity_chemical: float = field(default=1.0)
    """Multiplier on the internally-computed chemical eddy diffusivity.
    SPIDER: -eddy_diffusivity_chemical (1.0 default)."""

    # Constant-properties mode (SPIDER -use_const_properties parity).
    # When True, both SPIDER and Aragog bypass EOS tables and use
    # analytical T(S) = T_ref * exp((S - S_ref) / Cp) with constant
    # rho, Cp, alpha, k, visc. phi=1 always (no phase transitions).
    # For controlled parity comparisons with dummy structure + atmosphere.
    const_properties: bool = field(default=False)
    const_rho: float = field(default=4000.0, validator=gt(0))
    """Constant density [kg/m3]."""
    const_Cp: float = field(default=1000.0, validator=gt(0))
    """Constant heat capacity [J/kg/K]."""
    const_alpha: float = field(default=1e-5, validator=gt(0))
    """Constant thermal expansivity [1/K]."""
    const_cond: float = field(default=4.0, validator=gt(0))
    """Constant thermal conductivity [W/m/K]."""
    const_log10visc: float = field(default=2.0)
    """Constant log10 dynamic viscosity [Pa.s]."""
    const_T_ref: float = field(default=3500.0, validator=gt(0))
    """Reference temperature for T(S) = T_ref * exp((S-S_ref)/Cp) [K]."""
    const_S_ref: float = field(default=3000.0)
    """Reference entropy for T(S) [J/kg/K]. No positivity constraint
    since entropy reference states can be zero or negative."""

    # Phase transition thermodynamics.
    latent_heat_of_fusion: float = field(default=4e6, validator=gt(0))
    """Latent heat of fusion of silicate [J/kg]. Aragog uses this as a
    scalar in _PhaseMixedParameters. SPIDER derives it per-(P,S) from
    dS * T_fus via the EOS tables; the SPIDER derivation is more
    physically correct but this scalar matches Aragog's historical
    default (4e6) and is good to ~10% at Earth-mantle conditions. TODO:
    switch Aragog to SPIDER's derivation once the EntropyEOS exposes a
    dS_fus(P) method."""

    phase_transition_width: float = field(
        default=0.1,
        validator=(gt(0), lt(1)),
    )
    """Width [fraction] of the mushy-zone transition in Aragog's
    _PhaseMixedParameters. Sets the width of the phase boundary in
    Aragog's mixed-phase blending (viscosity, thermal conductivity
    etc.). Distinct from ``[interior_energetics.spider].matprop_smooth_width``
    which is SPIDER's analogous knob for its own solver."""

    # Core thermal model (Bower+2018 Table 2).
    core_tfac_avg: float = field(default=1.147, validator=gt(0))
    """Core T_avg / T_cmb ratio from adiabatic gradient (Bower+2018
    Table 2). Used by Aragog's _BoundaryConditionsParameters.tfac_core_avg.
    SPIDER derives its own internally."""

    # dSdr_cmb damping for the marginally-stable phi=0 CMB closure
    # (Aragog energy_balance BC only). Adds
    #   -dsdr_cmb_damping_gamma * (dSdr_cmb - dsdr_cmb_damping_target)
    # to the d(dSdr_cmb)/dt RHS. gamma=0 disables (default).
    # See memory/tcore_phi0_instrumented_diagnostic.md.
    dsdr_cmb_damping_gamma: float = field(default=0.0, validator=ge(0))
    """Damping rate [1/yr] for dSdr_cmb in Aragog's energy_balance BC.
    0 = no damping. From the diag3 eigenvalue analysis the loop
    eigenvalue is +2.3e-5/yr, so gamma > 5e-5/yr is needed to push it
    into the stable half-plane. Larger gamma stabilises faster but
    biases the steady-state dSdr toward the target."""

    dsdr_cmb_damping_target: float = field(default=0.0)
    """Target dSdr_cmb [J/(kg*K*m)] toward which damping pulls the
    state. 0 = pull toward zero (Aragog's current working point).
    SPIDER's quasi-equilibrium in the CHILI Earth solid regime is
    ~-6.3e-5 J/(kg*K*m), so setting this matches SPIDER's working
    point if the equilibrium offset is the underlying issue."""

    # Diagnostic flag for T_core investigations.
    write_flux_diagnostics: bool = field(default=False)
    """When True, Aragog's NetCDF output includes per-component flux
    decomposition (Jcond_b, Jconv_b, Jgrav_b, Jmix_b) and basic-node
    state variables (dSdr_b, eddy_diff_b, phi_basic_b, T/cp/rho_basic_b).
    Adds ~10 fields per snapshot; default False keeps output compact.
    Used for the T_core phi=0 CMB-closure instability investigation
    (2026-04-14). SPIDER path ignores this flag (uses SPIDER's own JSON
    output which already includes Jcond_b, Jconv_b, Jgrav_b, Jmix_b)."""

    def __attrs_post_init__(self):
        """Resolve Tier 4 deprecation aliases.

        Two legacy fields are accepted as backwards-compat aliases
        for ``interior_energetics.rtol``:
        ``num_tolerance`` (old top-level name) and
        ``[interior_energetics.spider].tolerance_rel`` (old
        per-solver name).

        The resolution rules are:

        - If only the alias is set, its value is copied to ``rtol``
          and a ``DeprecationWarning`` is emitted.
        - If both are set to the same value, the alias is silently
          ignored (the user is in a clean state).
        - If both are set to DISTINCT non-default values, ``ValueError``
          is raised because we cannot guess which one the user meant.

        All aliases will be removed in a future release.

        Note: ``matprop_smooth_width`` was briefly a top-level field
        under Tier 4 but reverted to SPIDER-only on 2026-04-09 after
        the Aragog Jgrav smoothing was made parameter-free. It lives
        on ``Spider`` now, no alias logic needed.
        """
        _default_rtol = 1e-10

        # --- num_tolerance (top-level) -> rtol ---
        num_tol_set = self.num_tolerance != _default_rtol
        rtol_set = self.rtol != _default_rtol
        if num_tol_set and rtol_set and self.num_tolerance != self.rtol:
            raise ValueError(
                'interior_energetics.num_tolerance and .rtol are both '
                'set to distinct non-default values '
                f'(num_tolerance={self.num_tolerance}, rtol={self.rtol}). '
                'num_tolerance is deprecated; set rtol only.'
            )
        if num_tol_set and not rtol_set:
            warnings.warn(
                'interior_energetics.num_tolerance is deprecated; use '
                'interior_energetics.rtol instead. Value copied to rtol '
                'for backwards compatibility. This alias will be removed '
                'in a future release.',
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(self, 'rtol', float(self.num_tolerance))
            rtol_set = True

        # --- spider.tolerance_rel -> rtol ---
        spider_tol = float(getattr(self.spider, 'tolerance_rel', -1.0))
        if spider_tol > 0:
            if rtol_set and spider_tol != self.rtol:
                raise ValueError(
                    'interior_energetics.spider.tolerance_rel is a '
                    'deprecated alias for interior_energetics.rtol, but '
                    f'both are set to distinct values ({spider_tol} vs '
                    f'{self.rtol}). Remove the [interior_energetics.spider] '
                    'section and set rtol at the top level.'
                )
            warnings.warn(
                'interior_energetics.spider.tolerance_rel is deprecated; '
                'use interior_energetics.rtol at the top level instead. '
                'Value copied to rtol for backwards compatibility. This '
                'alias will be removed in a future release.',
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(self, 'rtol', spider_tol)
