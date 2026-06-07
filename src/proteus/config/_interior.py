from __future__ import annotations

import warnings

from attrs import define, field
from attrs.validators import ge, gt, in_, lt

# Default relative tolerance for the interior ODE solver. ``rtol`` and its
# deprecated alias ``num_tolerance`` default to a sentinel so that "left at
# the default" can be told apart from "explicitly set to the default value";
# both resolve to this in Interior.__attrs_post_init__ when unset.
_DEFAULT_RTOL = 1e-10
_TOL_UNSET = -1.0


def _gt0_or_unset(instance, attribute, value):
    """Allow the unset sentinel; otherwise require a positive value."""
    if value == _TOL_UNSET:
        return
    if value <= 0:
        raise ValueError(f'`{attribute.name}` must be greater than 0, got {value}')


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


def valid_interiorboundary(instance, attribute, value):
    """Validate Boundary backend's solidus/liquidus ordering.

    Only fires when ``module == 'boundary'``; otherwise the subclass is
    constructed with defaults and never exercised.
    """
    if instance.module != 'boundary':
        return

    tsol = instance.boundary.T_solidus
    tliq = instance.boundary.T_liquidus
    if tliq <= tsol:
        raise ValueError(f'Boundary liquidus ({tliq}K) must be greater than solidus ({tsol}K)')


@define
class Spider:
    """SPIDER-specific parameters.

    ``solver_type`` is the SUNDIALS integrator choice.
    ``tolerance_rel`` is a deprecated alias for the top-level
    ``[interior_energetics].rtol``; set ``rtol`` instead.
    ``matprop_smooth_width`` sets the smoothing width for
    material-property blending across the solidus/liquidus, read by
    both SPIDER and Aragog.

    Attributes
    ----------
    solver_type: str
        SUNDIALS integrator choice. Choices: 'adams', 'bdf'.
    tolerance_rel: float
        Deprecated alias for ``Interior.rtol``. Set
        ``interior_energetics.rtol`` at the top level instead.
    matprop_smooth_width: float
        Melt-fraction window width for smoothing material properties
        across the solidus/liquidus. Passed to SPIDER as
        ``-matprop_smooth_width`` and to Aragog via
        ``_PhaseMixedParameters``.
    """

    solver_type: str = field(default='bdf', validator=in_(('adams', 'bdf')))

    # Sentinel -1 means "not set". A positive value is copied to the
    # top-level rtol in Interior.__attrs_post_init__ and emits a
    # DeprecationWarning. The validator allows -1 as a pass-through.
    tolerance_rel: float = field(default=-1.0)

    # Material-property smoothing width for the phase-boundary blend.
    # Passed to SPIDER as ``-matprop_smooth_width`` and to Aragog's
    # ``EntropyPhaseEvaluator`` via ``_PhaseMixedParameters``. Controls
    # the tanh transition between two-phase (Lever Rule) and single-
    # phase material properties near the solidus and liquidus.
    matprop_smooth_width: float = field(default=1.0e-2, validator=(gt(0), lt(1)))

    # Absolute mass tolerance [kg] for the secant solver in
    # determine_interior_radius. Tightens with mass scale; the default
    # 100 kg is below SPIDER's own internal mass-balance error on a 1
    # M_E planet.
    tolerance_struct: float = field(default=1e2, validator=gt(0))

    # When True (default), the SPIDER subprocess writes stdout/stderr
    # to ``<output>/spider_recent.log``. Set False to discard the
    # subprocess output entirely (sp.DEVNULL); useful for batch runs
    # where the log files accumulate disk pressure.
    log_output: bool = field(default=True)


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
    mass_coordinates: bool
        Whether to use mass coordinates in the model. Default is True.
        Uses uniform spacing in mass coordinate space, giving larger cells
        at the surface where density is lower, matching SPIDER's mesh.
    backend: str
        ODE backend selector. Default 'jax'.
          - 'jax'   : CVODE with JAX-derived RHS and JAX analytic Jacobian
                     (`jax.jacrev`). Recommended for production.
          - 'numpy' : CVODE with numpy RHS and CVODE finite-difference
                     Jacobian. Available for development and SPIDER-parity
                     comparisons; less robust than 'jax' at production
                     resolution because the FD-Jacobian noise can trip
                     Aragog's T_core-jump retry guard at tight tolerances.
        The diffrax direct-JAX integration path is research-only and
        gated on a code-level flag in `proteus.interior_energetics.aragog`.
    atol_temperature_equivalent: float
        Effective temperature-scale absolute tolerance [K] for Aragog's
        CVODE integrator. Aragog's state variable is entropy (J/kg/K),
        but users think in Kelvin, so this is exposed as a temperature
        equivalent that Aragog converts internally via Cp/T. Default is
        1e-8 K, matching SPIDER's atol=rtol=1e-8 setting; this tight
        tolerance eliminates the CVODE marginal-stability bifurcation
        at the first dt jump after equilibration.
    core_bc: str
        Core-mantle boundary condition mode. Default 'energy_balance'.
        Valid values:
          - 'quasi_steady': alpha-factor heat-flux partition; gives
                            about -19% T_core offset vs SPIDER.
          - 'energy_balance': SPIDER bit-parity BC with dSdr_cmb as a
                              new state variable (mirrors SPIDER
                              bc.c:76-131).
          - 'gradient': gradient-based state with two boundary entropies
                        as state variables.
          - 'bower2018': experimental, do not use for production.
    """

    mass_coordinates: bool = field(default=True)
    backend: str = field(default='jax', validator=in_(('numpy', 'jax')))
    atol_temperature_equivalent: float = field(default=1.0e-8, validator=gt(0))
    """Effective temperature-scale absolute tolerance [K] for Aragog's ODE integrator.
    Default 1e-8 matches SPIDER's atol=rtol=1e-8 setting; this tight tolerance
    avoids a marginal-stability bifurcation at the first dt jump after
    equilibration."""
    core_bc: str = field(
        default='energy_balance',
        validator=in_(('quasi_steady', 'energy_balance', 'gradient', 'bower2018')),
    )
    phase_smoothing: str = field(
        default='tanh',
        validator=in_(('tanh', 'cubic_hermite')),
    )
    """Phase-boundary smoothing for Jgrav and Jmix: 'tanh' (SPIDER parity) or 'cubic_hermite'."""
    solver_method: str = field(
        default='cvode',
        validator=in_(('cvode', 'radau', 'bdf')),
    )
    """ODE solver: 'cvode' (SUNDIALS, SPIDER parity), 'radau' (scipy), 'bdf' (scipy)."""
    scalar_gravity_override: bool = field(default=False)
    """Scalar-gravity comparison knob. When True, the external mesh file that
    Zalmoxis writes has its gravity column overwritten with a uniform scalar
    (the surface value from ``hf_row['gravity']``) before Aragog reads it, so
    Aragog's per-node gravity path interpolates to that scalar everywhere.
    False by default; set True only when running a paired scalar-gravity
    comparison."""
    phi_step_cap: float = field(default=0.0, validator=ge(0.0))
    """Per-call ΔΦ cap. When > 0 and at least one staggered cell sits in or
    near the mushy band at solve() entry, Aragog clamps the integration
    end_time so the projected per-cell |ΔΦ| over the requested window stays
    within this cap. The estimate uses |dΦ/dt| at t=start_time scaled by a 0.5
    safety factor, and the PROTEUS outer loop sees the truncated achieved time
    via ``sol.t[-1]``. Default 0.0 (disabled)."""

    tolerance_struct: float = field(default=1e2, validator=gt(0))
    """Absolute mass tolerance [kg] for the secant solver in
    determine_interior_radius. Default 100 kg; pairs with Spider's
    matching field so both backends drive the same outer-loop convergence
    criterion."""


def valid_interiordummy(instance, attribute, value):
    if instance.module != 'dummy':
        return

    pass  # dummy uses planet.tsurf_init for initial temperature

    tliq = instance.dummy.mantle_tliq
    tsol = instance.dummy.mantle_tsol
    if tliq <= tsol:
        raise ValueError(f'Dummy liquidus ({tliq}K) must be greater than solidus ({tsol}K)')


@define
class InteriorBoundary:
    """Parameters for Boundary interior module. Default values taken from Schaefer et al. 2016 (https://iopscience.iop.org/article/10.3847/0004-637X/829/2/63/pdf).

    Attributes
    ----------
    rtol: float
        ODE solver relative tolerance.
    atol: float
        ODE solver absolute tolerance.
    T_p_0: float
        Initial potential temperature [K] for boundary solver if zalmoxis module is not used.
    T_solidus: float
        Mantle solidus temperature [K].
    T_liquidus: float
        Mantle liquidus temperature [K].
    Tsurf_event_change: float
        Maximum change in surface temperature allowed during a single interior iteration before triggering an event [K].
    critical_rayleigh_number: float
        Critical Rayleigh number for onset of convection [-].
    heat_fusion_silicate: float
        Latent heat of fusion for silicates [J/kg].
    nusselt_exponent: float
        Nusselt-Rayleigh scaling exponent [-].
    silicate_heat_capacity: float
        Silicate heat capacity [J/kg/K].
    atm_heat_capacity: float
        Used as fallback for atmosphere heat capacity when layer-specific value is not available [J/kg/K].
    silicate_density: float
        Silicate density [kg/m^3]. Default taken from Fei et. al. 2021 (https://ui.adsabs.harvard.edu/abs/2021NatCo..12..876F).
    thermal_conductivity: float
        Thermal conductivity [W/m/K].
    thermal_diffusivity: float
        Thermal diffusivity [m^2/s].
    thermal_expansivity: float
        Thermal expansivity [1/K].
    viscosity_model: int
        Viscosity parameterisation model. Choices: 1 (constant), 2 (aggregate smooth transition), 3 (Arrhenius temperature-dependent).
    eta_constant: float
        Constant viscosity value [Pa s] for model 1.
    transition_width: float
        Width of viscosity transition in melt fraction space [-] for aggregate model.
    eta_solid_const: float
        Constant solid viscosity for aggregate formulation [Pa s].
    eta_melt_const: float
        Constant melt viscosity for aggregate formulation [Pa s].
    dynamic_viscosity: float
        Reference dynamic viscosity [Pa s] for Arrhenius solid mantle model.
    activation_energy: float
        Activation energy [J/mol] for Arrhenius solid mantle model.
    viscosity_prefactor: float
        Viscosity prefactor [Pa s] for Vogel-Fulcher-Tammann magma ocean model.
    viscosity_activation_temp: float
        Activation temperature [K] for Vogel-Fulcher-Tammann magma ocean model.
    logging: bool
        Whether to create diagnostic CSV data files from boundary interior module.
    """

    rtol: float = field(default=1e-6, validator=gt(0))
    atol: float = field(default=1e-9, validator=gt(0))

    T_p_0: float = field(default=3500.0, validator=ge(0))

    T_solidus: float = field(default=1420.0, validator=ge(0))
    T_liquidus: float = field(default=2020.0, validator=gt(0))

    Tsurf_event_change: float = field(default=20.0, validator=gt(0))  # K

    critical_rayleigh_number: float = field(default=1.1e3, validator=gt(0))  # -
    heat_fusion_silicate: float = field(default=4.0e5, validator=gt(0))  # J/kg
    nusselt_exponent: float = field(default=0.33, validator=gt(0))  # -
    silicate_heat_capacity: float = field(default=1.2e3, validator=gt(0))  # J/kg/K
    atm_heat_capacity: float = field(default=1.7e4, validator=gt(0))  # J/kg/K
    silicate_density: float = field(default=4103.0, validator=gt(0))  # kg/m^3
    thermal_conductivity: float = field(default=4.2, validator=gt(0))  # W/m/K
    thermal_diffusivity: float = field(default=1e-6, validator=gt(0))  # m^2/s
    thermal_expansivity: float = field(default=2e-5, validator=gt(0))  # 1/K

    # Viscosity parameterisation
    viscosity_model: int = field(
        default=2, validator=in_((1, 2, 3))
    )  # 1=constant, 2=aggregate, 3=Arrhenius
    eta_constant: float = field(default=1e2, validator=gt(0))  # Pa s, for model 1

    # Aggregate viscosity parameters
    transition_width: float = field(default=0.2, validator=(gt(0), lt(1)))  # -
    eta_solid_const: float = field(default=1e22, validator=gt(0))  # Pa s
    eta_melt_const: float = field(default=1e2, validator=gt(0))  # Pa s

    # Arrhenius solid mantle parameters
    dynamic_viscosity: float = field(default=3.8e9, validator=gt(0))  # Pa s
    activation_energy: float = field(default=3.5e5, validator=gt(0))  # J/mol
    creep_parameter: float = field(default=26.0, validator=gt(0))

    # Arrhenius magma ocean parameters (Vogel-Fulcher-Tammann)
    viscosity_prefactor: float = field(default=2.4e-4, validator=gt(0))  # Pa s
    viscosity_activation_temp: float = field(default=4600, validator=gt(0))  # K

    logging: bool = field(default=False)


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
        Fixed internal heating rate (e.g. radiogenic) [W kg-1].
        Tidal heating is handled separately via `heat_tidal` and is
        added on top of this value.
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

    Notes
    -----
    The ``melting_dir`` and ``eos_dir`` fields live on the parent
    ``[interior_struct]`` section (class ``Struct`` in
    ``config/_struct.py``), not here. They are shared across SPIDER,
    Aragog, and Zalmoxis and so belong with the structure config.
    """

    module: str = field(
        default='aragog', validator=in_(('spider', 'aragog', 'dummy', 'boundary'))
    )
    num_levels: int = field(default=80, validator=ge(40))

    # Unified ODE tolerance: both SPIDER and Aragog read from here.
    # num_tolerance is a deprecated alias (emits DeprecationWarning).
    # matprop_smooth_width lives on the Spider subclass but is read by both solvers.
    #
    # rtol and num_tolerance default to a sentinel so that an explicit
    # value equal to the resolved default is not mistaken for "unset".
    # Both resolve to _DEFAULT_RTOL in __attrs_post_init__.
    rtol: float = field(default=_TOL_UNSET, validator=_gt0_or_unset)
    """Relative numerical tolerance for the interior ODE solver.
    SPIDER: -ts_sundials_rtol (used internally via atol_sf scaling).
    Aragog: scipy solve_ivp rtol. The deprecated aliases num_tolerance and
    [interior_energetics.spider].tolerance_rel copy into this field.
    Resolves to 1e-10 when left unset."""

    atol: float = field(default=1e-10, validator=gt(0))
    """Absolute numerical tolerance for the interior ODE solver.
    SPIDER: -ts_sundials_atol (scaled by atol_sf at runtime). Aragog
    uses [interior_energetics.aragog].atol_temperature_equivalent
    instead because its state variable is entropy, not temperature, and
    a direct entropy-scale atol would be unintuitive to tune."""

    # Deprecated alias for rtol. Emits DeprecationWarning when set to a
    # positive value; will be removed in a future release.
    num_tolerance: float = field(default=_TOL_UNSET, validator=_gt0_or_unset)

    trans_conduction: bool = field(default=True)
    trans_convection: bool = field(default=True)
    trans_grav_sep: bool = field(default=True)
    trans_mixing: bool = field(default=True)
    heat_radiogenic: bool = field(default=True)
    heat_tidal: bool = field(default=False)

    spider: Spider = field(factory=Spider, validator=valid_spider)
    aragog: Aragog = field(factory=Aragog, validator=valid_aragog)
    dummy: InteriorDummy = field(factory=InteriorDummy, validator=valid_interiordummy)
    boundary: InteriorBoundary = field(
        factory=InteriorBoundary, validator=valid_interiorboundary
    )

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
    # EOS and rheology parameters exposed to config. Defaults match
    # SPIDER values for cross-solver consistency.
    # -----------------------------------------------------------------

    # Adams-Williamson mantle hydrostatic EOS parameters.
    adams_williamson_rhos: float = field(
        default=4078.95095544,
        validator=gt(0),
    )
    """Adams-Williamson surface density [kg/m^3]. Matches SPIDER
    -adams_williamson_rhos and Aragog _MeshParameters.surface_density."""

    adams_williamson_beta: float = field(
        default=1.1115348931000002e-07,
        validator=gt(0),
    )
    """Adams-Williamson density gradient [1/m]. Matches SPIDER
    -adams_williamson_beta. Aragog derives its own via bulk modulus, so
    this value applies to SPIDER only."""

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
    -solid_log10visc (22.0 = 1e22 Pa s). Shared by SPIDER and Aragog so
    both apply the same solid-phase rheology; a mis-set value diverges
    both solvers' solid-phase rheology by the same factor."""

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
    physically correct but this scalar is good to ~10% at Earth-mantle
    conditions. TODO: switch Aragog to SPIDER's derivation once the
    EntropyEOS exposes a dS_fus(P) method."""

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

    # Diagnostic flag for T_core investigations.
    write_flux_diagnostics: bool = field(default=False)
    """When True, Aragog's NetCDF output includes per-component flux
    decomposition (Jcond_b, Jconv_b, Jgrav_b, Jmix_b) and basic-node
    state variables (dSdr_b, eddy_diff_b, phi_basic_b, T/cp/rho_basic_b).
    Adds ~10 fields per snapshot; default False keeps output compact.
    Useful for diagnosing T_core and CMB-closure behaviour near phi=0.
    SPIDER path ignores this flag (uses SPIDER's own JSON output which
    already includes Jcond_b, Jconv_b, Jgrav_b, Jmix_b)."""

    def __attrs_post_init__(self):
        """Resolve deprecated tolerance aliases.

        Two deprecated fields are accepted as aliases for
        ``interior_energetics.rtol``: ``num_tolerance`` (top-level)
        and ``[interior_energetics.spider].tolerance_rel`` (per-solver).

        The resolution rules are:

        - If only the alias is set, its value is copied to ``rtol``
          and a ``DeprecationWarning`` is emitted.
        - If both are set to the same value, the alias is silently
          ignored (the user is in a clean state).
        - If both are set to DISTINCT values, ``ValueError`` is raised
          because we cannot guess which one the user meant.

        "Set" is detected via a sentinel default, so an explicit value
        equal to the resolved default still counts as set and is not
        overridden by an alias.

        All aliases will be removed in a future release.
        """
        # Whether each field was supplied is tracked via the sentinel
        # default, not by value-comparison against the resolved default.
        num_tol_set = self.num_tolerance != _TOL_UNSET
        rtol_set = self.rtol != _TOL_UNSET

        # --- num_tolerance (top-level) -> rtol ---
        if num_tol_set and rtol_set and self.num_tolerance != self.rtol:
            raise ValueError(
                'interior_energetics.num_tolerance and .rtol are both '
                'set to distinct values '
                f'(num_tolerance={self.num_tolerance}, rtol={self.rtol}). '
                'num_tolerance is deprecated; set rtol only.'
            )
        if num_tol_set and not rtol_set:
            warnings.warn(
                'interior_energetics.num_tolerance is deprecated; use '
                'interior_energetics.rtol instead. The value is copied to '
                'rtol. This alias will be removed in a future release.',
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(self, 'rtol', float(self.num_tolerance))
            rtol_set = True

        # --- spider.tolerance_rel -> rtol ---
        spider_tol = float(getattr(self.spider, 'tolerance_rel', _TOL_UNSET))
        if spider_tol > 0:
            if rtol_set and spider_tol != self.rtol:
                raise ValueError(
                    'interior_energetics.spider.tolerance_rel is a '
                    'deprecated alias for interior_energetics.rtol, but '
                    f'both are set to distinct values ({spider_tol} vs '
                    f'{self.rtol}). Remove the [interior_energetics.spider] '
                    'section and set rtol at the top level.'
                )
            # Only warn and copy when the alias actually changes rtol;
            # an alias equal to an explicit rtol is a silent no-op.
            if not (rtol_set and spider_tol == self.rtol):
                warnings.warn(
                    'interior_energetics.spider.tolerance_rel is deprecated; '
                    'use interior_energetics.rtol at the top level instead. '
                    'The value is copied to rtol. This alias will be removed '
                    'in a future release.',
                    DeprecationWarning,
                    stacklevel=2,
                )
            object.__setattr__(self, 'rtol', spider_tol)
            rtol_set = True

        # Resolve the sentinel to the real default if nothing set rtol.
        if not rtol_set:
            object.__setattr__(self, 'rtol', _DEFAULT_RTOL)
