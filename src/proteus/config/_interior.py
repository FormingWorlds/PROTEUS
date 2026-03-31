from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, lt


def valid_spider(instance, attribute, value):
    if instance.module != 'spider':
        return

    ini_entropy = instance.spider.ini_entropy
    if (not ini_entropy) or (ini_entropy <= 200.0):
        raise ValueError('`interior.spider.ini_entropy` must be >200')

    # at least one energy term enabled
    if not (
        instance.trans_conduction
        or instance.trans_convection
        or instance.trans_mixing
        or instance.trans_grav_sep
    ):
        raise ValueError('Must enable at least one energy transport term in SPIDER')


def valid_path(instance, attribute, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{attribute.name}' must be a non-empty string")


@define
class Spider:
    """Parameters for SPIDER module.

    Attributes
    ----------
    mixing_length: int
        Parameterisation used to determine convective mixing length.
    tolerance_rel: float
        Relative solver tolerance (SUNDIALS-specific).
    tsurf_atol: float
        Absolute tolerance on change in T_mantle during a single interior iteration.
    tsurf_rtol: float
        Relative tolerance on change in T_mantle during a single interior iteration.
    ini_entropy: float
        Initial specific surface entropy [J K-1 kg-1].
    ini_dsdr: float
        Initial interior specific entropy gradient [J K-1 kg-1 m-1].
    solver_type: str
        Numerical integrator. Choices: 'adams', 'bdf'.
    matprop_smooth_width: float
        Window width, in melt-fraction, for smoothing properties across liquidus and solidus
    """

    ini_entropy = field(default=None)
    ini_dsdr: float = field(default=-4.698e-6, validator=lt(0))
    mixing_length: int = field(default=2, validator=in_((1, 2)))
    tolerance_rel: float = field(default=1e-10, validator=gt(0))
    solver_type: str = field(default='bdf', validator=in_(('adams', 'bdf')))
    tsurf_atol: float = field(default=10.0, validator=gt(0))
    tsurf_rtol: float = field(default=0.01, validator=gt(0))
    matprop_smooth_width: float = field(default=1e-2, validator=(gt(0), lt(1)))


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
    """Parameters for Aragog module.

    Attributes
    ----------
    num_levels: int
        Number of Aragog grid levels (basic mesh).
    initial_condition: int
        How to define the intial temperature profile (1: linear, 2: user defined, 3: adiabat).
        Default is 3 (adiabat), which integrates dT/dP|_S from the surface downward.
    tolerance: float
        Solver tolerance.
    basal_temperature: float
        Temperature at the base of the mantle (if using a linear temperature profile to start)
    init_file: str
        File containing the initial temperature file for aragog
    inner_boundary_condition: int
        Type of inner boundary condition. Choices:  1 (core cooling), 2 (prescribed heat flux), 3 (prescribed temperature).
    inner_boundary_value: float
        Value of the inner boundary condition, either temperature or heat flux, depending on the chosen condition.
    trans_conduction: bool
        Whether to include conductive heat flux in the model. Default is True.
    trans_convection: bool
        Whether to include convective heat flux in the model. Default is True.
    trans_grav_sep: bool
        Whether to include gravitational separation flux in the model. Default is True (matches SPIDER).
    trans_mixing: bool
        Whether to include mixing flux in the model. Default is True (matches SPIDER).
    dilatation: bool
        Whether to include dilatation source term in the model. Default is False.
    mass_coordinates: bool
        Whether to use mass coordinates in the model. Default is True.
        Uses uniform spacing in mass coordinate space, giving larger cells
        at the surface where density is lower, matching SPIDER's mesh.
    tsurf_poststep_change: float
        Maximum change in surface temperature allowed during a single interior iteration [K].
    event_triggering: bool
        Whether to include event triggering in the solver. Default is True.
    bulk_modulus: float
        Adiabatic bulk modulus AW-EOS parameter [Pa].
    param_utbl: bool
        Whether to parameterize the ultra-thin thermal boundary layer at the surface.
        Reduces the effective radiating temperature to account for the unresolved
        boundary layer (Bower et al. 2018, Eq. 18). Default is True (matches SPIDER).
    param_utbl_const: float
        UTBL scaling constant [K^-2]. Default is 1e-7 (matches SPIDER).
    """

    basal_temperature: float = field(default=7000)
    init_file: str = field(default=None)
    initial_condition: int = field(
        default=3,
        validator=in_(
            (
                1,
                2,
                3,
            )
        ),
    )
    inner_boundary_condition: int = field(default=1, validator=ge(0))
    inner_boundary_value: float = field(default=4000, validator=ge(0))
    outer_boundary_condition: int = field(default=4, validator=in_((1, 4)))
    # 4 = prescribed flux from atmosphere (default, PROTEUS coupling mode)
    # 1 = native grey-body (sigma*T^4, standalone mode, bypasses atmosphere)
    dilatation: bool = field(default=False)
    mass_coordinates: bool = field(default=True)
    tsurf_poststep_change: float = field(default=30, validator=ge(0))
    event_triggering: bool = field(default=True)
    bulk_modulus: float = field(default=260e9, validator=gt(0))
    param_utbl: bool = field(default=True)
    param_utbl_const: float = field(default=1e-7, validator=gt(0))


def valid_interiordummy(instance, attribute, value):
    if instance.module != 'dummy':
        return

    pass  # dummy validation uses interior.tsurf_init (top-level)

    tliq = instance.dummy.mantle_tliq
    tsol = instance.dummy.mantle_tsol
    if tliq <= tsol:
        raise ValueError(f'Dummy liquidus ({tliq}K) must be greater than solidus ({tsol}K)')


@define
class InteriorDummy:
    """Parameters for Dummy interior module.

    Attributes
    ----------
    tsurf_init: float
        Initial magma surface temperature [K].
    tmagma_atol: float
        Max absolute change in surface temperature [K] during a single iteration.
    tmagma_rtol: float
        Max relative change in surface temperature [K] during a single iteration.
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

    tmagma_atol: float = field(default=30.0, validator=ge(0))
    tmagma_rtol: float = field(default=0.05, validator=ge(0))
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
    heat_radiogenic: bool
        Include radiogenic heat production?
    heat_tidal: bool
        Include tidal heating?
    rfront_loc: float
        Centre of rheological transition in terms of melt fraction
    rfront_wid: float
        Width of rheological transition in terms of melt fraction
    initial_thermal_state: str
        Mode for setting the initial surface temperature.
        'fixed': use tsurf_init from solver config (default).
        'self_consistent': compute from accretion + differentiation energy
        budget following White+Li (2025). Requires struct.module = 'zalmoxis'.
    thermal_state_T_eq: float
        Radiative equilibrium temperature [K] for self-consistent mode.
    thermal_state_f_accretion: float
        Heat retention efficiency for accretion energy [0-1].
    thermal_state_f_differentiation: float
        Heat retention efficiency for differentiation energy [0-1].
    thermal_state_C_iron: float
        Specific heat capacity of iron [J/kg/K].
    thermal_state_C_silicate: float
        Specific heat capacity of silicate [J/kg/K].

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

    module: str = field(validator=in_(('spider', 'aragog', 'dummy')))
    tsurf_init: float = field(default=3300.0, validator=gt(200))
    num_levels: int = field(default=100, validator=ge(40))
    num_tolerance: float = field(default=1e-10, validator=gt(0))
    trans_conduction: bool = field(default=True)
    trans_convection: bool = field(default=True)
    trans_grav_sep: bool = field(default=True)
    trans_mixing: bool = field(default=True)
    heat_radiogenic: bool = field(default=True)
    heat_tidal: bool = field(default=True)

    spider: Spider = field(factory=Spider, validator=valid_spider)
    aragog: Aragog = field(factory=Aragog, validator=valid_aragog)
    dummy: InteriorDummy = field(factory=InteriorDummy, validator=valid_interiordummy)

    grain_size: float = field(default=0.1, validator=gt(0))
    flux_guess: float = field(default=-1)
    rfront_loc: float = field(default=0.3, validator=(gt(0), lt(1)))
    rfront_wid: float = field(default=0.15, validator=(gt(0), lt(1)))

    # Initial thermal state mode:
    #   'fixed': use tsurf_init from the solver config (current default)
    #   'self_consistent': compute from accretion + differentiation energy
    #                      budget (White+Li 2025). Requires struct.module = 'zalmoxis'.
    initial_thermal_state: str = field(
        default='fixed', validator=in_(('fixed', 'self_consistent'))
    )
    # Parameters for self-consistent thermal state (ignored when 'fixed'):
    thermal_state_T_eq: float = field(default=255.0, validator=gt(0))
    thermal_state_f_accretion: float = field(default=0.04, validator=ge(0))
    thermal_state_f_differentiation: float = field(default=0.50, validator=ge(0))
    thermal_state_C_iron: float = field(default=450.0, validator=gt(0))  # Dulong-Petit (White+Li 2025)
    thermal_state_C_silicate: float = field(default=1250.0, validator=gt(0))  # Dulong-Petit (White+Li 2025)

    # Phase-dependent eddy diffusivity floor [m^2/s]. Default 0 = standard MLT.
    # When > 0, applies max(kh_MLT, floor * f(phi)) where f transitions from
    # 1 (liquid) to 0 (solid) at the rheological transition. Passed to both
    # SPIDER (-kappah_floor) and Aragog (kappah_floor in energy config).
    kappah_floor: float = field(default=0.0, validator=ge(0))
