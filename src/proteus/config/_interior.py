from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, lt


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


def valid_path(instance, attribute, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{attribute.name}' must be a non-empty string")


@define
class Spider:
    """Parameters for SPIDER module.

    Attributes
    ----------
    tolerance_rel: float
        Relative solver tolerance (SUNDIALS-specific).
    solver_type: str
        Numerical integrator. Choices: 'adams', 'bdf'.
    matprop_smooth_width: float
        Window width, in melt-fraction, for smoothing properties across liquidus and solidus
    """

    ini_entropy: float = field(default=3200.0, validator=gt(200.0))
    tolerance_rel: float = field(default=1e-10, validator=gt(0))
    solver_type: str = field(default='bdf', validator=in_(('adams', 'bdf')))
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
    trans_conduction: bool
        Whether to include conductive heat flux in the model. Default is True.
    trans_convection: bool
        Whether to include convective heat flux in the model. Default is True.
    trans_grav_sep: bool
        Whether to include gravitational separation flux in the model. Default is True (matches SPIDER).
    trans_mixing: bool
        Whether to include mixing flux in the model. Default is True (matches SPIDER).
    dilatation: bool
        Whether to include dilatation source term in the model. Default is True.
    mass_coordinates: bool
        Whether to use mass coordinates in the model. Default is True.
        Uses uniform spacing in mass coordinate space, giving larger cells
        at the surface where density is lower, matching SPIDER's mesh.
    event_triggering: bool
        Whether to include event triggering in the solver. Default is True.
    param_utbl: bool
        Whether to parameterize the ultra-thin thermal boundary layer at the surface.
        Reduces the effective radiating temperature to account for the unresolved
        boundary layer (Bower et al. 2018, Eq. 18). Default is True (matches SPIDER).
    param_utbl_const: float
        UTBL scaling constant [K^-2]. Default is 1e-7 (matches SPIDER).
    jax: bool
        Use JAX/diffrax solver backend instead of scipy BDF. Default is False.
        When True, the entropy ODE is integrated with diffrax Tsit5 instead of
        scipy solve_ivp (BDF). Requires jax, equinox, and diffrax packages.
    """

    dilatation: bool = field(default=True)
    mass_coordinates: bool = field(default=True)
    event_triggering: bool = field(default=True)
    param_utbl: bool = field(default=True)
    param_utbl_const: float = field(default=1e-7, validator=gt(0))
    jax: bool = field(default=False)


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

    module: str = field(validator=in_(('spider', 'aragog', 'dummy')))
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

    mixing_length: str = field(
        default='nearest', validator=in_(('nearest', 'constant'))
    )
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

    rfront_loc: float = field(default=0.3, validator=(gt(0), lt(1)))
    rfront_wid: float = field(default=0.15, validator=(gt(0), lt(1)))

    # Phase-dependent eddy diffusivity floor [m^2/s]. Default 0 = standard MLT.
    # When > 0, applies max(kh_MLT, floor * f(phi)) where f transitions from
    # 1 (liquid) to 0 (solid) at the rheological transition. Passed to both
    # SPIDER (-kappah_floor) and Aragog (kappah_floor in energy config).
    kappah_floor: float = field(default=10.0, validator=ge(0))
