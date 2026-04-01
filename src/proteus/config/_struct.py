from __future__ import annotations

from typing import Optional

from attrs import define, field
from attrs.validators import ge, gt, in_, le, lt


def valid_zalmoxis(instance, attribute, value):
    if instance.module == 'spider':
        return

    max_iterations_outer = instance.zalmoxis.max_iterations_outer
    max_iterations_inner = instance.zalmoxis.max_iterations_inner
    max_iterations_pressure = instance.zalmoxis.max_iterations_pressure
    core_eos = instance.zalmoxis.core_eos
    mantle_eos = instance.zalmoxis.mantle_eos
    ice_layer_eos = instance.zalmoxis.ice_layer_eos
    mantle_mass_fraction = instance.zalmoxis.mantle_mass_fraction

    if max_iterations_outer < 3:
        raise ValueError('`interior.zalmoxis.max_iterations_outer` must be > 2')
    if max_iterations_inner < 13:
        raise ValueError('`interior.zalmoxis.max_iterations_inner` must be > 12')
    if max_iterations_pressure < 13:
        raise ValueError('`interior.zalmoxis.max_iterations_pressure` must be > 12')

    # EOS format validation: must be "<source>:<material>"
    for name, eos_val in [('core_eos', core_eos), ('mantle_eos', mantle_eos)]:
        if ':' not in eos_val:
            raise ValueError(
                f"`struct.zalmoxis.{name}` must be in '<source>:<material>' format, "
                f"got '{eos_val}'"
            )
    if ice_layer_eos and ':' not in ice_layer_eos:
        raise ValueError(
            f"`struct.zalmoxis.ice_layer_eos` must be empty or '<source>:<material>' format, "
            f"got '{ice_layer_eos}'"
        )

    # WolfBower2018 EOS is limited to 1 TPa. For planets > 2 M_earth,
    # CMB pressure exceeds this and Zalmoxis will fail to converge.
    import logging as _logging

    _log = _logging.getLogger('fwl.' + __name__)
    # mushy_zone_factor only applies to PALEOS unified tables
    mzf = getattr(instance.zalmoxis, 'mushy_zone_factor', 0.8)
    if mzf < 1.0 and not mantle_eos.startswith('PALEOS:'):
        _log.warning(
            'mushy_zone_factor=%.2f has no effect with mantle EOS %s. '
            'The mushy zone factor only applies to PALEOS unified tables. '
            'For WolfBower2018/RTPress100TPa, the mushy zone is defined by '
            'the solidus/liquidus melting curve files.',
            mzf,
            mantle_eos,
        )

    # 2-layer model (no ice layer, non-T-dep mantle): mantle_mass_fraction must be 0
    _TDEP_PREFIXES = ('WolfBower2018', 'RTPress100TPa')
    if not ice_layer_eos and not mantle_eos.startswith(_TDEP_PREFIXES):
        if mantle_mass_fraction != 0:
            raise ValueError(
                '`struct.zalmoxis.mantle_mass_fraction` must be 0 for a 2-layer model '
                'without T-dependent mantle EOS (only core + mantle are modeled).'
            )

    # 3-layer model (with ice layer): mass fractions must not exceed 75%
    if ice_layer_eos:
        cmf = instance.core_frac if instance.core_frac_mode == 'mass' else 0.325
        if cmf + mantle_mass_fraction > 0.75:
            raise ValueError(
                '`core_frac` and `zalmoxis.mantle_mass_fraction` '
                'must add up to <= 75% for a 3-layer model (Seager 2007).'
            )


@define
class Zalmoxis:
    """Parameters for Zalmoxis module.

    Attributes
    ----------
    core_eos: str
        EOS for the core layer. Format: "<source>:<material>".
        Tabulated: "Seager2007:iron".
        Analytic: "Analytic:iron", "Analytic:MgFeSiO3", etc.
    mantle_eos: str
        EOS for the mantle layer. Format: "<source>:<material>".
        Tabulated: "Seager2007:MgSiO3", "WolfBower2018:MgSiO3".
        Analytic: "Analytic:MgSiO3", "Analytic:MgFeSiO3", etc.
    ice_layer_eos: str
        EOS for the ice/water layer (3-layer model). Empty string for
        2-layer model (core + mantle only).
        Tabulated: "Seager2007:H2O". Analytic: "Analytic:H2O".
    mushy_zone_factor: float
        Cryoscopic depression factor controlling the width of the mushy
        zone (partially molten region) in the PALEOS unified EOS.
        Defines the solidus as T_sol = T_liq * mushy_zone_factor.
        1.0 = sharp phase boundary (no mushy zone).
        0.8 = solidus at 80% of the liquidus temperature, roughly
        matching the Stixrude+2014 cryoscopic depression for MgSiO3.
        Must be in [0.7, 1.0]. Only applies to PALEOS unified EOS;
        ignored for WolfBower2018 and RTPress100TPa (which use explicit
        melting curve files). This factor is applied consistently across
        Zalmoxis (density interpolation), SPIDER (phase boundaries),
        and the VolatileProfile phi-blending.
    mantle_mass_fraction: float
        Fraction of the planet's interior mass corresponding to the mantle.
        Required for 3-layer models (with ice layer) and for T-dependent
        2-layer models (WolfBower2018, RTPress100TPa) where it partitions
        mass between core and mantle layers.
    temperature_mode: str
        Choice of input temperature profile: "isothermal", "linear",
        "prescribed", "adiabatic".
    surface_temperature: float
        Surface temperature (K), required for temperature_mode="isothermal",
        "linear", or "adiabatic", ignored otherwise.
    center_temperature: float
        Center temperature (K), required for temperature_mode="linear"
        or "adiabatic" (initial guess), ignored otherwise.
    temperature_profile_file: Optional[str]
        Filename containing a prescribed temperature profile, required for
        temperature_mode="prescribed".
    num_levels: int
        Number of Zalmoxis radius layers.
    max_iterations_outer: int
        Maximum number of iterations for the outer loop.
    tolerance_outer: float
        Convergence tolerance for the outer loop [kg].
    max_iterations_inner: int
        Maximum number of iterations for the inner loop.
    tolerance_inner: float
        Convergence tolerance for the inner loop [kg/m^3].
    relative_tolerance: float
        Relative tolerance for solve_ivp.
    absolute_tolerance: float
        Absolute tolerance for solve_ivp.
    maximum_step: float
        Maximum integration step size for solve_ivp (m).
    adaptive_radial_fraction: float
        Fraction (0-1) of the radial domain defining where solve_ivp
        transitions from adaptive to fixed-step integration when using
        the WolfBower2018 T-dependent mantle EOS.
    max_center_pressure_guess: float
        Maximum pressure guess at the center of the planet (Pa).
    target_surface_pressure: float
        Target surface pressure for the pressure adjustment [Pa].
    pressure_tolerance: float
        Convergence tolerance for the pressure adjustment [Pa].
    max_iterations_pressure: int
        Maximum number of iterations for the pressure adjustment.
    verbose: bool
        If true, logs detailed convergence info and warnings.
    iteration_profiles_enabled: bool
        If true, writes pressure and density profiles for each iteration.
    """

    core_eos: str = field(default='Seager2007:iron')
    mantle_eos: str = field(default='Seager2007:MgSiO3')
    ice_layer_eos: str = field(default='')

    mushy_zone_factor: float = field(default=0.8, validator=(ge(0.7), le(1.0)))

    mantle_mass_fraction: float = field(default=0, validator=(ge(0), lt(1)))
    temperature_mode: str = field(
        default='isothermal',
        validator=in_(('isothermal', 'linear', 'prescribed', 'adiabatic')),
    )
    surface_temperature: float = field(default=3500, validator=ge(0))
    center_temperature: float = field(default=6000, validator=ge(0))
    temperature_profile_file: Optional[str] = field(default=None)

    num_levels: int = field(default=150)

    max_iterations_outer: int = field(default=100, validator=ge(1))
    tolerance_outer: float = field(default=3e-3, validator=ge(0))
    max_iterations_inner: int = field(default=100, validator=ge(1))
    tolerance_inner: float = field(default=1e-4, validator=ge(0))
    relative_tolerance: float = field(default=1e-5, validator=ge(0))
    absolute_tolerance: float = field(default=1e-6, validator=ge(0))
    maximum_step: float = field(default=250000, validator=ge(0))
    adaptive_radial_fraction: float = field(default=0.98, validator=ge(0))
    max_center_pressure_guess: float = field(default=0.99e12, validator=ge(0))

    target_surface_pressure: float = field(default=101325, validator=ge(0))
    pressure_tolerance: float = field(default=1e9, validator=ge(0))
    max_iterations_pressure: int = field(default=200, validator=ge(1))

    verbose: bool = field(default=False)
    iteration_profiles_enabled: bool = field(default=False)

    def __attrs_post_init__(self):
        if self.temperature_mode == 'prescribed':
            if not self.temperature_profile_file:
                raise ValueError(
                    '`temperature_profile_file` must be provided when '
                    "`temperature_mode` is 'prescribed'."
                )


@define
class Struct:
    """Planetary structure (mass, radius).

    Attributes
    ----------
    core_frac: float
        Fraction of the planet's interior radius corresponding to the core.
    module: str
        Module for solving the planet's interior structure. Choices: 'self', 'zalmoxis'.
    zalmoxis: Zalmoxis or None
        Zalmoxis parameters if module is 'zalmoxis'.
    update_interval: float
        Maximum interval (ceiling) between structure re-computations [yr].
        Only used when module is 'zalmoxis'. 0 means only compute structure
        at init (no dynamic updates).
    update_min_interval: float
        Minimum interval (floor) between structure re-computations [yr].
        Prevents thrashing during rapid cooling. Only used when
        update_interval > 0.
    update_dtmagma_frac: float
        Fractional change in T_magma that triggers a structure update.
        Update fires when |T_new - T_ref| / T_ref >= this value.
    update_dphi_abs: float
        Absolute change in Phi_global that triggers a structure update.
        Update fires when |Phi_new - Phi_ref| >= this value.
    core_frac_mode: str
        How core_frac is interpreted. 'radius': fraction of planet radius.
        'mass': fraction of total planet mass. Only 'radius' is supported
        when module = 'self'.
    core_density: float or str
        Density of the planet's core [kg m-3]. Set to 'self' for
        self-consistent calculation by Zalmoxis (requires module = 'zalmoxis').
    core_heatcap: float or str
        Specific heat capacity of the planet's core [J kg-1 K-1]. Set to 'self'
        for self-consistent calculation by Zalmoxis (requires module = 'zalmoxis').
    """

    core_frac: float = field(validator=(gt(0), lt(1)))
    core_frac_mode: str = field(default='mass', validator=in_(('radius', 'mass')))

    module: Optional[str] = field(
        default='zalmoxis',
        validator=lambda inst, attr, val: val is None or val in ('spider', 'zalmoxis'),
    )
    zalmoxis: Optional[Zalmoxis] = field(
        default=None,
        validator=lambda inst, attr, val: val is None or valid_zalmoxis(inst, attr, val),
    )

    update_interval: float = field(default=10, validator=ge(0))
    update_min_interval: float = field(default=0, validator=ge(0))
    update_dtmagma_frac: float = field(default=0.03, validator=(gt(0), lt(1)))
    update_dphi_abs: float = field(default=0.05, validator=(gt(0), lt(1)))

    mesh_max_shift: float = field(default=0.05, validator=(gt(0), lt(1)))
    mesh_convergence_interval: float = field(default=10.0, validator=gt(0))

    equilibrate_init: bool = False
    equilibrate_max_iter: int = field(default=15, validator=ge(1))
    equilibrate_tol: float = field(default=0.01, validator=gt(0))

    global_miscibility: bool = field(default=False)
    miscibility_max_iter: int = field(default=10, validator=ge(1))
    miscibility_tol: float = field(default=0.01, validator=gt(0))

    core_density = field(default=10738.33)
    core_heatcap = field(default=880.0)

    melting_dir: str = field(default='Monteux-600')
    eos_dir: str = field(default='WolfBower2018_MgSiO3')
    def __attrs_post_init__(self):
        if self.update_interval > 0 and self.update_min_interval > self.update_interval:
            raise ValueError(
                f'`update_min_interval` ({self.update_min_interval}) must be '
                f'<= `update_interval` ({self.update_interval}), otherwise '
                f'the floor blocks all updates before the ceiling can fire.'
            )
        if self.global_miscibility and self.module == 'spider':
            raise ValueError(
                '`global_miscibility` requires `module = "zalmoxis"`. '
                'The binodal-aware structure solver is only available in Zalmoxis.'
            )

        # core_frac_mode = "mass" requires Zalmoxis
        if self.core_frac_mode == 'mass' and self.module == 'spider':
            raise ValueError(
                '`core_frac_mode = "mass"` requires `module = "zalmoxis"`. '
                'The spider module only supports radius-based core fraction.'
            )

        # core_density and core_heatcap: "self" requires Zalmoxis
        for param_name in ('core_density', 'core_heatcap'):
            val = getattr(self, param_name)
            if val == 'self':
                if self.module == 'spider':
                    raise ValueError(
                        f'`{param_name} = "self"` requires `module = "zalmoxis"`. '
                        f'Set a numerical value when using module = "spider".'
                    )
            elif not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(
                    f'`{param_name}` must be "self" or a positive number, got {val!r}'
                )
