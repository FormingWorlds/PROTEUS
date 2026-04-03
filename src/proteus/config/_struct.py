from __future__ import annotations

from typing import Optional

from attrs import define, field
from attrs.validators import ge, gt, in_, le, lt

from ._converters import none_if_none


def valid_zalmoxis(instance, attribute, value):
    if instance.module == 'spider':
        return

    core_eos = instance.zalmoxis.core_eos
    mantle_eos = instance.zalmoxis.mantle_eos
    ice_layer_eos = instance.zalmoxis.ice_layer_eos
    mantle_mass_fraction = instance.zalmoxis.mantle_mass_fraction

    # EOS format validation: must be "<source>:<material>"
    for name, eos_val in [('core_eos', core_eos), ('mantle_eos', mantle_eos)]:
        if ':' not in eos_val:
            raise ValueError(
                f"`interior_struct.zalmoxis.{name}` must be in '<source>:<material>' format, "
                f"got '{eos_val}'"
            )
    if ice_layer_eos is not None and ':' not in ice_layer_eos:
        raise ValueError(
            f"`interior_struct.zalmoxis.ice_layer_eos` must be 'none' or '<source>:<material>' format, "
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
    if ice_layer_eos is None and not mantle_eos.startswith(_TDEP_PREFIXES):
        if mantle_mass_fraction != 0:
            raise ValueError(
                '`interior_struct.zalmoxis.mantle_mass_fraction` must be 0 for a 2-layer model '
                'without T-dependent mantle EOS (only core + mantle are modeled).'
            )

    # 3-layer model (with ice layer): mass fractions must not exceed 75%
    if ice_layer_eos is not None:
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
    ice_layer_eos: str or None
        EOS for the ice/water layer (3-layer model). 'none' for
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
    num_levels: int
        Number of Zalmoxis radius layers.
    lookup_nP: int
        Number of pressure points in SPIDER P-S tables generated from PALEOS.
    lookup_nS: int
        Number of entropy points in SPIDER P-S tables generated from PALEOS.
    """

    core_eos: str = field(default='PALEOS:iron')
    mantle_eos: str = field(default='PALEOS:MgSiO3')
    ice_layer_eos = field(default=None, converter=none_if_none)

    mushy_zone_factor: float = field(default=0.8, validator=(ge(0.7), le(1.0)))

    mantle_mass_fraction: float = field(default=0, validator=(ge(0), lt(1)))

    num_levels: int = field(default=150)

    # Structure update triggers (during coupled evolution)
    update_interval: float = field(default=1e9, validator=ge(0))
    update_min_interval: float = field(default=0, validator=ge(0))
    update_dtmagma_frac: float = field(default=0.05, validator=(gt(0), lt(1)))
    update_dphi_abs: float = field(default=0.05, validator=(gt(0), lt(1)))

    # Mesh smoothing
    mesh_max_shift: float = field(default=0.05, validator=(gt(0), lt(1)))
    mesh_convergence_interval: float = field(default=10.0, validator=gt(0))

    # Pre-main-loop equilibration (CALLIOPE + Zalmoxis convergence)
    equilibrate_init: bool = field(default=True)
    equilibrate_max_iter: int = field(default=15, validator=ge(1))
    equilibrate_tol: float = field(default=0.01, validator=gt(0))

    # SPIDER P-S table resolution (generated from PALEOS)
    lookup_nP: int = field(default=1000, validator=ge(100))
    lookup_nS: int = field(default=250, validator=ge(50))

    # Binodal-aware miscibility (H2-MgSiO3 solvus)
    global_miscibility: bool = field(default=False)
    miscibility_max_iter: int = field(default=10, validator=ge(1))
    miscibility_tol: float = field(default=0.01, validator=gt(0))

    def __attrs_post_init__(self):
        if self.update_interval > 0 and self.update_min_interval > self.update_interval:
            raise ValueError(
                f'`update_min_interval` ({self.update_min_interval}) must be '
                f'<= `update_interval` ({self.update_interval}), otherwise '
                f'the floor blocks all updates before the ceiling can fire.'
            )


@define
class Struct:
    """Planetary structure (mass, radius).

    Attributes
    ----------
    core_frac: float
        Fraction of the planet's interior radius corresponding to the core.
    module: str
        Module for solving the planet's interior structure. Choices: 'dummy', 'spider', 'zalmoxis'.
    zalmoxis: Zalmoxis or None
        Zalmoxis parameters if module is 'zalmoxis'.
    core_frac_mode: str
        How core_frac is interpreted. 'radius': fraction of planet radius.
        'mass': fraction of total planet mass. Only 'radius' is supported
        when module = 'spider'.
    core_density: float or str
        Density of the planet's core [kg m-3]. Set to 'self' for
        self-consistent calculation by Zalmoxis (requires module = 'zalmoxis').
    core_heatcap: float or str
        Specific heat capacity of the planet's core [J kg-1 K-1]. Set to 'self'
        for self-consistent calculation by Zalmoxis (requires module = 'zalmoxis').
    """

    core_frac: float = field(default=0.325, validator=(gt(0), lt(1)))
    core_frac_mode: str = field(default='mass', validator=in_(('radius', 'mass')))

    module: Optional[str] = field(
        default='zalmoxis',
        validator=lambda inst, attr, val: val is None or val in ('dummy', 'spider', 'zalmoxis'),
    )
    zalmoxis: Optional[Zalmoxis] = field(
        factory=Zalmoxis,
        validator=lambda inst, attr, val: val is None or valid_zalmoxis(inst, attr, val),
    )

    core_density = field(default='self')
    core_heatcap = field(default='self')

    melting_dir = field(default=None, converter=none_if_none)
    eos_dir = field(default=None, converter=none_if_none)

    def __attrs_post_init__(self):
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

        # melting_dir and eos_dir: required for the spider struct module
        # (Zalmoxis and dummy derive EOS from their own config)
        if self.module == 'spider':
            if self.melting_dir is None:
                raise ValueError(
                    'interior_struct.melting_dir must be set when module = "spider". '
                    'Provide a melting curve folder name (e.g. "Monteux-600") from '
                    'FWL_DATA/interior_lookup_tables/Melting_curves/.'
                )
            if self.eos_dir is None:
                raise ValueError(
                    'interior_struct.eos_dir must be set when module = "spider". '
                    'Provide an EOS folder name (e.g. "WolfBower2018_MgSiO3") from '
                    'FWL_DATA/interior_lookup_tables/EOS/dynamic/.'
                )
