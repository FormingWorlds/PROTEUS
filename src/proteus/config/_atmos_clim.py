from __future__ import annotations

import logging

from attrs import define, field
from attrs.validators import ge, gt, in_, le

from ._converters import none_if_none

log = logging.getLogger('fwl.' + __name__)

def tmp_max_bigger_than_tmp_min(instance, attribute, value):
    if value <= instance.tmp_minimum:
        raise ValueError("'tmp_maximum' has to be bigger than 'tmp_minimum'.")

def warn_if_dummy(instance, attribute, value):
    if (instance.module == 'dummy') and value:
        raise ValueError('Dummy atmos_clim module is incompatible with Rayleigh scattering')

def check_overlap(instance, attribute, value):
    _overlaps = ("ro", "ee", "rorr")
    if value not in _overlaps:
        raise ValueError("Overlap type must be one of " + str(_overlaps))

def valid_agni(instance, attribute, value):
    if instance.module != "agni":
        return

    # agni must solve_energy=true if surf_state=skin
    if (not instance.agni.solve_energy) and (instance.surf_state == 'skin'):
        raise ValueError("Must set `agni.solve_energy=true` if using `surf_state='skin'`")

    # cannot set condensation and chemistry at the same time
    if instance.agni.chemistry and instance.agni.condensation:
        raise ValueError("`atmos_clim.agni`: Cannot enable condensation and chemistry at the same time")

    # set spectral files?
    if not instance.agni.spectral_group:
        raise ValueError("Must set atmos_clim.agni.spectral_group")
    if not instance.agni.spectral_bands:
        raise ValueError("Must set atmos_clim.agni.spectral_bands")

@define
class Agni:
    """AGNI atmosphere module.

    Attributes
    ----------
    p_top: float
        Top of atmosphere grid pressure [bar].
    spectral_group: str
        Spectral file codename defining the gas opacities to be included. See [documentation](https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/spectral_files.pdf).
    spectral_bands: str
        Number of wavenumer bands in k-table. See documentation.
    surf_material : str
        File name for material used to set surface single-scattering properties, relative to FWL data directory. Set to 'greybody' to use `surf_greyalbedo`. See [documentation](https://fwl-proteus.readthedocs.io/en/latest/data/#surfaces) for potential options.
    num_levels: str
        Number of atmospheric grid levels.
    chemistry: str | None
        Treatment of self-consistent atmospheric chemsitry. Choices: "none", "eq".
    solve_energy: bool
        Solve for an energy-conserving atmosphere solution.
    solution_atol: float
        Absolute tolerance on the atmosphere solution.
    solution_rtol: float
        Relative tolerance on the atmosphere solution.
    overlap_method: str
        Gas overlap method. Choices: random overlap ("ro"), RO with resorting+rebinning ("rorr"), equivalent extinction ("ee").
    condensation: bool
        Enable volatile condensation/phase change in the atmosphere.
    real_gas: bool
        Use real gas equations of state in atmosphere, where possible.
    psurf_thresh: float
        Use the transparent-atmosphere solver when P_surf is less than this value [bar].
    """

    spectral_group: str     = field(default=None)
    spectral_bands: str     = field(default=None)
    p_top: float            = field(default=1e-5, validator=gt(0))
    surf_material: str      = field(default="surface_albedos/Hammond24/lunarmarebasalt.dat")
    num_levels: int         = field(default=40, validator=ge(15))
    chemistry: str          = field(default="none",
                                    validator=in_((None, "eq")),
                                    converter=none_if_none)
    solve_energy: bool      = field(default=True)
    solution_atol: float    = field(default=0.5,  validator=gt(0))
    solution_rtol: float    = field(default=0.15,  validator=gt(0))
    overlap_method: str     = field(default='ee', validator=check_overlap)
    condensation: bool      = field(default=False)
    real_gas: bool          = field(default=False)
    psurf_thresh: bool      = field(default=0.1, validator=ge(0))

    @property
    def chemistry_int(self) -> int:
        """Return integer state for agni."""
        return 1 if self.chemistry else 0

def valid_janus(instance, attribute, value):
    if instance.module != "janus":
        return

    # set spectral files?
    if not instance.janus.spectral_group:
        raise ValueError("Must set atmos_clim.janus.spectral_group")
    if not instance.janus.spectral_bands:
        raise ValueError("Must set atmos_clim.janus.spectral_bands")

@define
class Janus:
    """JANUS atmosphere module.

    Attributes
    ----------
    p_top: float
        Top of atmosphere grid pressure [bar].
    p_obs: float
        Pressure level probed by observations [bar]
    spectral_group: str
        Spectral file codename defining the gas opacities to be included. See [documentation](https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/spectral_files.pdf).
    spectral_bands: str
        Number of wavenumer bands in k-table. See documentation.
    F_atm_bc: int
        Measure outgoing flux using value at TOA (0) or surface (1).
    num_levels: int
        Number of atmospheric grid levels.
    tropopause: str | None
        Scheme for determining tropopause location. Choices: "none", "skin", "dynamic".
    overlap_method: str
        Gas overlap method. Choices: random overlap ("ro"), RO with resorting+rebinning ("rorr"), equivalent extinction ("ee").
    """

    spectral_group: str     = field(default=None)
    spectral_bands: str     = field(default=None)
    p_top: float            = field(default=1e-5, validator=gt(0))
    p_obs: float            = field(default=2e-3, validator=gt(0))
    F_atm_bc: int           = field(default=0, validator=in_((0, 1)))
    num_levels: int         = field(default=90, validator=ge(15))
    tropopause: str | None  = field(default="none",
                                    validator=in_((None, 'skin', 'dynamic')),
                                    converter=none_if_none)
    overlap_method: str     = field(default='ee', validator=check_overlap)

@define
class Dummy:
    """Dummy atmosphere module.

    Attributes
    ----------
    gamma: float
        Atmosphere opacity between 0 and 1.
    """

    gamma: float = field(default=0.7, validator=(ge(0),le(1)) )


@define
class AtmosClim:
    """Atmosphere parameters, model selection.

    Attributes
    ----------
    prevent_warming: bool
        When True, require the planet to monotonically cool over time.
    surface_d: float
        Conductive skin thickness [m],
    surface_k: float
        Conductive skin thermal conductivity [W m-1 K-1].
    cloud_enabled: bool
        Enable water cloud radiative effects.
    cloud_alpha: float
        Condensate retention fraction (0 => full rainout, 1 => fully retained).
    surf_state: str
        Surface energy balance scheme. Choices: "mixed_layer", "fixed", "skin".
    surf_greyalbedo : float
        Grey surface albedo.
    albedo_pl: float
        Planetary/bold albedo used to emulate scattering [dimensionless].
    rayleigh: bool
        Include Rayleigh scattering in the radiative transfer calculations.
    tmp_minimum: float
        Minimum temperature throughout the atmosphere [K].
    tmp_maximum: float
        Maximum temperature throughout the atmosphere [K].
    module: str
        Which atmosphere module to use.
    agni: Agni
        Config parameters for AGNI atmosphere module
    janus: Janus
        Config parameters for JANUS atmosphere module
    dummy: Dummy
        Config parameters for dummy atmosphere module
    """

    module: str = field(validator=in_(('dummy', 'agni', 'janus')))

    agni: Agni   = field(factory=Agni,  validator=valid_agni)
    janus: Janus = field(factory=Janus, validator=valid_janus)
    dummy: Dummy = field(factory=Dummy)

    surf_state: str         = field(default='skin',
                                    validator=(
                                        in_(('mixed_layer', 'fixed', 'skin')),
                                    ))
    prevent_warming: bool   = field(default=False)
    surface_d: float        = field(default=0.01, validator=gt(0))
    surface_k: float        = field(default=2.0,  validator=gt(0))
    cloud_enabled: bool     = field(default=False)
    cloud_alpha: float      = field(default=0.0, validator=(ge(0), le(1)))
    surf_greyalbedo:float   = field(default=0.2, validator=(ge(0),le(1)))
    albedo_pl: float        = field(default=0.0, validator=(ge(0), le(1)))
    rayleigh: bool          = field(default=True,validator=warn_if_dummy)
    tmp_minimum: float      = field(default=0.5, validator=gt(0))
    tmp_maximum: float      = field(default=5000.0,
                                    validator=tmp_max_bigger_than_tmp_min)

    @property
    def surf_state_int(self) -> int:
        """Return integer state for agni."""
        match self.surf_state:
            case 'fixed':
                return 1
            case 'skin':
                return 2
            case _:
                raise ValueError(f"Invalid surf_state for AGNI: '{self.surf_state}'")
