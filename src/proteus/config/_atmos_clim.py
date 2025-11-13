from __future__ import annotations

import logging
import os

from attrs import define, field
from attrs.validators import ge, gt, in_, le

from ._converters import lowercase, none_if_none

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

    # ensure psurf_thresh is greater than p_top, to avoid upside-down atmosphere in transparent mode
    if instance.agni.p_top > instance.agni.psurf_thresh:
        raise ValueError("Must set `agni.p_top` to be less than `agni.psurf_thresh`")

    # ensure p_obs is greater than p_top
    if instance.agni.p_top > instance.agni.p_obs:
        raise ValueError("Must set `agni.p_top` to be less than `agni.p_obs`")

    # agni must solve_energy=true if surf_state=skin
    if (not instance.agni.solve_energy) and (instance.surf_state == 'skin'):
        raise ValueError("Must set `agni.solve_energy=true` if using `surf_state='skin'`")

    if instance.agni.latent_heat and not instance.agni.condensation:
        raise ValueError("`atmos_clim.agni`: Must set `condensation=true` if setting `latent_heat=true`")

    # set spectral files?
    if not instance.agni.spectral_group:
        raise ValueError("Must set atmos_clim.agni.spectral_group")
    if not instance.agni.spectral_bands:
        raise ValueError("Must set atmos_clim.agni.spectral_bands")

    # fastchem installed?
    if instance.agni.chemistry == "eq":
        FC_DIR = os.environ.get("FC_DIR")
        if FC_DIR:
            if not os.path.isdir(FC_DIR):
                raise FileNotFoundError(f"Fastchem not found at FC_DIR={FC_DIR}")
        else:
            raise EnvironmentError("Chemistry is enabled but environment variable `FC_DIR` is not set")

@define
class Agni:
    """AGNI atmosphere module.

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
    phs_timescale: float
        Characteristic timescale of phase changes [seconds].
    evap_efficiency: bool
        Efficiency of raindrop re-evaporation (0 to 1).
    rainout: bool
        Enable volatile rainout in the atmosphere and ocean formation below.
    latent_heat: bool
        Account for latent heat from condense/evap when solving temperature profile. Requires `rainout=true`.
    convection: bool
        Account for convective heat transport, using MLT.
    conduction: bool
        Account for conductive heat transport, using Fourier's law.
    real_gas: bool
        Use real gas equations of state in atmosphere, where possible.
    psurf_thresh: float
        Use the transparent-atmosphere solver when P_surf is less than this value [bar].
    dx_max: float
        Nominal maximum step size to T(p) during the solver process, although this is dynamic.
    dx_max_ini: float
        Initial maximum step size to T(p) when AGNI is called in the first few PROTEUS loops.
    max_steps: int
        Maximum number of iterations before giving up.
    perturb_all: bool
        Recalculate entire jacobian matrix at every iteration?
    mlt_criterion: str
        Convection criterion. Options: (l)edoux, (s)chwarzschild.
    fastchem_floor:float
        Minimum temperature allowed to be sent to FC
    fastchem_maxiter_chem:int
        Maximum FC iterations (chemistry)
    fastchem_maxiter_solv:int
        Maximum FC iterations (internal solver)
    fastchem_xtol_chem:float
        FC solver tolerance (chemistry)
    fastchem_xtol_elem:float
        FC solver tolerance (elemental)
    ini_profile: str
        Shape of initial T(p) guess: 'loglinear', 'isothermal', 'dry_adiabat', 'analytic'.
    ls_default: int
        Default linesearch method. 0: disabled, 1: goldensection, 2: backtracking.
    """

    spectral_group: str     = field(default=None)
    spectral_bands: str     = field(default=None)
    p_top: float            = field(default=1e-5, validator=gt(0))
    p_obs: float            = field(default=20e-3, validator=gt(0))
    surf_material: str      = field(default="surface_albedos/Hammond24/lunarmarebasalt.dat")
    num_levels: int         = field(default=40, validator=ge(15))
    chemistry: str          = field(default="none",
                                    validator=in_((None, "eq")),
                                    converter=none_if_none)
    solve_energy: bool      = field(default=True)
    solution_atol: float    = field(default=0.5,  validator=gt(0))
    solution_rtol: float    = field(default=0.15,  validator=gt(0))
    overlap_method: str     = field(default='ee', validator=check_overlap)
    phs_timescale: float    = field(default=1e6, validator=gt(0))
    evap_efficiency: float  = field(default=0.05, validator=(le(1), ge(0)))
    rainout: bool           = field(default=False)
    latent_heat: bool       = field(default=False)
    convection: bool        = field(default=False)
    conduction: bool        = field(default=False)
    real_gas: bool          = field(default=False)
    psurf_thresh: float     = field(default=0.1, validator=ge(0))
    dx_max: float           = field(default=35.0,  validator=gt(1))
    dx_max_ini: float       = field(default=300.0, validator=gt(1))
    max_steps: int          = field(default=70, validator=gt(2))
    perturb_all: bool       = field(default=True)
    mlt_criterion: str      = field(default='l', validator=in_(('l','s',)))
    fastchem_floor:float        = field(default=150.0, validator=gt(0.0))
    fastchem_maxiter_chem:int   = field(default=60000, validator=gt(200))
    fastchem_maxiter_solv:int   = field(default=20000, validator=gt(200))
    fastchem_xtol_chem:float    = field(default=1e-4,  validator=gt(0.0))
    fastchem_xtol_elem:float    = field(default=1e-4,  validator=gt(0.0))
    ini_profile: str        = field(default='loglinear',
                                    converter=lowercase,
                                    validator=in_(('loglinear','isothermal',
                                                   'dry_adiabat','analytic'))
                                    )
    ls_default: int         = field(default=2, validator=in_((0,1,2)))

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

    A parametrised model of the atmosphere designed for debugging. The greenhouse effect
    is captured by `gamma` which produces a transparent atmosphere when 0, and a completely
    opaque atmosphere when 1. The height of the atmosphere equals the scale height times
    the `height_factor` variable.

    Attributes
    ----------
    gamma: float
        Atmosphere opacity factor between 0 and 1.
    height_factor: float
        A multiplying factor applied to the ideal-gas scale height.
    """

    gamma: float         = field(default=0.7, validator=(ge(0),le(1)))
    height_factor: float = field(default=3.0, validator=ge(0))


def valid_albedo(instance, attribute, value):

    if isinstance(value, str):
        return

    elif isinstance(value, float):
        if not (0 <= value <= 1):
            raise ValueError("The value of `albedo_pl` must be between 0 and 1")
        else:
            return

    else:
        raise ValueError("The value of `albedo_pl` must be a string or a float")

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
    albedo_pl: float | str
        Planetary bond albedo used to emulate scattering. Can be float (0 to 1) or str (path to CSV file containing lookup data).
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
    albedo_pl               = field(default=0.0, validator=valid_albedo)
    rayleigh: bool          = field(default=True,validator=warn_if_dummy)
    tmp_minimum: float      = field(default=0.5, validator=gt(0))
    tmp_maximum: float      = field(default=5000.0,
                                    validator=tmp_max_bigger_than_tmp_min)

    @property
    def surf_state_int(self) -> int:
        """Return integer surface boundary condition for agni."""
        match self.surf_state:
            case 'fixed':
                return 1
            case 'skin':
                return 2
            case _:
                raise ValueError(f"Invalid surf_state for AGNI: '{self.surf_state}'")

    @property
    def albedo_from_file(self) -> bool:
        """Is albedo set by lookup table or not?"""
        if isinstance(self.albedo_pl, str):
            return True
        elif isinstance(self.albedo_pl, float):
            return False
        else:
            raise ValueError("Cannot determine configuration for setting `albedo_pl`")
