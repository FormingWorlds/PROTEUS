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

    prevent_warming: bool
    surface_d: float = field(validator=gt(0))
    surface_k: float = field(validator=gt(0))
    cloud_enabled: bool
    cloud_alpha: float = field(validator=(ge(0), le(1)))
    surf_state: str = field(validator=in_(('mixed_layer', 'fixed', 'skin')))
    surf_greyalbedo:float = field(validator=(ge(0),le(1)))
    albedo_pl: float = field(validator=(ge(0), le(1)))
    rayleigh: bool = field(validator=warn_if_dummy)
    tmp_minimum: float = field(validator=gt(0))
    tmp_maximum: float = field(validator=tmp_max_bigger_than_tmp_min)

    module: str = field(validator=in_(('dummy', 'agni', 'janus')))

    agni: Agni
    janus: Janus
    dummy: Dummy

    @property
    def surf_state_int(self) -> int:
        """Return integer state for agni."""
        return ('mixed_layer', 'fixed', 'skin').index(self.surf_state)


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
    """

    p_top: float = field(validator=gt(0))
    spectral_group: str
    spectral_bands: str
    surf_material: str
    num_levels: int = field(validator=ge(15))
    chemistry: str = field(validator=in_((None, "eq")), converter=none_if_none)
    solve_energy: bool
    solution_atol: float = field(validator=gt(0))
    solution_rtol: float = field(validator=gt(0))
    overlap_method: str = field(validator=check_overlap)
    condensation: bool

    @property
    def chemistry_int(self) -> int:
        """Return integer state for agni."""
        return 1 if self.chemistry else 0

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

    p_top: float = field(validator=gt(0))
    p_obs: float = field(validator=gt(0))
    spectral_group: str
    spectral_bands: str
    F_atm_bc: int = field(validator=in_((0, 1)))
    num_levels: int = field(validator=ge(15))
    tropopause: str | None = field(
        validator=in_((None, 'skin', 'dynamic')), converter=none_if_none
    )
    overlap_method: str = field(validator=check_overlap)

@define
class Dummy:
    """Dummy atmosphere module.

    Attributes
    ----------
    gamma: float
        Atmosphere opacity between 0 and 1.
    """

    gamma: float = field( validator=(ge(0),le(1)) )
