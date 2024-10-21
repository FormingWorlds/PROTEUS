from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


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
    surf_albedo: float
        Grey albedo applied to the surface of the planet [dimensionless].
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
    surface_d: float
    surface_k: float
    cloud_enabled: bool
    cloud_alpha: float
    surf_state: str = field(validator=validators.in_(('mixed_layer', 'fixed', 'skin')))
    surf_albedo: float
    albedo_pl: float
    rayleigh: bool
    tmp_minimum: float
    tmp_maximum: float

    module: str = field(validator=validators.in_(('dummy', 'agni', 'janus')))

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
        Spectral file codename defining the gas opacities to be included. See [documentation](See documentation: https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/spectral_files.pdf).
    spectral_bands: str
        Number of wavenumer bands in k-table. See documentation.
    num_levels: str
        Number of atmospheric grid levels.
    chemistry: str | None
        Treatment of self-consistent atmospheric chemsitry. Choices: "none", "eq", "kin".
    """
    p_top: float
    spectral_group: str
    spectral_bands: str
    num_levels: int = field(validator=validators.ge(15))
    chemistry: str | None = field(
        validator=validators.in_((None, 'eq', 'kin')), converter=none_if_none
    )

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
    """
    p_top: float
    spectral_group: str
    spectral_bands: str
    F_atm_bc: int = field(validator=validators.in_((0, 1)))
    num_levels: int = field(validator=validators.ge(15))
    tropopause: str | None = field(
        validator=validators.in_((None, 'skin', 'dynamic')), converter=none_if_none
    )


@define
class Dummy:
    """Dummy atmosphere module.

    Attributes
    ----------
    gamma: float
        Atmosphere opacity between 0 and 1.
    """
    gamma: float
