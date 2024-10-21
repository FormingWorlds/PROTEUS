from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Agni:
    """Agni atmosphere module.

    Attributes
    ----------
    p_top: float
        Bar, top of atmosphere grid pressure.
    spectral_group: str
        Which gas opacities to include.
    spectral_bands: str
        Number of spectral bands?
    num_levels: str
        Number of atmospheric grid levels.
    chemistry: str | None
        Choices: None, "eq", "kin"
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
    """Janus atmosphere module.

    Attributes
    ----------
    p_top: float
        Bar, top of atmosphere grid pressure.
    spectral_group: str
        Which gas opacities to include.
    spectral_bands: str
        Number of spectral bands.
    F_atm_bc: int
        Measure outgoing flux at: (0) TOA | (1) Surface.
    num_levels: int
        Number of atmospheric grid levels.
    tropopause: str | None
        Choices: None | skin | dynamic.
    """
    p_top: float
    spectral_group: str
    spectral_bands: str
    F_atm_bc: int = field(validator=validators.in_((0, 5)))
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


@define
class AtmosClim:
    """Atmosphere parameters, model selection.

    Attributes
    ----------
    prevent_warming: bool
        Do not allow the planet to heat up.
    surface_d: float
        M, conductive skin thickness.
    surface_k: float
        W m-1 K-1, conductive skin thermal conductivity.
    cloud_enabled: bool
        Enable water cloud radiative effects.
    cloud_alpha: float
        Condensate retention fraction (1 -> fully retained).
    surf_state: str
        Surface scheme: "mixed_layer", "fixed", "skin".
    surf_albedo: float
        Path to file ("string") or grey quantity (float).
    albedo_pl: float
        Bond albedo (scattering).
    rayleigh: bool
        Enable rayleigh scattering.
    tmp_minimum: float
        Temperature floor on solver.
    tmp_maximum: float
        Temperature ceiling on solver.
    module: str
        Which atmosphere module to use.
    agni: Agni
        Config parameters for Agni atmosphere module
    janus: Janus
        Config parameters for Janus atmosphere module
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
