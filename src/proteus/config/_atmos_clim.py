from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Agni:
    p_top: float
    spectral_group: str
    spectral_bands: str
    num_levels: int = field(validator=validators.ge(15))
    chemistry: str | None = field(
        validator=validators.in_((None, 'eq', 'kin')), converter=none_if_none
    )


@define
class Janus:
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
    gamma: float


@define
class AtmosClim:
    """Atmosphere parameters, model selection"""
    prevent_warming: bool
    surface_d: float
    surface_k: float
    cloud_enabled: bool
    cloud_alpha: float
    surf_state: str = field(validator=validators.in_(('mixed_layer', 'fixed', 'skin')))
    surf_albedo: float
    albedo_pl: float
    rayleigh: float
    tmp_minimum: float
    tmp_maximum: float

    module: str = field(validator=validators.in_(('dummy', 'agni', 'janus')))

    agni: Agni
    janus: Janus
    dummy: Dummy
