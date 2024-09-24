from __future__ import annotations

from attrs import define, field, validators


@define
class Agni:
    p_top: float
    spectral_group: str
    spectral_bands: str
    num_levels: int
    chemistry: str = field(validator=validators.in_(("none", "eq", "kin")))
    tmp_minimum: float

@define
class Janus:
    p_top: float
    spectral_group: str
    spectral_bands: str
    F_atm_bc: int = field(validator=validators.in_((0, 5)))
    num_levels: int
    tmp_minimum: float
    tropopause: str  = field(validator=validators.in_(("none", "skin", "dynamic")))

@define
class Dummy:
    gamma: float

@define
class Atmos:
    prevent_warming: bool
    surface_d: float
    surface_k: float
    cloud_enabled: bool
    cloud_alpha: float
    surf_state: str  = field(validator=validators.in_(("mixed_layer", "fixed", "skin")))
    surf_albedo: float
    rayleigh: float

    module: str  = field(validator=validators.in_(("dummy", "agni", "janus")))

    agni: Agni
    janus: Janus
    dummy: Dummy
