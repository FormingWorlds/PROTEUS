from __future__ import annotations

from typing import Literal

from attrs import define


@define
class Agni:
    p_top: float
    spectral_group: str
    spectral_bands: str
    num_levels: int
    chemistry: Literal["none", "eq", "kin"]
    tmp_minimum: float

@define
class Janus:
    p_top: float
    spectral_group: str
    spectral_bands: str
    F_atm_bc: Literal[0, 5]
    num_levels: int
    tmp_minimum: float
    tropopause: Literal["none", "skin", "dynamic"]

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
    surf_state: Literal["mixed_layer", "fixed", "skin"]
    surf_albedo: float
    rayleigh: float

    module: Literal["dummy", "agni", "janus"]

    agni: Agni
    janus: Janus
    dummy: Dummy
