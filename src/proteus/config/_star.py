from __future__ import annotations

from typing import Literal

from attrs import define


@define
class Mors:
    tracks: Literal["spada", "baraffe"]
    spec: str

@define
class Star:
    mass: float
    radius: float
    Teff: float
    Lbol: float
    omega: float
    age_now: float
    age_ini: float

    module: Literal["none", "mors"]

    mors: Mors
