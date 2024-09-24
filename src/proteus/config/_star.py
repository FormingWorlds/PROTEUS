from __future__ import annotations

from typing import Literal

from attrs import define, field, validators


@define
class Mors:
    tracks: str = field(validator=validators.in_(("spada", "baraffe")))
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
