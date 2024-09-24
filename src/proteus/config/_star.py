from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Mors:
    tracks: str = field(validator=validators.in_(('spada', 'baraffe')))
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

    module: str | None = field(
        validator=validators.in_((None, 'mors')),
        converter=none_if_none,
    )

    mors: Mors
