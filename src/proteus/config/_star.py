from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Mors:
    tracks: str = field(validator=validators.in_(('spada', 'baraffe')))
    spec: str
    Lbol: float
    omega: float
    age_now: float
    age_ini: float

@define
class StarDummy:
    rate: float

@define
class Star:
    """Stellar parameters, model selection"""
    mass: float
    radius: float
    Teff: float

    module: str | None = field(
        validator=validators.in_((None, 'mors', 'dummy')),
        converter=none_if_none,
    )

    mors: Mors
    dummy: StarDummy
