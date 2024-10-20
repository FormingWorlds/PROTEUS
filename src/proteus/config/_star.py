from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Mors:
    tracks: str = field(validator=validators.in_(('spada', 'baraffe')))
    spec: str


@define
class StarDummy:
    lowbreak: float

@define
class Star:
    """Stellar parameters, model selection"""
    radius: float
    mass: float
    Teff: float
    omega: float
    lum_now: float
    age_now: float
    age_ini: float

    module: str | None = field(
        validator=validators.in_((None, 'mors', 'dummy')),
        converter=none_if_none,
    )

    mors: Mors
    dummy: StarDummy
