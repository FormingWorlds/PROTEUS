from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Elements:
    CH_ratio: float
    H_oceans: float
    N_ppmw: float
    S_ppmw: float


@define
class Volatiles:
    H2O: float = field(default=0)
    CO2: float = field(default=0)
    N2: float = field(default=0)
    S2: float = field(default=0)
    SO2: float = field(default=0)
    H2: float = field(default=0)
    CH4: float = field(default=0)
    CO: float = field(default=0)


@define
class Delivery:
    """Initial volatile inventory, and delivery model selection"""
    initial: str = field(validator=validators.in_(('elements', 'volatile')))

    module: str | None = field(validator=validators.in_((None,)), converter=none_if_none)

    elements: Elements
    volatiles: Volatiles
