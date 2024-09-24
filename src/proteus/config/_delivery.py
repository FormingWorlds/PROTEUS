from __future__ import annotations

from attrs import define, field, validators


@define
class Elements:
    CH_ratio: float
    H_oceans: float
    N_ppmw: float
    S_ppmw: float

@define
class Volatiles:
    H2O: float
    CO2: float
    N2: float
    S2: float
    SO2: float
    H2: float
    CH4: float
    CO: float

@define
class Delivery:
    initial: str = field(validator=validators.in_(("elements", "volatile")))

    module: str = field(validator=validators.in_(("none",)))

    elements: Elements
    volatiles: Volatiles
