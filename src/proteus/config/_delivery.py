from __future__ import annotations

from typing import Literal

from attrs import define


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
    initial: Literal["elements", "volatile"]

    module: Literal["none"]

    elements: Elements
    volatiles: Volatiles
