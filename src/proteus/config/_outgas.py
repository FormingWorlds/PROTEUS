from __future__ import annotations

from attrs import define, field, validators


@define
class Calliope:
    include_H2O: bool
    include_CO2: bool
    include_N2: bool
    include_S2: bool
    include_SO2: bool
    include_H2: bool
    include_CH4: bool
    include_CO: bool


@define
class Atmodeller:
    some_parameter: str


@define
class Outgas:
    fO2_shift_IW: float

    module: str = field(validator=validators.in_(('calliope', 'atmodeller')))

    calliope: Calliope
    atmodeller: Atmodeller
