from __future__ import annotations

from attrs import define, field, validators


@define
class Spider:
    num_levels: int  = field(validator=validators.ge(40))
    mixing_length: int
    tolerance: float
    tsurf_atol: float
    tsurf_rtol: float
    ini_entropy: float
    ini_dsdr: float


@define
class Aragog:
    some_parameter: str


@define
class Interior:
    """Magma ocean model selection and parameters"""
    grain_size: float
    F_initial: float

    module: str = field(validator=validators.in_(('spider', 'aragog', 'dummy')))

    spider: Spider
    aragog: Aragog
