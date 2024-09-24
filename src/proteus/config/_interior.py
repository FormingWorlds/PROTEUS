from __future__ import annotations

from attrs import define, field, validators


@define
class Spider:
    num_levels: int
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
    grain_size: float
    F_initial: float

    module: str = field(validator=validators.in_(('spider', 'aragog')))

    spider: Spider
    aragog: Aragog
