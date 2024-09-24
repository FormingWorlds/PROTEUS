from __future__ import annotations

from typing import Literal

from attrs import define


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

    module: Literal["spider", "aragog"]

    spider: Spider
    aragog: Aragog
