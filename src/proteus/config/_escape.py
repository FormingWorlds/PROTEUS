from __future__ import annotations

from typing import Literal

from attrs import define


@define
class Zephyrus:
    some_parameter: str

@define
class EscapeDummy:
    rate: float

@define
class Escape:
    module: Literal["dummy", "zephyrus"]

    zephyrus: Zephyrus
    dummy: EscapeDummy
