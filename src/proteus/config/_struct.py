from __future__ import annotations

from typing import Literal

from attrs import define


@define
class Struct:
    mass: float
    radius: float
    corefrac: float

    module: Literal["none"]
