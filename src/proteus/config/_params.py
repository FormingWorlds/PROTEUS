from __future__ import annotations

from typing import Literal

from attrs import define


@define
class OutputParams:
    path: str
    logging: Literal["INFO", "DEBUG", "ERROR", "WARNING"]
    plot_mod: int
    plot_fmt: Literal["pdf", "png"]

@define
class DtProportional:
    propconst: float

@define
class DtAdaptive:
    atol: float
    rtol: float

@define
class TimeStepParams:
    minimum: float
    maximum: float
    initial: float
    starspec: float
    starinst: float
    method: Literal["proportional","adaptive", "maximum"]
    proportional: DtProportional
    adaptive: DtAdaptive

@define
class StopIters:
    enabled: bool
    minimum: int
    maximum: int

@define
class StopTime:
    enabled: bool
    minimum: float
    maximum: float

@define
class StopSolid:
    enabled: bool
    phi_crit: float

@define
class StopRadeqm:
    enabled: bool
    F_crit: float

@define
class StopSteady:
    enabled: bool
    F_crit: float
    dprel: float

@define
class StopEscape:
    enabled: bool
    mass_frac: float

@define
class StopParams:
    iters: StopIters
    time: StopTime
    solid: StopSolid
    radeqm: StopRadeqm
    steady: StopSteady
    escape: StopEscape

@define
class Params:

    out: OutputParams
    dt: TimeStepParams
    stop: StopParams
