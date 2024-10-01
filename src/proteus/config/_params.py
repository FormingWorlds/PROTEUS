from __future__ import annotations

from attrs import define, field, validators


@define
class OutputParams:
    path: str
    logging: str = field(validator=validators.in_(('INFO', 'DEBUG', 'ERROR', 'WARNING')))
    plot_mod: int
    plot_fmt: str = field(validator=validators.in_(('pdf', 'png')))


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
    method: str = field(validator=validators.in_(('proportional', 'adaptive', 'maximum')))
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
    """Parameters for code execution, output files, time-stepping, convergence"""
    out: OutputParams
    dt: TimeStepParams
    stop: StopParams
