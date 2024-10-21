"""This module describes the parameters for the data location, data output, and logging.
It also defines stopping criteria."""

from __future__ import annotations

from attrs import define, field, validators


@define
class Params:
    """Parameters for code execution, output files, time-stepping, convergence.

    Attributes
    ----------
    out: OutputParams
        Parameters for data / logging output
    dt: TimeStepParams
        Parameters for time-stepping
    stop: StopParams
        Parameters for stopping criteria
    """
    out: OutputParams
    dt: TimeStepParams
    stop: StopParams


@define
class OutputParams:
    """Parameters for output files and logging

    Attributes
    ----------
    path: str
        Path where to store output data
    logging: str
        Set loglevel, choices: 'INFO', 'DEBUG', 'ERROR', 'WARNING'
    plot_mod: int
        Plotting frequency, 0: wait until completion | n: every n iterations
    plot_fmt: str
        Plotting image file format, "png" or "pdf" recommended
    """
    path: str
    logging: str = field(validator=validators.in_(('INFO', 'DEBUG', 'ERROR', 'WARNING')))
    plot_mod: int
    plot_fmt: str = field(validator=validators.in_(('pdf', 'png')))


@define
class TimeStepParams:
    """Parameters for time-stepping parameters

    Attributes
    ----------
    minimum: float
        yr, minimum time-step
    maximum: float
        yr, maximum time-step
    initial: float
        yr, inital step size
    starspec: float
        yr, interval to re-calculate the stellar spectrum
    starinst: float
        yr, interval to re-calculate the instellation
    method: str
        Time-stepping method, choices: 'proportional', 'adaptive', 'maximum'
    proportional: DtProportional
        Parameters for proportional method
    adaptive: DtAdaptive
        Parameters for adaptive method
    """
    minimum: float
    maximum: float
    initial: float
    starspec: float
    starinst: float
    method: str = field(validator=validators.in_(('proportional', 'adaptive', 'maximum')))
    proportional: DtProportional
    adaptive: DtAdaptive


@define
class DtProportional:
    """Parameters for proportional time-stepping

    Attributes
    ----------
    propconst: float
        Proportionality constant.
    """
    propconst: float


@define
class DtAdaptive:
    """Parameters for adaptive time-stepping

    Attributes
    ----------
    atol: float
        Step size absolute tolerance
    rtol: float
        Step size relative tolerance
    """
    atol: float
    rtol: float


@define
class StopParams:
    """Parameters for termination criteria.

    Attributes
    ----------
    iters: StopIters
        Parameters for required number of iterations.
    time: StopTime
        Parameters for required time constraints.
    solid: StopSolid
        Parameters for solidification.
    radeqm: StopRadeqm
        Parameters for radiative equilibrium.
    steady: StopSteady
        Parameters for steady state.
    escape: StopEscape
        Parameters for escape.
    """
    iters: StopIters
    time: StopTime
    solid: StopSolid
    radeqm: StopRadeqm
    steady: StopSteady
    escape: StopEscape


@define
class StopIters:
    """Parameters for iteration stopping criteria.

    Attributes
    ----------
    enabled: bool
        Enable criterium if True
    minimum: int
        Minimum number of iterations
    maximum: int
        Maximum number of iterations
    """
    enabled: bool
    minimum: int
    maximum: int


@define
class StopTime:
    """Parameters for time constraints stopping criteria.

    Attributes
    ----------
    enabled: bool
        Enable criterium if True
    minimum: float
        yr, model will certainly run to t > minimum
    maximum: float
        yr, model will terminate when t > maximum
    """
    enabled: bool
    minimum: float
    maximum: float


@define
class StopSolid:
    """Parameters for solidification stopping criteria.

    Attributes
    ----------
    enabled: bool
        Enable criterium if True.
    phi_crit: float
        Non-dimensional, model will terminate when global melt fraction < phi_crit.
    """
    enabled: bool
    phi_crit: float


@define
class StopRadeqm:
    """Parameters for radiative equilibrium stopping criteria.

    Attributes
    ----------
    enabled: bool
        Enable criterium if True
    F_crit: float
        W m-2, model will terminate when |F_atm| < F_crit
    """
    enabled: bool
    F_crit: float


@define
class StopSteady:
    """Parameters for steady state stopping criteria.

    Attributes
    ----------
    enabled: bool
        Enable criterium if True
    F_crit: float
        Maximum absolute value of F_atm allowed for convergence
    dprel: float
        Percentage change in melt fraction over time (dp/p)/dt*100
    """
    enabled: bool
    F_crit: float
    dprel: float


@define
class StopEscape:
    """Parameters for escape stopping criteria.

    Attributes
    ----------
    enabled: bool
        Enable criterium if True
    mass_frac: float
        Stop when atm_mass < this frac of initial mass
    """
    enabled: bool
    mass_frac: float
