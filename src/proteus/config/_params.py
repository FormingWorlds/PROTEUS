"""This module describes the parameters for the data location, data output, and logging.
It also defines stopping criteria."""

from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, lt


def max_bigger_than_min(instance, attribute, value):
    if value <= instance.minimum:
        raise ValueError("'maximum' has to be bigger than 'minimum'.")


@define
class Params:
    """Parameters for code execution, output files, time-stepping, convergence.

    Attributes
    ----------
    out: OutputParams
        Parameters for data / logging output.
    dt: TimeStepParams
        Parameters for time-stepping.
    stop: StopParams
        Parameters for stopping criteria.
    """
    out: OutputParams
    dt: TimeStepParams
    stop: StopParams

    resume: bool = field(default = False)
    offline: bool = field(default = False)


@define
class OutputParams:
    """Parameters for output files and logging

    Attributes
    ----------
    path: str
        Path to output folder relative to `PROTEUS/output/`.
    logging: str
        Log verbosity. Choices: 'INFO', 'DEBUG', 'ERROR', 'WARNING'.
    plot_mod: int
        Plotting frequency. 0: wait until completion. n: every n iterations.
    plot_fmt: str
        Plotting output file format. Choices: "png", "pdf".
    """
    path: str
    logging: str = field(validator=in_(('INFO', 'DEBUG', 'ERROR', 'WARNING')))
    plot_mod: int = field(validator=ge(0))
    plot_fmt: str = field(validator=in_(('pdf', 'png')))


@define
class TimeStepParams:
    """Parameters for time-stepping parameters

    Attributes
    ----------
    minimum: float
        Minimum time-step size [yr].
    maximum: float
        Maximum time-step size [yr].
    initial: float
        Initial time-step size [yr].
    starspec: float
        Maximum interval at which to recalculate the stellar spectrum [yr].
    starinst: float
        Maximum interval at which to recalculate instellation flux [yr].
    method: str
        Time-stepping method. Choices: 'proportional', 'adaptive', 'maximum'.
    proportional: DtProportional
        Parameters used to configure the proportional time-stepping scheme.
    adaptive: DtAdaptive
        Parameters used to configure the adaptive time-stepping scheme.
    """
    minimum: float = field(validator=gt(0))
    maximum: float = field(validator=gt(0))
    initial: float = field(validator=gt(0))
    starspec: float = field(validator=ge(0))
    starinst: float = field(validator=ge(0))
    method: str = field(validator=in_(('proportional', 'adaptive', 'maximum')))
    proportional: DtProportional
    adaptive: DtAdaptive


@define
class DtProportional:
    """Parameters used to configure the proportional time-stepping scheme.

    Attributes
    ----------
    propconst: float
        Proportionality constant.
    """
    propconst: float = field(validator=gt(0))


@define
class DtAdaptive:
    """Parameters used to configure the adaptive time-stepping scheme.

    Attributes
    ----------
    atol: float
        Absolute tolerance on time-step size [yr].
    rtol: float
        Relative tolerance on time-step size [dimensionless].
    """
    atol: float = field(validator=gt(0))
    rtol: float = field(validator=gt(0))


@define
class StopParams:
    """Parameters for termination criteria.

    Attributes
    ----------
    iters: StopIters
        Parameters for iteration number criteria.
    time: StopTime
        Parameters for maximum time criteria.
    solid: StopSolid
        Parameters for solidification criteria.
    radeqm: StopRadeqm
        Parameters for radiative equilibrium criteria.
    escape: StopEscape
        Parameters for escape criteria.
    """
    iters: StopIters
    time: StopTime
    solid: StopSolid
    radeqm: StopRadeqm
    escape: StopEscape


@define
class StopIters:
    """Parameters for iteration number criteria.

    Attributes
    ----------
    enabled: bool
        Enable criteria if True
    minimum: int
        Minimum number of iterations.
    maximum: int
        Maximum number of iterations.
    """
    enabled: bool
    minimum: int = field(validator=ge(0))
    maximum: int = field(validator=max_bigger_than_min)


@define
class StopTime:
    """Parameters for maximum time criteria.

    Attributes
    ----------
    enabled: bool
        Enable criteria if True
    minimum: float
        Model will absolutely not terminate until at least this time is reached [yr].
    maximum: float
        Model will terminate when this time is reached [yr].
    """
    enabled: bool
    minimum: float = field(validator=ge(0))
    maximum: float = field(validator=max_bigger_than_min)


@define
class StopSolid:
    """Parameters for solidification criteria.

    Attributes
    ----------
    enabled: bool
        Enable criteria if True.
    phi_crit: float
        Model will terminate when global melt fraction is less than this value [dimensionless].
    """
    enabled: bool
    phi_crit: float = field(validator=(gt(0), lt(1)))


@define
class StopRadeqm:
    """Parameters for radiative equilibrium stopping criteria.

    Attributes
    ----------
    enabled: bool
        Enable criteria if True
    atol: float
        Absolute tolerance on energy balance [W m-2].
    rtol: float
        Relative tolerance on energy balance.
    """
    enabled: bool
    atol: float = field(validator=gt(0))
    rtol: float = field(validator=ge(0))


@define
class StopEscape:
    """Parameters for escape stopping criteria.

    Attributes
    ----------
    enabled: bool
        Enable criteria if True
    mass_frac: float
        Model will termiante when atmosphere mass is less than this fraction of the initial atmosphere mass [dimensionless].
    """
    enabled: bool
    mass_frac: float = field(validator=(gt(0), lt(1)))
