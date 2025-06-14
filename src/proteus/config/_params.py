"""This module describes the parameters for the data location, data output, and logging.
It also defines stopping criteria."""

from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, lt

from ._converters import none_if_none


def valid_path(instance, attribute, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{attribute.name}' must be a non-empty string")

def max_bigger_than_min(instance, attribute, value):
    if value <= instance.minimum:
        raise ValueError("'maximum' has to be bigger than 'minimum'.")

def valid_mod(instance, attribute, value):
    if value is None:
        return
    if value < 0:
        raise ValueError(f"Parameter '{attribute}' must be None or greater than 0.")

@define
class OutputParams:
    """Parameters for output files and logging

    Attributes
    ----------
    path: str
        Path to output folder relative to `PROTEUS/output/`.
    logging: str
        Log verbosity. Choices: 'INFO', 'DEBUG', 'ERROR', 'WARNING'.
    plot_fmt: str
        Plotting output file format. Choices: "png", "pdf".
    write_mod: int
        Write CSV frequency. 0: wait until completion. n: every n iterations.
    plot_mod: int | None
        Plotting frequency. 0: wait until completion. n: every n iterations. None: never plot.
    archive_mod: int | None
        Archive frequency. 0: wait until completion. n: every n iterations. None: never archive.
    remove_sf: bool
        Remove SOCRATES spectral files after model terminates.
    """
    path: str       = field(validator=valid_path)
    logging: str    = field(default='INFO',
                                    validator=in_(('INFO', 'DEBUG', 'ERROR', 'WARNING')))
    plot_fmt: str   = field(default='png',  validator=in_(('pdf', 'png')))
    write_mod:int   = field(default=1,      validator=ge(0))
    plot_mod        = field(default=10,     validator=valid_mod, converter=none_if_none)
    archive_mod     = field(default=None,   validator=valid_mod, converter=none_if_none)
    remove_sf:bool  = field(default=False)

@define
class DtProportional:
    """Parameters used to configure the proportional time-stepping scheme.

    Attributes
    ----------
    propconst: float
        Proportionality constant.
    """
    propconst: float = field(default=52.0, validator=gt(0))


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
    atol: float = field(default=0.02, validator=gt(0))
    rtol: float = field(default=0.10, validator=gt(0))

@define
class TimeStepParams:
    """Parameters for time-stepping parameters

    Attributes
    ----------
    minimum: float
        Minimum absolute time-step size [yr].
    minimum_rel: float
        Minimum relative time-step size [dimensionless].
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


    starspec: float = field(default=3e6, validator=ge(0))
    starinst: float = field(default=1e3, validator=ge(0))

    method: str = field(default='adaptive',
                        validator=in_(('proportional', 'adaptive', 'maximum')))

    proportional: DtProportional = field(factory=DtProportional)
    adaptive: DtAdaptive         = field(factory=DtAdaptive)

    minimum: float      = field(default=3e2,  validator=gt(0))
    minimum_rel: float  = field(default=1e-6, validator=gt(0))
    maximum: float      = field(default=1e7,  validator=gt(0))
    initial: float      = field(default=1e3,  validator=gt(0))


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
    enabled: bool = field(default=True)
    minimum: int  = field(default=5, validator=ge(0))
    maximum: int  = field(default=9000, validator=max_bigger_than_min)


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
    enabled: bool  = field(default=True)
    maximum: float = field(default=6e9, validator=max_bigger_than_min)
    minimum: float = field(default=1e3, validator=ge(0))


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
    phi_crit: float = field(default=0.01, validator=(gt(0), lt(1)))
    enabled: bool   = field(default=True)


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
    enabled: bool   = field(default=True)
    atol: float     = field(default=1.0,  validator=gt(0))
    rtol: float     = field(default=1e-3, validator=ge(0))


@define
class StopEscape:
    """Parameters for escape stopping criteria.

    Attributes
    ----------
    enabled: bool
        Enable criteria if True
    p_stop: float
        Model will terminate when surface pressure is less than this value [bar].
    """
    enabled: bool   = field(default=True)
    p_stop:float    = field(default=1, validator=(gt(0),lt(1e6)))


@define
class StopParams:
    """Parameters for termination criteria.

    Attributes
    ----------
    strict: bool
        Require termination criteria to be satisfied twice before the model exits.
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
    iters: StopIters   = field(factory=StopIters)
    time: StopTime     = field(factory=StopTime)
    solid: StopSolid   = field(factory=StopSolid)
    radeqm: StopRadeqm = field(factory=StopRadeqm)
    escape: StopEscape = field(factory=StopEscape)

    strict: bool        = field(default=False)

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
    out: OutputParams  = field(factory=OutputParams)
    dt: TimeStepParams = field(factory=TimeStepParams)
    stop: StopParams   = field(factory=StopParams)

    resume: bool       = field(default = False)
    offline: bool      = field(default = False)
