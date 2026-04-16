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
        Output folder name inside ``PROTEUS/output/``. Set to ``"auto"``
        (default) for a unique timestamped name (``run_YYYYMMDD_HHMMSS_xxxx``),
        or any string for a fixed folder (e.g. ``"my_earth_run"``).
    logging: str
        Log verbosity. Choices: 'INFO', 'DEBUG', 'ERROR', 'WARNING'.
    plot_fmt: str
        Plotting output file format. Choices: "png", "pdf".
    write_mod: int
        Write CSV frequency. 0: wait until completion. n: every n iterations.
    dt_write_rel: float
        Minimum elapsed simulation time between data writes, expressed as a
        fraction of the current simulation time. The effective minimum write
        interval is ``dt_write_rel * Time``. This gives logarithmic spacing:
        at Time=1e3 yr with dt_write_rel=1e-3 the guard is 1 yr; at
        Time=1e9 yr it is 1e6 yr. Set to 0 to write every time write_mod
        triggers (default, preserving existing behaviour).
    plot_mod: int | None
        Plotting frequency. 0: wait until completion. n: every n iterations. None: never plot.
    archive_mod: int | None
        Archive frequency. 0: wait until completion. n: every n iterations. None: never archive.
    remove_sf: bool
        Remove SOCRATES spectral files after model terminates.
    """

    path: str = field(default='auto', validator=valid_path)
    logging: str = field(default='INFO', validator=in_(('INFO', 'DEBUG', 'ERROR', 'WARNING')))
    plot_fmt: str = field(default='png', validator=in_(('pdf', 'png')))
    write_mod: int = field(default=1, validator=ge(0))
    dt_write_rel: float = field(default=0.0, validator=ge(0))
    # Type hint includes `str` so cattrs can structure the literal string
    # "none" before the `none_if_none` converter maps it to Python None.
    # Without `str` in the union, `int("none")` raises ValueError at
    # structure time. Same pattern as star._star.Mors.rot_period.
    plot_mod: int | str | None = field(default=5, validator=valid_mod, converter=none_if_none)
    archive_mod: int | str | None = field(
        default=None, validator=valid_mod, converter=none_if_none
    )
    remove_sf: bool = field(default=False)


@define
class TimeStepParams:
    """Parameters for time-stepping.

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
    propconst: float
        Proportionality constant (proportional method).
    atol: float
        Absolute tolerance on time-step size (adaptive method) [yr].
    rtol: float
        Relative tolerance on time-step size (adaptive method) [dimensionless].
    mushy_maximum: float
        Maximum time-step size [yr] during the mushy-zone transition
        (``phi_crit < Phi_global < mushy_upper``). Tighter than
        ``maximum`` because the interior solver hits stiffness
        cliffs in this regime (phase-boundary Jgrav + rheology
        contrast). Set to 0 (default) to disable the mushy-regime
        cap, in which case ``maximum`` applies throughout. A
        typical value for Aragog at 1 M_E is ~4e3 yr; see
        ``input/chili/evolution-proteus-earth-aragog-config.toml``.
    mushy_upper: float
        Upper bound of the mushy regime [dimensionless melt
        fraction]. When ``Phi_global < mushy_upper`` AND
        ``Phi_global > stop.solid.phi_crit``, ``mushy_maximum``
        takes over from ``maximum``. Default 0.99 so the cap kicks
        in as soon as the first cell crystallises.
    hysteresis_iters: int
        Number of PROTEUS iterations after an adaptive "slow down"
        decision during which the speed-up factor is suppressed.
        Prevents the controller from ramping dt straight back into
        the same stiffness cliff it just escaped from. Default 3;
        set to 0 to disable.
    hysteresis_sfinc: float
        Replacement speed-up factor applied while the hysteresis
        counter is active. Must be ``>= 1.0`` and ``<= SFINC``
        (1.6). Default 1.1 (gentle ramp-up).
    """

    starspec: float = field(default=1e8, validator=ge(0))
    starinst: float = field(default=1e2, validator=ge(0))

    method: str = field(
        default='adaptive', validator=in_(('proportional', 'adaptive', 'maximum'))
    )

    propconst: float = field(default=52.0, validator=gt(0))
    atol: float = field(default=0.02, validator=gt(0))
    rtol: float = field(default=0.10, validator=gt(0))

    minimum: float = field(default=1e4, validator=gt(0))
    minimum_rel: float = field(default=1e-5, validator=gt(0))
    maximum: float = field(default=1e7, validator=gt(0))
    initial: float = field(default=3e1, validator=gt(0))

    # Stiffness-aware adaptive time-stepping extensions.
    # Defaults OFF (mushy_maximum=0, hysteresis_iters=0) for
    # backwards compatibility; enable via positive config values.
    mushy_maximum: float = field(default=0.0, validator=ge(0))
    mushy_upper: float = field(default=0.99, validator=(gt(0), lt(1)))
    hysteresis_iters: int = field(default=0, validator=ge(0))
    hysteresis_sfinc: float = field(default=1.1, validator=ge(1.0))


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
    minimum: int = field(default=5, validator=ge(0))
    maximum: int = field(default=9000, validator=max_bigger_than_min)


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

    enabled: bool = field(default=True)
    maximum: float = field(default=6e9, validator=max_bigger_than_min)
    minimum: float = field(default=1e3, validator=ge(0))


@define
class StopSolid:
    """Parameters for solidification criteria.

    Attributes
    ----------
    enabled: bool
        Enable termination at solidification if True.
    phi_crit: float
        Model will terminate (if enabled) or freeze volatiles (if
        freeze_volatiles) when global melt fraction drops below this value.
    freeze_volatiles: bool
        When True, outgassing stops at crystallization (Phi_global < phi_crit)
        but the simulation continues. Dissolved volatiles are trapped in the
        solid mantle and preserved in the helpfile. The atmosphere retains
        its current composition. When False, outgassing continues regardless
        of melt fraction. Default True.
    """

    phi_crit: float = field(default=0.01, validator=(gt(0), lt(1)))
    enabled: bool = field(default=True)
    freeze_volatiles: bool = field(default=False)


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

    enabled: bool = field(default=True)
    atol: float = field(default=1.0, validator=gt(0))
    rtol: float = field(default=1e-3, validator=ge(0))


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

    enabled: bool = field(default=True)
    p_stop: float = field(default=3.0, validator=(gt(0), lt(1e6)))


@define
class StopDisint:
    """Parameters for planet disintegration stopping criteria.

    Attributes
    ----------
    enabled: bool
        Enable all planet disintegration criteria if True
    roche_enabled: bool
        Disable Roche limit criterion
    offset_roche: float
        Absolute correction (+/-) to (increase/decrease) calculated Roche limit [m].
    spin_enabled: bool
        Disable Breakup period criterion
    offset_spin: float
        Absolute correction (+/-) to (increase/decrease) calculated Breakup period [s].
    """

    enabled: bool = field(default=False)

    roche_enabled: bool = field(default=True)
    offset_roche: float = field(default=0)

    spin_enabled: bool = field(default=True)
    offset_spin: float = field(default=0)


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
    disint: StopDisint
        Parameters for planet disintegration criteria.
    """

    iters: StopIters = field(factory=StopIters)
    time: StopTime = field(factory=StopTime)
    solid: StopSolid = field(factory=StopSolid)
    radeqm: StopRadeqm = field(factory=StopRadeqm)
    escape: StopEscape = field(factory=StopEscape)
    disint: StopDisint = field(factory=StopDisint)

    strict: bool = field(default=False)


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

    out: OutputParams = field(factory=OutputParams)
    dt: TimeStepParams = field(factory=TimeStepParams)
    stop: StopParams = field(factory=StopParams)

    resume: bool = field(default=False)
    offline: bool = field(default=False)
