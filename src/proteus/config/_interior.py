from __future__ import annotations

from attrs import define, field, validators


@define
class Interior:
    """Magma ocean model selection and parameters.

    Attributes
    ----------
    grain_size: float
        Crystal settling grain size [m].
    F_initial: float
        Initial heat flux guess [W m-2].
    module: str
        Module for simulating the magma ocean. Choices: 'spider', 'aragog', 'dummy'.
    spider: Spider
        Parameters for running the SPIDER module.
    aragog: Aragog
        Parameters for running the aragog module.
    """
    grain_size: float
    F_initial: float

    module: str = field(validator=validators.in_(('spider', 'aragog', 'dummy')))

    spider: Spider
    aragog: Aragog


@define
class Spider:
    """Parameters for SPIDER module.

    Attributes
    ----------
    num_levels: int
        Number of SPIDER grid levels.
    mixing_length: int
        Parameterisation used to determine convective mixing length.
    tolerance: float
        Solver tolerance.
    tsurf_atol: float
        Absolute tolerance on change in T_mantle during a single interior iteration.
    tsurf_rtol: float
        Relative tolerance on change in T_mantle during a single interior iteration.
    ini_entropy: float
        Initial specific surface entropy [J K-1 kg-1].
    ini_dsdr: float
        Initial interior specific entropy gradient [J K-1 kg-1 m-1].
    """
    num_levels: int  = field(validator=validators.ge(40))
    mixing_length: int
    tolerance: float
    tsurf_atol: float
    tsurf_rtol: float
    ini_entropy: float
    ini_dsdr: float


@define
class Aragog:
    """Parameters for Aragog module.

    Attributes
    ----------
    some_parameter: str
        Not used.
    """
    some_parameter: str
