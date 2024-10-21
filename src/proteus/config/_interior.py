from __future__ import annotations

from attrs import define, field, validators


@define
class Spider:
    """Parameters for Spider module.

    Attributes
    ----------
    num_levels: int
        Number of SPIDER grid levels
    mixing_length: int
        Mixing length parameterization
    tolerance: float
        Solver tolerance
    tsurf_atol: float
        Tsurf_poststep_change
    tsurf_rtol: float
        Tsurf_poststep_change_frac
    ini_entropy: float
        Surface entropy conditions [J K-1 kg-1]
    ini_dsdr: float
        Interior entropy gradient [J K-1 kg-1 m-1]
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


@define
class Interior:
    """Magma ocean model selection and parameters.

    Attributes
    ----------
    grain_size: float
        Crystal settling grain size [m]
    F_initial: float
        Initial heat flux guess [W m-2]
    module: str
        Select interior model, choices: 'spider', 'aragog', 'dummy'
    spider: Spider
        Parameters for spider module
    aragog: Aragog
        Parameters for aragog module
    """
    grain_size: float
    F_initial: float

    module: str = field(validator=validators.in_(('spider', 'aragog', 'dummy')))

    spider: Spider
    aragog: Aragog
