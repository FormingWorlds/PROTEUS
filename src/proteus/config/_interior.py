from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, lt


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
    grain_size: float = field(validator=gt(0))
    F_initial: float = field(validator=gt(0))

    module: str = field(validator=in_(('spider', 'aragog', 'dummy')))

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
    num_levels: int  = field(validator=ge(40))
    mixing_length: int  = field(validator=in_((1,2)))
    tolerance: float = field(validator=gt(0))
    tsurf_atol: float = field(validator=gt(0))
    tsurf_rtol: float = field(validator=gt(0))
    ini_entropy: float = field(validator=gt(0))
    ini_dsdr: float = field(validator=lt(0))

@define
class Dummy:
    """Parameters for Dummy interior module.

    Attributes
    ----------
    mass: float
        Mass of planet's interior. Units of [M_earth].
    """
    mass: float = field(validator=gt(0))


@define
class Aragog:
    """Parameters for Aragog module.

    Attributes
    ----------
    num_levels: int
        Number of Aragog grid levels (basic mesh).
    tolerance: float
        Solver tolerance.
    """
    num_levels: int  = field(validator=ge(40))
    tolerance: float
