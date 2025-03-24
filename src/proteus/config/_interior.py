from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, lt


def no_radio_if_dummy(instance, attribute, value):
    if (instance.module == 'dummy') and value:
        raise ValueError("Radiogenic heating is not supported by the dummy interior module")


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
    num_levels: int     = field(default=80,         validator=ge(40))
    mixing_length: int  = field(default=2,          validator=in_((1,2)))
    tolerance: float    = field(default=1e-10,      validator=gt(0))
    tsurf_atol: float   = field(default=20.0,       validator=gt(0))
    tsurf_rtol: float   = field(default=0.01,       validator=gt(0))
    ini_entropy: float  = field(default=3300.0,     validator=gt(0))
    ini_dsdr: float     = field(default=-4.698e-6,  validator=lt(0))

@define
class Aragog:
    """Parameters for Aragog module.

    Attributes
    ----------
    num_levels: int
        Number of Aragog grid levels (basic mesh).
    tolerance: float
        Solver tolerance.
    ini_tmagma: float
        Initial magma surface temperature [K].
    """

    num_levels: int     = field(default=200,    validator=ge(40))
    ini_tmagma: float   = field(default=3200.0, validator=gt(0))
    tolerance: float    = field(default=1e-10,  validator=gt(0))

@define
class Dummy:
    """Parameters for Dummy interior module.

    Attributes
    ----------
    ini_tmagma: float
        Initial magma surface temperature [K].
    """

    ini_tmagma: float = field(default=3500.0, validator=gt(0))

@define
class Interior:
    """Magma ocean model selection and parameters.

    Attributes
    ----------
    grain_size: float
        Crystal settling grain size [m].
    F_initial: float
        Initial heat flux guess [W m-2].
    radiogenic_heat: bool
        Include radiogenic heat production?
    tidal_heat: bool
        Include tidal heating?
    rheo_phi_loc: float
        Centre of rheological transition in terms of melt fraction
    rheo_phi_wid: float
        Width of rheological transition in terms of melt fraction
    bulk_modulus: float
        Fixed bulk modulus of each layer [Pa].

    module: str
        Module for simulating the magma ocean. Choices: 'spider', 'aragog', 'dummy'.
    spider: Spider
        Parameters for running the SPIDER module.
    aragog: Aragog
        Parameters for running the aragog module.
    dummy: Dummy
        Parameters for running the dummy module.
    """

    module: str     = field(validator=in_(('spider', 'aragog', 'dummy')))

    radiogenic_heat: bool   = field(validator=no_radio_if_dummy)
    tidal_heat: bool

    spider: Spider  = field(factory=Spider)
    aragog: Aragog  = field(factory=Aragog)
    dummy: Dummy    = field(factory=Dummy)

    grain_size: float       = field(default=0.1,   validator=gt(0))
    F_initial: float        = field(default=1e5,   validator=gt(0))
    rheo_phi_loc: float     = field(default=0.3, validator=(gt(0),lt(1)))
    rheo_phi_wid: float     = field(default=0.15, validator=(gt(0),lt(1)))
    bulk_modulus: float     = field(default=260e9, validator=gt(0))


