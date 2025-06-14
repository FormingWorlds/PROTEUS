from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, lt


def valid_spider(instance, attribute, value):
    if instance.module != "spider":
        return

    ini_entropy = instance.spider.ini_entropy
    if (not ini_entropy) or (ini_entropy <= 200.0) :
        raise ValueError("`interior.spider.ini_entropy` must be >200")

def valid_path(instance, attribute, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{attribute.name}' must be a non-empty string")

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
        Absolute solver tolerance.
    tolerance_rel: float
        Relative solver tolerance.
    tsurf_atol: float
        Absolute tolerance on change in T_mantle during a single interior iteration.
    tsurf_rtol: float
        Relative tolerance on change in T_mantle during a single interior iteration.
    ini_entropy: float
        Initial specific surface entropy [J K-1 kg-1].
    ini_dsdr: float
        Initial interior specific entropy gradient [J K-1 kg-1 m-1].
    solver_type: str
        Numerical integrator. Choices: 'adams', 'bdf'.
    """
    ini_entropy          = field(default=None)
    ini_dsdr: float      = field(default=-4.698e-6,  validator=lt(0))
    num_levels: int      = field(default=190,        validator=ge(40))
    mixing_length: int   = field(default=2,          validator=in_((1,2)))
    tolerance: float     = field(default=1e-10,      validator=gt(0))
    tolerance_rel: float = field(default=1e-10,      validator=gt(0))
    solver_type: str     = field(default="bdf",      validator=in_(("adams", "bdf")))
    tsurf_atol: float    = field(default=10.0,       validator=gt(0))
    tsurf_rtol: float    = field(default=0.01,       validator=gt(0))


def valid_aragog(instance, attribute, value):
    if instance.module != "aragog":
        return

    ini_tmagma = instance.aragog.ini_tmagma
    if (not ini_tmagma) or (ini_tmagma <= 200.0) :
        raise ValueError("`interior.aragog.ini_tmagma` must be >200")

@define
class Aragog:
    """Parameters for Aragog module.

    Attributes
    ----------
    logging: str
        Log verbosity of Aragog. Choices: 'INFO', 'DEBUG', 'ERROR', 'WARNING'.
    num_levels: int
        Number of Aragog grid levels (basic mesh).
    tolerance: float
        Solver tolerance.
    ini_tmagma: float
        Initial magma surface temperature [K].
    inner_boundary_condition: int
        Type of inner boundary condition. Choices:  1 (core cooling), 2 (prescribed heat flux), 3 (prescribed temperature).
    inner_boundary_value: float
        Value of the inner boundary condition, either temperature or heat flux, depending on the chosen condition.
    conduction: bool
        Whether to include conductive heat transfer in the model. Default is True.
    convection: bool
        Whether to include convective heat transfer in the model. Default is True.
    gravitational_separation: bool
        Whether to include gravitational separation in the model. Default is False.
    mixing: bool
        Whether to include mixing in the model. Default is False.
    tsurf_poststep_change: float
        Maximum change in surface temperature allowed during a single interior iteration [K].
    event_triggering: bool
        Whether to include event triggering in the solver. Default is True.
    """

    logging: str                        = field(default='ERROR',validator=in_(('INFO', 'DEBUG', 'ERROR', 'WARNING')))
    ini_tmagma                          = field(default=None)
    num_levels: int                     = field(default=100,    validator=ge(40))
    tolerance: float                    = field(default=1e-10,  validator=gt(0))
    inner_boundary_condition: int       = field(default=1, validator=ge(0))
    inner_boundary_value:float          = field(default=4000, validator=ge(0))
    conduction: bool                    = field(default=True)
    convection: bool                    = field(default=True)
    gravitational_separation: bool      = field(default=False)
    mixing: bool                        = field(default=False)
    tsurf_poststep_change: float        = field(default=30, validator=ge(0))
    event_triggering:bool               = field(default=True)


def valid_interiordummy(instance, attribute, value):
    if instance.module != "dummy":
        return

    ini_tmagma = instance.dummy.ini_tmagma
    if (not ini_tmagma) or (ini_tmagma <= 200.0) :
        raise ValueError("`interior.dummy.ini_tmagma` must be >200")

    if instance.radiogenic_heat:
        raise ValueError("Dummy interior module does not support radiogenic heating")

@define
class InteriorDummy:
    """Parameters for Dummy interior module.

    Attributes
    ----------
    ini_tmagma: float
        Initial magma surface temperature [K].
    """

    ini_tmagma = field(default=None)

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
    melting_dir: str
        Set of melting curves to use in the model.
    """

    module: str             = field(validator=in_(('spider', 'aragog', 'dummy')))
    melting_dir: str        = field(default="Monteux-600",validator=valid_path)
    radiogenic_heat: bool   = field(default=True)
    tidal_heat: bool        = field(default=True)

    spider: Spider          = field(factory=Spider,        validator=valid_spider)
    aragog: Aragog          = field(factory=Aragog,        validator=valid_aragog)
    dummy: InteriorDummy    = field(factory=InteriorDummy, validator=valid_interiordummy)

    grain_size: float       = field(default=0.1,    validator=gt(0))
    F_initial: float        = field(default=1e3,    validator=gt(0))
    rheo_phi_loc: float     = field(default=0.3,    validator=(gt(0),lt(1)))
    rheo_phi_wid: float     = field(default=0.15,   validator=(gt(0),lt(1)))
    bulk_modulus: float     = field(default=260e9,  validator=gt(0))
