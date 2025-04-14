from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, le

from ._converters import none_if_none


def valid_mors(instance, attribute, value):
    if instance.module != "mors":
        return

    if (instance.mors.age_now is None) or (instance.mors.age_now <= 0):
        raise ValueError("mors.age_now must be > 0")

    if instance.mors.spec is None:
        raise ValueError("Must provide mors.spec")

    set_pcntle = instance.mors.rot_pcntle is not None
    set_period = instance.mors.rot_period is not None

    if set_pcntle and set_period:
        raise ValueError("Stellar rotation must be set by percentile or period, not both")
    if (not set_pcntle) and (not set_period):
        raise ValueError("Stellar rotation must be set either by percentile or period")

    if set_pcntle and not (0 <= instance.mors.rot_pcntle <= 100):
        raise ValueError("Rotation percentile must be >=0 and <=100")

    if set_period and (instance.mors.rot_period <= 0):
        raise ValueError("Rotation period must be greater than zero")

@define
class Mors:
    """Module parameters for MORS module.

    Attributes
    ----------
    rot_pcntle: float
        Rotation, as percentile of stellar population.
    rot_period: float
        Rotation rate [days].
    tracks: str
        Stellar evolution track to be used. Choices: 'spada', 'baraffe'.
    age_now: float
        Observed estimated age of the star [Gyr].
    spec: str
        Name of file containing stellar spectrum. See [documentation](https://fwl-proteus.readthedocs.io/en/latest/data/#stars) for potential file names.
    """
    age_now         = field(default=None)
    spec            = field(default=None)
    rot_pcntle      = field(default=None, converter=none_if_none)
    rot_period      = field(default=None, converter=none_if_none)
    tracks: str     = field(default='spada', validator=in_(('spada', 'baraffe')))

def valid_stardummy(instance, attribute, value):
    if instance.module != "dummy":
        return
    if (instance.dummy.radius is None) or (instance.dummy.radius <= 0):
        raise ValueError("star.dummy.radius must be > 0")
    if (instance.dummy.Teff is None) or (instance.dummy.Teff <= 0):
        raise ValueError("star.dummy.Teff must be > 0")

@define
class StarDummy:
    """Dummy star module.

    Attributes
    ----------
    radius: float
        Observed radius [R_sun].
    Teff: float
        Observed effective temperature [K].
    """
    radius = field(default=None)
    Teff   = field(default=None)

@define
class Star:
    """Stellar parameters, model selection.

    You can find useful reference data in the [documentation](https://fwl-proteus.readthedocs.io/en/latest/data/#stars).

    Attributes
    ----------
    mass: float
        Stellar mass [M_sun]. Note that for Mors,
        it should be between 0.1 and 1.25 solar masses.
        Values outside of the valid range will be clipped.
    age_ini: float
        Age of system at model initialisation [Gyr].
    module: str | None
        Select star module to use.
    mors: Mors
        Parameters for MORS module.
    dummy: StarDummy
        Parameters for the dummy star module
    """

    module: str | None = field(
        validator=in_((None, 'mors', 'dummy')),
        converter=none_if_none,
    )

    mass: float = field(validator=(ge(0.1), le(1.25)))
    age_ini: float = field(validator=gt(0))

    mors: Mors       = field(factory=Mors,      validator=valid_mors)
    dummy: StarDummy = field(factory=StarDummy, validator=valid_stardummy)
