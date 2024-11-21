from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, le

from ._converters import none_if_none


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
    mass: float = field(validator=(ge(0.1), le(1.25)))
    age_ini: float = field(validator=gt(0))

    module: str | None = field(
        validator=in_((None, 'mors', 'dummy')),
        converter=none_if_none,
    )

    mors: Mors
    dummy: StarDummy


@define
class Mors:
    """Module parameters for MORS module.

    Attributes
    ----------
    rot_pctle: float
        Rotation rate, as a percentile of stellar population with the same mass [%].
    tracks: str
        Stellar evolution track to be used. Choices: 'spada', 'baraffe'.
    age_now: float
        Observed estimated age of the star [Gyr].
    spec: str
        Name of file containing stellar spectrum. See [documentation](https://fwl-proteus.readthedocs.io/en/latest/data/#stars) for potential file names.
    """
    rot_pctle: float = field(validator=(ge(0), le(100)))
    tracks: str = field(validator=in_(('spada', 'baraffe')))
    age_now: float = field(validator=gt(0))
    spec: str

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
    radius: float = field(validator=gt(0))
    Teff: float = field(validator=gt(0))
