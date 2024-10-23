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
    radius: float
        Observed radius [R_sun].
    Teff: float
        Observed effective temperature [K].
    mass: float
        Stellar mass [M_sun]. Note that for Mors,
        it should be between 0.1 and 1.25 solar masses.
        Values outside of the valid range will be clipped.
    lum_now: float
        Observed bolometric luminosity [L_sun].
    omega: float
        Rotation rate, as a percentile of stellar population with the same mass [%].
    age_now: float
        Observed estimated age of the star [Gyr].
    age_ini: float
        Age of star at model initialisation [Gyr].
    module: str | None
        Select star module to use.
    mors: Mors
        Parameters for MORS module.
    """
    radius: float = field(validator=gt(0))
    mass: float = field(validator=(ge(0.1), le(1.25)))
    Teff: float = field(validator=gt(0))
    omega: float = field(validator=(ge(0), le(100)))
    lum_now: float = field(validator=gt(0))
    age_now: float = field(validator=gt(0))
    age_ini: float = field(validator=gt(0))

    module: str | None = field(
        validator=in_((None, 'mors', 'dummy')),
        converter=none_if_none,
    )

    mors: Mors


@define
class Mors:
    """Module parameters for MORS module.

    Attributes
    ----------
    tracks: str
        Stellar evolution track to be used. Choices: 'spada', 'baraffe'.
    spec: str
        Name of file containing stellar spectrum. See [documentation](https://fwl-proteus.readthedocs.io/en/latest/data/#stars) for potential file names.
    """
    tracks: str = field(validator=in_(('spada', 'baraffe')))
    spec: str
