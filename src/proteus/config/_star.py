from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Star:
    """Stellar parameters, model selection.

    Attributes
    ----------
    radius: float
        Observed radius [R_sun].
    Teff: float
        Observed effective temperature [K].
    mass: float
        Stellar mass [M_sun]
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
    radius: float
    mass: float
    Teff: float
    omega: float
    lum_now: float
    age_now: float
    age_ini: float

    module: str | None = field(
        validator=validators.in_((None, 'mors', 'dummy')),
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
        Name of file containing stellar spectrum. See
    """
    tracks: str = field(validator=validators.in_(('spada', 'baraffe')))
    spec: str
