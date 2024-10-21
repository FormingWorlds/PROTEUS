from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


@define
class Mors:
    """Module parameters for MORS module.

    Attributes
    ----------
    tracks: str
        Name of evolution tracks, choice: 'spada', 'baraffe'
    spec: str
        Path to stellar spectrum
    """
    tracks: str = field(validator=validators.in_(('spada', 'baraffe')))
    spec: str


@define
class Star:
    """Stellar parameters, model selection.

    Attributes
    ----------
    mass: float
        M_sun
    radius: float
        R_sun
    Teff: float
        K
    Lbol: float
        L_sun
    omega: float
        Rotation percentile
    age_now: float
        Gyr, current age of star used for scaling
    age_ini: float
        Gyr, model initialisation/start age
    module: str | None
        Select star module to use.
    mors: Mors
        Parameters for MORS module
    """
    mass: float
    radius: float
    Teff: float
    Lbol: float
    omega: float
    age_now: float
    age_ini: float

    module: str | None = field(
        validator=validators.in_((None, 'mors')),
        converter=none_if_none,
    )

    mors: Mors
