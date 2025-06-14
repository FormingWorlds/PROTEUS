from __future__ import annotations

from attrs import define, field
from attrs.validators import ge, gt, in_, lt

from ._converters import none_if_none


@define
class Platon:
    """Parameters for the PLATON module.

    Attributes
    ----------
    downsample: int
        Downsample binning factor for the spectrum.
    clip_vmr: float
        Minimum VMR for a species to be included in the radiative transfer.
    """

    downsample: int   = field(default=8, validator=ge(1))
    clip_vmr: float   = field(default=1e-8, validator=(gt(0), lt(1)))

@define
class Observe:
    """Synthetic observations.

    synthesis: str
        Module to use for calculating synthetic spectra.
    """

    synthesis: str  = field(validator=in_((None,'platon')),
                            converter=none_if_none)

    platon: Platon  = field(factory=Platon)
