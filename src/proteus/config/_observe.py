from __future__ import annotations

from attrs import define, field
from attrs.validators import in_, ge


@define
class Observe:
    """Synthetic observations.

    module: str
        Module to use for calculating synthetic spectra.
    samples: int
        How many samples to use from the simulated evolution.
    """

    module: str = field(validator=in_((None,'platon')), converter=none_if_none)
    samples: int = field(validator=ge(0))
