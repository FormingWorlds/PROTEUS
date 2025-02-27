from __future__ import annotations

from attrs import define, field
from attrs.validators import in_

from ._converters import none_if_none


@define
class Observe:
    """Synthetic observations.

    synthesis: str
        Module to use for calculating synthetic spectra.
    """

    synthesis: str = field(validator=in_((None,'platon')), converter=none_if_none)
