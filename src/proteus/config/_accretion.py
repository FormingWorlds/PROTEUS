from __future__ import annotations

from attr.validators import in_
from attrs import define, field

from ._converters import none_if_none


@define
class Accretion:
    """Late accretion / delivery model selection.

    Attributes
    ----------
    module: str or None
        Accretion module to use. Currently only None is supported.
    """

    module: str | None = field(default='none', validator=in_((None,)), converter=none_if_none)
