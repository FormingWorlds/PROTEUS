from __future__ import annotations

from attrs import define, field
from attrs.validators import gt, in_, lt

from ._converters import none_if_none


@define
class PetitRADTRANS:
    """Parameters for the petitRADTRANS module.

    Attributes
    ----------
    input_data_path: str or None
        Optional path to petitRADTRANS `input_data`. If `None`, the installed
        package location will be used.
    line_opacity_mode: str
        Opacity treatment: 'c-k' (correlated-k) or 'lbl' (line-by-line).
    include_rayleigh: bool
        Include Rayleigh scattering contributions.
    include_cia: bool
        Include collision-induced absorption contributions.
    """

    input_data_path: str = field(default=None, converter=none_if_none)
    line_opacity_mode: str = field(default='c-k', validator=in_(('c-k', 'lbl')))
    include_rayleigh: bool = field(default=True)
    include_cia: bool = field(default=True)


@define
class Observe:
    """Synthetic observations.

    module: str
        Module to use for calculating synthetic spectra.
    clip_vmr: float
        Minimum VMR to include a species in radiative transfer.
    reference_pressure: float
        Reference pressure for synthetic spectrum generation [bar].
    """

    module: str = field(
        default='none',
        validator=in_((None, 'petitRADTRANS')),
        converter=none_if_none,
    )
    clip_vmr: float = field(default=1e-8, validator=(gt(0), lt(1)))
    reference_pressure: float = field(default=10, validator=gt(0))

    petitRADTRANS: PetitRADTRANS = field(factory=PetitRADTRANS)
