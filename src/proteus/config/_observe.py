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

    downsample: int = field(default=8, validator=ge(1))
    clip_vmr: float = field(default=1e-8, validator=(gt(0), lt(1)))

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
    clip_vmr: float
        Minimum VMR to include a species in radiative transfer.
    include_rayleigh: bool
        Include Rayleigh scattering contributions.
    include_cia: bool
        Include collision-induced absorption contributions.
    remove_one_gas: bool
        If True, compute spectra with each gas removed (for diagnostics).
    """

    input_data_path: str = field(default=None, converter=none_if_none)
    line_opacity_mode: str = field(default="c-k", validator=in_(("c-k", "lbl")))
    clip_vmr: float = field(default=1e-8, validator=(gt(0), lt(1)))
    include_rayleigh: bool = field(default=True)
    include_cia: bool = field(default=True)
    remove_one_gas: bool = field(default=False)

@define
class Observe:
    """Synthetic observations.

    module: str
        Module to use for calculating synthetic spectra.
    """

    module: str = field(
        validator=in_((None, 'platon', 'petitRADTRANS')),
        converter=none_if_none,
    )

    platon: Platon = field(factory=Platon)
    petitRADTRANS: PetitRADTRANS = field(factory=PetitRADTRANS)
