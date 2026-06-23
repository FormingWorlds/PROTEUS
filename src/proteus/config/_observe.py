from __future__ import annotations

from attrs import define, field
from attrs.validators import gt, in_, lt

from ._converters import none_if_none


@define
class PetitRADTRANS:
    """Parameters for the petitRADTRANS module.

    Attributes
    ----------
    line_opacity_mode: str
        Opacity treatment: 'c-k' (correlated-k) or 'lbl' (line-by-line).
    include_rayleigh: bool
        Include Rayleigh scattering contributions.
    include_cia: bool
        Include collision-induced absorption contributions.
    silent: bool
        Suppress petitRADTRANS stdout/stderr during Radtrans initialization.

    Note
    ----
    Input data is discovered at runtime from dirs['fwl']/prt/input_data,
    where dirs['fwl'] is populated from the FWL_DATA environment variable
    during PROTEUS startup.
    """

    line_opacity_mode: str = field(default='c-k', validator=in_(('c-k', 'lbl')))
    include_rayleigh: bool = field(default=True)
    include_cia: bool = field(default=True)
    silent: bool = field(default=True)


@define
class Observe:
    """Synthetic observations.

    module: str
        Module to use for calculating synthetic spectra.
    clip_vmr: float
        Minimum VMR to include a species in radiative transfer.
    reference_pressure: float
        Reference pressure for synthetic spectrum generation [bar].
    source: str
        Composition source selection: 'all', 'outgas', 'profile', or 'offchem'.
    spectrum_type: str
        Synthetic spectrum products to compute: 'both', 'transit', or 'eclipse'.
    remove_one_gas: bool
        If True, generate additional leave-one-out spectra with each gas removed.
    """

    module: str | None = field(
        default='none',
        validator=in_((None, 'petitRADTRANS')),
        converter=none_if_none,
    )
    clip_vmr: float = field(default=1e-8, validator=(gt(0), lt(1)))
    reference_pressure: float = field(default=10, validator=gt(0))
    source: str = field(default='all', validator=in_(('all', 'outgas', 'profile', 'offchem')))
    spectrum_type: str = field(default='both', validator=in_(('both', 'transit', 'eclipse')))
    remove_one_gas: bool = field(default=True)

    petitRADTRANS: PetitRADTRANS = field(factory=PetitRADTRANS)
