from __future__ import annotations

import logging

from attrs import define, field
from attrs.validators import in_

from ._converters import none_if_none

log = logging.getLogger('fwl.' + __name__)

@define
class Vulcan:
    """VULCAN chemistry module.

    Attributes
    ----------
    clip_fl : float
        Stellar flux floor [ergs cm-2 s-1 nm-1].
    clip_vmr : float
        Neglect species with surface VMR < clip_vmr.
    make_funs : bool
        Make functions from chemical network.
    ini_mix : str
        Initial mixing ratios. Options: table, const_mix.
    fix_surf : bool
        Fix the surface mixing ratios based on outgassed composition.
    network : str
        Chemical network. Options: CHO, NCHO, SNCHO.
    save_frames : bool
        Save simulation state as plots.

    """

    clip_fl: float      = field(default=1e-20)
    clip_vmr:float      = field(default=1e-10)
    make_funs:bool      = field(default=True)
    ini_mix:str         = field(default="table",
                                validator=in_(("table", "const_mix")))
    fix_surf:bool       = field(default=False)
    network:str         = field(default="SNCHO",
                                validator=in_(("CHO", "NCHO", "SNCHO")))
    save_frames:bool    = field(default=False)


@define
class AtmosChem:
    """Atmosphere chemistry parameters, model selection.

    Attributes
    ----------
    module : str
        Chemistry module
    vulcan : Vulcan
        VULCAN  module options
    Kzz_on : bool
        Use Kzz.
    Kzz_const : float
        Constant Kzz value [cm2/s]. If 'none', Kzz is read from NetCDF file.
    moldiff_on : bool
        Use molecular diffusion.
    photo_on : bool
        Use photochemistry.
    updraft_on : bool
        Use updraft velocity.

    """

    module: str | None  = field(validator=in_((None,'vulcan')), converter=none_if_none)

    vulcan: Vulcan      = field(factory=Vulcan)

    Kzz_on:bool         = field(default=True)
    Kzz_const           = field(default=None, converter=none_if_none)
    moldiff_on:bool     = field(default=True)
    photo_on:bool       = field(default=True)
    updraft_on:bool     = field(default=False)
