from __future__ import annotations

import logging

from attrs import define, field
from attrs.validators import ge, gt, in_, le

from ._converters import none_if_none

log = logging.getLogger('fwl.' + __name__)

@define
class Vulcan:
    """VULCAN chemistry module.

    Attributes
    ----------
    clip_fl : float
        Stellar flux floor.
    clip_vmr : float
        Neglect species with surface VMR < clip_vmr.
    make_funs : bool
        Make functions from chemical network.
    ini_mix : str
        Initial mixing ratios (table, const_mix).
    fix_surf : bool
        Fix surface species.
    network : str
        Chemical network string (CHO, NCHO, SNCHO)
    save_frames : bool
        Save animation.

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
    moldiff_on : bool
        Use molecular diffusion.
    updraft_on : bool
        Use updraft velocity.
    photo_on : bool
        Use photochemistry.

    """

    module: str | None  = field(validator=in_((None,'vulcan')), converter=none_if_none)

    vulcan: Vulcan      = field(factory=Vulcan)

    Kzz_on:bool         = field(default=False)
    moldiff_on:bool     = field(default=False)
    updraft_on:bool     = field(default=False)
    photo_on:bool       = field(default=False)
