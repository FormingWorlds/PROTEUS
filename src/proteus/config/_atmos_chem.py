from __future__ import annotations

import math

from attrs import define, field
from attrs.validators import gt, in_, lt

from ._converters import none_if_none


def valid_use_vm_mol(instance, attribute, value):
    """Refuse the upwind molecular-diffusion scheme, which VULCAN aborts on."""
    if value:
        raise ValueError(
            f'{attribute.name}=true is not supported; '
            'VULCAN upwind scheme for molecular diffusion is in development'
        )


def valid_ini_mix(instance, attribute, value):
    """Refuse `ini_mix = 'previous'` unless the surface mixing ratios are fixed."""
    if (value == 'previous') and not instance.fix_surf:
        raise ValueError(f"{attribute.name} = 'previous' requires fix_surf = true")


def valid_mode(instance, attribute, value):
    """Refuse coupled chemistry, and automatic chemistry without a module."""
    if value == 'online':
        raise ValueError(
            "atmos_chem.mode = 'online' is not yet implemented; use 'offline' or 'none'"
        )
    if (value == 'offline') and (instance.module is None):
        raise ValueError("atmos_chem.mode = 'offline' requires atmos_chem.module != 'none'")


def times_to_floats(value):
    """Convert a list of chemistry times to floats [yr]; leave strings as they are."""
    if isinstance(value, (list, tuple)):
        return [float(t) for t in value]
    return value


def valid_times(instance, attribute, value):
    """Accept 'end', 'all', or a non-empty list of finite times >= 0 [yr]."""
    if isinstance(value, str):
        if value not in ('end', 'all'):
            raise ValueError(
                f"atmos_chem.times must be 'end', 'all' or a list of times, got '{value}'"
            )
        return
    if not isinstance(value, list):
        raise ValueError(
            f"atmos_chem.times must be 'end', 'all' or a list of times, got {value}"
        )
    if len(value) == 0:
        raise ValueError('atmos_chem.times must not be an empty list')
    for t in value:
        if not (math.isfinite(t) and (t >= 0)):
            raise ValueError(f'atmos_chem.times must be finite and >= 0 yr, got {t}')


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
        Initial mixing ratios. Options: profile (the climate profile), outgas (the surface mixing ratios at every level), previous (the result of the latest earlier chemistry snapshot, interpolated in log pressure; needs fix_surf = true; falls back to profile when there is none).
    fix_surf : bool
        Fix the surface mixing ratios based on outgassed composition.
    network : str
        Chemical network. Options: CHO, NCHO, SNCHO.
    save_frames : bool
        Save simulation state as plots.
    yconv_cri : float
        Steady state - max change in mixing ratio over test period
    slope_cri : float
        Steady state - max rate of change of mixing ratio over test period
    yconv_min : float
        Steady state - alternative max change in mixing ratio, accepted when the rate of change is also below VULCAN's diffusion-limited slope.
    st_factor : float
        Steady state - the test period starts at this fraction of the elapsed model time, in (0, 1).
    conv_step : int
        Steady state - maximum number of solver steps the test period reaches back.
    use_vm_mol : bool
        Use the upwind scheme for molecular diffusion. Must be false, because VULCAN does not support it yet.
    use_vz : bool
        Include vertical advection at the constant updraft velocity `atmos_chem.updraft_const`.
    use_condense : bool
        Include condensation of every gas that the selected network can condense.
    use_settling : bool
        Include gravitational settling of the condensate particles. Has an effect only with `use_condense`.
    r_p : float
        Default radius of every condensate particle [cm].
    runtime : float
        Maximum integrated model time [s].
    dttry : float
        Initial solver timestep [s].
    dt_min : float
        Minimum solver timestep [s].
    atol : float
        Absolute tolerance on number density [cm-3].
    mtol : float
        Mixing ratio below which a species is ignored in the truncation error.
    mtol_conv : float
        Mixing ratio below which a species is ignored in the steady-state test.
    flux_atol : float
        Absolute tolerance on the actinic flux [photons cm-2 s-1 nm-1].
    rtol : float
        Relative tolerance for adjusting the solver step size.
    post_conden_rtol : float
        Relative tolerance used once condensation stops.
    """

    clip_fl: float = field(default=1e-20)
    clip_vmr: float = field(default=1e-10)
    make_funs: bool = field(default=True)
    ini_mix: str = field(
        default='profile',
        validator=(in_(('profile', 'outgas', 'previous')), valid_ini_mix),
    )
    fix_surf: bool = field(default=False)
    network: str = field(default='SNCHO', validator=in_(('CHO', 'NCHO', 'SNCHO')))
    save_frames: bool = field(default=False)
    yconv_cri: float = field(default=0.05)
    slope_cri: float = field(default=1e-4)
    yconv_min: float = field(default=0.5, validator=gt(0))
    st_factor: float = field(default=0.5, validator=(gt(0), lt(1)))
    conv_step: int = field(default=100, validator=gt(0))
    use_vm_mol: bool = field(default=False, validator=valid_use_vm_mol)
    use_vz: bool = field(default=False)
    use_condense: bool = field(default=False)
    use_settling: bool = field(default=False)
    r_p: float = field(default=5e-3, validator=gt(0))
    runtime: float = field(default=1e22, validator=gt(0))
    dttry: float = field(default=1e-8, validator=gt(0))
    dt_min: float = field(default=1e-10, validator=gt(0))
    atol: float = field(default=0.1, validator=gt(0))
    mtol: float = field(default=1e-22, validator=gt(0))
    mtol_conv: float = field(default=1e-20, validator=gt(0))
    flux_atol: float = field(default=1.0, validator=gt(0))
    rtol: float = field(default=0.9, validator=gt(0))
    post_conden_rtol: float = field(default=0.1, validator=gt(0))


@define
class AtmosChem:
    """Atmosphere chemistry parameters, model selection.

    Attributes
    ----------
    module : str
        Chemistry module
    vulcan : Vulcan
        VULCAN  module options
    mode : str
        When to run the chemistry module automatically. Options: none (only with `proteus offchem`), offline (after the simulation ends), online (not yet implemented).
    times : str | list[float]
        Snapshots to run the chemistry on, with `mode = 'offline'` or `proteus offchem`. Options: end (the final state), all (every atmosphere snapshot), or a list of times [yr], each mapped to the nearest atmosphere snapshot.
    photo_on : bool
        Use photochemistry.
    Kzz_on : bool
        Use Kzz.
    Kzz_const : float
        Constant Kzz value [cm2/s]. If 'none', Kzz is read from NetCDF file.
    moldiff_on : bool
        Use molecular diffusion.
    updraft_const : float
        Updraft velocity [cm/s].

    """

    module: str | None = field(
        default='none', validator=in_((None, 'vulcan', 'dummy')), converter=none_if_none
    )

    vulcan: Vulcan = field(factory=Vulcan)

    mode: str | None = field(
        default='none',
        validator=(in_((None, 'offline', 'online')), valid_mode),
        converter=none_if_none,
    )
    times: str | list[float] = field(
        default='end', validator=valid_times, converter=times_to_floats
    )
    photo_on: bool = field(default=True)
    Kzz_on: bool = field(default=True)
    Kzz_const: float | str | None = field(default=None, converter=none_if_none)
    moldiff_on: bool = field(default=True)
    updraft_const: float = field(default=0.0)
