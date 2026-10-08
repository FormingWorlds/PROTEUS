"""Single source of the aragog phase parameters built from a PROTEUS config.

The aragog interior module builds phase parameters at three call sites:

- the numpy entropy solver's ``_PhaseMixedParameters`` (``aragog.py``);
- the JAX CVODE factory's ``PhaseParams`` (``aragog.py``);
- the JAX research runner's ``PhaseParams`` (``aragog_jax.py``).

The numpy and JAX types name their fields differently and the numpy type
carries constant-property and mush fields the JAX type does not, so each
site resolves the same config quantities on its own unless a single
builder feeds them. A field resolved at one site but forgotten at
another, or cast at one site but not another, drifts silently.

This module resolves every configured quantity once in
``_phase_params_from_config`` and exposes two thin type-constructors,
``build_mixed_phase_params`` and ``build_jax_phase_params``. The three
call sites carry no field list of their own, so a configured field
reaches every site it applies to or none, and a shared quantity is cast
in exactly one place.

Notes
-----
The module imports only the aragog library types, never a PROTEUS
module, so ``aragog.py`` and ``aragog_jax.py`` can both import it at top
without a cycle. ``_PhaseMixedParameters`` is imported at module top;
``PhaseParams`` is imported lazily inside ``build_jax_phase_params`` so
that importing this module, and ``aragog.py`` through it, does not
require the optional JAX stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aragog.parser import _PhaseMixedParameters

if TYPE_CHECKING:
    from aragog.jax.phase import PhaseParams

    from proteus.config import Config

# Fixed at both JAX sites, not exposed in the PROTEUS schema. The
# SPIDER-analogue cubic-Hermite Jgrav smoothing stays on for production;
# the width matches the hardcoded 1e-2 in the numpy entropy_state
# smoothing calls.
_JAX_BOTTOM_UP_GRAV_SEP = True
_JAX_PHASE_SMOOTHING_WIDTH = 0.01

# Fixed at the numpy site: the mixed-phase evaluator is always built in
# the 'mixed' phase.
_MIXED_PHASE = 'mixed'


@dataclass(frozen=True)
class _PhaseParamsInputs:
    """Configured quantities that feed the aragog phase parameters.

    Every field is resolved and cast once from a PROTEUS config, so the
    numpy and JAX constructors that consume this object cannot resolve a
    shared quantity differently.

    Attributes
    ----------
    rheological_transition_melt_fraction, rheological_transition_width : float
        Melt-fraction centre and width of the rheological transition.
        Shared: numpy ``rheological_transition_*`` and JAX
        ``phi_rheo`` / ``phi_width``.
    grain_size : float
        Grain size. Shared by both types under the same name.
    matprop_smooth_width : float
        SPIDER material-property blend width. Shared by both types under
        the same name.
    separation_viscosity : str
        Gravitational-separation drag viscosity source ('melt' or
        'mixture'). Shared: the numpy type stores the string, the JAX
        type stores it as the ``separation_viscosity_mixture`` flag.
    latent_heat_of_fusion, phase_transition_width : float
        Numpy-only mixed-phase quantities.
    const_properties : bool
        Numpy-only constant-properties switch.
    const_rho, const_Cp, const_alpha, const_cond, const_log10visc, const_T_ref, const_S_ref : float
        Numpy-only constant-property values.
    viscosity_solid, viscosity_liquid : float
        JAX-only solid and liquid viscosities, already raised from the
        configured base-10 logarithms.
    k_solid, k_liquid : float
        JAX-only solid and liquid thermal conductivities.
    conduction, convection, grav_sep, mixing : bool
        JAX-only transport-term switches.
    eddy_diff_thermal, eddy_diff_chemical, kappah_floor : float
        JAX-only eddy diffusivities and the mixing-length diffusivity
        floor.
    phase_smoothing : str
        JAX-only phase-boundary smoothing selection ('tanh' or
        'cubic_hermite').
    """

    # Shared between the numpy and JAX phase parameters.
    rheological_transition_melt_fraction: float
    rheological_transition_width: float
    grain_size: float
    matprop_smooth_width: float
    separation_viscosity: str

    # Numpy mixed-phase parameters only.
    latent_heat_of_fusion: float
    phase_transition_width: float
    const_properties: bool
    const_rho: float
    const_Cp: float
    const_alpha: float
    const_cond: float
    const_log10visc: float
    const_T_ref: float
    const_S_ref: float

    # JAX phase parameters only.
    viscosity_solid: float
    viscosity_liquid: float
    k_solid: float
    k_liquid: float
    conduction: bool
    convection: bool
    grav_sep: bool
    mixing: bool
    eddy_diff_thermal: float
    eddy_diff_chemical: float
    kappah_floor: float
    phase_smoothing: str


def _phase_params_from_config(config: Config) -> _PhaseParamsInputs:
    """Resolve the aragog phase-parameter inputs from a PROTEUS config.

    Parameters
    ----------
    config : Config
        The PROTEUS configuration. Only ``config.interior_energetics``
        and its nested ``spider`` and ``aragog`` sections are read.

    Returns
    -------
    _PhaseParamsInputs
        The configured quantities, each cast once to the type the
        aragog constructors expect.

    Notes
    -----
    ``kappah_floor`` is cast to ``float`` here so that both JAX sites
    store an identical value; the numpy site does not consume it.
    """
    ie = config.interior_energetics
    return _PhaseParamsInputs(
        rheological_transition_melt_fraction=ie.rfront_loc,
        rheological_transition_width=ie.rfront_wid,
        grain_size=ie.grain_size,
        matprop_smooth_width=float(ie.spider.matprop_smooth_width),
        separation_viscosity=ie.aragog.separation_viscosity,
        latent_heat_of_fusion=float(ie.latent_heat_of_fusion),
        phase_transition_width=float(ie.phase_transition_width),
        const_properties=bool(ie.const_properties),
        const_rho=float(ie.const_rho),
        const_Cp=float(ie.const_Cp),
        const_alpha=float(ie.const_alpha),
        const_cond=float(ie.const_cond),
        const_log10visc=float(ie.const_log10visc),
        const_T_ref=float(ie.const_T_ref),
        const_S_ref=float(ie.const_S_ref),
        viscosity_solid=10.0 ** float(ie.solid_log10visc),
        viscosity_liquid=10.0 ** float(ie.melt_log10visc),
        k_solid=float(ie.solid_cond),
        k_liquid=float(ie.melt_cond),
        conduction=ie.trans_conduction,
        convection=ie.trans_convection,
        grav_sep=ie.trans_grav_sep,
        mixing=ie.trans_mixing,
        eddy_diff_thermal=float(ie.eddy_diffusivity_thermal),
        eddy_diff_chemical=float(ie.eddy_diffusivity_chemical),
        kappah_floor=float(ie.kappah_floor),
        phase_smoothing=ie.aragog.phase_smoothing,
    )


def build_mixed_phase_params(
    config: Config, solidus: str, liquidus: str
) -> _PhaseMixedParameters:
    """Build the numpy mixed-phase parameters from a PROTEUS config.

    Parameters
    ----------
    config : Config
        The PROTEUS configuration.
    solidus, liquidus : str
        Runtime paths to the solidus and liquidus data files, resolved
        by the caller from the interior lookup directory.

    Returns
    -------
    _PhaseMixedParameters
        The mixed-phase parameters for the numpy entropy solver.

    Notes
    -----
    ``cp_blend`` is not passed, so it stays at the aragog library
    default. This builder is the only construction site of
    ``_PhaseMixedParameters`` in PROTEUS.
    """
    inputs = _phase_params_from_config(config)
    return _PhaseMixedParameters(
        latent_heat_of_fusion=inputs.latent_heat_of_fusion,
        rheological_transition_melt_fraction=inputs.rheological_transition_melt_fraction,
        rheological_transition_width=inputs.rheological_transition_width,
        solidus=solidus,
        liquidus=liquidus,
        phase=_MIXED_PHASE,
        phase_transition_width=inputs.phase_transition_width,
        grain_size=inputs.grain_size,
        separation_viscosity=inputs.separation_viscosity,
        matprop_smooth_width=inputs.matprop_smooth_width,
        const_properties=inputs.const_properties,
        const_rho=inputs.const_rho,
        const_Cp=inputs.const_Cp,
        const_alpha=inputs.const_alpha,
        const_cond=inputs.const_cond,
        const_log10visc=inputs.const_log10visc,
        const_T_ref=inputs.const_T_ref,
        const_S_ref=inputs.const_S_ref,
    )


def build_jax_phase_params(config: Config) -> PhaseParams:
    """Build the JAX phase parameters from a PROTEUS config.

    Parameters
    ----------
    config : Config
        The PROTEUS configuration.

    Returns
    -------
    PhaseParams
        The phase parameters for the JAX entropy solver, shared by the
        CVODE factory and the research runner.

    Notes
    -----
    ``PhaseParams`` is imported here rather than at module top so that
    importing this module does not require the optional JAX stack. This
    builder is the only construction site of ``PhaseParams`` in PROTEUS.
    """
    from aragog.jax.phase import PhaseParams

    inputs = _phase_params_from_config(config)
    return PhaseParams(
        phi_rheo=inputs.rheological_transition_melt_fraction,
        phi_width=inputs.rheological_transition_width,
        viscosity_solid=inputs.viscosity_solid,
        viscosity_liquid=inputs.viscosity_liquid,
        grain_size=inputs.grain_size,
        k_solid=inputs.k_solid,
        k_liquid=inputs.k_liquid,
        matprop_smooth_width=inputs.matprop_smooth_width,
        conduction=inputs.conduction,
        convection=inputs.convection,
        grav_sep=inputs.grav_sep,
        mixing=inputs.mixing,
        eddy_diff_thermal=inputs.eddy_diff_thermal,
        eddy_diff_chemical=inputs.eddy_diff_chemical,
        kappah_floor=inputs.kappah_floor,
        bottom_up_grav_sep=_JAX_BOTTOM_UP_GRAV_SEP,
        phase_smoothing=inputs.phase_smoothing,
        phase_smoothing_width=_JAX_PHASE_SMOOTHING_WIDTH,
        separation_viscosity=inputs.separation_viscosity,
    )
