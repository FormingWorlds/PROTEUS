# Depth-resolved mantle melt state for the zero-dimensional interior modules
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Melting curve folder used when `interior_struct.melting_dir` is left unset
# and the structure module is not Zalmoxis, which brings the PALEOS curves of
# its own. This is the folder that the basic interior lookup-table download already
# fetches, so a configuration that never mentions melting curves still finds
# a solidus and a liquidus on disk.
DEFAULT_MELTING_DIR = 'Wolf_Bower+2018'

# Lowest pressure at which the adiabat is stepped [Pa]. Below one bar the
# dT/dP = nabla T / P form divides by a pressure that carries no information
# about the mantle, so the temperature is carried through unchanged. Matches
# the guard in the Zalmoxis structure adiabat.
PRES_FLOOR = 1.0e5

# Temperature bounds applied to each adiabat step [K]. Wide enough never to
# bind on a physical mantle profile; they exist so that a corrupt gradient
# lookup produces a bounded profile instead of an overflow that propagates
# silently into the melt fraction. Matches the Zalmoxis structure adiabat.
TMP_MIN = 100.0
TMP_MAX = 1.0e5

# Largest number of levels the adiabat is integrated over. The gradient lookup
# is evaluated once per level per solve, so a very finely resolved structure
# mesh would dominate the cost of a step for no gain: the adiabat and the
# melting curves are both smooth on the scale of a mantle shell. A mesh with
# more levels than this is linearly resampled onto this many.
MAX_LEVELS = 400

# Constant melting temperatures the boundary backend falls back on under its
# linear melt model, named in the hints the errors below give.
_LINEAR_CONSTANTS = {
    'boundary': 'T_solidus and T_liquidus',
}


def _energetics_module(config: Config) -> str:
    """
    Name of the energetics module building the melt model, for error messages.

    Read defensively: an error raised while composing another error's message
    would hide the problem the message is about.
    """
    module = getattr(getattr(config, 'interior_energetics', None), 'module', None)
    return module if module in _LINEAR_CONSTANTS else 'boundary'


def _linear_switch(config: Config) -> str:
    """Setting that runs the energetics module without an adiabat."""
    return f'interior_energetics.{_energetics_module(config)}.melt_model = "linear"'


@dataclass(frozen=True)
class MantleProfile:
    """Radial structure of the mantle, ascending from the CMB to the surface.

    Attributes
    ----------
        radius : np.ndarray
            Radius of each level [m], strictly ascending.
        pressure : np.ndarray
            Pressure at each level [Pa], decreasing outward.
        density : np.ndarray
            Mass density at each level [kg m-3].
        gravity : np.ndarray
            Gravitational acceleration at each level [m s-2].
    """

    radius: np.ndarray
    pressure: np.ndarray
    density: np.ndarray
    gravity: np.ndarray

    @property
    def r_core(self) -> float:
        """Radius of the core-mantle boundary [m]."""
        return float(self.radius[0])

    @property
    def r_int(self) -> float:
        """Radius of the top of the mantle [m]."""
        return float(self.radius[-1])

    def shell_mass(self) -> float:
        """Mantle mass implied by the profile's own density [kg]."""
        return float(np.trapezoid(4.0 * np.pi * self.radius**2 * self.density, self.radius))


def mesh_path(outdir: str) -> str:
    """Path of the structure mesh written by the interior structure module."""
    return os.path.join(outdir, 'data', 'zalmoxis_output.dat')


def read_mantle_mesh(config: Config, outdir: str, hf_row: dict) -> MantleProfile:
    """
    Read the mantle profile produced by the interior structure module.

    The structure module writes `<outdir>/data/zalmoxis_output.dat` with five
    columns, `r [m]`, `P [Pa]`, `rho [kg m-3]`, `g [m s-2]`, `T [K]`, covering
    the mantle only and ascending in radius. Both structure backends write the
    same schema, and it is the same mesh the Aragog backend reads, so the melt
    state computed here is anchored on the structure the rest of the model sees.

    The temperature column is deliberately dropped. It holds the structure
    solve's own temperature, whereas the adiabat integrated here is anchored at
    the evolving magma temperature.

    Parameters
    ----------
        config : Config
            Model configuration.
        outdir : str
            Output directory of the current run.
        hf_row : dict
            Helpfile row, supplying the scalar radii and masses the mesh is
            checked against.

    Returns
    ----------
        profile : MantleProfile
            Radial mantle profile.
    """
    from proteus.interior_struct.zalmoxis import validate_zalmoxis_output_schema

    path = mesh_path(outdir)
    module = config.interior_struct.module

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'The {_energetics_module(config)} interior module needs the mantle '
            f'profile written by the structure module, but {path} does not exist. '
            f"The '{module}' structure module should have written it before the "
            'first interior call; check the structure solve for errors, or set '
            f'{_linear_switch(config)} to run without a radial profile.'
        )

    # Shape, top-of-mantle radius and mantle mass are the structure module's own
    # contract for this file. Reuse its checker rather than restating them here.
    validate_zalmoxis_output_schema(path, hf_row)

    data = np.loadtxt(path)
    radius, pressure, density, gravity = (data[:, i] for i in range(4))

    # After the schema check, which judges the file as written, and before the
    # invariant check, which should judge only the levels the solve filled.
    radius, pressure, density, gravity = _drop_unfilled_surface_nodes(
        radius, pressure, density, gravity, path
    )

    _check_mesh_invariants(radius, pressure, density, path, module)

    if len(radius) > MAX_LEVELS:
        log.debug(
            'Mantle profile has %d levels; resampling onto %d for the adiabat.',
            len(radius),
            MAX_LEVELS,
        )
        target = np.linspace(radius[0], radius[-1], MAX_LEVELS)
        pressure = np.interp(target, radius, pressure)
        density = np.interp(target, radius, density)
        gravity = np.interp(target, radius, gravity)
        radius = target

    return MantleProfile(radius=radius, pressure=pressure, density=density, gravity=gravity)


def _drop_unfilled_surface_nodes(
    radius: np.ndarray,
    pressure: np.ndarray,
    density: np.ndarray,
    gravity: np.ndarray,
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop the unfilled nodes a structure solve can leave outside the surface.

    When the solve reaches its surface-pressure target between two radial
    nodes, the node(s) outside that point are never integrated and are written
    with zero density and zero pressure. The damped-Picard outer solver does
    this on the first structure call, where the target is zero because nothing
    has outgassed yet. Zalmoxis treats these nodes as padding -- its own
    surface clean-up starts by walking inward past any non-positive density --
    and the invariant check has to do the same, or it rejects a valid structure.

    Only the trailing run is dropped. A non-positive density anywhere inside
    the mantle is left for the invariant check to reject.
    """
    top = len(density)
    while top > 0 and density[top - 1] <= 0.0:
        top -= 1

    # Nothing to drop, or nothing filled at all: leave the arrays whole so the
    # invariant check reports what is actually wrong with them.
    if top == len(density) or top == 0:
        return radius, pressure, density, gravity

    log.debug(
        'Dropped %d unfilled surface node(s) beyond r=%.6e m from %s; the '
        'mantle profile now ends at r=%.6e m, P=%.3e Pa.',
        len(density) - top,
        radius[top],
        path,
        radius[top - 1],
        pressure[top - 1],
    )
    return radius[:top], pressure[:top], density[:top], gravity[:top]


def _check_mesh_invariants(
    radius: np.ndarray,
    pressure: np.ndarray,
    density: np.ndarray,
    path: str,
    module: str,
) -> None:
    """Reject a mantle mesh that cannot support an adiabat integration."""
    if len(radius) < 2:
        raise ValueError(
            f'Mantle profile {path} written by the {module} structure module has '
            f'{len(radius)} level(s); at least two are needed to integrate an '
            'adiabat.'
        )
    if not np.all(np.diff(radius) > 0.0):
        raise ValueError(
            f'Mantle profile {path} is not strictly ascending in radius, so the '
            'columns are not in the documented r, P, rho, g, T order.'
        )
    if not np.all(np.diff(pressure) < 0.0):
        raise ValueError(
            f'Mantle profile {path} does not decrease in pressure outward, so the '
            'columns are not in the documented r, P, rho, g, T order.'
        )
    if not np.all(density > 0.0):
        raise ValueError(f'Mantle profile {path} contains a non-positive density.')
    if not np.all(pressure > 0.0):
        raise ValueError(f'Mantle profile {path} contains a non-positive pressure.')


def integrate_adiabat(
    profile: MantleProfile, tmp_top: float, nabla_ad: Callable[[float, float], float]
) -> np.ndarray:
    """
    Integrate a mantle adiabat inward from the top of the mantle.

    Steps `dT/dP = nabla_ad(P, T) T / P` down the profile's own pressure grid,
    anchored at `tmp_top` at the top of the mantle. This is the same explicit
    stepping, with the same low-pressure skip and temperature clamp, that the
    Zalmoxis structure solve uses for its surface-anchored adiabat, so the two
    agree on a shared profile.

    Parameters
    ----------
        profile : MantleProfile
            Radial mantle profile.
        tmp_top : float
            Temperature at the top of the mantle [K].
        nabla_ad : callable
            Maps (pressure [Pa], temperature [K]) to the dimensionless
            adiabatic gradient.

    Returns
    ----------
        temperature : np.ndarray
            Temperature at each level [K], ascending with the profile's radii.
    """
    if tmp_top <= 0.0:
        raise ValueError('Adiabat anchor temperature must be positive, got %.3e' % tmp_top)

    pres = profile.pressure
    n = len(pres)
    temp = np.zeros(n)
    temp[n - 1] = tmp_top

    for i in range(n - 2, -1, -1):
        p_eval = pres[i + 1]
        t_eval = temp[i + 1]

        if p_eval < PRES_FLOOR or pres[i] < PRES_FLOOR:
            temp[i] = t_eval
            continue

        nabla = nabla_ad(p_eval, t_eval)
        if nabla is None or not np.isfinite(nabla) or nabla <= 0.0:
            temp[i] = t_eval
            continue

        dtdp = nabla * t_eval / p_eval
        temp[i] = float(np.clip(t_eval + dtdp * (pres[i] - p_eval), TMP_MIN, TMP_MAX))

    return temp


def melt_profile(
    profile: MantleProfile,
    temperature: np.ndarray,
    tmp_sol: np.ndarray,
    tmp_liq: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Local melt fraction and the radius at which the adiabat meets the liquidus.

    The melt fraction is linear in temperature between the local solidus and
    the local liquidus, both evaluated at the level's own pressure, so the
    mushy zone follows the melting curves rather than a pair of constants. A
    level whose solidus and liquidus coincide is wholly molten at or above that
    temperature and wholly solid below it.

    Parameters
    ----------
        profile : MantleProfile
            Radial mantle profile.
        temperature : np.ndarray
            Mantle temperature at each level [K].
        tmp_sol : np.ndarray
            Solidus temperature at each level [K].
        tmp_liq : np.ndarray
            Liquidus temperature at each level [K].

    Returns
    ----------
        phi : np.ndarray
            Melt fraction at each level, within [0, 1].
        r_liq : float
            Radius at which the adiabat crosses the liquidus [m]. Equal to the
            core-mantle boundary when the whole mantle is above its liquidus,
            and to the top of the mantle when none of it is.
    """
    span = tmp_liq - tmp_sol
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.where(span > 0.0, (temperature - tmp_sol) / span, 0.0)
    # A degenerate melting curve has no mushy zone: the level is molten exactly
    # when it sits at or above the coincident solidus and liquidus.
    phi = np.where(span > 0.0, phi, np.where(temperature >= tmp_liq, 1.0, 0.0))
    phi = np.clip(phi, 0.0, 1.0)

    return phi, liquidus_crossing_radius(profile.radius, temperature - tmp_liq)


def liquidus_crossing_radius(radius: np.ndarray, excess: np.ndarray) -> float:
    """
    Radius at which the mantle temperature drops through the liquidus.

    Scans inward from the top of the mantle for the first level pair that
    brackets a sign change in `excess`, the amount by which the mantle exceeds
    its local liquidus, and interpolates linearly between them. Scanning from
    the top rather than the bottom picks the base of the surface magma ocean
    when a melting curve admits more than one crossing.

    Parameters
    ----------
        radius : np.ndarray
            Radius of each level [m], strictly ascending.
        excess : np.ndarray
            Mantle temperature minus the local liquidus at each level [K].

    Returns
    ----------
        r_liq : float
            Crossing radius [m].
    """
    n = len(radius)

    if excess[n - 1] < 0.0:
        # The top of the mantle is already below its liquidus: no magma ocean.
        return float(radius[n - 1])

    for i in range(n - 2, -1, -1):
        if excess[i] < 0.0 <= excess[i + 1]:
            # Linear interpolation for the zero of excess between the two levels.
            frac = excess[i + 1] / (excess[i + 1] - excess[i])
            return float(radius[i + 1] + frac * (radius[i] - radius[i + 1]))

    # Molten all the way down: the front sits at the core-mantle boundary.
    return float(radius[0])


def volume_melt_fraction(profile: MantleProfile, phi: np.ndarray) -> float:
    """
    Melt fraction of the mantle by volume.

    Parameters
    ----------
        profile : MantleProfile
            Radial mantle profile.
        phi : np.ndarray
            Melt fraction at each level.

    Returns
    ----------
        phi_vol : float
            Volume melt fraction, within [0, 1].
    """
    radius = profile.radius
    weight = 4.0 * np.pi * radius**2
    volume = float(np.trapezoid(weight, radius))
    if volume <= 0.0:
        return 0.0
    return float(np.clip(np.trapezoid(phi * weight, radius) / volume, 0.0, 1.0))


def mass_melt_fraction(profile: MantleProfile, phi: np.ndarray) -> float:
    """
    Melt fraction of the mantle by mass.

    Weighted by the profile's own density, so a mantle whose deep, dense levels
    have already crystallised reports a lower melt fraction by mass than by
    volume. The two agree only when the density is uniform.

    Parameters
    ----------
        profile : MantleProfile
            Radial mantle profile.
        phi : np.ndarray
            Melt fraction at each level.

    Returns
    ----------
        phi_mass : float
            Mass melt fraction, within [0, 1].
    """
    radius = profile.radius
    weight = 4.0 * np.pi * radius**2 * profile.density
    mass = float(np.trapezoid(weight, radius))
    if mass <= 0.0:
        return 0.0
    return float(np.clip(np.trapezoid(phi * weight, radius) / mass, 0.0, 1.0))


@dataclass(frozen=True)
class MeltState:
    """Melt state of the mantle at one magma temperature.

    Attributes
    ----------
        phi_mass : float
            Mass melt fraction of the mantle.
        phi_vol : float
            Volume melt fraction of the mantle.
        r_liq : float
            Radius at which the adiabat crosses the liquidus [m].
        temperature : np.ndarray
            Adiabatic temperature at each level [K].
        phi : np.ndarray
            Melt fraction at each level.
    """

    phi_mass: float
    phi_vol: float
    r_liq: float
    temperature: np.ndarray
    phi: np.ndarray


def solve_melt_state(
    profile: MantleProfile,
    tmp_top: float,
    nabla_ad: Callable[[float, float], float],
    tmp_sol: np.ndarray,
    tmp_liq: np.ndarray,
) -> MeltState:
    """
    Melt state of the mantle for a given temperature at the top of the mantle.

    Parameters
    ----------
        profile : MantleProfile
            Radial mantle profile.
        tmp_top : float
            Temperature at the top of the mantle [K].
        nabla_ad : callable
            Maps (pressure [Pa], temperature [K]) to the dimensionless
            adiabatic gradient.
        tmp_sol : np.ndarray
            Solidus temperature at each level [K].
        tmp_liq : np.ndarray
            Liquidus temperature at each level [K].

    Returns
    ----------
        state : MeltState
            Melt fractions, crystallisation front and profiles.
    """
    temperature = integrate_adiabat(profile, tmp_top, nabla_ad)
    phi, r_liq = melt_profile(profile, temperature, tmp_sol, tmp_liq)

    return MeltState(
        phi_mass=mass_melt_fraction(profile, phi),
        phi_vol=volume_melt_fraction(profile, phi),
        r_liq=r_liq,
        temperature=temperature,
        phi=phi,
    )


def uses_paleos_melting_curves(config: Config) -> bool:
    """
    Whether the mantle melt state is built on the PALEOS melting curves.

    True whenever the Zalmoxis structure module is in use. The solidus and
    liquidus are then the PALEOS MgSiO3 curves Zalmoxis uses for a PALEOS
    mantle, and the ones the `liquidus_super` initial condition is solved
    against, rather than a tabulated folder. `interior_struct.melting_dir` is
    not read by the melt model in that case.
    """
    return config.interior_struct.module == 'zalmoxis'


def resolve_melting_dir(config: Config) -> str:
    """
    Name of the melting curve folder to read the solidus and liquidus from.

    Falls back to the folder shipped with the basic interior lookup tables when
    the configuration does not name one, so a run that never mentions melting
    curves still gets a pressure-dependent melting model.
    """
    melting_dir = config.interior_struct.melting_dir
    if melting_dir is None:
        log.debug('No melting curve folder configured; using %s', DEFAULT_MELTING_DIR)
        return DEFAULT_MELTING_DIR
    return str(melting_dir)


def load_paleos_melting_curves(config: Config) -> tuple[Callable, Callable]:
    """
    PALEOS MgSiO3 solidus and liquidus as functions of pressure.

    The liquidus is the Zalmoxis `PALEOS-liquidus`: Belonoshko et al. (2005)
    below 2.55 GPa and Fei et al. (2021) above. The solidus is that liquidus
    scaled by `interior_struct.zalmoxis.mushy_zone_factor`, built by the same
    helper the structure solve uses, so the two share one mushy zone.

    Parameters
    ----------
        config : Config
            Model configuration.

    Returns
    ----------
        solidus, liquidus : callable
            Map pressure [Pa] to temperature [K].
    """
    try:
        from zalmoxis.melting_curves import (
            derive_solidus_from_liquidus,
            get_melting_curve_function,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f'The {_energetics_module(config)} interior module takes the PALEOS '
            'melting curves from Zalmoxis when the Zalmoxis structure module is '
            f'in use; import failed: {exc}. Install Zalmoxis, or set '
            f'{_linear_switch(config)} to run without melting curves.'
        )

    mushy_zone_factor = float(config.interior_struct.zalmoxis.mushy_zone_factor)
    energetics = getattr(config, 'interior_energetics', None)
    scale = float(getattr(energetics, 'paleos_melt_scale', 1.0))
    liquidus = get_melting_curve_function('PALEOS-liquidus')
    if scale != 1.0:
        # Both curves move together: the solidus is derived from the scaled
        # liquidus below, so the mushy zone keeps its shape.
        liquidus = _scaled_curve(liquidus, scale)
    solidus = derive_solidus_from_liquidus(liquidus, mushy_zone_factor)

    log.debug(
        'Mantle melting curves for the adiabat: %.4f x PALEOS liquidus, '
        'solidus = %.2f x liquidus',
        scale,
        mushy_zone_factor,
    )
    return solidus, liquidus


def _scaled_curve(curve: Callable, scale: float) -> Callable:
    """Melting curve whose temperature is `scale` times that of `curve`."""

    def scaled(pressure):
        return scale * curve(pressure)

    return scaled


def load_melting_curves(config: Config) -> tuple[Callable, Callable]:
    """
    Load the solidus and liquidus of the mantle as functions of pressure.

    With the Zalmoxis structure module these are the PALEOS curves, see
    `load_paleos_melting_curves`. Otherwise they are the two-column
    pressure-temperature curves shipped under
    `FWL_DATA/interior_lookup_tables/Melting_curves/`, the same files the
    Aragog and SPIDER backends are given.

    Parameters
    ----------
        config : Config
            Model configuration.

    Returns
    ----------
        solidus, liquidus : callable
            Map pressure [Pa] to temperature [K].
    """
    if uses_paleos_melting_curves(config):
        return load_paleos_melting_curves(config)

    from proteus.utils.data import FWL_DATA_DIR, load_melting_curve

    melting_dir = resolve_melting_dir(config)
    folder = FWL_DATA_DIR / 'interior_lookup_tables' / 'Melting_curves' / melting_dir

    curves = []
    for stem in ('solidus', 'liquidus'):
        path = folder / f'{stem}_P-T.dat'
        if not path.is_file():
            raise FileNotFoundError(
                f'The {_energetics_module(config)} interior module needs the '
                f'mantle {stem} at {path}. '
                "Run 'proteus get interiordata' to fetch the melting curves, "
                'choose another folder with interior_struct.melting_dir, or set '
                f'{_linear_switch(config)} to run with the configured '
                f'{_LINEAR_CONSTANTS[_energetics_module(config)]} instead.'
            )
        curve = load_melting_curve(path)
        if curve is None:
            raise ValueError(f'Could not read the mantle {stem} from {path}.')
        curves.append(curve)

    log.debug('Mantle melting curves for the adiabat read from %s', folder)
    return curves[0], curves[1]


def evaluate_melting_curve(curve: Callable, pressure: np.ndarray, name: str) -> np.ndarray:
    """
    Evaluate a melting curve over a pressure profile, clamped to its table.

    The tabulated curves return no value outside the pressure range they were
    fitted over. A deep mantle can reach past the top of the table, so the
    out-of-range levels are evaluated at the table edge rather than left
    undefined, and the extrapolation is reported once.

    Parameters
    ----------
        curve : callable
            Maps pressure [Pa] to temperature [K].
        pressure : np.ndarray
            Pressure at each level [Pa].
        name : str
            Curve name, used in the warning.

    Returns
    ----------
        temperature : np.ndarray
            Melting temperature at each level [K].
    """
    values = np.asarray(curve(pressure), dtype=float)
    if np.all(np.isfinite(values)):
        return values

    knots = getattr(curve, 'x', None)
    if knots is not None and len(knots) > 0:
        p_min, p_max = float(np.min(knots)), float(np.max(knots))
        log.warning(
            'Mantle %s is tabulated over %.1f to %.1f GPa but the mantle spans '
            '%.1f to %.1f GPa; %d level(s) outside the table are evaluated at '
            'its edge.',
            name,
            p_min * 1e-9,
            p_max * 1e-9,
            float(np.min(pressure)) * 1e-9,
            float(np.max(pressure)) * 1e-9,
            int(np.count_nonzero(~np.isfinite(values))),
        )
        values = np.asarray(curve(np.clip(pressure, p_min, p_max)), dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f'Mantle {name} returned no value at {int(np.count_nonzero(~np.isfinite(values)))} '
            'of the mantle levels, even after clamping to its tabulated pressure '
            'range. Check the melting curve data selected by '
            'interior_struct.melting_dir.'
        )
    return values


def zalmoxis_nabla_ad(
    config: Config, cache: dict, solidus: Callable, liquidus: Callable
) -> Callable[[float, float], float]:
    """
    Adiabatic gradient of the mantle, from the Zalmoxis equation of state.

    Returns the dimensionless gradient `nabla_ad = (dlnT/dlnP)_S` that Zalmoxis
    itself uses to build a structure adiabat, looked up from the equation of
    state selected by `interior_struct.zalmoxis.mantle_eos`. Routing through
    Zalmoxis rather than re-reading the tables keeps the boundary backend on
    the same gradient as the structure solve for every supported
    equation of state.

    Parameters
    ----------
        config : Config
            Model configuration.
        cache : dict
            Interpolation cache, shared across calls so the gradient tables are
            read from disk once per run.
        solidus, liquidus : callable
            Melting curves, mapping pressure [Pa] to temperature [K]. Two-phase
            equations of state weight the solid and liquid gradients by the
            local melt fraction, so they need the same curves the melt profile
            is built from.

    Returns
    ----------
        nabla_ad : callable
            Maps (pressure [Pa], temperature [K]) to the dimensionless gradient.
    """
    try:
        from zalmoxis.mixing import LayerMixture, get_mixed_nabla_ad
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f'The {_energetics_module(config)} interior module needs Zalmoxis '
            'for the mantle adiabatic gradient '
            '(zalmoxis.mixing.get_mixed_nabla_ad); import failed: '
            f'{exc}. Install Zalmoxis, or set {_linear_switch(config)} to run '
            'without an adiabat.'
        )

    from proteus.interior_struct.zalmoxis import load_zalmoxis_material_dictionaries

    eos_name = config.interior_struct.zalmoxis.mantle_eos
    backend = _energetics_module(config)
    switch = _linear_switch(config)
    mat_dicts = load_zalmoxis_material_dictionaries()
    mixture = LayerMixture(components=[eos_name], fractions=[1.0])

    def _nabla_ad(pressure: float, temperature: float) -> float:
        nabla = get_mixed_nabla_ad(
            pressure,
            temperature,
            mixture,
            mat_dicts,
            cache,
            solidus,
            liquidus,
        )
        if nabla is None:
            raise RuntimeError(
                f"The mantle equation of state '{eos_name}' provides no adiabatic "
                f'gradient, so the {backend} interior module cannot integrate an '
                f'adiabat. Choose a temperature-dependent mantle_eos, or set {switch}.'
            )
        return float(nabla)

    log.debug('Mantle adiabatic gradient from Zalmoxis EOS %s', eos_name)
    return _nabla_ad


def mesh_identity(outdir: str) -> tuple:
    """
    Identity of the structure mesh currently on disk.

    Used to notice that the structure module has re-solved and rewritten the
    mantle profile, so a cached profile is dropped rather than silently carried
    across a structure refresh.

    Parameters
    ----------
        outdir : str
            Output directory of the current run.

    Returns
    ----------
        identity : tuple
            Modification time and size of the mesh file, or an empty tuple when
            the file is absent.
    """
    try:
        stat = os.stat(mesh_path(outdir))
    except OSError:
        return ()
    return (stat.st_mtime_ns, stat.st_size)


class AdiabatMeltModel:
    """
    Melt state of the mantle from an adiabat and the mantle melting curves.

    Holds the structure module's mantle profile, the solidus and liquidus
    sampled on that profile's pressures, and the adiabatic gradient from the
    Zalmoxis equation of state. One instance serves every magma temperature
    evaluated against a single structure solve.

    Attributes
    ----------
        profile : MantleProfile
            Radial mantle profile from the structure module.
        tmp_sol, tmp_liq : np.ndarray
            Solidus and liquidus temperature at each level [K].
        identity : tuple
            Identity of the mesh file the profile was read from.
        eos_cache : dict
            Interpolation cache for the equation of state tables.
    """

    def __init__(self, config: Config, outdir: str, hf_row: dict, eos_cache: dict | None):
        self.eos_cache = {} if eos_cache is None else eos_cache
        self.identity = mesh_identity(outdir)
        self.profile = read_mantle_mesh(config, outdir, hf_row)

        solidus, liquidus = load_melting_curves(config)
        pres = self.profile.pressure
        self.tmp_sol = evaluate_melting_curve(solidus, pres, 'solidus')
        self.tmp_liq = evaluate_melting_curve(liquidus, pres, 'liquidus')

        self.nabla_ad = zalmoxis_nabla_ad(config, self.eos_cache, solidus, liquidus)

    def state(self, tmp_top: float) -> MeltState:
        """Melt state for a given temperature at the top of the mantle."""
        return solve_melt_state(
            self.profile, tmp_top, self.nabla_ad, self.tmp_sol, self.tmp_liq
        )
