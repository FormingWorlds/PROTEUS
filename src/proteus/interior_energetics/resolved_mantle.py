# Depth-resolved mantle shared by the zero-dimensional energetics backends
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import cumulative_trapezoid

from proteus.interior_energetics import adiabat

if TYPE_CHECKING:
    from proteus.config import Config
    from proteus.interior_energetics.common import Interior_t

log = logging.getLogger('fwl.' + __name__)

# Spacing of the magma temperatures at which the melt curve evaluates the
# adiabat [K]. Between these nodes the melt fraction is interpolated linearly,
# so the error is second order in the spacing times the curvature of the melt
# fraction, negligible against a mushy zone several hundred kelvin wide. One
# kelvin keeps the number of adiabat integrations per step close to the number
# of kelvin the mantle cools through.
DTMP_GRID = 1.0


def get_melt_model(
    config: Config, dirs: dict, hf_row: dict, interior_o: Interior_t
) -> adiabat.AdiabatMeltModel:
    """
    Melt model for the current structure, rebuilt when the structure changes.

    The mantle profile, the melting curves sampled on it and the adiabatic
    gradient all depend on the structure solve, so the model is cached against
    the identity of the mesh file and rebuilt when the structure module
    rewrites it. The equation-of-state interpolation cache outlives the rebuild
    so the gradient tables are read from disk once per run.

    Parameters
    ----------
        config : Config
            Model configuration.
        dirs : dict
            Model directories.
        hf_row : dict
            Current helpfile row.
        interior_o : Interior_t
            Interior state, carrying the cache.

    Returns
    ----------
        model : adiabat.AdiabatMeltModel
            Melt model for the current structure.
    """
    outdir = dirs['output']
    identity = adiabat.mesh_identity(outdir)

    model = getattr(interior_o, '_adiabat_melt_model', None)
    if (model is not None) and (model.identity == identity):
        return model

    eos_cache = getattr(interior_o, '_adiabat_eos_cache', None)
    if eos_cache is None:
        eos_cache = {}
        interior_o._adiabat_eos_cache = eos_cache

    model = adiabat.AdiabatMeltModel(config, outdir, hf_row, eos_cache)
    interior_o._adiabat_melt_model = model
    return model


class MeltCurve:
    """
    Mass melt fraction of the mantle as a function of the magma temperature.

    An ODE solver evaluates the melt fraction and its response to the magma
    temperature many times per step, too often to integrate an adiabat each
    time. The curve integrates the adiabat once per node of a uniform grid in
    the magma temperature, keeps every node it has computed for as long as the
    structure is unchanged, and interpolates linearly in between.

    The response is the slope of the segment the temperature falls in, so the
    latent heat drawn from the crystallisation front between two temperatures,
    the latent capacity integrated over that interval, equals the latent heat
    times the mantle mass times the change in melt fraction, whatever path the
    solver takes.

    Attributes
    ----------
        model : adiabat.AdiabatMeltModel
            Melt model the nodes are evaluated with.
        dtmp : float
            Node spacing [K].
    """

    def __init__(self, model: adiabat.AdiabatMeltModel, dtmp: float = DTMP_GRID):
        if not dtmp > 0.0:
            raise ValueError('Melt curve node spacing must be positive, got %.3e K' % dtmp)
        self.model = model
        self.dtmp = float(dtmp)
        self._nodes: dict[int, float] = {}

        # The adiabat never cools with depth, so an anchor at or above the
        # hottest liquidus anywhere in the mantle leaves every level molten.
        self._tmp_molten = float(np.max(model.tmp_liq))

    def _phi_node(self, k: int) -> float:
        """Mass melt fraction at node `k`, integrating the adiabat on first use."""
        phi = self._nodes.get(k)
        if phi is None:
            # The adiabat needs a positive anchor; no level is molten at the
            # lowest temperature it admits.
            tmp = max(k * self.dtmp, adiabat.TMP_MIN)
            if tmp >= self._tmp_molten:
                phi = 1.0
            else:
                phi = self.model.state(tmp).phi_mass
            self._nodes[k] = phi
        return phi

    def _segment(self, tmp: float) -> tuple[float, float, float]:
        """Fractional position within the segment holding `tmp`, and its end values."""
        if math.isnan(tmp):
            raise ValueError('Magma temperature is NaN; the melt fraction is undefined')
        if tmp >= self._tmp_molten:
            return 0.0, 1.0, 1.0
        k = math.floor(tmp / self.dtmp)
        frac = tmp / self.dtmp - k
        return frac, self._phi_node(k), self._phi_node(k + 1)

    def phi(self, tmp: float) -> float:
        """
        Mass melt fraction of the mantle at magma temperature `tmp` [K].

        Within [0, 1].
        """
        frac, lo, hi = self._segment(tmp)
        return float(np.clip(lo + frac * (hi - lo), 0.0, 1.0))

    def dphi_dtmp(self, tmp: float) -> float:
        """
        Response of the mass melt fraction to the magma temperature [K-1].

        Non-negative: melting is monotonic in temperature, so a negative slope
        between two nodes is integration noise and is not charged as latent
        heat.
        """
        _, lo, hi = self._segment(tmp)
        return max((hi - lo) / self.dtmp, 0.0)


def get_melt_curve(
    config: Config, dirs: dict, hf_row: dict, interior_o: Interior_t
) -> MeltCurve:
    """
    Melt curve for the current structure, rebuilt with its melt model.

    The nodes already computed are kept across steps for as long as the melt
    model is, so a mantle cooling slowly integrates the adiabat only at the
    temperatures it has not visited before.
    """
    model = get_melt_model(config, dirs, hf_row, interior_o)
    curve = getattr(interior_o, '_adiabat_melt_curve', None)
    if (curve is None) or (curve.model is not model):
        curve = MeltCurve(model)
        interior_o._adiabat_melt_curve = curve
    return curve


@dataclass(frozen=True)
class MantleShells:
    """Mantle split into radial shells for the tidal heating modules.

    Attributes
    ----------
        radius : np.ndarray
            Radius of the shell edges [m], ascending from the core-mantle
            boundary to the top of the mantle. One entry longer than the
            per-shell arrays.
        mass : np.ndarray
            Mass of each shell [kg], summing to the mantle mass.
        density : np.ndarray
            Mean density of each shell [kg m-3], its mass over its volume.
        pressure : np.ndarray
            Pressure at the mid-radius of each shell [Pa].
    """

    radius: np.ndarray
    mass: np.ndarray
    density: np.ndarray
    pressure: np.ndarray


def _enclosed_mass(profile: adiabat.MantleProfile, values: np.ndarray | None = None):
    """
    Mass enclosed between the core-mantle boundary and each level of the profile.

    With `values` given, the enclosed integral of `values` weighted by mass
    instead. Both use the trapezoid rule on the profile's own levels, the rule
    `adiabat.mass_melt_fraction` integrates with, so shell sums built from them
    reproduce the mantle-wide integrals exactly.
    """
    integrand = 4.0 * np.pi * profile.radius**2 * profile.density
    if values is not None:
        integrand = integrand * np.asarray(values, dtype=float)
    return cumulative_trapezoid(integrand, profile.radius, initial=0.0)


def shell_edges(profile: adiabat.MantleProfile, n_shells: int) -> np.ndarray:
    """
    Edges of `n_shells` shells of equal thickness spanning the mantle profile.

    Parameters
    ----------
        profile : adiabat.MantleProfile
            Radial mantle profile.
        n_shells : int
            Number of shells, at least one.

    Returns
    ----------
        edges : np.ndarray
            Shell edge radii [m], ascending from the core-mantle boundary.
    """
    if n_shells < 1:
        raise ValueError(f'The mantle needs at least one shell, got {n_shells}')
    return np.linspace(profile.r_core, profile.r_int, int(n_shells) + 1)


def shell_masses(profile: adiabat.MantleProfile, edges: np.ndarray) -> np.ndarray:
    """
    Mass of the profile between consecutive shell edges [kg].

    Differencing the enclosed mass makes the shells sum to the profile's mass
    whatever the shell edges, including edges that fall between the levels.
    """
    return np.diff(np.interp(edges, profile.radius, _enclosed_mass(profile)))


def shell_average(
    profile: adiabat.MantleProfile, edges: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """
    Mass-weighted mean of a per-level quantity over each shell.

    The shell means weighted by the shell masses recover the mass-weighted
    mean over the whole mantle, so the melt fraction of the shells agrees with
    the global mass melt fraction to rounding.

    Parameters
    ----------
        profile : adiabat.MantleProfile
            Radial mantle profile.
        edges : np.ndarray
            Shell edge radii [m], ascending.
        values : np.ndarray
            Quantity at each level of the profile.

    Returns
    ----------
        means : np.ndarray
            Mean of the quantity over each shell.
    """
    weighted = np.diff(np.interp(edges, profile.radius, _enclosed_mass(profile, values)))
    return weighted / shell_masses(profile, edges)


def mantle_shells(
    profile: adiabat.MantleProfile, n_shells: int, mantle_mass: float
) -> MantleShells:
    """
    Split the mantle profile into shells carrying the structure's mantle mass.

    The shell masses follow the profile's own density, rescaled so that they
    sum to `mantle_mass`, the mass the energetics backend uses for heat
    capacity and heating. The rescaling leaves every shell mean unchanged.

    Parameters
    ----------
        profile : adiabat.MantleProfile
            Radial mantle profile.
        n_shells : int
            Number of shells.
        mantle_mass : float
            Mantle mass [kg].

    Returns
    ----------
        shells : MantleShells
            Shell edges, masses, densities and pressures.
    """
    if not mantle_mass > 0.0:
        raise ValueError('Mantle mass must be positive, got %.3e kg' % mantle_mass)

    edges = shell_edges(profile, n_shells)
    mass = shell_masses(profile, edges)
    mass = mass * (mantle_mass / float(np.sum(mass)))

    volume = (4.0 * np.pi / 3.0) * np.diff(edges**3)
    mid = 0.5 * (edges[:-1] + edges[1:])
    pressure = np.interp(mid, profile.radius, profile.pressure)

    return MantleShells(radius=edges, mass=mass, density=mass / volume, pressure=pressure)


def tidal_power(tides, shell_mass: np.ndarray) -> float:
    """
    Tidal heat deposited in the mantle [W].

    Parameters
    ----------
        tides : array_like
            Tidal heating rate of each shell [W kg-1].
        shell_mass : np.ndarray
            Mass of each shell [kg].

    Returns
    ----------
        power : float
            Tidal heating summed over the shells [W].
    """
    tides = np.asarray(tides, dtype=float).ravel()
    shell_mass = np.asarray(shell_mass, dtype=float).ravel()
    if tides.size != shell_mass.size:
        raise ValueError(
            f'The tidal heating array has {tides.size} entries but the mantle has '
            f'{shell_mass.size} shell(s); the tides module and the interior '
            'disagree on the radial grid.'
        )
    return float(np.dot(tides, shell_mass))


def publish_shells(
    interior_o: Interior_t,
    shells: MantleShells,
    profile: adiabat.MantleProfile,
    state: adiabat.MeltState,
):
    """Publish the resolved mantle on the layer arrays read by the tides modules.

    The shells are ordered from the core-mantle boundary outward, the Aragog
    convention. The melt fraction and temperature of each shell are mass
    means of the adiabat over it. Viscosity is a placeholder here, filled in
    afterwards from the melt fraction by the backend or the interior wrapper.
    """
    n_shells = len(shells.mass)
    interior_o.radius = np.array(shells.radius, dtype=float)
    interior_o.mass = np.array(shells.mass, dtype=float)
    interior_o.density = np.array(shells.density, dtype=float)
    interior_o.pres = np.array(shells.pressure, dtype=float)
    interior_o.phi = np.clip(shell_average(profile, shells.radius, state.phi), 0.0, 1.0)
    interior_o.temp = shell_average(profile, shells.radius, state.temperature)
    interior_o.visc = np.ones(n_shells)  # placeholder to be updated elsewhere
