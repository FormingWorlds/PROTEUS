"""Melt drainage from a crystallising mush, and the timescales that limit it.

Melt leaves the pore space of a freezing front only if two things happen: the
melt percolates through the pores, and the solid matrix deforms to close them.
Each has its own timescale and the slower one controls, so the drainage rate of
a parcel at porosity ``phi`` is

    1 / tau(phi) = 1 / max( L / w_rel(phi), tau_s )

with ``w_rel`` the Darcy percolation velocity and ``tau_s`` the matrix
deformation time. A parcel entering the front at ``phi_top`` and crossing it in
a residence time ``t_res`` therefore retains

    dphi/dt = -phi / max( L / w_rel(phi), tau_s ),   F_vol = phi(t_res)

which is the quantity the trapping bracket needs. This replaces the linear law
``F_tl = -(phi_c tau / DeltaT) dT/dt`` of Sim et al. (2024) Eq. 7 and its two
free parameters: ``tau`` becomes the two timescales above, computed from the
solver's own front, and ``DeltaT`` disappears because the residence time is
geometric (front thickness over front speed) rather than thermal.

Why the geometric residence time is the better one
--------------------------------------------------
The published form writes ``t_res = DeltaT / |dT/dt|`` with the potential
temperature of the convecting magma ocean. The freezing front is by definition
the region that has stopped convecting, so that substitution assumes the front
cools at the rate of the ocean above it. Taking ``t_res = L / v_f`` from the
front geometry needs no such assumption and removes a parameter at the same
time.

Relation to the solver's own separation velocity
------------------------------------------------
``w_rel`` here is the same three-regime law the interior solver uses for
gravitational separation (Bower et al. 2018 section 2.1), with two deliberate
differences.

* The drag viscosity is always the melt viscosity. The solver's
  ``separation_viscosity`` defaults to ``'mixture'``, the
  rheological-transition-blended value, which exceeds the melt viscosity by ten
  orders of magnitude at the top of the front and by eighteen to twenty in its
  lower half. A drainage estimate built on that traps everything by
  construction. Trapping is a sub-grid process the solver does not resolve, so
  it uses the melt viscosity regardless of the solver's choice, and the
  inconsistency is deliberate rather than overlooked.
* The Darcy velocity of McKenzie (1984) carries a factor ``(1 - phi)`` that the
  solver's ``relative_velocity`` omits, because the solver wants the melt
  velocity relative to the solid rather than the melt flux relative to the
  mixture. The factor is applied here; it is 30 percent at ``phi = 0.3``.

The permeability law is transcribed rather than imported because the interior
solver does not yet expose it as a public function. ``test_compaction.py`` pins
this transcription against the solver's own ``relative_velocity``, so the two
cannot drift apart silently. Replace the transcription with the import once the
solver exposes a mobility function.

A caution on calibration
------------------------
Sim et al. (2024) and Hier-Majumder and Hirschmann (2017) evaluate their drag
coefficient with the tubule permeability of Hier-Majumder (2011),
``k = a^2 phi^2 / (72 pi)``, at a reference porosity of 0.05. That law scales as
``phi^2`` where the Blake-Kozeny-Carman branch used here scales as ``phi^3``,
and at their calibration point it gives a permeability roughly eighty times
larger. A timescale computed here is therefore a different estimate on a
different microphysical model, not the published number reproduced with the
solver's inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger('fwl.' + __name__)

# Regime boundaries of the three-part permeability, Bower et al. (2018)
# Eqs. 13a-c, matching the interior solver exactly. The upper crossing is
# exact; the lower sits 1.7e-5 below the analytic Blake-Kozeny-Carman to
# Rumpf-Gupte crossing. The blend widths are numerical smoothing, not physics.
MOBILITY_BOUND_LOW = 0.0769452
MOBILITY_BOUND_HIGH = 0.771462
MOBILITY_BLEND_LOW = 0.02
MOBILITY_BLEND_HIGH = 0.05

# Mush viscosity entering the matrix deformation time [log10 Pa s] when the
# configuration leaves it unset. The literature spans 1e18 to 1e22 Pa s for a
# mush near the solidus; this is the middle of that range.
DEFAULT_MUSH_LOG10VISC = 20.0

# Branch codes reported on every step, so a trajectory shows which regime it
# was in and how often the guards fired.
BRANCH_NONE = 0  # no trapping: mode off, warming, or no front
BRANCH_DARCY = 1  # dynamic, percolation-limited at the top of the front
BRANCH_MATRIX = 2  # dynamic, matrix-deformation-limited
BRANCH_GUARD = 3  # upper bound: front unresolved, doubled, too thick, or dense melt
BRANCH_UNUSABLE = 4  # first iteration, zero-length step, or remesh
BRANCH_PUBLISHED = 5  # constant or fixed-tau mode
BRANCH_FALLBACK = 6  # drainage requested but the interior state was unavailable


@dataclass
class FrontGeometry:
    """The freezing front located on the interior mesh at one instant."""

    index: np.ndarray  # staggered-node indices spanning the front
    r_top: float  # radius where the melt fraction crosses rfront_loc [m]
    r_base: float  # radius where the porosity falls to phi_min [m]
    thickness: float  # r_top - r_base [m]
    porosity_top: float  # porosity at the top of the front [1]
    delta_rho: float  # solid minus melt density across the front [kg m-3]
    rho_melt: float  # mean melt density across the front [kg m-3]
    rho_solid_base: float  # solid density at the base of the front [kg m-3]
    dense_melt: bool  # True where the melt is locally the denser phase
    guard_reason: str = ''  # why the front cannot be integrated; empty when it can


def mobility_function(porosity: float | np.ndarray, grain_size: float) -> float | np.ndarray:
    """Permeability over porosity, ``F = K / phi`` [m2].

    The three-regime form of Bower et al. (2018) section 2.1, blended with the
    same tanh weights the interior solver uses:

    * Blake-Kozeny-Carman, low porosity: ``a^2 phi^2 / ((1-phi)^2 * 1000)``
    * Rumpf-Gupte, intermediate:         ``a^2 phi^4.5 * 5/7``
    * Stokes settling, high porosity:    ``a^2 * 2/9``

    Parameters
    ----------
    porosity : float or ndarray
        Volume fraction of melt [1].
    grain_size : float
        Crystal grain size [m].
    """
    if grain_size <= 0.0:
        raise ValueError(f'grain_size must be positive, got {grain_size!r}')
    phi = np.asarray(porosity, dtype=float)
    por = np.maximum(phi, 1.0e-20)
    one_m_por = np.maximum(1.0 - phi, 1.0e-20)

    f_bkc = grain_size**2 * por**2 / (one_m_por**2 * 1000.0)
    f_rg = grain_size**2 * por**4.5 * (5.0 / 7.0)
    f_stokes = grain_size**2 * 2.0 / 9.0

    w_rg = 0.5 * (1.0 + np.tanh((phi - MOBILITY_BOUND_LOW) / MOBILITY_BLEND_LOW))
    w_st = 0.5 * (1.0 + np.tanh((phi - MOBILITY_BOUND_HIGH) / MOBILITY_BLEND_HIGH))
    f = (1.0 - w_rg) * f_bkc + (w_rg - w_st) * f_rg + w_st * f_stokes
    f = np.maximum(f, 0.0)
    return float(f) if np.ndim(porosity) == 0 else f


def darcy_velocity(
    porosity: float | np.ndarray,
    grain_size: float,
    delta_rho: float,
    gravity: float,
    melt_visc: float,
) -> float | np.ndarray:
    """Melt percolation speed relative to the matrix [m s-1].

    ``w_D = (1 - phi) |delta_rho| g F(phi) / eta_melt``. The ``(1 - phi)`` factor
    is the one the solver's ``relative_velocity`` omits; see the module
    docstring.
    """
    if melt_visc <= 0.0:
        raise ValueError(f'melt viscosity must be positive, got {melt_visc!r}')
    phi = np.asarray(porosity, dtype=float)
    f = mobility_function(phi, grain_size)
    w = (1.0 - phi) * abs(delta_rho) * gravity * f / melt_visc
    w = np.maximum(w, 0.0)
    return float(w) if np.ndim(porosity) == 0 else w


def matrix_time(mush_visc: float, delta_rho: float, gravity: float, thickness: float) -> float:
    """Matrix deformation time [s], McKenzie (1984) compaction viscosity.

    ``tau_s = mu_s / (delta_rho g L)``. The compaction pressure ``delta_rho g L``
    drives a volumetric strain rate against the bulk viscosity ``zeta = mu_s/phi``,
    and closing a porosity ``phi`` takes ``phi zeta / (delta_rho g L)``, which is
    independent of porosity.
    """
    denom = abs(delta_rho) * gravity * thickness
    if denom <= 0.0 or not np.isfinite(denom):
        return float('inf')
    return float(mush_visc) / denom


def drainage_integral(
    porosity_top: float,
    t_res: float,
    thickness: float,
    grain_size: float,
    delta_rho: float,
    gravity: float,
    melt_visc: float,
    mush_visc: float,
) -> tuple[float, float, float]:
    """Porosity a parcel retains after crossing the front [1].

    Integrates ``dphi/dt = -phi / max(L/w_D(phi), tau_s)`` from ``porosity_top``
    over the residence time. Bounded in ``[0, porosity_top]`` without a clamp,
    because the porosity decays but never reaches zero: the drainage is
    self-limiting, ``w_D`` falling as ``phi^2`` or steeper at low porosity.

    Returns
    -------
    tuple
        Retained volume fraction, the percolation time at the entry porosity
        [s], and the matrix deformation time [s].
    """
    from scipy.integrate import solve_ivp

    tau_s = matrix_time(mush_visc, delta_rho, gravity, thickness)
    w_top = darcy_velocity(porosity_top, grain_size, delta_rho, gravity, melt_visc)
    tau_d_top = thickness / w_top if w_top > 0.0 else float('inf')

    if not np.isfinite(t_res) or t_res <= 0.0 or porosity_top <= 0.0:
        return float(max(porosity_top, 0.0)), tau_d_top, tau_s

    def rhs(_t, y):
        phi = max(float(y[0]), 0.0)
        w = darcy_velocity(phi, grain_size, delta_rho, gravity, melt_visc)
        tau_d = thickness / w if w > 0.0 else float('inf')
        # The slower process controls: melt leaves only if it can both
        # percolate through the pores and have the matrix deform to close
        # them, so the drainage time is max(tau_D, tau_s), not min.
        rate = max(tau_d, tau_s)
        if not np.isfinite(rate) or rate <= 0.0:
            return [0.0]
        return [-phi / rate]

    sol = solve_ivp(
        rhs,
        (0.0, float(t_res)),
        [float(porosity_top)],
        method='LSODA',
        rtol=1.0e-8,
        atol=1.0e-12,
    )
    if not sol.success:
        log.warning(
            'Drainage integral did not converge (%s); retaining the entry '
            'porosity as an upper bound.',
            sol.message,
        )
        return float(porosity_top), tau_d_top, tau_s
    retained = float(sol.y[0, -1])
    return float(min(max(retained, 0.0), porosity_top)), tau_d_top, tau_s


def porosity_from_densities(
    rho: np.ndarray, rho_solid: np.ndarray, rho_melt: np.ndarray
) -> np.ndarray:
    """Volume fraction of melt from the mixture and end-member densities [1].

    ``phi = (rho_s - rho) / (rho_s - rho_l)``, the same lever rule on density
    the interior solver applies, clipped to [0, 1]. Where the tabulated melt is
    locally denser than the solid the denominator is floored, which drives the
    porosity to zero there rather than inverting it; those nodes are reported
    separately by :func:`locate_front` and take the guard branch.
    """
    drho = np.asarray(rho_solid, dtype=float) - np.asarray(rho_melt, dtype=float)
    drho = np.where(drho > 1.0e-6, drho, 1.0e-6)
    phi = (np.asarray(rho_solid, dtype=float) - np.asarray(rho, dtype=float)) / drho
    return np.clip(phi, 0.0, 1.0)


def _contiguous_runs(mask: np.ndarray) -> list[np.ndarray]:
    """Index runs of consecutive True values."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)


def _interp_crossing(r: np.ndarray, field: np.ndarray, value: float, k: int) -> float:
    """Radius where ``field`` crosses ``value`` on the interval next to node k."""
    k = int(np.clip(k, 0, r.size - 2))
    f0, f1 = float(field[k]), float(field[k + 1])
    if not np.isfinite(f0) or not np.isfinite(f1) or f1 == f0:
        return float(r[k])
    frac = (value - f0) / (f1 - f0)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(r[k] + frac * (r[k + 1] - r[k]))


def locate_front(
    radius: np.ndarray,
    melt_fraction: np.ndarray,
    porosity: np.ndarray,
    rho_solid: np.ndarray,
    rho_melt: np.ndarray,
    rfront_loc: float,
    phi_min: float,
    n_front_min: int = 3,
    max_front_fraction: float = 1.0,
    r_floor: float | None = None,
) -> tuple[FrontGeometry | None, int]:
    """Locate the freezing front on the interior mesh.

    The front runs from the node where the solver melt fraction crosses
    ``rfront_loc`` at the top down to the node where the porosity falls to
    ``phi_min`` at the base. Using the solver's own melt fraction and its own
    rheological transition for the top means the front trapping uses is the
    front the solver uses for its rheology, rather than a second threshold that
    could disagree with it.

    A mantle crystallising from the bottom up is often still porous at its
    lowest node. The front then rests on the core-mantle boundary, which is
    impermeable, and that boundary is its base: the melt in it drains upward
    only, as it does from a front sitting on compacted cumulate.

    Parameters
    ----------
    r_floor : float or None
        Radius of the impermeable lower boundary of the mantle [m], the base
        of a front that is still porous at the lowest node. Defaults to the
        lowest node radius. Must not lie above it.

    Returns
    -------
    tuple
        The geometry and a branch code. The geometry is ``None`` only when no
        front exists. A geometry paired with :data:`BRANCH_GUARD` is returned
        when the front is present but split in two, too thin to resolve, too
        thick for the thin-front picture, reaching the surface node, or holding
        melt denser than the solid, with the cause in ``guard_reason``: the
        caller should then take the entry porosity as an upper bound rather than
        integrate.
    """
    r = np.asarray(radius, dtype=float)
    r_lo = float(r[0]) if r_floor is None else float(r_floor)
    if r_lo > float(r[0]):
        raise ValueError(
            f'r_floor = {r_lo:.6g} m lies above the lowest node at {float(r[0]):.6g} m; '
            'the lower boundary of the mantle cannot be inside the mesh'
        )
    phi_s = np.asarray(melt_fraction, dtype=float)
    por = np.asarray(porosity, dtype=float)
    mask = (por > phi_min) & (phi_s < rfront_loc)
    runs = _contiguous_runs(mask)
    if not runs:
        return None, BRANCH_NONE

    index = runs[0]
    top, base = int(index[-1]), int(index[0])
    drho_arr = np.asarray(rho_solid, dtype=float) - np.asarray(rho_melt, dtype=float)
    dense = bool(np.nanmin(drho_arr[index]) <= 0.0)

    geom = FrontGeometry(
        index=index,
        r_top=float(r[top]),
        r_base=float(r[base]),
        thickness=float(r[top] - r[base]),
        porosity_top=float(por[top]),
        delta_rho=float(np.nanmean(drho_arr[index])),
        rho_melt=float(np.nanmean(np.asarray(rho_melt, dtype=float)[index])),
        rho_solid_base=float(np.asarray(rho_solid, dtype=float)[base]),
        dense_melt=dense,
    )

    reasons = []
    if len(runs) > 1:
        reasons.append(f'the mush is split into {len(runs)} separate layers')
    if index.size < n_front_min:
        reasons.append(
            f'the front spans {index.size} nodes, fewer than the {n_front_min} '
            'needed to resolve it'
        )
    if top >= r.size - 1:
        reasons.append('the front reaches the surface node')
    if dense:
        reasons.append('the melt is denser than the solid inside the front')
    if reasons:
        geom.guard_reason = '; '.join(reasons)
        return geom, BRANCH_GUARD

    geom.r_top = _interp_crossing(r, phi_s, rfront_loc, top)
    if base == 0:
        geom.r_base = r_lo
    else:
        geom.r_base = _interp_crossing(r, por, phi_min, base - 1)
    geom.thickness = geom.r_top - geom.r_base
    span = float(r[-1] - r_lo)
    if geom.thickness <= 0.0:
        geom.guard_reason = 'the front has no positive thickness'
        return geom, BRANCH_GUARD
    if span > 0.0 and geom.thickness > max_front_fraction * span:
        geom.guard_reason = (
            f'the front spans {geom.thickness / span:.0%} of the mantle, above the '
            f'{max_front_fraction:.0%} a thin front allows'
        )
        return geom, BRANCH_GUARD
    return geom, BRANCH_DARCY


def volume_to_mass_fraction(f_vol: float, rho_melt: float, rho_solid: float) -> float:
    """Convert a trapped melt volume fraction to the mass fraction the budget uses.

    ``F_tl`` is a volume fraction in the compaction physics, where the
    disaggregation limit is a geometric packing threshold, and a mass fraction in
    the volatile budget. The two differ by the melt-to-mixture density ratio,
    about 0.93 at a ten percent density contrast and a porosity of 0.3. The
    conversion is applied once, here, and only on the dynamic paths: the
    constant and fixed-tau modes report the published law's fraction unconverted
    so they match the papers and the offline calculator.
    """
    f = float(np.clip(f_vol, 0.0, 1.0))
    denom = f * rho_melt + (1.0 - f) * rho_solid
    if denom <= 0.0 or not np.isfinite(denom):
        return f
    return float(f * rho_melt / denom)
