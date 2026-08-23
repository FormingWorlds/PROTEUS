# Common tides model functions
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import netCDF4 as nc
import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftshift
from scipy.optimize import brentq

from proteus.config import Config
from proteus.interior_energetics.common import Interior_t

log = logging.getLogger("fwl."+__name__)


@dataclass
class TidalInteraction:
    primary: Any
    perturber: Any

    nmk: Optional[NDArray[np.floating]] = None
    sigma: Optional[NDArray[np.floating]] = None
    LNk: Optional[NDArray[np.floating]] = None


@dataclass
class Tides_t:
    interactions: List[TidalInteraction] = field(default_factory=list)

    def add(self, primary, perturber):
        try:
            return self.get(primary, perturber)
        except KeyError:
            interaction = TidalInteraction(primary, perturber)
            self.interactions.append(interaction)
            return interaction

    def get(self, primary, perturber):
        for interaction in self.interactions:
            if interaction.primary == primary and interaction.perturber == perturber:
                return interaction
        raise KeyError(f"No tidal interaction: {primary} <- {perturber}")

    def add_from_file(self, primary, perturber, file_path: str):
        interaction = self.add(primary, perturber)

        with nc.Dataset(file_path, 'r') as ds:
            n = ds.variables["n"][:]
            m = ds.variables["m"][:]
            k = ds.variables["k"][:]

            interaction.nmk = np.column_stack([n, m, k]).astype(int)
            interaction.sigma = ds.variables['sigma'][:]
            interaction.LNk = (
                ds.variables["LNk_real"][:]
                + 1j * ds.variables["LNk_imag"][:]
            )

        return interaction


# ---------------------------------------------------------------------------
# Hansen coefficients X_k^{n,m}(e), computed via FFT on the mean anomaly and
# cached so that the (expensive) FFTs only ever run once per PROTEUS
# process, not once per RHS evaluation.
#
# Caching strategy, and why: Hansen coefficients depend only on eccentricity
# (for a fixed degree n), which changes slowly relative to how many times an
# implicit orbital-evolution solver evaluates its right-hand side (Newton
# iterations, stage evaluations, and rejected trial steps all re-ask for the
# same or very similar e). A per-call memoization cache keyed on rounded e
# was tried and rejected: even at a 99% hit rate, the solver's own internal
# evaluations explored enough distinct e values that a 200 yr test still
# needed ~2500 fresh FFTs (~0.5s each -- five per e, one per m branch) --
# extrapolated to a full evolutionary run, that is many hours to days.
#
# Instead, get_all_m_hansen() below is backed by a table, built once over a
# grid of eccentricities and linearly interpolated at query time -- zero
# FFTs during the actual integration, and smooth by construction (Hansen
# coefficients have no sharp features in e, so linear interpolation is a
# good approximation and, unlike a memoized/rounded cache, never turns the
# right-hand side an implicit solver sees into a discontinuous step
# function of e).
#
# The k-range (which modes carry non-negligible weight) grows sharply with
# e: at e~0.7, the k=2 mode that dominates at e~0 has effectively vanished,
# while modes out past k~55 still carry >=1% weight. kmin_kmax_for_e()
# tabulates, once, the eccentricity-appropriate [kmin, kmax] window so
# callers only sum over the modes that actually matter at the current e
# (order 10 near e=0, growing to several hundred at e>0.8), instead of a
# fixed window that both wastes effort at low e and silently drops most of
# the tidal coupling at high e.
# ---------------------------------------------------------------------------


def nextpow2_int(x):
    """Return the integer p such that 2^p >= x.

    Attributes
    ----------
    x : int
        Input value.

    Returns
    -------
    p : int
        The smallest integer p such that 2^p >= x.
    """
    return int(np.ceil(np.log2(x))) if x > 0 else 0


def kepler_newton(M, e):
    """
    Solve Kepler's equation E - e*sin(E) = M using Newton iteration.

    Attributes
    ----------
    M : array_like
        Mean anomaly in radians.
    e : float
        Orbital eccentricity (0 <= e < 1).

    Returns
    -------
    E : ndarray
        Eccentric anomaly in radians, same shape as M.
    """
    M = np.array(M, dtype=float)
    E = np.copy(M)

    # Danby-style improved initial guess
    if e > 0:
        E = M + (e * np.sin(M)) / (1 - np.sin(M + e) + np.sin(M))

    # Newton iterations
    for _ in range(10):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if np.max(np.abs(dE)) < 1e-13:
            break

    return np.mod(E, 2 * np.pi)


def hansen_fft(n, m, e, kmin, kmax, N=None):
    """Compute Hansen coefficients X_k^{n,m}(e) using FFT on mean anomaly.

    Attributes
    ----------
    n : int
        Degree of the Hansen coefficient.
    m : int
        Order of the Hansen coefficient.
    e : float
        Orbital eccentricity (0 <= e < 1).
    kmin : int
        Minimum k value for which to compute the coefficient.
    kmax : int
        Maximum k value for which to compute the coefficient.
    N : int, optional
        Number of points for FFT. If None, it will be chosen adaptively.

    Returns
    -------
    k : ndarray
        Array of k values from kmin to kmax.
    Xkm : ndarray
        Corresponding Hansen coefficients X_k^{n,m}(e).
    """
    # Choose FFT size adaptively
    if N is None:
        width = max(64, 4 * (kmax - kmin + 1))
        target = width * max(8, int(np.ceil(16 / (1 - e + np.finfo(float).eps))))
        p = max(12, int(np.ceil(np.log2(target))))
        N = 2**p
    else:
        p = nextpow2_int(N)
        N = 2**p

    # Mean anomaly grid
    M = np.arange(N) * (2 * np.pi / N)

    # Solve Kepler
    E = kepler_newton(M, e)

    ce = np.cos(E)
    se = np.sin(E)
    r_over_a = 1 - e * ce
    v = np.arctan2(np.sqrt(1 - e**2) * se, ce - e) # true anomaly

    # Hansen integrand
    f = (r_over_a**n) * np.exp(1j * m * v)

    # FFT, normalized like Python’s fft(f)/N
    F = fftshift(fft(f.astype(complex))) / N

    k_all = np.arange(-N // 2, N // 2)
    mask = (k_all >= kmin) & (k_all <= kmax)

    k = k_all[mask]
    Zk = F[mask]
    Xkm = np.real(Zk)

    return k, Xkm


@dataclass
class _HansenTable:
    """Tabulated Hansen coefficients X_k^{n,m}(e), n fixed, over an
    eccentricity grid and a fixed [kmin, kmax] window, for fast linear
    interpolation. Built once by init_hansen_table(); never rebuilt except
    via force=True."""
    e_grid: NDArray[np.floating]
    kmin: int
    kmax: int
    n_deg: int
    values: Dict[int, NDArray[np.floating]]  # m -> array[len(e_grid), kmax-kmin+1]


@dataclass
class _KRangeTable:
    """Tabulated eccentricity-appropriate [kmin, kmax] window. Built once by
    init_k_range_table(); never rebuilt except via force=True."""
    e_grid: NDArray[np.floating]
    kmin: NDArray[np.integer]
    kmax: NDArray[np.integer]


_hansen_table: Optional[_HansenTable] = None
_k_range_table: Optional[_KRangeTable] = None

# Default eccentricity grid shared by both tables: fine near e=0 (where
# Hansen coefficients vary fastest in relative terms) and coarser at high e.
_DEFAULT_E_GRID = np.concatenate([
    np.arange(0.0, 0.05, 0.005),
    np.arange(0.05, 0.90, 0.01),
    np.arange(0.90, 0.951, 0.005),
])

def _select_k_range(e: float, threshold: float = 0.01,
                     k_search_max: int = 450, pad: int = 2) -> tuple[int, int]:
    """Widest [kmin, kmax] (padded) such that the m=0 and m=2 Hansen
    branches (the dissipative/heating-relevant ones) both have |X_k| below
    `threshold` everywhere outside it."""
    lo_all, hi_all = [], []
    for m in (0, 2):
        k, X = hansen_fft(-3, m, e, -k_search_max, k_search_max)
        above = k[np.abs(X) >= threshold]
        if len(above) == 0:
            lo_all.append(-2)
            hi_all.append(4)
        else:
            lo_all.append(above.min())
            hi_all.append(above.max())
    kmin = min(min(lo_all), -2) - pad
    kmax = max(max(hi_all), 4) + pad
    return int(kmin), int(kmax)


def init_k_range_table(e_grid: Optional[NDArray[np.floating]] = None,
                        force: bool = False) -> None:
    """Build the eccentricity -> [kmin, kmax] lookup table once.

    Safe to call more than once: a no-op unless `force=True`, so callers
    don't need to track whether this has already run.
    """
    global _k_range_table
    if _k_range_table is not None and not force:
        return

    e_grid = _DEFAULT_E_GRID if e_grid is None else np.asarray(e_grid, dtype=float)
    kmins = np.empty(len(e_grid), dtype=int)
    kmaxs = np.empty(len(e_grid), dtype=int)
    for i, e in enumerate(e_grid):
        kmins[i], kmaxs[i] = _select_k_range(e)
    _k_range_table = _KRangeTable(e_grid=e_grid, kmin=kmins, kmax=kmaxs)
    log.info(f"k-range table built: {len(e_grid)} grid points"
             f"(e in [{e_grid.min():.3f}, {e_grid.max():.3f}], "
             f"n_modes in [{(kmaxs-kmins+1).min()}, {(kmaxs-kmins+1).max()}])")


def kmin_kmax_for_e(e: float) -> tuple[int, int]:
    """Eccentricity-appropriate [kmin, kmax] window. Lazily builds the
    lookup table (with default settings) on first use if it hasn't been
    built yet, so this is safe to call without any setup step."""
    if _k_range_table is None:
        init_k_range_table()
    table = _k_range_table
    e = min(max(e, 0.0), table.e_grid[-1])
    idx = np.searchsorted(table.e_grid, e, side='right') - 1
    idx = min(max(idx, 0), len(table.e_grid) - 1)
    return int(table.kmin[idx]), int(table.kmax[idx])


def init_hansen_table(e_grid: Optional[NDArray[np.floating]] = None,
                       kmin: Optional[int] = None, kmax: Optional[int] = None,
                       n_deg: int = 2, force: bool = False) -> None:
    """Build the Hansen-coefficient value table once, over `e_grid` and
    [kmin, kmax].

    Safe to call more than once: a no-op unless `force=True`. If kmin/kmax
    are not given, they are derived from the k-range table's own realized
    bounds (building it first if needed) -- this keeps the two tables
    consistent by construction rather than by a hand-picked guess. (A fixed
    guess previously left a gap once e exceeded ~0.76, where the k-range
    table legitimately asked for a wider window than the value table
    covered; the resulting out-of-range slice did not raise an error, it
    silently returned a wrong-shaped empty array, which then broke with an
    unrelated-looking exception far from the actual cause. get_all_m_hansen
    below now raises a clear error in that situation instead.)
    """
    global _hansen_table
    if _hansen_table is not None and not force:
        return

    if kmin is None or kmax is None:
        init_k_range_table()
        kmin = int(_k_range_table.kmin.min()) if kmin is None else kmin
        kmax = int(_k_range_table.kmax.max()) if kmax is None else kmax

    e_grid = _DEFAULT_E_GRID if e_grid is None else np.asarray(e_grid, dtype=float)
    n_k = kmax - kmin + 1
    values = {m: np.zeros((len(e_grid), n_k)) for m in range(-n_deg, n_deg + 1)}

    for i, e in enumerate(e_grid):
        for m in range(-n_deg, n_deg + 1):
            _, X = hansen_fft(-(n_deg + 1), m, e, kmin, kmax)
            values[m][i, :] = X
    _hansen_table = _HansenTable(e_grid=e_grid, kmin=kmin, kmax=kmax, n_deg=n_deg, values=values)
    log.info(f"Hansen table built: {len(e_grid)} e-points x {n_k} k-modes x "
             f"{2*n_deg+1} m-branches")


def get_all_m_hansen(e: float, n_deg: int, kmin: int, kmax: int):
    """Hansen coefficients X_k^{n,m}(e) for all m = -n_deg..n_deg, by linear
    interpolation over the pre-tabulated values, sliced to [kmin, kmax].

    Lazily builds the table (with default settings) on first call if it
    hasn't been built yet -- this is what guarantees the expensive FFT
    sweep runs exactly once per process regardless of whether any setup
    code remembers to call init_hansen_table() explicitly: the first call
    (from wherever it happens to come) pays the one-time cost, and this
    function is called often (once per right-hand-side evaluation), so
    every call after that is a cheap array lookup, not a recomputation.

    Returns
    -------
    k_range : ndarray
        Array of k values from kmin to kmax.
    results : dict
        m -> ndarray of Hansen coefficients X_k^{n_deg,m}(e), same shape as k_range.
    """
    if _hansen_table is None:
        init_hansen_table(n_deg=n_deg)
    table = _hansen_table

    if kmin < table.kmin or kmax > table.kmax:
        raise ValueError(
            f"get_all_m_hansen: requested k-range [{kmin},{kmax}] at e={e:.4f} exceeds "
            f"the Hansen table's window [{table.kmin},{table.kmax}] -- rebuild with "
            f"init_hansen_table(kmin=..., kmax=..., force=True), or widen kmin/kmax "
            f"there to cover whatever kmin_kmax_for_e() can return."
        )

    e = min(max(e, 0.0), table.e_grid[-1])
    idx = np.searchsorted(table.e_grid, e, side='right') - 1
    idx = min(max(idx, 0), len(table.e_grid) - 2)
    e0, e1 = table.e_grid[idx], table.e_grid[idx + 1]
    w = 0.0 if e1 == e0 else (e - e0) / (e1 - e0)

    lo = kmin - table.kmin
    hi = kmax - table.kmin + 1
    k_range = np.arange(kmin, kmax + 1)
    results = {
        m: (1.0 - w) * values[idx, lo:hi] + w * values[idx + 1, lo:hi]
        for m, values in table.values.items()
    }
    return k_range, results


def get_C_planet(hf_row: dict, config: Config, interior_o: Interior_t):
    """Compute the planet's principal moment of inertia (C_planet) based on the interior structure.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        config : Config
            Model configuration.
        interior_o : Interior_t
            Interior object containing interior arrays
    """
    # Calculate the planet's principal moment of inertia (C_planet)
    # Assuming a spherically symmetric mass distribution, we can use the formula:
    # C = (8/3) * pi * integral_0^R (rho(r) * r^4 dr)
    # where rho(r) is the density profile and R is the radius of the planet.

    # Get the radial grid and density profile from the interior object
    arr_keys = ("density", "radius")
    lov = {k:np.array(getattr(interior_o, k), copy=True, dtype=float) for k in arr_keys}

    # Reverse arrays if using SPIDER
    #  Such that i=0 is at the CMB
    if config.interior_energetics.module == "spider":
        for k in arr_keys:
            lov[k] = lov[k][::-1]

    r_edges = lov["radius"]      # length N+1
    rho = lov["density"]         # length N

    r0 = r_edges[:-1]
    r1 = r_edges[1:]

    integral = np.sum(
        rho * (r1**5 - r0**5) / 5.0
    )

    C_planet = (8*np.pi/3.0) * integral

    # Store C_planet in the helpfile row for later use
    hf_row['C_planet'] = C_planet

    # Check if C_planet is physically reasonable
    C_factor_planet = C_planet / (hf_row['M_int'] * hf_row['R_int']**2)
    log.info(f"Computed C_planet: {C_planet:.3e} kg.m^2, C_factor_planet: {C_factor_planet:.3f}")


def _solve_e_stationary(a_prime, s_prime, Lambda, Omega_ratio):
    """Solve Rufu & Canup (2020), Eq. 12, for the stable stationary eccentricity e_s.

    Note:
    Rufu & Canup (2020) use the following normalization:
    - a' = a / R_p
    - s' = Omega / Omega_p
    - Lambda = sqrt(1.5 * J_star * Omega_p / Omega_star)
    - Omega_ratio = Omega_star / Omega_p

    where:
    Omega_p = sqrt(const_G * M_p / R_p**3)

    Parameters
    ----------
    a_prime : float
        Scaled semi-major axis (a / R_p).
    s_prime : float
        Scaled spin rate (Omega / Omega_p).
    Lambda : float
        Scaled angular momentum (L / (M_p * sqrt(G * M_p * R_p))).
    Omega_ratio : float
        Ratio of the planet's spin rate to the orbital mean motion (Omega / Omega_p).

    Returns
    -------
    e_s : float
        The stable stationary eccentricity, or np.nan if no solution exists.
    """
    if not (np.isfinite(a_prime) and np.isfinite(s_prime)):
        return np.nan

    def f(e):
        return (
            Lambda**2 * s_prime**2 / (a_prime**3.5 * (1.0 - e**2)**2)
            - 1.0
            - 3.0 * np.sqrt(1.0 - e**2) * a_prime**1.5 * Omega_ratio
        )

    lo, hi = 1e-8, 1.0 - 1e-8
    f_lo, f_hi = f(lo), f(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return np.nan
    return brentq(f, lo, hi)
