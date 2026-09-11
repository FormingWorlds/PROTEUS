"""Hansen Coefficients via Interpolated Tables

Computes Hansen coefficients using pre-computed, linearly interpolated lookup
tables instead of per-call FFTs.

#### Motivation

Implicit ODE solvers frequently evaluate right-hand sides at micro-varying
eccentricities (e). Running FFTs on-the-fly takes days over long integrations.
Nearest-neighbor caching creates step-function discontinuities that break
implicit solvers. Linear interpolation over a fixed grid provides smooth
derivatives with O(1) query times and zero FFTs during integration.

#### Hansen Mode Windowing

The required mode range [k_{min}, k_{max}] expands dramatically with eccentricity
(e.g., ~10 modes near e=0, several hundred at e > 0.8). Using dynamic windows
prevents wasting compute at low e and avoids silently truncating energy at high e.

1. `_k_range_table` (`_KRangeTable`): Pre-tabulates [k_{min}, k_{max}] by calling
   `hansen_fft` over a wide search window to find where X_k drops below tolerance.
   Queried via `kmin_kmax_for_e(e)`.
2. `_hansen_table`  (`_HansenTable`): Stores pre-computed coefficients across the
   global [k_{min}, k_{max}] envelope derived from `_k_range_table`.
   Queried via `get_all_m_hansen(e)`.

#### Production Flow

* Warm-up:  `orbit/wrapper.py` calls both `init_*` functions up front to offload the
  single ~1 minute FFT generation phase before timing-critical ODE substeps run.
* Fallback: Hot-path calls to `get_all_m_hansen` will lazily construct missing tables
  if the explicit initialization was skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftshift

log = logging.getLogger('fwl.' + __name__)


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
    Valid up to e ~ 0.9, fails to converge for e > 0.9.

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
    if e >= 0.90:
        log.warning(
            f"Eccentricity e={e:.4f} >= 0.90 exceeds stable convergence bound. "
            "Results near pericenter may lose precision."
        )

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
    v = np.arctan2(np.sqrt(1 - e**2) * se, ce - e)  # true anomaly

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
_DEFAULT_E_GRID = np.concatenate(
    [
        np.arange(0.0, 0.05, 0.005),
        np.arange(0.05, 0.85, 0.01),
        np.arange(0.85, 0.90, 0.005),
    ]
)


def _select_k_range(
    e: float, threshold: float = 0.001, k_search_max: int = 450, pad: int = 2
) -> tuple[int, int]:
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


def init_k_range_table(
    e_grid: Optional[NDArray[np.floating]] = None, force: bool = False
) -> None:
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
    log.info(
        f'k-range table built: {len(e_grid)} grid points'
        f'(e in [{e_grid.min():.3f}, {e_grid.max():.3f}], '
        f'n_modes in [{(kmaxs - kmins + 1).min()}, {(kmaxs - kmins + 1).max()}])'
    )


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


def padded_k_range_for_evection(
    e_now: float,
    de_dt_yr: float,
    dt_next_yr: float,
    padding_factor: float = 1.0,
    e_cap: float | None = None,
) -> tuple[int, int]:
    """[kmin, kmax] appropriate for where eccentricity is headed over the
    NEXT macro-step, not just where it is right now."""
    if e_cap is None:
        e_cap = float(_DEFAULT_E_GRID[-1])

    e_pad = min(
        e_cap,
        max(0.0, e_now) + abs(padding_factor) * abs(de_dt_yr) * max(dt_next_yr, 0.0),
    )
    return kmin_kmax_for_e(e_pad)


def init_hansen_table(
    e_grid: Optional[NDArray[np.floating]] = None,
    kmin: Optional[int] = None,
    kmax: Optional[int] = None,
    n_deg: int = 2,
    force: bool = False,
) -> None:
    """Build the Hansen-coefficient value table once, over `e_grid` and
    [kmin, kmax].

    Safe to call more than once: a no-op unless `force=True`. If kmin/kmax
    are not given, they are derived from the k-range table's own realized
    bounds (building it first if needed) -- this keeps the two tables
    consistent by construction rather than by a hand-picked guess.
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
    _hansen_table = _HansenTable(
        e_grid=e_grid, kmin=kmin, kmax=kmax, n_deg=n_deg, values=values
    )
    log.info(
        f'Hansen table built: {len(e_grid)} e-points x {n_k} k-modes x '
        f'{2 * n_deg + 1} m-branches'
    )


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
        log.warning(
            f"Requested k-range [{kmin}, {kmax}] exceeds pre-tabulated "
            f"[{table.kmin}, {table.kmax}]; results will be truncated."
        )
        kmin = max(kmin, table.kmin)
        kmax = min(kmax, table.kmax)

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
