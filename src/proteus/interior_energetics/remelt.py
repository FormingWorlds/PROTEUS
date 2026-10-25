"""Partial giant-impact re-melt of the mantle: the outer-shell rule.

An impact melts the mass fraction f of the mantle. The rule raises the
outermost shells that hold f times the mantle mass to the initial condition
and keeps the end-of-step state below them; the shell that holds the
boundary takes the mass-weighted mean of the two. This is the 1D radial
limit of a global magma ocean of uniform depth. f = 1 is the full re-melt,
f = 0 leaves the state unchanged, and no cell cools.

``remelt_weights`` holds the geometry. A melt-pool model replaces it with a
function of the same output contract: a weight in [0, 1] per cell, the
closure sum(w m) = f M, and w = 1 in every cell at f = 1.

The rule has no memory of earlier melt: each impact fills its budget f M from
the surface down, also where the shells already hold the initial condition.
Two impacts of f = 0.3 in one step melt 0.3 of the mantle plus w (1 - w) of the
boundary cell's mass, which the second impact blends again (at most a quarter
of one cell). The scalar mantle compounds: two impacts move it 1 - (1 - f)^2 of
the lift.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

# Relative closure of the melted mass sum(w m) against f M.
REMELT_CLOSURE_RTOL = 1e-10


def check_melt_fraction(f) -> float:
    """Return f as a float, or raise if it is not a finite real number in [0, 1]."""
    if isinstance(f, bool) or not isinstance(f, numbers.Real) or not 0.0 <= f <= 1.0:
        raise ValueError(f'melt fraction must be a finite real number in [0, 1], got {f!r}')
    return float(f)


def remelt_weights(f: float, cell_mass: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Melted mass share of each cell, filled from the surface down.

    Parameters
    ----------
    f : float
        Melted mass fraction of the mantle [1], in [0, 1].
    cell_mass : np.ndarray
        Mass of each mantle cell [kg], finite and positive, in any order.
    radii : np.ndarray
        Radius of each cell [m], same shape as ``cell_mass``; the cell with the
        largest radius melts first.

    Returns
    -------
    np.ndarray
        Weight w_i in [0, 1] per cell, in the order of ``cell_mass``, with
        sum(w m) = f sum(m).

    Raises
    ------
    ValueError
        If f is not a finite number in [0, 1], or a cell mass or radius is
        invalid.
    RuntimeError
        If the weights miss the melted mass closure.
    """
    f = check_melt_fraction(f)
    m = np.asarray(cell_mass, dtype=float)
    r = np.asarray(radii, dtype=float)
    if m.ndim != 1 or m.size == 0 or r.shape != m.shape:
        raise ValueError(f'cell masses {m.shape} and radii {r.shape} must be 1-D and match')
    if not (np.all(np.isfinite(m)) and np.all(m > 0.0) and np.all(np.isfinite(r))):
        raise ValueError('cell masses must be finite and positive and cell radii finite')
    if f == 1.0:
        return np.ones_like(m)

    order = np.argsort(-r, kind='stable')
    m_sorted = m[order]
    above = np.concatenate(([0.0], np.cumsum(m_sorted)[:-1]))
    total = float(m.sum())
    w = np.empty_like(m)
    w[order] = np.clip((f * total - above) / m_sorted, 0.0, 1.0)

    # A subnormal weight carries few bits, hence the 1-ulp-of-zero term.
    tol = REMELT_CLOSURE_RTOL * f * total + 5e-324 * float(m.max())
    miss = abs(float(np.sum(w * m)) - f * total)
    if miss > tol:
        raise RuntimeError(f'melted mass misses f M by {miss:.3e} kg (tolerance {tol:.3e} kg)')
    return w


def remelt_entropy(S_end: np.ndarray, S_ic: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Re-melted entropy profile for the cell weights w.

    A fully melted cell takes ``max(S_end, S_ic)``, the full re-melt; a cell
    melted by w < 1 moves the share w of the way up to the initial condition,
    the mass-weighted mean specific entropy of its melted and unmelted parts. That
    cell holds less heat than its parts, a deficit first order in dT/dS: for T(S)
    linear over the cell it is w (1 - w) m dS dT / 2, with dS = S_ic - S_end and dT
    the temperature rise over dS. Relative to the injected heat it is largest when
    all the melt fits in the top cell. The booked impact heat integrates the actual
    profile, so the energy budget closes.

    Parameters
    ----------
    S_end : np.ndarray
        End-of-step specific entropy per cell [J kg-1 K-1].
    S_ic : np.ndarray
        Initial-condition specific entropy per cell [J kg-1 K-1].
    w : np.ndarray
        Melted mass share per cell [1], in [0, 1].

    Returns
    -------
    np.ndarray
        Re-melted specific entropy per cell [J kg-1 K-1], never below ``S_end``.

    Raises
    ------
    ValueError
        If an entropy is non-finite, or a weight is not finite or lies outside [0, 1].
    """
    S_end = np.asarray(S_end, dtype=float)
    S_ic = np.asarray(S_ic, dtype=float)
    w = np.asarray(w, dtype=float)
    if not (np.all(np.isfinite(S_end)) and np.all(np.isfinite(S_ic))):
        raise ValueError('a re-melt entropy is non-finite')
    if not np.all(np.isfinite(w) & (w >= 0.0) & (w <= 1.0)):
        raise ValueError('every re-melt weight must be finite and in [0, 1]')
    S_new = np.where(
        w >= 1.0, np.maximum(S_end, S_ic), S_end + w * np.maximum(S_ic - S_end, 0.0)
    )
    return S_new


def remelt_temperature(T_end: float, T_init: float, f: float) -> float:
    """Re-melted temperature of a well-mixed scalar mantle [K].

    The 0D projection of the outer-shell rule for a box of constant heat
    capacity without latent heat: the share f is raised to ``T_init`` and
    mixed, so T = T_end + f (T_init - T_end). f = 1 gives ``max(T_init, T_end)``,
    the full re-melt, and a mantle above ``T_init`` keeps its temperature.
    The dummy interior holds one temperature for mantle and core (its heat
    capacity includes the core), so f also heats the core; f is not rescaled
    for it, so f = 1 stays the full re-melt.

    Parameters
    ----------
    T_end : float
        End-of-step mantle temperature [K].
    T_init : float
        Initial mantle temperature [K].
    f : float
        Melted mass fraction [1], in [0, 1].

    Returns
    -------
    float
        Re-melted mantle temperature [K].

    Raises
    ------
    ValueError
        If f is not a finite number in [0, 1], or a temperature is non-finite.
    """
    f = check_melt_fraction(f)
    if not (math.isfinite(T_end) and math.isfinite(T_init)):
        raise ValueError(f'a re-melt temperature is non-finite: {T_end!r}, {T_init!r}')
    if f == 1.0:
        return max(T_init, T_end)
    return T_end + f * max(T_init - T_end, 0.0)
