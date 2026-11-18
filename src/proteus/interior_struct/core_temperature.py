"""Core temperature table for the Zalmoxis structure solve.

Zalmoxis applies a core temperature T(P) to every node of the core. PROTEUS gives it as
``(P, T)`` pairs on the adiabat of an Aragog core profile through an anchor temperature at the CMB:
the live Aragog core with ``core_bc = "core_module"``, and a Gaussian profile fitted to the last
structure with ``core_bc = "energy_balance"``.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

# Pairs per table: uniform in radius from the centre to the CMB of the profile
CORE_T_POINTS = 161


def core_temperature_table(profiles, t_anchor, n=CORE_T_POINTS):
    """Pairs (P, T) of the core adiabat through ``t_anchor`` at the CMB, P ascending.

    Parameters
    ----------
    profiles : aragog.core.profiles.GaussianCoreProfiles or TabulatedCoreProfiles
        Core profiles with ``r_cmb`` [m], ``pressure(r)`` [Pa] and ``adiabat(r, t_cmb)`` [K].
    t_anchor : float
        Temperature at the CMB of the adiabat [K], finite and positive.
    n : int, optional
        Number of pairs, uniform in radius from the centre to ``profiles.r_cmb``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(P, T)``: pressure [Pa] strictly ascending from the CMB to the centre, temperature [K].

    Raises
    ------
    ValueError
        When ``t_anchor`` is not a finite positive number (a bool is refused), or the pairs are
        not finite, not strictly ascending in P or not positive in T.
    """
    if not (
        isinstance(t_anchor, numbers.Real)
        and not isinstance(t_anchor, bool)
        and math.isfinite(t_anchor)
        and t_anchor > 0
    ):
        raise ValueError(
            f'core temperature anchor must be a finite positive number, got {t_anchor!r}'
        )
    r = np.linspace(0.0, float(profiles.r_cmb), n)
    p = np.asarray(profiles.pressure(r), dtype=float)[::-1]
    t = np.asarray(profiles.adiabat(r, float(t_anchor)), dtype=float)[::-1]
    if not (
        np.all(np.isfinite(p))
        and np.all(np.isfinite(t))
        and np.all(np.diff(p) > 0)
        and np.all(t > 0)
    ):
        raise ValueError(
            'core temperature pairs are not finite, ascending in P and positive in T'
        )
    return p, t
