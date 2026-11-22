"""Core temperature table for the Zalmoxis structure solve.

Zalmoxis applies a core temperature T(P) to every node of the core. PROTEUS gives it as
``(P, T)`` pairs on the adiabat of an Aragog core profile through an anchor temperature at the CMB:
the live Aragog core with ``core_bc = "core_module"``, else (any other ``core_bc``, or SPIDER) a
Gaussian profile fitted to the last structure.
"""

from __future__ import annotations

import math
import numbers

import numpy as np
from zalmoxis.invariants import StructureInvariantError
from zalmoxis.structure_model import core_temperature

# Pairs per table: uniform in radius from the centre to the CMB of the profile
CORE_T_POINTS = 161
# Largest distance of a core node beyond the table, as a fraction of its pressure range
MAX_CORE_T_EXCESS = 0.1


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
        not finite, not strictly ascending in P, not positive in T, or fall in T with P (the
        Zalmoxis rule).
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
        and np.all(np.diff(t) >= 0)
    ):
        raise ValueError(
            'core temperature pairs are not finite, ascending in P and positive in T, '
            'or T falls with P'
        )
    return p, t


def check_core_arrays(core, table):
    """Raise unless the core arrays of a solve that used ``table`` lie on it.

    Parameters
    ----------
    core : dict
        Zalmoxis core arrays with ``r`` [m] ascending, ``P`` [Pa] and ``T`` [K].
    table : tuple of numpy.ndarray
        ``(P, T)`` core temperature table of the solve.

    Raises
    ------
    StructureInvariantError
        When ``T`` is missing, not finite, not positive, rises outward, or differs from the
        table temperature at ``P`` by more than a relative 1e-9.
    """
    t = np.asarray(core.get('T', []), dtype=float)
    want = core_temperature(np.asarray(core['P'], dtype=float), table)
    if not (
        t.shape == want.shape
        and np.all(np.isfinite(t))
        and np.all(t > 0)
        and np.all(np.diff(t) <= 0)
        and np.allclose(t, want, rtol=1e-9, atol=0.0)
    ):
        raise StructureInvariantError(
            'the core array temperature is not the core table temperature at the core array '
            'pressure (finite, positive, not rising outward, relative 1e-9)'
        )


def structure_core_table(config, hf_row, t_anchor, profiles=None):
    """Core table of a structure solve: the adiabat through ``t_anchor`` at the CMB.

    Parameters
    ----------
    config : Config
        Run configuration; ``interior_energetics.aragog.core_module`` gives alpha and c_p of the
        Gaussian fit.
    hf_row : dict
        Row with the last structure: ``M_core`` [kg], ``P_center``, ``P_cmb`` [Pa], ``R_core`` [m].
    t_anchor : float
        Temperature at the CMB [K]: the Aragog core temperature, else the row ``T_cmb``, else
        (a row without a positive ``T_cmb``) the deepest mantle temperature of the solve.
    profiles : aragog core profiles or None, optional
        Profiles of the live Aragog core (``core_bc = "core_module"``); None fits a Gaussian core
        to the row structure.

    Returns
    -------
    tuple of numpy.ndarray
        ``(P, T)`` of :func:`core_temperature_table`.
    """
    if profiles is None:
        from aragog.core.profiles import GaussianCoreProfiles

        c = config.interior_energetics.aragog.core_module
        profiles = GaussianCoreProfiles.from_structure(
            m_core=float(hf_row['M_core']),
            p_cen=float(hf_row['P_center']),
            r_cmb=float(hf_row['R_core']),
            p_cmb=float(hf_row['P_cmb']),
            alpha=float(c.alpha),
            c_p=float(c.c_p),
        )
    return core_temperature_table(profiles, t_anchor)
