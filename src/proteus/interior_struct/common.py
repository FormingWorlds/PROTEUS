"""Helpers shared by the interior structure modules and their consumers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config


def solvus_radius(
    config: Config,
    R_solvus: float | None,
    R_outer: float | None,
    R_inner: float | None = 0.0,
) -> float | None:
    """Return the solvus radius when the global-miscibility frame applies.

    With ``interior_struct.zalmoxis.global_miscibility`` the atmosphere lower
    boundary and the SPIDER domain move from the magma-ocean surface to the
    solvus. That frame applies only when a structure solve has written a
    physical solvus: finite and strictly between the inner (core) radius and
    the outer radius. The helpfile row initialises ``R_solvus`` to 0, and a
    zero must not reach a module as a boundary radius. Config validation
    currently rejects ``global_miscibility = true`` for every structure module
    (the Zalmoxis binodal handoff, tracker #64, is not implemented), so in a
    validated run this returns None.

    Parameters
    ----------
    config : Config
        PROTEUS configuration.
    R_solvus : float or None
        Solvus radius from the helpfile row [m].
    R_outer : float or None
        Outer radius the solvus must lie inside [m].
    R_inner : float or None
        Inner (core) radius the solvus must lie outside [m]; None or 0 means no
        inner bound beyond positivity.

    Returns
    -------
    float or None
        The solvus radius [m], or None when miscibility is off or no valid
        solvus is present.
    """
    if not config.interior_struct.zalmoxis.global_miscibility:
        return None
    if R_solvus is None or R_outer is None:
        return None
    R_sol, R_out = float(R_solvus), float(R_outer)
    R_in = float(R_inner) if R_inner is not None else 0.0
    if not (math.isfinite(R_sol) and math.isfinite(R_out) and math.isfinite(R_in)):
        return None
    if max(R_in, 0.0) < R_sol < R_out:
        return R_sol
    return None


def tracks_volatile_mass(config: Config) -> bool:
    """Whether ``M_volatile_change`` is recorded: only Zalmoxis solves a whole-planet target."""
    return config.interior_struct.module == 'zalmoxis'


def volatile_mass_change(hf_row: dict) -> float:
    """Return ``M_volatile_change`` [kg]; zero when absent (dummy structure, older helpfile).

    Raises
    ------
    RuntimeError
        If the column is not finite, which would corrupt the structure target.
    """
    change = float(hf_row.get('M_volatile_change') or 0.0)
    if not math.isfinite(change):
        raise RuntimeError(
            f'M_volatile_change is not finite ({change!r}); the ledger is corrupt.'
        )
    return change


def record_volatile_change(config: Config, hf_row: dict, delta: float) -> None:
    """Add ``delta`` [kg] to ``M_volatile_change`` when the Zalmoxis structure is in use.

    The Zalmoxis whole-planet target is ``mass_tot`` plus ``M_volatile_change``
    less the volatiles its mantle EOS does not hold, so a volatile change left
    out of the column would come back as rock at the next structure solve.

    Raises
    ------
    RuntimeError
        If the column or ``delta`` is not finite.
    """
    if not tracks_volatile_mass(config):
        return
    if not math.isfinite(delta):
        raise RuntimeError(f'volatile mass change is not finite ({delta!r})')
    hf_row['M_volatile_change'] = volatile_mass_change(hf_row) + delta


def debit_escaped_mass(config: Config, hf_row: dict, escaped: float) -> None:
    """Record the mass an escape step removed from the element budgets.

    Applied with the Zalmoxis structure, with or without accretion. The debit
    includes any element the step set to zero below the outgassing threshold,
    which ``esc_kg_cumulative`` does not count: that mass leaves the budgets, so
    the target drops with it, and oxygen the chemistry recomputes from the melt
    is mass the mantle supplied.

    Parameters
    ----------
    config : Config
        Model configuration; read for the structure module.
    hf_row : dict
        Current helpfile row; ``M_volatile_change`` is lowered in place.
    escaped : float
        Element mass the escape step removed from the budgets [kg].
    """
    if 0.0 < escaped < math.inf:
        record_volatile_change(config, hf_row, -escaped)
