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
