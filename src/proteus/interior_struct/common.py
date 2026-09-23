"""Helpers shared by the interior structure modules and their consumers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config


def solvus_radius(config: Config, hf_row: dict, R_outer: float | None = None) -> float | None:
    """Return the solvus radius when the global-miscibility frame applies.

    With ``interior_struct.zalmoxis.global_miscibility`` the atmosphere lower
    boundary and the SPIDER domain move from the magma-ocean surface to the
    solvus. That frame applies only when a structure solve has written a
    physical solvus: ``hf_row['R_solvus']`` finite and strictly between 0 and
    the outer radius. The helpfile row initialises ``R_solvus`` to 0, and a
    zero must not reach a module as a boundary radius.

    Parameters
    ----------
    config : Config
        PROTEUS configuration.
    hf_row : dict
        Current helpfile row.
    R_outer : float or None
        Outer radius the solvus must lie inside [m]. Defaults to
        ``hf_row['R_int']``.

    Returns
    -------
    float or None
        The solvus radius [m], or None when miscibility is off or no valid
        solvus is present.
    """
    if not config.interior_struct.zalmoxis.global_miscibility:
        return None
    R_sol = hf_row.get('R_solvus')
    if R_outer is None:
        R_outer = hf_row.get('R_int')
    if R_sol is None or R_outer is None:
        return None
    R_sol, R_outer = float(R_sol), float(R_outer)
    if not (math.isfinite(R_sol) and math.isfinite(R_outer)):
        return None
    if 0.0 < R_sol < R_outer:
        return R_sol
    return None
