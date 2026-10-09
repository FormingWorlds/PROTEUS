"""Where trapping buried each volatile, in melt-fraction coordinates.

A crystallising step buries mass into the solid that froze while the global
melt fraction fell from ``Phi(t-1)`` to ``Phi(t)``. The ledger records that
mass against that interval of melt fraction, one layer per step. The solid that
remelts first is the solid that froze last, so a remelt to ``Phi`` returns the
mass recorded below ``Phi``, and freezing and remelting the same interval
cancel exactly.

The ledger is rebuilt from the helpfile history, ``Phi_global`` and the
``_kg_trapped`` columns, so it keeps no state of its own and survives a resume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Layer:
    """Mass buried while the melt fraction fell from ``hi`` to ``lo``."""

    lo: float
    hi: float
    mass: np.ndarray  # per trapped column [kg]


def remelt_to(layers: list[Layer], phi: float) -> np.ndarray:
    """Remove and return the mass buried below melt fraction ``phi`` [kg].

    ``layers`` is ordered as built, the most recent last, which is also the
    lowest in melt fraction. It is changed in place.
    """
    released = np.zeros_like(layers[0].mass) if layers else np.zeros(0)
    while layers and layers[-1].lo < phi:
        top = layers[-1]
        if top.hi <= phi:
            released = released + top.mass
            layers.pop()
            continue
        share = (phi - top.lo) / (top.hi - top.lo)
        released = released + share * top.mass
        top.mass = (1.0 - share) * top.mass
        top.lo = phi
        break
    return released


def build_ledger(phi: np.ndarray, trapped: np.ndarray) -> list[Layer]:
    """Replay a history of melt fraction and trapped mass into burial layers.

    Parameters
    ----------
    phi : ndarray
        Global melt fraction of each completed step, shape (N,) [1].
    trapped : ndarray
        Trapped mass of each tracked column at each step, shape (N, K) [kg].

    Returns
    -------
    list of Layer
        The layers still buried at the end of the history, the most recent last.
    """
    phi = np.asarray(phi, dtype=float)
    trapped = np.nan_to_num(np.asarray(trapped, dtype=float))
    layers: list[Layer] = []
    for i in range(1, phi.size):
        before, after = phi[i - 1], phi[i]
        if not (np.isfinite(before) and np.isfinite(after)) or after == before:
            continue
        if after < before:
            buried = np.maximum(trapped[i] - trapped[i - 1], 0.0)
            layers.append(Layer(lo=float(after), hi=float(before), mass=buried))
        else:
            remelt_to(layers, float(after))
    return layers


def ledger_total(layers: list[Layer], n_columns: int) -> np.ndarray:
    """Mass the ledger holds per column [kg]."""
    total = np.zeros(n_columns)
    for layer in layers:
        total = total + layer.mass
    return total
