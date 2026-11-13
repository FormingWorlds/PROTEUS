"""The atmosphere column that the escape step reads.

The ZEPHYRUS dispatcher locates its wind base, exobase and photosphere on the
atmosphere profile, so the climate step keeps a copy of its last converged
column for the escape step of the next iteration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from proteus.atmos_clim.common import Atmos_t
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Mixing ratios at or below this are dropped from the stored column, as no
# species this rare changes the escaping composition.
VMR_DROP = 1.0e-30


def column_from_arrays(p, r, T, vmr: dict) -> dict:
    """Order a column base to top and keep the levels that rise strictly.

    Parameters
    ----------
    p, r, T : array_like
        Pressure [Pa], radius from the planet centre [m] and temperature [K]
        per level, in either order.
    vmr : dict
        Gas name to its volume mixing ratio per level.

    Returns
    -------
    column : dict
        Keys ``p``, ``r``, ``T`` and ``vmr``, ordered from the deepest level
        up, with pressure strictly decreasing and radius strictly increasing.

    Raises
    ------
    ValueError
        If the arrays differ in length or fewer than three usable levels
        remain.
    """
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    T = np.asarray(T, dtype=float)
    gases = {g: np.asarray(x, dtype=float) for g, x in vmr.items()}
    n = len(p)
    if len(r) != n or len(T) != n or any(len(x) != n for x in gases.values()):
        raise ValueError('atmosphere column arrays differ in length')

    order = np.argsort(-p, kind='stable')
    p, r, T = p[order], r[order], T[order]
    gases = {g: x[order] for g, x in gases.items()}

    # Keep each level that lies strictly above the last one kept.
    keep = np.zeros(n, dtype=bool)
    last_p, last_r = np.inf, -np.inf
    for i in range(n):
        ok = np.isfinite(p[i]) and np.isfinite(r[i]) and np.isfinite(T[i])
        if ok and 0.0 < p[i] < last_p and r[i] > last_r and T[i] > 0.0:
            keep[i] = True
            last_p, last_r = p[i], r[i]
    if keep.sum() < 3:
        raise ValueError('atmosphere column has fewer than three usable levels')

    out_vmr = {}
    for g, x in gases.items():
        x = np.clip(np.nan_to_num(x[keep], nan=0.0), 0.0, None)
        if np.any(x > VMR_DROP):
            out_vmr[g] = x
    if not out_vmr:
        raise ValueError('atmosphere column holds no gas')
    return {'p': p[keep], 'r': r[keep], 'T': T[keep], 'vmr': out_vmr}


def _agni_column(atm) -> dict:
    """Column of an AGNI atmosphere, from its cell-centre arrays."""
    vmr = {str(g): np.array(atm.gas_vmr[g], dtype=float) for g in list(atm.gas_names)}
    return column_from_arrays(
        np.array(atm.p, dtype=float),
        np.array(atm.r, dtype=float),
        np.array(atm.tmp, dtype=float),
        vmr,
    )


def _janus_column(atm, r_int: float) -> dict:
    """Column of the JANUS atmosphere last solved, whose heights start at the surface."""
    vmr = {str(g): np.array(x, dtype=float) for g, x in atm.x_gas.items()}
    return column_from_arrays(
        np.array(atm.p, dtype=float),
        np.array(atm.z, dtype=float) + float(r_int),
        np.array(atm.tmp, dtype=float),
        vmr,
    )


def store_escape_profile(atmos_o: Atmos_t, config: Config, hf_row: dict) -> None:
    """Keep the column of a converged solve for the next escape step.

    A rejected solve leaves the previous column in place and marks it as not
    current, so escape never reads a structure the solver rejected. Nothing is
    stored when escape does not use the ZEPHYRUS dispatcher, for the dummy
    atmosphere, or for an AGNI column in transparent mode (surface pressure
    below ``atmos_clim.agni.psurf_thresh``), which AGNI collapses onto its top
    level; escape then builds its own isothermal stand-in.

    Parameters
    ----------
    atmos_o : Atmos_t
        Atmosphere state, updated in place.
    config : Config
        Run configuration; selects the climate backend.
    hf_row : dict
        Helpfile row of this iteration, for the interior radius under JANUS.
    """
    module = config.atmos_clim.module
    dispatcher = bool(getattr(getattr(config, 'escape', None), 'uses_dispatcher', False))
    transparent = module == 'agni' and bool(getattr(atmos_o._atm, 'transparent', False))
    if module not in ('agni', 'janus') or not dispatcher or transparent:
        # A transparent AGNI column is collapsed onto its top level, so escape
        # builds its stand-in from the surface pressure instead.
        atmos_o.escape_profile = None
        atmos_o.escape_profile_current = False
        return
    if not atmos_o.converged:
        atmos_o.escape_profile_current = False
        return
    try:
        if module == 'agni':
            column = _agni_column(atmos_o._atm)
        else:
            column = _janus_column(atmos_o._atm_janus_last, hf_row['R_int'])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        log.warning('Atmosphere column not stored for escape: %s', exc)
        atmos_o.escape_profile_current = False
        return
    atmos_o.escape_profile = column
    atmos_o.escape_profile_current = True
