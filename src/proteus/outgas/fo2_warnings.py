"""Warnings for a fixed fO2 that leaves its range of validity.

Both checks read the helpfile after a row is appended and change nothing.

- With ``planet.fO2_source = 'user_constant'``, CALLIOPE and atmodeller derive the oxygen
  inventory from the fO2, and the Zalmoxis structure target does not keep it
  (``_structure_target_masses``): oxygen outside the structure leaves the mantle mass of the
  next structure solve. A rise of it since the first finite row above ``O_RISE_WARN_FRAC`` of
  the mantle warns once per crossing, when the structure is solved again during the run.
- The IW buffer is evaluated at ``max(T_magma, outgas.T_floor)``. Hirschmann (2021, GCA 313,
  74, Sect. 5) fits it over 1000-3000 K; CALLIOPE uses that fit at every temperature,
  atmodeller switches to Hirschmann (2008) below 1000 K. On a step that ran the outgassing,
  the first row below and the first row above the range of the module each warn once per run.

Each decision reads the rows of the helpfile only, so a resumed run warns as an uninterrupted
one.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Fraction of the mantle mass; the oxygen rise of an 8.6 M_E IW+2 case stays within 6e-5.
O_RISE_WARN_FRAC = 1e-3
# Temperature range [K] of the IW buffer of each outgassing module.
BUFFER_T_RANGE_K = {'calliope': (1000.0, 3000.0), 'atmodeller': (-math.inf, 3000.0)}


def _column(hf_all: pd.DataFrame, name: str) -> np.ndarray:
    """Column ``name`` as floats, NaN where absent, not numeric or not finite."""
    if name not in hf_all.columns:
        return np.full(len(hf_all), np.nan)
    v = pd.to_numeric(hf_all[name], errors='coerce').to_numpy(dtype=float)
    return np.where(np.isfinite(v), v, np.nan)


def _check_oxygen(config: Config, hf_all: pd.DataFrame) -> None:
    """Warn when the derived O outside the structure crosses ``O_RISE_WARN_FRAC`` of the mantle."""
    from proteus.outgas.wrapper import outgassing_derives_o_kg_total

    struct = config.interior_struct
    if (
        struct.module != 'zalmoxis'
        or not struct.zalmoxis.update_interval > 0
        or not outgassing_derives_o_kg_total(config)
    ):
        return
    col = 'O_kg_total' if struct.zalmoxis.dry_mantle else 'O_kg_atm'
    o, mantle = _column(hf_all, col), _column(hf_all, 'M_mantle')
    finite_o = o[np.isfinite(o)]
    if finite_o.size == 0:
        return
    with np.errstate(all='ignore'):
        rise = (o - finite_o[0]) / np.where(mantle > 0, mantle, np.nan)
    before = rise[:-1][np.isfinite(rise[:-1])]
    if not rise[-1] > O_RISE_WARN_FRAC or (before.size and before[-1] > O_RISE_WARN_FRAC):
        return
    log.warning(
        'Oxygen derived by the fixed fO2 (outgas.fO2_shift_IW = %+.2f) rose by %.3e kg since '
        'the first row, %.2f %% of the mantle (%.3e kg); %s leaves the structure mass with '
        'interior_struct.zalmoxis.dry_mantle = %s, so the mantle loses it at the next '
        'structure solve.',
        config.outgas.fO2_shift_IW,
        rise[-1] * mantle[-1],
        100.0 * rise[-1],
        mantle[-1],
        col,
        struct.zalmoxis.dry_mantle,
    )


def _check_buffer(config: Config, hf_all: pd.DataFrame) -> None:
    """Warn on the first row below and the first row above the buffer's temperature range."""
    span = BUFFER_T_RANGE_K.get(config.outgas.module)
    if span is None:
        return
    lo, hi = span
    T = np.maximum(_column(hf_all, 'T_magma'), config.outgas.T_floor)
    text = f'{lo:.0f}-{hi:.0f} K' if math.isfinite(lo) else f'(at most {hi:.0f} K)'
    for outside in (T < lo, T > hi):
        if outside[-1] and not outside[:-1].any():
            log.warning(
                'The IW buffer of %s is evaluated at %.1f K, outside the calibrated temperature '
                'range %s of Hirschmann (2021, GCA 313, 74); its fO2 there is an extrapolation. '
                'This warning is given once per run.',
                config.outgas.module,
                T[-1],
                text,
            )


def check_fo2_warnings(config: Config, hf_all: pd.DataFrame, outgassed: bool = True) -> None:
    """Log the oxygen and buffer warnings due at the last row of ``hf_all``.

    ``outgassed`` is False on a step whose volatile exchange ran no outgassing call
    (desiccated, crystallized, or under a held lid seal): no buffer was evaluated there.
    """
    if len(hf_all) == 0:
        return
    _check_oxygen(config, hf_all)
    if outgassed:
        _check_buffer(config, hf_all)
