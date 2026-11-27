"""Warnings for a fixed fO2 that leaves its range of validity.

Both checks read the helpfile after a row is appended and change nothing.

- With ``planet.fO2_source = 'user_constant'``, CALLIOPE and atmodeller derive the oxygen
  inventory from the fO2, and the Zalmoxis structure target does not keep it
  (``_structure_target_masses``): oxygen outside the structure leaves the mantle mass of the
  next structure solve. A rise of ``O_kg_total`` since the first finite row above
  ``O_RISE_WARN_FRAC`` of the mantle mass of that row warns once per crossing, when the
  structure is solved again during the run (``zalmoxis.update_interval > 0``, or an impact).
  With a wet mantle only ``O_kg_atm`` is outside the structure, and the message reports its
  change. Outside the guard: the oxygen solved before the first row (the baseline structure
  solve of a start other than ``liquidus_super``, or ``zalmoxis.equilibrate_init = false``),
  and oxygen that escape removes and the fixed fO2 derives again, which leaves ``O_kg_total``
  unchanged while ``M_volatile_change`` takes it from the structure.
- The IW buffer is evaluated at ``max(T_magma, outgas.T_floor)``. Hirschmann (2021, GCA 313,
  74, Sect. 5) fits it over 1000-3000 K; CALLIOPE uses the fit at every temperature as its
  default buffer from 26.11.8 (pyproject.toml: fwl-calliope>=26.11.8), atmodeller switches to
  Hirschmann (2008) below 1000 K. Among the rows whose step ran an outgassing solve
  (``outgas_called``), the first row below and the first row above the range of the module
  each warn once per run.

Each decision reads the rows of the helpfile only, so a run resumed from a written row warns
on the rows after it as an uninterrupted run. A warning first given on a row that was not
written, or in a helpfile without ``outgas_called`` (read as no call), can repeat once.
"""

from __future__ import annotations

import logging
import math
from functools import cache
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


def _column(col: pd.Series | None, n: int) -> np.ndarray:
    """``hf_all.get(name)`` as ``n`` floats, NaN where absent, not numeric or not finite."""
    if col is None:
        return np.full(n, np.nan)
    v = pd.to_numeric(col, errors='coerce').to_numpy(float)
    return np.where(np.isfinite(v), v, np.nan)


def _check_oxygen(config: Config, hf_all: pd.DataFrame) -> None:
    """Warn when the derived O crosses ``O_RISE_WARN_FRAC`` of the baseline-row mantle mass."""
    from proteus.outgas.wrapper import outgassing_derives_o_kg_total

    struct = config.interior_struct
    if struct.module != 'zalmoxis' or not outgassing_derives_o_kg_total(config):
        return
    if not (struct.zalmoxis.update_interval > 0 or config.accretion.module):
        return
    o, mantle = (
        _column(hf_all.get('O_kg_total'), len(hf_all)),
        _column(hf_all.get('M_mantle'), len(hf_all)),
    )
    base = np.isfinite(o).argmax()
    with np.errstate(all='ignore'):
        rise = (o - o[base]) / (mantle[base] if mantle[base] > 0 else np.nan)
        moved = rise[-1] * mantle[base]
    before = rise[:-1][np.isfinite(rise[:-1])]
    if not rise[-1] > O_RISE_WARN_FRAC or (before.size and before[-1] > O_RISE_WARN_FRAC):
        return
    if struct.zalmoxis.dry_mantle:
        where = 'with interior_struct.zalmoxis.dry_mantle = true all of it is outside the structure mass'
    else:
        atm = _column(hf_all.get('O_kg_atm'), len(hf_all))
        where = (
            'with interior_struct.zalmoxis.dry_mantle = false only O_kg_atm is outside the '
            f'structure mass (its change since that row: {atm[-1] - atm[base]:.3e} kg)'
        )
    log.warning(
        'Oxygen derived by the fixed fO2 (outgas.fO2_shift_IW = %+.2f) rose by %.3e kg since '
        'the first finite row, %.2f %% of the mantle mass of that row (%.3e kg); %s, and the '
        'mantle loses that mass at the next structure solve.',
        config.outgas.fO2_shift_IW,
        moved,
        100.0 * rise[-1],
        mantle[base],
        where,
    )


@cache
def _calliope_default_buffer() -> str | None:
    """CALLIOPE's default IW buffer, or None (logged once) for a version that does not name it."""
    import calliope.oxygen_fugacity as fo2

    model = getattr(fo2, 'DEFAULT_FO2_MODEL', None)
    if model is None:
        log.warning(
            'calliope.oxygen_fugacity names no DEFAULT_FO2_MODEL; no IW buffer range check.'
        )
    return model


def _check_buffer(config: Config, hf_all: pd.DataFrame) -> None:
    """Warn on the first outgassed row below and the first above the buffer's temperature range."""
    module = config.outgas.module
    if module not in BUFFER_T_RANGE_K:
        return
    if module == 'calliope' and _calliope_default_buffer() != 'hirschmann':  # Fischer: no range
        return
    lo, hi = BUFFER_T_RANGE_K[module]
    T = np.maximum(_column(hf_all.get('T_magma'), len(hf_all)), config.outgas.T_floor)
    T = np.where(_column(hf_all.get('outgas_called'), len(hf_all)) > 0, T, np.nan)
    text = f'{lo:.0f}-{hi:.0f} K' if math.isfinite(lo) else f'(at most {hi:.0f} K)'
    if config.planet.fO2_source == 'from_O_budget':
        effect = 'the reported fO2_shift_IW_derived is relative to an extrapolated buffer'
    else:
        effect = 'its fO2 there is an extrapolation'
    for outside in (T < lo, T > hi):
        if outside[-1] and not outside[:-1].any():
            log.warning(
                'The IW buffer of %s is evaluated at %.1f K, outside the calibrated temperature '
                'range %s of Hirschmann (2021, GCA 313, 74); %s. This warning is given once per '
                'run.',
                module,
                T[-1],
                text,
                effect,
            )


def check_fo2_warnings(config: Config, hf_all: pd.DataFrame) -> None:
    """Log the oxygen and buffer warnings due at the last row of ``hf_all``."""
    if len(hf_all) == 0:
        return
    _check_oxygen(config, hf_all)
    _check_buffer(config, hf_all)
