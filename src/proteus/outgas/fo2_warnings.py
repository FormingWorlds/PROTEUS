"""Warnings for a fixed fO2 that leaves its range of validity.

Both checks read the helpfile after a row is appended and change nothing.

- With ``planet.fO2_source = 'user_constant'``, CALLIOPE and atmodeller derive the oxygen
  inventory from the fO2, and the Zalmoxis structure target does not keep it
  (``_structure_target_masses``): oxygen outside the structure leaves the mantle mass of the
  next structure solve. A rise of it since the first row above ``O_RISE_WARN_FRAC`` of the
  mantle warns once per crossing.
- The IW buffer is evaluated at ``max(T_magma, outgas.T_floor)``. Hirschmann (2021, GCA 313,
  74, Sect. 5) fits it over 1000-3000 K; CALLIOPE uses that fit at every temperature,
  atmodeller switches to Hirschmann (2008) below 1000 K. A first row outside the range of the
  module warns once per run.

Each decision reads the rows of the helpfile only, so a resumed run warns as an uninterrupted
one.
"""

from __future__ import annotations

import logging
import math
import numbers
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


def _finite(value) -> float:
    """``value`` as a finite float; NaN when absent, boolean or not finite."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return math.nan
    return float(value) if math.isfinite(value) else math.nan


def _derived_o(config: Config, row) -> float:
    """Derived O [kg] that the structure target leaves out of the row."""
    dry = config.interior_struct.zalmoxis.dry_mantle
    return _finite(row.get('O_kg_total') if dry else row.get('O_kg_atm'))


def _oxygen_rise(config: Config, hf_all: pd.DataFrame, i: int) -> float:
    """Derived O outside the structure in row ``i`` minus that of row 0, per mantle mass."""
    row = hf_all.iloc[i]
    mantle = _finite(row.get('M_mantle'))
    if not mantle > 0.0:
        return math.nan
    return (_derived_o(config, row) - _derived_o(config, hf_all.iloc[0])) / mantle


def _outside_buffer_range(config: Config, hf_all: pd.DataFrame) -> np.ndarray:
    """Per row, whether the buffer temperature ``max(T_magma, T_floor)`` is outside the range."""
    lo, hi = BUFFER_T_RANGE_K[config.outgas.module]
    T = pd.to_numeric(hf_all['T_magma'], errors='coerce').to_numpy(dtype=float)
    T = np.maximum(T, config.outgas.T_floor)
    return (T < lo) | (T > hi)


def check_fo2_warnings(config: Config, hf_all: pd.DataFrame) -> None:
    """Log the oxygen and buffer warnings due at the last row of ``hf_all``."""
    from proteus.outgas.wrapper import outgassing_derives_o_kg_total

    if len(hf_all) == 0 or not outgassing_derives_o_kg_total(config):
        return
    last = len(hf_all) - 1

    if config.interior_struct.module == 'zalmoxis' and last > 0:
        rise = _oxygen_rise(config, hf_all, last)
        before = _oxygen_rise(config, hf_all, last - 1)
        if rise > O_RISE_WARN_FRAC and not before > O_RISE_WARN_FRAC:
            dry = config.interior_struct.zalmoxis.dry_mantle
            mantle = _finite(hf_all.iloc[last].get('M_mantle'))
            log.warning(
                'Oxygen derived by the fixed fO2 (outgas.fO2_shift_IW = %+.2f) rose by %.3e kg '
                'since the first row, %.2f %% of the mantle (%.3e kg); %s leaves the structure '
                'mass with interior_struct.zalmoxis.dry_mantle = %s, so the mantle loses it at '
                'the next structure solve.',
                config.outgas.fO2_shift_IW,
                rise * mantle,
                100.0 * rise,
                mantle,
                'O_kg_total' if dry else 'O_kg_atm',
                dry,
            )

    if 'T_magma' not in hf_all.columns:
        return
    outside = _outside_buffer_range(config, hf_all)
    if not outside[last] or outside[:last].any():
        return
    lo, hi = BUFFER_T_RANGE_K[config.outgas.module]
    T = max(float(pd.to_numeric(hf_all['T_magma'].iloc[last])), config.outgas.T_floor)
    span = f'{lo:.0f}-{hi:.0f} K' if math.isfinite(lo) else f'(at most {hi:.0f} K)'
    log.warning(
        'The IW buffer of %s is evaluated at %.1f K, outside the calibrated range %s of '
        'Hirschmann (2021, GCA 313, 74); its fO2 there is an extrapolation. This warning is '
        'given once per run.',
        config.outgas.module,
        T,
        span,
    )
