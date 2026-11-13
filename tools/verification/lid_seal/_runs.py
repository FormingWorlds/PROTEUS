"""Shared readers of the lid seal tutorial runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from proteus.utils.coupler import read_helpfile_table


def read_run(run: str | Path) -> pd.DataFrame:
    """The helpfile of a PROTEUS output folder, every float as written."""
    return read_helpfile_table(Path(run) / 'runtime_helpfile.csv')


def closing_row(seal: pd.DataFrame) -> int:
    """Index of the first row with an active seal."""
    active = np.flatnonzero(seal['seal_active'].values > 0.5)
    if active.size == 0:
        raise ValueError('the seal never closes in this run')
    return int(active[0])
