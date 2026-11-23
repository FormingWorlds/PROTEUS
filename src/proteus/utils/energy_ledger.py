"""Whole-run energy ledger of a stored helpfile in the heat frame (rho T dS).

Per step k (row k-1 to row k) the ledger sums the booked energy columns [J]:

- mantle and lid: ``R_m = S - (-L_int + C_cmb + Q + I_m) + I_m``, with the mantle heat change
  ``S = step_dE_state_heat_J``, the heat to the atmosphere ``L_int = -step_dE_F_int_J``, the CMB heat
  ``C_cmb = step_dE_F_cmb_J``, the radiogenic and tidal heat ``Q`` and the impact heat ``I_m``;
- core: ``R_c = D_core + C_cmb - B_core``. With ``core_bc = 'core_module'`` the core heat change is
  ``step_dE_core_J`` and the booked terms ``B_core`` are the impact, refit and structure-update core heat;
  with ``energy_balance`` and ``bower2018`` it is ``M_core core_heatcap core_tfac dT_cmb_node``
  (Bower et al. 2018, Eqs. 19-21) and nothing is booked, so a core heat jump at a structure update stays
  in ``R_c`` (the remap class);
- coupling: ``G = L_int - E_atm``, with the loss to space ``E_atm = F_atm A dt`` on the start row of the
  step (a step runs on the structure of its start row);
- planet: ``R_p = R_m + R_c - G``.

The mantle mass change of each step (a structure update can change the mantle mass) is reported in kg;
its energy depends on the zero of the EOS energy and is not booked. ``step_dE_compression_J`` is
reported as a series and not summed into the heat ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from proteus.utils.constants import secs_per_year

CORE_BCS = ('energy_balance', 'bower2018', 'core_module')
G_TOL = 1e-12  # coupling identity on a no-lid step, relative to |L_int| of the step
R_M_TOL = 1e-9  # R_m against the stored E_residual_cons_J difference, relative

_COMMON = [
    'Time',
    'R_int',
    'R_core',
    'M_core',
    'M_mantle',
    'F_atm',
    'step_dE_state_heat_J',
    'step_dE_F_int_J',
    'step_dE_F_cmb_J',
    'step_dE_Q_radio_J',
    'step_dE_Q_tidal_J',
    'step_dE_impact_J',
]
_CORE = {
    'core_module': [
        'step_dE_core_J',
        'step_dE_impact_core_J',
        'step_dE_impact_core_refit_J',
        'step_dE_struct_core_J',
    ],
    'energy_balance': ['core_heatcap', 'T_cmb_node'],
    'bower2018': ['core_heatcap', 'T_cmb_node'],
}
_OPTIONAL = [
    'lid_weight',
    'step_dE_compression_J',
    'E_residual_cons_J',
    'E_state_heat_cons_J',
    'dE_predicted_cons_J',
]


@dataclass(frozen=True)
class Ledger:
    """Energy ledger of the steps of a window; every array has one value per step [J unless noted].

    Attributes
    ----------
    time : np.ndarray
        Time at the end of each step [yr].
    terms : dict[str, np.ndarray]
        Booked terms: S_state, L_int, C_cmb, Q, I_mantle, D_core, B_core, E_atm, compression.
    R_m, R_c, G, R_p : np.ndarray
        Mantle, core, coupling and planet residuals.
    classes : dict[str, np.ndarray]
        Boolean step classes: lid, update, remap, zero, impact, negative_F_atm.
    dM_mantle : np.ndarray
        Mantle mass change of the step [kg].
    dM_mantle_frac : np.ndarray
        ``dM_mantle`` over the mantle mass at the start row [1].
    throughput : float
        Gross energy throughput T, the sum of the absolute booked terms [J].
    """

    time: np.ndarray
    terms: dict
    R_m: np.ndarray
    R_c: np.ndarray
    G: np.ndarray
    R_p: np.ndarray
    classes: dict
    dM_mantle: np.ndarray
    dM_mantle_frac: np.ndarray
    throughput: float

    @property
    def closure(self) -> float:
        """Planet closure ``sum R_p / T`` [1]."""
        return float(self.R_p.sum() / self.throughput) if self.throughput > 0 else 0.0

    def split(self, name: str = 'R_c') -> dict[str, float]:
        """Sum of a residual by step class: steps ending at an update, steps after one, other [J]."""
        x = getattr(self, name)
        up, rm = self.classes['update'], self.classes['remap']
        return {
            'update': float(x[up & ~rm].sum()),
            'remap': float(x[rm].sum()),
            'other': float(x[~up & ~rm].sum()),
        }


def _column(hf: pd.DataFrame, name: str) -> np.ndarray:
    """A numeric helpfile column as float, refusing a column of another type."""
    if name not in hf.columns:
        raise ValueError(f'the helpfile has no column {name!r}')
    col = hf[name]
    if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        raise ValueError(f'column {name!r} is not numeric (dtype {col.dtype})')
    return col.to_numpy(dtype=float)


def _window(n: int, start_row: int, end_row: int | None) -> tuple[int, int]:
    """Checked start and end rows of the window (end inclusive)."""
    end = n - 1 if end_row is None else end_row
    if not 0 <= start_row < n - 1:
        raise ValueError(f'start_row {start_row} must lie in [0, {n - 2}]')
    if not start_row < end <= n - 1:
        raise ValueError(f'end_row {end} must lie in [{start_row + 1}, {n - 1}]')
    return start_row, end


def _read(hf: pd.DataFrame, core_bc: str, rows: slice) -> dict[str, np.ndarray]:
    """The columns the ledger reads, checked to be finite on the rows of the window."""
    names = _COMMON + _CORE[core_bc]
    cols = {name: _column(hf, name)[rows] for name in names}
    for name in _OPTIONAL:
        cols[name] = _column(hf, name)[rows] if name in hf.columns else None
    for name, x in cols.items():
        if x is not None and not np.all(np.isfinite(x)):
            bad = int(np.flatnonzero(~np.isfinite(x))[0]) + rows.start
            raise ValueError(f'column {name!r} is not finite at row {bad}')
    return cols


def _terms(c: dict, core_bc: str, core_tfac: float, dt: np.ndarray) -> dict[str, np.ndarray]:
    """Booked terms per step from the columns of rows start..end (step k ends at row k + 1)."""
    e = slice(1, None)
    A = 4.0 * np.pi * c['R_int'] ** 2
    if core_bc == 'core_module':
        D = c['step_dE_core_J'][e]
        B = sum(c[k][e] for k in _CORE['core_module'][1:])
    else:
        heat_cap = c['M_core'] * c['core_heatcap'] * core_tfac
        D = heat_cap[e] * np.diff(c['T_cmb_node'])
        B = np.zeros_like(D)
    comp = c['step_dE_compression_J']
    return {
        'S_state': c['step_dE_state_heat_J'][e],
        'L_int': -c['step_dE_F_int_J'][e],
        'C_cmb': c['step_dE_F_cmb_J'][e],
        'Q': c['step_dE_Q_radio_J'][e] + c['step_dE_Q_tidal_J'][e],
        'I_mantle': c['step_dE_impact_J'][e],
        'D_core': D,
        'B_core': B,
        'E_atm': c['F_atm'][:-1] * A[:-1] * dt,
        'compression': comp[e] if comp is not None else np.full(dt.size, np.nan),
    }


def _classes(c: dict, t: dict, dt: np.ndarray) -> dict[str, np.ndarray]:
    """Step classes: a live lid at either end, the step ending on a new structure, the step after it,
    a step with dt > 0 and no energy, an impact step and a step whose start row has F_atm < 0."""
    new = np.zeros(c['R_int'].size, bool)
    for k in ('R_core', 'M_core', 'R_int'):
        new[1:] |= c[k][1:] != c[k][:-1]
    w = c['lid_weight']
    lid = (w[:-1] > 0) | (w[1:] > 0) if w is not None else np.zeros(dt.size, bool)
    return {
        'lid': lid,
        'update': new[1:],
        'remap': new[:-1],
        'zero': (dt > 0) & (t['S_state'] == 0) & (t['L_int'] == 0),
        'impact': t['I_mantle'] != 0,
        'negative_F_atm': c['F_atm'][:-1] < 0,
    }


def _check(t: dict, cls: dict, R_m: np.ndarray, G: np.ndarray, c: dict) -> None:
    """The coupling identity on every no-lid step with energy, and R_m against the stored column."""
    plain = ~cls['lid'] & ~cls['zero']
    over = plain & ~(np.abs(G) <= G_TOL * np.abs(t['L_int']))
    if over.any():
        k = int(np.flatnonzero(over)[0])
        raise RuntimeError(
            f'coupling identity G = L_int - E_atm fails on the no-lid step {k}: '
            f'G = {G[k]:.6e} J, L_int = {t["L_int"][k]:.6e} J'
        )
    er = c['E_residual_cons_J']
    if er is not None:
        stored, summed = er[-1] - er[0], float(R_m.sum())
        gross = sum(float(np.abs(t[k]).sum()) for k in ('S_state', 'L_int', 'C_cmb', 'Q'))
        eps = np.finfo(float).eps
        floor = 64 * eps * gross  # round-off of the per-step sums
        for name in (
            'E_state_heat_cons_J',
            'dE_predicted_cons_J',
        ):  # E_residual is their difference
            if c[name] is not None:
                floor += R_m.size * eps * float(np.abs(c[name]).max())
        if not abs(stored - summed) <= R_M_TOL * max(abs(stored), abs(summed)) + floor:
            raise RuntimeError(
                f'R_m {summed:.6e} J differs from the stored E_residual_cons_J change {stored:.6e} J'
            )


def whole_run_ledger(
    hf: pd.DataFrame,
    core_bc: str,
    core_tfac: float = 1.147,
    start_row: int = 2,
    end_row: int | None = None,
) -> Ledger:
    """Energy ledger of the steps from ``start_row`` to ``end_row`` of a helpfile table.

    Parameters
    ----------
    hf : pd.DataFrame
        Helpfile table, as ``proteus.utils.coupler.read_helpfile_table`` returns it.
    core_bc : str
        Aragog core boundary condition of the run: ``energy_balance``, ``bower2018`` or ``core_module``.
    core_tfac : float
        Core temperature factor ``interior_energetics.core_tfac_avg`` [1].
    start_row, end_row : int
        Rows that bound the window; the steps end at rows ``start_row + 1`` to ``end_row``. The default
        start row 2 is the last row at t = 0 of a fresh run.

    Returns
    -------
    Ledger
        Per-step terms, residuals, classes and mantle mass changes of the window.

    Raises
    ------
    ValueError
        On an unknown ``core_bc``, a window outside the table, a missing or non-numeric column, a
        non-finite value, or a time that does not increase across a step with energy.
    RuntimeError
        When the coupling identity fails on a no-lid step, or ``R_m`` differs from the stored
        ``E_residual_cons_J`` change.
    """
    if core_bc not in CORE_BCS:
        raise ValueError(f'core_bc {core_bc!r} has no core ledger here; use one of {CORE_BCS}')
    start, end = _window(len(hf), start_row, end_row)
    c = _read(hf, core_bc, slice(start, end + 1))
    dt = np.diff(c['Time']) * secs_per_year
    t = _terms(c, core_bc, core_tfac, dt)
    energy = np.abs(t['S_state']) + np.abs(t['L_int']) + np.abs(t['C_cmb']) > 0
    if np.any(dt < 0) or np.any((dt == 0) & energy):
        k = int(np.flatnonzero((dt < 0) | ((dt == 0) & energy))[0]) + start + 1
        raise ValueError(f'Time does not increase into row {k} across a step with energy')
    cls = _classes(c, t, dt)
    R_m = t['S_state'] - (-t['L_int'] + t['C_cmb'] + t['Q'] + t['I_mantle']) + t['I_mantle']
    R_c = t['D_core'] + t['C_cmb'] - t['B_core']
    G = t['L_int'] - t['E_atm']
    _check(t, cls, R_m, G, c)
    dM = np.diff(c['M_mantle'])
    booked = ('S_state', 'L_int', 'C_cmb', 'Q', 'I_mantle', 'B_core')
    return Ledger(
        time=c['Time'][1:],
        terms=t,
        R_m=R_m,
        R_c=R_c,
        G=G,
        R_p=R_m + R_c - G,
        classes=cls,
        dM_mantle=dM,
        dM_mantle_frac=dM / c['M_mantle'][:-1],
        throughput=float(sum(np.abs(t[k]).sum() for k in booked)),
    )
