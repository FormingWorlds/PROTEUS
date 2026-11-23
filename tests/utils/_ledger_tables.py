"""Synthetic helpfile tables for the energy-ledger tests (``tests/utils/test_energy_ledger.py`` and
``tests/tools/test_whole_run_ledger.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from proteus.utils.constants import secs_per_year

M_CORE, CP_CORE = 2.0e24, 800.0  # core mass [kg] and heat capacity [J kg-1 K-1]


def _table(core_bc='energy_balance', n_steps=7, seed=3, tfac=1.147):
    """Helpfile of 3 start rows at t = 0 and ``n_steps`` steps whose booked terms close exactly.

    Per step: mantle heat S = -L_int + C_cmb + Q; the loss to space F_atm(start) A(start) dt equals L_int;
    the core change is -C_cmb (energy_balance through T_cmb_node, core_module through step_dE_core_J).
    """
    rng = np.random.default_rng(seed)
    n = 3 + n_steps
    t = np.r_[0.0, 0.0, 0.0, np.cumsum(rng.uniform(5.0, 50.0, n_steps))]
    R_int = np.full(n, 6.4e6)
    A = 4.0 * np.pi * R_int**2
    L = np.r_[0.0, 0.0, 0.0, rng.uniform(1e27, 3e27, n_steps)]
    C = np.r_[0.0, 0.0, 0.0, rng.uniform(1e26, 4e26, n_steps)]
    Q = np.r_[0.0, 0.0, 0.0, rng.uniform(1e24, 5e24, n_steps)]
    F_atm = np.zeros(n)
    dt = np.diff(t) * secs_per_year
    F_atm[2:-1] = L[3:] / (A[2:-1] * dt[2:])
    d = {
        'Time': t,
        'R_int': R_int,
        'F_atm': F_atm,
        'step_dE_state_heat_J': -L + C + Q,
        'step_dE_F_int_J': -L,
        'step_dE_F_cmb_J': C,
        'step_dE_Q_radio_J': Q,
        'step_dE_Q_tidal_J': np.zeros(n),
        'step_dE_impact_J': np.zeros(n),
        'M_core': np.full(n, M_CORE),
        'R_core': np.full(n, 3.5e6),
        'M_mantle': np.full(n, 4.0e24),
        'core_heatcap': np.full(n, CP_CORE),
        'lid_weight': np.zeros(n),
        'step_dE_compression_J': np.zeros(n),
    }
    if core_bc == 'core_module':
        d |= {
            'step_dE_core_J': -C,
            'step_dE_impact_core_J': np.zeros(n),
            'step_dE_impact_core_refit_J': np.zeros(n),
            'step_dE_struct_core_J': np.zeros(n),
            'T_cmb_node': np.full(n, 4000.0),
        }
    else:
        d['T_cmb_node'] = 4000.0 - np.cumsum(C) / (M_CORE * CP_CORE * tfac)
    d['E_residual_cons_J'] = np.zeros(n)
    return pd.DataFrame(d)
