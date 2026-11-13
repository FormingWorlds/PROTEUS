"""Closed H-C-O gas at fixed element moles, total pressure P [bar] and temperature T [K].

Species H2O, H2, CO2, CO, CH4 and O2, with the equilibria and constants that CALLIOPE uses
(``calliope.chemistry.ModifiedKeq``: janaf_H2, janaf_CO, schaefer_CH4), and log10 fO2 [bar] as
the unknown:

    p_H2 = 10^(K_H - y/2) p_H2O,  p_CO = 10^(K_C - y/2) p_CO2,
    p_CH4 = 10^(K_M - y) p_CO2 p_H2^2,  p_O2 = 10^y.

The unknowns are ln p_H2O, ln p_CO2 and y; the equations are the total pressure and the H/C and
O/C ratios of the target.
"""

from __future__ import annotations

import numpy as np
from calliope.chemistry import ModifiedKeq
from scipy.optimize import least_squares

SP = ('H2O', 'H2', 'CO2', 'CO', 'CH4', 'O2')
EL = {
    'H2O': (2, 0, 1), 'H2': (2, 0, 0), 'CO2': (0, 1, 2),
    'CO': (0, 1, 1), 'CH4': (4, 1, 0), 'O2': (0, 0, 2),
}  # fmt: skip


def _logk(T):
    k = ModifiedKeq('janaf_H2')
    return k.janaf_H2(T)[0], k.janaf_CO(T)[0], k.schaefer_CH4(T)[0]


def pressures(u, T):
    """Partial pressures [bar] of the 6 species for the unknowns ``u`` at ``T``."""
    kh, kc, km = _logk(T)
    a, b, y = np.exp(u[0]), np.exp(u[1]), u[2]
    h2 = 10 ** (kh - 0.5 * y) * a
    return {
        'H2O': a, 'H2': h2, 'CO2': b, 'CO': 10 ** (kc - 0.5 * y) * b,
        'CH4': 10 ** (km - y) * b * h2**2, 'O2': 10**y,
    }  # fmt: skip


def elements(n):
    """Moles of H, C and O in the species moles ``n``."""
    return np.array([sum(n[s] * EL[s][j] for s in SP) for j in range(3)])


def solve(n_el, P, T, guess):
    """Species moles of the closed gas with element moles ``n_el`` (H, C, O) at ``P`` and ``T``,
    and its log10 fO2; ``guess`` is a dict of species moles."""
    tot = sum(guess.values())
    u0 = [
        np.log(guess['H2O'] / tot * P),
        np.log(guess['CO2'] / tot * P),
        np.log10(max(guess['O2'], 1e-300) / tot * P),
    ]

    def res(u):
        p = pressures(u, T)
        e = elements(p)
        return [
            np.log(sum(p.values()) / P),
            np.log(e[0] / e[1] / (n_el[0] / n_el[1])),
            np.log(e[2] / e[1] / (n_el[2] / n_el[1])),
        ]

    sol = least_squares(res, u0, xtol=1e-15, ftol=1e-15, gtol=1e-15, method='lm')
    assert sol.success and max(abs(np.array(sol.fun))) < 1e-12, sol
    p = pressures(sol.x, T)
    scale = n_el[1] / elements(p)[1]
    return {s: p[s] * scale for s in SP}, sol.x[2]
