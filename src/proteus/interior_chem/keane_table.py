"""Keane + Deng-thermal melt EOS of the FeO -> FeO1.5 volume change, for
``planet.interior_chem.redox_dV_eos = 'keane_table'`` (``eos_deng``).

Each 12.5 mol% Fe melt of Deng et al. (2020) (Mg14Fe2Si16O48 with Fe2+,
Mg14Fe2Si16O49 with Fe3+; 16 (Mg,Fe)SiO3 formula units and 2 Fe per cell) is

    P(V, T) = P_Keane(V) + (a - b x + c x^2) / 1000 (T - T_REF),  x = V / V0,

with the Keane (2000) cold curve on the T_REF = 3000 K isotherm (Attia et al.
2026 Eq 12) and the thermal term of Deng et al. (2020). The parameters in
``PARAMS`` are a least-squares fit to the 73 FPMD points of Deng et al. (2020)
Fig. 1 for the two melts; the fit residual is 0.08-0.09 GPa. ``build_table``
tabulates the oxidation integral int_{1 bar}^{P} dV dP' of FeO + 1/4 O2 =
FeO1.5 [J/mol] on a grid uniform in log10 P (1 bar - 4 TPa) and log10 T
(1000 - 20000 K), 100 points per decade, which ``eos_deng`` interpolates
bilinearly. The build takes a few seconds and runs once per process.

Liquid branch: V is the dense root of P(V, T) = P, below the volume of the
minimum of P(V) at that T; where P lies below that minimum the melt has no
liquid volume. Above T_b(1 bar), about 4206 K, the fit has no liquid at 1 bar,
so the integral evaluates dV at min(T, T_b(P') - 10 K) along the path, T_b(P)
being the hottest temperature at which both melts have a liquid at P.

Beyond the FPMD coverage (2000-4000 K, up to 17, 74 and 94 GPa on the 2000,
3000 and 4000 K isotherms) the table is an extrapolation of the cold-curve
form; ``in_fpmd`` marks the covered part of the grid.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.constants import N_A
from scipy.interpolate import RegularGridInterpolator

log = logging.getLogger('fwl.' + __name__)

# Fitted (V0 [A^3 per cell], K0 [GPa], K0', K'inf, a, b, c [MPa/K]) of the two melts.
PARAMS = {
    'Mg14Fe2Si16O48': (
        1178.7475575867927,
        28.806515185523534,
        1.97391309116932,
        4.861535624009775,
        36.92238150103547,
        74.05278307114249,
        38.51163050743906,
    ),
    'Mg14Fe2Si16O49': (
        1199.0827024253103,
        25.1406216008444,
        2.451282443240946,
        4.657618418770205,
        35.32674552882822,
        70.58694925576656,
        36.5628019923648,
    ),
}
RED, OX = 'Mg14Fe2Si16O48', 'Mg14Fe2Si16O49'
N_FE = 2  # Fe atoms per cell
T_REF = 3000.0  # K, isotherm of the cold curve
A3_TO_J_PER_GPA_MOL = 1.0e-30 * N_A * 1.0e9  # 1 A^3 per cell times 1 GPa, in J per mol of cells

P_MIN_GPA, P_MAX_GPA = 1.0e-4, 4000.0  # table range, 1 bar - 4 TPa
T_MIN, T_MAX = 1000.0, 20000.0  # K
N_PER_DECADE = 100  # grid points per decade in P and T
# Highest FPMD pressure [GPa] of the Fe3+ melt on each Deng et al. (2020) isotherm [K].
FPMD_P_MAX = {2000.0: 16.989, 3000.0: 73.616, 4000.0: 94.277}

_T_EDGE = np.geomspace(T_MIN, 30000.0, 600)  # K, isotherms of the liquid-branch edge
_N_EDGE = 3000  # volumes per isotherm in the search for the minimum of P(V)
_N_BISECT = 70  # bisection halvings of the volume solve
_N_PATH = 4000  # trapezoid nodes of each 1 bar -> 4 TPa isotherm
T_B_MARGIN = 10.0  # K, keeps the path inside the liquid region along its edge


def pressure(V, T, params) -> np.ndarray:
    """P [GPa] of one melt at volume V [A^3 per cell] and temperature T [K]."""
    V0, K0, Kp, Kinf, a, b, c = params
    Y, x = V0 / V, V / V0
    cold = K0 * (Kp / Kinf**2 * (Y**Kinf - 1.0) - (Kp / Kinf - 1.0) * np.log(Y))
    return cold + (a - b * x + c * x**2) / 1000.0 * (T - T_REF)


class _Melt:
    """One melt: the edge of its liquid branch, and the liquid volume V(P, T)."""

    def __init__(self, params):
        self.params = params
        V0 = params[0]
        Vg = np.geomspace(0.1 * V0, 2.0 * V0, _N_EDGE)
        vmax, pmin = [], []
        for T in _T_EDGE:
            rising = np.diff(pressure(Vg, T, params)) > 0
            k = np.argmax(rising) if np.any(rising) else Vg.size - 1
            vmax.append(Vg[k])
            pmin.append(pressure(Vg[k], T, params))
        self.vmax_T, self.pmin_T = np.array(vmax), np.array(pmin)

    def volume(self, P, T) -> np.ndarray:
        """Liquid volume [A^3 per cell] at P [GPa] and T [K]; NaN without a liquid."""
        P, T = np.broadcast_arrays(np.asarray(P, float), np.asarray(T, float))
        lo = np.full(P.shape, 0.1 * self.params[0])
        hi = np.interp(np.log(T), np.log(_T_EDGE), self.vmax_T)
        ok = P >= np.interp(np.log(T), np.log(_T_EDGE), self.pmin_T)
        for _ in range(_N_BISECT):
            mid = 0.5 * (lo + hi)
            above = pressure(mid, T, self.params) > P
            lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
        return np.where(ok, 0.5 * (lo + hi), np.nan)


def _hottest_liquid_T(melts):
    """T_b(P) [K]: hottest edge isotherm at which every melt has a liquid at P [GPa]."""
    p_need = np.maximum.reduce([np.maximum(m.pmin_T, P_MIN_GPA) for m in melts])
    p_need = np.maximum.accumulate(p_need)

    def t_b(P):
        k = np.searchsorted(p_need, np.maximum(P, P_MIN_GPA), side='right') - 1
        return _T_EDGE[np.clip(k, 0, _T_EDGE.size - 1)]

    return t_b


def _oxidation_integral(red, ox, t_b, T, P_out) -> np.ndarray:
    """int_{1 bar}^{P} dV dP' [J/mol] of FeO + 1/4 O2 = FeO1.5 on the isotherm T [K]
    at the pressures P_out [GPa], trapezoidal on a log-spaced path."""
    P_path = np.geomspace(P_MIN_GPA, P_MAX_GPA, _N_PATH)
    T_path = np.minimum(T, t_b(P_path) - T_B_MARGIN)
    dv = (ox.volume(P_path, T_path) - red.volume(P_path, T_path)) / N_FE * A3_TO_J_PER_GPA_MOL
    if np.any(np.isnan(dv)):
        raise RuntimeError(f'Keane melt EOS: undefined dV on the {T:.0f} K integration path')
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dv[1:] + dv[:-1]) * np.diff(P_path))])
    return np.interp(np.log(P_out), np.log(P_path), cum)


def _log_grid(lo, hi) -> np.ndarray:
    n = int(np.ceil(np.log10(hi / lo) * N_PER_DECADE)) + 1
    return np.linspace(np.log10(lo), np.log10(hi), n)


def build_table() -> dict:
    """Tabulate the oxidation integral and the FPMD coverage.

    Returns a dict with ``log10_T_K`` and ``log10_P_Pa`` (grid axes),
    ``int_dV_dP_ox`` [J/mol] and ``in_fpmd`` (bool), arrays indexed [T, P].
    """
    t0 = time.perf_counter()
    red, ox = _Melt(PARAMS[RED]), _Melt(PARAMS[OX])
    t_b = _hottest_liquid_T([red, ox])
    lp, lt = _log_grid(P_MIN_GPA * 1e9, P_MAX_GPA * 1e9), _log_grid(T_MIN, T_MAX)
    P_gpa, T = 10**lp / 1e9, 10**lt
    I = np.vstack([_oxidation_integral(red, ox, t_b, t, P_gpa) for t in T])
    t_near = np.array([min(FPMD_P_MAX, key=lambda k: abs(k - t)) for t in T])
    p_cap = np.array([FPMD_P_MAX[t] for t in t_near])
    in_fpmd = ((T >= 2000.0) & (T <= 4000.0))[:, None] & (P_gpa[None, :] <= p_cap[:, None])
    log.info(
        'Built the Keane + Deng-thermal melt EOS table in %.1f s', time.perf_counter() - t0
    )
    return {'log10_T_K': lt, 'log10_P_Pa': lp, 'int_dV_dP_ox': I, 'in_fpmd': in_fpmd}


def interpolators(table) -> dict:
    """Bilinear interpolator of a table in (log10 T, log10 P), its FPMD mask and its range."""
    lt, lp = table['log10_T_K'], table['log10_P_Pa']
    return {
        'interp': RegularGridInterpolator((lt, lp), table['int_dV_dP_ox']),
        'in_fpmd': RegularGridInterpolator(
            (lt, lp), np.asarray(table['in_fpmd']).astype(float), method='nearest'
        ),
        'T_range': (10 ** lt[0], 10 ** lt[-1]),
        'P_range': (10 ** lp[0] / 1e9, 10 ** lp[-1] / 1e9),
    }
