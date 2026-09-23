"""Pressure term for the Fe-disproportionation reaction.

Provides ``int_dV_dP(T, P)``, the integral of the reaction volume change

    3 FeO(silicate liq) = 2 FeO1.5(silicate liq) + Fe(liq metal)

from 1 bar to P along an isotherm at T, in J/mol. This is the term Schaefer
et al. (2024) always include in their disproportionation calculation
(``disproportionation.m`` passes the full basal pressure straight through to
``deltaGFeOFeO15_Deng.m``), in contrast to their surface fO2 routine
``fO2lowP_H22.m``, which correctly sets it to zero because that is evaluated
at 1 bar.

The integral assembles three volume terms, following Schaefer's construction:

    int dV dP = 2*int(dV_Deng) dP  -  int(V_FeO) dP  +  int(V_Fe) dP

where dV_Deng = V(FeO1.5) - V(FeO) comes from differencing two Deng et al.
(2020) melt endmembers (Mg14Fe2Si16O48 and Mg14Fe2Si16O49, which differ by
exactly one oxygen; the Mg-silicate part cancels), V_FeO is the Lange &
Carmichael (1987) liquid volume with a Murnaghan compression, and V_Fe is
liquid iron on the Komabayashi (2014) Vinet EOS.

Because the integral depends only on (P, T) -- never on composition, the
initial ferric fraction, or the timestep -- it is tabulated once per process
and interpolated thereafter. A 150 x 40 grid reproduces a direct 400-point
integration to ~0.1% median / 1.2% max; a 60 x 40 grid is NOT sufficient
(10% worst case).

VALIDITY
--------
Two independent limits, both enforced here and reported through the ``valid``
flag that every public call returns:

1. Deng's thermal-pressure term ``BH(x) = (a - b*x + c*x**2)/1000``, with
   x = V/V0, is a parabola whose minimum sits at x = b/(2c) ~ 0.973. Below
   that it decreases with expanding volume, as a thermal pressure must;
   above it the fit turns upward and diverges, which is unphysical. It was
   evidently fitted to compressed states only. We therefore cap the liquid
   at V <= V0 and splice a constant dV below the corresponding pressure

       P_splice(T) = P0 + BH(1)*(T - T0)

   using Deng's own uncompressed volume difference there. NOTE: this splice
   is our construction, not from the literature. Deng et al. (2020) Nat.
   Commun. 11, 2007 should be consulted for the V/V0 range they actually
   sampled, and the splice point set from that.

2. Above T ~ 4175 K the BM4 equation has no root at all at 1 bar (the rising
   thermal term overtakes the falling Birch-Murnaghan term), and Deng's fit
   was calibrated around T0 = 3000 K with a thermal pressure linear in
   (T - T0). We mark T above ``T_CEILING`` invalid rather than extrapolate.

Pressures above ``P_EXERCISED`` (136 GPa, roughly Earth's core-mantle
boundary and the deepest Schaefer et al. ran) are flagged but not refused.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import brentq

log = logging.getLogger('fwl.' + __name__)

# ── Deng et al. (2020) 4th-order Birch-Murnaghan + thermal pressure ─────────
# Row 0 = reduced   (Mg14Fe2Si16O48, ferrous)
# Row 1 = oxidised  (Mg14Fe2Si16O49, ferric)
# V0 converted from cubic Angstroms/mol to J/GPa/mol: 1 A^3/mol = 602.21 J/GPa
_V0 = 602.21 * np.array([1180.11401427979, 1204.76365212898])
_K0 = np.array([26.9471386120485, 23.1953006249046])
_KP = np.array([2.80253187140108, 3.21608935806420])
_KDP = np.array([0.0123134723892271, 0.00934018313745533])
_A = np.array([35.7939748339471, 34.5261639420686])
_B = np.array([71.1031366774265, 68.6442962291442])
_C = np.array([36.5954522514324, 35.2706911576929])

_P0 = 1.0e-4        # GPa, Deng reference pressure
_T0 = 3000.0        # K,   Deng reference temperature

# Uncompressed reaction volume, 0.5*(V0_ox - V0_red), in J/GPa/mol.
# The factor 0.5 is Schaefer's: the endmembers carry 2 Fe per formula unit,
# so the per-FeO volume difference is half the endmember difference.
_DV0 = 0.5 * (_V0[1] - _V0[0])

T_CEILING = 4175.0      # K,   above this the BM4 has no root at 1 bar
P_EXERCISED = 136.0     # GPa, deepest pressure Schaefer et al. (2024) ran

# ── Lange & Carmichael (1987) FeO liquid + Kress & Carmichael (1993) K ──────
_K0_FEO, _KP_FEO = 30.33, 4.0

# ── Komabayashi (2014) liquid Fe, Vinet ─────────────────────────────────────
_V298_FE, _ALPHA0_FE = 6.88, 9.0e-5
_K0_FE, _KP_FE, _DELTA0_FE, _KAPPA_FE = 148.0, 5.8, 5.1, 0.56


def _bh(x: np.ndarray, i: int) -> np.ndarray:
    """Deng thermal-pressure coefficient, GPa/K. x = V/V0."""
    return (_A[i] - _B[i] * x + _C[i] * x ** 2) / 1000.0


def p_splice(T: float) -> float:
    """Pressure at V = V0, below which the fit would need V > V0.

    Taken as the larger of the two endmembers: they have slightly different
    thermal coefficients, so the ferrous one loses its root first, and a
    splice set from the ferric endmember alone leaves a thin band near the
    temperature ceiling where the ferrous solve fails.
    """
    return _P0 + max(float(_bh(np.array(1.0), 0)),
                     float(_bh(np.array(1.0), 1))) * (T - _T0)


def _bm4_pressure(V: float, T: float, i: int) -> float:
    v0, k0, kp, kdp = _V0[i], _K0[i], _KP[i], _KDP[i]
    a1 = 3 * v0 * _P0
    a2 = 3 * v0 * (3 * k0 - 5 * _P0) / 2
    a3 = v0 * (9 * k0 * kp - 36 * k0 + 35 * _P0) / 2
    a4 = 3 * v0 * (9 * k0 ** 2 * kdp + 9 * k0 * kp ** 2
                   - 63 * k0 * kp + 143 * k0 - 105 * _P0) / 8
    f = 0.5 * ((v0 / V) ** (2.0 / 3.0) - 1.0)
    p_bm = (a1 + 2 * a2 * f + 3 * a3 * f ** 2 + 4 * a4 * f ** 3) \
        * (v0 ** (2.0 / 3.0)) / (3.0 * V ** (5.0 / 3.0))
    return p_bm + float(_bh(np.array(V / v0), i)) * (T - _T0)


def _solve_V(P: float, T: float, i: int) -> float:
    """Invert the BM4 + thermal EOS for V at (P, T).

    Bracketed on [0.15*V0, V0]: the upper bound is V0 by construction (we
    never use the expanded branch where the thermal parabola misbehaves),
    so the bracket is guaranteed whenever P >= p_splice(T). A bracketed
    solver is used deliberately -- Schaefer's MATLAB uses fsolve from a
    guess with Display='off', which returns a non-converged value silently.
    """
    lo, hi = 0.15 * _V0[i], _V0[i]
    f_lo = _bm4_pressure(lo, T, i) - P
    f_hi = _bm4_pressure(hi, T, i) - P
    if f_lo * f_hi > 0:
        return np.nan
    return brentq(lambda V: _bm4_pressure(V, T, i) - P, lo, hi, xtol=1e-10)


def _int_V_FeO(T: float, P: float) -> float:
    """Murnaghan integral of the Lange & Carmichael (1987) FeO liquid, J/mol."""
    V = (13650.0 + 2.92 * (T - 1673.0)) * 1e-3          # cm3/mol
    return (V * _K0_FEO / (_KP_FEO - 1.0)
            * ((1.0 + _KP_FEO * P / _K0_FEO) ** (1.0 - 1.0 / _KP_FEO) - 1.0)) * 1e3


def _V_Fe(P: float) -> float:
    """Vinet inversion for liquid Fe at 298 K reference, cm3/mol."""
    if P <= 1e-6:
        return _V298_FE

    def resid(V):
        x = (V / _V298_FE) ** (1.0 / 3.0)
        return P - 3 * _K0_FE * (1 - x) / x ** 2 \
            * np.exp(1.5 * (_KP_FE - 1) * (1 - x))

    return brentq(resid, 0.2 * _V298_FE, _V298_FE, xtol=1e-12)


def _cumint(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid, same length as x, starting at 0."""
    return np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))))


def _int_V_Fe_grid(T: float, Pgrid: np.ndarray) -> np.ndarray:
    """Komabayashi (2014) liquid Fe integrated along the whole P grid, J/mol.

    One cumulative sweep gives the integral at every P, so the Vinet
    inversions cost len(Pgrid) rather than len(Pgrid)**2.
    """
    VT = np.empty(len(Pgrid))
    for j, p in enumerate(Pgrid):
        Vp = _V_Fe(float(p))
        alpha = _ALPHA0_FE * np.exp(-_DELTA0_FE / _KAPPA_FE
                                    * (1.0 - (Vp / _V298_FE) ** _KAPPA_FE))
        VT[j] = Vp * np.exp(alpha * (T - 298.0))
    return _cumint(VT, Pgrid) * 1e3


class _Table:
    """Cached (P, T) grid of int dV dP for the disproportionation reaction."""

    def __init__(self, P_max: float, T_min: float, T_max: float,
                 nP: int, nT: int):
        # The grid starts at P0 = 1 bar, not at zero: that is the standard
        # state the Kowalski & Spencer Gibbs energies are referenced to, and
        # the BM4 residual has no sign change in [0.15*V0, V0] at P = 0
        # (P_BM(V0) = P0 by construction), so a zero first node would leave
        # an unsolvable point at the foot of every integral. Schaefer's
        # deltaGFeOFeO15_Deng.m likewise starts its PVector at 1e-4 GPa.
        self.P = np.concatenate(([_P0], np.linspace(_P0, P_max, nP)[1:]))
        self.T = np.linspace(T_min, min(T_max, T_CEILING), nT)
        self.I = np.zeros((nP, nT))

        for j, T in enumerate(self.T):
            Psp = p_splice(T)
            dV = np.empty(nP)
            for i, P in enumerate(self.P):
                if P < Psp:
                    dV[i] = _DV0                      # splice: uncompressed
                else:
                    v_ox = _solve_V(P, T, 1)
                    v_red = _solve_V(P, T, 0)
                    dV[i] = (np.nan if (np.isnan(v_ox) or np.isnan(v_red))
                             else 0.5 * (v_ox - v_red))
            # carry the last good value forward if the EOS failed anywhere
            if np.isnan(dV).any():
                bad = int(np.isnan(dV).sum())
                log.warning('Deng EOS: %d/%d grid points failed at T=%.0f K',
                            bad, nP, T)
                idx = np.where(~np.isnan(dV))[0]
                if idx.size == 0:
                    dV[:] = _DV0
                else:
                    dV = np.interp(self.P, self.P[idx], dV[idx])

            int_dV = _cumint(dV, self.P)
            # Referenced to P0, matching the two cumulative integrals above:
            # the Murnaghan form integrates from zero, so its value at the
            # reference pressure is subtracted off.
            int_FeO = np.array([_int_V_FeO(T, float(P)) for P in self.P])
            int_FeO = int_FeO - int_FeO[0]
            int_Fe = _int_V_Fe_grid(T, self.P)
            self.I[:, j] = 2.0 * int_dV - int_FeO + int_Fe

    def __call__(self, T, P):
        """Bilinear interpolation, clamped at the grid edges."""
        Pc = np.clip(P, self.P[0], self.P[-1])
        Tc = np.clip(T, self.T[0], self.T[-1])
        ip = np.clip(np.searchsorted(self.P, Pc) - 1, 0, len(self.P) - 2)
        it = np.clip(np.searchsorted(self.T, Tc) - 1, 0, len(self.T) - 2)
        wp = (Pc - self.P[ip]) / (self.P[ip + 1] - self.P[ip])
        wt = (Tc - self.T[it]) / (self.T[it + 1] - self.T[it])
        return ((1 - wp) * (1 - wt) * self.I[ip, it]
                + wp * (1 - wt) * self.I[ip + 1, it]
                + (1 - wp) * wt * self.I[ip, it + 1]
                + wp * wt * self.I[ip + 1, it + 1])


_TABLE: _Table | None = None


def build_table(P_max: float = P_EXERCISED * 1.05, T_min: float = 1500.0,
                T_max: float = T_CEILING, nP: int = 150, nT: int = 40) -> None:
    """Build and cache the (P, T) table. ~0.15 s; call once at init."""
    global _TABLE
    _TABLE = _Table(P_max, T_min, T_max, nP, nT)
    log.info('Deng EOS dV dP table built: %d x %d, P<=%.0f GPa, T=%.0f-%.0f K',
             nP, nT, P_max, _TABLE.T[0], _TABLE.T[-1])


def int_dV_dP(T, P):
    """int dV dP for 3FeO = 2FeO1.5 + Fe, from 1 bar to P at T. J/mol.

    Parameters
    ----------
    T : array_like, K
    P : array_like, GPa

    Returns
    -------
    (I, valid) : I is J/mol (0.0 where invalid, never silently so -- check
        ``valid``), valid is a boolean array marking points inside both the
        temperature ceiling and the pressure range Deng et al. were
        exercised over.
    """
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    if _TABLE is None:
        # Size the grid to the range that is actually usable. Everything
        # above P_EXERCISED is masked invalid below, so extending the table
        # to a deeper mantle would only spread the same number of nodes over
        # a wider span and coarsen the resolution where it is needed. A 5%
        # margin keeps the top of the valid range away from the grid edge.
        build_table(P_max=P_EXERCISED * 1.05)
    valid = (T <= T_CEILING) & (P <= P_EXERCISED) & np.isfinite(T) & np.isfinite(P)
    out = np.where(valid, _TABLE(T, P), 0.0)
    return out, valid
