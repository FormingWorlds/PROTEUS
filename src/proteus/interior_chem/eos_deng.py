"""Pressure term for the Fe-disproportionation reaction.

Provides ``int_dV_dP(T, P)``, the integral of the reaction volume change

    3 FeO(silicate liq) = 2 FeO1.5(silicate liq) + Fe(liq metal)

from 1 bar to P along an isotherm at T, in J/mol. This is the term Schaefer
et al. (2024) always include in their disproportionation calculation
(``disproportionation.m`` passes the full basal pressure straight through to
``deltaGFeOFeO15_Deng.m``), in contrast to their surface fO2 routine
``fO2lowP_H22.m``, which correctly sets it to zero because that is evaluated
at 1 bar.

The integral is evaluated exactly as ``deltaGFeOFeO15_Deng.m`` and
``BM4VolumeFunc.m`` (Schaefer et al. 2024 archived code) construct it:

    int dV dP = 2*int(dV_Deng) dP  -  int(V_FeO) dP  +  int(V_Fe) dP

* dV_Deng = V(FeO1.5) - V(FeO) = 0.5*(V_ox - V_red), from the two Deng et al.
  (2020) 12.5 mol% FeO* melt endmembers (Mg14Fe2Si16O48 and Mg14Fe2Si16O49,
  which differ by exactly one oxygen), each a fourth-order Birch-Murnaghan
  EOS plus the thermal pressure BH(V/V0)*(T - T0), T0 = 3000 K. Integrated
  with the trapezoid rule on Schaefer's 100-node grid: P0 = 1e-4 GPa, then
  P*k/99 for k = 1..99.
* V_FeO is the Lange & Carmichael (1987) liquid volume at T with a
  Murnaghan compression (K0 = 30.33 GPa, K' = 4), integrated analytically
  from P = 0 to P, as in Schaefer's code.
* V_Fe is liquid iron on the Komabayashi (2014) Vinet EOS with
  Anderson-Gruneisen thermal expansion, integrated with the trapezoid rule
  on 100 equally spaced nodes from P0 to P.

``int_dV_dP_oxidation(T, P)`` returns the first term on its own,
int(dV_Deng) dP, the volume integral of the oxidation reaction
FeO + 1/4 O2 = FeO1.5 (Schaefer et al. 2024 Eq 10), used for the radial fO2
profile (Hirschmann 2022 Eq 21 / Schaefer et al. 2024 Eq 13). The O2 gas is
not part of the condensed-phase volume change, as fO2 is a fugacity.

VOLUME INVERSION
----------------
Schaefer solves P(V, T) = P with MATLAB ``fsolve`` from V = 0.95*V0, output
suppressed and exit flag unchecked. Here the same root is found with a
bracketed, vectorised bisection:

* P >= P(V0, T): the root lies on the compressed branch, V in [0.15 V0, V0],
  where P(V) is monotonic, so it is the unique root fsolve converges to.
* P < P(V0, T) (only above T0, where the thermal term is positive): the
  liquid expands past V0. P(V) keeps falling to a minimum near V/V0 ~ 1.1-1.35
  and then rises; the root between V0 and that minimum is the one fsolve
  reaches from 0.95*V0.
* If that minimum lies above the requested P there is no root (at 1 bar this
  happens above ~4175 K for the ferric endmember). MATLAB's fsolve then
  returns, unconverged, the point where the squared residual is stationary,
  i.e. the minimum of P(V). That volume is used here, and a warning is
  logged once per process. The exact unconverged MATLAB value depends on its
  iteration history and cannot be reproduced; this is its limit point.

VALIDITY
--------
Schaefer's code has no validity envelope. PROTEUS keeps one, outside the EOS
formula: points with T > ``T_CEILING`` or P > ``P_EXERCISED`` (136 GPa,
roughly Earth's core-mantle boundary and the deepest Schaefer et al. ran) are
returned as 0 and flagged invalid. For Earth-like magma oceans Schaefer never
leaves this envelope, so inside it the results are hers.

``T_CEILING`` is 6500 K, an extrapolation 1500 K beyond ``T_DENG_MAX``
(5000 K, the upper limit of the Deng et al. 2020 thermodynamic modelling;
their FPMD spans 2000-4000 K). It is set to the hottest melt of an Earth-like
run that starts fully molten (~6500 K at the core-mantle boundary), so that
the whole homogeneous melt is tested for metal saturation from the first
check and the reaction extent is not set by a subset of it. Values above
``T_DENG_MAX`` are extrapolations of every term (Deng volumes, the FeO and Fe
volumes, the Gibbs energies) and are logged once per process.
"""

from __future__ import annotations

import logging

import numpy as np

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

_P0 = 1.0e-4  # GPa, Deng reference pressure and foot of Schaefer's grid
_T0 = 3000.0  # K,   Deng reference temperature
_N_GRID = 100  # nodes of Schaefer's trapezoid integrations
_N_BISECT = 60  # bisection halvings: bracket / 2**60 is far below fsolve's tolerance

T_DENG_MAX = 5000.0  # K,   upper limit of Deng et al. (2020) modelling
T_CEILING = 6500.0  # K,   validity envelope, extrapolated beyond T_DENG_MAX
P_EXERCISED = 136.0  # GPa, deepest pressure Schaefer et al. (2024) ran

# ── Lange & Carmichael (1987) FeO liquid + Kress & Carmichael K ─────────────
_K0_FEO, _KP_FEO = 30.33, 4.0

# ── Komabayashi (2014) liquid Fe, Vinet ─────────────────────────────────────
_V298_FE, _ALPHA0_FE = 6.88, 9.0e-5
_K0_FE, _KP_FE, _DELTA0_FE, _KAPPA_FE = 148.0, 5.8, 5.1, 0.56

_NO_ROOT_WARNED = False
_EXTRAPOLATION_WARNED = False


def _bh(x, i: int):
    """Deng thermal-pressure coefficient, GPa/K. x = V/V0."""
    return (_A[i] - _B[i] * x + _C[i] * x**2) / 1000.0


def _bm4_pressure(V, T, i: int):
    """Deng P(V, T) in GPa for endmember ``i``; V in J/GPa/mol, T in K.
    Broadcasts over V and T."""
    v0, k0, kp, kdp = _V0[i], _K0[i], _KP[i], _KDP[i]
    a1 = 3 * v0 * _P0
    a2 = 3 * v0 * (3 * k0 - 5 * _P0) / 2
    a3 = v0 * (9 * k0 * kp - 36 * k0 + 35 * _P0) / 2
    a4 = 3 * v0 * (9 * k0**2 * kdp + 9 * k0 * kp**2 - 63 * k0 * kp + 143 * k0 - 105 * _P0) / 8
    f = 0.5 * ((v0 / V) ** (2.0 / 3.0) - 1.0)
    p_bm = (
        (a1 + 2 * a2 * f + 3 * a3 * f**2 + 4 * a4 * f**3)
        * (v0 ** (2.0 / 3.0))
        / (3.0 * V ** (5.0 / 3.0))
    )
    return p_bm + _bh(V / v0, i) * (T - _T0)


def p_at_V0(T, i: int):
    """Pressure at which endmember ``i`` sits exactly at V0, in GPa.

    The cold term equals P0 at V0 by construction, so this is
    P0 + BH(1)*(T - T0). Below it the root is on the expanded branch
    (V > V0); it is below 1 bar whenever T < T0.
    """
    return _P0 + _bh(1.0, i) * (np.asarray(T, dtype=float) - _T0)


def _bisect_decreasing(func, lo, hi):
    """Vectorised bisection for a decreasing ``func`` with func(lo) >= 0 >= func(hi)."""
    for _ in range(_N_BISECT):
        mid = 0.5 * (lo + hi)
        pos = func(mid) > 0.0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


def _expanded_minimum(T, i: int):
    """Volume of the minimum of P(V) beyond V0 at each T, J/GPa/mol: a dense
    scan over V/V0 in [1, 1.6] refined by golden-section search."""
    T = np.asarray(T, dtype=float)
    v0 = _V0[i]
    xs = np.linspace(1.0, 1.6, 241)
    Ps = _bm4_pressure(xs[None, :] * v0, T[:, None], i)
    k = np.argmin(Ps, axis=1)
    lo = xs[np.maximum(k - 1, 0)]
    hi = xs[np.minimum(k + 1, xs.size - 1)]
    g = 0.5 * (np.sqrt(5.0) - 1.0)
    for _ in range(40):
        x1 = hi - g * (hi - lo)
        x2 = lo + g * (hi - lo)
        left = _bm4_pressure(x1 * v0, T, i) < _bm4_pressure(x2 * v0, T, i)
        hi = np.where(left, x2, hi)
        lo = np.where(left, lo, x1)
    return 0.5 * (lo + hi) * v0


def _volumes(P, T, i: int):
    """Endmember volume V(P, T) in J/GPa/mol, the root Schaefer's fsolve finds.

    P has shape (n, m) and T shape (n,). Returns (V, no_root), where no_root
    marks points with no solution, set to the minimum of P(V) (see module
    docstring).
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    Tc = T[:, None]
    v0 = _V0[i]

    def resid(V):
        return _bm4_pressure(V, Tc, i) - P

    V = np.full(P.shape, np.nan)
    comp = P >= p_at_V0(Tc, i)
    lo, hi = np.full(P.shape, 0.15 * v0), np.full(P.shape, v0)
    reach = resid(lo) >= 0.0
    root = _bisect_decreasing(resid, lo, hi)
    V = np.where(comp & reach, root, V)

    no_root = np.zeros(P.shape, dtype=bool)
    expd = ~comp
    if np.any(expd):
        v_min = np.broadcast_to(_expanded_minimum(T, i)[:, None], P.shape)
        has = expd & (resid(v_min) <= 0.0)
        root = _bisect_decreasing(resid, np.full(P.shape, v0), v_min.copy())
        V = np.where(has, root, V)
        no_root = expd & ~has
        V = np.where(no_root, v_min, V)
    return V, no_root


def _solve_V(P: float, T: float, i: int) -> float:
    """Scalar V(P, T) for endmember ``i``, J/GPa/mol (see ``_volumes``)."""
    V, _ = _volumes(np.array([[float(P)]]), np.array([float(T)]), i)
    return float(V[0, 0])


def _int_V_FeO(T, P):
    """Murnaghan integral of the Lange & Carmichael (1987) FeO liquid from
    P = 0 to P, J/mol (Schaefer's ``VdPFeO_LC``). T in K, P in GPa."""
    V = (13650.0 + 2.92 * (np.asarray(T, dtype=float) - 1673.0)) * 1e-3  # cm3/mol
    return (
        V
        * _K0_FEO
        / (_KP_FEO - 1.0)
        * (
            (1.0 + _KP_FEO * np.asarray(P, dtype=float) / _K0_FEO) ** (1.0 - 1.0 / _KP_FEO)
            - 1.0
        )
    ) * 1e3


def _vinet_resid(V, P):
    x = (V / _V298_FE) ** (1.0 / 3.0)
    return P - 3 * _K0_FE * (1 - x) / x**2 * np.exp(1.5 * (_KP_FE - 1) * (1 - x))


def _V_Fe(P):
    """Vinet inversion for liquid Fe at the 298 K reference, cm3/mol. P in GPa."""
    P = np.asarray(P, dtype=float)
    lo = np.full(P.shape, 0.2 * _V298_FE)
    hi = np.full(P.shape, _V298_FE)
    # The residual rises with V, so bisect its negative.
    V = _bisect_decreasing(lambda v: -_vinet_resid(v, P), lo, hi)
    V = np.where(P <= 1e-6, _V298_FE, V)
    return float(V) if V.ndim == 0 else V


def _VT_Fe(P, T):
    """Liquid Fe volume at (P, T), cm3/mol, with Anderson-Gruneisen expansion."""
    Vp = _V_Fe(P)
    alpha = _ALPHA0_FE * np.exp(-_DELTA0_FE / _KAPPA_FE * (1.0 - (Vp / _V298_FE) ** _KAPPA_FE))
    return Vp * np.exp(alpha * (T - 298.0))


def _trapz(y, x):
    """Trapezoid rule along the last axis."""
    return np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x, axis=-1), axis=-1)


def _integrals(T, P, deng_only=False):
    """The three terms for flat arrays T [K], P [GPa], in J/mol:
    (int dV_Deng dP, int V_FeO dP, int V_Fe dP); the last two are None
    when ``deng_only``."""
    global _NO_ROOT_WARNED
    k = np.arange(1, _N_GRID)
    grid_d = np.concatenate((np.full((P.size, 1), _P0), P[:, None] * k / (_N_GRID - 1)), axis=1)
    V_red, nr_red = _volumes(grid_d, T, 0)
    V_ox, nr_ox = _volumes(grid_d, T, 1)
    no_root = nr_red | nr_ox
    if np.any(no_root) and not _NO_ROOT_WARNED:
        _NO_ROOT_WARNED = True
        log.warning(
            'Deng EOS: no volume root at %d node(s) (T up to %.0f K, P down to %.3g GPa); '
            'using the residual minimum, as an unconverged fsolve in Schaefer et al. does',
            int(np.count_nonzero(no_root)),
            float(np.max(np.broadcast_to(T[:, None], no_root.shape)[no_root])),
            float(np.min(grid_d[no_root])),
        )
    I_deng = _trapz(0.5 * (V_ox - V_red), grid_d)
    if deng_only:
        return I_deng, None, None

    grid_fe = _P0 + (P[:, None] - _P0) * np.linspace(0.0, 1.0, _N_GRID)
    I_fe = _trapz(_VT_Fe(grid_fe, T[:, None]), grid_fe) * 1e3
    return I_deng, _int_V_FeO(T, P), I_fe


def _evaluate(T, P, combine, deng_only=False):
    """Shared driver: validity mask, evaluation on valid points, 0 elsewhere."""
    global _EXTRAPOLATION_WARNED
    T, P = np.broadcast_arrays(np.asarray(T, dtype=float), np.asarray(P, dtype=float))
    shape = T.shape
    T, P = T.ravel(), P.ravel()
    valid = np.isfinite(T) & np.isfinite(P) & (T <= T_CEILING) & (P <= P_EXERCISED)
    if not _EXTRAPOLATION_WARNED and np.any(valid & (T > T_DENG_MAX)):
        _EXTRAPOLATION_WARNED = True
        log.warning(
            'Deng EOS evaluated above its %.0f K modelling limit (up to %.0f K); '
            'values there are extrapolated',
            T_DENG_MAX,
            float(np.max(T[valid])),
        )
    out = np.zeros(T.shape)
    if np.any(valid):
        vals = combine(*_integrals(T[valid], P[valid], deng_only))
        # A compressed-branch point beyond the bracket is NaN: flag it, never return it.
        ok = np.isfinite(vals)
        out[np.flatnonzero(valid)[ok]] = vals[ok]
        valid[np.flatnonzero(valid)[~ok]] = False
    return out.reshape(shape), valid.reshape(shape)


def int_dV_dP(T, P):
    """int dV dP for 3FeO = 2FeO1.5 + Fe, from 1 bar to P at T. J/mol.

    Parameters
    ----------
    T : array_like, K
    P : array_like, GPa

    Returns
    -------
    (I, valid) : I is J/mol (0.0 where invalid, never silently so -- check
        ``valid``), valid is a boolean array marking points inside the
        temperature ceiling and the pressure range Schaefer et al. exercised.
    """
    return _evaluate(T, P, lambda deng, feo, fe: 2.0 * deng - feo + fe)


def int_dV_dP_oxidation(T, P):
    """int dV dP for FeO + 1/4 O2 = FeO1.5, from 1 bar to P at T. J/mol.

    dV = V(FeO1.5) - V(FeO) per Fe, from the two Deng et al. (2020) melt
    endmembers (Schaefer et al. 2024 Eq 10-11, ``deltaVdP`` in
    ``deltaGFeOFeO15_Deng.m``), with the same validity limits as
    ``int_dV_dP``. This is the pressure term of Hirschmann (2022) Eq 21 /
    Schaefer et al. (2024) Eq 13.

    Parameters
    ----------
    T : array_like, K
    P : array_like, GPa

    Returns
    -------
    (I, valid) : as for ``int_dV_dP``.
    """
    return _evaluate(T, P, lambda deng, feo, fe: deng, deng_only=True)
