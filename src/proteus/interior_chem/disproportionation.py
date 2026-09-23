"""Fe metal saturation in a silicate melt, via the disproportionation reaction

    3 FeO(silicate liq) = 2 FeO1.5(silicate liq) + Fe(liq metal)      [Eq 7]

following Schaefer et al. (2024) JGR Planets 129, e2023JE008262, Section 2.7.

Deliberately free of PROTEUS imports: it takes numbers and returns numbers so
it can be unit-tested without an Interior_t. ``interior_chem/redox.py`` owns
all state and every decision about whether to apply the result.

PHYSICS
-------
Standard-state Gibbs energies are the Kowalski & Spencer (1995) CALPHAD
polynomials used by Schaefer's ``deltaGFeOFeO15_Deng.m``. Only three are
needed -- the O2 and Fe-reference terms cancel by stoichiometry, since the
reaction balances 3 Fe = 2 + 1 and 3x1 O = 2x1.5 O:

    dG0(T) = 2*G_FeO1.5(l) + G_Fe(l) - 3*G_FeO(l)
    K(T,P) = exp( -(dG0(T) + int dV dP) / RT )

Metal saturation is then a test, not a solve. Rearranging the mass-action
expression K = a_FeO1.5^2 * a_Fe / a_FeO^3 for the metal activity, with
a_i = gamma_i * n_i / n_sil, the n_sil powers give 3 - 2 = 1 surviving:

    a_Fe = K * gamma_FeO^3 * n2^3 / ( gamma_FeO1.5^2 * n3^2 * n_sil )

a_Fe >= 1 means the melt is supersaturated and metal should exsolve. Note
that the *absolute* melt inventory enters, not just the Fe3+/Fe2+ ratio,
because the reaction consumes three dissolved species and produces two --
it is sensitive to how dilute the melt is. (Contrast Hirschmann's Eq 21 for
fO2, which is mole-conserving in the melt, so its concentration dependence
cancels and only the ratio matters.)

When saturated we solve for the reaction extent xi, which because the Fe
stoichiometric coefficient is 1 *is* the moles of metal formed:

    n_FeO = n2 - 3*xi     n_FeO1.5 = n3 + 2*xi     n_metal = xi

Fe and O conservation are identities under this parameterisation, not
constraints to impose, so the nominal 3x3 system collapses to one scalar
equation. It is monotonic in xi and brackets automatically: F(0) is
RHS(0)*(a_Fe - 1), so the saturation test *is* the bracket check, and
RHS -> +inf as xi -> n2/3.

xi < 0 is the back-reaction (metal redissolving). It is needed here, unlike
in Schaefer's code, because PROTEUS keeps the metal co-located with the melt
rather than segregating it to the core, so it must be allowed to dissolve
when conditions change.

ACTIVITY COEFFICIENTS
---------------------
gamma_FeO = 1.55, gamma_FeO1.5 = 1.0 (Hirschmann 2022, as adopted by
Schaefer). Only the combination GAMMA = gamma_FeO^3 / gamma_FeO1.5^2 is
identifiable. The paper describes these as "poorly known"; f_crit scales as
GAMMA^0.5, so a factor-2 error in GAMMA moves the threshold by ~40%. Worth
treating as a stated uncertainty rather than a fixed truth.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from proteus.interior_chem import eos_deng

R = 8.31447

GAMMA_FEO = 1.55        # Hirschmann (2022); Schaefer et al. (2024) Section 2.7
GAMMA_FEO15 = 1.0
GAMMA = GAMMA_FEO ** 3 / GAMMA_FEO15 ** 2    # the only identifiable combination


# ── Kowalski & Spencer (1995) standard-state Gibbs energies, J/mol ──────────
def _g_fe_ref(T):
    T = np.asarray(T, dtype=float)
    return np.where(
        T < 1811,
        1225.7 + 124.134 * T - 23.5143 * T * np.log(T) - 4.39752e-3 * T ** 2
        - 5.89269e-8 * T ** 3 + 77358.5 / T,
        -25383.451 + 299.31255 * T - 46 * T * np.log(T) + 2.2960305e31 / T ** 9,
    )


def _g_fe_liquid(T):
    T = np.asarray(T, dtype=float)
    return np.where(
        T < 1811,
        12040.17 - 6.55843 * T - 3.6751551e-21 * T ** 7 + _g_fe_ref(T),
        -10839.7 + 291.302 * T - 46 * T * np.log(T),
    )


def _g_feo_liquid(T):
    T = np.asarray(T, dtype=float)
    return (-279318 + 252.848 * T - 46.12826 * T * np.log(T)
            - 5.7402984e-3 * T ** 2) + 34008 - 20.969 * T


def _g_feo15_liquid(T):
    T = np.asarray(T, dtype=float)
    return 0.5 * (-858683 + 827.946 * T - 137.0089 * T * np.log(T)
                  + 1453810 / T) + 39712 - 20.007 * T


def dG0(T):
    """Standard-state dG of 3FeO = 2FeO1.5 + Fe, J/mol. 1 bar, no P term."""
    return 2 * _g_feo15_liquid(T) + _g_fe_liquid(T) - 3 * _g_feo_liquid(T)


def K_eq(T, P_gpa=None, use_P_term: bool = False):
    """Equilibrium constant. Returns (K, valid).

    ``valid`` is all-True when the pressure term is off; with it on, it marks
    points inside the Deng EOS validity envelope (see eos_deng).
    """
    T = np.asarray(T, dtype=float)
    dG = dG0(T)
    if use_P_term:
        if P_gpa is None:
            raise ValueError('use_P_term=True requires P_gpa')
        I, valid = eos_deng.int_dV_dP(T, P_gpa)
        dG = dG + I
    else:
        valid = np.ones_like(T, dtype=bool)
    return np.exp(-dG / (R * T)), valid


def activity_Fe_metal(T, n2, n3, n_sil, P_gpa=None, use_P_term: bool = False):
    """a_Fe = K * GAMMA * n2^3 / (n3^2 * n_sil). Returns (a_Fe, valid).

    n2, n3, n_sil are melt-wide totals (the melt is assumed homogeneous, as
    in redox.py Step 4). Only K varies from cell to cell, through T and P.
    """
    K, valid = K_eq(T, P_gpa, use_P_term)
    if n3 <= 0.0 or n2 <= 0.0 or n_sil <= 0.0:
        return np.full(np.shape(K), np.inf if n3 <= 0.0 < n2 else 0.0), valid
    return K * GAMMA * n2 ** 3 / (n3 ** 2 * n_sil), valid


def critical_ferric_fraction(T, n_FeT, n_sil, P_gpa=None,
                             use_P_term: bool = False) -> float:
    """Fe3+/FeT at which a_Fe = 1. Metal saturates BELOW this value.

    Diagnostic only -- the model never needs it, but it is the quantity an
    f_0 sensitivity scan is really measuring.
    """
    K, _ = K_eq(T, P_gpa, use_P_term)
    K = float(K)

    def g(f):
        return K * GAMMA * n_FeT * (1 - f) ** 3 / (f ** 2 * n_sil) - 1.0

    if g(0.5) > 0:
        return np.nan            # saturated even at f = 0.5
    return brentq(g, 1e-12, 0.5, xtol=1e-14)


def solve_extent(K: float, n2: float, n3: float, n_sil: float,
                 n_metal_avail: float = 0.0) -> float:
    """Reaction extent xi at a_Fe = 1, for ONE parcel. Moles.

    xi > 0 metal forms, xi < 0 metal redissolves, 0 nothing happens.
    Fe and O conservation hold identically; see the module docstring.
    """
    if n2 <= 0.0 or n3 <= 0.0 or n_sil <= 0.0:
        return 0.0

    def F(xi):
        n_feo = n2 - 3.0 * xi
        if n_feo <= 0.0:
            return -np.inf
        return K - (n3 + 2.0 * xi) ** 2 * (n_sil - xi) \
            / (GAMMA_FEO ** 3 * n_feo ** 3)

    f0 = F(0.0)

    if f0 > 0.0:                                   # supersaturated -> form metal
        hi = n2 / 3.0 * (1.0 - 1e-12)
        if F(hi) > 0.0:                            # should not happen: RHS -> inf
            return hi
        return brentq(F, 0.0, hi, xtol=1e-18, rtol=1e-12)

    if f0 < 0.0 and n_metal_avail > 0.0:           # undersaturated -> dissolve
        lo = -n_metal_avail
        if n3 + 2.0 * lo <= 0.0:                   # cannot un-make more Fe3+
            lo = -n3 / 2.0 * (1.0 - 1e-12)
        if F(lo) <= 0.0:
            return lo                              # all available metal dissolves
        return brentq(F, lo, 0.0, xtol=1e-18, rtol=1e-12)

    return 0.0
