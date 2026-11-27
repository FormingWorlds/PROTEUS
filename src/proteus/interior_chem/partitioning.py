"""Fe2+ partitioning between (Mg,Fe)SiO3 solid and melt from the Fe-Mg
exchange coefficient K_D(P, T), for the 'mgsio3_kd' prescription of the
melt-redox tracker (interior_chem/redox.py, Step 5-6).

K_D = (X_Fe^s X_Mg^l) / (X_Fe^l X_Mg^s), X_Fe = Fe2+ / (Fe2+ + Mg), is fitted
in both sources to

    K_D(P, T) = exp[-(dH - T dS + P dV) / (R T)]

with the exchange enthalpy, entropy and volume per mole of exchanged cation:
bridgmanite from Dragulet & Stixrude (2024, GRL 51, e2023GL107979, Fig. 4),
fitted at 23-140 GPa and 3000-6000 K; post-perovskite from Dragulet &
Stixrude (2026, arXiv:2609.13527, Eq. 8), fitted at 156-603 GPa and
6000-10000 K. Both papers write k_B in the fit but give the parameters per
mole, so R is used. The phase changes at the bridgmanite-post-perovskite-melt
triple point, 180 GPa, which the 2026 paper takes from Deng et al. (2023). Both papers
find K_D independent of X_Fe, which is why one K_D serves the whole
MgSiO3-FeSiO3 join.

Temperature is clamped to the fitted range of each phase, for bridgmanite
extended from 6000 K to 6500 K so that the 2026 paper's own use of the fit at
the triple point (0.057 at 180 GPa) is reproduced; the fit gives 0.057 there at
about 6430 K, a temperature inferred here, not stated in the paper. Pressure is not
clamped below 603 GPa: the 2026 paper itself evaluates the bridgmanite fit
up to 180 GPa; above 603 GPa post-perovskite takes the 603 GPa value.

Composition basis: the melt and the solid are (Mg,Fe)SiO3 with Fe3+ as
FeO1.5, so every Fe2+ carries its own SiO2 and the Mg of a phase is what its
mass leaves after FeSiO3 and FeO1.5. Mg is not tracked; it follows from the
masses, so it is conserved whenever the cell mass is.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import R as R_GAS  # J/mol/K

MU_FESIO3 = 0.131929  # molar mass of FeSiO3 in kg/mol (= 131.929 g/mol)
MU_MGSIO3 = 0.100389  # molar mass of MgSiO3 in kg/mol (= 100.389 g/mol)
MU_FEO15 = 0.07984  # molar mass of FeO1.5 in kg/mol (= 79.84 g/mol)

P_TRIPLE_GPA = 180.0  # bridgmanite below, post-perovskite at and above

# (dH [J/mol], dS [J/mol/K], dV [m^3/mol]) of MgSiO3(s) + FeSiO3(l) = MgSiO3(l) + FeSiO3(s)
KD_FIT = {
    'bdg': (-19.66e3, -12.04, 0.53e-6),
    'ppv': (122.0e3, 12.9, 0.52e-6),
}
# Temperature range [K] of each fit; T is clamped into it. Bridgmanite was fitted to
# 6000 K, but Dragulet & Stixrude (2026) evaluate it at the triple point (~6430 K, inferred).
KD_T_RANGE = {'bdg': (3000.0, 6500.0), 'ppv': (6000.0, 10000.0)}
P_PPV_MAX_GPA = 603.0  # highest pressure of the post-perovskite fit

_BISECT_ITERATIONS = 80


def kd_fe_mg(P_gpa, T) -> tuple[np.ndarray, np.ndarray]:
    """Fe-Mg exchange coefficient K_D (dimensionless) of the MgSiO3 phase
    stable at pressure ``P_gpa`` [GPa], evaluated at temperature ``T`` [K].

    Bridgmanite below P_TRIPLE_GPA, post-perovskite at and above it. Returns
    (K_D, clamped), where ``clamped`` is True for cells whose T or P lay
    outside the fit and were moved to its edge."""
    P_gpa = np.asarray(P_gpa, dtype=float)
    T = np.asarray(T, dtype=float)
    ppv = P_gpa >= P_TRIPLE_GPA
    kd = np.empty(np.broadcast(P_gpa, T).shape)
    clamped = np.zeros_like(kd, dtype=bool)
    P_b, T_b = np.broadcast_arrays(P_gpa, T)
    for phase, sel in (('bdg', ~ppv), ('ppv', ppv)):
        dH, dS, dV = KD_FIT[phase]
        T_lo, T_hi = KD_T_RANGE[phase]
        T_c = np.clip(T_b[sel], T_lo, T_hi)
        P_c = np.minimum(P_b[sel], P_PPV_MAX_GPA) if phase == 'ppv' else P_b[sel]
        clamped[sel] = (T_c != T_b[sel]) | (P_c != P_b[sel])
        kd[sel] = np.exp(-(dH + P_c * 1e9 * dV) / (R_GAS * T_c) + dS / R_GAS)
    return kd, clamped


def solid_x_fe(kd, x_liq):
    """Fe2+ / (Fe2+ + Mg) of the solid in equilibrium with a melt of
    ``x_liq``, the K_D definition solved for the solid (Dragulet & Stixrude
    2024 Eq. 10). Tends to K_D x_liq for x_liq -> 0 and to 1 for x_liq -> 1."""
    kd = np.asarray(kd, dtype=float)
    x_liq = np.asarray(x_liq, dtype=float)
    return kd * x_liq / (1.0 - (1.0 - kd) * x_liq)


def melt_n_mg(M_melt, n_fe2, n_fe3):
    """Moles of Mg [mol] in a melt of mass ``M_melt`` [kg] holding ``n_fe2``
    and ``n_fe3`` [mol]: the mass left after FeSiO3 and FeO1.5, as MgSiO3.
    Negative when the iron alone weighs more than the melt."""
    return (
        np.asarray(M_melt) - np.asarray(n_fe2) * MU_FESIO3 - np.asarray(n_fe3) * MU_FEO15
    ) / (MU_MGSIO3)


def _x_fe(n_fe2, n_mg):
    """Fe2+ / (Fe2+ + Mg), 0 where the phase holds neither."""
    tot = n_fe2 + n_mg
    return np.where(tot > 0.0, n_fe2 / np.where(tot > 0.0, tot, 1.0), 0.0)


def d_fe2_equilibrium(kd, x_liq, c_fe2_liq, c_fe3_solid):
    """Solid/melt ratio of Fe2+ mass concentration (mol/kg over mol/kg) for
    an infinitesimal amount of solid in equilibrium with the melt.

    ``c_fe2_liq`` [mol/kg] is the melt Fe2+ concentration, ``c_fe3_solid``
    [mol/kg] the Fe3+ the solid takes. The solid's mass is FeSiO3 + MgSiO3
    + FeO1.5 at the cation ratio of ``solid_x_fe``."""
    xs = solid_x_fe(kd, x_liq)
    c_fe2_solid = (
        xs * (1.0 - c_fe3_solid * MU_FEO15) / (xs * MU_FESIO3 + (1.0 - xs) * MU_MGSIO3)
    )
    return c_fe2_solid / c_fe2_liq


def fe2_into_solid_kd(
    n_fe2_c, n_fe3_c, n_fe3_solid, M_melt_c, s, kd, m_other_c=0.0
) -> np.ndarray:
    """Moles of Fe2+ [mol] that enter the solid formed in each cell this
    step, with the solid in equilibrium with the cell's residual melt at
    the exchange coefficient ``kd``.

    A cell's melt of mass ``M_melt_c`` [kg] holds ``n_fe2_c`` and ``n_fe3_c``
    [mol]; the fraction ``s`` of it crystallises and takes ``n_fe3_solid``
    [mol] of the Fe3+. ``m_other_c`` [kg] of that melt mass is neither
    (Mg,Fe)SiO3 nor FeO1.5 (the free SiO2 metal separation leaves, see
    redox._free_sio2_mass); it holds no Mg and stays in the residual melt.
    The unknown n2_s is the root of

        X_Fe(solid) - K_D X / (1 - (1 - K_D) X) = 0,   X = X_Fe(residual melt)

    with each phase's Mg from its mass (``melt_n_mg``). The left side rises
    with n2_s, between the bound where the residual melt has no Mg left
    (X = 1) and the one where the solid has none (or holds all the Fe2+),
    so bisection finds the single root. Fe2+ is conserved exactly, and Mg
    too because it follows from the masses. ``s = 1`` moves all Fe2+ to the
    solid; ``s = 0`` none."""
    n2 = np.asarray(n_fe2_c, dtype=float)
    n3 = np.asarray(n_fe3_c, dtype=float)
    n3s = np.asarray(n_fe3_solid, dtype=float)
    s = np.asarray(s, dtype=float)
    m_s = s * np.asarray(M_melt_c, dtype=float)
    m_l = (1.0 - s) * np.asarray(M_melt_c, dtype=float) - np.asarray(m_other_c, dtype=float)
    n3l = n3 - n3s

    lo = np.clip(n2 - (m_l - n3l * MU_FEO15) / MU_FESIO3, 0.0, n2)
    hi = np.clip((m_s - n3s * MU_FEO15) / MU_FESIO3, 0.0, n2)
    # An infeasible cell (lo > hi) cannot hold its iron at all; it keeps the
    # residual melt at its Mg-free bound, and redox Step 3d holds the step.
    hi = np.maximum(hi, lo)

    def h(n2s):
        xs = _x_fe(n2s, (m_s - n2s * MU_FESIO3 - n3s * MU_FEO15) / MU_MGSIO3)
        xl = _x_fe(n2 - n2s, (m_l - (n2 - n2s) * MU_FESIO3 - n3l * MU_FEO15) / MU_MGSIO3)
        return xs - solid_x_fe(kd, xl)

    for _ in range(_BISECT_ITERATIONS):
        mid = 0.5 * (lo + hi)
        up = h(mid) > 0.0
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    n2s = 0.5 * (lo + hi)
    n2s = np.where(s >= 1.0, n2, n2s)
    return np.where((s <= 0.0) | (n2 <= 0.0), 0.0, n2s)
