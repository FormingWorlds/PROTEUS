"""Melt Fe3+/Fe2+ redox tracking during fractional crystallization.

Implements the radial Fe3+/Fe2+ tracking framework referenced by
planet.fO2_source = 'from_mantle_redox' (issue #653). Tracks the global
Fe3+/Fe2+ reservoirs in the melt as the mantle crystallizes, using
prescribed (not equilibrium-derived) partition coefficients between melt
and newly formed solid -- the same fractional-crystallization bookkeeping
prototyped standalone in tools/redox_step.py, restructured here to
run one step at a time inside PROTEUS's live coupling loop rather than
post-processing a full SPIDER output series.

Physics summary (see tools/redox_step.py for the full derivation
this was validated against):

  Step 1-2  initial melt iron inventory: Fe2+ from the ferrous FeO fraction
            W_FET, Fe3+ = Fe2+ * f_0 / (1 - f_0) on top of it
  Step 3    new solid mass each step, from the decrease in local melt
            fraction (delta_phi), using the mass-invariant _s grid. A cell
            with phi < PHI_SOLID (0.05) counts as solid throughout (phi set
            to 0 on entry), so its last trace of melt crystallises when it
            crosses the threshold
  Step 4    redistribute the *previous* step's global Fe3+/Fe2+ reservoirs
            across cells in proportion to local melt mass -- this is what
            makes the Fe3+/Fe2+ ratio spatially uniform: only the absolute
            per-cell amounts vary, not the ratio itself
  Step 5-6  Fe3+/Fe2+ moles entering the new solid, via depth-dependent
            partition coefficients (bridgmanite below P_CUTOFF_GPA,
            Cpx/Opx above it)
  Step 7-9  update the global reservoirs and the resulting ferric
            fraction / Fe3+/Fe2+ redox ratio
  Step 9a-9h  Fe metal saturation, run unconditionally between every pair of
            crystallization steps. Whether the melt is supersaturated is a
            property of the melt rather than a user option, and carrying a
            supersaturation would leave the model metastable in the sense
            Schaefer et al. flag for their Figures 2 and 4.
            Between crystallization and the fO2 evaluation -- Schaefer
            et al. (2024) Section 2.7 "add an additional step in between each
            crystallization step to check for metal saturation" -- the melt
            is tested against the disproportionation reaction
            3FeO = 2FeO1.5 + Fe. The metal activity

                a_Fe = K(T,P) * GAMMA * n2^3 / (n3^2 * n_sil)

            is evaluated per cell; a_Fe >= 1 means supersaturated. Only the
            most supersaturated cell can host equilibrium, because Step 4
            makes the melt homogeneous, so one scalar root find on the
            reaction extent xi suffices. Fe and O conservation are identities
            under that parameterisation. Metal is deposited where it formed
            and does not move, so n_fe_metal_cell is spatially resolved while
            the melt stays uniform; only the forward reaction is applied
            (xi >= 0), so metal is never redissolved. Because metal
            removes Fe but leaves the O behind, saturation drives Fe3+/FeT
            *up*, and the melt is buffered at the saturation value regardless
            of f_0. The equilibrium constant always carries int(dV dP) from
            the Deng et al. (2020) EOS (interior_chem/eos_deng.py): at depth
            2*V(FeO1.5) + V(Fe) < 3*V(FeO), and that volume change is what
            makes disproportionation favourable there. Cells outside the EOS
            envelope (T > 5000 K or P > 136 GPa) are excluded from the
            check. See interior_chem/disproportionation.py for the
            derivation.
  Step 10   surface fO2 via Hirschmann (2022) GCA 313 Eq 21 (= Schaefer
            et al. 2024 Eq 13) at 1 bar, with the pressure/EOS term
            (integral of Delta V dP) zero, using the single melt
            Fe3+/Fe2+ after Step 9a-9h, at the outgassing temperature
            max(T_magma, outgas.T_floor). This is Schaefer's surface
            calculation (fO2lowP_H22.m, "without the high pressure term",
            P = 0.0001 GPa), while the disproportionation in Step 9 always
            carries the pressure term. Delta-IW uses the O'Neill & Eggins
            (2002) buffer as given in Bower et al. (2022) PSJ 3, 93,
            Eq 7-8, at the same temperature. Separately, a radial fO2
            profile is evaluated in every melt cell at that cell's T and P
            WITH int(dV dP) for FeO + 1/4 O2 = FeO1.5 (Schaefer Eq 10-11,
            interior_chem/eos_deng.int_dV_dP_oxidation; the oxidation dV,
            not the disproportionation dV of Step 9c) and stored in the
            interior snapshot as a diagnostic; it does not feed the
            outgassing.

BSE starting composition and partition coefficients are the same
literature values used in tools/redox_step.py (McDonough 2003 via
Hirschmann 2022 for the oxide wt%; Schaefer et al. 2024 Table 3 / Eq 6
for the partition coefficients); they are module constants here rather
than config fields, matching the "physics defaults, not user knobs"
treatment of comparable fixed stoichiometry elsewhere in the outgas
package (e.g. proteus.outgas.dummy._ELEMENT_TO_SPECIES).

The one exception is the initial ferric fraction f_0, which IS a config
field (``planet.ferric_fraction_initial``, default 0.1). It is the least
well constrained of these numbers and the one a parameter study most
naturally varies, so it is exposed rather than fixed.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from proteus.interior_chem import disproportionation as dispro
from proteus.interior_chem import eos_deng
from proteus.utils.constants import element_mmw

if TYPE_CHECKING:
    from proteus.config import Config
    from proteus.interior_energetics.common import Interior_t

log = logging.getLogger('fwl.' + __name__)

# ── Step 0: fixed input parameters (McDonough 2003 BSE, Schaefer et al. 2024
# Table 1; partition coefficients Schaefer et al. 2024 Table 3 / Eq 6) ──────
# Ferrous FeO mass fraction of the melt. Schaefer et al. (2024) Table 1 lists
# FeO (7.82 wt%) and FeO1.5 as separate oxides, so this is Fe2+ only: it sets
# the initial Fe2+ reservoir, and Fe3+ is added on top of it from f_0 (Step 2).
# The non-iron mass that becomes MgSiO3 in _metal_saturation_step is
# therefore 1 - W_FET - w_FeO1.5, with w_FeO1.5 fixed at initialisation.
W_FET = 0.08  # ferrous FeO mass fraction in the melt (dimensionless)
MU_FEO = 0.07184  # molar mass of FeO in kg/mol (= 71.84 g/mol)
MU_FEO15 = 0.07984  # molar mass of FeO1.5 in kg/mol (= 79.84 g/mol)
MU_MGSIO3 = 0.100389  # molar mass of MgSiO3 in kg/mol (= 100.389 g/mol)
# Default initial ferric fraction Fe3+/FeT. The value actually used is
# config.planet.ferric_fraction_initial, which defaults to this; the
# constant is kept so the dataclass has a sane placeholder before
# _init_state overwrites it, and to document the published default.
F_0 = 0.10  # initial ferric fraction Fe3+/FeT (Schaefer et al. 2024)

# Melt-fraction threshold below which a cell counts as solid everywhere in
# the tracker (melt inventory, crystallization, Step 4 homogenisation, the
# metal check and the fO2 profile). The interior solver's phi is only
# exactly 0 below the solidus, so near-solid mush with a trace of melt would
# otherwise stay "melt": it would keep its share of the Fe reservoirs and
# could host the metal-saturation binding cell. A cell crossing the
# threshold crystallises its remaining melt in that step (Step 3).
PHI_SOLID = 0.05

D_FE2_BRG = 0.85  # bridgmanite/melt partition coefficient for Fe2+ (both regimes)

P_CUTOFF_GPA = 22.0  # halt bridgmanite assemblage above this pressure
D_FE3_BRG = 0.75  # bridgmanite/melt (Table 3)
D_FE3_CPX = 0.45  # clinopyroxene/melt (Table 3, Mallmann & O'Neill 2009)
D_FE3_OPX = 0.70 * D_FE3_CPX  # orthopyroxene/melt = D_opx/cpx x D_cpx/melt (Eq 6)
D_FE3_SHALLOW = (D_FE3_CPX + D_FE3_OPX) / 2  # plain mean, no modal Cpx/Opx given

# BSE starting composition (wt%)
_WT_OXIDES = {
    'FeO': 7.82,
    'MgO': 38.3,
    'SiO2': 45.5,
    'CaO': 3.58,
    'Al2O3': 4.49,
    'FeO15': 0.36,
    'Na2O': 0.0,
    'K2O': 0.0,
    'TiO2': 0.0,
    'P2O5': 0.0,
}

# Hirschmann (2022) Eq 21 fixed coefficients
_R = 8.31447
_A = 0.19317
_B = -4.51412 / 2.303
_C_PARAM = 9574.293 / 2.303
_DELTA_CP = 33.25
_T0 = 1673.15
_YS = [
    y / 2.303
    for y in (
        -1198.4,
        -426.82,
        1138.371,
        4232.933,
        6650.972,
        7998.434,
        -10298.6,
        -2866.92,
        -2663.74,
    )
]


def _compute_mole_fractions():
    """Single-cation-basis oxide mole fractions for Hirschmann (2022) Eq 21.

    Na2O, K2O, Al2O3 and P2O5 have 2 cations per formula unit and are
    expressed per single cation, matching Hirschmann's point (2)
    modification to Eq 20.
    """
    M = {
        'SiO2': 60.08,
        'TiO2': 79.87,
        'MgO': 40.30,
        'CaO': 56.08,
        'Na2O': 61.98,
        'K2O': 94.20,
        'P2O5': 141.94,
        'Al2O3': 101.96,
        'FeO': 71.84,
        'FeO15': 79.84,
    }
    cations_per_unit = {
        'SiO2': 1,
        'TiO2': 1,
        'MgO': 1,
        'CaO': 1,
        'Na2O': 2,
        'K2O': 2,
        'P2O5': 2,
        'Al2O3': 2,
        'FeO': 1,
        'FeO15': 1,
    }
    n_cation = {k: cations_per_unit[k] * _WT_OXIDES[k] / M[k] for k in M}
    n_total = sum(n_cation.values())
    return {k: v / n_total for k, v in n_cation.items()}


def _log10_fO2(redox_ratio, T, X: dict, int_dV_dP=0.0):
    """log10(fO2), Hirschmann (2022) Eq 21 / Schaefer et al. (2024) Eq 13.

    ``T`` [K] and ``int_dV_dP`` [J/mol, oxidation reaction FeO -> FeO1.5,
    1 bar to P] may be arrays of the same shape; ``redox_ratio`` is
    X_FeO1.5 / X_FeO. Eq 13 carries the pressure term as
    -int(dV dP)/(R T ln10) on the Fe3+/Fe2+ side, so at fixed ratio it
    raises log10(fO2) by int(dV dP)/(a R T ln10)."""
    T = np.asarray(T, dtype=float)
    Xi = [X['SiO2'], X['TiO2'], X['MgO'], X['CaO'], X['Na2O'], X['K2O'], X['P2O5'], X['Al2O3']]
    loggammas_const = (
        _YS[0] * Xi[0]
        + _YS[1] * Xi[1]
        + _YS[2] * Xi[2]
        + _YS[3] * Xi[3]
        + _YS[4] * Xi[4]
        + _YS[5] * Xi[5]
        + _YS[6] * Xi[6]
        + _YS[7] * Xi[7] * Xi[0]
        + _YS[8] * Xi[0] * Xi[2]
    )
    logXFe3Fe2 = math.log10(redox_ratio)
    dG_RT = (
        _B + _C_PARAM / T - (_DELTA_CP / _R / math.log(10)) * (1 - _T0 / T - np.log(T / _T0))
    )
    pressure_term = np.asarray(int_dV_dP, dtype=float) / (_R * T * math.log(10))
    loggammas = loggammas_const / T
    return (logXFe3Fe2 - dG_RT + pressure_term - loggammas) / _A


def _log10_fO2_surface(redox_ratio: float, T: float, X: dict) -> float:
    """log10(fO2) at 1 bar: Eq 13 with int(dV dP) = 0."""
    return float(_log10_fO2(redox_ratio, T, X, 0.0))


def _log10_fO2_profile(
    redox_ratio: float, temp: np.ndarray, P_gpa: np.ndarray, X: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell log10(fO2) from Eq 13 at each cell's (T, P).

    Returns (log10_fO2, valid). Cells outside the Deng EOS envelope
    (T > T_CEILING or P > P_EXERCISED) have no pressure term and are NaN,
    flagged False in ``valid``."""
    I_ox, valid = eos_deng.int_dV_dP_oxidation(temp, P_gpa)
    prof = np.where(valid, _log10_fO2(redox_ratio, temp, X, I_ox), np.nan)
    return prof, valid


def _iw_buffer_bower2022(T: float) -> float:
    """log10(fO2) of the IW buffer: O'Neill & Eggins (2002) as given in
    Bower et al. (2022) PSJ 3, 93, Eq 7. T-only, no pressure term."""
    return (-244118 + 115.559 * T - 8.474 * T * math.log(T)) / (0.5 * math.log(10) * _R * T)


def _update_ratios(state: MeltRedoxState) -> None:
    """Step 8-9 (Eq 20-24): derive ferric fraction and redox ratio from the
    reservoirs. Called after crystallization, and again after the metal step
    if any metal formed or dissolved."""
    n_FeT_melt = state.n_fe3_melt + state.n_fe2_melt
    if n_FeT_melt > 0:
        state.ferric_frac = state.n_fe3_melt / n_FeT_melt
        state.redox_ratio = state.ferric_frac / (1.0 - state.ferric_frac)


@dataclass
class MeltRedoxState:
    """Persistent Fe3+/Fe2+ tracking state, one instance per run (held at
    Interior_t.redox_state), carried across coupling-loop timesteps."""

    n_fe3_melt: float
    n_fe2_melt: float
    phi_prev: np.ndarray
    D_fe3_cell: np.ndarray
    # Per-cell Fe metal inventory [mol]. Persistent and spatially resolved:
    # metal is a dense separate phase that stays where it formed, while the
    # silicate melt is homogenised every step by Step 4. This is the field a
    # later percolation/transport model consumes.
    n_fe_metal_cell: np.ndarray
    # Per-cell metal activity, overwritten each step. Diagnostic only, but
    # logged even when below 1 so the distance from the buffer is visible.
    a_fe_cell: np.ndarray
    # Index (into the interior_o staggered-grid arrays) of the cell with the
    # largest a_Fe this step, i.e. the binding cell cstar that receives any
    # metal formed. -1 when no cell was tested. Overwritten each step.
    a_fe_max_cell: int = -1
    ferric_frac: float = F_0
    redox_ratio: float = field(default=F_0 / (1.0 - F_0))
    # Initial FeO1.5 mass fraction of the melt, set from f_0 in Step 2 and held
    # fixed; enters the non-iron melt mass in the metal step.
    w_feo15: float = 0.0
    # Per-cell log10(fO2) from Eq 13 with int(dV dP) (Step 10a), NaN outside
    # the melt or the EOS envelope. Overwritten each step. Diagnostic only;
    # fO2_cell is the uppermost melt cell, the shallowest point of it.
    log10_fO2_cell: np.ndarray | None = None
    fO2_cell: int = -1
    melt_exhausted: bool = False
    eos_coverage_logged: bool = False
    X: dict = field(default_factory=_compute_mole_fractions)


def _init_state(
    phi: np.ndarray, mass: np.ndarray, pres: np.ndarray, f_0: float
) -> MeltRedoxState:
    """Step 1-2 (Eq 1-3): seed the global Fe3+/Fe2+ reservoirs from the
    interior state at the first call, and the per-cell bridgmanite/Cpx-Opx
    split from the (time-invariant) pressure profile.

    W_FET is the ferrous FeO fraction, so it fixes n_Fe2 directly. Fe3+ is
    then set so that n_Fe3 / (n_Fe2 + n_Fe3) = f_0, which rearranges to
    n_Fe3 = n_Fe2 * f_0 / (1 - f_0). Total iron is n_Fe2 + n_Fe3.

    ``f_0`` is the initial ferric fraction Fe3+/FeT, from
    config.planet.ferric_fraction_initial. The config validator confines
    it to the open interval (0, 1); it is re-checked here because this
    function is also called directly by tests and callers that bypass
    the config layer, and f_0 in {0, 1} silently produces a degenerate
    redox ratio rather than an error."""
    if not 0.0 < f_0 < 1.0:
        raise ValueError(f'initial ferric fraction must lie in (0, 1), got {f_0}')

    M_melt_0 = float(np.sum(phi * mass))
    n_fe2_0 = (W_FET * M_melt_0) / MU_FEO
    n_fe3_0 = n_fe2_0 * f_0 / (1.0 - f_0)
    w_feo15 = n_fe3_0 * MU_FEO15 / M_melt_0 if M_melt_0 > 0.0 else 0.0

    D_fe3_cell = np.where(pres >= P_CUTOFF_GPA * 1e9, D_FE3_BRG, D_FE3_SHALLOW)

    return MeltRedoxState(
        n_fe3_melt=n_fe3_0,
        n_fe2_melt=n_fe2_0,
        phi_prev=phi.copy(),
        D_fe3_cell=D_fe3_cell,
        n_fe_metal_cell=np.zeros_like(phi),
        a_fe_cell=np.zeros_like(phi),
        ferric_frac=f_0,
        redox_ratio=f_0 / (1.0 - f_0),
        w_feo15=w_feo15,
    )


def _warn_clamped_cells(temp: np.ndarray, P_gpa: np.ndarray, usable: np.ndarray) -> None:
    """Warn when usable melt cells lie outside the tabulated Deng EOS grid, so
    their int dV dP is the nearest grid-edge value rather than a computed one.
    In practice this is melt below the table's lowest temperature (1500 K),
    which the validity mask does not exclude."""
    clamped = usable & eos_deng.clamped_mask(temp, P_gpa)
    if not np.any(clamped):
        return
    log.warning(
        'Out of the bounds of the Deng EOS: %d melt cell(s) at T=%.0f-%.0f K, '
        'P=%.1f-%.1f GPa; values of int(dV dP) are clamped to the nearest '
        'available grid value',
        int(np.count_nonzero(clamped)),
        float(np.min(temp[clamped])),
        float(np.max(temp[clamped])),
        float(np.min(P_gpa[clamped])),
        float(np.max(P_gpa[clamped])),
    )


def _metal_saturation_step(
    state: MeltRedoxState,
    temp: np.ndarray,
    pres: np.ndarray,
    phi: np.ndarray,
    mass: np.ndarray,
) -> float:
    """Step 9a-9h: check the melt for Fe-metal saturation and, if it is
    supersaturated, react it to equilibrium. Returns the reaction extent xi
    [mol], which because the Fe stoichiometric coefficient is 1 is also the
    moles of metal formed. Always >= 0: the back-reaction is disabled.

    This is an intrinsic step of the crystallization algorithm, not an
    option: Schaefer et al. (2024) Section 2.7 "add an additional step in
    between each crystallization step to check for metal saturation". The
    melt either is or is not supersaturated, and holding Fe3+/FeT through a
    supersaturation would leave the model metastable.

    Runs between crystallization (Steps 3-9) and the surface fO2 (Step 10),
    matching Schaefer et al. (2024) Section 2.7, who "add an additional step
    in between each crystallization step to check for metal saturation".
    Step 10 must see the corrected redox ratio, hence the ordering.

    Mutates ``state`` in place: n_fe2_melt, n_fe3_melt, n_fe_metal_cell and
    a_fe_cell. Does NOT recompute the derived ratios -- the caller does that
    via _update_ratios, so the update is skipped when nothing happened.
    """
    # Step 9b: melt inventory from the physics, not from a BSE composition.
    # PROTEUS carries an MgSiO3 mantle, so the non-Fe melt species are
    # MgO + SiO2 -- two single-cation oxide units per formula unit, matching
    # the basis Schaefer's nsil is built on (her MW array lists MgO and SiO2
    # separately). Dropping the factor 2 would halve n_sil and inflate a_Fe
    # by ~1.9x, since a_Fe ~ 1/n_sil.
    state.a_fe_max_cell = -1
    M_melt = float(np.sum(phi * mass))
    if M_melt <= 0.0:
        return 0.0
    # Non-iron mass excludes both the ferrous FeO and the ferric FeO1.5 that
    # Step 2 adds on top of it.
    n_other = 2.0 * M_melt * (1.0 - W_FET - state.w_feo15) / MU_MGSIO3
    n2, n3 = state.n_fe2_melt, state.n_fe3_melt
    n_sil = n_other + n2 + n3

    # Step 9d: a_Fe per cell. Only K varies with cell (through T and P); the
    # composition factor is one scalar because Step 4 makes the melt uniform.
    melt = phi > 0.0
    P_gpa = pres / 1e9
    # The pressure term is always included: 2*V(FeO1.5) + V(Fe) < 3*V(FeO) at
    # depth, so int(dV dP) is what makes disproportionation favourable there.
    # Omitting it would leave the metal activity a function of temperature
    # alone and invert the depth trend, with the shallow cool melt saturating
    # before the deep hot melt. Schaefer et al. likewise always pass the full
    # basal pressure through to the equilibrium constant.
    a_fe, eos_valid = dispro.activity_Fe_metal(
        temp, n2, n3, n_sil, P_gpa=P_gpa, use_P_term=True
    )
    a_fe = np.where(melt & eos_valid, a_fe, 0.0)
    state.a_fe_cell = a_fe
    _warn_clamped_cells(temp, P_gpa, melt & eos_valid)

    # Step 9e: the binding cell. Because the melt is homogeneous, only the
    # most supersaturated cell can host equilibrium -- once it reaches
    # a_Fe = 1 every other cell is necessarily below 1. This is the analogue
    # of Schaefer evaluating at the base of the magma ocean.
    usable = melt & eos_valid
    if not np.any(usable):
        # Every melt cell is outside the Deng EOS envelope (T above 5000 K or
        # P above 136 GPa), so the melt cannot be tested this step. Common
        # early in a hot deep magma ocean, which is also when metal is most
        # likely to form, so say it once rather than pass silently.
        if not state.eos_coverage_logged:
            state.eos_coverage_logged = True
            log.warning(
                'No melt cell lies inside the Fe-disproportionation EOS '
                'envelope (T <= %.0f K, P <= %.0f GPa); metal saturation '
                'cannot be evaluated while that holds.',
                eos_deng.T_CEILING,
                eos_deng.P_EXERCISED,
            )
        return 0.0
    cstar = int(np.argmax(np.where(usable, a_fe, -np.inf)))
    state.a_fe_max_cell = cstar

    # Forward reaction only: metal that has formed is never redissolved, so an
    # undersaturated melt is left untouched even if metal is present.
    if a_fe[cstar] < 1.0:
        return 0.0  # undersaturated, no metal forms

    # Step 9f: the only equation solved. One scalar unknown, monotonic in xi,
    # bracketed automatically because F(0) = RHS(0)*(a_Fe - 1).
    # n_metal_avail = 0 disables the back-reaction branch, so xi >= 0.
    K, _ = dispro.K_eq(temp[cstar], P_gpa[cstar], use_P_term=True)
    xi = dispro.solve_extent(float(K), n2, n3, n_sil, n_metal_avail=0.0)
    if xi <= 0.0:
        return 0.0

    # Step 9g: apply. Fe and O conservation are identities here.
    # Metal is deposited where it formed and does not move; the radial
    # field builds up over many steps as cstar migrates with the front.
    state.n_fe2_melt = n2 - 3.0 * xi
    state.n_fe3_melt = n3 + 2.0 * xi
    state.n_fe_metal_cell[cstar] += xi
    return xi


def write_fO2_profile_ncdf(fpath: str, state: MeltRedoxState | None) -> bool:
    """Append the Step 10 fO2 profile to an interior snapshot ``_int.nc``.

    Adds (or overwrites) ``log10_fO2_s`` on the snapshot's ``staggered``
    dimension -- absolute log10(fO2/bar) from Eq 13 at each cell's (T, P),
    NaN in solid cells and outside the Deng EOS envelope -- and the scalar
    ``fO2_top_index``, the staggered index of the uppermost melt cell
    (diagnostic: ``fO2_shift_IW_mantle`` is evaluated at 1 bar). Returns True if written.

    The snapshot is written by the interior backend before the redox step
    runs, so this is called afterwards on the same file. Skips, with a log
    message, when there is no profile yet, no file, or a length mismatch.
    """
    import netCDF4 as nc

    if state is None or state.log10_fO2_cell is None:
        return False
    if not os.path.isfile(fpath):
        log.debug('No interior snapshot at %s; fO2 profile not written', fpath)
        return False
    prof = np.asarray(state.log10_fO2_cell, dtype=float)
    with nc.Dataset(fpath, mode='a') as ds:
        if 'staggered' not in ds.dimensions or len(ds.dimensions['staggered']) != prof.size:
            log.warning(
                'fO2 profile (%d cells) does not match the staggered grid of %s; not written',
                prof.size,
                fpath,
            )
            return False
        _put_fO2_profile(ds, state)
    return True


def write_redox_ncdf(
    fpath: str,
    state: MeltRedoxState | None,
    time: float,
    interior_o: Interior_t,
    hf_row: dict,
) -> bool:
    """Write a standalone per-step redox snapshot ``<time>_redox.nc``.

    For interior backends with no ``_int.nc`` to append to (SPIDER, whose
    per-step output is JSON written by the C binary). Self-contained: the
    fO2 profile is stored with the staggered P, T and phi it was computed
    on, plus the step's scalars. Returns True if written.
    """
    import netCDF4 as nc

    if state is None or state.log10_fO2_cell is None:
        return False
    prof = np.asarray(state.log10_fO2_cell, dtype=float)
    with nc.Dataset(fpath, mode='w') as ds:
        ds.description = 'PROTEUS melt redox snapshot (interior_chem/redox.py)'
        ds.createDimension('staggered', prof.size)

        def _add(name, data, units):
            v = ds.createVariable(name, np.float64, ('staggered',))
            v[:] = np.asarray(data, dtype=float)
            v.units = units

        _add('pres_s', np.asarray(interior_o.pres, dtype=float) / 1e9, 'GPa')
        _add('temp_s', interior_o.temp, 'K')
        _add('phi_s', interior_o.phi, '')
        _put_fO2_profile(ds, state)

        for name, value, units in (
            ('time', time, 'yr'),
            ('fO2_shift_IW_mantle', hf_row['fO2_shift_IW_mantle'], 'log10 units rel. IW'),
            ('ferric_frac_mantle', hf_row['ferric_frac_mantle'], ''),
        ):
            v = ds.createVariable(name, np.float64)
            v.assignValue(float(value))
            v.units = units
    return True


def _put_fO2_profile(ds, state: MeltRedoxState) -> None:
    """Create or overwrite log10_fO2_s and fO2_top_index in an open dataset
    that already has a ``staggered`` dimension of the profile's length."""
    if 'log10_fO2_s' in ds.variables:
        v = ds['log10_fO2_s']
    else:
        v = ds.createVariable('log10_fO2_s', np.float64, ('staggered',))
    v[:] = np.asarray(state.log10_fO2_cell, dtype=float)
    v.units = 'log10(bar)'
    v.long_name = 'melt oxygen fugacity, Hirschmann (2022) Eq 21 / Schaefer et al. (2024) Eq 13'
    v.comment = (
        'Evaluated at each cell T and P with the FeO-FeO1.5 '
        'int(dV dP) (Deng et al. 2020); NaN in solid cells and '
        'outside the EOS envelope'
    )
    if 'fO2_top_index' in ds.variables:
        t = ds['fO2_top_index']
    else:
        t = ds.createVariable('fO2_top_index', np.int32)
    t.assignValue(int(state.fO2_cell))
    t.comment = (
        'staggered index of the uppermost melt cell (diagnostic; '
        'fO2_shift_IW_mantle is evaluated at 1 bar, not here)'
    )


def effective_melt_fraction(phi) -> np.ndarray:
    """Solver melt fraction with cells below PHI_SOLID set to 0 (solid)."""
    phi = np.asarray(phi, dtype=float)
    return np.where(phi < PHI_SOLID, 0.0, phi)


def update_melt_redox(interior_o: Interior_t, hf_row: dict, config: Config) -> None:
    """Advance the melt Fe3+/Fe2+ tracking by one coupling-loop timestep,
    compute the radial Eq 13 fO2 profile (diagnostic), and write the
    surface (1 bar) Delta-IW into hf_row.

    No-op unless config.planet.fO2_source == 'from_mantle_redox'. Call
    once per run_interior invocation, after interior_o.phi/mass/pres/temp
    are finalised for this step.

    Writes:
        hf_row['fO2_shift_IW_mantle'] - surface (1 bar) Delta-IW this step
        hf_row['ferric_frac_mantle']  - global melt Fe3+/FeT this step
    """
    if config.planet.fO2_source != 'from_mantle_redox':
        return

    # Effective melt fraction: cells with phi < PHI_SOLID are solid (phi = 0)
    # for every step below. Applied once here so all steps agree.
    phi = effective_melt_fraction(interior_o.phi)
    mass = np.asarray(interior_o.mass, dtype=float)
    pres = np.asarray(interior_o.pres, dtype=float)

    first_call = interior_o.redox_state is None

    if first_call:
        f_0 = float(config.planet.ferric_fraction_initial)
        interior_o.redox_state = _init_state(phi, mass, pres, f_0)
        log.info(
            'Melt redox tracking initialised: n_FeT_melt=%.3e mol, f_0=%.3f',
            interior_o.redox_state.n_fe3_melt + interior_o.redox_state.n_fe2_melt,
            f_0,
        )
    else:
        state = interior_o.redox_state

        if not state.melt_exhausted:
            # Step 3 (Eq 4-6): new solid mass from the decrease in melt
            # fraction since the previous call (mass is time-invariant per
            # cell on the _s grid, so 'mass' here serves for both steps)
            delta_phi = state.phi_prev - phi
            new_solid_frac = np.clip(delta_phi, 0.0, None)
            new_solid_mass = mass * new_solid_frac

            # Step 4 (Eq 9-11): redistribute the *previous* step's global
            # reservoirs across cells in proportion to local melt mass
            M_melt_cell = state.phi_prev * mass
            M_melt_all = float(np.sum(M_melt_cell))

            if M_melt_all > 0:
                safe_M = np.where(M_melt_cell > 0, M_melt_cell, 1.0)
                n_fe3_melt_cell = np.where(
                    M_melt_cell > 0, (M_melt_cell / M_melt_all) * state.n_fe3_melt, 0.0
                )
                n_fe2_melt_cell = np.where(
                    M_melt_cell > 0, (M_melt_cell / M_melt_all) * state.n_fe2_melt, 0.0
                )
                C_fe3 = n_fe3_melt_cell / safe_M
                C_fe2 = n_fe2_melt_cell / safe_M

                # Step 5-6 (Eq 12-17): Fe3+/Fe2+ moles entering the new solid
                delta_n_fe3 = state.D_fe3_cell * C_fe3 * new_solid_mass
                delta_n_fe2 = D_FE2_BRG * C_fe2 * new_solid_mass

                # Step 7 (Eq 18-19): update the global reservoirs
                state.n_fe3_melt -= float(np.sum(delta_n_fe3))
                state.n_fe2_melt -= float(np.sum(delta_n_fe2))

                # Step 8-9 (Eq 20-24): ferric fraction and redox ratio
                _update_ratios(state)
            else:
                state.melt_exhausted = True
                log.info(
                    'Melt redox tracking: mantle fully solidified, '
                    'freezing ferric_frac=%.6f for remaining steps',
                    state.ferric_frac,
                )

        state.phi_prev = phi.copy()

    state = interior_o.redox_state

    # Step 9a-9h: Fe metal saturation, run between every pair of
    # crystallization steps. Not optional: the melt either is or is not
    # supersaturated, and carrying a supersaturation would leave the model
    # metastable. Skipped on the first call, where no crystallization has
    # happened yet, matching Schaefer et al., who begin the check only after
    # the first solid layer forms.
    xi = 0.0
    if not first_call and not state.melt_exhausted:
        temp = np.asarray(interior_o.temp, dtype=float)
        xi = _metal_saturation_step(state, temp, pres, phi, mass)
        if xi != 0.0:
            _update_ratios(state)

    # Step 10a: radial fO2 profile from Eq 13 at each cell's (T, P),
    # including int(dV dP). Diagnostic only (written to the interior
    # snapshot, cf. Schaefer et al. 2024 Fig S4); it does not set Delta-IW.
    temp = np.asarray(interior_o.temp, dtype=float)
    P_gpa = pres / 1e9
    melt = phi > 0.0
    prof, _ = _log10_fO2_profile(state.redox_ratio, temp, P_gpa, state.X)
    state.log10_fO2_cell = np.where(melt, prof, np.nan)
    candidates = melt if np.any(melt) else np.ones_like(melt, dtype=bool)
    state.fO2_cell = int(np.argmin(np.where(candidates, pres, np.inf)))

    # Step 10b: surface Delta-IW handed to the outgassing, as Schaefer et al.
    # (2024) compute the fO2 of the outgassing atmosphere (fO2lowP_H22.m):
    # Eq 13 at the surface, 1 bar, int(dV dP) = 0, with the single melt
    # Fe3+/Fe2+ (homogeneous by Step 4). The depth physics reaches the
    # surface only through that ratio (crystallization and the Step 9
    # metal step, which do carry the pressure term). Evaluating a cell at
    # depth instead would add its pressure term against a 1-bar buffer,
    # tens of log units once the shallow melt has frozen.
    # Both Eq 13 and the buffer are taken at the temperature the outgassing
    # solves at: CALLIOPE and atmodeller raise T_magma to outgas.T_floor,
    # and below the floor the relation gives offsets of -10 or lower that
    # the chemistry cannot solve.
    T_floor = float(config.outgas.T_floor)
    T_magma = float(hf_row['T_magma'])
    T_out = max(T_magma, T_floor)
    if T_out > T_magma:
        log.warning(
            'Melt redox: T_magma = %.1f K is below outgas.T_floor = %.1f K; '
            'surface fO2 and Delta-IW are evaluated at %.1f K, the temperature '
            'the outgassing uses',
            T_magma,
            T_floor,
            T_out,
        )
    log10_fO2_surf = _log10_fO2_surface(state.redox_ratio, T_out, state.X)
    dIW = log10_fO2_surf - _iw_buffer_bower2022(T_out)

    hf_row['fO2_shift_IW_mantle'] = dIW
    hf_row['ferric_frac_mantle'] = state.ferric_frac

    # Per-step summary at INFO, so the metal state is visible without DEBUG
    n_metal_total = float(np.sum(state.n_fe_metal_cell))
    if first_call:
        metal_msg = 'not checked (first step)'
    elif state.melt_exhausted:
        metal_msg = 'not checked (mantle solidified)'
    elif xi > 0.0:
        metal_msg = 'metal formed (%.3e mol this step)' % xi
    else:
        metal_msg = 'no metal formed'
    log.info(
        'Metal redox state: %s; cumulative metal=%.3e mol, '
        'Fe3+/FeT=%.4f, surface dIW=%+.3f (1 bar, T=%.0f K)',
        metal_msg,
        n_metal_total,
        state.ferric_frac,
        dIW,
        T_out,
    )

    # Step 11: metal-saturation diagnostics. Written unconditionally so the
    # columns exist whether or not the check ran; a_Fe is reported even when
    # below 1, because how close the melt runs to the buffer is the useful
    # diagnostic during an f_0 scan.
    hf_row['a_fe_max_mantle'] = float(np.max(state.a_fe_cell))
    # Where that maximum sits: the binding cell, which is also where any
    # metal formed this step was deposited. -1 on steps with no check.
    checked = not first_call and not state.melt_exhausted
    hf_row['a_fe_max_cell_mantle'] = float(state.a_fe_max_cell if checked else -1)
    hf_row['n_fe_metal_mantle'] = float(np.sum(state.n_fe_metal_cell))
    # Same cumulative metal as a mass: moles times the molar mass of Fe.
    hf_row['fe_metal_kg_mantle'] = hf_row['n_fe_metal_mantle'] * element_mmw['Fe']
    # Reaction extent this step: moles of metal formed (>= 0; no
    # redissolution).
    hf_row['n_fe_metal_step_mantle'] = xi
