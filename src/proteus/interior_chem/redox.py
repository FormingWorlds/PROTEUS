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

  Step 1-2  initial melt iron inventory and Fe3+/Fe2+ split, from f_0
  Step 3    new solid mass each step, from the decrease in local melt
            fraction (delta_phi), using the mass-invariant _s grid
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
            the melt stays uniform; xi < 0 redissolves it. Because metal
            removes Fe but leaves the O behind, saturation drives Fe3+/FeT
            *up*, and the melt is buffered at the saturation value regardless
            of f_0. The equilibrium constant always carries int(dV dP) from
            the Deng et al. (2020) EOS (interior_chem/eos_deng.py): at depth
            2*V(FeO1.5) + V(Fe) < 3*V(FeO), and that volume change is what
            makes disproportionation favourable there. Cells outside the EOS
            envelope (T > 4175 K or P > 136 GPa) are excluded from the
            check. See interior_chem/disproportionation.py for the
            derivation.
  Step 10   surface fO2 via Hirschmann (2022) GCA 313 Eq 21 (= Schaefer
            et al. 2024 Eq 13), with the pressure/EOS term (integral of
            Delta V dP) fixed at zero. This is exact rather than an
            approximation: the equation is evaluated at the surface, 1 bar.
            Schaefer does the same in fO2lowP_H22.m ("without the high
            pressure term", P = 0.0001 GPa) while always passing the full
            basal pressure to the disproportionation, so including the term
            in Step 9c and omitting it here is her structure, not a
            shortcut. Reads the redox ratio AFTER Step 9a-9h, so any metal
            that formed is reflected in the outgassing fO2. Delta-IW
            uses the O'Neill & Eggins (2002) buffer as given in Bower
            et al. (2022) PSJ 3, 93, Eq 7-8 (T-only, no pressure term).

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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from proteus.interior_chem import disproportionation as dispro
from proteus.interior_chem import eos_deng

if TYPE_CHECKING:
    from proteus.config import Config
    from proteus.interior_energetics.common import Interior_t

log = logging.getLogger('fwl.' + __name__)

# ── Step 0: fixed input parameters (McDonough 2003 BSE via Hirschmann 2022;
# Schaefer et al. 2024 Table 3 / Eq 6) ──────────────────────────────────────
# FeO (not Fe) mass fraction of the melt: it is divided by the molar mass of
# FeO below, so it must be the oxide fraction. BSE is 7.82 wt% FeO against
# 6.08 wt% Fe, and 0.08 matches the former. This matters beyond naming --
# (1 - W_FET) is the non-FeO mass that becomes MgSiO3 in _metal_saturation_step.
W_FET  = 0.08       # FeO mass fraction in the melt (dimensionless)
MU_FEO = 0.07184    # molar mass of FeO in kg/mol (= 71.84 g/mol)
MU_MGSIO3 = 0.100389  # molar mass of MgSiO3 in kg/mol (= 100.389 g/mol)
# Default initial ferric fraction Fe3+/FeT. The value actually used is
# config.planet.ferric_fraction_initial, which defaults to this; the
# constant is kept so the dataclass has a sane placeholder before
# _init_state overwrites it, and to document the published default.
F_0    = 0.10       # initial ferric fraction Fe3+/FeT (Schaefer et al. 2024)

D_FE2_BRG = 0.85    # bridgmanite/melt partition coefficient for Fe2+ (both regimes)

P_CUTOFF_GPA = 22.0        # halt bridgmanite assemblage above this pressure
D_FE3_BRG    = 0.75        # bridgmanite/melt (Table 3)
D_FE3_CPX    = 0.45        # clinopyroxene/melt (Table 3, Mallmann & O'Neill 2009)
D_FE3_OPX    = 0.70 * D_FE3_CPX      # orthopyroxene/melt = D_opx/cpx x D_cpx/melt (Eq 6)
D_FE3_SHALLOW = (D_FE3_CPX + D_FE3_OPX) / 2   # plain mean, no modal Cpx/Opx given

# BSE starting composition (wt%)
_WT_OXIDES = {
    'FeO': 7.82, 'MgO': 38.3, 'SiO2': 45.5, 'CaO': 3.58, 'Al2O3': 4.49,
    'FeO15': 0.36, 'Na2O': 0.0, 'K2O': 0.0, 'TiO2': 0.0, 'P2O5': 0.0,
}

# Hirschmann (2022) Eq 21 fixed coefficients
_R       = 8.31447
_A       = 0.19317
_B       = -4.51412 / 2.303
_C_PARAM = 9574.293 / 2.303
_DELTA_CP = 33.25
_T0      = 1673.15
_YS = [y / 2.303 for y in
       (-1198.4, -426.82, 1138.371, 4232.933, 6650.972, 7998.434,
        -10298.6, -2866.92, -2663.74)]


def _compute_mole_fractions():
    """Single-cation-basis oxide mole fractions for Hirschmann (2022) Eq 21.

    Na2O, K2O, Al2O3 and P2O5 have 2 cations per formula unit and are
    expressed per single cation, matching Hirschmann's point (2)
    modification to Eq 20.
    """
    M = {'SiO2': 60.08, 'TiO2': 79.87, 'MgO': 40.30, 'CaO': 56.08,
         'Na2O': 61.98, 'K2O': 94.20, 'P2O5': 141.94, 'Al2O3': 101.96,
         'FeO': 71.84, 'FeO15': 79.84}
    cations_per_unit = {'SiO2': 1, 'TiO2': 1, 'MgO': 1, 'CaO': 1,
                        'Na2O': 2, 'K2O': 2, 'P2O5': 2, 'Al2O3': 2,
                        'FeO': 1, 'FeO15': 1}
    n_cation = {k: cations_per_unit[k] * _WT_OXIDES[k] / M[k] for k in M}
    n_total = sum(n_cation.values())
    return {k: v / n_total for k, v in n_cation.items()}


def _log10_fO2_surface(redox_ratio: float, T: float, X: dict) -> float:
    """log10(fO2), Hirschmann (2022) Eq 21, Delta V dP = 0 (see module
    docstring). T-only: no radial/pressure dependence."""
    Xi = [X['SiO2'], X['TiO2'], X['MgO'], X['CaO'],
          X['Na2O'], X['K2O'], X['P2O5'], X['Al2O3']]
    loggammas_const = (
        _YS[0] * Xi[0] + _YS[1] * Xi[1] + _YS[2] * Xi[2] + _YS[3] * Xi[3]
        + _YS[4] * Xi[4] + _YS[5] * Xi[5] + _YS[6] * Xi[6]
        + _YS[7] * Xi[7] * Xi[0] + _YS[8] * Xi[0] * Xi[2]
    )
    logXFe3Fe2 = math.log10(redox_ratio)
    dG_RT = _B + _C_PARAM / T - (_DELTA_CP / _R / math.log(10)) * (
        1 - _T0 / T - math.log(T / _T0)
    )
    loggammas = loggammas_const / T
    return (logXFe3Fe2 - dG_RT - loggammas) / _A


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
    ferric_frac: float = F_0
    redox_ratio: float = field(default=F_0 / (1.0 - F_0))
    melt_exhausted: bool = False
    eos_coverage_logged: bool = False
    X: dict = field(default_factory=_compute_mole_fractions)


def _init_state(
    phi: np.ndarray, mass: np.ndarray, pres: np.ndarray, f_0: float
) -> MeltRedoxState:
    """Step 1-2 (Eq 1-3): seed the global Fe3+/Fe2+ reservoirs from the
    interior state at the first call, and the per-cell bridgmanite/Cpx-Opx
    split from the (time-invariant) pressure profile.

    ``f_0`` is the initial ferric fraction Fe3+/FeT, from
    config.planet.ferric_fraction_initial. The config validator confines
    it to the open interval (0, 1); it is re-checked here because this
    function is also called directly by tests and callers that bypass
    the config layer, and f_0 in {0, 1} silently produces a degenerate
    redox ratio rather than an error."""
    if not 0.0 < f_0 < 1.0:
        raise ValueError(
            f'initial ferric fraction must lie in (0, 1), got {f_0}'
        )

    M_melt_0 = float(np.sum(phi * mass))
    n_FeT_0 = (W_FET * M_melt_0) / MU_FEO

    D_fe3_cell = np.where(pres >= P_CUTOFF_GPA * 1e9, D_FE3_BRG, D_FE3_SHALLOW)

    return MeltRedoxState(
        n_fe3_melt=f_0 * n_FeT_0,
        n_fe2_melt=(1.0 - f_0) * n_FeT_0,
        phi_prev=phi.copy(),
        D_fe3_cell=D_fe3_cell,
        n_fe_metal_cell=np.zeros_like(phi),
        a_fe_cell=np.zeros_like(phi),
        ferric_frac=f_0,
        redox_ratio=f_0 / (1.0 - f_0),
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
    moles of metal formed (negative = metal redissolved).

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
    M_melt = float(np.sum(phi * mass))
    if M_melt <= 0.0:
        return 0.0
    n_other = 2.0 * M_melt * (1.0 - W_FET) / MU_MGSIO3
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

    # Step 9e: the binding cell. Because the melt is homogeneous, only the
    # most supersaturated cell can host equilibrium -- once it reaches
    # a_Fe = 1 every other cell is necessarily below 1. This is the analogue
    # of Schaefer evaluating at the base of the magma ocean.
    usable = melt & eos_valid
    if not np.any(usable):
        # Every melt cell is outside the Deng EOS envelope (T above 4175 K or
        # P above 136 GPa), so the melt cannot be tested this step. Common
        # early in a hot deep magma ocean, which is also when metal is most
        # likely to form, so say it once rather than pass silently.
        if not state.eos_coverage_logged:
            state.eos_coverage_logged = True
            log.warning(
                'No melt cell lies inside the Fe-disproportionation EOS '
                'envelope (T <= %.0f K, P <= %.0f GPa); metal saturation '
                'cannot be evaluated while that holds.',
                eos_deng.T_CEILING, eos_deng.P_EXERCISED,
            )
        return 0.0
    cstar = int(np.argmax(np.where(usable, a_fe, -np.inf)))

    n_metal_avail = float(np.sum(state.n_fe_metal_cell))
    if a_fe[cstar] < 1.0 and n_metal_avail <= 0.0:
        return 0.0                      # undersaturated, nothing to dissolve

    # Step 9f: the only equation solved. One scalar unknown, monotonic in xi,
    # bracketed automatically because F(0) = RHS(0)*(a_Fe - 1).
    K, _ = dispro.K_eq(temp[cstar], P_gpa[cstar], use_P_term=True)
    xi = dispro.solve_extent(float(K), n2, n3, n_sil, n_metal_avail)
    if xi == 0.0:
        return 0.0

    # Step 9g: apply. Fe and O conservation are identities here.
    state.n_fe2_melt = n2 - 3.0 * xi
    state.n_fe3_melt = n3 + 2.0 * xi
    if xi > 0.0:
        # Metal is deposited where it formed and does not move; the radial
        # field builds up over many steps as cstar migrates with the front.
        state.n_fe_metal_cell[cstar] += xi
    else:
        # Back-reaction: dissolve proportionally from wherever metal sits.
        total = float(np.sum(state.n_fe_metal_cell))
        if total > 0.0:
            state.n_fe_metal_cell += xi * (state.n_fe_metal_cell / total)
            np.clip(state.n_fe_metal_cell, 0.0, None, out=state.n_fe_metal_cell)
    return xi


def update_melt_redox(interior_o: Interior_t, hf_row: dict, config: Config) -> None:
    """Advance the melt Fe3+/Fe2+ tracking by one coupling-loop timestep and
    write the resulting surface Delta-IW into hf_row.

    No-op unless config.planet.fO2_source == 'from_mantle_redox'. Call
    once per run_interior invocation, after interior_o.phi/mass/pres and
    hf_row['T_magma'] are finalised for this step (any per-step clamping
    already applied), so the outgas dispatch that runs afterwards sees a
    value consistent with the T_magma it also uses.

    Writes:
        hf_row['fO2_shift_IW_mantle'] - surface Delta-IW this step
        hf_row['ferric_frac_mantle']  - global melt Fe3+/FeT this step
    """
    if config.planet.fO2_source != 'from_mantle_redox':
        return

    phi  = np.asarray(interior_o.phi, dtype=float)
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
            log.debug(
                'Metal saturation: xi=%.4e mol, total metal=%.4e mol, f=%.6f',
                xi, float(np.sum(state.n_fe_metal_cell)), state.ferric_frac,
            )

    # Step 10: surface fO2 and Delta-IW, evaluated at the true surface T
    # (T_magma, the same temperature the outgas dispatch uses)
    T_surf = float(hf_row['T_magma'])
    log10_fO2_surf = _log10_fO2_surface(state.redox_ratio, T_surf, state.X)
    dIW = log10_fO2_surf - _iw_buffer_bower2022(T_surf)

    hf_row['fO2_shift_IW_mantle'] = dIW
    hf_row['ferric_frac_mantle'] = state.ferric_frac

    # Step 11: metal-saturation diagnostics. Written unconditionally so the
    # columns exist whether or not the check ran; a_Fe is reported even when
    # below 1, because how close the melt runs to the buffer is the useful
    # diagnostic during an f_0 scan.
    hf_row['a_fe_max_mantle'] = float(np.max(state.a_fe_cell))
    hf_row['n_fe_metal_mantle'] = float(np.sum(state.n_fe_metal_cell))
    # Reaction extent this step: positive where metal exsolved, negative
    # where it redissolved.
    hf_row['n_fe_metal_step_mantle'] = xi
