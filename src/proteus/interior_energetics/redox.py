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
  Step 10   surface fO2 via Hirschmann (2022) GCA 313 Eq 21 (= Schaefer
            et al. 2024 Eq 13), with the pressure/EOS term (integral of
            Delta V dP) fixed at zero -- matching every place Schaefer's
            own reference code evaluates this equation, and consistent
            with the rest of this model using prescribed partition
            coefficients rather than an equilibrium constant. Delta-IW
            uses the O'Neill & Eggins (2002) buffer as given in Bower
            et al. (2022) PSJ 3, 93, Eq 7-8 (T-only, no pressure term).

BSE starting composition and partition coefficients are the same
literature values used in tools/redox_step.py (McDonough 2003 via
Hirschmann 2022 for the oxide wt%; Schaefer et al. 2024 Table 3 / Eq 6
for the partition coefficients); they are module constants here rather
than config fields, matching the "physics defaults, not user knobs"
treatment of comparable fixed stoichiometry elsewhere in the outgas
package (e.g. proteus.outgas.dummy._ELEMENT_TO_SPECIES).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from proteus.config import Config
    from proteus.interior_energetics.common import Interior_t

log = logging.getLogger('fwl.' + __name__)

# ── Step 0: fixed input parameters (McDonough 2003 BSE via Hirschmann 2022;
# Schaefer et al. 2024 Table 3 / Eq 6) ──────────────────────────────────────
W_FET  = 0.08       # bulk iron mass fraction in the melt (dimensionless)
MU_FEO = 0.07184    # molar mass of FeO in kg/mol (= 71.84 g/mol)
F_0    = 0.10       # initial ferric fraction Fe3+/FeT

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


@dataclass
class MeltRedoxState:
    """Persistent Fe3+/Fe2+ tracking state, one instance per run (held at
    Interior_t.redox_state), carried across coupling-loop timesteps."""
    n_fe3_melt: float
    n_fe2_melt: float
    phi_prev: np.ndarray
    D_fe3_cell: np.ndarray
    ferric_frac: float = F_0
    redox_ratio: float = field(default=F_0 / (1.0 - F_0))
    melt_exhausted: bool = False
    X: dict = field(default_factory=_compute_mole_fractions)


def _init_state(phi: np.ndarray, mass: np.ndarray, pres: np.ndarray) -> MeltRedoxState:
    """Step 1-2 (Eq 1-3): seed the global Fe3+/Fe2+ reservoirs from the
    interior state at the first call, and the per-cell bridgmanite/Cpx-Opx
    split from the (time-invariant) pressure profile."""
    M_melt_0 = float(np.sum(phi * mass))
    n_FeT_0 = (W_FET * M_melt_0) / MU_FEO

    D_fe3_cell = np.where(pres >= P_CUTOFF_GPA * 1e9, D_FE3_BRG, D_FE3_SHALLOW)

    return MeltRedoxState(
        n_fe3_melt=F_0 * n_FeT_0,
        n_fe2_melt=(1.0 - F_0) * n_FeT_0,
        phi_prev=phi.copy(),
        D_fe3_cell=D_fe3_cell,
    )


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

    if interior_o.redox_state is None:
        interior_o.redox_state = _init_state(phi, mass, pres)
        log.info(
            'Melt redox tracking initialised: n_FeT_melt=%.3e mol, f_0=%.3f',
            interior_o.redox_state.n_fe3_melt + interior_o.redox_state.n_fe2_melt,
            F_0,
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
                n_FeT_melt = state.n_fe3_melt + state.n_fe2_melt
                if n_FeT_melt > 0:
                    state.ferric_frac = state.n_fe3_melt / n_FeT_melt
                    state.redox_ratio = state.ferric_frac / (1.0 - state.ferric_frac)
            else:
                state.melt_exhausted = True
                log.info(
                    'Melt redox tracking: mantle fully solidified, '
                    'freezing ferric_frac=%.6f for remaining steps',
                    state.ferric_frac,
                )

        state.phi_prev = phi.copy()

    state = interior_o.redox_state

    # Step 10: surface fO2 and Delta-IW, evaluated at the true surface T
    # (T_magma, the same temperature the outgas dispatch uses)
    T_surf = float(hf_row['T_magma'])
    log10_fO2_surf = _log10_fO2_surface(state.redox_ratio, T_surf, state.X)
    dIW = log10_fO2_surf - _iw_buffer_bower2022(T_surf)

    hf_row['fO2_shift_IW_mantle'] = dIW
    hf_row['ferric_frac_mantle'] = state.ferric_frac
