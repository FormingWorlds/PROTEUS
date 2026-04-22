"""
Mantle partitioning hooks for the unified redox module (#57).

This module hosts three live hooks for follow-on physics, each
shipped as a documented stub:

1. :func:`advance_fe_reservoirs` — stateful Fe³⁺/Fe²⁺/Fe⁰ evolution
   (issue #653, Mariana). Stub-bulk implementation: treats the whole
   mantle as one cell and evaluates the configured oxybarometer at
   the surface; returns zero per-cell solid uptake and zero
   Fe⁰-to-core flux. Full implementation replaces the body with the
   10-step Mariana algorithm (remelt-symmetric per plan v6 §3.9).

2. :func:`mineral_melt_KD_fe` — Schaefer+24 Table 3 ferric-iron
   mineral/melt partition coefficients (part of #653). Stub returns
   (1.0, 1.0) for every mineral.

3. :func:`metal_silicate_KD` — metal-silicate partitioning of
   H/O/Si/C/S/N between silicate melt and core metal during core
   formation (issue #526). Stub returns 0.0 everywhere.

4. :func:`eos_density_correction` — EOS corrections for Fe³⁺-rich
   mantle and light-element core (issue #432). Stub returns identity
   factors. No consumer in PROTEUS today; the stub locks the API for
   when #432 wires SPIDER / Aragog / Zalmoxis EOS-lookup paths.

See plan v6 §6.1-§6.3 for full handover checklists.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

log = logging.getLogger('fwl.' + __name__)

# Clamp to avoid division-by-zero in the remelt branch for cells that
# are about to fully remelt (plan v6 §3.9; numerics-review F5 fix).
_EPS_SOLID = 1e-6


# ==================================================================
# Issue #653 — Mariana depth-resolved Fe³⁺/Fe²⁺ evolution
# ==================================================================


@dataclass
class Fe3EvolutionResult:
    """
    Output of :func:`advance_fe_reservoirs`. See plan v6 §6.1 for the
    full contract; this dataclass is the stable interface Mariana must
    satisfy when she replaces the function body.
    """
    n_Fe3_melt: float                      # mol, updated global
    n_Fe2_melt: float                      # mol, updated global
    n_Fe3_solid_cell: np.ndarray           # mol per cell
    n_Fe2_solid_cell: np.ndarray           # mol per cell
    Fe3_frac_bulk_melt: float              # f^(t) = n_Fe3_melt / (n_Fe3_melt + n_Fe2_melt)
    log10_fO2_surface: float               # warm start for redox.solver
    log10_fO2_profile: np.ndarray | None = None
    dm_Fe0_to_core: float = 0.0            # kg this step; Schaefer §2.7
    dn_Fe3_solid_cell: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dn_Fe2_solid_cell: np.ndarray = field(default_factory=lambda: np.zeros(0))


def advance_fe_reservoirs(
    *,
    pressure_profile: np.ndarray,
    temperature_profile: np.ndarray,
    melt_fraction_profile: np.ndarray,
    melt_fraction_profile_prev: np.ndarray,
    cell_mass_profile: np.ndarray,
    n_Fe3_melt_prev: float,
    n_Fe2_melt_prev: float,
    n_Fe3_solid_cell_prev: np.ndarray,
    n_Fe2_solid_cell_prev: np.ndarray,
    mantle_comp,
    oxybarometer: str = 'schaefer2024',
    kd_provider: Callable = None,
    mineral_of: Callable = None,
    phi_crit: float = 0.4,
) -> Fe3EvolutionResult:
    """
    Advance melt Fe³⁺/Fe²⁺ reservoirs by one step.

    The **stub-bulk** implementation below:
      * Treats the mantle as one cell.
      * Makes no crystallisation uptake and no remelt release (no K_D
        applied). This is the null step that preserves existing Fe
        balance while the redox machinery ships.
      * Evaluates the configured oxybarometer (default
        `schaefer2024`) at the surface (P[0], T[0]) using the bulk
        mantle composition + previous-step Fe split. Returns the
        result as the warm-start suggestion for the outer solver.
      * Returns zero Fe⁰-to-core flux (no metal-saturation check).

    Plan v6 §6.1.6 defines the 10-item handover checklist for Mariana.
    Key rules:
      1. Do not change this signature or `Fe3EvolutionResult`.
      2. Implement both Δφ > 0 (freeze, plan §3.9) and Δφ < 0
         (thaw) branches with the `_EPS_SOLID` clamp.
      3. Set Schaefer §2.7 metal saturation only when MO exists
         (φ_max > φ_crit).
      4. Flip `IMPLEMENTATION_STATUS['fe3_evolution']` to
         `'mariana-schaefer2024'`.

    Parameters
    ----------
    pressure_profile : np.ndarray, shape (N,)
        Per-cell pressure in Pa (from Aragog mesh).
    temperature_profile : np.ndarray, shape (N,)
        Per-cell temperature in K.
    melt_fraction_profile : np.ndarray, shape (N,)
        Current φ_i.
    melt_fraction_profile_prev : np.ndarray, shape (N,)
        Previous step's φ_i. Used by Mariana to compute Δφ_i.
    cell_mass_profile : np.ndarray, shape (N,)
        M_i in kg.
    n_Fe3_melt_prev, n_Fe2_melt_prev : float
        Previous-step global melt reservoirs (mol).
    n_Fe3_solid_cell_prev, n_Fe2_solid_cell_prev : np.ndarray, shape (N,)
        Previous-step per-cell solid accumulations (mol).
    mantle_comp :
        config.interior_struct.mantle_comp (MantleComp dataclass).
    oxybarometer : str
        Key into `redox.buffers.OXYBAROMETERS`.
    kd_provider : Callable
        Returns (D_Fe3, D_Fe2) for (mineral, P, T). Defaults to
        `mineral_melt_KD_fe` below.
    mineral_of : Callable
        Returns mineral name for (P, T, mantle_comp). Defaults to
        `liquidus_mineral_at_node`.
    phi_crit : float
        Threshold for MO-active regime.
    """
    # Stub: identity evolution (no freeze, no thaw, no metal drain).
    n_nodes = len(melt_fraction_profile)
    n_Fe3_solid_cell = np.array(n_Fe3_solid_cell_prev, copy=True)
    n_Fe2_solid_cell = np.array(n_Fe2_solid_cell_prev, copy=True)

    total_fe = n_Fe3_melt_prev + n_Fe2_melt_prev
    fe3_frac = n_Fe3_melt_prev / total_fe if total_fe > 0 else 0.0

    # Warm-start fO2 via the configured oxybarometer. The MO-active /
    # -inactive dispatch is handled by the caller (coupling.py) via
    # `log10_fO2_mantle` in buffers.py; here we only report a naive
    # warm start evaluated at surface conditions using current
    # mantle composition. Commit C's coupling layer replaces this
    # call with the proper dispatcher.
    from proteus.redox.buffers import log10_fO2_mantle

    # Aragog staggered arrays are ordered CMB → surface; the surface
    # is at index -1, not 0 (round-6 review N-2 fix).
    T_surf = float(temperature_profile[-1]) if n_nodes > 0 else 0.0
    P_surf = float(pressure_profile[-1]) if n_nodes > 0 else 0.0
    phi_max = (
        float(np.max(melt_fraction_profile)) if n_nodes > 0 else 0.0
    )
    # Stub cannot resolve a Schaefer Eq 13 warm-start when the mantle
    # is fully reduced (no Fe³⁺ in melt). Rather than silently
    # oxidising to Fe3_frac=0.02 (which happens to be Earth BSE and
    # would import ~5 log units of spurious fO2), we return
    # NaN — downstream callers should treat NaN as "warm start
    # undefined, solver must widen bracket" in Commit C.
    if fe3_frac <= 0:
        log.warning(
            'advance_fe_reservoirs stub: Fe3_frac=%.3e; warm-start fO2 '
            'is undefined for fully reduced mantle. Returning NaN; '
            'caller (redox solver) should use previous ΔIW or widen '
            'bracket.', fe3_frac,
        )
        log10_fO2_surf = float('nan')
    else:
        try:
            log10_fO2_surf = log10_fO2_mantle(
                Fe3_frac=fe3_frac,
                temperature=T_surf,
                pressure=P_surf,
                phi_max=phi_max,
                mantle_comp=mantle_comp,
                oxybarometer=oxybarometer,
                phi_crit=phi_crit,
            )
        except (NotImplementedError, ValueError) as exc:
            log.warning(
                'advance_fe_reservoirs stub: log10_fO2_mantle failed '
                '(%s); returning NaN.', exc,
            )
            log10_fO2_surf = float('nan')

    return Fe3EvolutionResult(
        n_Fe3_melt=n_Fe3_melt_prev,
        n_Fe2_melt=n_Fe2_melt_prev,
        n_Fe3_solid_cell=n_Fe3_solid_cell,
        n_Fe2_solid_cell=n_Fe2_solid_cell,
        Fe3_frac_bulk_melt=fe3_frac,
        log10_fO2_surface=log10_fO2_surf,
        log10_fO2_profile=None,
        dm_Fe0_to_core=0.0,
        dn_Fe3_solid_cell=np.zeros(n_nodes),
        dn_Fe2_solid_cell=np.zeros(n_nodes),
    )


def mineral_melt_KD_fe(
    mineral: str,
    *,
    pressure: float,
    temperature: float,
    C_Fe2O3_melt_wt: float = 0.0,
) -> tuple[float, float]:
    """
    Mineral/melt partition coefficients (D_Fe3+, D_Fe2+).

    Stub implementation returns **(1.0, 1.0)** for every mineral —
    i.e. no net partitioning. This is the identity case that makes
    `advance_fe_reservoirs` stub-bulk reproduce pre-#57 Fe balance.

    Full implementation (issue #653, Schaefer+24 Table 3):

      =========  ============================================
      mineral    D_Fe3+^{mineral/melt}
      =========  ============================================
      sp         exp(0.87·10000/T − 4.6 + 0.24·ln(C_Fe2O3_wt))
      ol         0.0       (Mallmann & O'Neill 2009)
      cpx        0.45 ± 0.20 (Mallmann & O'Neill 2009)
      opx        0.70 · D_{Fe3+}^{cpx/melt}
      gt         1.40 ± 0.70 (Holycross & Cottrell 2023)
      wad        1.18 ± 0.58 (O'Neill+93 + Armstrong EOS)
      maj        1.59 ± 0.77
      ring       1.60 ± 1.03
      bg         0.75 ± 0.65 (Boujibar+16, Huang+23, Kuwahara+Nakada 23)
      mw         0.0
      =========  ============================================

    D_Fe2+ values from Elkins-Tanton 2008 Table S2.
    """
    # Identity.
    return (1.0, 1.0)


def liquidus_mineral_at_node(
    *,
    pressure: float,
    temperature: float,
    mantle_comp=None,
) -> str:
    """
    Liquidus mineral at (P, T) in the fractional-crystallisation
    sequence.

    Stub returns 'bg' (bridgmanite) regardless of (P, T), matching
    Mariana's minimal single-mineral algorithm.

    Full implementation (Elkins-Tanton 2008 / Schaefer+24 §2.3)
    routes:
      * P < 2 GPa             → ol + opx + cpx + sp  (returns 'opx')
      * 2 ≤ P < 13 GPa        → ol + opx + cpx + gt
      * 13 ≤ P < 23 GPa       → wad
      * 23 ≤ P < 27 GPa       → ring
      * P ≥ 27 GPa            → bg + mw
    """
    return 'bg'


# ==================================================================
# Issue #526 — metal-silicate partitioning (core-forming equilibrium)
# ==================================================================


@dataclass
class MetalSilicatePartitionResult:
    """Return of :func:`apply_metal_silicate_partitioning`."""
    fluxes: dict          # element → kg mantle→core this step (+ = into core)
    dR_budget_mantle: float
    dR_budget_core: float


def metal_silicate_KD(
    element: str,
    *,
    pressure: float,
    temperature: float,
    fO2_dIW: float,
    mantle_comp=None,
    core_comp=None,
) -> float:
    """
    Metal/silicate partition coefficient D_metal/silicate.

    Stub returns **0.0** for every element. No partitioning into
    the core.

    Full implementation (issue #526):

      =======  ========================================================
      element  calibration source(s)
      =======  ========================================================
      H        Young+2023; Schlichting+Young 2022; Luo+2024
               (strongly fO2-dependent; D(H) ≈ 0.01 at IW−2, ≈ 1 at IW−5)
      O        Badro+2015; Fischer+2015 (rises with T, P)
      Si       Hirose+2017; Badro+2018 (rises with T)
      C        Dasgupta+2013; Fischer+2020
      S        Boujibar+2014
      N        Dalou+2017; Grewal+2019
      =======  ========================================================

    Parameters
    ----------
    element : str
        One of 'H', 'O', 'Si', 'C', 'S', 'N'.
    pressure, temperature : float
        Conditions at the core-mantle boundary. Typical 20-60 GPa,
        3000-5000 K.
    fO2_dIW : float
        ΔIW at those conditions (log10 units).
    """
    if element not in {'H', 'O', 'Si', 'C', 'S', 'N'}:
        raise KeyError(f'metal_silicate_KD: unknown element {element!r}')
    return 0.0


def apply_metal_silicate_partitioning(
    hf_row: dict,
    dt: float,
    *,
    config=None,
    kd_provider: Callable = metal_silicate_KD,
) -> MetalSilicatePartitionResult:
    """
    Compute per-step net mass flux from silicate melt to core metal
    for each light element.

    Stub returns zero fluxes (since default K_D = 0.0). Call this
    once per PROTEUS step, between the interior solve and the redox
    solve.

    Issue #526.
    """
    return MetalSilicatePartitionResult(
        fluxes={e: 0.0 for e in ('H', 'O', 'Si', 'C', 'S', 'N')},
        dR_budget_mantle=0.0,
        dR_budget_core=0.0,
    )


# ==================================================================
# Issue #432 — EOS density corrections for Fe³⁺ and core light elements
# ==================================================================


@dataclass
class EOSCorrection:
    """Multiplicative correction factors on the mantle and core EOS."""
    mantle_factor: float = 1.0
    core_factor: float = 1.0


def eos_density_correction(
    *,
    Fe3_frac: float = 0.04,
    core_H_wt: float = 0.0,
    core_O_wt: float = 0.0,
    core_Si_wt: float = 0.0,
) -> EOSCorrection:
    """
    EOS density corrections driven by redox state and core
    light-element composition.

    Stub returns **identity factors** (mantle=1.0, core=1.0); no
    consumer in PROTEUS today.

    Full implementation (issue #432):
      * Mantle: Dorfman+2024 parametrisation ρ_mantle(Fe³⁺/Fe^T);
        typically +2 % density per 0.04 rise in Fe³⁺/Fe^T.
      * Core: Hirose+2019 ρ_core(H, O, Si); H has the largest leverage
        (-1 % per 0.1 wt% H).

    The function exists purely to lock the API; when #432 ships,
    SPIDER / Aragog / Zalmoxis EOS lookup paths will call it.
    """
    return EOSCorrection(mantle_factor=1.0, core_factor=1.0)
