"""
Redox main-loop orchestration (#57 Commit C / C.1).

Single entry point that `proteus.py`'s main loop calls once per step,
between `run_escape` and `run_outgassing` (plan v6 §3.1).

Static mode pins `fO2_shift_IW` to the config value and writes passive
diagnostic columns (R_budget_atm / R_mantle / R_core / Fe3_frac) so
that every run — even pre-plan-compatible ones — has visible redox
state. Evolving modes run the full sequence:

1. Bootstrap `R_budget_atm` from the current atmosphere inventory
   (necessary on step 0 where the column would otherwise be zero).
2. Debit ZEPHYRUS per-species escape flux into `R_budget_atm`.
3. Run Mariana's stateful Fe-reservoir evolution (stub-bulk today).
   Convert the stub's log10 fO2 suggestion to ΔIW at (T_surf, P_surf)
   before storing as the solver warm-start.
4. Transactional Brent over ΔIW via `solve_fO2`. Residual balances the
   FULL R_atm + R_mantle + R_core against R_total_prev (NO partial
   pre-freeze — CALLIOPE writes `_mol_liquid` keys that `redox_budget_mantle`
   reads, so the mantle budget is not invariant across probes).
5. Soft-assert redox-budget conservation.

The downstream `run_outgassing` call at `proteus.py:681` uses the ΔIW
the solver wrote; the Brent solver itself does NOT commit a final
outgas — its deep-copy probes are pure.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

log = logging.getLogger('fwl.' + __name__)


_MODE_ENUM = {'static': 0, 'fO2_init': 1, 'composition': 2}


def record_mode_active(hf_row: dict, mode: str) -> None:
    """Stamp `hf_row['redox_mode_active']` with an integer enum."""
    hf_row['redox_mode_active'] = float(_MODE_ENUM.get(mode, 0))


def _write_passive_diagnostics(hf_row: dict, mantle_comp) -> None:
    """
    Populate R_budget_* + Fe3_frac diagnostic columns in any mode
    (plan v6 §3.6 specifies passive diagnostic writes in static mode).
    """
    from proteus.redox.budget import (
        redox_budget_atm,
        redox_budget_core,
        redox_budget_mantle,
    )
    hf_row['R_budget_atm'] = redox_budget_atm(hf_row)
    hf_row['R_budget_mantle'] = redox_budget_mantle(hf_row, mantle_comp)
    hf_row['R_budget_core'] = redox_budget_core(hf_row)
    hf_row['R_budget_total'] = (
        hf_row['R_budget_atm']
        + hf_row['R_budget_mantle']
        + hf_row['R_budget_core']
    )


def run_redox_step(
    hf_row: dict,
    hf_row_prev: dict,
    dirs: dict,
    config,
    *,
    mesh_state=None,
    escape_fluxes_species_kg: Optional[dict[str, float]] = None,
) -> None:
    """
    Per-step redox orchestration.

    Parameters
    ----------
    hf_row : dict
        Live helpfile row; mutated in place.
    hf_row_prev : dict
        Previous committed helpfile row (for ΔR_escape and warm start).
    dirs, config : objects
        Passed through to the inner solver's outgas call.
    mesh_state :
        Read-only Aragog mesh (from
        `interior_energetics.aragog.get_mesh_state_for_redox`). Consumed
        by the Mariana stub; ignored by the Brent solver.
    escape_fluxes_species_kg : dict
        Per-species kg outflow from ZEPHYRUS over the current step.
    """
    mode = getattr(config.redox, 'mode', 'static')
    record_mode_active(hf_row, mode)

    mantle_comp = getattr(
        getattr(config, 'interior_struct', None), 'mantle_comp', None,
    )

    # Static mode: pin fO2_shift_IW, write passive diagnostics, return.
    if mode == 'static':
        hf_row['fO2_shift_IW'] = float(config.outgas.fO2_shift_IW)
        _write_passive_diagnostics(hf_row, mantle_comp)
        return

    # --- Evolving mode path ---
    from proteus.redox.buffers import log10_fO2_IW
    from proteus.redox.conservation import (
        assert_redox_conserved,
        debit_escape,
    )
    from proteus.redox.partitioning import advance_fe_reservoirs
    from proteus.redox.solver import solve_fO2

    # Step 1: bootstrap R_budget_atm from the atmosphere inventory if
    # we have not already written it this step. This is critical on
    # step 0 where the column would otherwise be zero and the solver
    # would drive the post-outgas R_atm to zero regardless of true
    # state.
    _write_passive_diagnostics(hf_row, mantle_comp)

    # Step 2: debit escape flux.
    if escape_fluxes_species_kg:
        try:
            debit_escape(hf_row, escape_fluxes_species_kg)
        except ValueError as exc:
            log.warning('debit_escape rejected flux: %s', exc)

    # Step 3: Mariana's Fe-reservoir evolution (stub-bulk today).
    if mesh_state is not None:
        fe_result = advance_fe_reservoirs(
            pressure_profile=mesh_state.pressure_profile,
            temperature_profile=mesh_state.temperature_profile,
            melt_fraction_profile=mesh_state.melt_fraction_profile,
            melt_fraction_profile_prev=mesh_state.melt_fraction_profile_prev,
            cell_mass_profile=mesh_state.cell_mass_profile,
            n_Fe3_melt_prev=float(hf_row_prev.get('n_Fe3_melt', 0.0)),
            n_Fe2_melt_prev=float(hf_row_prev.get('n_Fe2_melt', 0.0)),
            n_Fe3_solid_cell_prev=mesh_state.n_Fe3_solid_cell_prev,
            n_Fe2_solid_cell_prev=mesh_state.n_Fe2_solid_cell_prev,
            mantle_comp=mantle_comp,
            oxybarometer=getattr(config.redox, 'oxybarometer', 'schaefer2024'),
            phi_crit=float(getattr(config.redox, 'phi_crit', 0.4)),
        )
        hf_row['n_Fe3_melt'] = fe_result.n_Fe3_melt
        hf_row['n_Fe2_melt'] = fe_result.n_Fe2_melt
        hf_row['Fe3_frac'] = fe_result.Fe3_frac_bulk_melt
        hf_row['dm_Fe0_to_core_cum'] = (
            float(hf_row.get('dm_Fe0_to_core_cum', 0.0))
            + fe_result.dm_Fe0_to_core
        )
        # Stash per-cell Fe / fO2 arrays for the NEXT Aragog NetCDF
        # write (plan v6 §3.7, E.1). Main-loop order is run_interior
        # → run_redox_step, so the snapshot at iter N carries the
        # reservoir state computed at the end of iter N-1. Stub-bulk
        # today (zeros + NaN); wired through to real per-cell arrays
        # once #653 (Mariana) lands. Shape matches the mesh.
        n_stag = mesh_state.pressure_profile.size
        log10_fO2_profile = fe_result.log10_fO2_profile
        if log10_fO2_profile is None:
            log10_fO2_profile = np.full(n_stag, np.nan)
        hf_row['_redox_per_cell'] = {
            'n_Fe3_solid_cell': np.asarray(fe_result.n_Fe3_solid_cell),
            'n_Fe2_solid_cell': np.asarray(fe_result.n_Fe2_solid_cell),
            'log10_fO2_profile': np.asarray(log10_fO2_profile),
        }
        # Convert Mariana's log10 fO2 to ΔIW at the melt surface so the
        # solver's warm-start has the right units. The key name
        # `redox_delta_IW_suggested_by_mariana` is intentionally ΔIW,
        # not raw log10 fO2 (round-5 reviewer caught the unit
        # mismatch; see plan §8 Commit C.1).
        log10_fO2_raw = fe_result.log10_fO2_surface
        if math.isnan(log10_fO2_raw) or mesh_state.pressure_profile.size == 0:
            hf_row['redox_delta_IW_suggested_by_mariana'] = float('nan')
        else:
            # Aragog staggered nodes run CMB → surface; the surface is
            # at index -1, not 0 (round-6 review N-2 fix). Using [0]
            # would evaluate the IW buffer at ~130 GPa / ~4000 K
            # instead of ~1 bar / ~2000 K, pushing ΔIW off by many
            # log units.
            T_surf = float(mesh_state.temperature_profile[-1])
            P_surf = float(mesh_state.pressure_profile[-1])
            iw_at_surf = log10_fO2_IW(T_surf, P_surf)
            hf_row['redox_delta_IW_suggested_by_mariana'] = (
                log10_fO2_raw - iw_at_surf
            )
    else:
        log.debug(
            'run_redox_step: no mesh_state supplied; skipping '
            'advance_fe_reservoirs.'
        )

    # Step 4: transactional Brent over ΔIW. Residual balances the
    # FULL R_atm + R_mantle + R_core against R_total_prev (plan v6
    # §3.3 + round-5 physics review).
    solve_fO2(hf_row, hf_row_prev, dirs, config, mesh_state=mesh_state)

    # NB: the previous C/C.1 implementation called
    # _write_passive_diagnostics a second time here. Round-6 review
    # caught that the second call recomputes R_budget_atm from
    # `{species}_mol_atm` keys — which were NEVER modified by
    # debit_escape (escape only touched `R_budget_atm` directly) —
    # silently reverting the escape debit. The second call is removed.
    # `R_budget_total` is recomputed below to stay in sync with the
    # debited `R_budget_atm` and the frozen `R_budget_mantle` /
    # `R_budget_core`.
    hf_row['R_budget_total'] = (
        float(hf_row.get('R_budget_atm', 0.0))
        + float(hf_row.get('R_budget_mantle', 0.0))
        + float(hf_row.get('R_budget_core', 0.0))
    )

    # Step 5: soft-assert conservation.
    soft_tol = float(getattr(config.redox, 'soft_conservation_tol', 1e-3))
    include_atm = bool(getattr(config.redox, 'include_atm', True))
    include_mantle = bool(getattr(config.redox, 'include_mantle', True))
    include_core = bool(getattr(config.redox, 'include_core', True))
    assert_redox_conserved(
        hf_row, hf_row_prev,
        soft_tol=soft_tol,
        include_atm=include_atm,
        include_mantle=include_mantle,
        include_core=include_core,
    )
