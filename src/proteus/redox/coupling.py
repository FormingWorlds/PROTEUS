"""
Redox main-loop orchestration (#57 Commit C).

Single entry point that `proteus.py`'s main loop calls once per step,
between `run_escape` and `run_outgassing` (plan v6 §3.1). Static mode
is a no-op on physics (only pins `fO2_shift_IW`); evolving modes run
the full sequence:

1. Debit ZEPHYRUS per-species escape flux into `R_budget_atm`.
2. Run Mariana's stateful Fe-reservoir evolution (stub-bulk today).
3. Pre-freeze R_mantle and R_core for the Brent residual.
4. Transactional Brent over ΔIW via `solve_fO2`.
5. Soft-assert redox-budget conservation.

The downstream `run_outgassing` call (at `proteus.py:681`) runs AFTER
`run_redox_step` and uses the ΔIW the solver wrote.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger('fwl.' + __name__)


_MODE_ENUM = {'static': 0, 'fO2_init': 1, 'composition': 2}


def record_mode_active(hf_row: dict, mode: str) -> None:
    """Stamp `hf_row['redox_mode_active']` with an integer enum."""
    hf_row['redox_mode_active'] = float(_MODE_ENUM.get(mode, 0))


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
        by the Mariana stub and ignored by the Brent solver in Commit C.
    escape_fluxes_species_kg : dict
        Per-species kg outflow from ZEPHYRUS over the current step, as
        computed by
        `escape.wrapper.atm_escape_fluxes_species_kg`. Missing/None →
        no escape debit this step.
    """
    mode = getattr(config.redox, 'mode', 'static')
    record_mode_active(hf_row, mode)

    # Static mode: pin fO2_shift_IW to the config value. No solver work.
    if mode == 'static':
        hf_row['fO2_shift_IW'] = float(config.outgas.fO2_shift_IW)
        return

    # --- Evolving mode path ---
    from proteus.redox.budget import redox_budget_core, redox_budget_mantle
    from proteus.redox.conservation import (
        assert_redox_conserved,
        debit_escape,
    )
    from proteus.redox.partitioning import advance_fe_reservoirs
    from proteus.redox.solver import solve_fO2

    mantle_comp = getattr(config.interior_struct, 'mantle_comp', None)

    # Step 1: debit escape flux.
    if escape_fluxes_species_kg:
        try:
            debit_escape(hf_row, escape_fluxes_species_kg)
        except ValueError as exc:
            log.warning('debit_escape rejected flux: %s', exc)

    # Step 2: Mariana's Fe-reservoir evolution (stub-bulk today).
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
        hf_row['redox_delta_IW_suggested_by_mariana'] = (
            fe_result.log10_fO2_surface
        )
    else:
        log.debug(
            'run_redox_step: no mesh_state supplied; skipping '
            'advance_fe_reservoirs (stub-bulk does nothing anyway).'
        )

    # Step 3: pre-freeze R_mantle, R_core for the Brent residual.
    R_mantle_pre = redox_budget_mantle(hf_row, mantle_comp)
    R_core_pre = redox_budget_core(hf_row)
    hf_row['R_budget_mantle'] = R_mantle_pre
    hf_row['R_budget_core'] = R_core_pre

    # Step 4: transactional Brent.
    solve_fO2(
        hf_row, hf_row_prev, dirs, config,
        mesh_state=mesh_state,
        R_mantle_pre_frozen=R_mantle_pre,
        R_core_pre_frozen=R_core_pre,
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
