"""
Redox main-loop orchestration (#57 — Commit C wiring point).

This module is the single entry point that `proteus.py`'s main loop
calls once per step to:

1. Debit escape RB.
2. Run Mariana's stateful Fe-reservoir evolution (stub in Commit B).
3. Compute the pre-frozen `R_mantle` and `R_core` for the solver.
4. Call the outer Brent solver.
5. Soft-assert redox conservation.
6. Update `hf_row['redox_mode_active']` and related diagnostics.

Commit B ships the skeleton so tests can import this module; Commit C
wires each call site into `proteus.py` after the interior step.
"""
from __future__ import annotations

import logging

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
    escape_fluxes_species_kg: dict[str, float] | None = None,
) -> None:
    """
    Run the per-step redox sequence (plan v6 §3.1).

    Commit-B stub: records the active mode and leaves fO2_shift_IW
    pinned to its static value (or previous CSV value in resume).

    Commit C will:
      1. `debit_escape(hf_row, escape_fluxes_species_kg)`.
      2. `advance_fe_reservoirs(mesh_state, hf_row_prev, ...)` — run
         Mariana's stub (or real implementation if #653 landed).
      3. Pre-freeze `R_mantle`, `R_core`.
      4. `solve_fO2(hf_row, hf_row_prev, mesh_state, R_mantle, R_core,
         dirs, config)` — transactional Brent.
      5. `assert_redox_conserved(hf_row, hf_row_prev, soft_tol=...)`.
    """
    mode = getattr(config.redox, 'mode', 'static')
    record_mode_active(hf_row, mode)

    if mode == 'static':
        # Pin fO2_shift_IW to the config value.
        hf_row['fO2_shift_IW'] = float(config.outgas.fO2_shift_IW)
        return

    # Evolving modes: Commit C wires in debit_escape, Mariana,
    # solve_fO2, and assert_redox_conserved. For Commit B we leave
    # the scalar at the previous step's value.
    prev = hf_row_prev.get('fO2_shift_IW', None)
    if prev is None or prev == 0.0:
        prev = float(config.outgas.fO2_shift_IW)
    hf_row['fO2_shift_IW'] = float(prev)
