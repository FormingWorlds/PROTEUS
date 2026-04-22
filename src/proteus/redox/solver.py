"""
Redox solver (#57 — Commit C stub).

The transactional Brent root-find over ΔIW lives here. Commit B ships
only the stub + dataclass; Commit C replaces the body with the full
implementation described in plan v6 §3.3, which:

- Accepts mesh state (P, T, φ, cell_mass profiles) from the Aragog
  wrapper so `R_mantle` is pre-frozen before the Brent loop.
- Freezes desiccation decision at step boundary (no step discontinuity
  in the residual).
- Runs `run_outgassing` as the inner function on a deep-copy of
  `hf_row` per probe (idempotency fix).
- Widens bracket ±halfwidth → ±2·halfwidth on first failure before
  falling back to previous ΔIW.
- Commits one outgas call into the real `hf_row` at the converged ΔIW.

See plan v6 §3.3 and test_solver.py for the detailed contract.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger('fwl.' + __name__)


@dataclass
class SolverResult:
    """
    Output of :func:`solve_fO2`. Stable across Commits B → C.
    """
    delta_IW: float                      # solved log10 fO2 − log10 fO2_IW
    converged: bool
    final_residual: float                # mol e- at the committed ΔIW
    iterations: int = 0
    widened_bracket: bool = False
    fell_back_to_previous: bool = False


def solve_fO2(
    hf_row: dict,
    hf_row_prev: dict,
    dirs: dict,
    config,
    *,
    mesh_state=None,
    R_mantle_pre_frozen: float = 0.0,
    R_core_pre_frozen: float = 0.0,
    outgas_callable=None,
) -> SolverResult:
    """
    Drive `hf_row['fO2_shift_IW']` to satisfy the Evans/Hirschmann
    redox-budget closure.

    Commit-B stub: returns `hf_row_prev['fO2_shift_IW']` (or
    `config.outgas.fO2_shift_IW` if no prior row) unchanged. Static
    mode works correctly today; `fO2_init` and `composition` modes
    are no-ops until Commit C wires in the Brent solver.

    Parameters
    ----------
    hf_row : dict
        Live helpfile row. In Commit C this will be mutated in place
        after Brent convergence (one outgas commit at the solved ΔIW);
        the stub leaves it untouched.
    hf_row_prev : dict
        Last successfully committed step's hf_row (source of the
        warm-start ΔIW).
    dirs, config : objects
        Passed through to the inner outgas call in Commit C.
    mesh_state : AragogMeshState, optional
        Per-cell (P, T, φ, cell_mass) profiles. Commit C reads these
        via `aragog.get_mesh_state_for_redox()`.
    R_mantle_pre_frozen, R_core_pre_frozen : float
        Pre-computed R_mantle and R_core (by the caller) so the
        Brent residual depends only on ΔIW via the atmosphere.
    outgas_callable : Callable
        Inner solve; defaults to `proteus.outgas.wrapper.run_outgassing`.
    """
    prev = hf_row_prev.get('fO2_shift_IW', None)
    if prev is None or prev == 0.0:
        prev = float(getattr(config.outgas, 'fO2_shift_IW', 0.0))
    delta = float(prev)

    log.debug('solve_fO2 stub: returning previous ΔIW = %.4f', delta)

    return SolverResult(
        delta_IW=delta,
        converged=True,
        final_residual=0.0,
    )
