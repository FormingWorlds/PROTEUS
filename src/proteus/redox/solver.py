"""
Redox solver: transactional Brent root-find over ΔIW (#57 Commit C).

Given a frozen R_target (R_atm_prev + R_mantle + R_core − ΔR_escape −
ΔR_dispro), find the `fO2_shift_IW` value that makes
`run_outgassing(ΔIW) → atmosphere state` satisfy the Evans/Hirschmann
redox-budget closure.

Contract (plan v6 §3.3):
- Transactional: inner Brent probes operate on `copy.deepcopy(hf_row)`;
  the caller's hf_row is NOT mutated until the solver commits a final
  write of `fO2_shift_IW` on convergence.
- Desiccation freeze: the `check_desiccation` gate is evaluated ONCE
  at entry and frozen for the whole Brent sweep. Inside the loop the
  residual function always takes the same (`run_desiccated` vs
  `run_outgassing`) branch so the residual is continuous.
- Bracket widening: try ±halfwidth first; on `ValueError` (no sign
  change) widen to ±2·halfwidth; then fall back to previous ΔIW with
  a logged warning + `redox_solver_fallback_count` increment.
- Warm start: prefer Mariana's Schaefer Eq 13 warm start
  (`log10_fO2_suggested_by_mariana` on hf_row). If NaN (fully reduced
  mantle) or missing, fall back to previous step's ΔIW.

The solver does NOT commit the final outgas result into hf_row. It
writes `hf_row['fO2_shift_IW'] = delta_IW_solved` only. The downstream
main-loop call to `run_outgassing(dirs, config, hf_row)` (at
`proteus.py:681`) is then the single authoritative commit at the
converged ΔIW.
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger('fwl.' + __name__)


@dataclass
class SolverResult:
    """
    Output of :func:`solve_fO2`. Stable API across Commits B → C → D.
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
    outgas_callable: Optional[Callable] = None,
) -> SolverResult:
    """
    Drive `hf_row['fO2_shift_IW']` to satisfy the Evans/Hirschmann
    redox-budget closure.

    Parameters
    ----------
    hf_row, hf_row_prev : dict
        Current and previous helpfile rows. Only `hf_row['fO2_shift_IW']`
        is mutated; nothing else.
    dirs, config : objects
        Passed through to the inner outgas call via deep-copy probes.
    mesh_state :
        Read-only Aragog mesh state (unused in the Commit-C solver
        because R_mantle is pre-frozen by the caller; kept in the
        signature for Commit D extensions).
    R_mantle_pre_frozen, R_core_pre_frozen : float
        Pre-computed R_mantle and R_core. Frozen by the caller so the
        residual depends only on ΔIW via R_atm.
    outgas_callable :
        Inner solve. Defaults to
        `proteus.outgas.wrapper.run_outgassing`. The desiccation /
        crystallisation branches in the main loop are handled here by
        checking hf_row state and calling `run_desiccated` in-line
        when appropriate.

    Returns
    -------
    SolverResult
    """
    if outgas_callable is None:
        from proteus.outgas.wrapper import check_desiccation, run_desiccated, run_outgassing
        outgas_callable = run_outgassing
        _check_desiccation_local = check_desiccation
        _run_desiccated_local = run_desiccated
    else:
        from proteus.outgas.wrapper import check_desiccation, run_desiccated
        _check_desiccation_local = check_desiccation
        _run_desiccated_local = run_desiccated

    from proteus.redox.budget import redox_budget_atm

    # ---- freeze targets at entry ----
    R_escape_this_step = (
        float(hf_row.get('R_escaped_cum', 0.0))
        - float(hf_row_prev.get('R_escaped_cum', 0.0))
    )
    # R_target: the R_atm value the post-outgas atmosphere should have
    # so that R_atm + R_mantle + R_core = R_atm_prev + R_mantle_pre +
    # R_core_pre - (R_escape - R_dispro). Since mantle/core are frozen,
    # the target simplifies.
    R_atm_prev = float(hf_row_prev.get('R_budget_atm', 0.0))
    dR_dispro = float(hf_row.get('dm_Fe0_to_core_this_step_as_RB', 0.0))
    R_target = R_atm_prev - R_escape_this_step - dR_dispro

    # Freeze desiccation decision.
    desiccated_at_entry = _check_desiccation_local(config, hf_row)

    # Warm start: Mariana's Schaefer Eq 13 suggestion if finite, else
    # previous step's ΔIW, else config's IC.
    warm = hf_row.get('redox_delta_IW_suggested_by_mariana', None)
    if warm is None or (isinstance(warm, float) and math.isnan(warm)):
        warm = hf_row_prev.get('fO2_shift_IW', None)
    if warm is None or (isinstance(warm, float) and (math.isnan(warm) or warm == 0.0)):
        # Fall back to config IC.
        warm = float(getattr(config.outgas, 'fO2_shift_IW', 0.0))
    warm = float(warm)

    halfwidth = float(getattr(config.redox, 'bracket_halfwidth', 4.0))
    rtol = float(getattr(config.redox, 'rtol', 1e-3))
    atol = float(getattr(config.redox, 'atol', 1e-4))
    max_iter = int(getattr(config.redox, 'max_iter', 20))

    def residual(delta_IW: float) -> float:
        """Post-outgas R_atm − R_atm_target on a deep-copy probe."""
        hf_row_tx = copy.deepcopy(hf_row)
        hf_row_tx['fO2_shift_IW'] = float(delta_IW)
        if desiccated_at_entry:
            _run_desiccated_local(config, hf_row_tx)
        else:
            outgas_callable(dirs, config, hf_row_tx)
        return redox_budget_atm(hf_row_tx) - R_target

    # Brent with bracket-widening.
    from scipy.optimize import brentq

    delta_solved = None
    widened = False
    fell_back = False
    iterations = 0

    for attempt, factor in enumerate([1.0, 2.0]):
        bracket_lo = warm - halfwidth * factor
        bracket_hi = warm + halfwidth * factor
        try:
            delta_solved, result_obj = brentq(
                residual, bracket_lo, bracket_hi,
                rtol=rtol, xtol=atol, maxiter=max_iter,
                full_output=True,
            )
            iterations = result_obj.iterations
            widened = attempt > 0
            log.debug(
                'redox Brent converged in %d iter at ΔIW=%.4f (bracket '
                '[%.2f, %.2f], %s)',
                iterations, delta_solved, bracket_lo, bracket_hi,
                'widened' if widened else 'nominal',
            )
            break
        except (ValueError, RuntimeError) as exc:
            log.warning(
                'redox Brent failed at halfwidth=%.1f: %s',
                halfwidth * factor, exc,
            )

    if delta_solved is None:
        fell_back = True
        delta_solved = warm
        hf_row['redox_solver_fallback_count'] = (
            float(hf_row.get('redox_solver_fallback_count', 0.0)) + 1
        )
        log.warning(
            'redox solver fallback: using ΔIW=%.4f (previous step)',
            delta_solved,
        )

    # Commit: write only the scalar. Do NOT call outgas_callable on
    # the real hf_row — the main loop's downstream run_outgassing
    # call at proteus.py:681 is the authoritative commit.
    hf_row['fO2_shift_IW'] = float(delta_solved)
    final_residual = residual(delta_solved) if not fell_back else float('nan')
    return SolverResult(
        delta_IW=float(delta_solved),
        converged=(delta_solved is not None and not fell_back),
        final_residual=float(final_residual) if math.isfinite(final_residual) else float('nan'),
        iterations=iterations,
        widened_bracket=widened,
        fell_back_to_previous=fell_back,
    )
