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
    outgas_callable: Optional[Callable] = None,
) -> SolverResult:
    """
    Drive `hf_row['fO2_shift_IW']` to satisfy the Evans/Hirschmann
    redox-budget closure.

    Residual (Commit C.1, round-5 physics review fix): balances the
    FULL conserved quantity R_atm + R_mantle + R_core on the post-
    outgas probe against R_total_prev - ΔR_escape - ΔR_dispro. Earlier
    C implementation pre-froze R_mantle / R_core and solved on R_atm
    alone, but CALLIOPE + atmodeller write `{species}_mol_liquid` on
    every probe, which changes R_mantle. Pre-freezing was therefore
    wrong. The current form uses the same `redox_budget_*` functions
    on each probe and is invariant to whichever keys the outgas step
    writes.

    Parameters
    ----------
    hf_row, hf_row_prev : dict
        Current and previous helpfile rows. Only `hf_row['fO2_shift_IW']`
        (and the `redox_solver_fallback_count` counter on fallback) is
        mutated by this function.
    dirs, config : objects
        Passed through to the inner outgas call via deep-copy probes.
    mesh_state :
        Read-only Aragog mesh state (unused in the Brent residual;
        kept in the signature for Commit D extensions).
    outgas_callable :
        Inner solve. Defaults to
        `proteus.outgas.wrapper.run_outgassing`. Desiccation branch is
        handled in-line by calling `run_desiccated` when `check_desiccation`
        returns True at entry (gate frozen for the whole Brent sweep).

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

    from proteus.redox.budget import (
        redox_budget_atm,
        redox_budget_core,
        redox_budget_mantle,
    )

    # ---- freeze target at entry ----
    R_escape_this_step = (
        float(hf_row.get('R_escaped_cum', 0.0))
        - float(hf_row_prev.get('R_escaped_cum', 0.0))
    )
    # R_total_prev from the *previous committed* row. R_budget_total
    # is a passive diagnostic written by _write_passive_diagnostics
    # every step in both static and evolving modes (coupling.py), so
    # it should always be present. Fall back to sum-from-components if
    # not.
    R_total_prev = float(hf_row_prev.get('R_budget_total', float('nan')))
    if math.isnan(R_total_prev):
        R_total_prev = (
            float(hf_row_prev.get('R_budget_atm', 0.0))
            + float(hf_row_prev.get('R_budget_mantle', 0.0))
            + float(hf_row_prev.get('R_budget_core', 0.0))
        )
    dR_dispro = float(hf_row.get('dm_Fe0_to_core_this_step_as_RB', 0.0))
    # Conservation: R_total_new = R_total_prev + ΔR_atm_escape - ΔR_dispro.
    # `R_escape_this_step` is the cumulative delta in R_atm stored by
    # `debit_escape` (positive for reducer-species escape since R_atm
    # rises when reducers leave). `ΔR_dispro` is the RB transferred
    # from mantle+atm to core by Fe disproportionation (positive =
    # mantle loses reducing power to core). Both appear with opposite
    # signs in the target. Round-6 review found earlier C/C.1
    # implementations had a sign error (subtracting instead of adding
    # R_escape_this_step), driving Brent to the wrong ΔIW by ~4·Δn
    # every step with escape active.
    R_target = R_total_prev + R_escape_this_step - dR_dispro

    mantle_comp = getattr(
        getattr(config, 'interior_struct', None), 'mantle_comp', None,
    )

    # Freeze desiccation decision.
    desiccated_at_entry = _check_desiccation_local(config, hf_row)

    # Warm start: Mariana's ΔIW suggestion if finite, else previous
    # step's ΔIW, else config IC. `warm == 0.0` is a legitimate value
    # (IW+0), so do NOT treat it as a sentinel; only NaN / None mean
    # "unavailable".
    warm = hf_row.get('redox_delta_IW_suggested_by_mariana', None)
    if warm is None or (isinstance(warm, float) and math.isnan(warm)):
        warm = hf_row_prev.get('fO2_shift_IW', None)
    if warm is None or (isinstance(warm, float) and math.isnan(warm)):
        warm = float(getattr(config.outgas, 'fO2_shift_IW', 0.0))
    warm = float(warm)

    halfwidth = float(getattr(config.redox, 'bracket_halfwidth', 4.0))
    rtol = float(getattr(config.redox, 'rtol', 1e-3))
    atol = float(getattr(config.redox, 'atol', 1e-4))
    max_iter = int(getattr(config.redox, 'max_iter', 20))

    # Closure variable: record the most-recent residual value so we
    # can return it without calling residual() a second time (round-5
    # numerics review MAJOR-2 fix).
    last_residual = [float('nan')]

    def residual(delta_IW: float) -> float:
        """
        Full-budget residual on a deep-copy probe:
            [R_atm + R_mantle + R_core](post-outgas) - R_target
        """
        hf_row_tx = copy.deepcopy(hf_row)
        hf_row_tx['fO2_shift_IW'] = float(delta_IW)
        if desiccated_at_entry:
            _run_desiccated_local(config, hf_row_tx)
        else:
            outgas_callable(dirs, config, hf_row_tx)
        r = (
            redox_budget_atm(hf_row_tx)
            + redox_budget_mantle(hf_row_tx, mantle_comp)
            + redox_budget_core(hf_row_tx)
            - R_target
        )
        last_residual[0] = r
        return r

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
            iterations = getattr(result_obj, 'iterations', 0)
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
                'redox Brent %s at halfwidth=%.1f: %s',
                type(exc).__name__, halfwidth * factor, exc,
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
    # Return the last residual Brent actually evaluated. No extra
    # outgas call.
    final_residual = last_residual[0] if not fell_back else float('nan')
    return SolverResult(
        delta_IW=float(delta_solved),
        converged=(delta_solved is not None and not fell_back),
        final_residual=float(final_residual) if math.isfinite(final_residual) else float('nan'),
        iterations=iterations,
        widened_bracket=widened,
        fell_back_to_previous=fell_back,
    )
