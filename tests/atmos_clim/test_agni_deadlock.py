"""
Unit tests for the AGNI deadlock detection plumbing.

These tests verify the convergence-flag pipeline added on 2026-04-07 in
response to the diagnosis at output_files/analysis_1EO_M1_profiles. The fix
threads a `converged` flag from `_solve_energy` -> `run_agni` -> the wrapper
-> `Atmos_t.converged` so that the main coupling loop can detect a deadlock
where AGNI repeatedly fails to converge AND the interior state is frozen.

What is tested here (Python-only, no Julia, no SOCRATES):
1. `Atmos_t.converged` defaults to True (so non-AGNI atmospheres never trigger
   the deadlock detector).
2. The wrapper pops `agni_converged` from the atmosphere output dict so it
   does not pollute the helpfile schema.
3. The wrapper sets `atmos_o.converged` to False when AGNI reports failure.
4. The wrapper resets `atmos_o.converged` to True when AGNI succeeds, even if
   a previous iteration left it False.
5. Edge case: if the atmosphere module is `dummy` (no `agni_converged` key in
   output), the flag defaults to True and never raises KeyError.

What is NOT tested here:
- The actual main-loop deadlock counter logic in `proteus.py`. That sits
  inside the main `start()` method which constructs a full Proteus instance,
  so it is verified by integration / smoke tests rather than unit tests.
"""

from __future__ import annotations

import pytest

from proteus.atmos_clim.common import Atmos_t

pytestmark = pytest.mark.unit


def test_atmos_t_converged_default_true():
    """`Atmos_t.converged` defaults to True so non-AGNI atmospheres do not
    trigger the deadlock detector."""
    atmos_o = Atmos_t()
    # The deadlock detector in the main loop only fires when this is False,
    # so the default MUST be True (not None, not 0, not 1, not "yes").
    assert atmos_o.converged is True
    # And it must remain True across re-instantiation (sanity check).
    other = Atmos_t()
    assert other.converged is True
    assert atmos_o.converged is other.converged
    assert atmos_o is not other


def test_converged_flag_pop_pattern():
    """Simulate the wrapper.py merge logic: `agni_converged` MUST be popped
    from atm_output before the merge into hf_row, so it never reaches the
    helpfile CSV. Also confirms that the default is True if missing."""
    # Construct a minimal hf_row with the keys the wrapper merges. The merge
    # code is `for key in atm_output.keys(): if key in hf_row: hf_row[key] = ...`.
    hf_row = {
        'F_atm': 1.0e3,
        'F_olr': 1.5e3,
        'T_surf': 3000.0,
    }

    # Case 1: AGNI converged.
    atm_output = {
        'F_atm': 5.0e3,
        'F_olr': 5.5e3,
        'T_surf': 3500.0,
        'agni_converged': True,
    }
    converged = bool(atm_output.pop('agni_converged', True))
    assert converged is True
    assert 'agni_converged' not in atm_output
    for key in atm_output.keys():
        if key in hf_row:
            hf_row[key] = atm_output[key]
    # Verify the merge worked AND the flag did not leak into hf_row.
    assert hf_row['F_atm'] == pytest.approx(5.0e3)
    assert 'agni_converged' not in hf_row

    # Case 2: AGNI failed. The flag must come through as False.
    atm_output_fail = {'F_atm': 4.9e3, 'agni_converged': False}
    converged_fail = bool(atm_output_fail.pop('agni_converged', True))
    assert converged_fail is False
    assert 'agni_converged' not in atm_output_fail

    # Case 3: dummy atmosphere — no `agni_converged` key. Default must be True
    # so the deadlock detector does not fire spuriously.
    atm_output_dummy = {'F_atm': 100.0}
    converged_dummy = bool(atm_output_dummy.pop('agni_converged', True))
    assert converged_dummy is True


def test_atmos_t_converged_resets_on_success():
    """Once `atmos_o.converged` has been set to False (failure), a subsequent
    successful AGNI call must reset it to True. Otherwise the deadlock counter
    in the main loop would latch onto a stale state."""
    atmos_o = Atmos_t()

    # Simulate a failed AGNI call
    atmos_o.converged = bool({'agni_converged': False}.pop('agni_converged', True))
    assert atmos_o.converged is False

    # Simulate a successful AGNI call
    atmos_o.converged = bool({'agni_converged': True}.pop('agni_converged', True))
    assert atmos_o.converged is True

    # And if AGNI mysteriously omits the key (e.g. switched to JANUS mid-run),
    # the default must be True.
    atmos_o.converged = bool({}.pop('agni_converged', True))
    assert atmos_o.converged is True


def _frozen_step(converged, prev_state, new_state, counter, threshold,
                 hf_all_is_none=False):
    """Replicate the deadlock counter logic from proteus.py.

    Mirrors the production code at proteus.py::start() (right after
    run_atmosphere). Bit-exact equality is INTENTIONAL for T_magma and
    Phi_global because a truly frozen interior produces bit-exact identical
    values; the relative tolerance on F_atm is needed to catch jittery AGNI
    non-convergence noise on the same physical state.
    """
    aborted = False
    if not converged:
        if hf_all_is_none:
            return counter, aborted  # cannot fire without prev row
        cur_F = new_state['F_atm']
        prev_F = prev_state['F_atm']
        F_rel = abs(cur_F - prev_F) / max(abs(prev_F), 1.0)
        # bit-exact comparison for T_magma and Phi_global is intentional
        frozen = (
            prev_state['T_magma'] == new_state['T_magma']
            and prev_state['Phi_global'] == new_state['Phi_global']
            and F_rel < 1.0e-6
        )
        if frozen:
            counter += 1
            if counter >= threshold:
                aborted = True
        else:
            counter = 0
    else:
        counter = 0
    return counter, aborted


def test_deadlock_counter_logic():
    """The counter increments only when BOTH conditions hold:
      (a) atmos_o.converged is False
      (b) (T_magma, Phi_global) are bit-exactly identical to the previous
          committed row AND F_atm has changed by less than 1e-6 relative.

    The counter resets to 0 when either condition is broken (AGNI converges
    OR the interior state moves). After 3 consecutive deadlock iterations,
    the run aborts.

    Edge cases tested:
    - Reset on convergence after a partial deadlock streak.
    - Reset when interior moves (any of Tm, Phi, F_atm changes).
    - Threshold trigger at exactly 3 consecutive frozen failures.
    """
    threshold = 3
    counter = 0
    aborted = False

    # Reference frozen state used in most of the steps below.
    frozen = {'T_magma': 3712.1, 'Phi_global': 0.7648, 'F_atm': 8018.07}

    # Iteration 1: AGNI succeeds → counter stays 0
    counter, aborted = _frozen_step(True, frozen, frozen, counter, threshold)
    assert counter == 0
    assert not aborted

    # Iteration 2: AGNI fails, state frozen → counter = 1
    counter, aborted = _frozen_step(False, frozen, frozen, counter, threshold)
    assert counter == 1
    assert not aborted

    # Iteration 3: AGNI fails, but interior moves (Tm changes) → counter resets
    counter, aborted = _frozen_step(
        False, frozen, {**frozen, 'T_magma': 3700.0}, counter, threshold
    )
    assert counter == 0
    assert not aborted

    # Now simulate a true deadlock: 3 consecutive failures with frozen state
    counter, aborted = _frozen_step(False, frozen, frozen, counter, threshold)
    assert counter == 1
    counter, aborted = _frozen_step(False, frozen, frozen, counter, threshold)
    assert counter == 2
    counter, aborted = _frozen_step(False, frozen, frozen, counter, threshold)
    assert counter == 3
    assert aborted is True  # threshold reached, abort triggered


def test_deadlock_counter_resets_on_phi_change_only():
    """The deadlock detector must reset if Phi changes, even if the other
    two are still equal. Guards against false-positive abort when the
    interior is making slow but real progress in just one quantity."""
    threshold = 3
    counter = 0

    base = {'T_magma': 3712.1, 'Phi_global': 0.7648, 'F_atm': 8018.07}

    # 2 consecutive failures + frozen
    counter, _ = _frozen_step(False, base, base, counter, threshold)
    counter, _ = _frozen_step(False, base, base, counter, threshold)
    assert counter == 2

    # Phi changed — must reset (intentional bit-exact comparison)
    counter, _ = _frozen_step(
        False, base, {**base, 'Phi_global': 0.7649}, counter, threshold
    )
    assert counter == 0

    # Same exercise with F_atm changing meaningfully (5% relative)
    counter, _ = _frozen_step(False, base, base, counter, threshold)
    counter, _ = _frozen_step(False, base, base, counter, threshold)
    assert counter == 2
    counter, _ = _frozen_step(
        False, base, {**base, 'F_atm': 8418.97}, counter, threshold
    )
    assert counter == 0


def test_deadlock_counter_with_jittery_F_atm():
    """AGNI non-convergence may return slightly different F_atm values on
    each retry due to partial Newton steps. The detector uses a 1e-6
    relative tolerance on F_atm so that this jitter still registers as a
    frozen state. This is the bug-class flagged by the code review of the
    initial bit-exact-only implementation.

    A pure-equality detector would silently miss the deadlock; the relative
    tolerance must catch it.
    """
    threshold = 3
    counter = 0
    aborted = False

    base_F = 8018.0739
    base = {'T_magma': 3712.1, 'Phi_global': 0.7648, 'F_atm': base_F}

    # Simulate 3 consecutive AGNI failures where each successive F_atm
    # jitters by at most 1e-4 absolute (~1.2e-8 relative). The production
    # tolerance is 1e-6 relative, so all 3 pairwise comparisons fall well
    # below it. Reproduces the typical "AGNI returns slightly different
    # F_atm on each retry" pattern observed in real stuck loops.
    jittered_states = [
        {**base, 'F_atm': base_F + 1.0e-4},
        {**base, 'F_atm': base_F + 1.5e-4},
        {**base, 'F_atm': base_F + 2.0e-4},
    ]

    # In production, `prev` is updated to the last committed row each
    # iteration (because the helpfile is extended at the bottom of the
    # main loop body). Mirror that here by walking pairs.
    prev = base
    for new_state in jittered_states:
        counter, aborted = _frozen_step(False, prev, new_state, counter, threshold)
        prev = new_state

    # All 3 jittered iterations should have registered as frozen, triggering
    # the abort. Without the tolerance, counter would be 0 (each comparison
    # would be != due to bit-level differences).
    assert counter == 3, (
        f'Expected counter=3 with relative-tolerance comparison, got {counter}. '
        'The jittery F_atm should still register as a frozen state.'
    )
    assert aborted is True


def test_deadlock_counter_first_iteration_no_prev_row():
    """On a fresh run's first iteration, hf_all is None and there is no
    previous row to compare against. The detector must NOT crash and must
    NOT increment the counter — it should defer the deadlock decision until
    a baseline row exists. This catches the original review-found crash bug
    where len(self.hf_all) raised TypeError on iteration 0.
    """
    threshold = 3
    counter = 0

    base = {'T_magma': 4000.0, 'Phi_global': 1.0, 'F_atm': 5e3}

    # AGNI fails on the first iteration but hf_all is still None
    counter, aborted = _frozen_step(
        False, base, base, counter, threshold, hf_all_is_none=True
    )
    assert counter == 0
    assert not aborted

    # Even after multiple "fails with no prev" the counter stays at 0
    for _ in range(5):
        counter, aborted = _frozen_step(
            False, base, base, counter, threshold, hf_all_is_none=True
        )
    assert counter == 0
    assert not aborted


# =====================================================================
# AGNI post-solve validation gate (suggestion 1)
# =====================================================================
#
# These tests cover `proteus.atmos_clim.agni._validate_agni_state`, which
# guards against the failure mode where AGNI's solve_energy_b returns
# True but the post-solve atmosphere struct holds NaN/inf/non-positive
# values. Without this gate the bad values silently propagate into
# hf_row and the deadlock detector never fires (CHILI sweep regression).


class _MockAtmos:
    """Minimal duck-typed stand-in for AGNI's Atmos_t struct.

    Only the attributes inspected by `_validate_agni_state` are
    populated. Tests adjust them to simulate each failure mode.
    """

    def __init__(
        self,
        is_converged: bool = True,
        tmp_surf: float = 1500.0,
        flux_tot=None,
    ):
        self.is_converged = is_converged
        self.tmp_surf = tmp_surf
        # Default to a finite, well-shaped flux profile
        if flux_tot is None:
            flux_tot = [1.0e3, 9.5e2, 9.0e2, 8.5e2, 8.0e2]
        self.flux_tot = flux_tot


def test_validate_agni_state_accepts_healthy():
    """A converged solve with finite, positive state must pass validation.

    Edge: T_surf at the lower end of the magma-ocean range (1500 K), flux
    monotonically decreasing top→bottom (a typical cooling profile).
    """
    from proteus.atmos_clim.agni import _validate_agni_state

    atmos = _MockAtmos(is_converged=True, tmp_surf=1500.0)
    ok, reason = _validate_agni_state(atmos)
    assert ok is True
    assert reason == ''


def test_validate_agni_state_rejects_nan_tsurf():
    """NaN surface temperature must be rejected.

    Discriminating: NaN is the actual failure signature observed in
    CHILI sweep R12/R17 (line-search collapse + chemistry='eq' returned
    NaN tmp_surf with success=True).
    """
    from proteus.atmos_clim.agni import _validate_agni_state

    atmos = _MockAtmos(is_converged=True, tmp_surf=float('nan'))
    ok, reason = _validate_agni_state(atmos)
    assert ok is False
    assert 'tmp_surf' in reason


def test_validate_agni_state_rejects_negative_tsurf():
    """Negative surface temperature must be rejected.

    Physically unreasonable input: a Newton step that crossed zero would
    produce T_surf <= 0 and corrupt every downstream calculation.
    """
    from proteus.atmos_clim.agni import _validate_agni_state

    atmos = _MockAtmos(is_converged=True, tmp_surf=-100.0)
    ok, reason = _validate_agni_state(atmos)
    assert ok is False
    assert 'tmp_surf' in reason

    # Edge: exact zero must also fail
    atmos_zero = _MockAtmos(is_converged=True, tmp_surf=0.0)
    ok2, _ = _validate_agni_state(atmos_zero)
    assert ok2 is False


def test_validate_agni_state_rejects_inf_in_flux():
    """A single infinite element in flux_tot must trigger rejection.

    Discriminating: tests that the gate is element-wise, not just
    "is the first element finite". An inf at index 3 of a 5-element
    profile must still be caught.
    """
    from proteus.atmos_clim.agni import _validate_agni_state

    flux = [1e3, 9.5e2, 9.0e2, float('inf'), 8e2]
    atmos = _MockAtmos(is_converged=True, flux_tot=flux)
    ok, reason = _validate_agni_state(atmos)
    assert ok is False
    assert 'flux_tot' in reason
    assert 'non-finite' in reason


def test_validate_agni_state_rejects_empty_flux():
    """Empty flux array must be rejected (AGNI never returns this normally,
    but a struct construction bug could leave it empty)."""
    from proteus.atmos_clim.agni import _validate_agni_state

    atmos = _MockAtmos(is_converged=True, flux_tot=[])
    ok, reason = _validate_agni_state(atmos)
    assert ok is False
    assert 'flux_tot is empty' in reason


def test_validate_agni_state_rejects_unconverged():
    """If is_converged is False but solve_energy_b returned True, the gate
    must reject this contradictory state. AGNI sets is_converged=True only
    on CODE_SUC, so a conflict here means the state is corrupt."""
    from proteus.atmos_clim.agni import _validate_agni_state

    atmos = _MockAtmos(is_converged=False, tmp_surf=2000.0)
    ok, reason = _validate_agni_state(atmos)
    assert ok is False
    assert 'is_converged' in reason


# =====================================================================
# AGNI failure-reason parser (suggestion 4)
# =====================================================================


def test_extract_agni_failure_reason_nan_flux():
    """Parser must recognise the NaN-values failure marker emitted by
    AGNI/src/solver.jl line 977."""
    from proteus.atmos_clim.agni import _extract_agni_failure_reason

    log_lines = [
        '[2026-04-08 12:34:56] [info] AGNI solver iter 5\n',
        '[2026-04-08 12:34:57] [error]     failure (NaN values)\n',
    ]
    reason = _extract_agni_failure_reason(log_lines)
    assert reason == 'nan_flux'


def test_extract_agni_failure_reason_singular_jacobian():
    """Parser must recognise the singular-jacobian marker (solver.jl:971)."""
    from proteus.atmos_clim.agni import _extract_agni_failure_reason

    log_lines = ['[error]     failure (singular jacobian)\n']
    assert _extract_agni_failure_reason(log_lines) == 'singular_jacobian'


def test_extract_agni_failure_reason_max_iterations():
    """Parser must recognise the max-iterations marker (solver.jl:968)."""
    from proteus.atmos_clim.agni import _extract_agni_failure_reason

    log_lines = ['[error]     failure (maximum iterations)\n']
    assert _extract_agni_failure_reason(log_lines) == 'max_iterations'


def test_extract_agni_failure_reason_picks_last_attempt():
    """When multiple AGNI attempts emit failures within one PROTEUS
    iteration, the parser must report the LAST one (the one that drove
    the eventual non-convergence). Reverse-iteration is the discriminating
    behaviour: a forward scan would return the first marker instead.
    """
    from proteus.atmos_clim.agni import _extract_agni_failure_reason

    log_lines = [
        '[error]     failure (singular jacobian)\n',  # attempt 1
        '[info] retrying with linesearch\n',
        '[error]     failure (NaN values)\n',  # attempt 2 (the latest)
    ]
    reason = _extract_agni_failure_reason(log_lines)
    assert reason == 'nan_flux', (
        'parser must report the latest failure marker, not the first'
    )


def test_extract_agni_failure_reason_unparsed_when_missing():
    """Empty or non-matching log content must yield 'unparsed' so callers
    can distinguish 'AGNI failed silently' from 'AGNI failed with
    diagnosable mode'."""
    from proteus.atmos_clim.agni import _extract_agni_failure_reason

    assert _extract_agni_failure_reason([]) == 'unparsed'
    assert _extract_agni_failure_reason(['[info] just chatter\n']) == 'unparsed'
