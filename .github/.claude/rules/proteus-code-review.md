---
description: PROTEUS-specific code review criteria for the generator-evaluator pattern. Applies domain expertise (physics, coupling, pitfall patterns) to all code review in this repo.
---

# PROTEUS Code Review Criteria

When reviewing PROTEUS code (either your own or via code-reviewer agents), apply these domain-specific checks in addition to standard code quality review.

> **Discovery note.** PROTEUS keeps its Claude-Code rule files under `.github/.claude/rules/` (not the conventional repo-root `.claude/`) so they can be tracked in git and shared across collaborators. Claude does NOT auto-discover them at this path; the repo-root `CLAUDE.md` (symlinked to `.github/copilot-instructions.md`) names this file and `proteus-tests.md` explicitly. **Before opening any review pass, read both this file and `proteus-tests.md`.**

## Physics plausibility

- Temperature must be positive everywhere (Kelvin). Flag any code path where T could reach zero or go negative.
- Pressure must be positive and monotonically increasing with depth in interior profiles.
- Mass fractions must sum to 1.0. Flag any volatile partitioning code that doesn't enforce or verify normalization.
- Escape rates must not exceed total atmospheric mass. Flag unbounded escape calculations.
- Outgassing rates must be non-negative.
- Energy fluxes at module boundaries (atmosphere-interior, interior-core) must be consistent. If two modules independently compute the same flux, verify they agree.
- Stefan-Boltzmann: F = sigma * T^4. When reviewing radiative flux code, check the exponent is 4, not 3 or 5.

## Unit convention boundaries

PROTEUS has a split unit convention:
- **Config values**: "human" units (M_sun, bar, Gyr, K)
- **Internal hf_row values**: SI-ish (kg, Pa, yr, K)
- **Submodule APIs**: may expect either convention

When reviewing code that passes values between config, hf_row, and submodule calls, verify the unit is correct at each boundary. The ZEPHYRUS stellar-mass bug (audit 1.1) was exactly this class of error.

## Config mutability

The `Config` attrs object must not be mutated at runtime. Flag any code that sets `config.X.Y = value` outside of config initialization. Use local variables instead. Known violation: Zalmoxis sets `config.orbit.module = 'dummy'`; this is a known debt, not a pattern to replicate.

## Coupling parameter echo-back

When module A computes a quantity self-consistently (e.g., Zalmoxis computes core mass from EOS) and module B has its own internal model for the same quantity (e.g., SPIDER's `-rho_core`), module B's output can overwrite A's value in hf_row. Review any new submodule integration for this pattern.

## Whole-element aggregation symmetry (issue #677 lesson)

When reviewing code that aggregates element masses, all four sites of the cycle must use the same element set:

1. Initial-budget population (`calc_target_elemental_inventories`, `_resolve_oxygen_budget`)
2. M_planet bookkeeping (`update_planet_mass`)
3. Structure dry-mass target (`load_zalmoxis_configuration`)
4. Escape rate distribution (`calc_unfract_fluxes`, `calc_new_elements`)
5. Desiccation gate (`check_desiccation`)
6. First-call baseline (`M_vol_initial` in `run_escape`)

Issue #677 surfaced when one site (M_atm via `gas_list` sum) implicitly included oxygen mass while every other site (M_ele via `element_list` with `if e == 'O': continue`) excluded it. The fix is to make the element set explicit and consistent across all aggregation sites. A new `if e == 'O': continue` skip in any of these sites is a red flag during review; it likely re-introduces the asymmetry.

The runtime invariant `assert_mass_conservation(hf_row)` is called at the end of every iteration to hard-fail on a regression. If a review finds someone has weakened or removed that assertion, push back: it's the safety net that catches future O-skip reintroductions.

## IC consistency checks at unit boundaries

When a user supplies a value via config that gets re-derived by a downstream solver (e.g., O_budget from `planet.elements.O_mode` vs CALLIOPE's IC equilibrium), a one-shot reconciliation check at IC catches mis-specifications loudly rather than letting them silently corrupt the trajectory. Pattern from issue #677:

1. Stash the user-supplied value in `hf_row` under a sentinel-style key (e.g., `O_kg_user_ic`).
2. After the first solver call, compare the solver-derived value against the user budget.
3. Hard-fail if relative divergence exceeds a threshold (50% for O; threshold can be tuned per case).
4. Flip the sentinel so subsequent init-stage calls don't re-fire the check.

Applies to any future user-specified quantity that has a solver-derived equivalent. Examples worth retro-fitting: T_magma_init vs SPIDER's IC entropy, fO2_shift_IW vs the atmospheric chemistry it implies, surface gravity vs Zalmoxis's structure output.

## hf_row temporary overrides

When overriding hf_row values to pass different boundary conditions to a submodule, require a save/restore pattern:
```python
saved = {k: hf_row[k] for k in keys_to_override}
try:
    hf_row[k] = override_value
    result = call_submodule(hf_row)
finally:
    hf_row.update(saved)
```
Without restore, the helpfile CSV records override values instead of true planet state.

## Cross-module constant duplication

Physical constants (G, year length, solar mass, Stefan-Boltzmann) are defined independently in PROTEUS, CALLIOPE, ZEPHYRUS, and other submodules. When reviewing code that uses physical constants, check which definition is used and whether it matches the expected value.

## PALEOS / EOS tables

- SPIDER needs P-S tables (phase-specific S ranges, complete rectangles, uniform P spacing).
- Aragog needs P-T tables (full rectangular grid, identical for solid and melt).
- Phase-filtering P-T tables for Aragog creates irregular grids that cause scipy to use slow unstructured interpolation. Flag any table generation that filters by phase before writing Aragog tables.

## Interior-atmosphere coupling timing

The main loop advances Time before the atmosphere step runs. Any function comparing hf_row (current, time-advanced) with hf_all.iloc[-1] (previous) must get argument ordering right.

## Validator liveness

attrs validators can silently become dead code if they compare a dataclass instance against a primitive (e.g., `StopEscape is False`). When reviewing validators, check that both valid and invalid inputs are tested.

## Test marker discipline

Every test file must begin with a module-level `pytestmark = [pytest.mark.<tier>, pytest.mark.timeout(<budget>)]` (unit/30 s, smoke/60 s, integration/300 s, slow/3600 s). Per-function markers are additive but do not replace the module-level marker; CI runs `pytest -m "unit and not skip and not slow and not integration"` and any file missing the tier marker ships untested.

## Test quality (cross-reference)

Test-content rules (anti-happy-path, discriminating-value guards, physics-invariant tiering, `physics_invariant` / `reference_pinned` certification markers, adversarial-review trigger, mocking discipline, `importorskip` + module-constant-monkeypatch traps) live in [`proteus-tests.md`](proteus-tests.md). When reviewing tests, apply both files: this one for marker discipline and review-pass gate, the deep-dive for the content contract.

## Sister rules (cross-link)

- [`.github/copilot-instructions.md`](../../copilot-instructions.md) "Testing Standards" -- high-level rules visible to all readers. Repo-root `CLAUDE.md` is a symlink to this file.
- [`proteus-tests.md`](proteus-tests.md) -- test quality deep-dive; the canonical source for anti-happy-path patterns and the validation certification markers.
