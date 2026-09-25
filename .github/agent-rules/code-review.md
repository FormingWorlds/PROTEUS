# PROTEUS Code Review Criteria

The detail behind the Review section of `AGENTS.md`. Apply these domain checks in addition to general code quality review.


## Physics plausibility

- Temperature must be positive everywhere (Kelvin). Flag any code path where T could reach zero or go negative.
- Pressure must be positive and monotonically increasing with depth in interior profiles.
- Mass fractions must sum to 1.0. Flag any volatile partitioning code that doesn't enforce or verify normalization.
- The mass escaped in one step must stay within its reservoir; `limit_escape_step` in `src/proteus/escape/wrapper.py` caps it at `ESCAPE_STEP_MAX_FRAC` of the escapable reservoir. Flag escape calculations that bypass the cap.
- Outgassing rates must be non-negative.
- Energy fluxes at module boundaries (atmosphere-interior, interior-core) must be consistent. If two modules independently compute the same flux, verify they agree.
- Stefan-Boltzmann: F = sigma * T^4. When reviewing radiative flux code, check the exponent is 4, not 3 or 5.

## Unit convention boundaries

PROTEUS has a split unit convention:
- **Config values**: the unit each key states in `docs/Reference/config/` (for example stellar mass in M_sun, planet mass in M_earth, stellar age in Gyr)
- **Internal hf_row values**: SI, except atmospheric pressure in bar and model time in years (`GetHelpfileKeys` in `src/proteus/utils/coupler.py` states each key; interior pressures such as `P_cmb` are in Pa and orbital periods in s)
- **Submodule APIs**: may expect either convention

When reviewing code that passes values between config, hf_row, and submodule calls, verify the unit is correct at each boundary. A stellar mass passed to ZEPHYRUS in the wrong unit is an example of this class of error.

## Config mutability

The `Config` attrs object must not be mutated at runtime. Flag any code that sets `config.X.Y = value` outside of config initialization. Use local variables instead. Known violation: Zalmoxis sets `config.orbit.module = 'dummy'`; this is a known debt, not a pattern to replicate.

## Coupling parameter echo-back

When module A computes a quantity self-consistently (e.g., Zalmoxis computes core mass from EOS) and module B has its own internal model for the same quantity (e.g., SPIDER's `-rho_core`), module B's output can overwrite A's value in hf_row. Review any new submodule integration for this pattern.

## Whole-element aggregation symmetry

When reviewing code that aggregates element masses, every site of the cycle must include oxygen. Sites 1 and 2 sum `vol_element_list + noble_gases` and leave the rock-vapour elements of `vap_element_list` out on purpose, because rock vapour enters the atmosphere without being debited from the interior; sites 3, 4 and 6 sum `element_list`; site 5 tests `vol_element_list + noble_gases` against the threshold and sums `element_list` for the escape balance, to match the site-6 baseline. The sites:

1. Initial-budget population (`calc_target_elemental_inventories`, `_resolve_oxygen_budget`)
2. M_planet bookkeeping (`update_planet_mass`)
3. Structure dry-mass target (`load_zalmoxis_configuration`)
4. Escape rate distribution (`calc_unfract_fluxes`, `calc_new_elements`)
5. Desiccation gate (`check_desiccation`)
6. First-call baseline (`M_vol_initial` in `run_escape`)

If one site sums over `gas_list` (oxygen included) while another sums over `element_list` with `if e == 'O': continue`, `M_atm` can exceed `M_planet` (issue #677). Each site names its element list explicitly and includes oxygen; a new `if e == 'O': continue` in any of them is a red flag.

The runtime invariant `assert_mass_conservation(hf_row)` runs after the outgassing step of every iteration to hard-fail on a regression. If a review finds someone has weakened or removed that assertion, push back: it's the safety net that catches future O-skip reintroductions.

The one sanctioned relaxation is `require_atm_le_planet=False`, which the main loop passes when `outgas.vapourise = true`. It disables the `M_atm <= M_planet` half only, replaces it with a warning raised when the excess over `M_planet` exceeds `M_vaps`, and leaves the `M_vol_atm` species-sum half and the `atol_frac` tolerance untouched. It applies only while `M_vaps > 0`. Flag any change that widens `atol_frac`, that disables the species-sum half, or that silences the warning.

## IC consistency checks at unit boundaries

When a user supplies a value via config that gets re-derived by a downstream solver (e.g., O_budget from `planet.elements.O_mode` vs CALLIOPE's IC equilibrium), a one-shot reconciliation check at IC catches mis-specifications loudly rather than letting them silently corrupt the trajectory. The pattern (oxygen budget, issue #677):

1. Stash the user-supplied value in `hf_row` under a sentinel-style key (e.g., `O_kg_user_ic`).
2. After the first solver call, compare the solver-derived value against the user budget.
3. Hard-fail if relative divergence exceeds a threshold (50% for O; threshold can be tuned per case).
4. Flip the sentinel so subsequent init-stage calls don't re-fire the check.

Applies to any future user-specified quantity that has a solver-derived equivalent. Example worth retro-fitting: `fO2_shift_IW` against the atmospheric chemistry it implies.

## hf_row temporary overrides

When overriding hf_row values to pass different boundary conditions to a submodule, require a save/restore pattern:
```python
saved = {k: hf_row[k] for k in overrides}
try:
    hf_row.update(overrides)
    result = call_submodule(hf_row)
finally:
    hf_row.update(saved)
```
Without restore, the helpfile CSV records override values instead of true planet state.

## Cross-module constant duplication

Physical constants (G, year length, solar mass, Stefan-Boltzmann) are defined independently in PROTEUS, CALLIOPE, ZEPHYRUS, and other submodules. When reviewing code that uses physical constants, check which definition is used and whether it matches the expected value.

## PALEOS / EOS tables

- SPIDER and the Aragog entropy solver read P-S tables (phase-specific S ranges, complete rectangles, uniform P spacing).
- Aragog also reads P-T tables: phase-specific ones generated from the PALEOS-2phase tables when they exist, one shared table otherwise. Each must be a full rectangular grid.
- Removing the points of the other phase from a P-T table leaves an irregular grid, on which scipy falls back to slow unstructured interpolation. Flag any table generation that filters points by phase before writing Aragog tables.

## Interior-atmosphere coupling timing

The main loop advances Time before the atmosphere step runs. Any function comparing hf_row (current, time-advanced) with hf_all.iloc[-1] (previous) must get argument ordering right.

## Validator liveness

attrs validators can silently become dead code if they compare a dataclass instance against a primitive (e.g., `StopEscape is False`). When reviewing validators, check that both valid and invalid inputs are tested.
