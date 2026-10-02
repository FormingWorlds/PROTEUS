# PROTEUS review detail

The detail behind the Review section of `AGENTS.md`: the whole-planet oxygen and element accounting, the initial-condition checks and the `hf_row` override pattern. Background: issue #677.

## Oxygen in the config

The `planet.elements.O_mode` details that `AGENTS.md` leaves out:

- `"ppmw"`: `O_kg_total = O_budget * 1e-6` times the volatile reservoir mass (`M_mantle` or `M_int`, per `planet.volatile_reservoir`).
- `"FeO_mantle_wt_pct"`: `O_kg_total = M_mantle * (wt% / 100) * (M_O / M_FeO)`. The mantle EOS density does not change; PALEOS keeps its built-in FeO content, so the mode only sets the volatile O budget in familiar units.

## Element aggregation sites

Every site that sums element masses includes oxygen; that keeps `M_atm` below `M_planet` at high H budgets. If one site sums over `gas_list` (oxygen included) while another sums over `element_list` with `if e == 'O': continue`, `M_atm` can exceed `M_planet`. Each site names its element list explicitly; a new `if e == 'O': continue` in any of them is a red flag. The sites:

1. Initial-budget population (`calc_target_elemental_inventories`, `_resolve_oxygen_budget`)
2. M_planet bookkeeping (`update_planet_mass`)
3. Structure dry-mass target (`load_zalmoxis_configuration`)
4. Escape rate distribution (`calc_unfract_fluxes`, `calc_new_elements`); O is in the unfractionated partitioning, so `sum(esc_rate_e) == esc_rate_total` to within rounding
5. Desiccation gate (`check_desiccation`)
6. First-call baseline (`M_vol_initial` in `run_escape`)

Sites 1 and 2 sum `vol_element_list + noble_gases` and leave the rock-vapour elements of `vap_element_list` out on purpose, because rock vapour enters the atmosphere without being debited from the interior; sites 3, 4 and 6 sum `element_list`; site 5 tests `vol_element_list + noble_gases` against the threshold and sums `element_list` for the escape balance, to match the site-6 baseline. The different lists are intended, not an asymmetry to repair.

## Mass-conservation check

Flag any change that weakens or removes `assert_mass_conservation(hf_row)`; both of its halves use the tolerance `atol_frac`. The main loop passes `require_atm_le_planet=False` when `outgas.vapourise = true`, and the `M_atm <= M_planet` half turns into a warning only while `M_vaps > 0`; with no vapour column it is enforced whatever the keyword. The intended non-conservation is described in `docs/Explanations/model.md` ("Whole-planet mass is not conserved when vapourisation is enabled"). Flag any change that widens `atol_frac`, disables the species-sum half or silences the warning.

## Initial-condition checks

When a user supplies a value that a solver derives again, a one-time check at the initial condition catches a mis-specification loudly instead of letting it corrupt the trajectory. The oxygen pattern (`check_ic_oxygen_budget`):

1. Stash the user value in `hf_row` under a sentinel key (`O_kg_user_ic`, the resolved `O_budget` in kg).
2. After the first solver call, compare the solver value (`O_kg_total` from CALLIOPE or atmodeller) with it.
3. Fail when the relative difference exceeds a threshold (50 % for O).
4. Reset the sentinel (-1.0) so later calls do not fire the check again.

The check runs only with `planet.fO2_source = 'user_constant'` and an `O_mode` other than `'ic_chemistry'`; it is called in the outgassing step of each iteration and fires once. The same pattern fits any user value with a solver-derived counterpart, for example `fO2_shift_IW` against the atmospheric chemistry it implies.

## hf_row temporary overrides

A module call that needs different boundary values saves and restores them, or the helpfile records the override instead of the planet state:

```python
saved = {k: hf_row[k] for k in overrides}
try:
    hf_row.update(overrides)
    result = call_submodule(hf_row)
finally:
    hf_row.update(saved)
```
