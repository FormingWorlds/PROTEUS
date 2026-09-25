# Whole-planet oxygen accounting

Read this before touching element budgets, `M_planet` bookkeeping, escape
partitioning, the desiccation gate, or anything that sets or consumes
`planet.elements.O_mode`. Background: issue #677.


## Config contract

`planet.elements.O_mode` defaults to `"ic_chemistry"`. The four modes:

- `"ic_chemistry"`: defer the IC O budget to CALLIOPE's fO2-buffered
  equilibrium.
- `"ppmw"`, `"kg"`: parallel to the H/C/N/S modes. `"kg"` sets the initial
  `O_kg_total` to `O_budget`; `"ppmw"` sets it to `O_budget * 1e-6` times the volatile
  reservoir mass (`M_mantle` or `M_int`, per `planet.volatile_reservoir`).
- `"FeO_mantle_wt_pct"`: alternative unit for petrologists. The number is
  interpreted as `O_kg_total = M_mantle * (wt% / 100) * (M_O / M_FeO)`. The mantle EOS
  density is NOT modified; PALEOS still assumes its built-in FeO content. The
  mode is a unit-of-convenience for setting the volatile-O budget in familiar
  terms.

## Design (D1A)

Under D1A, the chosen design, CALLIOPE / atmodeller chemistry is unchanged.
Oxygen is treated as a buffered element at the chemistry step but a tracked
element in PROTEUS-side mass accounting.

`M_atm` stays below `M_planet` at high H budgets because O is included in:

- `M_ele`
- the Zalmoxis dry-mass subtraction
- the proportional escape distribution
- the desiccation gate

Escape includes O in the unfractionated partitioning so
`sum(esc_rate_e) == esc_rate_total` to within rounding.

## Runtime guards

- `assert_mass_conservation` in the main loop enforces `M_atm <= M_planet` every
  iteration. If a change weakens or removes it, push back: it is the safety net
  that catches O-skip reintroductions. The exception is `outgas.vapourise =
  true`: rock vapourisation deliberately moves rock mass into `M_atm` without
  subtracting it from the interior, and rock-vapour elements dilute the escape
  outflow without being debited from a tracked reservoir. In that mode the main
  loop passes `require_atm_le_planet=False`, which disables that one half and
  replaces it with a logged warning whenever the excess over `M_planet` is
  larger than `M_vaps`, i.e. larger than vapourisation explains. The relaxation
  applies only while `M_vaps > 0`; with no vapour column present the invariant
  is enforced regardless of the keyword. The other half of the check
  (`M_vol_atm` equals the summed masses of the volatile species, rock vapour
  excluded) stays enforced
  in both modes, and `atol_frac` is never loosened. Treat that non-conservation
  as intentional, not a bug to repair; see `docs/Explanations/model.md`,
  "Whole-planet mass is not conserved when vapourisation is enabled".
- `check_ic_oxygen_budget` compares the user oxygen mass (`O_kg_user_ic`, the
  resolved `O_budget` in kg) with the chemistry result of CALLIOPE or
  atmodeller (`O_kg_total`) and
  hard-fails above 50 % divergence. It runs only with
  `planet.fO2_source = 'user_constant'` and an `O_mode` other than
  `'ic_chemistry'`; it is called in the outgassing step of each iteration and fires once, because it
  resets `O_kg_user_ic` to the -1.0 sentinel.

## Aggregation symmetry

Every aggregation site includes oxygen. A new `if e == 'O': continue` skip in
any of them is a red flag: it lets `M_atm` exceed `M_planet` again. The mass
sites leave rock-vapour elements out on purpose; that is not an asymmetry to
repair. The sites are listed under "Whole-element aggregation symmetry" in
[`code-review.md`](code-review.md).
