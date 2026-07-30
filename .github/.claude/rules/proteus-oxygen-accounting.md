# Whole-planet oxygen accounting

Read this before touching element budgets, `M_planet` bookkeeping, escape
partitioning, the desiccation gate, or anything that sets or consumes
`planet.elements.O_mode`. Originates from issue #677.

> **Discovery note.** PROTEUS keeps its Claude-Code rule files under
> `.github/.claude/rules/` (not the conventional repo-root `.claude/`, which is
> gitignored and so cannot be shared with collaborators). Claude does NOT
> auto-discover them at this path; `.github/copilot-instructions.md` names this
> file so readers and AI tooling know to load it.

## Config contract

Every config must declare an explicit `planet.elements.O_mode`. Four valid modes:

- `"ic_chemistry"`: defer the IC O budget to CALLIOPE's fO2-buffered
  equilibrium. Preserves pre-fix behaviour; backwards-compatible.
- `"ppmw"`, `"kg"`: parallel to the H/C/N/S modes; sets O_kg directly.
- `"FeO_mantle_wt_pct"`: alternative unit for petrologists. The number is
  interpreted as `O_kg = M_mantle * (wt% / 100) * (M_O / M_FeO)`. The mantle EOS
  density is NOT modified; PALEOS still assumes its built-in FeO content. The
  mode is a unit-of-convenience for setting the volatile-O budget in familiar
  terms.

## Design (D1A)

Under D1A, the chosen design, CALLIOPE / atmodeller chemistry is unchanged.
Oxygen is treated as a buffered element at the chemistry step but a tracked
element in PROTEUS-side mass accounting.

The asymmetry that previously let `M_atm > M_planet` at high H budgets is closed
by including O in:

- `M_ele`
- the Zalmoxis dry-mass subtraction
- the proportional escape distribution
- the desiccation gate

Escape includes O in the unfractionated partitioning so
`sum(esc_rate_e) == esc_rate_total` to within rounding.

## Runtime guards

- `assert_mass_conservation` in the main loop enforces `M_atm <= M_planet` every
  iteration. If a change weakens or removes it, push back: it is the safety net
  that catches O-skip reintroductions.
- `check_ic_oxygen_budget`, called once after the first outgas call, hard-fails
  on >50% divergence between the user-supplied O_budget and CALLIOPE's
  equilibrium value.

## Aggregation symmetry

All aggregation sites must use the same element set. A new `if e == 'O':
continue` skip in any of them is a red flag; it likely re-introduces the
asymmetry that issue #677 closed. The sites are enumerated under "Whole-element
aggregation symmetry" in [`proteus-code-review.md`](proteus-code-review.md).

## Sister rules

- [`.github/copilot-instructions.md`](../../copilot-instructions.md): repo-wide rules.
- [`proteus-code-review.md`](proteus-code-review.md): the aggregation-site list and review criteria.
