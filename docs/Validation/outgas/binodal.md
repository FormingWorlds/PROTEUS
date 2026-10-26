# Validation: `src/proteus/outgas/binodal.py`

## What this page checks

This page checks how `proteus.outgas.binodal` splits H$_2$ between the atmosphere and the molten mantle with the miscibility model of Rogers, Young and Schlichting (2025). A weight $\sigma$ from that model sets the dissolved fraction: $\sigma$ of the H$_2$ goes into the melt and $1 - \sigma$ into the atmosphere. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the H$_2$ mole values of each reservoir are wrong. The case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends the case.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_binodal.py::test_h2_mole_total_uses_h2_molecular_mass_per_rogers2025` | Rogers, Young & Schlichting (2025), MNRAS, 544, 3496 (doi:10.1093/mnras/staf1940) | Pins the molar-mass conversion the binodal partition applies to every H2 reservoir against an independent computation using `scipy.constants.Avogadro` and `M(H2) = 2.016e-3` kg/mol. The kg-closure `atm + liquid + solid = total` is structurally trivial for the `sigma * T + (1 - sigma) * T + 0` assignment, so the discriminating assertion is the molar-mass pin (`rel=1e-12`): a regression mistaking `M(H2)` for the atomic mass `M(H) = 1.008e-3` kg/mol would inflate the mole total by a factor of two and be caught. |

## Moles of H2 from the molecular mass

**Why this check exists.** The partition converts the mass of H$_2$ in each reservoir to moles. The atomic mass of H, 1.008e-3 kg/mol, in place of the molecular mass of H$_2$ doubles the mole total without an error message.

**What it tests.** The test sets the miscibility weight to 0.4 and the H$_2$ budget to 5e17 kg. The mole total must equal the mass over $M(\mathrm{H_2})$ = 2.016e-3 kg/mol and the Avogadro constant, and the moles of the 3 reservoirs must sum to it. It also checks that the kg of the 3 reservoirs sum to the budget, that the reservoirs are not negative and that the solid holds no H$_2$.

**How we know the right answer.** The test computes the mole total independently, with `scipy.constants.Avogadro` and $M(\mathrm{H_2})$ = 2.016e-3 kg/mol.

**Why it matters.** The outgassing wrapper applies this partition (`apply_binodal_h2`) when `outgas.h2_binodal = true` and Zalmoxis `global_miscibility` is off.

**Known limit.** The config refuses `outgas.h2_binodal = true` (`_reject_enabled_binodal` in `src/proteus/config/_outgas.py`), so no run reaches this code now. The stored mole values are the mass over $M(\mathrm{H_2})$ and the Avogadro constant, so they are moles divided by 6.022e23.

**Known limit.** The kg closure, atmosphere + liquid + solid = total, holds by construction for any $\sigma$ between 0 and 1, so only the mole conversion tests something. The weight $\sigma$ itself comes from `zalmoxis.binodal` and is replaced by 0.4 here. No test compares this partition with the binodal handling of atmodeller.

??? info "Test details"
    - Reference: Rogers, Young & Schlichting (2025), MNRAS, 544, 3496 (doi:10.1093/mnras/staf1940); mole conversion with `scipy.constants.Avogadro` and M(H2) = 2.016e-3 kg/mol.
    - Tolerance: relative 1e-12 on the mole total and on both closures.
    - Test id: `tests/outgas/test_binodal.py::test_h2_mole_total_uses_h2_molecular_mass_per_rogers2025`
    - Command: `pytest tests/outgas/test_binodal.py::test_h2_mole_total_uses_h2_molecular_mass_per_rogers2025`

## Re-derivation note

The Rogers et al. (2025) binodal partitions H2 between the atmosphere
and the dissolved (liquid mantle) reservoir at the mass mixing-ratio set
by thermodynamic miscibility, not Henry's law:

```
sigma = rogers2025_suppression_weight(P_surf, T_magma, w_H2, w_sil)
H2_kg_liquid = sigma * H2_kg_total
H2_kg_atm    = (1 - sigma) * H2_kg_total
H2_kg_solid  = 0  # H2 does not partition into solid silicate
```

By construction the three reservoirs sum to `H2_kg_total` for any `sigma`
in \[0, 1], so the kg-closure is structurally trivial. The discriminating
check is instead the molar-mass conversion: the reference-pinned test patches
`rogers2025_suppression_weight` to a specific value (0.4), then verifies that
`H2_mol_total = H2_kg_total / M(H2) / N_av` uses the molecular mass
`M(H2) = 2.016e-3` kg/mol rather than the atomic mass `M(H)`.
