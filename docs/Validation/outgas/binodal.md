# Validation: `src/proteus/outgas/binodal.py`

## What this page checks

This page checks how `proteus.outgas.binodal` splits H$_2$ between the reservoirs of the planet, the atmosphere, the molten mantle and the solid, with the miscibility model of Rogers, Young and Schlichting (2025). Miscibility is the ability of H$_2$ and silicate melt to mix into one phase; the binodal is the boundary in pressure, temperature and composition where they separate into 2 phases. A weight $\sigma$ from that model sets the dissolved fraction: $\sigma$ of the H$_2$ goes into the melt and $1 - \sigma$ into the atmosphere. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the H$_2$ mole values of each reservoir are wrong. The case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends the case.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_binodal.py::test_h2_mole_total_uses_h2_molecular_mass_per_rogers2025` | Rogers, Young & Schlichting (2025), MNRAS, 544, 3496 (doi:10.1093/mnras/staf1940) | Pins the molar mass in the conversion the binodal partition applies to every H2 reservoir, against the same division with `scipy.constants.Avogadro` and `M(H2) = 2.016e-3` kg/mol. The stored mole values are moles divided by the Avogadro constant; the test does not check this unit. The kg-closure `atm + liquid + solid = total` is structurally trivial for the `sigma * T + (1 - sigma) * T + 0` assignment, so the discriminating assertion is the molar-mass pin (`rel=1e-12`): a regression mistaking `M(H2)` for the atomic mass `M(H) = 1.008e-3` kg/mol would inflate the mole total by a factor of two and be caught. |

## Moles of H2 from the molecular mass

**Why this check exists.** The partition converts the mass of H$_2$ in each reservoir to moles. The atomic mass of H, 1.008e-3 kg/mol, in place of the molecular mass of H$_2$ doubles the mole total without an error message.

**What it tests.** The test sets the miscibility weight to 0.4 and the H$_2$ budget to 5e17 kg. The mole total must equal the mass over $M(\mathrm{H_2})$ = 2.016e-3 kg/mol and the Avogadro constant, and the moles of the 3 reservoirs must sum to it. Mass over molar mass already gives moles, so the stored values are moles divided by 6.022e23, against keys such as `H2_mol_total` that name moles. The test asserts the same formula. It fails with the atomic mass of H, but it passes with this wrong unit. It also checks that the kg of the 3 reservoirs sum to the budget, that the reservoirs are not negative and that the solid holds no H$_2$.

**How we know the right answer.** The test repeats the division with `scipy.constants.Avogadro` and $M(\mathrm{H_2})$ = 2.016e-3 kg/mol, so it checks the molar mass in the code but not the unit of the result.

**Why it matters.** The outgassing wrapper applies this partition (`apply_binodal_h2`) to set the H$_2$ mass of the atmosphere and the melt when `outgas.h2_binodal = true` and Zalmoxis `global_miscibility` is off.

**Known limit.** The config refuses `outgas.h2_binodal = true` (`_reject_enabled_binodal` in `src/proteus/config/_outgas.py`). No run reaches this code now. The kg closure, atmosphere + liquid + solid = total, holds by construction for any $\sigma$ between 0 and 1, so only the mole conversion tests something; this test has no assert on the split itself; the unit test `test_sigma_partitions_h2_linearly` checks it. The weight $\sigma$ itself comes from `zalmoxis.binodal` and is replaced by 0.4 here. No test compares this partition with the binodal handling of atmodeller.

![H2 mass in the 3 reservoirs and the H2 mole total after the binodal partition](../../assets/validation/outgas_binodal_h2_moles.svg#only-light){ width="80%" }
![H2 mass in the 3 reservoirs and the H2 mole total after the binodal partition](../../assets/validation/outgas_binodal_h2_moles_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The left panel stacks the asserted H$_2$ masses of the gas, the melt and the solid (0) against the budget of 5e17 kg (dashed). The right panel shows the code quantity `H2_mol_total`, the mass divided by the H$_2$ molar mass and by the Avogadro constant. Its circle is the asserted code value, and its hollow square the same division done by the test. The grey band is the asserted bracket, $3\times10^{-4}$ to $5\times10^{-4}$. The cross is the value with the H atomic mass, which the test rejects. A pass means that the stack reaches the budget line and the circle sits in the square inside the band.

??? info "Test details"
    - Reference: Rogers, Young & Schlichting (2025), MNRAS, 544, 3496 (doi:10.1093/mnras/staf1940); mole conversion with `scipy.constants.Avogadro` and M(H2) = 2.016e-3 kg/mol.
    - Tolerance: relative 1e-12 on the mole total and on both closures.
    - Test id: `tests/outgas/test_binodal.py::test_h2_mole_total_uses_h2_molecular_mass_per_rogers2025`
    - Command: `pytest -m reference_pinned tests/outgas/test_binodal.py --validation-plots`

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

`tests/outgas/test_binodal.py::test_sigma_partitions_h2_linearly` checks this split. By construction the three reservoirs sum to `H2_kg_total` for any `sigma`
in \[0, 1], so the kg-closure is structurally trivial. The discriminating
check of `tests/outgas/test_binodal.py::test_h2_mole_total_uses_h2_molecular_mass_per_rogers2025` is instead the molar-mass conversion: the reference-pinned test patches
`rogers2025_suppression_weight` to a specific value (0.4), then verifies that
`H2_mol_total = H2_kg_total / M(H2) / N_av` uses the molecular mass
`M(H2) = 2.016e-3` kg/mol rather than the atomic mass `M(H)`.
