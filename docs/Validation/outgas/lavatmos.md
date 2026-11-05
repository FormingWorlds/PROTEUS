# Validation: `src/proteus/outgas/lavatmos.py`

## What this page checks

This page checks the molar masses of the species in the LavAtmos rock-vapour coupling, `proteus.outgas.lavatmos`. LavAtmos computes the vapour above a lava surface, and the coupling converts its species between mass and moles. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If a molar mass is wrong, every mass of that species in the vapour is wrong. The case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends the case.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_lavatmos_run_lavatmos.py::test_molecular_weights_match_iupac_atomic_weight_table` | IUPAC atomic weights, as tabulated in `proteus.utils.constants.element_mmw` (source: https://iupac.qmul.ac.uk/AtWt/) | Pins the molar masses `species_lib` derives from FastChem formula strings against sums taken from the IUPAC table, converted from kg/mol to g/mol. Four cases span a light hydride (H2O, 18.015 g/mol), a rock-forming oxide (SiO2, 60.083 g/mol), a bare heavy metal (Fe, 55.845 g/mol), and the electron (5.4858e-4 g/mol), so no single wrong code path satisfies all four. |

## Molar masses from FastChem formulas

**Why this check exists.** The coupling builds each molar mass from a FastChem formula string. A misread stoichiometric count, a missing conversion from kg/mol to g/mol, or a zeroed electron mass gives wrong species masses without an error message.

**What it tests.** `species_lib` holds a molar mass for each species. The test compares 4 cases with sums over the element table `element_mmw`: H$_2$O (18.015 g/mol), SiO$_2$ (60.083 g/mol), Fe (55.845 g/mol) and the electron (5.4858e-4 g/mol). It also checks that reading SiO$_2$ as SiO (44.084 g/mol) misses by more than 1 g/mol, that H$_2$O, SiO$_2$ and Fe lie between 1 and 1000 g/mol, that the electron lies between 0 and 1e-3 g/mol, and the mass order `e- << H2O < Fe < SiO2`.

**How we know the right answer.** The molar mass of a molecule is the sum of the atomic masses of its atoms, from the IUPAC atomic weights that `proteus.utils.constants.element_mmw` tabulates.

**Why it matters.** `species_lib` sets the molar masses with which the coupling converts every vapour species.

**Known limit.** The test sums the same `element_mmw` table that the code uses, and compares the electron with the code's own `electron_molar_mass`. So a wrong table value fails only the 1 to 1000 g/mol window, and a wrong electron mass only the 0 to 1e-3 g/mol window.

![Molar masses in species_lib against sums over the element table](../../assets/validation/outgas_lavatmos_molar_masses.svg#only-light){ width="80%" }
![Molar masses in species_lib against sums over the element table](../../assets/validation/outgas_lavatmos_molar_masses_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the molar mass summed from the element table, and the y-axis the molar mass in `species_lib`, both on log scales. For the electron, the x value is the code constant `electron_molar_mass`. The filled circles are the 4 asserted species (the electron, H$_2$O, Fe and SiO$_2$), and the line is 1:1. The hollow square is SiO$_2$ read as SiO, 44.084 g/mol; the test requires it to lie more than 1 g/mol from the SiO$_2$ value. A pass means that the 4 circles lie on the line.

??? info "Test details"
    - Reference: IUPAC atomic weights, as tabulated in `proteus.utils.constants.element_mmw` (source: https://iupac.qmul.ac.uk/AtWt/).
    - Tolerance: relative 1e-12 for the 4 molar masses; SiO misses SiO$_2$ by more than 1 g/mol; windows 1 to 1000 g/mol and 0 to 1e-3 g/mol.
    - Test id: `tests/outgas/test_lavatmos_run_lavatmos.py::test_molecular_weights_match_iupac_atomic_weight_table`
    - Command: `pytest -m reference_pinned tests/outgas/test_lavatmos_run_lavatmos.py --validation-plots`

## Re-derivation note

`_fastchem_weight` decomposes a FastChem formula string into an element-count
mapping via `mol_to_ele`, then sums the per-element molar masses weighted by
their stoichiometric counts:

```
weight[g/mol] = sum(count_el * element_mmw[el] for el, count in atoms) * 1000
```

The factor 1000 converts the table's kg/mol to the g/mol that FastChem and
LavAtmos expect. Free electrons cannot be decomposed into elements, so the
library injects `electron_molar_mass` for the `e-` entry directly rather than
routing it through the formula sum.

The test also asserts the order `e- < H2O < Fe < SiO2`. The 1e-12 pins imply it.

## Known limits of this page

The rock-vapour composition itself has no check against published LavAtmos output. No test compares the FastChem-recombined vapour composition with the tabulated results of van Buchem et al. (2023), or the derived `fO2_vapourise_derived` with an independent buffer calculation. Whole-planet mass conservation does not hold while `outgas.vapourise` is enabled, by design, so mass-closure invariants are not available as validation checks on this path; see
[Model description](../../Explanations/model.md#whole-planet-mass-is-not-conserved-when-vapourisation-is-enabled).
