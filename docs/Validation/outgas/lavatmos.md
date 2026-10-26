# Validation: `src/proteus/outgas/lavatmos.py`

## What this page checks

This page checks the molar masses of the species in the LavAtmos rock-vapour coupling, `proteus.outgas.lavatmos`. LavAtmos computes the vapour above a lava surface, and the coupling converts its species between mass and moles. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If a molar mass is wrong, every mass of that species in the vapour is wrong. The case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends the case.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_lavatmos_run_lavatmos.py::test_molecular_weights_match_iupac_atomic_weight_table` | IUPAC atomic weights, as tabulated in `proteus.utils.constants.element_mmw` (source: https://iupac.qmul.ac.uk/AtWt/) | Pins the molar masses `species_lib` derives from FastChem formula strings against sums taken independently from the IUPAC table, converted from kg/mol to g/mol. Four cases span a light hydride (H2O, 18.015 g/mol), a rock-forming oxide (SiO2, 60.083 g/mol), a bare heavy metal (Fe, 55.845 g/mol), and the electron (5.4858e-4 g/mol), so no single wrong code path satisfies all four. |

## Molar masses from FastChem formulas

**Why this check exists.** The coupling builds each molar mass from a FastChem formula string. A misread stoichiometric count, a missing conversion from kg/mol to g/mol, or a wrong electron mass gives wrong species masses without an error message.

**What it tests.** `species_lib` holds a molar mass for each species. The test compares 4 cases with sums over the element table `element_mmw`: H$_2$O (18.015 g/mol), SiO$_2$ (60.083 g/mol), Fe (55.845 g/mol) and the electron (5.4858e-4 g/mol). It also checks that reading SiO$_2$ as SiO (44.084 g/mol) misses by more than 1 g/mol, that every molecule lies between 1 and 1000 g/mol, that the electron lies between 0 and 1e-3 g/mol, and the mass order `e- << H2O < Fe < SiO2`.

**How we know the right answer.** The molar mass of a molecule is the sum of the atomic masses of its atoms, from the IUPAC atomic weights that `proteus.utils.constants.element_mmw` tabulates.

**Why it matters.** `species_lib` sets the molar masses with which the coupling converts every vapour species.

**Known limit.** The test sums the same `element_mmw` table that the code uses, and compares the electron with the code's own `electron_molar_mass`. So a wrong table value fails only the 1 to 1000 g/mol window, and a wrong electron mass only the 0 to 1e-3 g/mol window.

??? info "Test details"
    - Reference: IUPAC atomic weights, as tabulated in `proteus.utils.constants.element_mmw` (source: https://iupac.qmul.ac.uk/AtWt/).
    - Tolerance: relative 1e-12 for the 4 molar masses; SiO misses SiO$_2$ by more than 1 g/mol; windows 1 to 1000 g/mol and 0 to 1e-3 g/mol.
    - Test id: `tests/outgas/test_lavatmos_run_lavatmos.py::test_molecular_weights_match_iupac_atomic_weight_table`
    - Command: `pytest tests/outgas/test_lavatmos_run_lavatmos.py::test_molecular_weights_match_iupac_atomic_weight_table`

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

The pinned test has 3 kinds of guards:

- **Stoichiometric coefficient**: reading `SiO2` as `SiO` yields 44.084 g/mol,
  16 g/mol away from the correct value, against a 1 g/mol threshold.
- **Unit conversion**: omitting the kg-to-g factor leaves 0.060 kg/mol and
  applying it twice gives 6.0e4, both outside the pinned `1 < w < 1000` g/mol
  window.
- **Electron special case**: a zeroed or formula-derived electron mass fails
  the `0 < w < 1e-3` g/mol bound, since any element sum is at least 1.008.

A mass ordering assertion (`e- << H2O < Fe < SiO2`) pins all four cases
relative to one another, so a systematic offset that preserved ratios would
still have to reproduce the absolute values.

## Mutation check

Mutations of `_fastchem_weight` and of the electron entry in `src/proteus/outgas/lavatmos.py` fail tests in this file: removing the kg-to-g conversion fails 5 tests, ignoring the stoichiometric count fails 3, and zeroing the electron mass fails 3. The unmutated source passes all 24 tests in the file.

## Known limits of this page

The rock-vapour composition itself has no check against published LavAtmos output. No test compares the FastChem-recombined vapour composition with the tabulated results of van Buchem et al. (2023), or the derived `fO2_vapourise_derived` with an independent buffer calculation. Whole-planet mass conservation does not hold while `outgas.vapourise` is enabled, by design, so mass-closure invariants are not available as validation checks on this path; see
[Model description](../../Explanations/model.md#whole-planet-mass-is-not-conserved-when-vapourisation-is-enabled).
