# Validation: `src/proteus/outgas/lavatmos.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.outgas.lavatmos`, the LavAtmos rock-vapourisation
coupling.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_lavatmos_run_lavatmos.py::test_molecular_weights_match_iupac_atomic_weight_table` | IUPAC atomic weights, as tabulated in `proteus.utils.constants.element_mmw` (source: https://iupac.qmul.ac.uk/AtWt/) | Pins the molar masses `species_lib` derives from FastChem formula strings against sums taken independently from the IUPAC table, converted from kg/mol to g/mol. Four cases span a light hydride (H2O, 18.015 g/mol), a rock-forming oxide (SiO2, 60.083 g/mol), a bare heavy metal (Fe, 55.845 g/mol), and the electron (5.4858e-4 g/mol), so no single wrong code path satisfies all four. |

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

The pinned test carries three guard classes:

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

The guards were confirmed by mutating `_fastchem_weight` and the electron
injection in `src/proteus/outgas/lavatmos.py` and observing test failures:
removing the kg-to-g conversion fails 5 tests, ignoring the stoichiometric
count fails 3, and zeroing the electron mass fails 3. The unmutated source
passes all 24 tests in the file.

## Pending verification

The rock-vapour composition itself is not yet pinned against published
LavAtmos output. A cross-check of the FastChem-recombined vapour composition
against van Buchem et al. (2023) tabulated results, and of the derived
`fO2_vapourise_derived` against an independent buffer calculation, remain open
work items. Note also that whole-planet mass conservation does not hold while
`outgas.vapourise` is enabled by design, so mass-closure invariants are not
available as validation checks on this path; see
[Model description](../../Explanations/model.md#whole-planet-mass-is-not-conserved-when-vapourisation-is-enabled).
