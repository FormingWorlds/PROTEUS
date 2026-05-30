# Validation: `src/proteus/outgas/binodal.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.outgas.binodal` against the Rogers, Young &
Schlichting (2025) H2-MgSiO3 miscibility model.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_binodal.py::test_h2_mole_total_uses_h2_molecular_mass_per_rogers2025` | Rogers, Young & Schlichting (2025), MNRAS, 544, 3496 (doi:10.1093/mnras/staf1940) | Pins the molar-mass conversion the binodal partition applies to every H2 reservoir against an independent computation using `scipy.constants.Avogadro` and `M(H2) = 2.016e-3` kg/mol. The kg-closure `atm + liquid + solid = total` is structurally trivial for the `sigma * T + (1 - sigma) * T + 0` assignment, so the discriminating assertion is the molar-mass pin (`rel=1e-12`): a regression mistaking `M(H2)` for the atomic mass `M(H) = 1.008e-3` kg/mol would inflate the mole total by a factor of two and be caught. |

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
in [0, 1], so the kg-closure is structurally trivial. The discriminating
check is instead the molar-mass conversion: the reference-pinned test patches
`rogers2025_suppression_weight` to a specific value (0.4), then verifies that
`H2_mol_total = H2_kg_total / M(H2) / N_av` uses the molecular mass
`M(H2) = 2.016e-3` kg/mol rather than the atomic mass `M(H)`.

## Pending verification

The cross-implementation comparison against atmodeller's binodal handling
is a separate work item, tracked under the CALLIOPE-vs-atmodeller science
verification handover.
