# Validation: `src/proteus/outgas/binodal.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.outgas.binodal` against the Rogers et al. (2025)
H2-MgSiO3 miscibility model.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/outgas/test_binodal.py::test_h2_mass_is_conserved_per_rogers2025_partition` | Rogers et al. (2025), preprint arXiv:2502.xxxxx, Section 3.2 | Pre-print | Pins mass conservation across the binodal partition: for any suppression weight `sigma` returned by `zalmoxis.binodal.rogers2025_suppression_weight`, the post-partition reservoirs (atm + liquid + solid) must sum to `H2_kg_total`. Asserts each reservoir is non-negative and bounded above by the total. |

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
in [0, 1]. The test patches `rogers2025_suppression_weight` to a specific
value (0.4) to exercise the partition arithmetic without depending on the
upstream Zalmoxis implementation.

## Pending verification

The cross-implementation comparison against atmodeller's binodal handling
is a separate work item, tracked under the CALLIOPE-vs-atmodeller science
verification handover.
