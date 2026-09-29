# Validation: `src/proteus/outgas/compaction.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor
`proteus.outgas.compaction`, the melt drainage of the front scheme.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_compaction.py::test_mobility_reproduces_the_interior_solver_in_every_regime` | Cross-implementation check against the interior solver: `aragog.eos.entropy_phase.EntropyPhaseEvaluator.relative_velocity` (fwl-aragog), the three-regime permeability of Bower et al. (2018), section 2.1 | Calls the solver's own separation velocity on a stand-in mush and backs the mobility out of it, for three grain sizes and porosities spanning all three regimes and both blends. Agreement is `rtol=2e-4` from a porosity of 0.05 up and `1.5e-3` below, where the solver's soft porosity clip (width 1e-3) lifts the porosity by about `1e-6 / (4 phi)`. Also pins the drainage velocity as the solver's velocity times `(1 - phi)`. |
| `tests/outgas/test_compaction.py::test_drainage_reproduces_the_published_scaling_table` | Table 2 of the PROTEUS compaction note (Lichtenberg, 21 September 2026) | With the `(1 - phi)` matrix fraction removed, as in the note, the drainage integral reproduces the percolation time, the matrix time and the trapped fractions at 70 and 5 cm/yr for all five cases, to the table's printed precision (0.006 on the two-decimal fractions). With the factor, the percolation time is longer by exactly `1 / (1 - phi_c)` and the retained melt never lower. |

## Analytic limit checked alongside

`test_drainage_is_limited_by_the_slower_of_the_two_processes` pins the
matrix-limited case to its closed form `phi_c exp(-t_res / tau_s)` at
`rel=1e-6`.
