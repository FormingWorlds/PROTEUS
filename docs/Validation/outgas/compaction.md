# Validation: `src/proteus/outgas/compaction.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor
`proteus.outgas.compaction`, the melt drainage of the front scheme.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_compaction.py::test_mobility_reproduces_the_interior_solver_in_every_regime` | Cross-implementation check against the interior solver: `aragog.eos.entropy_phase.EntropyPhaseEvaluator.relative_velocity` (fwl-aragog), the three-regime permeability of Bower et al. (2018), section 2.1 | Calls the solver's own separation velocity on a stand-in mush and backs the mobility out of it, for three grain sizes and porosities spanning all three regimes and both blends. Agreement is `rtol=2e-4` from a porosity of 0.05 up and `1.5e-3` below, where the solver's soft porosity clip (width 1e-3) lifts the porosity by about `1e-6 / (4 phi)`. The comparison runs under each `separation_buoyancy`, against the mixture (McKenzie 1984, Eq. 3.7) and against the solid (Abe 1995, Eq. 40), and also pins the drainage velocity as the solver's velocity under the same buoyancy, so the drainage uses the law the separation velocity uses. The modes differ by `1 - phi` at `phi = 0.3` and agree within `1e-3` at `phi = 0.99`, where a grain settles alone. |
| `tests/outgas/test_compaction.py::test_the_lever_rule_porosity_is_the_solvers_where_the_melt_is_lighter` | Cross-implementation check against the interior solver's porosity, `aragog.eos.entropy_phase.EntropyPhaseEvaluator._porosity_from` (fwl-aragog 26.10.03), which it writes as `porosity_b` | On a two-phase mush with the melt lighter, the lever rule the trapping step applies to the solver's basic-node densities equals the solver's porosity to `atol=2e-5`, the soft clip of `2.5e-7 / phi` at each edge. Where the melt is the denser phase the solver's porosity reads below 1e-3, as its docstring warns, while the lever rule returns the melt volume, and a fully molten node takes its melt fraction, 1, where the lever rule on its single-phase density gives 0.95. |
| `tests/outgas/test_compaction.py::test_drainage_matches_an_independent_quadrature_over_five_martian_fronts` | Analytic limits and a reference calculation: the closed-form timescales `tau_D = L eta_m / (drho g F(phi))`, with `F` the solver's mobility under the default `separation_buoyancy = 'mixture'`, `(5/7) (1 - phi) a^2 phi^4.5` in the Rumpf-Gupte regime, and `tau_s = mu_s / (drho g L)`; the separable form of the drainage equation, `t_res = int_F^phi_c max(tau_D, tau_s) dphi / phi`, evaluated by quadrature; and its series solution in the Rumpf-Gupte regime, `H(phi) = sum_k phi^(k - 4.5) / (k - 4.5)` falling by `C t_res / L` | A 60 km martian front entered at `phi_c = 0.3` (`drho = 330` kg/m3, `g = 3.711` m/s2), crossed at 70 and 5 cm/yr, for five sets of grain size, melt viscosity and mush viscosity. The model under the default buoyancy matches the quadrature to `rel=1e-6` in all ten trapped fractions and the series to `1e-5` in the three Rumpf-Gupte cases; the fractions are pinned at `rel=1e-4` (0.27165 and 0.18134 down to 0.0094342 and 0.0025379) and the timescale ratios likewise (`tau_D / t_res` from 8.1653 to 8.1653e-5, `tau_s / t_res` 0.50314, 50.314 and 0.0050314). Under `separation_buoyancy = 'solid'` the percolation-limited fractions drain further, by more than 0.007, and the drainage integral matches the quadrature under that buoyancy too. |

## Drainage velocity against the solver's separation velocity

### Plot: drainage velocity

![Drainage velocity against the interior solver's separation velocity](../../assets/validation/compaction_drainage_velocity.svg#only-light){ width="95%" }
![Drainage velocity against the interior solver's separation velocity](../../assets/validation/compaction_drainage_velocity_dark.svg#only-dark){ width="95%" }

(a) Relative deviation of the drainage velocity `darcy_velocity` from the interior solver's `relative_velocity`, divided by the tolerance at each point, for a grain size of 1 mm under `separation_buoyancy = 'mixture'` (circles) and `'solid'` (squares), at 48 porosities from 0.02 to 0.99. The low-porosity points sit near the edge of the band because the solver soft-clips its porosity, which lifts the mobility of the `phi^2` branch by about `1e-6 / (2 phi^2)`: the drainage velocity is 1.24e-3 below the solver's at a porosity of 0.02 and 1.86e-4 below at 0.05 under `'solid'`, and 1.23e-3 and 1.81e-4 below under `'mixture'`. (b) Velocity ratio `'mixture'` / `'solid'` from the drainage (line) and from the solver (circles, every fourth point), against `1 - phi` on every regime (dashed), a law the test rejects. The ratio follows `1 - phi` on the Blake-Kozeny-Carman and Rumpf-Gupte regimes, falls to 0.35 near a porosity of 0.68, and returns to 1 in the Stokes regime, where a grain settles alone in the melt; at a porosity of 0.7 the rejected law is 16 percent below the solver, at 0.9 it is 90 percent below.

- Reference: cross-implementation check against `aragog.eos.entropy_phase.EntropyPhaseEvaluator.relative_velocity` (fwl-aragog 26.10.22), the three-regime permeability of Bower et al. (2018), section 2.1, with the buoyancy against the mixture (McKenzie 1984, Eq. 3.7) or the solid (Abe 1995, Eq. 40).
- Tolerance: `rtol=2e-4` from a porosity of 0.05 up and `1.5e-3` below, at all 48 porosities, under both buoyancies and grain sizes of 0.1, 1 and 10 mm.
- Test id: `tests/outgas/test_compaction.py::test_mobility_reproduces_the_interior_solver_in_every_regime`
- Command: `pytest -m reference_pinned tests/outgas/test_compaction.py --validation-plots`

## Analytic limit checked alongside

`test_drainage_is_limited_by_the_slower_of_the_two_processes` pins the
matrix-limited case to its closed form `phi_c exp(-t_res / tau_s)` at
`rel=1e-6`.
