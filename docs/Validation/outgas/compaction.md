# Validation: `src/proteus/outgas/compaction.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor
`proteus.outgas.compaction`, the melt drainage of the front scheme.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_compaction.py::test_darcy_velocity_follows_the_closed_form_mobility_in_every_regime` | Closed forms of the three regimes from Bower, Sanan & Wolf (2018), Eqs. 12 and 13a-c (PDF page 6 of the manuscript), in the melt volume fraction; the factor `1 - phi` against the mixture from McKenzie (1984), Eq. 3.7, p. 721 | `darcy_velocity` at 9 porosities inside the Blake-Kozeny-Carman (0.005 to 0.02), Rumpf-Gupte (0.2 to 0.5) and Stokes (0.95, 0.99) regimes, for grain sizes of 0.1, 1 and 10 mm, under each `separation_buoyancy`. The tolerance at each point is twice the weight the blends of the solver law give the other branches there, from 7.5e-4 at 0.005 down to 5e-7 at 0.3. The modes differ by `1 - phi` from 0.2 to 0.5 (`rtol=1e-3`) and agree within 3e-3 in the Stokes regime. A `1 - phi` missing on one porous-flow branch or added on the Stokes branch fails the test. | 2026-10-24 |
| `tests/integration/test_slow_aragog_separation_buoyancy.py::test_the_configured_buoyancy_reaches_the_built_solver` | Wiring check against the interior solver: `relative_velocity` of the evaluator of the aragog solver PROTEUS builds (fwl-aragog 26.10.22), on the tables of a short run | Slow tier. At unit drag, `darcy_velocity` at the evaluator's own porosity equals the evaluator's separation velocity to `rtol=1e-12` on every mush node with a positive contrast, in each mode, so the drainage and the solver share one mobility law. | 2026-10-24 |
| `tests/outgas/test_compaction.py::test_the_lever_rule_porosity_is_the_solvers_where_the_melt_is_lighter` | Cross-implementation check against the interior solver's porosity, `aragog.eos.entropy_phase.EntropyPhaseEvaluator._porosity_from` (fwl-aragog 26.10.03), which it writes as `porosity_b` | On a two-phase mush with the melt lighter, the lever rule the trapping step applies to the solver's basic-node densities equals the solver's porosity to `atol=2e-5`, the soft clip of `2.5e-7 / phi` at each edge. Where the melt is the denser phase the solver's porosity reads below 1e-3, as its docstring warns, while the lever rule returns the melt volume, and a fully molten node takes its melt fraction, 1, where the lever rule on its single-phase density gives 0.95. | 2026-10-24 |
| `tests/outgas/test_compaction.py::test_drainage_matches_an_independent_quadrature_over_five_martian_fronts` | Analytic limits and a reference calculation: the closed-form timescales `tau_D = L eta_m / (drho g F(phi))`, with `F` the solver's mobility under the default `separation_buoyancy = 'mixture'`, `(5/7) (1 - phi) a^2 phi^4.5` in the Rumpf-Gupte regime, and `tau_s = mu_s / (drho g L)`; the separable form of the drainage equation, `t_res = int_F^phi_c max(tau_D, tau_s) dphi / phi`, evaluated by quadrature; and its series solution in the Rumpf-Gupte regime, `H(phi) = sum_k phi^(k - 4.5) / (k - 4.5)` falling by `C t_res / L` | A 60 km martian front entered at `phi_c = 0.3` (`drho = 330` kg/m3, `g = 3.711` m/s2), crossed at 70 and 5 cm/yr, for five sets of grain size, melt viscosity and mush viscosity. The model under the default buoyancy matches the quadrature to `rel=1e-6` in all ten trapped fractions and the series to `1e-5` in the three Rumpf-Gupte cases; the fractions are pinned at `rel=1e-4` (0.27165 and 0.18134 down to 0.0094342 and 0.0025379) and the timescale ratios likewise (`tau_D / t_res` from 8.1653 to 8.1653e-5, `tau_s / t_res` 0.50314, 50.314 and 0.0050314). Under `separation_buoyancy = 'solid'` the percolation-limited fractions drain further, by more than 0.007, and the drainage integral matches the quadrature under that buoyancy too. | 2026-10-24 |

## Drainage velocity against the closed forms

### Plot: drainage velocity

![Drainage velocity against the closed-form mobility](../../assets/validation/compaction_drainage_velocity.svg#only-light){ width="95%" }
![Drainage velocity against the closed-form mobility](../../assets/validation/compaction_drainage_velocity_dark.svg#only-dark){ width="95%" }

(a) Relative deviation of the drainage velocity `darcy_velocity` from the closed form of its regime, divided by the tolerance at each point, for a grain size of 1 mm under `separation_buoyancy = 'mixture'` (circles) and `'solid'` (squares), at the 9 porosities of the test. (b) Velocity ratio `'mixture'` / `'solid'` of the drainage (line) and of the closed forms (circles), against `1 - phi` (dashed). The ratio follows `1 - phi` on the Blake-Kozeny-Carman and Rumpf-Gupte regimes, falls to 0.35 at a porosity of 0.68, and returns to 1 in the Stokes regime, where a grain settles alone in the melt.

- Reference: Bower, Sanan & Wolf (2018), Eqs. 12 and 13a-c, with the melt mass fraction converted to the volume fraction, which gives the mobility `F = k / phi` of each regime: `a^2 phi^2 / (1000 (1 - phi)^2)`, `(5/7) a^2 phi^4.5` and `2 a^2 / 9`, with the regime bounds at `phi = 1 / 12.993` and `1 / 1.29624`; against the mixture, McKenzie (1984), Eq. 3.7, p. 721, puts `1 - phi` on the two porous-flow regimes. The slow-tier test above checks that the interior solver uses the same law.
- Tolerance: twice the weight of the blends on the other branches at each porosity, under both buoyancies and grain sizes of 0.1, 1 and 10 mm.
- Test id: `tests/outgas/test_compaction.py::test_darcy_velocity_follows_the_closed_form_mobility_in_every_regime`
- Command: `pytest -m reference_pinned tests/outgas/test_compaction.py --validation-plots`

### Scope of the buoyancy law in a run

A front is entered at a porosity near `interior_energetics.rfront_loc`. Up to a porosity of 0.4 (the default `rfront_loc`), the velocity under `'mixture'` equals `(1 - phi)` times the velocity under `'solid'` within 1e-5 (`test_mixture_drainage_leaves_the_uniform_factor_only_near_the_stokes_regime`), so there the law changes the drainage by 1e-5 at most against that uniform factor. Near the Stokes regime the two depart: at a porosity of 0.9 the `'mixture'` velocity is 9.9 times the uniform factor, and the percolation time of a front entered there is about a tenth of that under the uniform factor.

## Analytic limit checked alongside

`test_drainage_is_limited_by_the_slower_of_the_two_processes` pins the
matrix-limited case to its closed form `phi_c exp(-t_res / tau_s)` at
`rel=1e-6`.
