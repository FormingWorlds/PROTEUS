# Validation: `src/proteus/outgas/compaction.py`

## What this page checks

This page checks the melt drainage of the front scheme, `proteus.outgas.compaction`. As the freezing front passes through the mush, melt drains out of it before the solid closes; the melt that stays behind traps volatiles. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the trapped melt fraction, and with it the volatiles held in the solid, is wrong. Each case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends each case.

Terms used below:

- the mush is the partly molten layer behind the front;
- the porosity $\phi$ is its melt volume fraction;
- the lever rule gives $\phi$ from the densities of the mixture, the solid and the melt;
- a quadrature is a numerical integration;
- Blake-Kozeny-Carman, Rumpf-Gupte and Stokes name the 3 regimes of the drainage law, from low to high porosity;
- `separation_buoyancy` sets the density contrast of the drainage: `'mixture'` against the mixture, `'solid'` against the solid.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_compaction.py::test_darcy_velocity_follows_the_closed_form_mobility_in_every_regime` | Closed forms of the three regimes from Bower, Sanan & Wolf (2018), Eqs. 12 and 13a-c (PDF page 6 of the manuscript), in the melt volume fraction; the factor `1 - phi` against the mixture from McKenzie (1984), Eq. 3.7, p. 721 | `darcy_velocity` at 9 porosities inside the Blake-Kozeny-Carman (0.005 to 0.02), Rumpf-Gupte (0.2 to 0.5) and Stokes (0.95, 0.99) regimes, for grain sizes of 0.1, 1 and 10 mm, under each `separation_buoyancy`. The tolerance at each point is twice the weight the blends of the solver law give the other branches there, from 7.5e-4 at 0.005 down to 5e-7 at 0.3. The modes differ by `1 - phi` from 0.2 to 0.5 (`rtol=1e-3`) and agree within 3e-3 in the Stokes regime. A `1 - phi` missing on one porous-flow branch or added on the Stokes branch fails the test. | 2026-10-24 |
| `tests/integration/test_slow_aragog_separation_buoyancy.py::test_the_configured_buoyancy_reaches_the_built_solver` | Wiring check against the interior solver: `relative_velocity` of the evaluator of the Aragog solver PROTEUS builds (fwl-aragog 26.10.22), on the tables of a short run | Slow tier. At unit drag, `darcy_velocity` at the evaluator's own porosity equals the evaluator's separation velocity to `rtol=1e-12` on every mush node with a positive contrast, in each mode, so the drainage and the solver share one mobility law. | 2026-10-24 |
| `tests/outgas/test_compaction.py::test_the_lever_rule_porosity_is_the_solvers_where_the_melt_is_lighter` | Cross-implementation check against the interior solver's porosity, `aragog.eos.entropy_phase.EntropyPhaseEvaluator._porosity_from` (fwl-aragog 26.10.03), which it writes as `porosity_b` | On a two-phase mush with the melt lighter, the lever rule the trapping step applies to the solver's basic-node densities equals the solver's porosity to `atol=2e-5`, the soft clip of `2.5e-7 / phi` at each edge. Where the melt is the denser phase the solver's porosity reads below 1e-3, as its docstring warns, while the lever rule returns the melt volume, and at a fully molten node `porosity_from_densities` takes the melt fraction, 1, while the solver's porosity, the lever rule on the single-phase density, is 0.95. | 2026-10-24 |
| `tests/outgas/test_compaction.py::test_drainage_matches_an_independent_quadrature_over_five_martian_fronts` | Analytic limits and a reference calculation: the closed-form timescales `tau_D = L eta_m / (drho g F(phi))`, with `F` the solver's mobility under the default `separation_buoyancy = 'mixture'`, `(5/7) (1 - phi) a^2 phi^4.5` in the Rumpf-Gupte regime, and `tau_s = mu_s / (drho g L)`; the separable form of the drainage equation, `t_res = int_F^phi_c max(tau_D, tau_s) dphi / phi`, evaluated by quadrature; and its series solution in the Rumpf-Gupte regime, `H(phi) = sum_k phi^(k - 4.5) / (k - 4.5)` falling by `C t_res / L` | A 60 km martian front entered at `phi_c = 0.3` (`drho = 330` kg/m3, `g = 3.711` m/s2), crossed at 70 and 5 cm/yr, for five sets of grain size, melt viscosity and mush viscosity. The model under the default buoyancy matches the quadrature to `rel=1e-6` in all ten trapped fractions and the series to `1e-5` in the three Rumpf-Gupte cases; the fractions are pinned at `rel=1e-4` (0.27165 and 0.18134 down to 0.0094342 and 0.0025379) and the timescale ratios likewise (`tau_D / t_res` from 8.1653 to 8.1653e-5, `tau_s / t_res` 0.50314, 50.314 and 0.0050314). Under `separation_buoyancy = 'solid'` the percolation-limited fractions drain further, by more than 0.007, and the drainage integral matches the quadrature under that buoyancy too. | 2026-10-24 |

## Drainage velocity in each regime

**Why this check exists.** The drainage velocity follows a different law in each porosity regime, with a factor $1 - \phi$ that depends on the buoyancy setting. A missing or extra $1 - \phi$ on one branch, or a wrong regime law, gives a wrong velocity without an error message.

**What it tests.** `darcy_velocity` at 9 porosities in the Blake-Kozeny-Carman (0.005 to 0.02), Rumpf-Gupte (0.2 to 0.5) and Stokes (0.95, 0.99) regimes, for grain sizes of 0.1, 1 and 10 mm, under each `separation_buoyancy`. Each value must match the closed form of its regime. The 2 buoyancy settings must differ by $1 - \phi$ from 0.2 to 0.5 and agree in the Stokes regime.

**How we know the right answer.** Bower, Sanan and Wolf (2018), Eqs. 12 and 13a-c, give the closed forms; McKenzie (1984), Eq. 3.7, gives the factor $1 - \phi$ against the mixture.

**Why it matters.** `drainage_integral` uses this velocity to set how much melt drains from the front.

![Drainage velocity against the closed-form mobility](../../assets/validation/compaction_drainage_velocity.svg#only-light){ width="95%" }
![Drainage velocity against the closed-form mobility](../../assets/validation/compaction_drainage_velocity_dark.svg#only-dark){ width="95%" }

**How to read the plot.** Panel (a) shows, at the 9 porosities $\phi$ of the test, the asserted deviation of `darcy_velocity` from the closed form of its regime. The deviation is divided by the tolerance at each point, for a grain size of 1 mm. Circles are `separation_buoyancy = 'mixture'`, squares `'solid'`, and the shaded band is $\pm 1$, the tolerance. Panel (b) shows the velocity ratio `'mixture'` / `'solid'` of the drainage (line, context) and of the closed forms (circles), against $1 - \phi$ (dashed). The ratio follows $1 - \phi$ in the Blake-Kozeny-Carman and Rumpf-Gupte regimes, falls to 0.35 at $\phi$ = 0.68, and returns to 1 in the Stokes regime. A pass means that every point of (a) lies inside the band, and the circles of (b) on the line.

??? info "Test details"
    - Reference: Bower, Sanan & Wolf (2018), Eqs. 12 and 13a-c, with the melt mass fraction converted to the volume fraction, which gives the mobility `F = k / phi` of each regime: `a^2 phi^2 / (1000 (1 - phi)^2)`, `(5/7) a^2 phi^4.5` and `2 a^2 / 9`, with the regime bounds at `phi = 1 / 12.993` and `1 / 1.29624`. For the `'mixture'` buoyancy, McKenzie (1984), Eq. 3.7, p. 721, puts `1 - phi` on the two porous-flow regimes.
    - Tolerance: at each point twice the weight that the blends of the solver law give the other branches, from 7.5e-4 at 0.005 to 5e-7 at 0.3; relative 1e-3 on the factor $1 - \phi$; relative 3e-3 in the Stokes regime.
    - Test id: `tests/outgas/test_compaction.py::test_darcy_velocity_follows_the_closed_form_mobility_in_every_regime`
    - Command: `pytest -m reference_pinned tests/outgas/test_compaction.py --validation-plots`

## The same mobility law in the interior solver

**Why this check exists.** The drainage and the interior solver must use one mobility law. A buoyancy setting that does not reach the solver makes the 2 parts disagree without an error message.

**What it tests.** In a short run, at unit drag, `darcy_velocity` at the porosity of each mush node with a positive density contrast must equal the separation velocity of the Aragog solver that PROTEUS builds, in each buoyancy mode.

**How we know the right answer.** `darcy_velocity` takes its mobility from the Aragog function `mobility_function`, so this case checks the wiring: that the configured buoyancy reaches the solver PROTEUS builds.

**Why it matters.** The separation velocity sets the melt drainage in the interior solver and in the trapping step.

![darcy_velocity against the separation velocity of the built Aragog solver at the mush nodes](../../assets/validation/compaction_solver_mobility.svg#only-light){ width="80%" }
![darcy_velocity against the separation velocity of the built Aragog solver at the mush nodes](../../assets/validation/compaction_solver_mobility_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the separation velocity of the built Aragog solver multiplied by its drag viscosity, in m s$^{-1}$ Pa s on a log scale. The drag viscosity is the mixture or the melt viscosity, as `interior_energetics.aragog.separation_viscosity` sets. The y-axis is `darcy_velocity` for a viscosity of 1 Pa s, also on a log scale. Each point is a mush node (a node with both melt and solid) of a short run with a positive density contrast. Circles are the `'mixture'` buoyancy and squares the `'solid'` buoyancy; these are the asserted pairs (relative 1e-12). The line is 1:1, and a pass means that every point lies on it.

??? info "Test details"
    - Reference: `relative_velocity` of the evaluator of the Aragog solver PROTEUS builds (fwl-aragog 26.10.22), on the tables of a short run.
    - Tolerance: relative 1e-12 on every mush node with a positive contrast; further ratio checks at 1e-9, 1e-6, 1e-3 and 1e-2.
    - Test id: `tests/integration/test_slow_aragog_separation_buoyancy.py::test_the_configured_buoyancy_reaches_the_built_solver`
    - Command: `pytest -m reference_pinned tests/integration/test_slow_aragog_separation_buoyancy.py --validation-plots` (slow tier)

## Porosity from the lever rule

**Why this check exists.** The trapping step computes the porosity from the densities of the solid, the melt and the mixture with the lever rule. A wrong rule gives a porosity that differs from the one the interior solver uses, without an error message.

**What it tests.** On a 2-phase mush with the melt lighter than the solid, the lever-rule porosity must equal the porosity of the interior solver. Where the melt is the denser phase the solver's porosity reads below 1e-3 while the lever rule returns the melt volume, At a fully molten node `porosity_from_densities` takes the melt fraction, 1, while the solver's porosity, the lever rule on the single-phase density, is 0.95.

**How we know the right answer.** The porosity of the Aragog solver (`EntropyPhaseEvaluator._porosity_from`, fwl-aragog 26.10.03), a second implementation.

**Why it matters.** The trapping step applies the lever rule to the densities of the interior solver; the porosity locates the freezing front and sets the porosity at its top, where the drainage starts.

![Porosity from the lever rule and from the interior solver against melt fraction](../../assets/validation/compaction_lever_rule_porosity.svg#only-light){ width="80%" }
![Porosity from the lever rule and from the interior solver against melt fraction](../../assets/validation/compaction_lever_rule_porosity_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the melt volume fraction and the y-axis the porosity $\phi$. The line is $\phi$ equal to the melt volume fraction (context). Where the melt is the lighter phase, the filled circles are the asserted lever-rule porosities and the hollow squares the asserted solver porosities, within 2e-5. Where the melt is the denser phase, the triangles are the asserted solver porosities, all below 1e-3. At the fully molten node, the star is `porosity_from_densities` (1) and the cross the solver porosity (0.95, the lever rule on density). A pass means that circles and squares lie on the line, and the triangles at 0.

??? info "Test details"
    - Reference: `aragog.eos.entropy_phase.EntropyPhaseEvaluator._porosity_from` (fwl-aragog 26.10.03), written as `porosity_b`.
    - Tolerance: absolute 2e-5, the soft clip of `2.5e-7 / phi` at each edge.
    - Test id: `tests/outgas/test_compaction.py::test_the_lever_rule_porosity_is_the_solvers_where_the_melt_is_lighter`
    - Command: `pytest -m reference_pinned tests/outgas/test_compaction.py --validation-plots`

## Drainage over 5 martian fronts

**Why this check exists.** The trapped melt fraction comes from integrating the drainage over the time the front takes to pass. A wrong timescale or a wrong integration gives a wrong trapped fraction without an error message.

**What it tests.** A 60 km martian front entered at a porosity of 0.3, crossed at 70 and 5 cm/yr, for 5 sets of grain size, melt viscosity and mush viscosity. The model must match a quadrature of the drainage equation in all 10 trapped fractions, and its series solution in the 3 Rumpf-Gupte cases. It also checks the closed-form timescales and the stored fractions and timescale ratios. The porosity $\phi$ falls as $d\phi/dt = -\phi / \max(\tau_D, \tau_s)$ over the residence time $t_\mathrm{res} = L / v_f$ of a front of width $L$ that moves at speed $v_f$; the retained melt fraction is $\phi$ at $t_\mathrm{res}$. Here $\tau_D = L \eta_m / (\Delta\rho\, g\, F(\phi))$ is the time the melt needs to drain through the mush, with $\eta_m$ the melt viscosity in Pa s, $\Delta\rho$ the density contrast in kg m$^{-3}$, $g$ the gravity in m s$^{-2}$ and $F$ the mobility in m$^2$. $\tau_s = \mu_s / (\Delta\rho\, g\, L)$ is the time the solid matrix needs to compact, with $\mu_s$ the mush viscosity in Pa s.

**How we know the right answer.** The closed-form timescales $\tau_D$ and $\tau_s$, a quadrature of the separable drainage equation and its series solution in the Rumpf-Gupte regime (the table above gives the series).

**Why it matters.** The trapping step takes its trapped melt fraction from this drainage.

![Retained melt fraction of the model against the quadrature and the series solution](../../assets/validation/compaction_martian_fronts.svg#only-light){ width="80%" }
![Retained melt fraction of the model against the quadrature and the series solution](../../assets/validation/compaction_martian_fronts_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the reference value and the y-axis the retained melt fraction of the model, both on log scales. The filled circles are the 10 asserted pairs of model and quadrature (relative 1e-6). They are 5 fronts crossed at 70 and at 5 cm/yr. The hollow squares are the 3 pairs of model and series solution in the Rumpf-Gupte cases (relative 1e-5). The line is 1:1, and a pass means that every point lies on it.

??? info "Test details"
    - Reference: closed-form timescales, the quadrature of the drainage equation and its Rumpf-Gupte series solution.
    - Tolerance: relative 1e-6 against the quadrature; relative 1e-5 against the series; relative 1e-4 on the stored fractions and ratios.
    - Test id: `tests/outgas/test_compaction.py::test_drainage_matches_an_independent_quadrature_over_five_martian_fronts`
    - Command: `pytest -m reference_pinned tests/outgas/test_compaction.py --validation-plots`

## Scope of the buoyancy law in a run

A front is entered at a porosity near `interior_energetics.rfront_loc`. Up to a porosity of 0.4 (the default `rfront_loc`), the velocity under `'mixture'` equals `(1 - phi)` times the velocity under `'solid'` within 1e-5 (`test_mixture_drainage_leaves_the_uniform_factor_only_near_the_stokes_regime`), so there the law changes the drainage by 1e-5 at most against that uniform factor. Near the Stokes regime the two depart: at a porosity of 0.9 the `'mixture'` velocity is 9.9 times the uniform factor, and the percolation time of a front entered there is about a tenth of that under the uniform factor.

## Analytic limit checked alongside

`test_drainage_is_limited_by_the_slower_of_the_two_processes` pins the
matrix-limited case to its closed form `phi_c exp(-t_res / tau_s)` at
`rel=1e-6`.
