# Validation: `src/proteus/outgas/lid_seal.py`

## What this page checks

This page checks the lid seal, `proteus.outgas.lid_seal`. When a crust closes over the magma ocean, the seal holds the volatiles dissolved in the melt, and they reach the atmosphere only by a slow leak. The leak stops at a floor: the mass that melt and crystals hold in equilibrium with the atmosphere. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the atmosphere gets the held volatiles too early, too late or in the wrong amount, or more than the melt would release, and the mass of an element is not conserved. Each case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends each case.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_lid_seal_smoke.py::test_equilibrium_fractions_are_the_dissolved_fractions_of_a_solved_row` | Real CALLIOPE solve | The floor fractions against the dissolved fractions of solved rows. | 2026-11-11 |
| `tests/outgas/test_lid_seal.py::test_relaxation_matches_the_exponential_and_not_the_step_length` | Analytical solution of $dM/dt = -(M - M_\mathrm{eq})/\tau$ | The held mass at a constant floor against the exponential, for steps of 1, 10 and 100 kyr. | 2026-11-11 |
| `tests/outgas/test_lid_seal.py::test_volcanic_rate_closes_the_mass_balance_of_the_sealed_reservoir` | Schaefer & Sasselov (2015)[^cite-schaefer2015], Eq. 14 and Eq. 17; mass balance | The volcanic rate against its closed form over the global melt fraction. | 2026-11-11 |
| `tests/outgas/test_lid_seal_convergence.py::test_leak_step_converges_to_the_continuous_law` | Radau integration of the continuous law | The end state of 120 and 480 steps against the reference, for $\tau$ from 100 yr to 10 Myr. | 2026-11-11 |
| `tests/outgas/test_lid_seal.py::test_a_short_tau_gives_the_equilibrium_partition` | Gas-melt equilibrium, solved independently | The end state of a step with $\tau$ = $10^{-3}\,\Delta t$. | 2026-11-11 |
| `tests/outgas/test_lid_seal_tutorial.py::test_the_tutorial_atmosphere_lies_between_the_held_and_the_open_run` | Tutorial runs with and without the seal | The water of the tutorial run against its floor and the 2 limits. | 2026-11-11 |

## The floor uses the dissolved fractions of the chemistry solve

**Why this check exists.** The floor of the leak must use the same solubility laws, pressures, temperature and oxygen buffer as the chemistry solve that filled the held reservoir. A wrong argument, for example the configured IW offset in place of the one the solve derived, moves the floor and lets a volatile leak too much or too little. The test fails then, because the N$_2$ and S$_2$ fractions move by more than 1 %.

**What it tests.** CALLIOPE solves 3 rows: at IW-2 and IW+2 (IW, the iron-wüstite buffer) with He and Ar on, and one with the IW offset derived from an oxygen budget (`from_O_budget`). One row has its magma temperature below `outgas.T_floor`. For each species, the floor fraction $X_{\mathrm{eq},i}$ of the seal, at the partial pressures of the row, must equal the dissolved mass of the solve over $\Phi M_\mathrm{mantle}$.

**How we know the right answer.** The solve computes its dissolved mass with the same laws; both sides must agree to round-off.

**Why it matters.** At closure the held mass is the dissolved mass of this solve, so the floor starts where the chemistry left off.

This case has no plot: it compares numbers that agree to round-off.

??? info "Test details"
    - Reference: the dissolved masses of the real CALLIOPE solve.
    - Tolerance: relative 1e-12 per species; H$_2$O, CO$_2$, CO, CH$_4$, N$_2$, S$_2$, He and Ar dissolve.
    - Test ids: `tests/outgas/test_lid_seal_smoke.py::test_equilibrium_fractions_are_the_dissolved_fractions_of_a_solved_row`, `tests/outgas/test_lid_seal_smoke.py::test_equilibrium_fractions_take_the_derived_fo2_under_from_o_budget`
    - Command: `pytest tests/outgas/test_lid_seal_smoke.py`

## At a constant floor the held mass relaxes as the exponential, for any step length

**Why this check exists.** Above its floor, the leak removes a fixed fraction of the distance to the floor per unit time. A step formula that is only first-order accurate, for example $M - (M - M_\mathrm{eq})\,\Delta t/\tau$, gives a result that depends on the step length. With 10 steps of 100 kyr and $\tau$ = 1 Myr, that formula misses the exact distance to the floor by 5.2 % after 1 Myr, and the test requires more than 5 %.

**What it tests.** The step of the leak runs at a constant floor $M_\mathrm{eq} = 0.4\,M_0$, with $M_0$ the held mass at the closure time $t_c$, the relaxation time $\tau$ = 1 Myr and no volcanism. It advances the held mass with steps of 1, 10 and 100 kyr up to 1 Myr. At every step the held mass must equal $M_\mathrm{eq} + (M_0 - M_\mathrm{eq})\,e^{-(t - t_c)/\tau}$, and the 1 kyr and 100 kyr steps must agree at the times of the 100 kyr steps. A second test runs the full leak step with solubility off, where every floor is 0, against $M_0\,e^{-(t - t_c)/\tau}$.

**How we know the right answer.** $dM/dt = -(M - M_\mathrm{eq})/\tau$ has the exact solution above. The code uses this solution over each step, so only round-off can remain.

**Why it matters.** The adaptive step of a PROTEUS run changes from step to step. Where the floor does not move, the held mass must not depend on that choice.

![Distance of the held mass to a constant floor against time since closure, for 3 step lengths, with the exponential](../../assets/validation/outgas_lid_seal_relaxation.svg#only-light){ width="80%" }
![Distance of the held mass to a constant floor against time since closure, for 3 step lengths, with the exponential](../../assets/validation/outgas_lid_seal_relaxation_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The upper panel shows the distance to the floor, $(M - M_\mathrm{eq})/(M_0 - M_\mathrm{eq})$, against the time since closure in units of $\tau$. The line is the exponential; the markers are the code for the 3 step lengths. The lower panel shows the relative error of the code, rounded to the tolerance 1e-12, because the round-off below it changes with the platform. A pass means that every marker lies on the line and every error is below 1e-12.

??? info "Test details"
    - Reference: analytical solution of $dM/dt = -(M - M_\mathrm{eq})/\tau$.
    - Tolerance: relative 1e-12 at every step, and between the 1 kyr and 100 kyr steps; the first-order step must be off by more than 5 %.
    - Test ids: `tests/outgas/test_lid_seal.py::test_relaxation_matches_the_exponential_and_not_the_step_length`, `tests/outgas/test_lid_seal.py::test_with_no_solubility_the_leak_decays_to_zero_whatever_the_step`
    - Command: `pytest -m reference_pinned tests/outgas/test_lid_seal.py --validation-plots`

## The volcanic rate closes the mass balance of the held volatile

**Why this check exists.** The volcanic leak needs the mass fraction of a volatile in the melt below the crust. A code that divides by the melt fraction of only the cells that hold melt, and not by the global melt fraction, gives a melt concentration that is too small. With half the mantle solid and half at 40 % melt, that error halves the volcanic rate, and the test fails.

**What it tests.** The volcanic rate of species $i$ is $k_i (M_i - M_{\mathrm{eq},i})$, with $k_i = \chi_d \dot M_c / (M_\mathrm{melt} + D_i M_\mathrm{solid})$. Here $M_i$ is the held mass, $M_{\mathrm{eq},i}$ its floor, $\chi_d$ the degassing efficiency, $\dot M_c$ the melt mass that reaches the crust per year, $D_i$ the crystal/melt partition coefficient, and $M_\mathrm{melt}$ and $M_\mathrm{solid}$ the melt and solid mass of the mantle. The test checks $k_i$ in 3 ways:

- the mass balance: the melt concentration times $M_\mathrm{melt} + D_i M_\mathrm{solid}$ gives back $M_i$, for CO$_2$ ($D$ = 0) and water ($D$ = 0.0017);
- the 2-region mantle: at a global melt fraction $\Phi$ = 0.2 and $D$ = 0, the melt concentration is $M_i/(0.2\,M_\mathrm{mantle})$, and the local melt fraction 0.4 would give half of it;
- one state written out by hand from Eq. 14 of Schaefer & Sasselov (2015), with their melt production term replaced by $\dot M_c$.

It also checks the closed form $1/(D + \Phi(1 - D))$ of the scaled rate over $\Phi$ from 0.01 to 1, for $D$ = 0, 0.01, 0.1 and 1.

**How we know the right answer.** The held volatile sits in the melt and, at $D > 0$, in the crystals in equilibrium with it. Mass balance then gives the melt concentration. The volcanic rate has the form of Eq. 14 and Eq. 17 of Schaefer & Sasselov (2015); replacing their melt production by $\dot M_c$, their degree of melting by $\Phi$, and their concentration by its excess over the solubility is an adaptation. The hand-written state checks the adapted formula, not a published value.

**Why it matters.** With `crust_rate > 0`, this rate sets how fast the held volatiles reach the atmosphere.

![Scaled volcanic rate against the global melt fraction for 4 partition coefficients](../../assets/validation/outgas_lid_seal_volcanic_rate.svg#only-light){ width="80%" }
![Scaled volcanic rate against the global melt fraction for 4 partition coefficients](../../assets/validation/outgas_lid_seal_volcanic_rate_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the global melt fraction $\Phi$, the y-axis the scaled volcanic rate $k_i M_\mathrm{mantle}/(\chi_d \dot M_c)$, both on log scales. Each line is the closed form $1/(D + \Phi(1 - D))$ for one partition coefficient $D$; the markers are the code. A pass means that every marker lies on its line. At $D$ = 1 the rate does not depend on $\Phi$, because the volatile is spread evenly over melt and crystals.

??? info "Test details"
    - Reference: Schaefer & Sasselov (2015), Eq. 14 and Eq. 17; mass balance of the held volatile.
    - Tolerance: relative 1e-14 for the mass balance and the 2-region case, 1e-12 for the hand-written state, 1e-13 for the closed form; the local melt fraction must be more than 40 % off.
    - Test id: `tests/outgas/test_lid_seal.py::test_volcanic_rate_closes_the_mass_balance_of_the_sealed_reservoir`
    - Command: `pytest -m reference_pinned tests/outgas/test_lid_seal.py --validation-plots`

## The leak stops at the floor of the end state

**Why this check exists.** The floor rises with the gas that the leak puts into the atmosphere. A step that takes the floor at its start leaks past the end state, and a step that ignores the floor drains the held mass to 0. In both cases the atmosphere gets more gas than the melt would release.

**What it tests.** Held water and CO$_2$ start above their floors and leak for one step of 1 kyr, with $\tau$ from 1 to 100 kyr, with and without volcanism, and with volcanism alone ($\tau$ infinite). The floor $M_{\mathrm{eq},i}$ is computed in the test from the atmosphere after the step, with the solubility laws written out. Each species must end at or above its floor, and its distance to it must be $e^{-\lambda_i \Delta t}$ times the distance it started at, with $\lambda_i = 1/\tau + k_i$. A species below its floor must not move.

**How we know the right answer.** For a constant floor the leak law has the exact solution of the previous cases; the step takes the floor of its end state.

**Why it matters.** This sets how much of each volatile a sealed planet releases.

This case has no plot: it checks one step against an identity.

??? info "Test details"
    - Reference: the exact step with the floor of the end state; Sossi et al. and Dixon et al. laws written in the test.
    - Tolerance: relative 1e-9 for the distance to the floor, 1e-12 for the mass moved; exact for a species below its floor.
    - Test ids: `tests/outgas/test_lid_seal.py::test_leak_relaxes_toward_the_floor_of_the_end_state`, `tests/outgas/test_lid_seal.py::test_sealed_mass_below_its_floor_does_not_leak`
    - Command: `pytest tests/outgas/test_lid_seal.py`

## The step converges to the continuous law

**Why this check exists.** Within a step the floor moves, because the melt fraction falls, but the step takes the floor of its end state. The result is then first order in the step length. This check measures how large that error is, and that it falls with the step as a first-order method must.

**What it tests.** Held water, CO$_2$ and N$_2$ start at the melt equilibrium of a closing state, with escape off. The global melt fraction falls linearly from 0.19 to 0.135 over 43 kyr, as in the [tutorial](../../Tutorials/lid_seal.md). The leak step runs this path in 120 and in 480 equal steps, for $\tau$ from 100 yr to 10 Myr. Each atmospheric and held mass at the end is compared with the reference, as a fraction of the inventory of its species: the held mass plus the atmospheric mass. That error must be below $10^{-4}$. Where the step is short against the leak ($\lambda \Delta t < 0.1$), 4 times more steps must cut the largest error by at least 3.5 times.

**How we know the right answer.** A stiff ODE solver (Radau, relative tolerance $10^{-11}$) integrates the same law in continuous time. The test writes the solubility laws out itself, without the CALLIOPE classes.

**Why it matters.** The adaptive step of a PROTEUS run changes from step to step; in the held part of the tutorial run it is 40 to 160 yr, shorter than the 358 yr steps of this test. The error of the step must stay small against the gas each species holds.

This case has no plot: its result is a table of errors, given below.

**Known limit.** Per mass, the error can be larger than per inventory. At 120 steps of 358 yr it reaches $1.9\times10^{-3}$ of the held CO$_2$ and N$_2$, which sit at a floor that moves with the melt fraction. It reaches $1.1\times10^{-3}$ of the atmospheric water at $\tau$ = 10 Myr, because the first leaking step is resolved only to one step. The error over the inventory is at most $4.4\times10^{-5}$.

??? info "Test details"
    - Reference: Radau integration of $dM_i/dt = -(1/\tau) \max(M_i - M_{\mathrm{eq},i}, 0)$ with the laws written in the test.
    - Tolerance: $10^{-4}$ of the species inventory; error cut of at least 3.5 times for 4 times more steps where $\lambda \Delta t < 0.1$.
    - Test id: `tests/outgas/test_lid_seal_convergence.py::test_leak_step_converges_to_the_continuous_law`
    - Command: `pytest tests/outgas/test_lid_seal_convergence.py`

## A short relaxation time gives the gas-melt equilibrium

**Why this check exists.** For $\tau$ much shorter than the step, the leak must end at the gas-melt equilibrium of the frozen species: each held mass equals its floor. A step that solved the species one at a time, without the coupling through the surface pressure and the mean molar mass, would miss it.

**What it tests.** Held water and CO$_2$ leak for one step with $\tau$ = $10^{-3}\,\Delta t$. The test finds the equilibrium on its own, with nested root finds on the 2 atmospheric masses, and compares the atmospheric and held masses.

**How we know the right answer.** At equilibrium each held mass equals its floor and the total of each species is unchanged; the test solves these 2 conditions independently of the code.

**Why it matters.** It is the fast limit of the leak: a crust that lets gas through at once.

This case has no plot: it checks one end state.

??? info "Test details"
    - Reference: the gas-melt equilibrium with frozen species, solved in the test.
    - Tolerance: relative 1e-9.
    - Test id: `tests/outgas/test_lid_seal.py::test_a_short_tau_gives_the_equilibrium_partition`
    - Command: `pytest tests/outgas/test_lid_seal.py`

## A finite relaxation time lies between the held and the open atmosphere

**Why this check exists.** With escape off and one dissolving species, the water in the atmosphere of a sealed run must lie between 2 limits: the seal that leaks nothing, and the open equilibrium of the melt. A leak that overshoots its floor, or one that runs the wrong way, leaves this band.

**What it tests.** Water starts at the melt equilibrium of a closing state and leaks with $\tau$ = 3 kyr while the global melt fraction falls from 0.3 to 0.15 in 15 steps. At every step the atmospheric water must be below the open equilibrium on the melt alone and at or above its value at closure. Once it leaks, it must stay above that value.

**How we know the right answer.** The open state is the equilibrium on the melt alone, which lies below the floor on melt and crystals; a finite $\tau$ lags behind both. The test computes the open state with its own root find.

**Why it matters.** The seal must never make a planet release more water than the open melt would.

This case has no plot: the next case shows the same band on the tutorial run.

??? info "Test details"
    - Reference: the open equilibrium on the melt alone and the held state, both solved in the test.
    - Tolerance: strict inequalities at every step.
    - Test id: `tests/outgas/test_lid_seal.py::test_finite_tau_lies_between_the_frozen_and_the_open_atmosphere`
    - Command: `pytest tests/outgas/test_lid_seal.py`

## The first held steps of the tutorial follow the floor

**Why this check exists.** At closure the held water is the dissolved water of the melt, but its floor also counts the crystals. So the held water starts below its floor and must wait until the floor falls under it. A code that leaked from the first held step, or never, would fail.

**What it tests.** The test replays the closing row of the [tutorial](../../Tutorials/lid_seal.md) and its next 13 rows, stored in `tests/data/lid_seal/tutorial_closing_rows.csv`, with escape off. CO$_2$, CO and CH$_4$ ($D$ = 0) must leak from the first held step. Water ($D$ = 0.0017) must stay below its floor until the first row where the global melt fraction $\Phi$ falls below $(\Phi_c - D)/(1 - D)$, and leak from that row on.

**How we know the right answer.** With the pressures fixed, the held water $X_\mathrm{eq} \Phi_c M_\mathrm{mantle}$ equals its floor $X_\mathrm{eq} (\Phi + D(1 - \Phi)) M_\mathrm{mantle}$ at $\Phi = (\Phi_c - D)/(1 - D)$. Here $\Phi_c$ is the global melt fraction at closure.

**Why it matters.** This delay shows how the reservoir of the floor, melt plus crystals, differs from the melt-only reservoir of the chemistry solve.

This case has no plot: it checks the row of the first leak of each species.

??? info "Test details"
    - Reference: the closed form of the first leaking row, from stored rows of the tutorial run.
    - Tolerance: exact row numbers.
    - Test id: `tests/outgas/test_lid_seal_tutorial.py::test_the_first_held_steps_of_the_tutorial_follow_the_floor`
    - Command: `pytest tests/outgas/test_lid_seal_tutorial.py`

## The tutorial atmosphere lies between the held and the open run

**Why this check exists.** The leak law must keep the floor on every row of a full PROTEUS run, with escape, all species and the adaptive step. A run that leaks past the floor fails this check.

**What it tests.** The [tutorial](../../Tutorials/lid_seal.md) runs 3 times: with the seal and $\tau$ = 100 kyr, with the seal and $\tau$ infinite, and without the seal. Their water columns are stored in `tests/data/lid_seal/tutorial_water.csv`. On every held row where the held water fell, it must end at or above its floor `H2O_kg_eq`. From closure on, the atmospheric water of the run with $\tau$ = 100 kyr must lie at or above the run with $\tau$ infinite and below the open run.

**How we know the right answer.** The floor check follows from the law. The run with $\tau$ infinite leaks nothing, and the open run keeps its melt in equilibrium with the atmosphere at all times, so they bound a run that leaks toward the floor.

**Why it matters.** This is the result a user sees: how much water a planet under a closed crust releases.

![Water in the atmosphere of the 3 tutorial runs against time, and the distance of the held water to its floor](../../assets/validation/outgas_lid_seal_tutorial_water.svg#only-light){ width="80%" }
![Water in the atmosphere of the 3 tutorial runs against time, and the distance of the held water to its floor](../../assets/validation/outgas_lid_seal_tutorial_water_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The upper panel shows the water in the atmosphere, in units of $10^{15}$ kg, against time in kyr for the open run, the seal with $\tau$ = 100 kyr, and the seal with $\tau$ infinite. The seal closes at 97 kyr. The lower panel shows, for the seal with $\tau$ = 100 kyr, how far the held water lies above its floor, as a fraction of the held water. Below the dotted zero line the held water is below its floor and does not leak. A pass means that the seal curve lies between the 2 other curves, and the lower curve is at or above zero on every row where the held water fell.

??? info "Test details"
    - Reference: tutorial runs with the seal ($\tau$ = 100 kyr and infinite) and without it.
    - Tolerance: relative 1e-12 for the floor; inequalities on every row.
    - Test id: `tests/outgas/test_lid_seal_tutorial.py::test_the_tutorial_atmosphere_lies_between_the_held_and_the_open_run`
    - Command: `pytest -m reference_pinned tests/outgas/test_lid_seal_tutorial.py --validation-plots`

## Invariants checked alongside

The tests in `tests/outgas/test_lid_seal.py` and the code itself also check, without a published reference:

- closing and opening the seal moves every species, element and mole column between the melt and the held reservoir without loss, and a row that cannot be read is refused before any mass moves;
- with the defaults (`tau_leak` infinite, `crust_rate = 0`) no floor is computed and every column stays the same to the last bit, with solubility on or off;
- after every step that leaks, the code checks that each species ends at or above its floor, at $e^{-\lambda_i \Delta t}$ of its starting distance; a step that ignores the floor or leaks too little stops the run;
- a negative or non-finite floor, a closing temperature that is not finite and above 0 K, a non-finite IW offset, and sweeps over the species that do not settle stop the run with an error;
- a step that leaves CO$_2$ above $83.15\,T/23$ bar, where the Dixon et al. CO$_2$ law falls with pressure, stops the run with an error;
- a sealed row resumes only with a finite closing temperature above 0 K;
- in one step the 2 leak parts of each element sum to the drop of its held mass and to the rise of its atmospheric mass; each species splits its leak in the ratio of its 2 rates;
- in one step no more melt than the melt mass reaches the crust, and below `phi_solid` none does;
- after a leak, the surface pressure is $M_\mathrm{atm}\,g/(4\pi R^2)$ and the mixing ratios follow the moles;
- the seal opens only when the top cell holds `phi_open` or more melt, when the crust is thinner than `delta_open`, or on a giant-impact step; a held step runs no chemistry solve;
- the crust thickness on the mesh reaches the lower face of the top run of solid cells;
- on every held row the melt columns are 0 and the held masses are finite and not negative (an error stops the run otherwise);
- the per-element closure includes the held mass, and the held mass blocks the desiccation check.

[^cite-schaefer2015]: L. Schaefer, D. Sasselov, *[The persistence of oceans on Earth-like planets: insights from the deep-water cycle](https://doi.org/10.1088/0004-637X/801/1/40)*, The Astrophysical Journal, 801, 40, 2015. [SciX](https://scixplorer.org/abs/2015ApJ...801...40S/abstract).
