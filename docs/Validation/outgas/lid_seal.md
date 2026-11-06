# Validation: `src/proteus/outgas/lid_seal.py`

## What this page checks

This page checks the lid seal, `proteus.outgas.lid_seal`. When a crust closes over the magma ocean, the seal holds the volatiles dissolved in the melt, and they reach the atmosphere only by a slow leak. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the atmosphere gets the held volatiles too early, too late or in the wrong amount, and the mass of an element is not conserved. Each case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends each case.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_lid_seal.py::test_relaxation_matches_the_exponential_and_not_the_step_length` | Analytical solution of $dM/dt = -M/\tau$ | The held mass against $M_c\,e^{-(t - t_c)/\tau}$ at every step, for steps of 1, 10 and 100 kyr. |
| `tests/outgas/test_lid_seal.py::test_volcanic_rate_closes_the_mass_balance_of_the_sealed_reservoir` | Schaefer & Sasselov (2015), ApJ 801:40, Eq. 14 and Eq. 17; mass balance | The volcanic rate against its closed form over the global melt fraction, the mass balance of the held volatile, and one state written out by hand. |

## The held mass decays as the exponential, for any step length

**Why this check exists.** The leak removes a fixed fraction of the held mass per unit time. A step formula that is only first-order accurate, for example $M - M\,\Delta t/\tau$, loses the wrong mass and gives a result that depends on the step length. With 10 steps of 100 kyr and $\tau$ = 1 Myr, that formula is 5.5 % off after 1 Myr, and the test requires more than 5 %.

**What it tests.** A row with water and CO$_2$ held under the crust leaks with the relaxation time $\tau$ = 1 Myr and no volcanism (`crust_rate = 0`). Here $M_c$ is the held mass at the closure time $t_c$. The code advances the held mass with steps of 1, 10 and 100 kyr up to 1 Myr. At every step the held water must equal $M_c\,e^{-(t - t_c)/\tau}$, and the 3 step lengths must give the same mass at the common times.

**How we know the right answer.** $dM/dt = -M/\tau$ has the exact solution $M_c\,e^{-(t - t_c)/\tau}$. The code uses this solution over each step, so only round-off can remain.

**Why it matters.** The adaptive step of a PROTEUS run changes from step to step. The held mass must not depend on that choice.

![Held water mass against time since closure, for 3 step lengths, with the exponential](../../assets/validation/outgas_lid_seal_relaxation.svg#only-light){ width="80%" }
![Held water mass against time since closure, for 3 step lengths, with the exponential](../../assets/validation/outgas_lid_seal_relaxation_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The upper panel shows the held fraction $M/M_c$ against the time since closure in units of $\tau$. The line is the exponential; the markers are the code for the 3 step lengths. The lower panel shows the relative error of the code against the exponential, rounded to the tolerance 1e-12, because the round-off below it changes with the platform; the largest error before rounding is 3.0e-14 on macOS. A pass means that every marker lies on the line and every error is below 1e-12.

??? info "Test details"
    - Reference: analytical solution of $dM/dt = -M/\tau$.
    - Tolerance: relative 1e-12 at every step and between the step lengths; the first-order step must be off by more than 5 %.
    - Test id: `tests/outgas/test_lid_seal.py::test_relaxation_matches_the_exponential_and_not_the_step_length`
    - Command: `pytest -m reference_pinned tests/outgas/test_lid_seal.py --validation-plots`

## The volcanic rate closes the mass balance of the held volatile

**Why this check exists.** The volcanic leak needs the mass fraction of a volatile in the melt below the crust. A code that divides by the melt fraction of only the cells that hold melt, and not by the global melt fraction, gives a melt concentration that is too small. With half the mantle solid and half at 40 % melt, that error halves the volcanic rate, and the test fails.

**What it tests.** The volcanic rate of species $i$ is $F_{\mathrm{volc},i} = k_i M_i$, with $k_i = \chi_d \dot M_c / (M_\mathrm{melt} + D_i M_\mathrm{solid})$. Here $M_i$ is the held mass, $\chi_d$ the degassing efficiency, $\dot M_c$ the melt mass that reaches the crust per year, $D_i$ the crystal/melt partition coefficient, and $M_\mathrm{melt}$ and $M_\mathrm{solid}$ the melt and solid mass of the mantle. The test checks 3 things:

- the mass balance: the melt concentration times $M_\mathrm{melt} + D_i M_\mathrm{solid}$ gives back $M_i$, for CO$_2$ ($D$ = 0) and water ($D$ = 0.0017);
- the 2-region mantle: at a global melt fraction $\Phi$ = 0.2 and $D$ = 0, the melt concentration is $M_i/(0.2\,M_\mathrm{mantle})$, and the local melt fraction 0.4 would give half of it;
- one state written out by hand from Eq. 14 of Schaefer and Sasselov (2015), with their melt production term replaced by $\dot M_c$.

It also checks the closed form $1/(D + \Phi(1 - D))$ of the scaled rate over $\Phi$ from 0.01 to 1, for $D$ = 0, 0.01, 0.1 and 1.

**How we know the right answer.** The held volatile sits in the melt and, at $D > 0$, in the crystals in equilibrium with it. Mass balance then gives the melt concentration. The volcanic rate has the form of Eq. 14 and Eq. 17 of Schaefer and Sasselov (2015); replacing their melt production by $\dot M_c$ and their degree of melting by $\Phi$ is an adaptation, so the hand-written state checks the adapted formula, not a published value.

**Why it matters.** With `crust_rate > 0`, this rate sets how fast the held volatiles reach the atmosphere.

![Scaled volcanic rate against the global melt fraction for 4 partition coefficients](../../assets/validation/outgas_lid_seal_volcanic_rate.svg#only-light){ width="80%" }
![Scaled volcanic rate against the global melt fraction for 4 partition coefficients](../../assets/validation/outgas_lid_seal_volcanic_rate_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the global melt fraction $\Phi$, the y-axis the scaled volcanic rate $k_i M_\mathrm{mantle}/(\chi_d \dot M_c)$, both on log scales. Each line is the closed form $1/(D + \Phi(1 - D))$ for one partition coefficient $D$; the markers are the code. A pass means that every marker lies on its line. At $D$ = 1 the rate does not depend on $\Phi$, because the volatile is spread evenly over melt and crystals.

??? info "Test details"
    - Reference: Schaefer & Sasselov (2015), ApJ 801:40, Eq. 14 and Eq. 17; mass balance of the held volatile.
    - Tolerance: relative 1e-14 for the mass balance and the 2-region case, 1e-12 for the hand-written state, 1e-13 for the closed form; the local melt fraction must be more than 40 % off.
    - Test id: `tests/outgas/test_lid_seal.py::test_volcanic_rate_closes_the_mass_balance_of_the_sealed_reservoir`
    - Command: `pytest -m reference_pinned tests/outgas/test_lid_seal.py --validation-plots`

## Invariants checked alongside

The tests in `tests/outgas/test_lid_seal.py` and the code itself also check, without a published reference:

- closing and opening the seal moves every species, element and mole column between the melt and the held reservoir without loss, and a row that cannot be read is refused before any mass moves;
- with the defaults (`tau_leak` infinite, `crust_rate = 0`) every column stays the same to the last bit;
- in one step the 2 leak parts of each element sum to the drop of its held mass and to the rise of its atmospheric mass, and they split in the ratio of the 2 rates;
- in one step no more melt than the melt mass reaches the crust, and below `phi_solid` none does;
- after a leak, the surface pressure is $M_\mathrm{atm}\,g/(4\pi R^2)$ and the mixing ratios follow the moles;
- the seal opens only past its band, or on a giant-impact step, and a held step runs no chemistry solve;
- the crust thickness on the mesh reaches the lower face of the top run of solid cells;
- on every held row the melt columns are 0 and the held masses are finite and not negative (an error stops the run otherwise);
- the per-element closure includes the held mass, and the held mass blocks the desiccation check.
