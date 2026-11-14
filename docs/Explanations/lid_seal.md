# Volatiles held under a closed crust

This page answers one question: **when a crust closes over the magma ocean, can the volatiles dissolved in the melt still reach the atmosphere?**

The short answer: with `outgas.lid_seal.enabled = true`, a crust that is thick enough holds the dissolved volatiles in the melt below it. This is the **seal**. While the seal holds, the chemistry is frozen. The held volatiles reach the atmosphere only by a slow **leak**, which stops where the melt holds as much of a volatile as it can dissolve at the pressure of the atmosphere.

The seal opens again in 3 cases: the crust thins, the top cell of the mesh holds a set share of melt again (35 % by default), or a giant impact happens. With the default, `false`, the seal never closes.

**Why it matters.** Without the seal, the chemistry keeps the melt in equilibrium with the atmosphere at every step, so a crust does not change where the volatiles go. With the seal, the atmosphere gets the gas of the melt only through the leak. This changes the atmospheric mass, the surface pressure and the mass that escape can take. In the [tutorial](../Tutorials/lid_seal.md) the effect is small: at 140 kyr the seal run holds 1.730 % more atmospheric water than the open run of its start in each of 6 starts that differ by round-off; [Step 6](../Tutorials/lid_seal.md#step-6-compare-the-2-atmospheres) gives their spread. The largest term is the frozen chemistry of the held gas ([Limits of the leak](#limits-of-the-leak)).

The seal is one path of the outgassing step; [Outgassing, rock vapour and volatile trapping](outgas.md) gives the order of the steps around it.

## Terms used on this page

- **Seal**: the state in which a closed crust holds the volatiles of the melt below it.
- **Held mass**: the mass of a species or element that the seal holds below the crust, in the `_kg_sealed` columns.
- **Floor**: the mass of a species that the melt holds in equilibrium with the atmosphere, in the `_kg_eq` columns; with front trapping it also counts the solid formed since closure. The leak does not go below it.
- **Front trapping**: the option `outgas.trap_mode = "front"`, in which the freezing mantle keeps a share of each volatile; [Volatile trapping in the freezing mantle](outgas.md#volatile-trapping-in-the-freezing-mantle) explains it.
- **Open run**: the same planet with the seal off; its chemistry keeps the melt in equilibrium with the atmosphere at every step.
- **CALLIOPE**: the outgassing chemistry module, which splits the volatiles between the atmosphere and the melt; the seal uses its solubility laws.
- **Aragog**: the interior solver; its mesh is a set of radial cells, and the seal reads their melt fraction.

## When the seal closes

In each step the seal reads 2 quantities; it does not close during the initialisation stage or in a step with a giant impact:

- the crust thickness $\delta$ \[m\];
- the melt fraction $\phi_\mathrm{top}$ \[1\] of the top cell of the interior mesh.

Where $\delta$ comes from depends on the state of the crust. The Aragog sub-grid lid and its weight `lid_weight` are explained in [Sub-grid lid and the surface skin](lid_coupling.md).

| State of the crust | Thickness $\delta$ | What decides the closure |
|---|---|---|
| Aragog sub-grid lid acts (`lid_weight > 0`), and the seal is open or was closed by the lid | `lid_delta`, the lid above the top mesh cell | both $\delta$ and $\phi_\mathrm{top}$ |
| crust on the mesh (no sub-grid lid) | depth of the top run of mesh cells with a melt fraction below `phi_solid` (below `phi_open` while the seal holds) | $\delta$ |
| no crust | 0 | the seal stays open |

The seal closes when $\delta \ge$ `delta_close` and $\phi_\mathrm{top} <$ `phi_close`. It closes after the chemistry solve of that step, so it holds the melt inventory of the state where the rule fires. A thin crust over a molten top cell does not close the seal.

The default `delta_close` is 10 km. A coarse mesh moves the crust thickness in steps of one cell. On the 79-cell mesh of the [tutorial](../Tutorials/lid_seal.md) the top cell is 65.8 km thick (the mesh is not uniform in radius). So the seal closes when the top cell freezes, for any `delta_close` below 65.8 km, 0.3 m included. Only the Aragog sub-grid lid can see a crust thinner than one cell, so a thin chill crust closes the seal earlier only through that lid.

The seal covers the whole surface at once. A planet with a molten dayside and a frozen nightside, without heat transport between them, is outside this rule.

## A small sub-grid lid

An open seal reads the sub-grid lid at any `lid_weight` above 0, however small. Aragog writes a weight of $10^{-10}$ or less as 0, so in a run the smallest weight the seal reads is just above $10^{-10}$. A lid with a small weight can therefore close the seal. The seal then holds the whole melt inventory, not a share scaled by the weight.

## When the seal opens

A closed seal opens in 3 cases:

- the top mesh cell holds `phi_open` of melt or more (default `phi_close + 0.05`);
- the crust is thinner than `delta_open` (default `0.5 delta_close`);
- a giant impact happens in the step.

The gap between the close and the open values stops a seal that would otherwise open and close in each step. On opening, the held mass returns to the melt, and the chemistry solve of the same step takes it. The atmosphere then gets the dissolved gas of the melt at once.

**Which crust a closed seal reads.**

- On the mesh, a closed seal counts every top cell with a melt fraction below `phi_open` as crust, and an open seal only those below `phi_solid`. So a top cell that crosses `phi_solid` does not open the seal.
- A seal closed on the mesh keeps reading the mesh and ignores a sub-grid lid that appears later.
- A seal closed by the sub-grid lid reads the lid while `lid_weight > 0`. From the first step where `lid_weight` is 0 it reads the mesh crust below `phi_open`, and it stays on the mesh after that. That crust need not hold a solid cell, so the seal can stay closed while every mesh cell holds some melt.
- **On a coarse mesh, `phi_close` and `delta_open` never decide anything.** A crust on the mesh means a solid top cell, and a solid cell is below `phi_close`, so the `phi_close` condition of the closure is always met. A held mesh crust is at least one cell thick, which is above the default `delta_open`, so `delta_open` does not act. The seal then opens only when the top cell holds `phi_open` of melt, or on an impact.

## The leak

While the seal holds, the held mass $M_i$ \[kg\] of each species $i$ leaks toward a floor $M_{\mathrm{eq},i}$ \[kg\], the mass that the melt holds in equilibrium with the present atmosphere:

$$
\frac{dM_i}{dt} = -\left(\frac{1}{\tau} + k_i\right) \max\left(M_i - M_{\mathrm{eq},i},\ 0\right),
\qquad
k_i = \frac{\chi_d\, \dot M_c}{R_i},
\qquad
R_i = M_\mathrm{melt}.
$$

The symbols are:

- $\tau$: `tau_leak` \[yr\], the relaxation time of the held volatiles. The default, infinity, holds them.
- $\dot M_c$: `crust_rate` \[kg yr$^{-1}$\], the mass of melt that reaches the crust per year. The default is 0. It is 0 when $\Phi$ is below `phi_solid`, and in one step it is cut to the melt mass; `seal_volc_capped` marks such a step.
- $\chi_d$: `chi_d` \[1\], the fraction of the volatiles in that melt that degasses; 0.02 by default, from Table 3 of Schaefer & Sasselov (2015)[^cite-schaefer2015].
- $M_\mathrm{melt} = \Phi M_\mathrm{mantle}$: the melt mass of the mantle \[kg\], with the global melt fraction $\Phi$.
- $R_i$: the reservoir of species $i$ \[kg\], the melt, as in the CALLIOPE solve; with front trapping it also counts the solid formed since closure, crystals and trapped melt ([The reservoir with front trapping](#the-reservoir-with-front-trapping)).
- $D_i$: `outgas.D_const_<species>`, the crystal/melt partition coefficient \[1\]; the seal uses it only with front trapping.
- $\lambda_i = 1/\tau + k_i$: the leak rate of species $i$ \[yr$^{-1}$\].

The first term relaxes the held mass with the time $\tau$. The second is volcanism: melt with the concentration $X_{\mathrm{melt},i} = M_i/R_i$ reaches the crust at the rate $\dot M_c$ and degasses the share $\chi_d$ of its excess over the solubility, $X_{\mathrm{melt},i} - X_{\mathrm{eq},i}$. Both terms act on the same distance to the floor, so the law is one relaxation. The leaked mass splits between them in the ratio $(1/\tau) : k_i$.

The volcanic term has the form of Eq. 14 and Eq. 17 of Schaefer & Sasselov (2015), with the melt production of plate spreading replaced by $\dot M_c$ and their degree of melting replaced by $\Phi$. Their Eq. 14 degasses $\chi_d X_{\mathrm{melt},i}$ with no solubility term. The limit to the excess over $X_{\mathrm{eq},i}$ and the adaptation to a mantle under a closed crust are parts of this model, not results of that paper.

## The floor of the leak

The floor is $M_{\mathrm{eq},i} = X_{\mathrm{eq},i}\, R_i$, with $X_{\mathrm{eq},i}$ \[1\] the mass fraction of species $i$ that the melt dissolves. It uses the solubility laws of the CALLIOPE chemistry solve ([solubility laws](https://proteus-framework.org/CALLIOPE/Explanations/solubility.html), [noble gases](https://proteus-framework.org/CALLIOPE/Explanations/noble_gases.html)):

- each law takes the partial pressure of the species in the atmosphere; the CO, CH$_4$ and N$_2$ laws also take the surface pressure;
- the CO$_2$ law also takes the temperature `seal_Tmagma_close` of the solve that closed the seal; the N$_2$ and S$_2$ laws take that temperature and its IW offset `fO2_shift_IW_derived` (IW: the iron-wüstite oxygen buffer);
- CO, CH$_4$ and the noble gases have a floor only where the matching `outgas.calliope.include_*` option is on;
- below $10^{-20}$ bar, where the CALLIOPE S$_2$ law returns 0, the S$_2$ floor follows the same square-root law of Gaillard et al. (2022)[^cite-gaillard2022] in its partial pressure, so it stays continuous;
- H$_2$, O$_2$, SO$_2$, H$_2$S and NH$_3$ have no law, so their floor is 0;
- with `outgas.calliope.solubility = false` every floor is 0, and the held mass decays at the rate $\lambda_i$ of the next section;
- without trapping (the default) the reservoir is the melt, $R_i = M_\mathrm{melt}$, because the CALLIOPE solve dissolves volatiles in the melt only. At closure each held mass therefore equals its floor;
- with front trapping (`outgas.trap_mode = "front"`) the reservoir also counts the solid formed since closure ([The reservoir with front trapping](#the-reservoir-with-front-trapping)). The volcanic rate $k_i$ takes the same reservoir.

The floor is one way: a species below its floor does not move, because gas does not dissolve back through the crust.

**Example.** Without front trapping (the default), in the [tutorial](../Tutorials/lid_seal.md) the seal closes at $\Phi_c = 0.19873$. The held water is the dissolved water of the melt and equals its floor. When $\Phi$ falls, the floor falls with it, so water, CO$_2$, CO and CH$_4$ leak from the first held step.

### The reservoir with front trapping

With front trapping, the solid that froze before closure keeps its own trapped volatiles and is not part of the reservoir. The reservoir counts only the solid formed since closure, as [trapping](outgas.md#the-burial-rule) buries it. A step that freezes the mass $\Delta M$ \[kg\] splits it by the trapped melt fraction $F_\mathrm{tl}$ \[1\] (`trap_F_tl`):

- $(1 - F_\mathrm{tl})\,\Delta M$ is crystals, which hold the species at $D_i$ times its melt concentration;
- $F_\mathrm{tl}\,\Delta M$ is trapped melt, which holds the species at the melt concentration.

With $M_\mathrm{xtal}$ (`seal_xtal_kg`) and $M_\mathrm{tl}$ (`seal_tl_kg`) the sums of these 2 masses since closure \[kg\], the reservoir is

$$
R_i = M_\mathrm{melt} + D_i M_\mathrm{xtal} + M_\mathrm{tl},
$$

so each step adds $[(1 - F_\mathrm{tl}) D_i + F_\mathrm{tl}]\,\Delta M$, the effective partition coefficient of trapping. A remelt returns the youngest solid first, as in trapping, and a remelt past the closing melt fraction leaves the melt alone. Both columns are 0 at closure and while the seal is open.

## One step of the leak

The floor rises with the gas the leak puts into the atmosphere: for water at the closure of the tutorial, 1 kg more in the atmosphere raises the floor by about 3200 kg. A floor taken at the start of a step would therefore let a step leak far past the end state. The code takes the floor at the end of the step instead. Over a step of length $\Delta t$ it solves for the leak $L_i$ \[kg\]:

$$
L_i = \left(1 - e^{-\lambda_i \Delta t}\right) \max\left(M_i - M_{\mathrm{eq},i}(L),\ 0\right),
\qquad \lambda_i = \frac{1}{\tau} + k_i.
$$

- For a constant floor this is the exact solution, whatever the step length.
- For $\tau \to 0$ it gives the gas-melt equilibrium of the frozen species; for $\lambda_i = 0$ nothing moves.
- A root finder (Brent's method) finds $L_i$ for each species; sweeps over the species repeat until no leak changes by more than $10^{-14}$ of its held mass, because the floors couple through the surface pressure and the mean molar mass.

After each step the code checks that every species that leaked ends at or above its floor, at $e^{-\lambda_i \Delta t}$ of its starting distance from it, and that a sample of the step of every species with held mass, a leak rate and a solubility law, also one whose leak came out as 0, shows one solution ([Laws that fall with pressure](#laws-that-fall-with-pressure)); otherwise the run stops with an error. With `tau_leak` infinite and `crust_rate = 0` the step computes no floor and changes no column.

## Limits of the leak

- **Laws that fall with pressure.** The CO$_2$, CO and CH$_4$ laws fall with pressure at high pressure; [Laws that fall with pressure](#laws-that-fall-with-pressure) says when a step can then have more than one solution.
- **First order in the step.** Within a step the floor moves for 3 reasons: the melt fraction falls, the leak fills the atmosphere, and escape thins it. The step takes the floor of its end state, so its error falls in proportion to $\Delta t$ where the step is short against the leak ($\lambda_i \Delta t < 0.1$). [Validation](../Validation/outgas/lid_seal.md#the-step-converges-to-the-continuous-law) gives the size of the error.
- **Crystals.** Without trapping the floor counts no crystals, as the CALLIOPE solve, so closure is a fixed point of the leak. With front trapping it counts the solid formed since closure at $(1 - F_\mathrm{tl}) D_i + F_\mathrm{tl}$ but at one melt concentration; layers fixed at the concentration of their own step, as in trapping, give 0.58 % more water in the atmosphere of the [tutorial setup with front trapping](../Validation/outgas/lid_seal.md#with-front-trapping-the-floor-counts-the-solid-formed-since-closure) at 140 kyr. At opening all held mass returns to the melt.
- **Erupted melt.** Of the volatile in the melt that reaches the crust, only the degassed share leaves the held mass; the rest stays counted as held.
- **Frozen chemistry.** The held and leaked gas keep the species they had at closure; the floor does not see a change of speciation in the atmosphere. The open run keeps its atmosphere in equilibrium with the redox buffer of the melt, which gives oxygen to the gas as the magma cools and turns CO into CO$_2$. In the [tutorial](../Tutorials/lid_seal.md) at 140 kyr the sealed atmosphere therefore holds 24.7 % more CO, 6.3 % less CO$_2$ and 1.7 % more water than the open run, at a 2.6 % lower surface pressure. A gas that re-speciates at fixed atmospheric elements does not close this gap, because the oxygen comes from the melt ([Validation](../Validation/outgas/lid_seal.md#the-tutorial-atmosphere-lies-above-the-held-run-and-crosses-the-open-run)).

## Laws that fall with pressure

The CO$_2$ law of Dixon et al. (1995)[^cite-dixon1995] falls with pressure above $83.15\,T/23$ bar, about 5400 bar at 1500 K. At a fixed composition the CO and CH$_4$ laws also fall with the surface pressure, CO above about 7000 bar and CH$_4$ above about 5200 bar. A fall alone does not give the step a second solution: the floor must fall faster than the leak adds gas to the atmosphere. The code checks each step for more than one solution and stops the run if it finds one; [Validation](../Validation/outgas/lid_seal.md#invariants-checked-alongside) says what this check can and cannot see.

## Settings of the seal

The settings of `[outgas.lid_seal]`, with their units and defaults, are in the [config reference](../Reference/config/escape_outgas.md#lid-seal-outgaslid_seal). The 3 you change first are `tau_leak` (default infinite: the volatiles stay held), `crust_rate` (default 0: no volcanic leak) and `delta_close` (default 10 km).

## What the seal needs

The seal needs the Aragog interior solver, because it reads the melt fraction and the radius of each mesh cell. [Lid seal](../How-to/usage_lid_seal.md) gives the settings that turn it on, the settings it needs, and the output columns that show what it does. Front trapping (`outgas.trap_mode = "front"`) can run with it. While the seal holds, the melt columns are empty, so trapping buries nothing; mass that a remelt releases from the solid joins the held mass.

## What the seal changes elsewhere

- Escape takes mass from the atmosphere only, as on a mantle frozen with `freeze_volatiles`.
- The escape step does not count the held mass; the desiccation check does, so a planet that holds volatiles under its crust is not desiccated. If the planet desiccates, a held rest below `outgas.mass_thresh` leaves with the atmosphere.
- The per-element mass closure includes the held mass: atmosphere + melt + solid + condensed + held = total.

**Where this is checked.** [Volatiles under a closed crust](../Validation/outgas/lid_seal.md).

[^cite-schaefer2015]: L. Schaefer, D. Sasselov, *[The persistence of oceans on Earth-like planets: insights from the deep-water cycle](https://doi.org/10.1088/0004-637X/801/1/40)*, The Astrophysical Journal, 801, 40, 2015. [SciX](https://scixplorer.org/abs/2015ApJ...801...40S/abstract).
[^cite-gaillard2022]: F. Gaillard, F. Bernadou, M. Roskosz, M. A. Bouhifd, Y. Marrocchi, G. Iacono-Marziano, M. Moreira, B. Scaillet, G. Rogerie, *[Redox controls during magma ocean degassing](https://doi.org/10.1016/j.epsl.2021.117255)*, Earth and Planetary Science Letters, 577, 117255, 2022. [SciX](https://scixplorer.org/abs/2022E%26PSL.57717255G/abstract).
[^cite-dixon1995]: J. E. Dixon, E. M. Stolper, J. R. Holloway, *[An experimental study of water and carbon dioxide solubilities in mid-ocean ridge basaltic liquids. Part I: Calibration and solubility models](https://doi.org/10.1093/oxfordjournals.petrology.a037267)*, Journal of Petrology, 36, 1607-1631, 1995. [SciX](https://scixplorer.org/abs/1995JPet...36.1607D/abstract).
