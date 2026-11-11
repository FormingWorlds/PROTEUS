# Volatiles held under a closed crust

This page answers one question: **when a crust closes over the magma ocean, can the volatiles dissolved in the melt still reach the atmosphere?**

The short answer: with `outgas.lid_seal.enabled = true`, a crust that is thick enough holds the dissolved volatiles in the melt below it. This is the **seal**. While the seal holds, the chemistry is frozen. The held volatiles reach the atmosphere only by a slow **leak**, which stops where the mantle holds as much of a volatile as the melt can dissolve at the pressure of the atmosphere.

The seal opens again in 3 cases: the crust thins, the top cell of the mesh holds a set share of melt again (35 % by default), or a giant impact happens. With the default, `false`, the seal never closes.

**Why it matters.** Without the seal, the chemistry keeps the melt in equilibrium with the atmosphere at every step, so a crust does not change where the volatiles go. With it, the atmosphere gets the gas of the melt only through the leak, which changes the atmospheric mass, the surface pressure and the mass that escape can take. In the [tutorial](../Tutorials/lid_seal.md) the effect is small, 0.43 % less water in the atmosphere at 140 kyr in each of 5 starts that differ only by round-off (their atmospheric water spreads by 0.64 %), because water is very soluble and most of it stays in the mantle in both runs.

The seal is one path of the outgassing step; [Outgassing, rock vapour and volatile trapping](outgas.md) gives the order of the steps around it.

## Terms used on this page

- **Seal**: the state in which a closed crust holds the volatiles of the melt below it.
- **Held mass**: the mass of a species or element that the seal holds below the crust, in the `_kg_sealed` columns.
- **Floor**: the mass of a species that the mantle holds in equilibrium with the atmosphere, in the `_kg_eq` columns. The leak does not go below it.
- **CALLIOPE**: the outgassing chemistry module, which splits the volatiles between the atmosphere and the melt; the seal uses its solubility laws.
- **Aragog**: the interior solver; its mesh is a set of radial cells, and the seal reads their melt fraction.

## When the seal closes

In each step the seal reads 2 quantities; it does not close during the initialisation stage:

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
- **On a coarse mesh, 2 of the opening rules never fire.** A crust on the mesh means a solid top cell, and a solid cell is below `phi_close`, so `phi_close` does not act. A held mesh crust is at least one cell thick, which is above the default `delta_open`, so `delta_open` does not act either. The seal then opens only when the top cell holds `phi_open` of melt, or on an impact.

## The leak

While the seal holds, the held mass $M_i$ \[kg\] of each species $i$ leaks toward a floor $M_{\mathrm{eq},i}$ \[kg\], the mass that the mantle holds in equilibrium with the present atmosphere:

$$
\frac{dM_i}{dt} = -\left(\frac{1}{\tau} + k_i\right) \max\left(M_i - M_{\mathrm{eq},i},\ 0\right),
\qquad
k_i = \frac{\chi_d\, \dot M_c}{R_i},
\qquad
R_i = M_\mathrm{melt} + D_i\, M_\mathrm{solid}.
$$

The symbols are:

- $\tau$: `tau_leak` \[yr\], the relaxation time of the held volatiles. The default, infinity, holds them.
- $\dot M_c$: `crust_rate` \[kg yr$^{-1}$\], the mass of melt that reaches the crust per year. The default is 0.
- $\chi_d$: `chi_d` \[1\], the fraction of the volatiles in that melt that degasses; 0.02 by default, from Table 3 of Schaefer & Sasselov (2015)[^cite-schaefer2015].
- $M_\mathrm{melt} = \Phi M_\mathrm{mantle}$ and $M_\mathrm{solid} = (1 - \Phi) M_\mathrm{mantle}$: the melt and the solid mass of the mantle \[kg\], with the global melt fraction $\Phi$.
- $D_i$: `outgas.D_const_<species>`, the crystal/melt partition coefficient \[1\].
- $R_i$: the reservoir of species $i$ \[kg\], the melt and the crystals in equilibrium with it; with front trapping only the crystals formed after closure ([The floor of the leak](#the-floor-of-the-leak)).
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
- with front trapping (`outgas.trap_mode = "front"`) the crystals that froze before closure keep their own trapped volatiles, so the reservoir counts only the crystals formed after it: $R_i = M_\mathrm{melt} + D_i \max(\Phi_c - \Phi, 0)\, M_\mathrm{mantle}$, with $\Phi_c$ the global melt fraction at closure (`seal_phi_close`). Above $\Phi_c$ it counts the melt alone: remelted crystals have returned their trapped mass to the held mass. The volcanic rate $k_i$ takes the same reservoir.

The floor is one way: a species below its floor does not move, because gas does not dissolve back through the crust.

**Example.** Without front trapping (the default), in the [tutorial](../Tutorials/lid_seal.md) the seal closes at $\Phi_c = 0.19250$. The held water is the dissolved water of the melt, while its floor also counts the crystals at $D = 0.0017$. The floor is therefore 1.0071 times the held water, and water starts to leak only when $\Phi$ falls below 0.19113. CO$_2$ ($D = 0$) leaks from the first held step.

## One step of the leak

The floor rises with the gas the leak puts into the atmosphere: for water at the closure of the tutorial, 1 kg more in the atmosphere raises the floor by about 3200 kg. A floor taken at the start of a step would therefore let a step leak far past the end state. The code takes the floor at the end of the step instead. Over a step of length $\Delta t$ it solves for the leak $L_i$ \[kg\]:

$$
L_i = \left(1 - e^{-\lambda_i \Delta t}\right) \max\left(M_i - M_{\mathrm{eq},i}(L),\ 0\right),
\qquad \lambda_i = \frac{1}{\tau} + k_i.
$$

- For a constant floor this is the exact solution, whatever the step length.
- For $\tau \to 0$ it gives the gas-melt equilibrium of the frozen species; for $\lambda_i = 0$ nothing moves.
- A root finder (Brent's method) finds $L_i$ for each species; sweeps over the species repeat until no leak changes by more than $10^{-14}$ of its held mass, because the floors couple through the surface pressure and the mean molar mass.

After each step the code checks that every species that leaked ends at or above its floor, at $e^{-\lambda_i \Delta t}$ of its starting distance from it, and that a sample of its step shows one solution ([Laws that fall with pressure](#laws-that-fall-with-pressure)); otherwise the run stops with an error. With `tau_leak` infinite and `crust_rate = 0` the step computes no floor and changes no column.

## Limits of the leak

- **Laws that fall with pressure.** The CO$_2$, CO and CH$_4$ laws fall with pressure at high pressure; [Laws that fall with pressure](#laws-that-fall-with-pressure) says when a step can then have more than one solution.
- **First order in the step.** Within a step the floor moves for 3 reasons: the melt fraction falls, the leak fills the atmosphere, and escape thins it. The step takes the floor of its end state, so its error falls in proportion to $\Delta t$ where the step is short against the leak ($\lambda_i \Delta t < 0.1$); at $\tau$ = 100 yr, where $\lambda_i \Delta t$ is 3.6, 4 times more steps cut it only 2.35 times. In the test on a melt-fraction path like that of the tutorial (120 steps of 358 yr, escape off) the error is at most $4.4\times10^{-5}$ of the inventory of each species, which is the held plus the atmospheric mass; per held or atmospheric mass it can reach $1.9\times10^{-3}$. [Validation](../Validation/outgas/lid_seal.md#the-step-converges-to-the-continuous-law) gives the table.
- **Crystals at closure.** The CALLIOPE solve dissolves volatiles in the melt only. Without front trapping, $R_i$ also counts the crystals, so at closure the held mass is below its floor by the factor $1 + D_i (1 - \Phi)/\Phi$. With front trapping the factor is 1 at closure.
- **Crystals at opening.** When the seal opens, the whole held mass returns to the melt, the share counted in the crystals included.
- **Erupted melt.** Of the volatile in the melt that reaches the crust, only the degassed share leaves the held mass; the rest stays counted as held.
- **Frozen chemistry.** The held and leaked gas keep their species; the floor does not see a change of speciation in the atmosphere.

## Laws that fall with pressure

The CO$_2$ law of Dixon et al. (1995)[^cite-dixon1995] falls with pressure above $83.15\,T/23$ bar, about 5400 bar at 1500 K. At a fixed composition the CO and CH$_4$ laws also fall with the surface pressure, CO above about 7000 bar and CH$_4$ above about 5200 bar. A fall alone does not give the step a second solution: the floor must fall faster than the leak adds gas to the atmosphere. The code checks each step for more than one solution and stops the run if it finds one; [Validation](../Validation/outgas/lid_seal.md#invariants-checked-alongside) says what this check can and cannot see.

## Settings of the seal

The settings of `[outgas.lid_seal]` and their defaults ([full reference](../Reference/config/escape_outgas.md)):

| Setting | Meaning | Default |
|---|---|---|
| `delta_close` | crust thickness at or above which the seal can close \[m\] | 10 km |
| `phi_close` | the top mesh cell must hold less melt than this to close \[1\] | 0.3 |
| `phi_open` | a closed seal opens at or above this top-cell melt fraction \[1\] | `phi_close + 0.05` |
| `delta_open` | a closed seal opens below this crust thickness \[m\] | `0.5 delta_close` |
| `phi_solid` | a mesh cell below this melt fraction counts as crust \[1\] | 0.001 |
| `tau_leak` | relaxation time of the leak \[yr\] | infinite |
| `crust_rate` | melt mass that reaches the crust per year \[kg yr$^{-1}$\] | 0 |
| `chi_d` | share of the volatiles of that melt that degasses \[1\] | 0.02 |

## How to turn the seal on and read the result

The seal needs the Aragog interior solver, because it reads the melt fraction and the radius of each mesh cell:

```toml
[interior_energetics]
module = "aragog"

[outgas.lid_seal]
enabled = true
delta_close = 1e4     # crust thickness at which the seal closes [m]
tau_leak = 1e5        # relaxation time of the held volatiles [yr]
```

The seal needs `outgas.module = "calliope"`, an escape reservoir of `"outgas"` or `"bulk"`, and, with the Zalmoxis structure, `interior_struct.zalmoxis.dry_mantle = true`. It cannot be combined with `params.stop.solid.freeze_volatiles = true`, which also freezes the melt inventory, or with `outgas.vapourise = true`. Front trapping (`outgas.trap_mode = "front"`) can run with it. While the seal holds, the melt columns are empty, so trapping buries nothing; mass that a remelt releases from the solid joins the held mass.

These output columns show what the seal does ([full list](../Reference/output.md)):

- `seal_active` \[1\]: 1 while the seal holds, else 0.
- `seal_delta` \[m\] and `seal_phi_top` \[1\]: the crust thickness and the melt fraction of the top mesh cell that the seal reads in the step. On the mesh, `seal_delta` is the depth of the top run of cells below `phi_solid` while the seal is open, and below `phi_open` while it holds and on the step where it opens, so it can jump on the first held step.
- `seal_path` \[1\]: the crust that a closed seal reads in the step: 1 for the crust on the mesh, 2 for the sub-grid lid; 0 while the seal is open, also when `seal_delta` comes from the sub-grid lid.
- `{sp}_kg_sealed`, `{sp}_mol_sealed` and `{e}_kg_sealed`: the held mass of each species and element.
- `{sp}_kg_eq` \[kg\]: the floor of each soluble species at the end of the step; 0 while the seal is open, on the step where it closes, with `outgas.calliope.solubility = false`, and on held steps with no leak rate (`tau_leak` infinite and no crust production) or nothing held.
- `seal_Tmagma_close` \[K\]: the magma temperature of the chemistry solve that closed the seal.
- `seal_phi_close` \[1\]: the global melt fraction of the step that closed the seal.
- `{e}_kg_leak_tau` and `{e}_kg_leak_volc` \[kg\]: the mass of each element that leaked in the step by relaxation and by volcanism.
- `seal_leak_kg_cum` \[kg\]: the leaked mass since the start of the run.
- `seal_open_kg` \[kg\]: the held mass that returned to the melt in a step where the seal opened.

## What the seal changes elsewhere

- Escape takes mass from the atmosphere only, as on a mantle frozen with `freeze_volatiles`.
- The escape step does not count the held mass; the desiccation check does, so a planet that holds volatiles under its crust is not desiccated. If the planet desiccates, a held rest below `outgas.mass_thresh` leaves with the atmosphere.
- The per-element mass closure includes the held mass: atmosphere + melt + solid + condensed + held = total.

**Where this is checked.** [Volatiles under a closed crust](../Validation/outgas/lid_seal.md).

[^cite-schaefer2015]: L. Schaefer, D. Sasselov, *[The persistence of oceans on Earth-like planets: insights from the deep-water cycle](https://doi.org/10.1088/0004-637X/801/1/40)*, The Astrophysical Journal, 801, 40, 2015. [SciX](https://scixplorer.org/abs/2015ApJ...801...40S/abstract).
[^cite-gaillard2022]: F. Gaillard, F. Bernadou, M. Roskosz, M. A. Bouhifd, Y. Marrocchi, G. Iacono-Marziano, M. Moreira, B. Scaillet, G. Rogerie, *[Redox controls during magma ocean degassing](https://doi.org/10.1016/j.epsl.2021.117255)*, Earth and Planetary Science Letters, 577, 117255, 2022. [SciX](https://scixplorer.org/abs/2022E%26PSL.57717255G/abstract).
[^cite-dixon1995]: J. E. Dixon, E. M. Stolper, J. R. Holloway, *[An experimental study of water and carbon dioxide solubilities in mid-ocean ridge basaltic liquids. Part I: Calibration and solubility models](https://doi.org/10.1093/oxfordjournals.petrology.a037267)*, Journal of Petrology, 36, 1607-1631, 1995. [SciX](https://scixplorer.org/abs/1995JPet...36.1607D/abstract).
