# Mantle redox evolution

This page describes how PROTEUS tracks the redox state of a crystallizing magma ocean and uses it to set the oxygen fugacity ($f_{\rm O_2}$) of outgassing.

The implementation is in `src/proteus/interior_chem/redox.py`.

See [Planet configuration](../Reference/config/planet.md) for configuration fields and [Output reference](../Reference/output.md) for output columns.

## Why track the redox state

With the default `planet.fO2_source = "user_constant"`, outgassing uses a fixed oxygen-fugacity offset from the iron-wüstite (IW) buffer:

`outgas.fO2_shift_IW`.

In a crystallizing magma ocean, this offset can evolve. Fe$^{3+}$ and Fe$^{2+}$ partition differently between the melt and newly formed crystals, changing the Fe$^{3+}$/Fe$^{2+}$ ratio of the remaining melt and therefore its $f_{\rm O_2}$.

Setting `planet.fO2_source = "from_mantle_redox"` enables this calculation. PROTEUS then derives the outgassing $f_{\rm O_2}$ from the tracked Fe$^{3+}$/Fe$^{2+}$ ratio at every coupling step.

!!! info "Physical picture"

    As the magma ocean freezes, each new crystal removes some iron from the melt. Crystals take up Fe$^{2+}$ more readily than Fe$^{3+}$, so the shrinking melt is left progressively richer in ferric iron.

    The tracker follows this in two steps:

    1. **In the interior,** it follows how crystallization changes the melt Fe$^{3+}$/Fe$^{2+}$ ratio.
    2. **At the surface,** it converts that ratio into an oxygen fugacity through the equilibrium FeO + ¼O$_2$ ⇌ FeO$_{1.5}$.

    The result is a single number for the outgassing code: the offset $\Delta$IW of the surface $f_{\rm O_2}$ from the iron-wüstite buffer.

<figure markdown="span">
  ![Model domain seen by the redox tracker](../assets/redox/redox_domain.svg#only-light){ width="760" }
  ![Model domain seen by the redox tracker](../assets/redox/redox_domain_dark.svg#only-dark){ width="760" }
  <figcaption>Model domain as seen by the redox tracker (schematic; the cell count, the depths of the 22 and 180 GPa lines and the base of the melt are illustrative). The interior module supplies the mass, pressure, temperature and melt fraction of each cell. The iron of the melt is held in two global reservoirs, shared by the melt cells in proportion to their melt mass. Solid formed in a cell takes iron with the partition coefficients of that cell's depth and returns it if the cell remelts. With the metal step on, metal forms in one melt cell, the base of the melt by default. The surface oxygen fugacity follows from the melt redox ratio at 1 bar, and CALLIOPE converts its offset from the IW buffer back with the same buffer.</figcaption>
</figure>

!!! note "Approach based on Schaefer et al. (2024)[^cite-schaefer2024]"

    Schaefer, Pahlevan and Elkins-Tanton track Fe$^{3+}$ and Fe$^{2+}$ during fractional crystallization using mineral/melt partition coefficients. They then convert the melt ferric/ferrous ratio to surface $f_{\rm O_2}$ using Hirschmann (2022)[^cite-hirschmann2022].

    In upper-mantle peridotite Fe$^{2+}$ is more compatible than Fe$^{3+}$, so crystallization there preferentially removes Fe$^{2+}$ and the remaining melt becomes more oxidized. In their lower-mantle and transition-zone minerals Fe$^{3+}$ is taken up more readily, so the melt Fe$^{3+}$/Fe$_{\rm T}$ first falls and then rises once the upper mantle crystallizes. The magnitude of these effects depends strongly on the poorly constrained Fe$^{3+}$ partitioning into lower-mantle minerals, especially bridgmanite.

    PROTEUS follows this approach with the simplifications listed in [Assumptions and limitations](#assumptions-and-limitations).

## Configuration

| Key | Meaning | Default |
|---|---|---|
| `planet.fO2_source` | Set to `"from_mantle_redox"` to enable the tracker. Other values disable it. | `"user_constant"` |
| `planet.interior_chem.ferric_fraction_initial` | Initial melt Fe$^{3+}$/Fe$_{\rm T}$, $f_0$, in the interval $(0,1)$. | `0.1` |
| `planet.interior_chem.metal_saturation` | Off by default. `true` adds a check for Fe-metal saturation (3FeO = 2FeO$_{1.5}$ + Fe) between crystallization steps, following Schaefer et al. (2024, Section 2.7), and counts cells below $\phi=0.15$ as solid. This page describes the model with `false`; the module docstring of `interior_chem/redox.py` describes the metal step. | `false` |
| `planet.interior_chem.partition_coefficients` | Fe$^{2+}$ and Fe$^{3+}$ partition coefficients: `"mgsio3"` (bridgmanite and pyroxenes), `"bulk_BSE"` (bulk values of the Schaefer et al. layers) or `"mgsio3_kd"` (`"mgsio3"` with a pressure- and temperature-dependent Fe$^{2+}$ coefficient in bridgmanite and post-perovskite); see [step 4](#4-partition-iron-into-the-new-solid). | `"mgsio3"` |
| `planet.interior_chem.fO2_pressure_term` | `true` adds $\int \Delta V\,dP$ of FeO + ¼O$_2$ = FeO$_{1.5}$ (melt EOS `redox_dV_eos`) to the radial fO$_2$ profile, and takes its ΔIW against the IW buffer at the cell pressure, without the metal step. The profile is a diagnostic: Fe$^{3+}$/Fe$_{\rm T}$ and the 1 bar surface fO$_2$ are unchanged. `false` evaluates the profile at 1 bar. Always on with `metal_saturation = true`. | `true` |

The tracker needs a melt fraction for every radial cell, so `interior_energetics.module` must be `spider` or `aragog`.

The default $f_0=0.1$ is the nominal Earth value from Schaefer et al. (2024). Hirschmann (2022) estimates Fe$^{3+}$/Fe$_{\rm T}=0.034$-$0.10$ for the Earth's magma ocean at core formation.

## Model workflow

The tracker runs once per coupling step inside `run_interior`, after the interior module provides the radial profiles:

- melt fraction $\phi_i$,
- cell mass $m_i$,
- pressure $P_i$,
- temperature $T_i$.

It carries three pieces of state between steps:

- total melt Fe$^{2+}$, $n^{2+}$,
- total melt Fe$^{3+}$, $n^{3+}$,
- previous-step melt fraction in each cell, $\phi_i^{\rm prev}$.

!!! abstract "Algorithm: one call of `update_melt_redox`"

    1. If `planet.fO2_source` is not `"from_mantle_redox"`, return.
    2. Read $\phi_i$, $m_i$, $P_i$ and $T_i$ from the interior module.
    3. **First call:** initialize $n^{2+}$, $n^{3+}$, $f=f_0$ and the per-cell $D_i$ ([step 1](#1-initialize-the-iron-inventory)). Go to 6.
    4. **Later calls:** if the global melt fraction is below 0.05, hold the state and go to 6. Otherwise:
        1. Return to the melt the iron of the solid that remelted since the last tracked call.
        2. Compute the newly formed solid $\Delta M_{s,i}$ from the decrease in $\phi_i$ ([step 2](#2-determine-newly-formed-solid)).
        3. Compute the uniform melt concentrations $C^{2+}$ and $C^{3+}$ ([step 3](#3-assume-a-homogeneous-melt)) and remove $\Delta n^{2+}$ and $\Delta n^{3+}$ into the new solid ([step 4](#4-partition-iron-into-the-new-solid)).
        4. If the melt would then hold more FeO + FeO$_{1.5}$ than its own mass, hold the state and go to 6.
        5. Update $n^{2+}$, $n^{3+}$, $f$ and $R$ ([step 5](#5-update-the-reservoirs-and-redox-ratio)) and set $\phi_i^{\rm prev}\leftarrow\phi_i$.
    5. **With `metal_saturation = true`:** run the metal step in one melt cell (module docstring of `interior_chem/redox.py`).
    6. Evaluate the radial $\log_{10}f_{\rm O_2}$ profile at each melt cell's $T_i$, at 1 bar or, with the pressure term, at $P_i$, and store the index of the uppermost melt cell (diagnostic only).
    7. Set $T_{\rm out}=\max(T_{\rm magma},T_{\rm floor})$. Evaluate the surface $f_{\rm O_2}$ from $R$ at $T_{\rm out}$ and 1 bar, then $\Delta{\rm IW}$ relative to the Hirschmann (2021) buffer.
    8. Write `ferric_frac_mantle` and `fO2_shift_IW_mantle` to the helpfile. CALLIOPE or atmodeller then uses $\Delta{\rm IW}$ as the outgassing $f_{\rm O_2}$ constraint; CALLIOPE applies it to the same Hirschmann (2021) buffer.

<figure markdown="span">
  ![Position of the redox tracker in one coupling step](../assets/redox/redox_flow.svg#only-light){ width="760" }
  ![Position of the redox tracker in one coupling step](../assets/redox/redox_flow_dark.svg#only-dark){ width="760" }
  <figcaption>One call of update_melt_redox in a PROTEUS coupling step. Warm boxes are tests on the tracker state; cool boxes are the computational steps described below. The first call and a held state go straight to the fO<sub>2</sub> evaluation, without the metal step. The radial profile (dashed) is a diagnostic: it is stored in the interior snapshot and does not feed the outgassing.</figcaption>
</figure>

### 1. Initialize the iron inventory

At the first call, the whole melt gets a uniform composition: a fixed mass fraction of FeO, plus enough FeO$_{1.5}$ that a fraction $f_0$ of the iron is ferric. These two numbers become the global iron reservoirs that every later step modifies.

On the first call, the total melt mass is

$$
M_{\rm melt}=\sum_i \phi_i m_i.
$$

The iron reservoirs are initialized as

$$
n^{2+}=\frac{w_{\rm FeO}M_{\rm melt}}{\mu_{\rm FeO}},
\qquad
n^{3+}=n^{2+}\frac{f_0}{1-f_0},
$$

where

- $w_{\rm FeO}=0.08$ is the FeO mass fraction,
- $\mu_{\rm FeO}=71.84$ g mol$^{-1}$,
- $f_0$ is `planet.interior_chem.ferric_fraction_initial`.

Fe$^{3+}$ is added on top of the ferrous inventory, so

$$
\frac{n^{3+}}{n^{2+}+n^{3+}}=f_0.
$$

The adopted $w_{\rm FeO}$ is the rounded bulk-silicate-Earth value from Schaefer et al. (2024) Table 1 (7.82 wt%).

On a single-cation basis, one mole of Fe$^{2+}$ is one mole of FeO units and one mole of Fe$^{3+}$ is one mole of FeO$_{1.5}$ units. So $n^{2+}$ and $n^{3+}$ count both iron ions and oxide units.

The per-cell partition coefficients $D_i^{2+}$ and $D_i^{3+}$ (step 4) are also fixed at this call, from the pressure profile of the first step.

### 2. Determine newly formed solid

Steps 2 to 5 run on every later call while the global melt fraction is at least 0.05 (see [step 5](#5-update-the-reservoirs-and-redox-ratio)). They are a step-by-step form of fractional crystallization: the solid formed in one step equilibrates with the residual melt of that step, as in Schaefer et al. (2024), and is then removed from contact with the melt and never re-equilibrates with it.

The interior grid keeps each cell's mass $m_i$ fixed in time. The melt mass in a cell was therefore $\phi_i^{\rm prev}m_i$ at the previous call and is $\phi_i m_i$ now. Any decrease is solid formed during this step.

If a cell's melt fraction decreases between two steps, the corresponding solid mass is

$$
\Delta M_{s,i}
=
m_i\max\left(\phi_i^{\rm prev}-\phi_i,0\right).
$$

The tracker uses the melt fraction returned by the interior module directly. A cell is fully solid only when $\phi_i=0$.

If a cell remelts, its increasing melt fraction does not create new solid. Instead it returns iron to the melt: the tracker keeps, per cell, the solid mass it has formed, $M_{s,i}$, and the Fe$^{2+}$ and Fe$^{3+}$ that solid holds, $n_{s,i}^{2+}$ and $n_{s,i}^{3+}$. A rise in melt fraction remelts the mass $m_i(\phi_i-\phi_i^{\rm prev})$, taken first from $M_{s,i}$, the most recently formed solid of the cell. Treating that solid as homogeneous, the cell returns

$$
\Delta n_i^{2+,3+}
=
\frac{\min\left[m_i(\phi_i-\phi_i^{\rm prev}),\,M_{s,i}\right]}{M_{s,i}}\,n_{s,i}^{2+,3+}
$$

to the melt reservoirs. Solid that was present at the first step holds no tracked iron, so remelting it adds melt mass but no iron. Fe metal is not redissolved. Metal that forms leaves the melt: from the next step on, the melt mass over which the reservoirs are spread (Steps 4-6) and against which Step 3d weighs the iron oxides is the interior's melt mass less the metal formed so far, $\left(3\mu_{\rm FeO}-2\mu_{\rm FeO_{1.5}}\right)\sum_i n_{{\rm met},i}$. The metal step's MgSiO$_3$ count is unchanged by this, since the metal takes none with it.

### 3. Assume a homogeneous melt

The melt is assumed to be well mixed by convection, following Schaefer et al. (2024, Section 3.1).

The Fe$^{3+}$ and Fe$^{2+}$ concentrations are therefore uniform throughout the melt:

$$
C^{3+}=\frac{n^{3+}}{M_{\rm melt}^{\rm prev}},
\qquad
C^{2+}=\frac{n^{2+}}{M_{\rm melt}^{\rm prev}},
$$

with

$$
M_{\rm melt}^{\rm prev}=\sum_i\phi_i^{\rm prev}m_i.
$$

Therefore, Fe$^{3+}$/Fe$^{2+}$ ratio is the same everywhere.

In practice, the global reservoirs from the previous step are shared among the cells that contained melt, in proportion to their melt mass $\phi_i^{\rm prev}m_i$. The melt mass of each cell cancels when its iron is divided by its melt mass, which is why every melt cell has the same concentration.

The previous step's melt is used because the new solid crystallized from that melt.

### 4. Partition iron into the new solid

A partition coefficient is the ratio of a species' concentration in the crystallizing solid to its concentration in the melt, $D=C_{\rm solid}/C_{\rm melt}$. Following Schaefer et al. (2024), mass and mole ratios are treated as proportional.

Cell $i$ holds the share $n_i=n\,\phi_i^{\rm prev}m_i/M_{\rm melt}^{\rm prev}$ of each reservoir, and its new solid is the fraction $s_i=\Delta M_{s,i}/(\phi_i^{\rm prev}m_i)$ of its melt. The new solid equilibrates with the residual melt of the cell at concentration ratio $D$, so the mass balance $n_i = D\,C_{\rm res}\Delta M_{s,i} + C_{\rm res}(1-s_i)\phi_i^{\rm prev}m_i$ gives the moles entering the solid:

$$
\Delta n^{3+}
=
\sum_i n_i^{3+}\,\frac{s_i D_i^{3+}}{1-s_i+s_i D_i^{3+}},
$$

$$
\Delta n^{2+}
=
\sum_i n_i^{2+}\,\frac{s_i D_i^{2+}}{1-s_i+s_i D_i^{2+}}.
$$

Each term lies between 0 and $n_i$ for any $s_i$ and $D_i\geq0$, and a cell whose melt freezes completely ($s_i=1$) moves all of its share into the solid. The same step is Schaefer et al.'s mass balance, so fed their coefficients and solid fractions the tracker reproduces their run to round-off.

Both coefficients depend on the mineral assemblage, selected from the cell pressure at the first step. `planet.interior_chem.partition_coefficients` chooses between three prescriptions; the third, `"mgsio3_kd"`, recomputes the lower-mantle $D^{2+}$ at every step.

#### `"mgsio3"` (default): bridgmanite and pyroxenes

| Assemblage | Pressure | $D^{3+}$ | $D^{2+}$ | Source |
|---|---:|---:|---:|---|
| Bridgmanite | $P\geq22$ GPa | 0.75 | 0.44 | Schaefer et al. (2024), Table 3 ($D^{3+}$) and Table S1 ($D^{2+}$) |
| Clinopyroxene + orthopyroxene | $P<22$ GPa | 0.3825 | 0.535 | Mean of Cpx and Opx values |

The shallow-mantle value is

$$
D^{3+}=0.3825
=
\frac{0.45+0.315}{2},
$$

where $D^{3+}_{\rm cpx}=0.45$ from Mallmann & O'Neill (2009)[^cite-mallmann2009] and

$$
D^{3+}_{\rm opx}=0.70D^{3+}_{\rm cpx}=0.315
$$

from Schaefer et al. (2024, Eq. 6).

PROTEUS does not use modal proportions of the two pyroxenes.

For Fe$^{2+}$, $D^{2+}$ is the experimental mineral/melt FeO partition coefficient compiled in Schaefer et al. (2024) Table S1: bridgmanite $0.44\pm0.19$, Cpx $0.38\pm0.23$, Opx $0.69\pm0.46$; the shallow value is $(0.38+0.69)/2=0.535$. Schaefer et al. use these values to derive Fe/Mg exchange coefficients; their model applies those exchange coefficients, and the $D^{2+}$ it produces against a magma-ocean liquid is lower (bridgmanite 0.385, Cpx 0.33, Opx 0.36 in their archived 1000 km run). With Table S1, $D^{3+}>D^{2+}$ for bridgmanite, so Fe$^{3+}$/Fe$_{\rm T}$ falls while the lower mantle crystallizes, and $D^{3+}<D^{2+}$ for the pyroxenes, so it rises at pressures below 22 GPa, as in Schaefer et al. In their models the shallow rise comes mainly from olivine, which takes up no Fe$^{3+}$.

#### `"bulk_BSE"`: bulk coefficients of the Schaefer et al. layers

Each cell takes the bulk coefficient of the full mineral assemblage of the Schaefer et al. (2024) layer holding its pressure. The values are medians over all 1000 draws of their archived 1000 km run for a bulk-silicate-Earth magma ocean.

| Layer | Pressure | Minerals (fraction) | $D^{3+}$ | $D^{2+}$ |
|---|---:|---|---:|---:|
| 1 | $P<2.5$ GPa | olivine, pyroxenes, spinel or plagioclase | 0.76 | 0.97 |
| 2 | $2.5\leq P<15$ GPa | ol 0.5, cpx 0.2, opx 0.2, gt 0.1 | 0.30 | 0.39 |
| 3 | $15\leq P<18$ GPa | wad 0.40, maj 0.35, cpx 0.25 | 1.14 | 0.42 |
| 4 | $18\leq P<22$ GPa | maj 0.55, ring 0.45 | 1.60 | 0.59 |
| 5 | $P\geq22$ GPa | bg 0.95, mw 0.05 | 0.83 | 0.45 |

Their runs never reach $P<1$ GPa, so that layer repeats the 1-2.5 GPa values. On their crystallization sequence this prescription follows their mean Fe$^{3+}$/Fe$_{\rm T}$ within 0.005; the `"mgsio3"` prescription follows its shape but differs by up to 0.04, because it has no transition-zone phases and no olivine. [The validation page](../Validation/interior_chem/redox.md) gives both comparisons.

#### `"mgsio3_kd"`: pressure- and temperature-dependent Fe$^{2+}$ in bridgmanite and post-perovskite

Below 22 GPa and for Fe$^{3+}$ everywhere this prescription keeps the `"mgsio3"` values. At $P\geq22$ GPa the Fe$^{2+}$ split follows the Fe-Mg exchange coefficient

$$
K_D=\frac{X^{s}_{\rm Fe}\,X^{l}_{\rm Mg}}{X^{l}_{\rm Fe}\,X^{s}_{\rm Mg}}
=\exp\!\left[-\frac{\Delta H-T\Delta S+P\Delta V}{RT}\right],
\qquad X_{\rm Fe}=\frac{\rm Fe^{2+}}{\rm Fe^{2+}+Mg},
$$

with the exchange parameters of first-principles molecular dynamics and thermodynamic integration:

| Phase | Pressure | $\Delta H$ (kJ/mol) | $\Delta S$ (J/mol/K) | $\Delta V$ (cm$^3$/mol) | Fitted range | Source |
|---|---:|---:|---:|---:|---|---|
| Bridgmanite | $22\leq P<180$ GPa | $-19.66\pm10.90$ | $-12.04\pm3.28$ | $0.53\pm0.05$ | 23-140 GPa, 3000-6000 K | Dragulet & Stixrude (2024)[^cite-dragulet2024], Fig. 4 |
| Post-perovskite | $P\geq180$ GPa | $122\pm64$ | $12.9\pm11.2$ | $0.52\pm0.09$ | 156-603 GPa, 6000-10000 K | Dragulet & Stixrude (2026)[^cite-dragulet2026], Eq. 8 |

The phase changes at the 180 GPa bridgmanite-post-perovskite-melt triple point that the 2026 study takes from Deng et al. (2023). Both studies find $K_D$ independent of $X_{\rm Fe}$, so one value serves the whole MgSiO$_3$-FeSiO$_3$ join. The cell temperature is clamped into the fitted range, for bridgmanite extended to 6500 K because the 2026 study evaluates the 2024 fit at the triple point and quotes $K_D=0.057$ there; the fit gives that value at about 6430 K, a temperature inferred here, not stated in the study. Post-perovskite above 603 GPa takes the 603 GPa value. A run that clamps a crystallising cell logs a warning once. At 4000 K bridgmanite gives $K_D\approx0.29$ at 25 GPa and 0.049 at 136 GPa.

Because $K_D$ fixes cation ratios, this prescription counts the melt in the (Mg,Fe)SiO$_3$ basis: Fe$^{2+}$ as FeSiO$_3$, Fe$^{3+}$ as FeO$_{1.5}$, and the rest of the melt mass as MgSiO$_3$. Mg is not tracked; a phase of mass $m$ holds

$$
n_{\rm Mg}=\frac{m-n^{2+}\mu_{\rm FeSiO_3}-n^{3+}\mu_{\rm FeO_{1.5}}}{\mu_{\rm MgSiO_3}},
$$

so Mg is conserved whenever the cell mass is. When metal forms, $3\,{\rm FeSiO_3}=2\,{\rm FeO_{1.5}}+{\rm Fe}+3\,{\rm SiO_2}$: the metal leaves the melt and the three SiO$_2$ stay in it, so the melt mass entering $n_{\rm Mg}$ is the melt left after metal separation (as for every prescription) less the free SiO$_2$, in all $\xi(\mu_{\rm Fe}+3\mu_{\rm SiO_2})$ summed over the metal formed. This keeps the reaction from adding MgSiO$_3$ (2.35 mol of Mg per mole of metal otherwise). The free SiO$_2$ is taken as incompatible: it stays in the melt as it crystallizes, and adds to the iron-oxide mass of Step 3d. In each deep cell the Fe$^{3+}$ split is the constant-$D$ one above, and the Fe$^{2+}$ entering the solid, $\Delta n_i^{2+}$, is the root of

$$
X_{\rm Fe}\big(\text{solid}\big)-\frac{K_D\,X}{1-(1-K_D)\,X}=0,
\qquad X=X_{\rm Fe}\big(\text{residual melt of the cell}\big),
$$

so the solid formed in a step is in exchange equilibrium with the melt it leaves; the left side rises monotonically with $\Delta n_i^{2+}$ and the root is found by bisection. Many small steps converge to fractional crystallization. Under this basis a melt and solid of the same composition have $D^{2+}=1$, so the melt approaches pure FeSiO$_3$ instead of passing it; with Fe$^{2+}$ counted as FeO, the two phases would differ in concentration even then. Step 3d of the tracker (melt iron heavier than the melt) counts Fe$^{2+}$ as FeSiO$_3$, so it means the melt has no Mg left, and the metal step counts its non-iron cations (MgO, SiO$_2$ of MgSiO$_3$ and FeSiO$_3$, and the free SiO$_2$) from the current reservoirs.

The stored `D_fe2_cell` is the concentration ratio of solid that would form from the current melt. It is lower than the 0.44 of `"mgsio3"` throughout the lower mantle and falls with depth: in a melt with the starting composition about 0.30 at 30 GPa and 3600 K, 0.23 at 50 GPa and 4800 K. It rises as the melt enriches in iron. Fe$^{3+}$ keeps $D^{3+}=0.75$ in bridgmanite and post-perovskite alike; the $K_D$ studies treat ferrous iron only, and no post-perovskite $D^{3+}$ is constrained, so the ratio $D^{3+}/D^{2+}$, which drives the change of Fe$^{3+}$/Fe$_{\rm T}$, is the least constrained part of this prescription.

### 5. Update the reservoirs and redox ratio

The iron removed into the new solid is subtracted from the melt reservoirs:

$$
n^{3+}\leftarrow n^{3+}-\Delta n^{3+},
\qquad
n^{2+}\leftarrow n^{2+}-\Delta n^{2+}.
$$

The melt ferric fraction is

$$
f=\frac{n^{3+}}{n^{2+}+n^{3+}},
$$

and the ferric/ferrous ratio is

$$
R
=
\frac{X_{\rm FeO_{1.5}}}{X_{\rm FeO}}
=
\frac{f}{1-f}
=
\frac{n^{3+}}{n^{2+}}.
$$

`f` is written to the helpfile as `ferric_frac_mantle`.

Iron that entered the solid returns to the melt when its cell remelts (Section 2), which changes $R$ towards the ratio of the remelted solid.

When the global melt fraction $\sum_i\phi_i m_i/\sum_i m_i$ falls below 0.05 (`MELT_FRACTION_MIN`), the tracker holds its state: the reservoirs, $f$ and the last tracked melt fractions $\phi_i^{\rm prev}$. With constant $D<1$, fractional crystallization enriches the melt in iron as $C\propto F^{D-1}$, which has no bound as the melt fraction $F$ goes to zero; with the `"mgsio3"` $D^{2+}$ of 0.44-0.535, the melt FeO + FeO$_{1.5}$, 0.09 of the melt mass at the start, reaches 0.36-0.48 of it at $F=0.05$ and would exceed the melt mass below $F=0.006$-0.014. As a mass-balance check on every tracked step, a step that would leave more FeO + FeO$_{1.5}$ in the melt than the melt mass is not applied: the tracker holds in the same way and logs a warning with the mass ratio, which covers any case the 0.05 threshold misses. Surface $\Delta$IW is still evaluated on every step from the held $f$. If the melt fraction rises above 0.05 again, for example after a giant impact, the next call applies the whole change since the last tracked profile: cells that froze further crystallize and cells that remelted return their iron.

## From melt redox to surface $f_{\rm O_2}$

At the surface, the melt exchanges oxygen with the atmosphere through the ferric-ferrous equilibrium

$$
{\rm FeO}+\tfrac14{\rm O_2}\rightleftharpoons{\rm FeO_{1.5}}.
$$

For a given redox ratio $R$ of the well-mixed melt and a given temperature, this equilibrium fixes the oxygen fugacity. More ferric iron (larger $R$) gives a higher $f_{\rm O_2}$.

### Hirschmann (2022) relation

Hirschmann (2022), Eq. 21, relates the ferric/ferrous ratio of the melt to its oxygen fugacity:

$$
\log_{10}R
=
a\log_{10}f_{\rm O_2}
+b+\frac{c}{T}
-\frac{\Delta C_p}{R_g\ln10}
\left[
1-\frac{T_0}{T}-\ln\frac{T}{T_0}
\right]
-\frac{\int_{P_0}^{P}\Delta V\,dP}{R_gT\ln10}
+\frac{1}{T}
\left[
\sum_{k=1}^{7}Y_kX_k
+Y_8X_{\rm SiO_2}X_{\rm AlO_{1.5}}
+Y_9X_{\rm SiO_2}X_{\rm MgO}
\right].
$$

Here, $R_g$ is the gas constant and the $X_k$ are oxide mole fractions on a single-cation basis:

$X_{\rm SiO_2}$, $X_{\rm TiO_2}$, $X_{\rm MgO}$, $X_{\rm CaO}$, $X_{\rm NaO_{0.5}}$, $X_{\rm KO_{0.5}}$, and $X_{\rm PO_{2.5}}$.

PROTEUS inverts this equation to obtain $\log_{10}f_{\rm O_2}$.

The coefficients are the unrounded values from Schaefer et al. (2024) Table 4:

- $a=0.19317$,
- $b=-4.51412/2.303$,
- $c=9574.293/2.303$ K,
- $\Delta C_p=33.25$ J K$^{-1}$ mol$^{-1}$,
- $T_0=1673.15$ K.

The $Y_k$ coefficients are those implemented in `redox.py`.

The oxide mole fractions are fixed to the bulk-silicate-Earth composition from Schaefer et al. (2024) Table 1.

### Surface $\Delta$IW used by outgassing

Outgassing exchanges oxygen with the melt at the surface. PROTEUS evaluates the Hirschmann (2022) relation at 1 bar, so the pressure integral is zero.

The temperature is

$$
T_{\rm out}
=
\max(T_{\rm magma},T_{\rm floor}),
$$

where `T_floor` is `outgas.T_floor`.

The resulting $f_{\rm O_2}$ is converted to an offset from the Hirschmann (2021)[^cite-hirschmann2021] IW buffer at the same temperature and 1 bar:

$$
\Delta{\rm IW}
=
\log_{10}f_{\rm O_2}(R,T_{\rm out},1\,{\rm bar})
-
\log_{10}f_{\rm O_2}^{\rm IW,H21}(T_{\rm out},1\,{\rm bar}).
$$

This value is written as `fO2_shift_IW_mantle`.

CALLIOPE or atmodeller uses this value instead of `outgas.fO2_shift_IW` for that coupling step. Because the interior calculation runs before outgassing, the outgassing offset always reflects the current mantle redox state.

CALLIOPE and atmodeller themselves raise $T_{\rm magma}$ to `outgas.T_floor` before solving. Evaluating both the melt $f_{\rm O_2}$ and the buffer at $T_{\rm out}$ therefore keeps $\Delta$IW consistent with the temperature actually used downstream. When $T_{\rm out}>T_{\rm magma}$, a warning is logged.

At fixed $f$, $\Delta$IW decreases with temperature, because the IW buffer rises with $T$ faster than the melt $f_{\rm O_2}$ does. The same melt can therefore lie above IW in a cool magma ocean and below it in a hot one.

!!! warning "Temperature range of the IW buffer"

    The Hirschmann (2021) IW buffer is calibrated from 1000 to 3000 K and up to 100 GPa.

### Radial $f_{\rm O_2}$ profile

The tracker also evaluates the Hirschmann (2022) relation in every melt cell $i$, at the cell temperature $T_i$ and with the same melt redox ratio $R$, since the melt is uniform. Which pressure enters depends on two options:

- With `fO2_pressure_term = true` (the default) or `metal_saturation = true`, cell $i$ is evaluated at its pressure $P_i$: the integral $\int_{P_0}^{P_i}\Delta V\,dP$ of FeO + ¼O$_2$ = FeO$_{1.5}$ comes from the melt EOS `redox_dV_eos` (`eos_deng.int_dV_dP_oxidation`), and a cell outside that EOS's range gets NaN.
- With both `false`, every cell is evaluated at 1 bar, so the pressure integral is zero and the profile varies with depth only through $T_i$.

Each cell's value is also written as an offset from the Hirschmann (2021) IW buffer, taken at the same temperature and pressure as the cell's $f_{\rm O_2}$:

$$
\Delta{\rm IW}_i
=
\log_{10}f_{\rm O_2}(R,T_i,P_i)
-
\log_{10}f_{\rm O_2}^{\rm IW,H21}(T_i,P_i),
$$

with $P_i$ replaced by 1 bar when both options are `false`. This is the convention Schaefer et al. (2024) use for $\Delta$IW at depth. The surface $\Delta$IW passed to the outgassing always uses 1 bar.

The buffer is the empirical fit of Hirschmann (2021) Table 1, as in Schaefer et al.'s `IW_H21.m` (`_iw_buffer_hirschmann2021` in `redox.py`):

$$
\log_{10}f_{\rm O_2}^{\rm IW,H21}(T,P)
=
A(P)+B(P)\,T+C(P)\,T\ln T+\frac{D(P)}{T},
$$

where each of $A$, $B$, $C$, $D$ is a polynomial in $P$ [GPa] with terms in $1$, $P$, $P^2$, $P^3$ and $\sqrt{P}$, and $T$ is in K. There are two coefficient sets, one for fcc/bcc iron and one for hcp iron; the hcp set applies where

$$
P \geq -18.640 + 0.04359\,T - 5.069\times10^{-6}\,T^2 .
$$

The fit is calibrated from $10^{-4}$ to 100 GPa and from 1000 to 3000 K; deeper or hotter cells, as in super-Earth mantles, are extrapolated.

The profile is diagnostic only: it is written to the interior snapshot and does not feed back into the model.

## Outputs

| Output | Where | Meaning |
|---|---|---|
| `ferric_frac_mantle` | Helpfile | Melt Fe$^{3+}$/Fe$_{\rm T}$ after the current crystallization step |
| `fO2_shift_IW_mantle` | Helpfile | Surface $\Delta$IW passed to outgassing \[log$_{10}$ units\] |
| `log10_fO2_s` | Interior snapshot | Per-cell $\log_{10}f_{\rm O_2}$ \[log$_{10}$ bar\]; NaN in solid cells and, with the pressure term, outside the `redox_dV_eos` range |
| `dIW_H21_s` | Interior snapshot | Same profile relative to the Hirschmann (2021) IW buffer at each cell's $T_i$ and, with the pressure term, $P_i$ (otherwise 1 bar) \[log$_{10}$ units\] |
| `fO2_top_index` | Interior snapshot | Staggered-grid index of the uppermost melt cell |
| `redox_*` | Interior snapshot | Tracker state used when resuming a run |

With Aragog, the profile and state are appended to `data/<time>_int.nc`.

With SPIDER, which writes JSON output, they are stored in `data/<time>_redox.nc`, written every step.

Both profile variables have a `pressure_term` attribute, which is `0` because the profiles are evaluated at 1 bar.

## Resuming a run

The tracker stores its Fe$^{2+}$ and Fe$^{3+}$ reservoirs in memory, so resumed runs must restore this state.

Each snapshot stores the full tracker state, including:

- Fe$^{2+}$ and Fe$^{3+}$ reservoirs,
- $f$,
- previous melt fractions $\phi^{\rm prev}$,
- $D^{3+}$,
- remaining bookkeeping variables.

`proteus start -r` restores this state from the snapshot corresponding to the resumed helpfile row. The run then continues as if it had not been stopped.

Snapshots created before redox state storage was implemented cannot restore the tracker. In that case, the resume logs a warning and reinitializes the tracker from `planet.interior_chem.ferric_fraction_initial`, causing $f$ to reset to $f_0$.

## Assumptions and limitations

- **Fixed composition in the $f_{\rm O_2}$ relation.**  
  The Hirschmann (2022) oxide mole fractions remain fixed at bulk-silicate-Earth values. Schaefer et al. (2024) instead evolve the melt composition during fractional crystallization; PROTEUS evolves only the iron redox state.

- **Temperature range of the $f_{\rm O_2}$ relation.**  
  Hirschmann (2022) calibrated Eq. 21 at 100 kPa on 478 experiments spanning 1468-2197 K. The surface $f_{\rm O_2}$ is evaluated at $\max(T_{\rm magma}, T_{\rm floor})$, which lies well above 2197 K early in a run and can fall to the 700 K default of `outgas.T_floor` late, so the relation is extrapolated at both ends.

- **Restricted mineralogy or fixed bulk coefficients.**  
  Schaefer et al. track a full mineral sequence, including olivine, pyroxenes, garnet, majorite, wadsleyite, ringwoodite, and bridgmanite. The `"mgsio3"` prescription uses only bridgmanite and the pyroxenes; `"bulk_BSE"` uses their bulk layer coefficients but holds them fixed rather than letting them follow the melt composition.

- **Fe/Mg exchange only in the lower mantle, and only with `"mgsio3_kd"`.**  
  With `"mgsio3"` and `"bulk_BSE"`, $D^{2+}$ is a fixed coefficient per assemblage. With `"mgsio3_kd"`, $D^{2+}$ at $P\geq22$ GPa follows the pressure- and temperature-dependent Fe-Mg exchange coefficient of bridgmanite and post-perovskite and the evolving melt composition ([step 4](#4-partition-iron-into-the-new-solid)); shallower it is the fixed `"mgsio3"` value. In Schaefer et al. it follows from a fixed Fe/Mg exchange coefficient and the evolving melt composition.

- **Uncertain partition coefficients.**  
  Schaefer et al. report $D^{3+}_{\rm bg}=0.75\pm0.65$ and $D^{3+}_{\rm cpx}=0.45\pm0.20$. Their Monte Carlo calculations therefore produce a wide range of Fe$^{3+}$/Fe$_{\rm T}$ and $f_{\rm O_2}$. PROTEUS uses only the central values.

- **Surface temperature.**  
  Schaefer et al. derive the surface temperature from an adiabat anchored at the magma-ocean base. PROTEUS instead uses the coupled $T_{\rm magma}$, with a lower bound set by `outgas.T_floor`.

- **No atmospheric back-reaction.**  
  Oxygen exchanged with the atmosphere does not modify the melt reservoirs. The coupling is one-way: melt redox $\rightarrow$ outgassing.

- **Remelted solid is homogeneous.**  
  A remelting cell returns the average Fe$^{3+}$/Fe$^{2+}$ of the solid the tracker formed in it, not that of the last layer to crystallize. Solid present at the first step holds no tracked iron.

- **Finite time-step effects.**  
  Each step equilibrates its solid with the residual melt, so a coarse step is closer to batch than to fractional crystallization. The melt iron inventory depends on the step size, but $f$ hardly does: for a single shallow-regime cell frozen to 1% melt, $f$ differs by less than $10^{-3}$ between 10 and 2000 steps.

- **Last 5% of the melt.**  
  The tracker holds its state below a global melt fraction of 0.05, so the redox evolution of the last melt is not followed, and the surface $\Delta$IW late in a run comes from the held $f$ at the current temperature.

## Validation

The unit tests in `tests/interior_chem/test_redox.py` check that:

- the initial iron split reproduces $f_0$ exactly and rejects values outside $(0,1)$;
- $D^{2+}$ and $D^{3+}$ switch between the two assemblages at 22 GPa, and `"bulk_BSE"` assigns each cell the coefficients of its Schaefer et al. layer;
- the crystallization step reproduces an archived Schaefer et al. (2024) run when fed its coefficients, and both prescriptions give the fall and later rise of their Fe$^{3+}$/Fe$_{\rm T}$ ([validation page](../Validation/interior_chem/redox.md));
- increasing $f_0$ increases surface $\Delta$IW at fixed temperature;
- surface $\Delta$IW uses the Hirschmann (2022) relation at 1 bar and $T_{\rm out}$ minus the Hirschmann (2021) IW buffer;
- the real melt-fraction change determines how much iron crystallizes;
- a step that freezes the whole melt of a cell moves all of that cell's iron into the solid for any $D$, the reservoirs stay non-negative, and the melt FeO + FeO$_{1.5}$ stays below the melt mass down to the 0.05 cutoff, below which the state is held;
- the Hirschmann (2021) buffer agrees with an independent 1-bar IW calibration near 1500 K and has the expected pressure dependence;
- a remelting cell returns the remelted share of its solid's iron, total iron is conserved between melt and solid, and remelting back to the initial melt fraction restores $f_0$;
- a run resumed from a snapshot reaches the same final reservoirs as an uninterrupted run.

### Comparison with Schaefer et al. (2024)

The figure feeds the crystallization sequence of an archived Schaefer et al. (2024)[^cite-schaefer2024] run (the 1000 km Earth magma ocean, $f_0=0.1$, their run 570) to the crystallization step of the tracker, with each set of partition coefficients in turn. With `"bulk_BSE"` the melt Fe$^{3+}$/Fe$_{\rm T}$ stays within 0.01 of their run at every step and turns from falling to rising between 12 and 18 GPa, as theirs does. With `"mgsio3"` it turns at 22 GPa, where the pyroxenes take over from bridgmanite, and lies up to 0.04 above their run near 3 GPa, because that prescription has no transition-zone phases and no olivine. The [validation page](../Validation/interior_chem/redox.md) gives the tolerances and the other reference checks.

<figure markdown="span">
  ![Melt ferric fraction on the Schaefer et al. (2024) crystallization sequence](../assets/validation/redox_schaefer2024_crystallisation.svg#only-light){ width="640" }
  ![Melt ferric fraction on the Schaefer et al. (2024) crystallization sequence](../assets/validation/redox_schaefer2024_crystallisation_dark.svg#only-dark){ width="640" }
  <figcaption>Melt Fe<sup>3+</sup>/Fe<sub>T</sub> against the pressure of the layer that crystallizes, from the base of the magma ocean (left) to the surface (right). Circles: Schaefer et al. (2024) run 570, every fourth step and the last; band: that run within 0.01, the tolerance of the test. Lines: the tracker's crystallization step with the <code>"bulk_BSE"</code> and <code>"mgsio3"</code> coefficients. The dotted line marks 22 GPa. Written by <code>tests/interior_chem/test_redox.py::test_bulk_prescription_follows_the_schaefer_2024_run</code>.</figcaption>
</figure>

This is a check of the crystallization step on their sequence, not a coupled run: the solid fraction of each step comes from their run, not from an interior module. No coupled run with the current prescriptions has yet been compared with their evolution.

---

**See also:** [Model description](model.md) | [Planet configuration](../Reference/config/planet.md) | [Output reference](../Reference/output.md) | [Coupling loop](coupling_loop.md)

[^cite-schaefer2024]: Schaefer, L., Pahlevan, K. & Elkins-Tanton, L.T., *Ferric Iron Evolution During Crystallization of the Earth and Mars*, Journal of Geophysical Research: Planets, 129, e2023JE008262, 2024.

[^cite-hirschmann2022]: Hirschmann, M.M., *Magma oceans, iron and chromium redox, and the origin of comparatively oxidized planetary mantles*, Geochimica et Cosmochimica Acta, 328, 221-241, 2022.

[^cite-hirschmann2021]: Hirschmann, M.M., *Iron-wüstite revisited: A revised calibration accounting for variable stoichiometry and the effects of pressure*, Geochimica et Cosmochimica Acta, 313, 74-84, 2021.

[^cite-dragulet2024]: Dragulet, F. & Stixrude, L., *Partitioning of Iron Between (Mg,Fe)SiO3 Liquid and Bridgmanite*, Geophysical Research Letters, 51, e2023GL107979, 2024.

[^cite-dragulet2026]: Dragulet, F. & Stixrude, L., *Incompatibility of iron in post-perovskite and the stability of basal magma oceans in super-Earths*, arXiv:2609.13527 (preprint), 2026.

[^cite-mallmann2009]: Mallmann, G. & O'Neill, H.St.C., *The crystal/melt partitioning of V during mantle melting as a function of oxygen fugacity compared with some other elements (Al, P, Ca, Sc, Ti, Cr, Fe, Ga, Y, Zr and Nb)*, Journal of Petrology, 50, 1765-1794, 2009.
