# Outgassing, rock vapour and volatile trapping

This page answers one question: **where do the volatiles of a planet go during each step of a PROTEUS run, and which part of the code moves them?**

The short answer: in each step, a chemistry solver splits the volatile inventory between the atmosphere and the magma ocean. Before that solve, a trapping step can bury part of the dissolved volatiles in the mantle that froze since the last step. After the solve, an optional step adds rock vapour from the hot melt surface to the atmosphere.

The first section gives the order of the steps. The later sections explain the species-to-element split, the H$_2$ binodal, rock vapour, trapping, melt drainage, the burial ledger and the limits of the trapping inputs. A section whose part of the code has a Validation case ends with a link to it. For the config keys, see [Escape and outgassing configuration](../Reference/config/escape_outgas.md#outgassing-outgas). For the output columns, see [Output format](../Reference/output.md).

## Terms used on this page

- **Volatile**: an element or molecule that can be a gas at the surface. PROTEUS tracks 5 volatile elements (H, C, N, O, S) and 11 species made from them: H$_2$O, CO$_2$, H$_2$, CO, CH$_4$, N$_2$, NH$_3$, S$_2$, SO$_2$, H$_2$S and O$_2$. It also tracks the noble gases He, Ne, Ar, Kr and Xe.
- **Reservoir**: a place where a volatile can be. Each species and each element has 3: the atmosphere (`_kg_atm`), the melt (`_kg_liquid`) and the solid mantle (`_kg_solid`). The 5 volatile elements have a fourth, `_kg_condensed`, for condensates.
- **Condensate**: a solid or liquid that forms from the gas at the surface, for example graphite. Only atmodeller makes condensates, and only with `outgas.atmodeller.include_condensates = true`.
- **Element total**: `{e}_kg_total`, the mass of element `{e}` in the whole planet \[kg\]. Here `{e}` stands for an element symbol, and `{sp}` below stands for a species name.
- **Escape**: the loss of atmospheric gas to space. The escape step lowers the element totals. The chemistry solve does not change them, except for oxygen (see the next section).
- **Chemistry solver**: the module that computes how a volatile inventory divides between gas and melt at the surface. It is set by `outgas.module`: CALLIOPE and atmodeller compute the chemical equilibrium, and the dummy module uses a fixed rule. See [Model description](model.md#volatile-outgassing-calliope-atmodeller).
- **Interior solver**: the module that evolves the mantle temperature and melt fraction, set by `interior_energetics.module` (for example Aragog or SPIDER).
- **Melt fraction** $\Phi$: the mass fraction of the mantle that is liquid \[1\], `Phi_global` in the output. Do not confuse it with the porosity $\phi$ (lower case) below, which is a volume fraction.
- **Oxygen fugacity** (fO$_2$): a measure of how oxidising the melt is, in the form of an effective O$_2$ pressure. PROTEUS gives it relative to the iron-wustite (IW) buffer, the fO$_2$ at which iron metal and FeO are in equilibrium. `outgas.fO2_shift_IW` is that shift in log10 units, so a value of 2 is 100 times the IW fO$_2$.
- **Mass threshold**: `outgas.mass_thresh` \[kg\]. The chemistry solve treats H, C, N or S with less mass than this as absent. Escape and the desiccation check also use it (see [the config reference](../Reference/config/escape_outgas.md#outgassing-outgas)).
- **Desiccated**: the state of a planet that has lost its volatiles. Every volatile and noble gas that escape can still reach is below the mass threshold.
- **Initialisation stage**: the first iterations of a run. They set up the starting state. The mantle does not evolve in them, and the time returns to zero after each.
- **`M_atm`**: the total mass of the atmosphere \[kg\].

## The order of the volatile steps

**Question.** In what order do the volatile steps run, and what does each step change?

**Short answer.** Trapping runs first, then escape, then outgassing. So escape and outgassing both see the mass that trapping buried in the same step.

Each iteration of the [coupling loop](coupling_loop.md#execution-order-per-iteration) runs these steps, in this order:

1. **Interior and structure.** The interior solver advances the mantle temperature and the melt fraction $\Phi$.
2. **Trapping** (`run_trapping`). If `outgas.trap_mode = "front"`, part of the dissolved volatiles moves from the melt to the solid mantle. See [Volatile trapping](#volatile-trapping-in-the-freezing-mantle).
3. The orbit and stellar steps run. They do not change the volatiles.
4. **Escape** (`run_escape`). Escape lowers the element totals.
5. **Outgassing** (`run_outgassing_and_vapourisation`). The chemistry solver splits the inventory between the atmosphere and the melt. If `outgas.vapourise = true`, rock vapour follows.
6. **Atmosphere climate.** The climate solver reads the new composition.

The outgassing step takes 1 of 3 paths:

- **Desiccated** (`run_desiccated`). This path runs when the planet is desiccated and escape accounts for the lost mass.
    - The atmosphere is emptied.
    - The melt is emptied, except with `outgas.trap_mode = "front"`. Then the melt keeps its mass, so that mass a remelt returns stays in the planet.
    - The solid mantle and the condensates keep their mass.
    - Rock vapour can still enter the atmosphere (see [Rock vapour](#rock-vapour-from-the-melt-surface)).
- **Crystallised** (`run_crystallized`). This path needs `params.stop.solid.freeze_volatiles = true`. It starts when $\Phi$ falls to `params.stop.solid.phi_crit` or below, and it stays on after that; only a giant impact (see [Accretion](model.md#accretion-morrigan)) turns it off. The melt and the atmosphere then exchange no volatiles. Escape still removes gas, so the code scales every atmospheric reservoir down by the same fraction, and the composition does not change.
- **Normal** (`run_outgassing`). The chemistry solver runs. This is the usual path while a magma ocean exists.

On the normal path, the step does 3 things:

1. It takes out the mass that the solver must not change: the trapped mass, and, with `outgas.prevent_desaturation = true`, the condensed mass.
2. It runs the chemistry solver.
3. It puts the withheld mass back and adds up the atmospheric mass `M_atm`.

**Oxygen.** With CALLIOPE or atmodeller and `planet.fO2_source = "user_constant"`, the solver holds the fO$_2$ at the set value. To do that, it adds or removes oxygen, so `O_kg_total` is a result of each solve and not a fixed budget. With `"from_O_budget"`, the O budget is the input, and the solver finds the fO$_2$ that matches it. The dummy module computes an O total only in the initialisation stage and only when the O total is empty. Otherwise it keeps the O total.

**Initialisation check.** In the initialisation stage, the code writes the element budgets from `planet.elements` to `{e}_kg_total`. You can also set an O budget, with `planet.elements.O_mode` set to `"kg"`, `"ppmw"` or `"FeO_mantle_wt_pct"`. With such an O budget and `planet.fO2_source = "user_constant"`, the code compares your budget with the oxygen of the first solve. It stops the run if the two differ by more than 50 % of the solve value. The default, `O_mode = "ic_chemistry"`, skips this check.

**Where this is checked.** No Validation case covers the order of the steps. After each outgassing step, a runtime check tests the mass budget; see [Mass conservation](coupling_loop.md#mass-conservation).

## From species to elements

**Question.** The chemistry solver returns masses of molecules. How does the code turn them into the element masses that escape and the budget use?

**Short answer.** It splits each molecule's mass over its elements by their share of the molecule's molar mass.

For an element $e$ with $n_e$ atoms in a molecule, that share is

$$
f_e = \frac{n_e\, m_e}{\sum_i n_i\, m_i},
$$

where $m_i$ is the molar mass of element $i$ \[kg mol$^{-1}$\] and the sum runs over every element of the molecule.

**Example.** For water, $f_\mathrm{O} = 0.88809$ and $f_\mathrm{H} = 0.11191$. So 1 kg of dissolved water adds 0.888 kg to `O_kg_liquid` and 0.112 kg to `H_kg_liquid`.

The shares of one molecule always add up to 1, so the split conserves mass, whatever the table of molar masses says. The atmodeller module writes this split for the atmosphere, melt and solid reservoirs. It does not write `{e}_kg_total`, because the escape step owns that total. Trapping uses the same rule (`element_masses_from_species`) to move element masses with the species it buries.

**Where this is checked.** [Species-to-element mass split](../Validation/outgas/atmodeller.md).

## Hydrogen and silicate melt: the binodal

**Question.** Hot hydrogen and silicate melt can mix completely. Does PROTEUS include that?

**Short answer.** Not in a run. The config loader rejects both switches for it: `outgas.h2_binodal = true` and `interior_struct.zalmoxis.global_miscibility = true`. The rest of this section describes the code that these switches would use.

H$_2$ and silicate melt are **miscible** when they form 1 phase. Where they separate into a hydrogen-rich gas and a silicate melt, they are immiscible. The line in pressure and temperature where they separate is the **binodal**. Rogers et al. [^cite-rogers2025] compute it for H$_2$ and MgSiO$_3$, which stands in for the silicate melt. In their sub-Neptune models, the 2 are miscible above about 4000 K. In the code, the binodal temperature depends on the pressure and on the H$_2$ mass fraction.

With `outgas.h2_binodal = true`, `apply_binodal_h2` would replace the H$_2$ split of the chemistry solver. It takes the fraction $\sigma$ of H$_2$ that dissolves in the melt from the binodal model of the Zalmoxis structure code. The inputs are the surface pressure, the magma temperature and the H$_2$ mass fraction. The split is then

$$
M_\mathrm{H_2,\,melt} = \sigma\, M_\mathrm{H_2,\,total}, \qquad M_\mathrm{H_2,\,atm} = (1-\sigma)\, M_\mathrm{H_2,\,total},
$$

where $M$ is a mass \[kg\]. $\sigma$ is between 0 (immiscible, all H$_2$ in the atmosphere) and 1 (all H$_2$ in the melt).

**Where this is checked.** [H2-MgSiO3 binodal](../Validation/outgas/binodal.md). That page covers the H$_2$ reservoir bookkeeping of this function. No case compares the split with Rogers et al.

## Rock vapour from the melt surface

**Question.** A very hot magma ocean also evaporates rock. How does that vapour enter the atmosphere?

**Short answer.** With `outgas.vapourise = true`, the LavAtmos code [^cite-vanbuchem2023] computes the vapour above the melt after the volatile solve, and adds it to the atmosphere.

LavAtmos computes the vapour that is in equilibrium with the melt surface. FastChem, a separate gas-chemistry package, then combines it with the volatile gas. The vapour adds species such as SiO, Na, Mg and Fe.

The inputs are:

- the surface temperature, $\max(T_\mathrm{min}, T_\mathrm{magma})$ \[K\], with $T_\mathrm{min}$ = `outgas.lavatmos.T_min` and $T_\mathrm{magma}$ the magma surface temperature from the interior;
- the volatile surface pressure from the chemistry solve \[bar\];
- the number (mole) fractions of the volatile elements in the atmosphere;
- the melt composition file `outgas.lavatmos.melt_comp_name`.

To turn masses into number fractions, the code needs the molar mass of each species. It sums the atomic masses of the formula and gives the result in g mol$^{-1}$, the unit that FastChem and LavAtmos use. For SiO$_2$ this is 60.083 g mol$^{-1}$.

The vapour step runs after the chemistry solve on the normal path and on the desiccated path. It is skipped when $\Phi$ is at `params.stop.solid.phi_crit` or below. Rock vapour enters the atmosphere without being taken from the mantle, so with `outgas.vapourise = true` whole-planet mass is not conserved; see [Model description](model.md#whole-planet-mass-is-not-conserved-when-vapourisation-is-enabled).

**Where this is checked.** [Rock-vapour molar masses](../Validation/outgas/lavatmos.md).

## Volatile trapping in the freezing mantle

**Question.** When the magma ocean freezes, how much of its dissolved water and gas stays in the solid mantle instead of reaching the atmosphere?

**Short answer.** Each kilogram of frozen mantle keeps some volatiles: a little inside the crystals, and all of those in the melt that stays between the crystals (the **pore melt**). With `outgas.trap_mode = "front"`, the code computes the pore melt from how fast melt drains out of the freezing layer. With `"none"`, the default, nothing is trapped.

Trapped mass is volatile mass that moved from the melt to the solid mantle. The text also says the mass is "buried"; the code calls it trapped (`_kg_trapped`).

### How to turn trapping on and read the result

Trapping needs the Aragog interior solver, because only Aragog gives the radial profiles that the freezing front needs:

```toml
[interior_energetics]
module = "aragog"

[outgas]
trap_mode = "front"
```

Trapping does nothing in the initialisation stage and in the first step of a run, because it needs a previous step to compare with. These output columns show what it did ([full list](../Reference/output.md#solid-phase-volatile-trapping)):

- `trap_kg_step` \[kg\]: the volatile mass buried in this step; negative when a remelt returned mass.
- `trap_kg_cumulative` \[kg\]: the net mass buried since the start of the run.
- `{sp}_kg_trapped` \[kg\]: the mass of each species, and of each element, that is trapped at the end of the step.
- `trap_F_tl` \[1\]: the trapped melt fraction of the step, $F_\mathrm{tl}$ below.
- `trap_branch`: which path the step took (see [When the front cannot be integrated](#when-the-front-cannot-be-integrated)).
- `trap_tau_D`, `trap_tau_s`, `trap_t_res` \[yr\] and `trap_L_front` \[m\]: the 2 timescales, the residence time and the front thickness of the drainage section; they show why a step took branch 1, 2 or 3.

The other trapping inputs, such as `outgas.D_const_<species>` and `outgas.trap_phi_min`, are in the [config reference](../Reference/config/escape_outgas.md#outgassing-outgas).

### The burial rule

In one step, the mantle mass $\Delta M$ \[kg\] freezes. The code gets it from the drop of the melt fraction: $\Delta M = M_\mathrm{mantle}\,(\Phi_{t-1} - \Phi_t)$, where $M_\mathrm{mantle}$ is the mantle mass \[kg\] and $t$ counts the steps. Each volatile species $Z$ then moves this mass from the melt to the solid [^cite-sim2024]:

$$
\Delta m_Z = \left[(1 - F_\mathrm{tl})\, D_Z + F_\mathrm{tl}\right] C_Z\, \Delta M.
$$

The symbols are:

- $C_Z$: the mass fraction of $Z$ in the melt \[1\], from the last chemistry solve.
- $D_Z$: the crystal/melt partition coefficient \[1\]. It is the mass fraction of $Z$ in the crystals divided by that in the melt, set by `outgas.D_const_<species>`.
- $F_\mathrm{tl}$: the trapped liquid fraction \[1\]. It is the mass fraction of the new solid that is pore melt.

The bracket is the effective partition coefficient. If $F_\mathrm{tl} = 0$, it is $D_Z$: only the crystals take up the volatile. If $F_\mathrm{tl} = 1$, it is 1: the frozen mass takes the volatile at the melt concentration, as if it were all melt.

The default $D_Z$ is 0.0017 for H$_2$O and 0 for every other species. A species with $D_Z = 0$, such as CO$_2$ or a noble gas, is buried only with the pore melt. For water, the pore melt buries more than the crystals when $F_\mathrm{tl}$ is above about 0.0017.

**Example.** Take water at $C_Z = 10^{-3}$ (1000 ppm by mass), with $D_Z = 0.0017$ and $F_\mathrm{tl} = 0.02$. Let $2\times10^{23}$ kg of mantle freeze. The bracket is 0.021666, so $4.33\times10^{18}$ kg of water is buried. Of this, the pore melt buries $4.0\times10^{18}$ kg and the crystals take up $3.3\times10^{17}$ kg.

The buried mass moves from `{sp}_kg_liquid` to `{sp}_kg_solid`, and `{sp}_kg_trapped` records it. The element reservoirs move with it, by the split of the [species-to-element section](#from-species-to-elements). The element totals do not change. Buried mass cannot escape or outgas: the escape step and the desiccation check do not count it. The buried mass of a species is never more than the mass that the melt holds.

**Where this is checked.** [Solid-phase volatile trapping](../Validation/outgas/trapping.md).

## Melt drainage across the freezing front

**Question.** How much melt is still in the pores when a layer of crystals has frozen?

**Short answer.** Melt leaves the freezing layer only if 2 things happen. The melt must flow out through the pores, and the crystal framework must deform to close the pores. The slower of the 2 sets the rate. The melt that is left when the layer is fully frozen gives $F_\mathrm{tl}$.

### Terms for this section

- **Mush**: a mix of crystals and melt that is partly molten.
- **Matrix**: the connected framework of crystals in a mush.
- **Freezing front**: the layer of mush between the magma ocean above and the solid, nearly melt-free mantle below. The code uses "front" for this layer.
- **Porosity** $\phi$: the volume fraction of melt in the mush \[1\]. It is a volume fraction, while the melt fraction $\Phi$ is a mass fraction.
- **Permeability**: how easily melt flows through the pores \[m$^2$\].
- **Parcel**: a small piece of mush. As the front moves up, a parcel enters the front at the top and leaves it at the base.
- **Mesh and nodes**: the radial grid of the interior solver, and its grid points.
- **Critical melt fraction** $\Phi_c$: `interior_energetics.rfront_loc` (default 0.4), the melt fraction at which the interior solver switches the mantle from melt-like to solid-like flow.

### Where the front is

The code finds the front on the interior mesh (`locate_front`):

- **Top**: where the melt fraction of the interior solver falls below $\Phi_c$. So trapping and the interior use the same transition.
- **Base**: where the porosity falls below `outgas.trap_phi_min` (default 0.01). If the lowest node is still porous, the core-mantle boundary is the base.

The interior solver gives a melt fraction for each node, so the top uses it directly. For the base, the code turns the densities of the mush $\rho$, the solid $\rho_s$ and the melt $\rho_l$ into a porosity: $\phi = (\rho_s - \rho)/(\rho_s - \rho_l)$.

### The 2 timescales

The **percolation time** $\tau_D$ is how long melt takes to flow through a front of thickness $L$ \[m\]:

$$
\tau_D = \frac{L}{w_D}, \qquad w_D = \frac{|\Delta\rho|\, g\, k(\phi)}{\eta_\mathrm{melt}}.
$$

Here $w_D$ is the percolation speed \[m s$^{-1}$\] and $\Delta\rho$ is the solid density minus the melt density \[kg m$^{-3}$\]. $g$ is the gravity \[m s$^{-2}$\] and $\eta_\mathrm{melt}$ is the melt viscosity \[Pa s\]. $k(\phi)$ \[m$^2$\] is the permeability divided by the porosity, called the mobility.

$k(\phi)$ is the same law that Aragog uses for the separation of melt and solid [^cite-bower2018]. It has 3 regimes: Blake-Kozeny-Carman flow at low porosity, Rumpf-Gupte flow at middle porosity and Stokes settling of single grains at high porosity. The [drainage Validation page](../Validation/outgas/compaction.md) gives the formula of each. $k$ grows with the grain size, `interior_energetics.grain_size` \[m\], and with $\phi$. With the default `interior_energetics.aragog.separation_buoyancy = "mixture"`, the 2 porous-flow regimes include a factor $1 - \phi$.

The **matrix time** $\tau_s$ is how long the matrix takes to deform and close the pores:

$$
\tau_s = \frac{\mu_s}{|\Delta\rho|\, g\, L},
$$

where $\mu_s$ is the mush viscosity \[Pa s\], $10^{20}$ Pa s by default.

### The drainage equation

A parcel takes the **residence time** $t_\mathrm{res} = L / v_f$ to cross the front. Here $v_f$ is the front speed \[m s$^{-1}$\], from the mass that froze in the step. While the parcel crosses the front, its porosity falls as

$$
\frac{d\phi}{dt} = -\frac{\phi}{\max(\tau_D(\phi),\ \tau_s)}.
$$

The code integrates this from the porosity at the top of the front, over the time $t_\mathrm{res}$ (`drainage_integral`). The porosity at the end is the trapped melt as a volume fraction. The code then turns it into the mass fraction $F_\mathrm{tl}$ that the burial rule uses (`volume_to_mass_fraction`).

**Example.** Take the Mars-like front of the [drainage Validation case](../Validation/outgas/compaction.md). It is 60 km thick, with $\Delta\rho = 330$ kg m$^{-3}$ and $g = 3.711$ m s$^{-2}$, and a parcel enters it at $\phi = 0.3$. Use 1 mm grains, a melt viscosity of 100 Pa s and the default mush viscosity. For the mass fraction, take melt and solid densities of 2800 and 3130 kg m$^{-3}$.

| Front speed | $t_\mathrm{res}$ | $\tau_D$ at entry | $\tau_s$ | Trapped melt, volume fraction | $F_\mathrm{tl}$ (mass fraction) |
|---|---|---|---|---|---|
| 70 cm yr$^{-1}$ | $8.6\times10^4$ yr | $7.0\times10^4$ yr | $4.3\times10^4$ yr | 0.194 | 0.177 |
| 5 cm yr$^{-1}$ | $1.2\times10^6$ yr | $7.0\times10^4$ yr | $4.3\times10^4$ yr | 0.109 | 0.099 |

A slower front gives the melt more time to drain, so it traps less. $\tau_D$ grows as $\phi$ falls, so here $\tau_D$ is longer than $\tau_s$ during the whole crossing. Percolation therefore limits the drainage, and the code records `trap_branch` = 1. Because this case is limited by percolation, the mush viscosity does not change the result.

### When the front cannot be integrated

The code does not integrate a front that the mesh cannot resolve. Then it uses an upper bound for $F_\mathrm{tl}$: the porosity at the top of the front, turned into a mass fraction, and not more than $\Phi_c$. This happens when:

- the front spans fewer than `outgas.trap_n_front_min` nodes (default 3);
- the front is thicker than `outgas.trap_max_front_fraction` times the mantle thickness (the default, 1, sets no limit);
- the mush is split into separate layers;
- the front reaches the surface node;
- the front has no positive thickness;
- the melt is denser than the solid somewhere in the front;
- the front speed is zero.

Each such step logs a warning with the cause. The column `trap_branch` records the path of every step:

| `trap_branch` | Name in the output reference | Meaning |
|---|---|---|
| 0 | no front | no front this step; a freezing step still buries at $D_Z$ |
| 1 | Darcy | drained, limited by percolation at the top of the front |
| 2 | matrix | drained, limited by matrix deformation at the top of the front |
| 3 | guard | upper bound, the front could not be integrated |
| 6 | no interior state | the interior profiles were missing; the step buries at $D_Z$ alone and logs a warning |
| 7 | remelting | the step returns trapped mass to the melt |

The values 4 and 5 are never written.

**Where this is checked.** [Melt drainage across the freezing front](../Validation/outgas/compaction.md).

## Remelting and the burial ledger

**Question.** If the mantle melts again, which buried volatiles come back to the melt?

**Short answer.** The solid that froze last is the first to melt. So a remelt returns the mass that was buried most recently.

The **burial ledger** stores the mass buried in each step as a layer. Each layer has 2 melt fractions: $\Phi$ before the step and $\Phi$ after it. When $\Phi$ rises to a new value, the code returns the mass of each layer whose melt fractions are below that value (`remelt_to`). A layer that the new value cuts in 2 returns the remelted share of its mass.

**Example.** The mantle freezes from $\Phi = 1$ to 0.7 and buries 3 kg. Then it freezes from 0.7 to 0.5 and buries 2 kg. Then it remelts to $\Phi = 0.6$. That remelts half of the second layer, so 1 kg returns to the melt. The first layer and the other half of the second layer stay buried.

At each remelt, the code builds the ledger from the output columns `Phi_global` and `{sp}_kg_trapped`, so a resumed run has the same ledger.

If the history does not account for all of the trapped mass, the rest returns in proportion to the share of the solid that remelted, $(\Phi_t - \Phi_{t-1})/(1 - \Phi_{t-1})$.

**Where this is checked.** [Burial ledger for remelting](../Validation/outgas/burial_ledger.md).

## Limits of the trapping inputs

**Question.** Which values can the trapping inputs take?

**Short answer.** The config loader checks each limit below. A value outside a limit stops the load with an error.

- `outgas.trap_mode`: `"none"` or `"front"`. `"front"` needs `interior_energetics.module = "aragog"`.
- `outgas.trap_phi_min`: above 0 and below 1.
- `outgas.trap_n_front_min`: 2 or more.
- `outgas.trap_max_front_fraction`: above 0 and at most 1.
- `outgas.D_const_<species>`: 0 or more.
- `interior_energetics.rfront_loc`: above 0 and below 1.
- `interior_energetics.grain_size`: above 0 \[m\].
- `outgas.trap_mush_log10visc`: log10 of the mush viscosity in Pa s. A value above 0 is used as given. A value of 0 or below, such as the default -1, selects 20, that is $10^{20}$ Pa s.

During a run, an $F_\mathrm{tl}$ outside 0 to 1 stops the run with an error. A front that the mesh cannot resolve gives a warning and an upper bound, not an error; see [When the front cannot be integrated](#when-the-front-cannot-be-integrated).

**Where this is checked.** No Validation case covers these limits.

[^cite-rogers2025]: Rogers, J.G., Young, E.D. and Schlichting, H.E., *[Redefining interiors and envelopes: hydrogen-silicate miscibility and its consequences for the structure and evolution of sub-Neptunes](https://doi.org/10.1093/mnras/staf1940)*, Monthly Notices of the Royal Astronomical Society, 544, 3496, 2025.
[^cite-sim2024]: Sim, S.J., Hirschmann, M.M. and Hier-Majumder, S., *[Volatile and trace element storage in a crystallizing martian magma ocean](https://doi.org/10.1029/2024JE008346)*, Journal of Geophysical Research: Planets, 129, e2024JE008346, 2024. Eq. 6.
[^cite-bower2018]: Bower, D.J., Sanan, P. and Wolf, A.S., *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49, 2018.
[^cite-vanbuchem2023]: van Buchem, C.P.A., Miguel, Y., Zilinskas, M. and van Westrenen, W., *[LavAtmos: An open-source chemical equilibrium vaporization code for lava worlds](https://doi.org/10.1111/maps.13994)*, Meteoritics & Planetary Science, 58, 1149-1161, 2023.
