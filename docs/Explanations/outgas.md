# Outgassing, rock vapour and volatile trapping

This page answers one question: **where do the volatiles of a planet go during each step of a PROTEUS run, and which part of the code moves them?**

The short answer: in each step, a chemistry solver splits the volatiles of the planet between the atmosphere and the magma ocean. Before that solve, a trapping step can trap part of the volatiles dissolved in the melt in the mantle that froze since the last step. After the solve, an optional step adds rock vapour from the hot melt surface to the atmosphere.

The first section gives the order of the steps. The later sections explain the species-to-element split, the H$_2$ binodal, rock vapour, trapping, melt drainage, the layer scheme, the burial ledger and the limits of the trapping inputs. A section whose part of the code has a Validation case ends with a link to it. For the config keys, see [Escape and outgassing configuration](../Reference/config/escape_outgas.md#outgassing-outgas). For the output columns, see [Output format](../Reference/output.md).

## Terms used on this page

- **Volatile**: an element or molecule that can be a gas at the surface. PROTEUS tracks 5 volatile elements: H, C, N, O and S. It also tracks 11 species made from them: H$_2$O, CO$_2$, H$_2$, CO, CH$_4$, N$_2$, NH$_3$, S$_2$, SO$_2$, H$_2$S and O$_2$. The noble gases He, Ne, Ar, Kr and Xe are tracked too.
- **Reservoir**: a place where a volatile can be. Each species and each element has 3: the atmosphere (`_kg_atm`), the melt (`_kg_liquid`) and the solid mantle (`_kg_solid`). The 5 volatile elements have a fourth, `_kg_condensed`, for condensates.
- **Condensate**: a solid or liquid that forms from the gas at the surface, for example graphite. Only atmodeller makes condensates, and only with `outgas.atmodeller.include_condensates = true`. With `outgas.prevent_desaturation = true`, condensed mass stays condensed: later solves cannot turn it back into gas.
- **Element total**: `{e}_kg_total`, the mass of element `{e}` in the whole planet \[kg\]. Here `{e}` stands for an element symbol, and `{sp}` below stands for a species name.
- **Budget**: the mass of an element that the planet starts with, from the config.
- **Escape**: the loss of atmospheric gas to space. The escape step lowers the element totals. The chemistry solve does not change them, except for oxygen (see the next section).
- **Chemistry solver**: the module that computes how a volatile inventory divides between gas and melt at the surface. It is set by `outgas.module`: CALLIOPE and atmodeller compute the chemical equilibrium, and the dummy module uses a fixed rule. See [Model description](model.md#volatile-outgassing-calliope-atmodeller).
- **Interior solver**: the module that evolves the mantle temperature and melt fraction, set by `interior_energetics.module` (for example Aragog or SPIDER).
- **Melt fraction** $\Phi$: the mass fraction of the whole mantle that is liquid \[1\], `Phi_global` in the output. The interior solver also has a local melt fraction for each point of its radial grid. Do not confuse either with the porosity $\phi$ (lower case) below, which is a volume fraction.
- **Solid limit**: `params.stop.solid.phi_crit`, the melt fraction at or below which the mantle counts as solid.
- **Oxygen fugacity** (fO$_2$): a measure of how oxidising the melt is, in the form of an effective O$_2$ pressure. PROTEUS gives it relative to the iron-wustite (IW) buffer, the fO$_2$ at which iron metal and FeO are in equilibrium. `outgas.fO2_shift_IW` ([config](../Reference/config/escape_outgas.md#outgassing-outgas)) is that shift in log10 units, so a value of 2 is 100 times the IW fO$_2$.
- **Mass threshold**: `outgas.mass_thresh` \[kg\]. The chemistry solve treats H, C, N or S with less mass than this as absent. Escape and the desiccation check (see **Desiccated** below) also use it (see [the config reference](../Reference/config/escape_outgas.md#outgassing-outgas)).
- **Trapping**: the step that moves volatiles from the melt into the mantle that froze in the step, set by `outgas.trap_mode`.
- **Node**: one cell of the 1-D radial mesh of the interior solver (or of the atmosphere), with its mass, melt and volatiles. Aragog calls them staggered cells.
- **Layer**: a contiguous set of nodes in one phase state: a liquid, mush or solid layer.
- **Run**: one PROTEUS simulation.
- **State**: the phase state of a node (liquid, mush or solid), or the state of a run (running, completed, errored).
- **Trapped mass**: volatile mass that moved from the melt into the mantle that froze, recorded in `_kg_trapped`. It is the sum of the buried and the locked mass. See [Volatile trapping](#volatile-trapping-in-the-freezing-mantle).
- **Buried**: volatiles held at mush nodes, in the pore melt and crystals of nodes below the rheological transition $\Phi_c$. Only the layer scheme has buried mass.
- **Locked**: volatiles held at solid nodes, whose porosity is at or below `outgas.trap_phi_min`, including the melt pockets frozen in them. The front scheme puts its trapped mass straight into the frozen mantle, so all of it is locked.
- **Desiccated**: the state of a planet that has lost its volatiles. For each volatile and noble gas, the mass that can still escape or outgas is below the mass threshold. Trapped mass, and condensed mass kept with `outgas.prevent_desaturation = true`, do not count.
- **Rock vapour**: gas of rock-forming elements and their compounds, such as SiO, Na, Mg and Fe, above a hot melt surface.
- **Giant impact**: a collision of the planet with another large body, which adds mass and can melt the mantle again. See [Accretion](model.md#accretion-morrigan).
- **Initialisation stage**: the first 4 iterations of a run. They set up the starting state, and the mantle does not evolve in them. The time is set back to 0 after each of them except the last, which ends the stage and keeps its time.
- **`M_atm`**: the total mass of the atmosphere \[kg\].

## The order of the volatile steps

**Question.** In what order do the volatile steps run, and what does each step change?

**Short answer.** Trapping runs first, then escape, then outgassing. So escape and outgassing both see the mass that trapping took in the same step.

Each iteration of the [coupling loop](coupling_loop.md#execution-order-per-iteration) runs these steps, in this order:

1. **Interior and structure.** The interior solver advances the mantle temperature and the melt fraction $\Phi$. A giant impact, if one is due, is applied here.
2. **Trapping** (`run_trapping`). If `outgas.trap_mode` is `"front"` or `"layers"`, part of the dissolved volatiles moves from the melt to the solid mantle. See [Volatile trapping](#volatile-trapping-in-the-freezing-mantle).
3. The orbit and stellar steps run. They do not change the volatiles.
4. **Escape** (`run_escape`). Escape lowers the element totals.
5. **Outgassing** (`run_outgassing_and_vapourisation`). The chemistry solver splits the inventory between the atmosphere and the melt. If `outgas.vapourise = true`, rock vapour follows.
6. **Atmosphere climate.** The climate solver reads the updated composition.

The outgassing step takes the first of these 3 paths that applies:

- **Desiccated** (`run_desiccated`). This path runs when the planet is desiccated and the escape step has already removed the lost mass.
    - The atmosphere is emptied.
    - The melt is emptied, except while trapping is on (`outgas.trap_mode` is `"front"` or `"layers"`). Then the melt keeps its mass, so that trapped mass can return to the melt if the mantle melts again.
    - The solid mantle and the condensates keep their mass.
    - Rock vapour can still enter the atmosphere (see [Rock vapour](#rock-vapour-from-the-melt-surface)).
- **Crystallised** (`run_crystallized`). This path needs `params.stop.solid.freeze_volatiles = true`.
    - It starts when $\Phi$ falls to the solid limit `params.stop.solid.phi_crit` or below. It stays on after that; only a giant impact turns it off.
    - The melt and the atmosphere exchange no volatiles.
    - Escape still removes gas, so the code scales every atmospheric reservoir down by the same fraction. The composition does not change.
    - This path accepts only `escape.reservoir` ([config](../Reference/config/escape_outgas.md#atmospheric-escape-escape)) = `"outgas"` or `"bulk"`; any other value stops the run with an error.
- **Normal** (`run_outgassing`). The chemistry solver runs. This is the usual path while a magma ocean exists.

On the normal path, the step does 3 things:

1. It takes out the mass that the solver must not change: the trapped mass, and, with `outgas.prevent_desaturation = true`, the condensed mass.
2. It runs the chemistry solver.
3. It puts the withheld mass back and adds up the atmospheric mass `M_atm`.

**Oxygen.** With CALLIOPE or atmodeller and `planet.fO2_source = "user_constant"`, the solver holds the fO$_2$ at the set value. To do that, it adds or removes oxygen, so `O_kg_total` is a result of each solve and not a fixed budget.

With `"from_O_budget"`, the O budget is the input, and the solver finds the fO$_2$ that matches it. The dummy module computes an O total in each call of the initialisation stage while the O total is still 0. In all other calls it keeps the O total as it is.

**Initialisation check.** In the initialisation stage, the code writes the element budgets to `{e}_kg_total`. They come from `planet.elements` with the default `planet.volatile_mode = "elements"` ([config](../Reference/config/planet.md#parameters-planet)), or from the initial partial pressures of `planet.gas_prs` with `"gas_prs"` ([config](../Reference/config/planet.md#partial-pressures-planetgas_prs)).

You can also set an O budget, with `planet.elements.O_mode` set to `"kg"`, `"ppmw"` or `"FeO_mantle_wt_pct"`. With such an O budget and `planet.fO2_source = "user_constant"`, the code compares your budget with the oxygen of the first solve. It stops the run if the two differ by more than 50 % of the solve value. The default, `O_mode = "ic_chemistry"`, skips this check.

**Where this is checked.** No Validation case covers the order of the steps. After each outgassing step, a runtime check tests the mass budget; see [Mass conservation](coupling_loop.md#mass-conservation).

## From species to elements

**Question.** The chemistry solver returns masses of molecules. How does the code turn them into the element masses that escape and the budget use?

**Short answer.** It splits each molecule's mass over its elements by their share of the molecule's molar mass.

For an element $e$ with $n_e$ atoms in a molecule, the share $f_e$ of the molecule's mass that is element $e$ is

$$
f_e = \frac{n_e\, m_e}{\sum_i n_i\, m_i},
$$

where $n_i$ is the number of atoms of element $i$ in the molecule, $m_i$ is the molar mass of element $i$ \[kg mol$^{-1}$\], and the sum runs over every element of the molecule.

**Example.** For water, $f_\mathrm{O} = 0.88809$ and $f_\mathrm{H} = 0.11191$. So 1 kg of dissolved water adds 0.888 kg to `O_kg_liquid` and 0.112 kg to `H_kg_liquid`.

The shares of one molecule always add up to 1, so the split conserves mass for any table of molar masses.

The atmodeller module writes this split for the atmosphere, melt and solid reservoirs. It does not write `{e}_kg_total` for H, C, N or S: the escape step keeps those totals up to date. Oxygen is the exception described above. Trapping uses the same rule (`element_masses_from_species`) to move element masses with the species it traps.

**Where this is checked.** [Species-to-element mass split](../Validation/outgas/atmodeller.md).

## Hydrogen and silicate melt: the binodal

**Question.** Hot hydrogen and silicate melt can mix completely. Does PROTEUS include that?

**Short answer.** Not in a run. The config loader rejects both switches for it: `outgas.h2_binodal = true` and `interior_struct.zalmoxis.global_miscibility = true` ([config](../Reference/config/interior.md#zalmoxis-interior_structzalmoxis)). The rest of this section describes the code that these switches would use.

H$_2$ and silicate melt are **miscible** when they form 1 phase. Where they separate into a hydrogen-rich gas and a silicate melt, they are immiscible. The line in pressure and temperature where they separate is the **binodal**.

Rogers et al. [^cite-rogers2025] compute the binodal for H$_2$ and MgSiO$_3$, which stands in for the silicate melt. In their sub-Neptune models, the 2 are miscible above about 4000 K. In the code, the binodal temperature depends on the pressure and on the H$_2$ mass fraction.

With `outgas.h2_binodal = true`, `apply_binodal_h2` would replace the H$_2$ split of the chemistry solver. It takes the fraction $\sigma$ of H$_2$ that dissolves in the melt from the binodal model of the Zalmoxis interior-structure module ([Model description](model.md#interior-structure-zalmoxis)). The inputs are the surface pressure, the magma temperature and the H$_2$ mass fraction. The dissolved mass is $\sigma$ times the total H$_2$ mass, and the rest stays in the atmosphere:

$$
M_\mathrm{H_2,\,melt} = \sigma\, M_\mathrm{H_2,\,total}, \qquad M_\mathrm{H_2,\,atm} = (1-\sigma)\, M_\mathrm{H_2,\,total},
$$

where $M$ is a mass \[kg\] and $M_\mathrm{H_2,\,total}$ is the H$_2$ in the melt and the atmosphere together. $\sigma$ is between 0 (immiscible, all H$_2$ in the atmosphere) and 1 (all H$_2$ in the melt).

**Where this is checked.** [H2-MgSiO3 binodal](../Validation/outgas/binodal.md). That page covers the H$_2$ reservoir bookkeeping of this function. No case compares the split with Rogers et al.

## Rock vapour from the melt surface

**Question.** A very hot magma ocean also evaporates rock. How does that vapour enter the atmosphere?

**Short answer.** With `outgas.vapourise = true`, the LavAtmos code [^cite-vanbuchem2023] computes the vapour above the melt after the volatile solve, and adds it to the atmosphere.

LavAtmos computes the vapour that is in equilibrium with the melt surface. FastChem, a separate gas-chemistry package, then combines it with the volatile gas.

The inputs are:

- the surface temperature, $\max(T_\mathrm{min}, T_\mathrm{magma})$ \[K\], with $T_\mathrm{min}$ = `outgas.lavatmos.T_min` and $T_\mathrm{magma}$ the magma surface temperature from the interior;
- the volatile surface pressure from the chemistry solve \[bar\];
- the number (mole) fractions of the volatile elements in the atmosphere;
- the melt composition file `outgas.lavatmos.melt_comp_name`.

To turn masses into number fractions, the code needs the molar mass of each species. It sums the atomic masses of the formula and gives the result in g mol$^{-1}$, the unit that FastChem and LavAtmos use. For SiO$_2$ this is 60.083 g mol$^{-1}$.

The vapour step runs on the normal path, after the chemistry solve, and on the desiccated path. It is skipped when $\Phi$ is at the solid limit `params.stop.solid.phi_crit` or below.

Rock vapour enters the atmosphere without being taken from the mantle. So with `outgas.vapourise = true`, whole-planet mass is not conserved; see [Model description](model.md#whole-planet-mass-is-not-conserved-when-vapourisation-is-enabled).

**Where this is checked.** [Rock-vapour molar masses](../Validation/outgas/lavatmos.md).

## Volatile trapping in the freezing mantle

**Question.** When the magma ocean freezes, how much of its dissolved water and gas stays in the solid mantle instead of reaching the atmosphere?

**Short answer.** Each kilogram of frozen mantle keeps some volatiles: a little inside the crystals, and all of those in the melt that stays between the crystals (the **pore melt**). With `outgas.trap_mode = "front"`, the code computes the pore melt from how fast melt drains out of the freezing mush layer, the **freezing front**. With `"layers"`, every node of the interior mesh drains its own pore melt (see [Layer drainage node by node](#layer-drainage-node-by-node)). With `"none"`, the default, nothing is trapped.

### How to turn trapping on and read the result

Trapping needs the Aragog interior solver, because only Aragog gives the radial profiles that the freezing front and the layer drainage need:

```toml
[interior_energetics]
module = "aragog"

[outgas]
trap_mode = "front"
```

Trapping compares each step with the step before it. So it does nothing in the initialisation stage, when the output has no earlier row, or when the time is 0. The first step after the initialisation stage compares with the last initialisation row, and it can trap; nothing is trapped when no mass froze.

These output columns show what trapping did ([full list](../Reference/output.md#solid-phase-volatile-trapping)):

- `trap_kg_step` \[kg\]: the net volatile mass moved from the melt into the solid in this step; negative when trapping returned more than it took.
- `trap_kg_returned` \[kg\]: the volatile mass trapping returned to the melt in this step, by a remelt or, in `"layers"`, by drainage.
- `trap_kg_cumulative` \[kg\]: the net mass trapped since the start of the run.
- `{sp}_kg_trapped` and `{e}_kg_trapped` \[kg\]: the mass of each species and of each element that is trapped at the end of the step.
- `trap_F_tl` \[1\]: the trapped melt fraction of the step, $F_\mathrm{tl}$ below.
- `trap_branch`: which path the step took (see [When the front cannot be integrated](#when-the-front-cannot-be-integrated)).
- `trap_tau_D`, `trap_tau_s` and `trap_t_res` \[yr\]: the 2 timescales and the residence time of the drainage section. The equations below use seconds; the output gives years. These columns show why a step took branch 1 or 2. On branch 3 they are not computed (NaN).
- `trap_L_front` \[m\]: the front thickness.

The other trapping inputs, such as `outgas.D_const_<species>` and `outgas.trap_phi_min`, are in the [config reference](../Reference/config/escape_outgas.md#outgassing-outgas).

### The trapping rule

In one step, the mantle mass $\Delta M$ \[kg\] freezes. The code gets it from the drop of the melt fraction: $\Delta M = M_\mathrm{mantle}\,(\Phi_{n-1} - \Phi_n)$, where $M_\mathrm{mantle}$ is the mantle mass \[kg\] and the index $n$ counts the steps.

Each volatile species $Z$ then locks the mass $\Delta m_Z$ \[kg\] in that frozen mantle [^cite-sim2024]:

$$
\Delta m_Z = \left[(1 - F_\mathrm{tl})\, D_Z + F_\mathrm{tl}\right] C_Z\, \Delta M.
$$

The symbols are:

- $C_Z$: the mass fraction of $Z$ in the melt \[1\], from the last chemistry solve.
- $D_Z$: the crystal/melt partition coefficient \[1\]. It is the mass fraction of $Z$ in the crystals divided by that in the melt, set by `outgas.D_const_<species>`.
- $F_\mathrm{tl}$: the trapped melt fraction \[1\]. It is the mass fraction of the frozen mantle that is pore melt.

The bracket is the effective partition coefficient. If $F_\mathrm{tl} = 0$, it is $D_Z$: only the crystals take up the volatile. If $F_\mathrm{tl} = 1$, it is 1: the frozen mass takes the volatile at the melt concentration, as if it were all melt.

The default $D_Z$ is 0.0017 for H$_2$O and 0 for every other species. A species with $D_Z = 0$, such as CO$_2$ or a noble gas, is locked only with the pore melt. For water, the pore melt locks more than the crystals when $F_\mathrm{tl}$ is above about 0.0017.

**Example.** Take water at $C_Z = 10^{-3}$ (1000 ppm by mass), with $D_Z = 0.0017$ and $F_\mathrm{tl} = 0.02$. Let $2\times10^{23}$ kg of mantle freeze. The bracket is 0.021666, so $4.33\times10^{18}$ kg of water is locked. Of this, the pore melt locks $4.0\times10^{18}$ kg and the crystals take up $3.3\times10^{17}$ kg.

The trapped mass moves from `{sp}_kg_liquid` to `{sp}_kg_solid`, and `{sp}_kg_trapped` records it. The element reservoirs move with it, by the split of the [species-to-element section](#from-species-to-elements). The element totals do not change.

Trapped mass cannot escape or outgas: the escape step and the desiccation check do not count it. The mass a step traps of a species is never more than the mass that the melt holds.

**Where this is checked.** [Solid-phase volatile trapping](../Validation/outgas/trapping.md).

## Melt drainage across the freezing front

**Question.** How much melt is still in the pores when a layer of crystals has frozen?

**Short answer.** Melt leaves the freezing layer only if 2 things happen. The melt must flow out through the pores, and the crystal framework must deform to close the pores. The slower of the 2 sets the rate. The melt that is left when the layer is fully frozen gives $F_\mathrm{tl}$.

### Terms for this section

- **Mush**: a mix of crystals and melt.
- **Matrix**: the connected framework of crystals in a mush.
- **Freezing front**: the layer of mush between the magma ocean above and the solid, nearly melt-free mantle below. The code uses "front" for this layer.
- **Porosity** $\phi$: the volume fraction of melt in the mush \[1\]. It is a volume fraction, while the melt fraction $\Phi$ is a mass fraction.
- **Permeability** $k$: how easily melt flows through the pores \[m$^2$\].
- **Percolation (Darcy flow)**: the flow of melt through the pores of the matrix. The output reference calls branch 1 "Darcy" after it.
- **Parcel**: a small piece of mush. As the front moves up, a parcel enters the front at the top and leaves it at the base.
- **Mesh and nodes**: the radial grid of the interior solver, and its cells (see [Terms used on this page](#terms-used-on-this-page)).
- **Critical melt fraction** $\Phi_c$: `interior_energetics.rfront_loc` (default 0.4), the melt fraction at the centre of the transition where the interior solver changes the mantle from melt-like to solid-like flow. It is not the solid limit `params.stop.solid.phi_crit`.

### Where the front is

The code finds the front on the interior mesh (`locate_front`):

- **Top**: where the local melt fraction of the interior solver falls below $\Phi_c$. So trapping and the interior use the same transition.
- **Base**: where the porosity falls below `outgas.trap_phi_min` (default 0.01). If the lowest node is still porous, the core-mantle boundary is the base.

The interior solver gives a local melt fraction for each node, and the top uses it directly.

The code also computes a porosity for each node from the densities \[kg m$^{-3}$\] of the mush $\rho$, the solid $\rho_s$ and the melt $\rho_l$: $\phi = (\rho_s - \rho)/(\rho_s - \rho_l)$. The base uses this porosity. The porosity of the top node is where the drainage equation starts.

### The 2 timescales

The **percolation time** $\tau_D$ is how long melt takes to flow through a front of thickness $L$ \[m\]:

$$
\tau_D = \frac{L}{w_D}, \qquad w_D = \frac{|\Delta\rho|\, g\, F(\phi)}{\eta_\mathrm{melt}}.
$$

Here $w_D$ is the percolation speed \[m s$^{-1}$\] and $\Delta\rho$ is the solid density minus the melt density \[kg m$^{-3}$\], averaged over the front. $g$ is the gravity \[m s$^{-2}$\] from the interior solver, and $\eta_\mathrm{melt}$ is the melt viscosity \[Pa s\], `10**interior_energetics.melt_log10visc`. $F(\phi)$ \[m$^2$\] is the mobility: the permeability $k$ divided by the porosity, as on the [drainage Validation page](../Validation/outgas/compaction.md).

$F(\phi)$ is the same law that Aragog uses for the separation of melt and solid [^cite-bower2018]. It has 3 regimes: Blake-Kozeny-Carman flow at low porosity, Rumpf-Gupte flow at middle porosity and Stokes settling of single grains at high porosity. The Validation page gives the formula of each.

$F$ grows with the square of the grain size, `interior_energetics.grain_size` \[m\]. With the default buoyancy setting, it rises with $\phi$ and then levels off at the Stokes value $2a^2/9$, with $a$ the grain size.

That buoyancy setting is `interior_energetics.aragog.separation_buoyancy`. With the default, `"mixture"`, the buoyancy of the melt is taken against the mush, which adds a factor $1 - \phi$ to the 2 porous-flow regimes. With `"solid"` it is taken against the solid.

The **matrix time** $\tau_s$ is how long the matrix takes to deform and close the pores:

$$
\tau_s = \frac{\mu_s}{|\Delta\rho|\, g\, L},
$$

where $\mu_s$ is the mush viscosity \[Pa s\], set by `outgas.trap_mush_log10visc` ($10^{20}$ Pa s by default).

### The drainage equation

A parcel takes the **residence time** $t_\mathrm{res} = L / v_f$ to cross the front. Here $v_f$ is the speed at which the front rises \[m s$^{-1}$\], from the mass $\Delta M$ that froze in the step.

While the parcel crosses the front, its porosity falls with the time $t$ \[s\]. The slower of the 2 processes sets the rate:

$$
\frac{d\phi}{dt} = -\frac{\phi}{\max(\tau_D(\phi),\ \tau_s)}.
$$

The code integrates this from the porosity at the top of the front, over the time $t_\mathrm{res}$ (`drainage_integral`). The porosity $f$ at the end is the trapped melt as a volume fraction. The code then turns it into the mass fraction that the trapping rule uses (`volume_to_mass_fraction`):

$$
F_\mathrm{tl} = \frac{\rho_l\, f}{\rho_l\, f + \rho_s\,(1 - f)}.
$$

**Example.** Take the Mars-like front of the [drainage Validation case](../Validation/outgas/compaction.md). It is 60 km thick, with $\Delta\rho = 330$ kg m$^{-3}$ and $g = 3.711$ m s$^{-2}$, and a parcel enters it at $\phi = 0.3$. Use 1 mm grains, a melt viscosity of 100 Pa s and the default mush viscosity. For the mass fraction, take melt and solid densities of 2800 and 3130 kg m$^{-3}$.

| Front speed | $t_\mathrm{res}$ | $\tau_D$ at entry | $\tau_s$ | Trapped melt, volume fraction | $F_\mathrm{tl}$ (mass fraction) |
|---|---|---|---|---|---|
| 70 cm yr$^{-1}$ | $8.6\times10^4$ yr | $7.0\times10^4$ yr | $4.3\times10^4$ yr | 0.194 | 0.177 |
| 5 cm yr$^{-1}$ | $1.2\times10^6$ yr | $7.0\times10^4$ yr | $4.3\times10^4$ yr | 0.109 | 0.099 |

A slower front gives the melt more time to drain, so it traps less.

$\tau_D$ grows as $\phi$ falls, so here $\tau_D$ is longer than $\tau_s$ during the whole crossing. Percolation therefore limits the drainage, and the code records `trap_branch` = 1.

$\tau_s$ is proportional to $\mu_s$, and it equals $\tau_D$ at entry when $\mu_s$ is about $1.6\times10^{20}$ Pa s. Above that value, the code records branch 2 (matrix-limited). The trapped melt then changes slowly at first: at $1.8\times10^{20}$ Pa s it is still 0.194 for the fast front. At $10^{21}$ Pa s it is 0.246. A lower mush viscosity gives the result of the table.

### When the front cannot be integrated

The code does not integrate a front that the mesh cannot resolve. Then it uses an upper bound for $F_\mathrm{tl}$: the porosity at the top of the front, turned into a mass fraction and capped at $\Phi_c$. This happens when:

- the front spans fewer than `outgas.trap_n_front_min` nodes (default 3);
- the front is thicker than `outgas.trap_max_front_fraction` times the mantle thickness (the default, 1, sets no limit);
- the mush is split into 2 or more porous layers with solid between them;
- the front reaches the top node of the mesh, at the surface;
- the front has no positive thickness;
- the melt is denser than the solid somewhere in the front;
- the front speed is zero.

Each such step logs a warning with the cause. The column `trap_branch` records the path of every step:

| `trap_branch` | Name in the output reference | Meaning |
|---|---|---|
| 0 | no front | no front this step; a freezing step still locks at $D_Z$ |
| 1 | Darcy | drained, limited by percolation at the top of the front |
| 2 | matrix | drained, limited by matrix deformation at the top of the front |
| 3 | guard | upper bound, the front could not be integrated |
| 6 | no interior state | the interior profiles were missing; the step locks at $D_Z$ alone and logs a warning |
| 7 | remelting | the step returns trapped mass to the melt; a diagnostic, since escape and desiccation read `trap_kg_returned` |

The code does not use the values 4 and 5.

**Where this is checked.** [Melt drainage across the freezing front](../Validation/outgas/compaction.md).

## Layer drainage node by node

**Question.** Can the pore melt drain through the whole mush, instead of across one front in one step?

**Short answer.** Yes, with `outgas.trap_mode = "layers"`. Every node of the Aragog mesh that is in the mush keeps its own pore melt from step to step. The melt drains up into the node above, and a node keeps what it still holds when it freezes. This is the per-node drainage of Miyazaki and Korenaga (2019, section 2.4) on the interior mesh [^cite-miyazaki2019]. It needs no front, front speed or residence time.

### Phase states

Each node is classified at every step from its melt fraction $x$ and the melt volume fraction $\phi = x\rho_s / (x\rho_s + (1 - x)\rho_l)$. Here $\rho_s$ and $\rho_l$ \[kg m$^{-3}$\] are the solid and melt densities at the phase boundaries:

- **liquid**: $x \ge \Phi_c$ (`interior_energetics.rfront_loc`), part of the magma ocean;
- **mush**: $x < \Phi_c$ and $\phi >$ `outgas.trap_phi_min`, so the pore melt leaves only by drainage;
- **solid**: $\phi \le$ `outgas.trap_phi_min`; the pore melt left in the node is locked.

A contiguous set of nodes in one phase state is a liquid, mush or solid layer. A node that enters the mush from above starts with the pore melt of the transition, whose mass fraction is exactly $\Phi_c$.

### Drainage

The pore melt $\psi$ of a node changes only by drainage. Take the nodes of a mush layer, counted from its base. The melt volume flux \[m$^3$ s$^{-1}$\] out of the top of node $k$ is

$$
q_k = \min\left(A_k\,\psi_k\,w_D(\psi_k),\; q_{k-1} + \frac{\psi_k V_k}{\tau_{s,k}}\right), \qquad q_\mathrm{base} = 0 .
$$

$A_k$ \[m$^2$\] is the area of the node's upper face and $V_k$ \[m$^3$\] its volume. $w_D$ is the percolation speed of the front scheme at the node's own density contrast and gravity. $\tau_{s,k} = \mu_s / (|\Delta\rho|\,g\,d_k)$ is the matrix time at the depth $d_k$ \[m\] of the node's centre below the top of its mush layer. The first term is the most the permeability passes. The second is the melt that compaction of the node and of the nodes below it supplies.

A single node with $L = V/A$ follows the front scheme's drainage equation exactly on the percolation branch. On the matrix branch it does not: its matrix time uses the depth of its centre, $L/2$, so it is twice the front's $\mu_s / (|\Delta\rho|\,g\,L)$.

The flowing melt is $\psi$ capped at the solver's own porosity, the mean of its values at the start and end of the step. Pore melt the record holds beyond it has crystallised in place, keeps its volatiles and does not flow.

Melt leaving the top of a mush layer rejoins the magma ocean, or leaves through the surface when the layer reaches the top of the mesh. A mush layer with a solid node above it is sealed and keeps its melt. A node whose tables put the melt at or above the solid density cannot drive its melt upward. It takes the front scheme's no-drainage bound and keeps its pore melt, and so do the mush nodes below it in the same mush layer. In Miyazaki and Korenaga (2019) such melt sinks instead; the layer scheme does not move melt downward.

The code integrates the fluxes with a second-order Runge-Kutta scheme. Each sub-step is a tenth (`SUBSTEP_FRACTION = 0.1`) of the shortest time any node takes to pass on what it holds.

### Volatiles

A node's pore melt leaves contact with the magma ocean when it enters the mush. All the volatiles in it are buried then: $C_Z\,F\,m$ of each species, with $F = \Phi_c$ the melt mass fraction of its pore melt and $m$ \[kg\] the node mass. They move with the melt as it drains. Melt that leaves a mush layer returns its share to the melt reservoir, and what a node holds when it freezes is locked.

Crystals that grow in a liquid node take $D_Z\,C_Z$ of their mass from the ocean. Crystals that grow in the pores of the mush draw on the pore melt, which the record already holds. Over the life of a node that does not drain, the two add up to $[(1 - \Phi_c)D_Z + \Phi_c]\,C_Z\,m$. That is the [trapping rule](#the-trapping-rule) at the no-drainage bound $F_\mathrm{tl} = \Phi_c$.

The node masses are Aragog's cell masses, rescaled so that they sum to $M_\mathrm{mantle}$. Aragog writes each cell mass as its current density times a fixed volume, and that sum grows past the mantle mass as the mantle cools.

Each step moves the net mass, new burials minus returns, between `{sp}_kg_liquid` and `{sp}_kg_solid`. It uses the same `{sp}_kg_trapped` bookkeeping as the front scheme. A step can therefore return more than it buries.

### The chemistry sees the ocean melt

The pore melt of the mush and solid nodes is cut off from the ocean. So with `"layers"` the chemistry solver dissolves volatiles into the melt of the liquid nodes alone. For the length of the solve, `Phi_global` reads `trap_Phi_ocean`: the liquid nodes' share of the melt times the global melt fraction. A step without a value, which the helpfile stores as 0, leaves `Phi_global` as it is. It is restored after the solve. The next step takes its melt concentration on that same ocean melt.

### Remelting node by node

A node that becomes liquid again returns everything trapped at it to the melt. A liquid or mush node whose melt fraction rises from $x_0$ to $x$ melts the share $(x - x_0)/(1 - x_0)$ of its crystals. That share of its crystal store goes with the new melt. A liquid node returns it to the ocean, so a liquid node at $x = 1$ holds nothing. A mush node puts it into its pore melt, which drains. Nodes that stay solid keep theirs, even when the global melt fraction rises. The burial ledger of the next section is not used in this mode.

Every return reaches escape and desiccation through `trap_kg_returned`, whether a node remelted or melt drained back to the ocean. On a step with a return, the escape floor is off and a desiccated planet is checked again. `trap_branch = 7` marks a net return by remelting nodes, as a diagnostic only.

### Resume, mesh changes and missing profiles

The per-node record lives on the interior object between steps. It is saved in each interior snapshot as `trap_*_s` variables, and a resumed run reads it back. Without a saved record, or with one sized for another mesh, the trapped mass on the helpfile is assigned to the nodes that are not liquid, in proportion to their mass. The mush the new record finds then buries nothing new, since it already holds that mass.

A step without the interior profiles traps at $D_Z\,C_Z\,\Delta M$ alone when the mantle freezes. When it remelts, the step returns the remelted share of the solid and takes the same share out of the record. It does not use the burial ledger, so nothing is returned twice.

### Output

The trapping columns have one meaning in both modes. For `"layers"` they read as follows:

- `trap_F_tl` \[1\]: the mass-weighted trapped melt fraction of the nodes that froze in the step, 0 when none did.
- `trap_n_front`, `trap_L_front` \[m\], `trap_tau_D` and `trap_tau_s` \[yr\]: the node count, thickness, percolation time and matrix time of the thickest mush layer, which stands for the front.
- `trap_v_front` \[m s$^{-1}$\] and `trap_front_courant` \[1\]: the front speed from the mass that froze, at the base of the thickest mush layer, and its advance in that layer's thicknesses. `trap_w_matrix_over_vf` is not computed, since no single front top exists, and the helpfile stores it as 0.
- `trap_t_res` \[yr\]: the mass-weighted time the nodes that froze spent in the mush.
- `trap_n_mush`: the mush nodes of the whole mesh. `trap_n_exited`: nodes that froze in the step.
- `trap_frac_bound` \[1\]: the mass share of the nodes that froze on the no-drainage bound. It includes nodes that drained before their mush layer was sealed, so it is not the share that kept their entry pore melt.
- `trap_kg_returned` \[kg\]: the mass returned to the melt this step. `trap_Phi_ocean` \[1\]: the ocean melt the chemistry dissolves into. `trap_n_substeps`: the drainage sub-steps of the step.

### Limits

- With `params.dt.mushy_maximum` set, the interior time step is capped while `params.stop.solid.phi_crit` < `Phi_global` < `params.dt.mushy_upper`, and not below that band. Many nodes can then cross the floor in one step, which the record follows only at that step's resolution.
- Before the record starts, the code knows no earlier melt fraction per node. The crystals grown in the step before it are not counted.

**Where this is checked.** [Pore-melt drainage of the layer scheme](../Validation/outgas/layer_drainage.md), [Per-node record of the layer scheme](../Validation/outgas/node_record.md) and [Layer trapping step](../Validation/outgas/layer_trapping.md).

## Remelting and the burial ledger

**Question.** If the mantle melts again, which locked volatiles come back to the melt?

**Short answer.** In the `"front"` mode, the solid that froze last is the first to melt. So a remelt returns the mass that was locked most recently. The `"layers"` mode decides remelting per node instead (see [Remelting node by node](#remelting-node-by-node)).

The **burial ledger** stores the mass locked in each step as an entry. Each entry has 2 melt fractions: $\Phi$ before the step and $\Phi$ after it.

When $\Phi$ rises to a higher value, the code returns the mass of each entry whose melt fractions are below that value (`remelt_to`). An entry that the higher value cuts in 2 returns the remelted share of its mass.

**Example.** The mantle freezes from $\Phi = 1$ to 0.7 and locks 3 kg. Then it freezes from 0.7 to 0.5 and locks 2 kg. Then it remelts to $\Phi = 0.6$. That remelts half of the second entry, so 1 kg returns to the melt. The first entry and the other half of the second entry stay locked.

At each remelt, the code builds the ledger from the output columns `Phi_global` and `{sp}_kg_trapped`, so a resumed run has the same ledger.

If the history does not account for all of the trapped mass, the rest returns in proportion to the share of the solid that remelted. With the step index $n$, that share is $(\Phi_n - \Phi_{n-1})/(1 - \Phi_{n-1})$.

**Where this is checked.** [Burial ledger for remelting](../Validation/outgas/burial_ledger.md).

## Limits of the trapping inputs

**Question.** Which values can the trapping inputs take?

**Short answer.** The config loader checks each limit below. A value outside a limit stops the load with an error.

- `outgas.trap_mode`: `"none"`, `"front"` or `"layers"`. `"front"` and `"layers"` need `interior_energetics.module = "aragog"`.
- `outgas.trap_phi_min`: above 0 and below 1.
- `outgas.trap_n_front_min`: 2 or more.
- `outgas.trap_max_front_fraction`: above 0 and at most 1.
- `outgas.D_const_<species>`: 0 or more.
- `interior_energetics.rfront_loc`: above 0 and below 1.
- `interior_energetics.grain_size`: above 0 \[m\].
- `outgas.trap_mush_log10visc`: log10 of the mush viscosity in Pa s. It has no limit. Set it above 0 to give your own value. The default, -1, and any value of 0 or below give $10^{20}$ Pa s.

During a run, an $F_\mathrm{tl}$ outside 0 to 1 stops the run with an error. A front that the mesh cannot resolve gives a warning and an upper bound, not an error; see [When the front cannot be integrated](#when-the-front-cannot-be-integrated).

**Where this is checked.** No Validation case covers these limits.

[^cite-rogers2025]: Rogers, J.G., Young, E.D. and Schlichting, H.E., *[Redefining interiors and envelopes: hydrogen-silicate miscibility and its consequences for the structure and evolution of sub-Neptunes](https://doi.org/10.1093/mnras/staf1940)*, Monthly Notices of the Royal Astronomical Society, 544, 3496, 2025.
[^cite-sim2024]: Sim, S.J., Hirschmann, M.M. and Hier-Majumder, S., *[Volatile and trace element storage in a crystallizing martian magma ocean](https://doi.org/10.1029/2024JE008346)*, Journal of Geophysical Research: Planets, 129, e2024JE008346, 2024. Eq. 6.
[^cite-bower2018]: Bower, D.J., Sanan, P. and Wolf, A.S., *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49, 2018.
[^cite-miyazaki2019]: Miyazaki, Y. and Korenaga, J., *[On the timescale of magma ocean solidification and its chemical consequences: 2. Compositional differentiation under crystal accumulation and matrix compaction](https://doi.org/10.1029/2018JB016928)*, Journal of Geophysical Research: Solid Earth, 124, 3399-3419, 2019. Section 2.4 moves $\phi v_\mathrm{perc} \Delta t$ of melt at each of 200 grid cells per step, in the direction of the density contrast, with a compaction time $\eta_s / ((\rho_s - \rho_l) g L)$.
[^cite-vanbuchem2023]: van Buchem, C.P.A., Miguel, Y., Zilinskas, M. and van Westrenen, W., *[LavAtmos: An open-source chemical equilibrium vaporization code for lava worlds](https://doi.org/10.1111/maps.13994)*, Meteoritics & Planetary Science, 58, 1149-1161, 2023.
