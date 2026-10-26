# Outgassing, rock vapour and volatile trapping

This page answers one question: **where do the volatiles of a planet go during each step of a PROTEUS run, and which part of the code moves them?**

The short answer: each step, a chemistry solver splits the volatile inventory between the atmosphere and the magma ocean. Before that solve, a trapping step can bury part of the dissolved volatiles in the mantle that froze since the last step. After it, an optional step adds rock vapour from the hot melt surface to the atmosphere.

The sections below follow that order. Each section ends with a link to the page that checks it. For the config keys see [Escape and outgassing configuration](../Reference/config/escape_outgas.md#outgassing-outgas), and for the output columns see [Output format](../Reference/output.md).

## Terms used on this page

- **Volatile**: a light element or molecule that can be a gas at the surface. PROTEUS tracks the elements H, C, N, O and S, the species built from them (H$_2$O, CO$_2$, H$_2$, CO, CH$_4$, N$_2$, NH$_3$, S$_2$, SO$_2$, H$_2$S, O$_2$) and the noble gases He, Ne, Ar, Kr and Xe.
- **Reservoir**: a place where a volatile can be. The code keeps 4 for each element: the atmosphere (`_kg_atm`), the melt (`_kg_liquid`), the solid mantle (`_kg_solid`) and condensates on the surface (`_kg_condensed`). The whole-planet total is `_kg_total`.
- **Element total**: `{e}_kg_total`, the mass of element `e` in the whole planet \[kg\]. Escape lowers it; the chemistry solve only splits it between reservoirs. Oxygen is the exception (see the next section).
- **Melt fraction** $\Phi$: the mass fraction of the mantle that is liquid \[1\], `Phi_global` in the output.
- **fO$_2$**: the oxygen fugacity of the melt, given as a shift from the iron-wüstite (IW) buffer in log10 units, `outgas.fO2_shift_IW`.
- **`mass_thresh`**: `outgas.mass_thresh` \[kg\]. An element with less mass than this counts as absent.

## Where outgassing sits in each step

**Question.** In what order do the volatile steps run, and what does each step change?

Each iteration of the [coupling loop](coupling_loop.md#execution-order-per-iteration) runs the volatile steps in this order (`src/proteus/proteus.py`):

1. **Interior and structure.** The interior solver advances the mantle temperature and the melt fraction $\Phi$.
2. **Trapping** (`run_trapping`). If `outgas.trap_mode = "front"`, part of the dissolved volatiles moves from the melt to the solid mantle. See [Volatile trapping](#volatile-trapping-in-the-freezing-mantle).
3. **Escape** (`run_escape`). Escape lowers the element totals.
4. **Outgassing** (`run_outgassing_and_vapourisation`). The chemistry solver splits the remaining inventory between the atmosphere and the melt. If `outgas.vapourise = true`, rock vapour follows.
5. **Atmosphere climate.** The climate solver reads the new composition.

Trapping runs before escape and outgassing, so both see the mass that was buried in this step.

The outgassing step takes 1 of 3 paths:

- **Desiccated** (`run_desiccated`): every volatile and noble gas that escape can still reach is below `mass_thresh`, and escape explains the mass that was lost. The atmosphere and the melt are emptied; the solid mantle and the condensates keep their mass. With trapping on, the melt keeps its mass as well, so mass that a remelt returns stays in the planet.
- **Crystallised** (`run_crystallized`): with `params.stop.solid.freeze_volatiles = true`, $\Phi$ fell to `params.stop.solid.phi_crit` or below. The melt and the atmosphere no longer exchange volatiles. Escape still removes gas, so the code scales every atmospheric reservoir down by the same fraction, and the composition stays the same.
- **Normal** (`run_outgassing`): the chemistry solver runs. This is the usual path while a magma ocean exists.

On the normal path, `run_outgassing` does 3 things:

1. It takes out of the inventory the mass that the solver must not touch: the trapped mass (`trapped_mass_withheld`) and, with `outgas.prevent_desaturation = true`, the condensed mass (`condensed_mass_withheld`).
2. It calls the solver set by `outgas.module`: CALLIOPE, atmodeller or the dummy module.
3. It puts the withheld mass back and computes the atmospheric mass `M_atm`.

**Oxygen.** Under `planet.fO2_source = "user_constant"` the solver holds the fO$_2$ fixed, so it must add or remove oxygen to do so. `O_kg_total` is then an output of each solve, not a conserved budget. Under `"from_O_budget"` the O budget is the input, and the solver finds the fO$_2$ that matches it.

**First step.** In the initialisation stage, `calc_target_elemental_inventories` writes the delivered element budgets to `{e}_kg_total`. If you set an O budget in `planet.elements` and `planet.fO2_source = "user_constant"`, `check_ic_oxygen_budget` compares that budget with the oxygen of the first solve. It stops the run if the two differ by more than 50 %.

**Where this is checked.** Mass closure after each outgassing step: [The coupling loop](coupling_loop.md#mass-conservation).

## From species to elements

**Question.** The solver returns masses of molecules. How does the code turn them into the element masses that escape and the budget use?

Each species mass is split over its elements by their share of the formula mass. For an element $e$ with $n_e$ atoms in a molecule, the share is

$$
f_e = \frac{n_e\, m_e}{\sum_i n_i\, m_i},
$$

where $m_i$ is the molar mass of element $i$ \[kg mol$^{-1}$\] and the sum runs over every element of the molecule.

**Example.** For water, $f_\mathrm{O} = 0.88809$ and $f_\mathrm{H} = 0.11191$. So 1 kg of dissolved water adds 0.888 kg to `O_kg_liquid` and 0.112 kg to `H_kg_liquid`.

The shares of one molecule always add up to 1, so the split conserves mass whatever the table of molar masses says. The atmodeller wrapper (`_populate_volatile_element_reservoirs`) writes the split for the atmosphere, melt and solid reservoirs. It does not write `{e}_kg_total`, because escape owns that total. The function `element_masses_from_species` in `proteus.outgas.common` uses the same rule for trapping.

**Where this is checked.** [Species-to-element mass split](../Validation/outgas/atmodeller.md).

## Hydrogen and silicate melt: the binodal

**Question.** At high temperature, hydrogen and silicate melt mix in any ratio. How does the code treat that?

At the pressures inside a sub-Neptune and above about 4000 K, H$_2$ and MgSiO$_3$ melt are miscible: they form one phase. Below that temperature they separate into a hydrogen-rich gas and a silicate melt. The line in pressure and temperature where the two phases separate is the **binodal** [^cite-rogers2025].

With `outgas.h2_binodal = true`, `apply_binodal_h2` would replace the H$_2$ split of the chemistry solver. It computes a weight $\sigma$ from the surface pressure, the magma temperature and the H$_2$ mass fraction, with the model of Zalmoxis, and then sets

$$
M_\mathrm{H_2,\,melt} = \sigma\, M_\mathrm{H_2,\,total}, \qquad M_\mathrm{H_2,\,atm} = (1-\sigma)\, M_\mathrm{H_2,\,total},
$$

where $M$ is a mass \[kg\] and $\sigma$ is between 0 (immiscible, all H$_2$ in the atmosphere) and 1.

This switch is **not available yet**. The config loader rejects `h2_binodal = true` with an error. When `interior_struct.zalmoxis.global_miscibility` is on, Zalmoxis handles the miscibility in its structure solve, and the outgassing step does not apply the bulk binodal.

**Where this is checked.** [H2-MgSiO3 binodal](../Validation/outgas/binodal.md).

## Rock vapour from the melt surface

**Question.** A very hot magma ocean also evaporates rock. How does that vapour enter the atmosphere?

With `outgas.vapourise = true`, `run_vapourisation` calls LavAtmos [^cite-vanbuchem2023] after the volatile solve. LavAtmos computes the vapour that is in equilibrium with the melt surface, and the FastChem chemistry code recombines it with the volatile gas. The vapour adds species such as SiO, Na, Mg and Fe to the atmosphere.

The inputs are:

- the surface temperature, $\max(T_\mathrm{min}, T_\mathrm{magma})$ \[K\], with $T_\mathrm{min}$ = `outgas.lavatmos.T_min`;
- the volatile surface pressure from the chemistry solve \[bar\];
- the element number fractions of the volatile atmosphere;
- the melt composition file `outgas.lavatmos.melt_comp_name`.

To turn masses into number fractions, the code needs the molar mass of each species. `_fastchem_weight` sums the atomic masses of the formula and converts kg mol$^{-1}$ to the g mol$^{-1}$ that FastChem and LavAtmos use. For SiO$_2$ this gives 60.083 g mol$^{-1}$. The electron, which has no formula, takes its own molar mass.

The vapour step is skipped once the mantle is crystallised. Rock vapour enters the atmosphere without being taken from the mantle, so whole-planet mass is not conserved on this path; see [Model description](model.md#whole-planet-mass-is-not-conserved-when-vapourisation-is-enabled).

**Where this is checked.** [Rock-vapour molar masses](../Validation/outgas/lavatmos.md).

## Volatile trapping in the freezing mantle

**Question.** When the magma ocean freezes, how much of its dissolved water and gas stays locked in the solid mantle instead of reaching the atmosphere?

**Short answer.** Each kilogram of new solid takes 2 things with it: a small amount of volatile inside the crystals, and the melt that stays in the pores between the crystals. The second part usually dominates. With `outgas.trap_mode = "front"`, the code computes the pore melt from how fast melt can drain out of the freezing layer. With `"none"` (the default), nothing is trapped.

### The burial rule

Over one step, the mantle mass $\Delta M$ \[kg\] freezes. The code computes it from the drop of the melt fraction, $\Delta M = M_\mathrm{mantle}\,(\Phi_{t-1} - \Phi_t)$. Each volatile species $Z$ then loses this mass from the melt to the solid [^cite-sim2024]:

$$
\Delta m_Z = \left[(1 - F_\mathrm{tl})\, D_Z + F_\mathrm{tl}\right] C_Z\, \Delta M.
$$

The symbols are:

- $C_Z$: the mass fraction of $Z$ in the melt \[1\], from the last chemistry solve;
- $D_Z$: the crystal/melt partition coefficient \[1\], the ratio of the mass fraction in the crystals to that in the melt, `outgas.D_const_<species>`;
- $F_\mathrm{tl}$: the trapped melt fraction \[1\], the mass fraction of the new solid that is pore melt.

The bracket is the effective partition coefficient. If $F_\mathrm{tl} = 0$, it is $D_Z$: only the crystals take up the volatile. If $F_\mathrm{tl} = 1$, it is 1: the whole melt is buried. The default $D_Z$ is 0.0017 for H$_2$O and 0 for every other species. A species with $D_Z = 0$, such as CO$_2$ or a noble gas, is buried only with the pore melt.

**Example.** Water at $C_Z = 10^{-3}$ (1000 ppm), with $D_Z = 0.0017$ and $F_\mathrm{tl} = 0.02$, as $2\times10^{23}$ kg of mantle freezes. The bracket is 0.021666, so $4.33\times10^{18}$ kg of water is buried. With $F_\mathrm{tl} = 0$ only $3.4\times10^{17}$ kg would be buried. At $F_\mathrm{tl} = 0.02$ the pore melt buries $4.0\times10^{18}$ kg, 12 times the $3.3\times10^{17}$ kg that the crystals take up.

The buried mass moves from `{sp}_kg_liquid` to `{sp}_kg_solid` and is recorded in `{sp}_kg_trapped`. The element totals do not change. Buried mass cannot escape or outgas: the escape step and the desiccation check subtract it from the mass they can reach.

### The trapped melt fraction

`outgas.trap_mode` selects how $F_\mathrm{tl}$ is found:

- `"none"`: no trapping. This is the default.
- `"front"`: $F_\mathrm{tl}$ comes from the drainage of melt through the freezing front, which the next section explains. This mode needs `interior_energetics.module = "aragog"`, because only Aragog writes the profiles that the front needs.

If the interior profiles are missing on a step, that step buries with $D_Z$ alone ($F_\mathrm{tl} = 0$) and logs a warning.

**Where this is checked.** [Solid-phase volatile trapping](../Validation/outgas/trapping.md).

## Melt drainage across the freezing front

**Question.** How much melt is still in the pores when a layer of crystals has frozen solid?

**Short answer.** The **freezing front** is the layer of the mantle that is partly solid but still porous. Melt leaves it only if 2 things happen: the melt flows out through the pores, and the crystal matrix deforms to close them. The slower of the 2 sets the drainage rate. The melt that is left when a parcel reaches the bottom of the front is $F_\mathrm{tl}$.

### Where the front is

`locate_front` (in `proteus.outgas.compaction`) finds the front on the interior mesh:

- **Top**: where the melt fraction of the interior solver falls below $\phi_c$ = `interior_energetics.rfront_loc` (default 0.4). This is the melt fraction at which the solver switches the mantle from melt-like to solid-like flow, so trapping and the interior rheology use the same transition.
- **Base**: where the porosity falls below `outgas.trap_phi_min` (default 0.01), or the core-mantle boundary if the lowest node is still porous.

The porosity $\phi$ is the volume fraction of melt \[1\]. The code gets it from the densities of the mixture, solid and melt, $\phi = (\rho_s - \rho)/(\rho_s - \rho_l)$.

### The 2 timescales

The **percolation time** $\tau_D$ is how long melt takes to flow through the front, of thickness $L$ \[m\]:

$$
\tau_D = \frac{L}{w_D}, \qquad w_D = \frac{|\Delta\rho|\, g\, F(\phi)}{\eta_\mathrm{melt}},
$$

where $w_D$ is the percolation speed \[m s$^{-1}$\], $\Delta\rho$ is the solid minus melt density \[kg m$^{-3}$\], $g$ is the gravity \[m s$^{-2}$\], $\eta_\mathrm{melt}$ is the melt viscosity \[Pa s\] and $F(\phi)$ \[m$^2$\] is the permeability divided by the porosity. $F$ is the same law that Aragog uses for melt-solid separation [^cite-bower2018]. It depends on the grain size, `interior_energetics.grain_size`.

The **matrix time** $\tau_s$ is how long the crystal matrix takes to deform and close the pores:

$$
\tau_s = \frac{\mu_s}{|\Delta\rho|\, g\, L},
$$

where $\mu_s$ is the mush viscosity \[Pa s\], $10^{20}$ Pa s by default (`outgas.trap_mush_log10visc`).

### The drainage equation

A parcel enters the front at the porosity of the front top and moves through it in the **residence time** $t_\mathrm{res} = L / v_f$, where $v_f$ is the front speed \[m s$^{-1}$\] from the mass that froze in the step. Along the way, its porosity falls as

$$
\frac{d\phi}{dt} = -\frac{\phi}{\max(\tau_D(\phi),\ \tau_s)}.
$$

`drainage_integral` integrates this from 0 to $t_\mathrm{res}$. The porosity at the end is the trapped melt as a volume fraction. `volume_to_mass_fraction` converts it to the mass fraction $F_\mathrm{tl}$ that the burial rule uses.

**Example.** A front 60 km thick, with $\Delta\rho = 330$ kg m$^{-3}$ and $g = 3.711$ m s$^{-2}$ (a Mars-like case), entered at $\phi = 0.3$, with 1 mm grains, a melt viscosity of 100 Pa s and the default mush viscosity:

| Front speed | $t_\mathrm{res}$ | $\tau_D$ at entry | $\tau_s$ | Trapped melt volume fraction |
|---|---|---|---|---|
| 70 cm yr$^{-1}$ | $8.6\times10^4$ yr | $7.0\times10^4$ yr | $4.3\times10^4$ yr | 0.194 |
| 5 cm yr$^{-1}$ | $1.2\times10^6$ yr | $7.0\times10^4$ yr | $4.3\times10^4$ yr | 0.109 |

A slower front gives the melt more time to drain, so it traps less. Here the percolation time is the longer one, so percolation limits the drainage.

### When the front cannot be integrated

The code does not integrate a front that the mesh cannot resolve. It then uses an upper bound: the porosity at the top of the front, converted to a mass fraction and capped at $\phi_c$. This happens when:

- the front spans fewer than `outgas.trap_n_front_min` nodes (default 3);
- the front is thicker than `outgas.trap_max_front_fraction` of the mantle (default 1, no limit);
- the porous layer is split into separate parts;
- the front reaches the surface node;
- the front has no positive thickness;
- the melt is denser than the solid somewhere in the front;
- the front speed is zero.

Each such step logs a warning with the cause. The column `trap_branch` records the path of every step: 0 no front, 1 percolation-limited, 2 matrix-limited, 3 upper bound, 6 no interior profiles, 7 remelting.

**Where this is checked.** [Melt drainage across the freezing front](../Validation/outgas/compaction.md).

## Remelting and the burial ledger

**Question.** If the mantle melts again, which buried volatiles come back to the melt?

**Short answer.** The solid that froze last is the first to melt. So a remelt returns the mass that was buried most recently, layer by layer.

The **burial ledger** (`proteus.outgas.burial_ledger`) stores each step's buried mass as a layer between 2 melt fractions: the melt fraction before and after the step. When $\Phi$ rises to a new value, `remelt_to` returns the mass of every layer below that value. A layer that is only partly remelted returns the remelted share of its mass.

**Example.** The mantle freezes from $\Phi = 1$ to 0.7 and buries 3 kg, then from 0.7 to 0.5 and buries 2 kg. It then remelts to $\Phi = 0.6$. That is half of the second layer, so 1 kg returns to the melt. The first layer and the other half of the second layer stay buried.

The code rebuilds the ledger from the output columns `Phi_global` and `{sp}_kg_trapped` on each remelt, so a resumed run has the same ledger. If the history does not explain all the trapped mass, the rest is returned in proportion to the share of the solid that remelted, $(\Phi_t - \Phi_{t-1})/(1 - \Phi_{t-1})$.

**Where this is checked.** [Burial ledger for remelting](../Validation/outgas/burial_ledger.md).

## Limits of the trapping inputs

Each limit below is checked when the config loads, and a value outside it stops the load with an error:

- `outgas.trap_mode`: `"none"` or `"front"`; `"front"` needs `interior_energetics.module = "aragog"`.
- `outgas.trap_phi_min`: between 0 and 1, both excluded.
- `outgas.trap_n_front_min`: at least 2.
- `outgas.trap_max_front_fraction`: above 0 and at most 1.
- `outgas.D_const_<species>`: 0 or more.
- `outgas.trap_mush_log10visc`: any number; zero or below selects $10^{20}$ Pa s.

At run time, $F_\mathrm{tl}$ outside \[0, 1\] or a negative $D_Z$ raises an error. The buried mass of a species is capped at the mass that the melt holds; the code then records the species in the step's capped list.

[^cite-rogers2025]: Rogers, J.G., Young, E.D. and Schlichting, H.E., *[Redefining interiors and envelopes: hydrogen-silicate miscibility and its consequences for the structure and evolution of sub-Neptunes](https://doi.org/10.1093/mnras/staf1940)*, Monthly Notices of the Royal Astronomical Society, 544, 3496, 2025.
[^cite-sim2024]: Sim, S.J., Hirschmann, M.M. and Hier-Majumder, S., *[Volatile and trace element storage in a crystallizing martian magma ocean](https://doi.org/10.1029/2024JE008346)*, Journal of Geophysical Research: Planets, 129, e2024JE008346, 2024. Eq. 6.
[^cite-bower2018]: Bower, D.J., Sanan, P. and Wolf, A.S., *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49, 2018.
[^cite-vanbuchem2023]: van Buchem, C.P.A., Miguel, Y., Zilinskas, M. and van Westrenen, W., *[LavAtmos: An open-source chemical equilibrium vaporization code for lava worlds](https://doi.org/10.1111/maps.13994)*, Meteoritics & Planetary Science, 58, 1149-1161, 2023.
