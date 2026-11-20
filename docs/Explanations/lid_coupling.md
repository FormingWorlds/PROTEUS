# Sub-grid lid and the surface skin

How does a thin solid crust on the magma ocean change the heat that the interior loses to the atmosphere?

**Short answer.** Aragog carries the crust as a sub-grid lid with a weight $w$ from 0 (no lid) to 1 (a full lid). The surface then has 2 parts at one surface temperature: the lid part conducts through the crust, and the open part conducts through a thin skin over the melt. PROTEUS gives the atmosphere both parts as one skin. It gives the interior the atmosphere flux and the terms that split it between the 2 parts, so that 3 energy rules between the melt, the lid and the atmosphere hold. Without these rules the interior would lose more or less heat than the atmosphere takes.

This page assumes you know the [coupling loop](coupling_loop.md): the interior runs first in each iteration with the flux `F_atm` of the last atmosphere call, and the atmosphere then solves for the surface temperature `T_surf`. The lid itself, its profile, its growth and its hand-over to the mesh, is described on the [Surface lid page](https://proteus-framework.org/aragog/Explanations/surface_lid.html) of the Aragog documentation. This page covers the PROTEUS side: what the atmosphere sees and what the interior takes. It applies to `interior_energetics.module = "aragog"` with the default `interior_energetics.surface_bc_mode = "flux"`. The lid seal of the melt volatiles reads the thickness of this lid; see [Volatiles held under a closed crust](lid_seal.md). The code paths, the guard codes, the resume fields and the run numbers behind this page are on [Lid coupling details](../Reference/lid_coupling.md).

## The skin and the lid

**The skin.** With the default `atmos_clim.surf_state = "skin"`, a thin conductive layer sits between the melt and the surface. It stands for the boundary layer at the top of a molten surface. Its thickness $d$ is `atmos_clim.surface_d` (default 0.01 m) and its conductivity $k$ is `atmos_clim.surface_k` (default 2 W m$^{-1}$ K$^{-1}$). Each iteration writes them into the helpfile columns `boundary_layer_thickness` and `boundary_layer_conductivity`. The skin conductance is

$$G_\mathrm{skin} = \frac{k}{d},$$

200 W m$^{-2}$ K$^{-1}$ for the defaults. The atmosphere module (AGNI or the dummy) finds the surface temperature $T_\mathrm{surf}$ at which the heat conducted through the skin equals the net flux the atmosphere carries away:

$$F_\mathrm{atm}(T_\mathrm{surf}) = G_\mathrm{skin}\,(T_\mathrm{magma} - T_\mathrm{surf}).$$

It returns `F_atm`, in W m$^{-2}$ per total area; in AGNI this is the total flux at the top level. Without a lid the interior takes `F_atm` as its surface flux for the next step.

**The lid.** A crust that forms at the surface starts metres thick or less, while the top cell of the interior mesh is kilometres thick. The mesh cannot resolve it, so Aragog carries it as a sub-grid lid: a conductive solid layer of thickness $\delta$ at the top of the mantle, with its base $T_b$ at the solidus at the pressure of the top-cell centre. Aragog holds $\delta$ and the lid energy per area $E_\mathrm{lid}$ as two extra solver states. A weight $w$ from 0 to 1 says how strongly the lid acts: $w = 0$ is no lid, $w = 1$ is a full lid, and values between blend the lid with the open surface. A lid is live where its weight and its conductance are both above 0. The lid conducts with

$$G_\mathrm{lid} = \frac{k_\mathrm{lid}\, f'(0)}{\delta},$$

the helpfile column `lid_conductance`. Here $k_\mathrm{lid}$ is `interior_energetics.solid_cond` and $f'(0)$ is the slope of the temperature profile of the lid at its surface, 1 for a straight profile and at least 0.1 in this conductance.

Set `interior_energetics.aragog.lid_mush_base = true` to run the lid. PROTEUS then passes `subgrid_lid` and `lid_mush_base` to Aragog. The Aragog lid also needs the solid rheology, `interior_energetics.aragog.rheology.enabled = true`.

[![Schematic of the top of the column with the sub-grid lid: the atmosphere above one surface temperature, a lid part of weight w with a solid lid over melt, an open part of weight 1 - w with a thin skin over melt, and the temperature with depth in both parts](../assets/lid_coupling_schematic.svg#only-light){ width="100%" }](../assets/lid_coupling_schematic.svg)
[![Schematic of the top of the column with the sub-grid lid: the atmosphere above one surface temperature, a lid part of weight w with a solid lid over melt, an open part of weight 1 - w with a thin skin over melt, and the temperature with depth in both parts](../assets/lid_coupling_schematic_dark.svg#only-dark){ width="100%" }](../assets/lid_coupling_schematic_dark.svg)

**How to read the figure.** Left: the column top, with one surface temperature $T_\mathrm{surf}$ for both parts. The lid part, weight $w$, conducts from its base $T_b$; the top-cell melt under it is at $T_b$ or warmer. The open part, weight $1 - w$, conducts through the skin from the melt at $T_m = \max(T_\mathrm{node}, T_b)$, with $T_\mathrm{node}$ the top basic node of the Aragog mesh ([What the atmosphere call sees](#what-the-atmosphere-call-sees)).

The arrow in the atmosphere is the flux $F_\mathrm{atm}$ of the atmosphere call. The arrow in the lid is the flux $F_\mathrm{surf}$ that leaves the lid surface; [Rule 3](#rule-3-the-atmosphere-flux-is-the-largest-loss) caps it. Under the lid, $F_\mathrm{melt}$ is the heat that the melt passes to the lid ([Rule 2](#rule-2-the-lid-ledger)). In the open part, $F_\mathrm{open}$ is its flux per open area; per total area it is $A_\mathrm{open}$ ([The open part within a step](#the-open-part-within-a-step)).

Right: temperature $T$ against depth $z$ in each part; dotted lines: the melt of the top cell. The lid profile uses the shape parameter $s = 1$ of the Aragog lid ([The onset thickness](#the-onset-thickness)), for illustration.

The lid thickness (centimetres to decimetres) and the top cell (65.8 km) are values of the coupled runs of this page. Thicknesses are not to scale.

## Three energy rules

The lid must not create or lose energy at the surface. Three rules state this.

### Rule 1: a molten surface

**Rule.** Without a lid the atmosphere sets the heat loss of the interior, exactly as in the default coupling:

$$F_\mathrm{int} = F_\mathrm{atm}\ \text{of the last atmosphere call}.$$

**How the code keeps it.** Every lid branch acts only for $w > 0$. With $w = 0$ the atmosphere call, the outer flux of Aragog and the booked `F_int` are those of a run without the lid ([Lid coupling details](../Reference/lid_coupling.md#code-that-keeps-rule-1) lists the functions).

The lid switches on without a step. The lid holds its surface at the `T_surf` of the last atmosphere call. Its weight $w = 1 - g\,(1 - w_\mathrm{s})$ grows from 0 as the held surface falls below $T_b$ (the existence weight $w_\mathrm{s}$) or as the top cell cools to its floor near the solidus (the floor weight $g$ falls from 1 to 0; [The onset thickness](#the-onset-thickness)). In the stored coupled runs of the Validation page the lid first acts through $g$: the surface lies 0.005 and 0.14 K above $T_b$ and the top node 10.6 and 5.8 K below it. In the step in which the lid switches on, it acts at the end of the step, so `F_int` is the step integral of Rule 2.

Each atmosphere call uses the lid weight of the last interior row. Near the onset this map from one row to the next can have more than one balance; [Lid coupling details](../Reference/lid_coupling.md#several-balances-at-the-onset) describes it with the first lid rows of 2 AGNI runs.

### Rule 2: the lid ledger

**Rule.** Under a lid, the melt below can lose less heat than the surface only by the heat the lid releases, and more only by the heat the lid stores. Per unit area,

$$F_\mathrm{melt} = F_\mathrm{surf} + \frac{dE_\mathrm{lid}}{dt},$$

with $F_\mathrm{melt}$ the heat the melt passes to the lid, $F_\mathrm{surf}$ the heat that leaves the lid surface, and $E_\mathrm{lid}$ in J m$^{-2}$.

**How the code keeps it.** Aragog holds $E_\mathrm{lid}$ as a solver state. Its energy budget counts the melt and the lid together, so the energy of the mantle changes only by the heat that crosses the surface and the core boundary. PROTEUS books what crossed the surface. With a lid at the start or at the end of the step, `F_int` is the mean surface flux of the step:

$$F_\mathrm{int} = -\frac{\Delta E_\mathrm{surf}}{A\,\Delta t},$$

with $\Delta E_\mathrm{surf}$ the surface energy integral of the step from Aragog (`step_dE_F_int_J`, negative for a loss), $A = 4\pi r^2$ at the top of the mesh and $\Delta t$ the step length. Inside the step the lid follows its own surface flux, so `F_int` can differ from `F_atm`. The column `lid_F_diff` records `F_int - F_atm`. A step in which the lid hands over to the mesh books this integral too; the coupling step does not end at the hand-over. With the mush base the lid hands over at 0.7 of the top-cell width.

### Rule 3: the atmosphere flux is the largest loss

**Rule.** The lid surface loses what the atmosphere takes from the lid part at the surface. In the next interior step, where the lid held at $T_\mathrm{surf}$ would conduct more than the cap

$$C = G_\mathrm{lid}'\,(T_b - T_\mathrm{surf}), \qquad G_\mathrm{lid}' = \min(G_\mathrm{lid}, G_\mathrm{skin}),$$

its surface warms to $T_s$ and carries $F_s = C + F' (T_s - T_\mathrm{surf})$. Here $F'$, in W m$^{-2}$ K$^{-1}$, is the slope of `F_atm` with the surface temperature, so $F_s$ is the flux of the atmosphere at the warmer surface to first order.

$C$ is the lid term of the last atmosphere balance (see the next section). At $w = 1$ that balance has only its lid term, so $C$ equals `F_atm` to the accuracy of the atmosphere solve.

**How the code keeps it.** The atmosphere call limits $G_\mathrm{lid}$ to $G_\mathrm{skin}$. PROTEUS passes $C$ and $F'$ to Aragog (`lid_surface_flux_cap`, `lid_surface_flux_slope`), and re-solves a step once where the lid surface ends above the held one. [Code that keeps Rule 3](../Reference/lid_coupling.md#code-that-keeps-rule-3) lists the functions.

The cap is close to the flux of the held lid at the start of the step, but not equal to it. [Lid coupling details](../Reference/lid_coupling.md#code-that-keeps-rule-3) says when it acts and gives numbers from dummy runs.

## What the atmosphere call sees

Under a lid the surface has two parts that share one surface temperature. The lid part, weight $w$, conducts from the lid base $T_b$ (column `lid_T_base`). The open part, weight $1 - w$, conducts through the skin from the melt under it. That melt is held at least at the lid base,

$$T_m = \max(T_\mathrm{node}, T_b),$$

with $T_\mathrm{node}$ the top basic node of the Aragog mesh (column `lid_T_node`). The floor is a model construct of the coupling, not a result from the literature: the open part is melt, so it cannot sit below the solidus at the lid base, while the extrapolated top node of the mesh can (see the floor in [Limits](#limits)). The two parts act in parallel:

$$F = w\,G_\mathrm{lid}'\,(T_b - T_\mathrm{surf}) + (1 - w)\,G_\mathrm{skin}\,(T_m - T_\mathrm{surf}).$$

AGNI and the dummy solve against one skin, so `lid_boundary` writes this sum in the form of one skin. With the blended conductance

$$G = w\,G_\mathrm{lid}' + (1 - w)\,G_\mathrm{skin},$$

the call takes the skin thickness $d_\mathrm{call} = k / G$ (with the same $k$) and the melt temperature

$$T_\mathrm{magma,call} = \frac{w\,G_\mathrm{lid}'\,T_b + (1 - w)\,G_\mathrm{skin}\,T_m}{G}.$$

Then $\frac{k}{d_\mathrm{call}}\,(T_\mathrm{magma,call} - T_\mathrm{surf})$ equals $F$ above for every $T_\mathrm{surf}$, so the atmosphere solves the parallel balance with the skin balance it already has. These values hold for the call only: `lid_boundary` writes nothing to the row. The row keeps the `T_magma` that Aragog writes under a lid, $w T_b + (1 - w) T_m$, the melt temperature that outgassing reads.

**Why the 10 mm skin is not added on top of the lid.** The skin stands for the boundary layer at the top of a molten surface. Under a solid lid no melt reaches the surface, so that boundary layer does not exist: the lid is the conductive layer, and $T_\mathrm{surf}$ is the temperature at its top. Adding the skin in series, $1/G = 1/G_\mathrm{lid} + 1/G_\mathrm{skin}$, would count one layer twice; at $G_\mathrm{lid} = G_\mathrm{skin}$ it would halve the conductance of the lid part.

**Where this is checked.** [The atmosphere call sees the lid and the open surface in parallel](../Validation/atmos_clim/common.md#the-atmosphere-call-sees-the-lid-and-the-open-surface-in-parallel) checks the parallel sum of the call against its closed form.

## What the interior takes

What does the interior take from the atmosphere step under a lid, and why does it not count the lid twice?

**Short answer.** The interior takes `F_atm`, per total area, as its outer boundary value. Under a live lid it also takes 4 terms of the atmosphere call balance. Aragog uses them to split `F_atm` between the lid part and the open part at its own lid weight.

Under a live lid in flux mode with the mush base, `lid_open_fields` (in `atmos_clim/common.py`) builds the 4 terms, and `AragogRunner.update_solver` sets them on the boundary of the Aragog solver:

- `open_lid_flux`, $L = G_\mathrm{lid}'\,(T_b - T_\mathrm{surf})$ in W m$^{-2}$: the flux of the lid part in the call, with its sign, inward for a surface above $T_b$;
- `open_surface_slope`, $F'$ in W m$^{-2}$ K$^{-1}$, after [the slope guard](#the-open-part-within-a-step);
- `open_skin_conductance`, $G_\mathrm{skin}$ in W m$^{-2}$ K$^{-1}$;
- `open_lid_conductance`, $G_\mathrm{lid}'$ in W m$^{-2}$ K$^{-1}$.

Otherwise the 4 terms are unset, and Aragog takes `F_atm` as the flux of the whole surface.

**Why the lid is not counted twice.** `F_atm` already holds the lid term $w L$ of the call. The open part therefore starts from $F_\mathrm{atm} - w L$ per total area, and the lid part adds its own loss. An open part at `F_atm` would count the lid term a second time.

**Inside the step.** Aragog sets the lid weight from its own state at each evaluation, and the open part takes that weight. The lid part loses its own conduction, not $L$. Aragog holds the lid surface at $T_\mathrm{surf}$ and at least 1 K below $T_b$, so near $T_b$ the lid part loses at least $G_\mathrm{lid}$ times 1 K. Where the held lid would conduct more than $C$, its surface carries $F_s$ of [Rule 3](#rule-3-the-atmosphere-flux-is-the-largest-loss). What the step loses is `F_int`, and `lid_F_diff` records `F_int - F_atm` ([Rule 2](#rule-2-the-lid-ledger)).

**A diagnostic column.** `lid_F_open` is the start value of the open part per open area; Aragog does not read it. [The open-part diagnostic](../Reference/lid_coupling.md#the-open-part-diagnostic) gives its formula and limits.

**Where this is checked.** [The two parts of the surface sum to the atmosphere flux](../Validation/atmos_clim/common.md#the-two-parts-of-the-surface-sum-to-the-atmosphere-flux) checks the sum of the two parts at the weight of the row against its closed form.

## The open part within a step

How does the flux of the open part change while the interior step runs?

**Short answer.** The open part starts from what is left of `F_atm` after the lid part, and then follows its melt. When the melt under it warms, its loss rises by the extra flux that the atmosphere would take at the warmer surface of the call balance. A fixed start flux would let a warming melt lose no more heat.

At the lid weight $w$ of each evaluation, the open part loses, per total area,

$$A_\mathrm{open} = F_\mathrm{atm} - w\,L + (1 - w)\,G_\mathrm{eff}(w)\,(T_m - T_{m,0}),$$

with $T_{m,0}$ the value of $T_m$ at the start of the step. The response per open area, in W m$^{-2}$ K$^{-1}$, is

$$G_\mathrm{eff}(w) = \frac{G_\mathrm{skin}\,F'}{F' + w\,G_\mathrm{lid}' + (1 - w)\,G_\mathrm{skin}}.$$

The surface loses $w F_s + A_\mathrm{open}$. With $F_s = L$ and no rise of $T_m$ that is exactly `F_atm`. The form has no division by $1 - w$, so $A_\mathrm{open}$ stays finite at $w = 1$.

**Where $G_\mathrm{eff}$ comes from.** A rise $dT_m$ of the open melt warms the surface of the call balance by $dT_\mathrm{surf} = (1 - w)\,G_\mathrm{skin}\,dT_m / (F' + w\,G_\mathrm{lid}' + (1 - w)\,G_\mathrm{skin})$. The atmosphere then takes $F'\,dT_\mathrm{surf}$ more. Aragog holds the lid surface, so the open part alone carries that extra flux: $(1 - w)\,G_\mathrm{eff}(w)\,dT_m = F'\,dT_\mathrm{surf}$. At $w = 0$ this is the skin and the atmosphere in series. The [Aragog page](https://proteus-framework.org/aragog/Explanations/surface_lid.html#the-open-part-within-a-call) describes the same term from the solver side.

**The slope guard.** $F'$ is the secant of `F_atm` over `T_surf` of the last 2 rows. With $F' = 0$ the held lid surface would carry a fixed flux, with no feedback of the atmosphere on the crust. Where that secant is unusable, a live lid takes the newest usable earlier secant, and where the run has none, the grey tangent $4 \sigma T_\mathrm{surf}^3$, with $\sigma$ the Stefan-Boltzmann constant. The open part also cuts $F'$ at the slope of a bare grey surface of emissivity 1 over the same 2 rows. The column `lid_slope_guard` records which case acted; [Slope guard codes](../Reference/lid_coupling.md#slope-guard-codes) gives the rules, the thresholds and the log line.

**Where this is checked.** [The open part follows the melt within the interior step](../Validation/atmos_clim/common.md#the-open-part-follows-the-melt-within-the-interior-step) checks $G_\mathrm{eff}$ against the derivative of the surface balance and gives it in numbers for a stored lid-onset row, and [Coupled runs through the lid onset](../Validation/atmos_clim/common.md#coupled-runs-through-the-lid-onset) the surface temperature of 2 stored run segments.

## The onset thickness

Aragog switches the lid on through its weight $w = 1 - g\,(1 - w_\mathrm{s})$, with $g$ the floor weight of the top cell and $w_\mathrm{s}$ the existence weight of the steady crust under the held surface, the `T_surf` of the last atmosphere call. $w_\mathrm{s}$ compares the thickness $\delta_\mathrm{eq} = k_\mathrm{lid} (T_b - T_\mathrm{surf}) / F_b$ with the onset thickness $\delta_\mathrm{on}$, with $F_b$ the base flux. $w_\mathrm{s}$ is 0 while the held surface lies at or above $T_b$. For $F_b > 0$ it rises without a step as the surface falls below $T_b$, and it is 0.5 where $\delta_\mathrm{eq} = \delta_\mathrm{on}$; for $F_b \le 0$ it is 1. A weight below $10^{-10}$ counts as 0, so the first lid row comes some K below $T_b$. PROTEUS derives $\delta_\mathrm{on}$ from the skin:

$$\delta_\mathrm{on} = \frac{k_\mathrm{lid}\, d}{k}.$$

For the defaults ($k_\mathrm{lid} = 4$ W m$^{-1}$ K$^{-1}$, $d = 0.01$ m, $k = 2$ W m$^{-1}$ K$^{-1}$) this gives $\delta_\mathrm{on} = 0.02$ m. At this thickness a straight lid has $k_\mathrm{lid}/\delta_\mathrm{on} = k/d = G_\mathrm{skin}$, so it conducts like the skin it replaces. The lid enters, however, at the phantom thickness $\delta_\mathrm{lo} = \delta_\mathrm{on}/2$ (see [Resume](#resume)), so its conductance at the onset is $2 f'(0)\,G_\mathrm{skin}$. A growing lid has $f'(0) \ge 1$, so this is at least twice $G_\mathrm{skin}$ ([Lid coupling details](../Reference/lid_coupling.md#the-lid-conductance-at-the-onset)). The atmosphere call caps $G_\mathrm{lid}$ at $G_\mathrm{skin}$ (Rule 3), so it does not see that step while $G_\mathrm{lid} > G_\mathrm{skin}$.

**Setting it.** `interior_energetics.aragog.lid_delta_on` \[m\] defaults to the skin value above. A finite number above 0 is used as given; any other value is refused at config load. [Setting the onset thickness](../Reference/lid_coupling.md#setting-the-onset-thickness) gives the warning band, the log line and the column.

## Resume

Each interior snapshot of a run with the lid stores the lid thickness, the lid energy and the time of a hand-over. On `proteus start -r`, PROTEUS restores the lid of the snapshot first and then its entropy profile. A stored lid that is absent or not finite restarts, with a warning, as the phantom lid of Aragog at `lid_delta_lo`: by the Aragog default half of `lid_delta_on`. A stored lid that Aragog refuses stops the resume with an error. [Resume of a run with the lid](../Reference/lid_coupling.md#resume-of-a-run-with-the-lid) gives the snapshot fields and every case.

A resume that changes `lid_mush_base` from the value in the run's `init_coupler.toml` stops with status 20, unless `interior_energetics.aragog.allow_resume_law_change` is true; see [Resume stops on a separation law](../How-to/troubleshooting.md#resume-stops-on-a-separation-law).

## Helpfile columns

The interior writes every column except `lid_F_open` and `lid_F_resid` at each step; the atmosphere wrapper writes those 2. With the lid off the columns read 0, except `lid_T_node` (the top basic node of the Aragog mesh) and `lid_F_open` (`F_atm`). With the lid on and $w = 0$, `lid_delta` holds the thin phantom lid, so read it with `lid_weight`. The columns `lid_weight`, `lid_conductance`, `lid_delta`, `lid_delta_on`, `lid_T_base` and `lid_T_node` hold $w$, $G_\mathrm{lid}$, $\delta$, $\delta_\mathrm{on}$, $T_b$ and $T_\mathrm{node}$. [Output format](../Reference/output.md) gives the unit and meaning of every lid column.

## Limits

- **Refused settings and stops.** JANUS does not read the lid. The lid is refused at load with JANUS, with an `atmos_clim.surf_state` other than `"skin"`, with the dummy at `atmos_clim.dummy.fixed_flux > 0`, with `planet.prevent_warming = true` and with `surface_d = "saturation"`. A clip of `T_magma` in a step where a lid acts stops the run, and so does a live lid with a zero skin. [Settings the lid refuses](../Reference/lid_coupling.md#settings-the-lid-refuses) gives the reasons.
- **One column.** The lid and the open part share one surface temperature in a 1-D column, blended by $w$. The model has no map of where on the planet a crust lies.
- **The cap.** Rule 3 caps the flux of the lid part only. The open part takes `F_atm` less the lid term and follows its melt with $G_\mathrm{eff}(w)$, a first-order response with the slope of the rows.
- **The floor.** $T_m = \max(T_\mathrm{node}, T_b)$ is a model construct of the coupling. The mesh keeps its own top node below $T_b$; only the call, the open flux and the row `T_magma` read the floored value. With `lid_mush_base` the row `T_magma` is floored at every lid weight, $w = 0$ included. It can lie above the `T_magma` of a run without the lid by up to $T_b - T_\mathrm{node}$: 5.8 and 10.6 K on the 2 stored live-lid rows at lid weights 0.97 and 0.70.
- **The slope with AGNI.** With AGNI the cap takes $F'$ from the rows alone, and `_correct_lid_cap` does not act. Where `F_atm` levels off with the surface temperature within one step, a slope from the rows before lies above the slope across the step. The lid part can then lose more than the atmosphere at the end surface. This case is tested only on the dummy flux with a soft ceiling; [Lid coupling details](../Reference/lid_coupling.md#the-slope-with-agni) gives the size of an AGNI secant error.
- **Lid onset within a step.** Each step takes the atmosphere of its start, while the crust grows within it, so the interior lags the atmosphere. In a 1 Earth-mass run with the open part and the slope guard of this page, `F_int` lay below the smaller `F_atm` at the 2 ends of its step in 7 steps, by 5 to 13 % of the `F_atm` at its start. From 50890 to 51300 yr, the energy outside that band was 5.8 % of the energy the atmosphere took at the normal step of about 35 yr, and 3.0 % at half that step. The gap scales with the step. [Lid onset within a step](../Reference/lid_coupling.md#lid-onset-within-a-step) gives the steps.

For the config keys, see [Aragog](../Reference/config/interior.md#aragog-interior_energeticsaragog) and [Atmosphere climate](../Reference/config/atmosphere.md#atmosphere-climate-atmos_clim).
