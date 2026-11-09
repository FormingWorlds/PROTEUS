# Sub-grid lid and the surface skin

A thin solid crust at the top of the Aragog mantle changes the heat path from the melt to the atmosphere. The lid enters the surface balance of the atmosphere, and the coupling keeps three energy rules between the melt, the lid and the atmosphere.

This page assumes you know the [coupling loop](coupling_loop.md): the interior runs first in each iteration with the flux `F_atm` of the last atmosphere call, and the atmosphere then solves for the surface temperature `T_surf`. The lid itself, its profile, its growth and its hand-over to the mesh, is described on the [Surface lid page](https://proteus-framework.org/aragog/Explanations/surface_lid.html) of the Aragog documentation. This page covers the PROTEUS side: what the atmosphere sees and what the interior takes. It applies to `interior_energetics.module = "aragog"` with the default `interior_energetics.surface_bc_mode = "flux"`.

## The skin and the lid

**The skin.** With the default `atmos_clim.surf_state = "skin"`, a thin conductive layer sits between the melt and the surface. It stands for the boundary layer at the top of a molten surface. Its thickness $d$ is `atmos_clim.surface_d` (default 0.01 m) and its conductivity $k$ is `atmos_clim.surface_k` (default 2 W m$^{-1}$ K$^{-1}$). Each iteration writes them into the helpfile columns `boundary_layer_thickness` and `boundary_layer_conductivity`. The skin conductance is

$$G_\mathrm{skin} = \frac{k}{d},$$

200 W m$^{-2}$ K$^{-1}$ for the defaults. The atmosphere module (AGNI or the dummy) finds the surface temperature $T_\mathrm{surf}$ at which the heat conducted through the skin equals the net flux the atmosphere carries away:

$$F_\mathrm{atm}(T_\mathrm{surf}) = G_\mathrm{skin}\,(T_\mathrm{magma} - T_\mathrm{surf}).$$

It returns `F_atm`; in AGNI this is the total flux at the top level. Without a lid the interior takes `F_atm` as its surface flux for the next step.

![Schematic of the top of the column with the sub-grid lid: the atmosphere above one surface temperature, a lid part of weight w with a solid lid over melt, an open part of weight 1 - w with a thin skin over melt, and the temperature with depth in both parts](../assets/lid_coupling_schematic.svg#only-light){ width="100%" }
![Schematic of the top of the column with the sub-grid lid: the atmosphere above one surface temperature, a lid part of weight w with a solid lid over melt, an open part of weight 1 - w with a thin skin over melt, and the temperature with depth in both parts](../assets/lid_coupling_schematic_dark.svg#only-dark){ width="100%" }

Top of the column with the sub-grid lid. Left: the atmosphere sets one surface temperature $T_\mathrm{surf}$ for both parts of the surface. The lid part, weight $w$, is a solid lid of thickness $\delta$ over melt; it conducts from its base $T_b$ at the solidus. The open part, weight $1 - w$, is the skin of thickness $d$ over melt; it conducts from $T_\mathrm{node}$, the top basic node of the Aragog mesh directly under the skin, with $T_\mathrm{node} \ge T_b$. The arrows mark the cap $C$ on the flux of the lid part, the heat $F_\mathrm{melt}$ that the melt passes to the lid, the flux $F_\mathrm{open}$ of the open part and the flux $F_\mathrm{atm}$ of the atmosphere call. Right: temperature $T$ against depth $z$ in each part. Thicknesses are not to scale. The lid profile uses the shape $s = 1$ for illustration.

**The lid.** A crust that forms at the surface starts metres thick or less, while the top cell of the interior mesh is kilometres thick. The mesh cannot resolve it, so Aragog carries it as a sub-grid lid: a conductive solid layer of thickness $\delta$ at the top of the mantle, with its base $T_b$ at the solidus. Aragog holds $\delta$ and the lid energy per area $E_\mathrm{lid}$ as two extra solver states. A weight $w$ from 0 to 1 says how strongly the lid acts: $w = 0$ is no lid, $w = 1$ is a full lid, and values between blend the lid with the open surface. The lid conducts with

$$G_\mathrm{lid} = \frac{k_\mathrm{lid}\, f'(0)}{\delta},$$

the helpfile column `lid_conductance`. Here $k_\mathrm{lid}$ is `interior_energetics.solid_cond` and $f'(0)$ is the slope of the temperature profile of the lid at its surface, 1 for a straight profile and at least 0.1 in this conductance.

Set `interior_energetics.aragog.lid_mush_base = true` to run the lid. PROTEUS then passes `subgrid_lid` and `lid_mush_base` to Aragog. The Aragog lid also needs the solid rheology, `interior_energetics.aragog.rheology.enabled = true`.

## Three energy rules

The lid must not create or lose energy at the surface. Three rules state this.

### Rule 1: a molten surface

**Rule.** Without a lid the atmosphere sets the heat loss of the interior, exactly as in the default coupling:

$$F_\mathrm{int} = F_\mathrm{atm}\ \text{of the last atmosphere call}.$$

**How the code keeps it.** Every lid branch acts only for $w > 0$. With $w = 0$:

- the atmosphere call receives the skin columns and `T_magma` unchanged (`lid_boundary` in `atmos_clim/common.py`);
- `lid_F_open` is set to `F_atm` (`lid_open_flux`);
- Aragog takes `F_atm` as its surface flux (the boundary update in `interior_energetics/aragog.py`);
- `F_int` is assigned `F_atm` when no lid acts at the start or at the end of the step (`_surface_flux_of_step`).

In the step in which the lid switches on, the lid acts at its end, so `F_int` is the step integral of Rule 2. The lid holds its surface at the `T_surf` of the last atmosphere call. Its weight is the existence weight of the steady crust under that surface (Aragog), so it grows from 0 as the surface falls below $T_b$, without a step. In two AGNI runs (drain step $\times 1$ and $\times 0.5$) the first lid row had $w = 3.0$ and $3.6 \times 10^{-10}$ with the held surface 13.7 and 15.3 K below $T_b$, so that step booked the full `F_atm` it started from; the first rows with $w$ above 0.5 came 284 and 267 yr later, with the surface 77 and 79 K below $T_b$.

**Several balances at the onset.** The atmosphere call uses the lid weight of the last interior row, so each call has one surface temperature. One row therefore maps the held surface to the next one: the interior step sets $w$ from the held surface, and the atmosphere balance at that $w$ sets the next surface. Where the weight changes fast with the surface, this map can have 3 fixed points: a stable branch with a nearly full lid and a cold surface, a stable branch with a small weight and a hot surface, and an unstable one between them. On such a state the rows move monotonically to the stable branch on the side of the previous row and do not alternate between branches. A jump is still possible where one branch ends (a fold): the rows then move to the other branch, in one direction, over several rows. On the state of the Aragog test `test_rows_follow_the_branch_of_the_previous_row` (in the Aragog repository; grey atmosphere, top cells 67 J kg$^{-1}$ K$^{-1}$ above the solidus) the stable branches have $w$ = 0.967 and 0.100 and the unstable one $w$ = 0.71. Whether a coupled run meets a state with 3 fixed points is not known: in a coupled dummy run through the onset the row-to-row changes of $w$ and $T_\mathrm{surf}$ kept their sign, and in two AGNI runs with the same weight those of $w$ and `F_atm` did.

### Rule 2: the lid ledger

**Rule.** Under a lid, the melt below can lose less heat than the surface only by the heat the lid releases, and more only by the heat the lid stores. Per unit area,

$$F_\mathrm{melt} = F_\mathrm{surf} + \frac{dE_\mathrm{lid}}{dt},$$

with $F_\mathrm{melt}$ the heat the melt passes to the lid, $F_\mathrm{surf}$ the heat that leaves the lid surface, and $E_\mathrm{lid}$ in J m$^{-2}$.

**How the code keeps it.** Aragog holds $E_\mathrm{lid}$ as a solver state. Its energy budget counts the melt and the lid together, so the energy of the mantle changes only by the heat that crosses the surface and the core boundary. PROTEUS books what crossed the surface. With a lid at the start or at the end of the step, `F_int` is the mean surface flux of the step:

$$F_\mathrm{int} = -\frac{\Delta E_\mathrm{surf}}{A\,\Delta t},$$

with $\Delta E_\mathrm{surf}$ the surface energy integral of the step from Aragog (`step_dE_F_int_J`, negative for a loss), $A = 4\pi r^2$ at the top of the mesh and $\Delta t$ the step length. Inside the step the lid follows its own surface flux, so `F_int` can differ from `F_atm`. The column `lid_F_diff` records `F_int - F_atm`. A step in which the lid hands over to the mesh books this integral too; the coupling step does not end at the hand-over. With the mush base the lid hands over at 0.7 of the top-cell width; in the coupled runs of this page the top cell is 65.8 km thick, and the lid stays near 0.18 m in the dummy runs and below 0.06 m in the two AGNI onset runs, far from it.

### Rule 3: the atmosphere flux is the largest loss

**Rule.** The lid surface loses what the atmosphere takes from the lid part at the surface. In the next interior step, where the lid held at $T_\mathrm{surf}$ would conduct more than

$$C = \min(G_\mathrm{lid}, G_\mathrm{skin})\,(T_b - T_\mathrm{surf}),$$

its surface warms to $T_s$ and carries $F_s = C + F' (T_s - T_\mathrm{surf})$, with $F'$ the slope of `F_atm` with the surface temperature: the flux of the atmosphere at the warmer surface to first order. $F' = 0$ is the cap alone, $F_s = C$.

$C$ is the lid term of the last atmosphere balance (see the next section). At $w = 1$ that balance has only its lid term, so $C$ equals `F_atm` to the accuracy of the atmosphere solve.

**How the code keeps it.** Two parts:

- In the atmosphere call, `lid_boundary` caps $G_\mathrm{lid}$ at $G_\mathrm{skin}$, so a lid never conducts more than the open surface. With $T_b \le T_\mathrm{node}$ and $T_\mathrm{surf} \le T_\mathrm{node}$, the lid part then never loses more than the open part would at the same surface temperature.
- `lid_flux_cap` computes $C$ from the row after the atmosphere step. `_set_lid_surface` passes it to Aragog as `lid_surface_flux_cap`, next to the held surface temperature `lid_surface_temperature`. Without a lid, or for a surface at or above $T_b$ ($C \le 0$), no cap is set.
- `lid_flux_slope` takes $F'$ from the last 2 rows, $\Delta F_\mathrm{atm} / \Delta T_\mathrm{surf}$, and sets it to 0 for a surface step below 0.01 K or a slope of 0 or below. `_set_lid_surface` passes it with the cap as `lid_surface_flux_slope`.
- After the step, where the lid surface ends above the held one, `_correct_lid_cap` takes the secant of the atmosphere flux between the held and the end surface, $(F_\mathrm{atm}(T_\mathrm{end}) - F_\mathrm{atm}) / (T_\mathrm{end} - T_\mathrm{surf})$, at least 0, and runs the step again from its start with the secant as $F'$, once. A secant within $10^{-3}$ of $F'$ changes nothing, so that re-solve is skipped. The log names each re-solve with its wall time and the running total. The flux at $T_\mathrm{end}$ comes from the atmosphere module: the dummy without `fixed_flux` has it as a formula; the other modules have no corrector and take the slope of the rows alone.

Aragog applies the cap: where the lid held at $T_\mathrm{surf}$ would conduct more than $C$, its surface carries $C + F' (T_s - T_\mathrm{surf})$ at a warmer surface. With $G_\mathrm{lid} \le G_\mathrm{skin}$ the cap is close to the flux of the held lid at the start of the step, so it acts where the held lid would conduct more than at that start. The two are not equal: $C$ takes $G_\mathrm{lid}$ from the profile shape at the end surface of the last step, and the held lid has another shape at the new `T_surf` (in the first dummy row under a flux ceiling the held lid conducted 24 % less than $C$). The held lid conducts $k f_s'(0) (T_b - T_\mathrm{surf}) / \delta$, and $f_s'(0)$ rises with the shape parameter $s$ of the profile, so the cap acts while the lid thins and while $s$ rises. Near the onset, where $G_\mathrm{lid} > G_\mathrm{skin}$, the cap lies below the flux of the held lid and can act from the start of the step.

**Where the guard and the corrector act.** In dummy runs with this rule: after a kick of $-5$ K on the held surface, the row slope (87.4 W m$^{-2}$ K$^{-1}$) lies within 2.2 % of the exact slope of the dummy flux (85.5), and the re-solve moves the end surface by 0.001 K. In the first row under a soft ceiling of the dummy flux, the surface rose from 729 to 819 K while `F_atm` fell, so the row slope is below 0 and the guard sets $F' = 0$. The lid then ended at 1406.3 K, far above the held 819.3 K; the corrector took the secant 0.0766 W m$^{-2}$ K$^{-1}$ and the step ended at 1406.1 K, as with the exact slope (secant 0.0770). The guard and the corrector, not the row slope alone, carry such a row.

## What the atmosphere call sees

Under a lid the surface has two parts that share one surface temperature. The lid part, weight $w$, conducts from the lid base $T_b$ (column `lid_T_base`). The open part, weight $1 - w$, conducts through the skin from the top node of the mesh $T_\mathrm{node}$ (column `lid_T_node`). The two act in parallel:

$$F = w\,G_\mathrm{lid}'\,(T_b - T_\mathrm{surf}) + (1 - w)\,G_\mathrm{skin}\,(T_\mathrm{node} - T_\mathrm{surf}), \qquad G_\mathrm{lid}' = \min(G_\mathrm{lid}, G_\mathrm{skin}).$$

AGNI and the dummy solve against one skin, so `lid_boundary` writes this sum in the form of one skin. With the blended conductance

$$G = w\,G_\mathrm{lid}' + (1 - w)\,G_\mathrm{skin},$$

the call takes the skin thickness $d_\mathrm{call} = k / G$ (with the same $k$) and the melt temperature

$$T_\mathrm{magma,call} = \frac{w\,G_\mathrm{lid}'\,T_b + (1 - w)\,G_\mathrm{skin}\,T_\mathrm{node}}{G}.$$

Then $\frac{k}{d_\mathrm{call}}\,(T_\mathrm{magma,call} - T_\mathrm{surf})$ equals $F$ above for every $T_\mathrm{surf}$, so the atmosphere solves the parallel balance with the skin balance it already has. These values hold for the call only: `lid_boundary` writes nothing to the row. The row keeps the `T_magma` that Aragog writes under a lid, $w T_b + (1 - w) T_\mathrm{node}$, the melt temperature that outgassing reads. The large-increase clip of `tmagma_atol` and `tmagma_rtol` in the interior wrapper acts on this row value, not on the lid branch of the call. `planet.prevent_warming` is refused with the lid (see Limits).

**Why the 10 mm skin is not added on top of the lid.** The skin stands for the boundary layer at the top of a molten surface. Under a solid lid no melt reaches the surface, so that boundary layer does not exist: the lid is the conductive layer, and $T_\mathrm{surf}$ is the temperature at its top. Adding the skin in series, $1/G = 1/G_\mathrm{lid} + 1/G_\mathrm{skin}$, would count one layer twice; at $G_\mathrm{lid} = G_\mathrm{skin}$ it would halve the conductance of the lid part.

**What the interior takes.** After the atmosphere step, `run_atmosphere` writes `lid_F_open`, the flux of the open part:

$$F_\mathrm{open} = r\,G_\mathrm{skin}\,(T_\mathrm{node} - T_\mathrm{surf}), \qquad r = \frac{F_\mathrm{atm}}{\frac{k}{d_\mathrm{call}}\,(T_\mathrm{magma,call} - T_\mathrm{surf})},$$

and `F_atm` where no lid acts. The ratio $r$ compares the flux of the atmosphere with the parallel skin balance of its call. Where the atmosphere closes that balance, $r = 1$ and the weighted lid and open parts sum to `F_atm`. Where it does not, only the open part is scaled by $r$ and the cap $C$ of the lid part is not, so with the lid at its cap the weighted parts $w C + (1 - w) F_\mathrm{open}$ miss `F_atm` by $w C (r - 1)$; in the two AGNI onset runs they summed to 0.956 to 1.000 of `F_atm`. This sum needs a surface below $T_b$. For a surface at or above $T_b$ no cap is set and Aragog holds the lid surface 1 K below $T_b$, so the flux of the lid part is no longer $C$ and the sum does not hold. $F_\mathrm{open}$ tends to `F_atm` as $w$ goes to 0, so the boundary of the open part does not step by the difference when the weight first exceeds 0. Where the balance of the call is not positive (a surface at or above the melt side of the call), $F_\mathrm{open}$ is the skin flux alone. In the next interior step Aragog takes `lid_F_open` as the flux of the part $1 - w$ and holds the lid surface at $T_\mathrm{surf}$. `F_atm` itself holds the lid term too, so as the flux of the open part it would count the lid part a second time.

**Where this is checked.** [Validation: the lid and the open surface in the atmosphere call](../Validation/atmos_clim/common.md) checks the parallel sum of the call and the sum of the two parts against their closed forms.

## The onset thickness

Aragog switches the lid on through its weight $w = 1 - g\,(1 - w_\mathrm{s})$, with $g$ the floor weight of the top cell and $w_\mathrm{s}$ the existence weight of the steady crust under the held surface, the `T_surf` of the last atmosphere call: the thickness $\delta_\mathrm{eq} = k_\mathrm{lid} (T_b - T_\mathrm{surf}) / F_b$ against the onset thickness $\delta_\mathrm{on}$, with $F_b$ the base flux. $w_\mathrm{s}$ is 0 while the held surface lies at or above $T_b$; for $F_b > 0$ it rises without a step as the surface falls below and is 0.5 where $\delta_\mathrm{eq} = \delta_\mathrm{on}$, and for $F_b \le 0$ it is 1. A weight below $10^{-10}$ counts as 0, so the first lid row comes some K below $T_b$ (13.7 and 15.3 K in the two AGNI runs above). PROTEUS derives $\delta_\mathrm{on}$ from the skin:

$$\delta_\mathrm{on} = \frac{k_\mathrm{lid}\, d}{k}.$$

For the defaults ($k_\mathrm{lid} = 4$ W m$^{-1}$ K$^{-1}$, $d = 0.01$ m, $k = 2$ W m$^{-1}$ K$^{-1}$) this gives $\delta_\mathrm{on} = 0.02$ m. At this thickness a straight lid has $k_\mathrm{lid}/\delta_\mathrm{on} = k/d = G_\mathrm{skin}$, so it conducts like the skin it replaces. The lid enters, however, at the phantom thickness $\delta_\mathrm{lo} = \delta_\mathrm{on}/2$ (see [Resume](#resume)), so its conductance at the onset is $2 f'(0)\,G_\mathrm{skin}$. For the profile family of the Aragog lid, $f'(0) = 2\sqrt{s}/(\sqrt{\pi}\,\mathrm{erf}\sqrt{s})$ for a growing lid of shape $s > 0$: 1.03 at $s = 0.1$ and 1.34 at $s = 1$; a thinning lid ($s < 0$) has $f'(0) < 1$. The atmosphere call caps $G_\mathrm{lid}$ at $G_\mathrm{skin}$ (Rule 3), so it does not see that step while $G_\mathrm{lid} > G_\mathrm{skin}$.

**Setting it.** `interior_energetics.aragog.lid_delta_on` \[m\]:

- Default none: PROTEUS takes the skin value above.
- A finite number above 0: used as given. PROTEUS logs a warning when it differs from the skin value by more than 10 %, for the defaults outside 0.018 to 0.022 m.
- A bool, a string, NaN, 0 or a negative value is refused at config load.

At setup PROTEUS logs the value it passes, `Aragog sub-grid lid onset thickness lid_delta_on = ... m`. The helpfile column `lid_delta_on` holds the value of the solver, 0 without the lid. Aragog refuses an onset thickness at or above 0.7 of the top-cell width, where the lid with the mush base hands over.

## Resume

Each interior snapshot (`data/<time>_int.nc`) of a run with the lid stores:

| Snapshot field | Unit | Content |
|---|---|---|
| `lid_delta_state` | m | lid thickness $\delta$, 0 after a hand-over to the mesh |
| `lid_E_state` | J m$^{-2}$ | lid energy $E_\mathrm{lid}$ |
| `lid_handover_time_state` | yr | time of the hand-over, only when one happened |

A run with the lid off writes none of them. A non-finite value is not written, with a warning.

On `proteus start -r`, PROTEUS restores the lid of the snapshot first and then its entropy profile (`restore_lid_start`, then `set_initial_entropy`, in `interior_energetics/aragog.py`). A new solver without a set lid would otherwise start the phantom lid, which refuses a top cell below the full floor gap or in the heat-jump band. If the snapshot entropy does not fit the cells of the resumed mesh, the resume stops with an error when a structure update was pending at the stop; otherwise PROTEUS logs an error, restarts from the initial entropy profile and the lid starts as the phantom lid. Otherwise:

- A solver without the lid takes nothing.
- If the lid thickness or the lid energy is absent or not finite, PROTEUS logs a warning that names the status of both. The lid then restarts as the phantom lid of Aragog at `lid_delta_lo`; by the Aragog default that is half of `lid_delta_on`, 0.01 m for the default skin.
- Otherwise PROTEUS calls the Aragog `set_initial_lid` with the stored thickness, energy and hand-over time. If Aragog refuses the stored lid, the resume stops with an error that names the snapshot time.

A helpfile written without the lid columns resumes with them read as zero, which means no lid.

A resume that changes `lid_mush_base` from the value in the run's `init_coupler.toml` stops with status 20, unless `interior_energetics.aragog.allow_resume_law_change` is true; see [Resume stops on a separation law](../How-to/troubleshooting.md#resume-stops-on-a-separation-law).

## Helpfile columns

The interior writes every column except `lid_F_open` at each step; the atmosphere wrapper writes `lid_F_open`. With the lid off the columns read 0, except `lid_T_node` (the top node of the mesh) and `lid_F_open` (`F_atm`). With the lid on and $w = 0$, `lid_delta` holds the thin phantom lid, so read it with `lid_weight`. The full list is in [Output format](../Reference/output.md).

| Column | Unit | Meaning |
|---|---|---|
| `lid_weight` | 1 | weight $w$ of the lid |
| `lid_conductance` | W m$^{-2}$ K$^{-1}$ | conductance $G_\mathrm{lid}$ of the lid |
| `lid_delta` | m | thickness $\delta$ of the lid; 0 without the lid, read with `lid_weight` |
| `lid_delta_on` | m | onset thickness $\delta_\mathrm{on}$; 0 without the lid |
| `lid_T_surface` | K | surface temperature of the lid |
| `lid_T_base` | K | base temperature $T_b$ of the lid |
| `lid_T_node` | K | top basic node $T_\mathrm{node}$ of the Aragog mesh |
| `lid_F_open` | W m$^{-2}$ | flux of the open part $1 - w$ of the surface |
| `lid_F_diff` | W m$^{-2}$ | `F_int` of the step minus the `F_atm` it started from |

## Limits

- **JANUS.** JANUS does not read the lid. With `interior_energetics.module = "aragog"` and `lid_mush_base = true`, a config with `atmos_clim.module = "janus"` is refused at load, and so is any `atmos_clim.surf_state` other than `"skin"`. The dummy with `atmos_clim.dummy.fixed_flux > 0` is refused too: it returns its flux without the skin, so the lid does not reach the atmosphere.
- **prevent_warming.** `planet.prevent_warming = true` is refused with the lid at load: it clamps `F_atm` while the open part of the lid keeps the skin flux, so the flux balance of the lid does not close.
- **Saturation skin.** With `surface_d = "saturation"` the skin is the stack of condensate layers, and it has no thickness while no layer has formed; a condensate layer over the lid is not modelled. A config with the lid and a saturation skin is refused at load. A live lid with a zero skin, as on a resume from a helpfile without `boundary_layer_conductivity`, stops the run with an error, because the open part then has no conductance to blend.
- **One column.** The lid and the open part share one surface temperature in a 1-D column, blended by $w$. The model has no map of where on the planet a crust lies.
- **The cap.** Rule 3 caps the flux of the lid part only; the open part keeps the boundary flux `lid_F_open` for the whole step.
- **The slope with AGNI.** With AGNI the cap takes $F'$ from the last 2 rows alone, and `_correct_lid_cap` does not act. A secant from AGNI needs a second solve at a fixed surface temperature, and that solve differs from the flux of the row by up to about $2\times10^{-3}$ at `solution_rtol` 0.01 and $2.5\times10^{-3}$ at 0.15. In one AGNI onset passage at `solution_rtol` 0.01, with steps of 12 to 25 K, a secant from the row flux at the held end and a second solve at the end surface was off by 2.5 to 3.6 per cent, as much as the correction itself (0.3 to 3.4 per cent). Where `F_atm` levels off with the surface temperature within one step, a slope from the rows before lies above the slope across the step, and the lid part can lose more than the atmosphere at the end surface. That case is tested only on the dummy flux with a soft ceiling: it has no AGNI form and is not tested with AGNI.

For the config keys, see [Aragog](../Reference/config/interior.md#aragog-interior_energeticsaragog) and [Atmosphere climate](../Reference/config/atmosphere.md#atmosphere-climate-atmos_clim).
