# Sub-grid lid and the surface skin

A thin solid crust at the top of the Aragog mantle changes the heat path from the melt to the atmosphere. The lid enters the surface balance of the atmosphere, and the coupling keeps three energy rules between the melt, the lid and the atmosphere.

This page assumes you know the [coupling loop](coupling_loop.md): the interior runs first in each iteration with the flux `F_atm` of the last atmosphere call, and the atmosphere then solves for the surface temperature `T_surf`. The lid itself, its profile, its growth and its hand-over to the mesh, is described on the Surface lid page of the Aragog documentation. This page covers the PROTEUS side: what the atmosphere sees and what the interior takes. It applies to `interior_energetics.module = "aragog"` with the default `interior_energetics.surface_bc_mode = "flux"`.

## The skin and the lid

**The skin.** With the default `atmos_clim.surf_state = "skin"`, a thin conductive layer sits between the melt and the surface. It stands for the boundary layer at the top of a molten surface. Its thickness $d$ is `atmos_clim.surface_d` (default 0.01 m) and its conductivity $k$ is `atmos_clim.surface_k` (default 2 W m$^{-1}$ K$^{-1}$). Each iteration writes them into the helpfile columns `boundary_layer_thickness` and `boundary_layer_conductivity`. The skin conductance is

$$G_\mathrm{skin} = \frac{k}{d},$$

200 W m$^{-2}$ K$^{-1}$ for the defaults. The atmosphere module (AGNI or the dummy) finds the surface temperature $T_\mathrm{surf}$ at which the heat conducted through the skin equals the net flux the atmosphere carries away:

$$F_\mathrm{atm}(T_\mathrm{surf}) = G_\mathrm{skin}\,(T_\mathrm{magma} - T_\mathrm{surf}).$$

It returns `F_atm`; in AGNI this is the total flux at the top level. Without a lid the interior takes `F_atm` as its surface flux for the next step.

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

In the step in which the lid switches on, the lid acts at its end, so `F_int` is the step integral of Rule 2. The lid holds its surface at the `T_surf` of the last atmosphere call. Its weight is the existence weight of the steady crust under that surface (Aragog), so it grows from 0 as the surface falls below $T_b$, without a step. In two AGNI runs (drain step $\times 1$ and $\times 0.5$) the first lid row had $w = 3 \times 10^{-10}$ with the held surface 13.7 and 15.3 K below $T_b$, so that step booked the full `F_atm` it started from; $w$ passed 0.5 about 270 to 280 yr later, with the surface 77 and 79 K below $T_b$.

**Several balances at the onset.** The atmosphere call uses the lid weight of the last interior row, so each call has one surface temperature. One row therefore maps the held surface to the next one: the interior step sets $w$ from the held surface, and the atmosphere balance at that $w$ sets the next surface. Where the weight changes fast with the surface, this map can have 3 fixed points: a stable branch with a nearly full lid and a cold surface, a stable branch with a small weight and a hot surface, and an unstable one between them. The rows then move monotonically to the stable branch on the side of the previous row and do not alternate between branches. A jump is still possible where one branch ends (a fold): the rows then move to the other branch, in one direction, over several rows. On the Aragog test state of `test_rows_follow_the_branch_of_the_previous_row` (grey atmosphere, top cells 67 J kg$^{-1}$ K$^{-1}$ above the solidus) the stable branches have $w$ = 0.967 and 0.100 and the unstable one $w$ = 0.709. In a coupled dummy run through the onset the row-to-row changes of $w$ and $T_\mathrm{surf}$ kept their sign; in two AGNI runs with the same weight, those of $w$ and `F_atm` did.

### Rule 2: the lid ledger

**Rule.** Under a lid, the melt below can lose less heat than the surface only by the heat the lid releases, and more only by the heat the lid stores. Per unit area,

$$F_\mathrm{melt} = F_\mathrm{surf} + \frac{dE_\mathrm{lid}}{dt},$$

with $F_\mathrm{melt}$ the heat the melt passes to the lid, $F_\mathrm{surf}$ the heat that leaves the lid surface, and $E_\mathrm{lid}$ in J m$^{-2}$.

**How the code keeps it.** Aragog holds $E_\mathrm{lid}$ as a solver state. Its energy budget counts the melt and the lid together, so the energy of the mantle changes only by the heat that crosses the surface and the core boundary. PROTEUS books what crossed the surface. With a lid at the start or at the end of the step, `F_int` is the mean surface flux of the step:

$$F_\mathrm{int} = -\frac{\Delta E_\mathrm{surf}}{A\,\Delta t},$$

with $\Delta E_\mathrm{surf}$ the surface energy integral of the step from Aragog (`step_dE_F_int_J`, negative for a loss), $A = 4\pi r^2$ at the top of the mesh and $\Delta t$ the step length. Inside the step the lid follows its own surface flux, so `F_int` can differ from `F_atm`. The column `lid_F_diff` records `F_int - F_atm`. A step in which the lid hands over to the mesh books this integral too; the coupling step does not end at the hand-over. With the mush base the lid hands over at 0.7 of the top-cell width; in the coupled runs of this page the top cell is 65.8 km thick and the lid stays near 0.18 m, far from it.

### Rule 3: the atmosphere flux is the largest loss

**Rule.** The lid surface can lose at most what the atmosphere sustains through the lid part of the surface. In the next interior step the lid surface flux $F_s$ obeys

$$F_s \le C = \min(G_\mathrm{lid}, G_\mathrm{skin})\,(T_b - T_\mathrm{surf}).$$

$C$ is the lid term of the last atmosphere balance (see the next section). At $w = 1$ that balance has only its lid term, so $C$ equals `F_atm` to the accuracy of the atmosphere solve.

**How the code keeps it.** Two parts:

- In the atmosphere call, `lid_boundary` caps $G_\mathrm{lid}$ at $G_\mathrm{skin}$, so a lid never conducts more than the open surface. With $T_b \le T_\mathrm{node}$ and $T_\mathrm{surf} \le T_\mathrm{node}$, the lid part then never loses more than the open part would at the same surface temperature.
- `lid_flux_cap` computes $C$ from the row after the atmosphere step. `_set_lid_surface` passes it to Aragog as `lid_surface_flux_cap`, next to the held surface temperature `lid_surface_temperature`. Without a lid, or for a surface at or above $T_b$ ($C \le 0$), no cap is set.

Aragog applies the cap: where the lid held at $T_\mathrm{surf}$ would conduct more than $C$, its surface carries $C$ at a warmer surface. With $G_\mathrm{lid} \le G_\mathrm{skin}$ the cap equals the flux of the held lid at the start of the step, so it acts where the held lid would conduct more than at that start. The held lid conducts $k f_s'(0) (T_b - T_\mathrm{surf}) / \delta$, and $f_s'(0)$ rises with the shape parameter $s$ of the profile, so the cap acts while the lid thins and while $s$ rises. Near the onset, where $G_\mathrm{lid} > G_\mathrm{skin}$, the cap lies below the flux of the held lid and can act from the start of the step.

## What the atmosphere call sees

Under a lid the surface has two parts that share one surface temperature. The lid part, weight $w$, conducts from the lid base $T_b$ (column `lid_T_base`). The open part, weight $1 - w$, conducts through the skin from the top node of the mesh $T_\mathrm{node}$ (column `lid_T_node`). The two act in parallel:

$$F = w\,G_\mathrm{lid}'\,(T_b - T_\mathrm{surf}) + (1 - w)\,G_\mathrm{skin}\,(T_\mathrm{node} - T_\mathrm{surf}), \qquad G_\mathrm{lid}' = \min(G_\mathrm{lid}, G_\mathrm{skin}).$$

AGNI and the dummy solve against one skin, so `lid_boundary` writes this sum in the form of one skin. With the blended conductance

$$G = w\,G_\mathrm{lid}' + (1 - w)\,G_\mathrm{skin},$$

the call takes the skin thickness $d_\mathrm{call} = k / G$ (with the same $k$) and the melt temperature

$$T_\mathrm{magma,call} = \frac{w\,G_\mathrm{lid}'\,T_b + (1 - w)\,G_\mathrm{skin}\,T_\mathrm{node}}{G}.$$

Then $\frac{k}{d_\mathrm{call}}\,(T_\mathrm{magma,call} - T_\mathrm{surf})$ equals $F$ above for every $T_\mathrm{surf}$, so the atmosphere solves the parallel balance with the skin balance it already has. These values hold for the call only: `lid_boundary` writes nothing to the row. The row keeps the `T_magma` that Aragog writes under a lid, $w T_b + (1 - w) T_\mathrm{node}$, the melt temperature that outgassing reads. The `T_magma` limiters of the interior wrapper (`planet.prevent_warming` and the large-increase clip of `tmagma_atol` and `tmagma_rtol`) act on this row value, not on the lid branch of the call. With `planet.prevent_warming` the wrapper also limits `F_int` to that of the previous row (after the init stage) and to at least $10^{-8}$ W m$^{-2}$, so a row it limits does not book the step integral of Rule 2.

**Why the 10 mm skin is not added on top of the lid.** The skin stands for the boundary layer at the top of a molten surface. Under a solid lid no melt reaches the surface, so that boundary layer does not exist: the lid is the conductive layer, and $T_\mathrm{surf}$ is the temperature at its top. Adding the skin in series, $1/G = 1/G_\mathrm{lid} + 1/G_\mathrm{skin}$, would count one layer twice; at $G_\mathrm{lid} = G_\mathrm{skin}$ it would halve the conductance of the lid part.

**What the interior takes.** After the atmosphere step, `run_atmosphere` writes `lid_F_open`, the flux of the open part:

$$F_\mathrm{open} = r\,G_\mathrm{skin}\,(T_\mathrm{node} - T_\mathrm{surf}), \qquad r = \frac{F_\mathrm{atm}}{\frac{k}{d_\mathrm{call}}\,(T_\mathrm{magma,call} - T_\mathrm{surf})},$$

and `F_atm` where no lid acts. The ratio $r$ compares the flux of the atmosphere with the parallel skin balance of its call. Where the atmosphere closes that balance, $r = 1$ and the weighted lid and open parts sum to `F_atm`. Where it does not, the open part takes the difference in the same ratio, so $F_\mathrm{open}$ tends to `F_atm` as $w$ goes to 0: the boundary of the open part does not step by the difference when the weight first exceeds 0. Where the balance of the call is not positive (a surface at or above the melt side of the call), $F_\mathrm{open}$ is the skin flux alone. In the next interior step Aragog takes `lid_F_open` as the flux of the part $1 - w$ and holds the lid surface at $T_\mathrm{surf}$. `F_atm` itself holds the lid term too, so as the flux of the open part it would count the lid part a second time.

## The onset thickness

Aragog switches the lid on through its weight $w = 1 - g\,(1 - w_\mathrm{g})$, with $g$ the floor weight of the top cell and $w_\mathrm{g}$ the weight of a grey-body crust (emissivity 1 at `T_eqm`) whose steady thickness passes the onset thickness $\delta_\mathrm{on}$. In the flux mode $w_\mathrm{g}$ is 0 while the held surface, the `T_surf` of the last atmosphere call, lies at or above $T_b$; below $T_b$ it does not depend on how far below. The lid therefore switches on in the first step whose held surface lies below $T_b$. PROTEUS derives $\delta_\mathrm{on}$ from the skin:

$$\delta_\mathrm{on} = \frac{k_\mathrm{lid}\, d}{k}.$$

For the defaults ($k_\mathrm{lid} = 4$ W m$^{-1}$ K$^{-1}$, $d = 0.01$ m, $k = 2$ W m$^{-1}$ K$^{-1}$) this gives $\delta_\mathrm{on} = 0.02$ m. At this thickness a straight lid has $k_\mathrm{lid}/\delta_\mathrm{on} = k/d = G_\mathrm{skin}$, so it conducts like the skin it replaces. The lid enters, however, at the phantom thickness $\delta_\mathrm{lo} = \delta_\mathrm{on}/2$ (see [Resume](#resume)), so its conductance at the onset is $2 f'(0)\,G_\mathrm{skin}$. For the profile family of the Aragog lid, $f'(0) = 2\sqrt{s}/(\sqrt{\pi}\,\mathrm{erf}\sqrt{s})$ for a growing lid of shape $s > 0$: 1.03 at $s = 0.1$ and 1.34 at $s = 1$; a thinning lid ($s < 0$) has $f'(0) < 1$. The atmosphere call caps $G_\mathrm{lid}$ at $G_\mathrm{skin}$ (Rule 3), so it does not see that step while $G_\mathrm{lid} > G_\mathrm{skin}$.

**Setting it.** `interior_energetics.aragog.lid_delta_on` \[m\]:

- Default none: PROTEUS takes the skin value above.
- A finite number above 0: used as given. PROTEUS logs a warning when it differs from the skin value by more than 10 %, for the defaults outside 0.018 to 0.022 m.
- A bool, a string, NaN, 0 or a negative value is refused at config load.
- With `atmos_clim.surface_d = "saturation"` the skin has no fixed thickness. The config is then refused unless `lid_delta_on` is set.

At setup PROTEUS logs the value it passes, `Aragog sub-grid lid onset thickness lid_delta_on = ... m`. The helpfile column `lid_delta_on` holds the value of the solver, 0 without the lid. Aragog refuses an onset thickness at or above 0.8 of the top-cell width.

## Resume

Each interior snapshot (`data/<time>_int.nc`) of a run with the lid stores:

| Snapshot field | Unit | Content |
|---|---|---|
| `lid_delta_state` | m | lid thickness $\delta$, 0 after a hand-over to the mesh |
| `lid_E_state` | J m$^{-2}$ | lid energy $E_\mathrm{lid}$ |
| `lid_handover_time_state` | yr | time of the hand-over, only when one happened |

A run with the lid off writes none of them. A non-finite value is not written, with a warning.

On `proteus start -r`, PROTEUS restores the entropy profile of the snapshot and then the lid (`restore_lid_start` in `interior_energetics/aragog.py`):

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
| `lid_F_open` | W m$^{-2}$ | flux of the surface without a lid |
| `lid_F_diff` | W m$^{-2}$ | `F_int` of the step minus the `F_atm` it started from |

## Limits

- **JANUS.** JANUS does not read the lid. With `interior_energetics.module = "aragog"` and `lid_mush_base = true`, a config with `atmos_clim.module = "janus"` is refused at load, and so is any `atmos_clim.surf_state` other than `"skin"`.
- **Saturation skin.** With `surface_d = "saturation"` the skin is the stack of condensate layers, and it has no thickness while no layer has formed. The config needs an explicit `lid_delta_on`. A live lid with a zero skin stops the run with an error, because the open part then has no conductance to blend.
- **One column.** The lid and the open part share one surface temperature in a 1-D column, blended by $w$. The model has no map of where on the planet a crust lies.
- **The cap.** Rule 3 caps the flux of the lid part only; the open part keeps the boundary flux `lid_F_open` for the whole step.

For the config keys, see [Aragog](../Reference/config/interior.md#aragog-interior_energeticsaragog) and [Atmosphere climate](../Reference/config/atmosphere.md#atmosphere-climate-atmos_clim).
