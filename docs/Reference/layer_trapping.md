# Layer trapping details

This page holds the integration, the bookkeeping, the resume path, the output columns and the limits of the layer scheme, `outgas.trap_mode = "layers"`. The physics and the symbols are explained on [Layer drainage node by node](../Explanations/outgas.md#layer-drainage-node-by-node); read that page first.

Symbols used here:

- $\psi$: the pore melt of a node, as a fraction of its volume \[1\];
- $\phi$: the solver's melt volume fraction of the node, its porosity \[1\];
- $V$, $A$: the volume \[m$^3$\] and the upper face area \[m$^2$\] of a node;
- $q$: the melt volume that leaves the top of a node per second \[m$^3$ s$^{-1}$\];
- $D_Z$: the crystal/melt partition coefficient of species $Z$, `outgas.D_const_<Z>` \[1\].

## The porosity bound within a step

`drain_mush_layers` in `outgas/layer_drainage.py` holds the pore melt of every mush node to the solver porosity:

- The bound starts at the porosity of the previous step's melt fraction, taken with this step's phase-boundary densities, and ends at this step's porosity. Between the two it is linear in time.
- Pore melt beyond the bound crystallises at the start of the step, after each sub-step against the bound at its end, and at the end of the step. A node drains from the step after the one in which it enters the mush, so the first crystallisation includes the melt that crystallised between the transition and the end of its entry step.
- The crystals keep $D_Z$ times the concentration of the melt they grow from. `_drain` in `outgas/node_record.py` moves that mass from the node's pore store to its crystal store, which does not drain. The rest of the volatiles stays in the pore melt that is left.
- Crystallised melt does not rejoin the pore melt. A rising porosity opens pore space that melt from below can fill; the crystals that melt give their share of the crystal store back to the pore melt ([Remelting node by node](#remelting-node-by-node)).
- Nodes of a sealed or held mush layer do not drain, but crystallise to the bound at the start and the end of the step.

## Sub-steps

The fluxes are integrated with Heun's second-order Runge-Kutta scheme. The first stage takes the fluxes of the pore melt against the bound at the start of the sub-step, the second those of the trial state, held to the bound at its end, against that bound.

- Each sub-step spans at most a tenth (`SUBSTEP_FRACTION = 0.1`) of the shortest time any node takes to pass on what it holds, $\psi V / q$. This keeps the pore melt and the volatiles of every node positive.
- While the porosity of a draining node changes within the step, a sub-step also spans at most a tenth of the step, so the bound is followed in ten pieces or more.
- What a node receives is capped in each stage at what it passes on plus its free pore space, taken from the top of the layer down (`_limit_inflow`). The cap needs no sub-step limit of its own, so a nearly closed node holds the melt below it without shortening the sub-steps.
- After `MAX_SUBSTEPS = 20000` sub-steps the rest of the step is left undrained, with a warning: an upper bound on what the mush keeps. A draining column slows as it drains, so this is a safety net.

## Sealed layers and dense melt

Melt that leaves the top of a mush layer rejoins the magma ocean, or leaves through the surface when the layer reaches the top of the mesh. A mush layer with a solid node above it is sealed and keeps its melt. A node whose tables put the melt at or above the solid density cannot drive its melt upward, since the solver's separation velocity sinks it. It takes the front scheme's no-drainage bound and keeps its pore melt, and so do the mush nodes below it in the same mush layer (`split_at_dense`). In Miyazaki and Korenaga (2019) such melt sinks instead; the layer scheme does not move melt downward. The time a node spends held, `t_held`, counts towards the no-drainage branch of its exit.

## Node masses

The node masses are Aragog's cell masses, rescaled so that they sum to $M_\mathrm{mantle}$, the mass the melt concentrations of the chemistry refer to. Aragog writes each cell mass as its current density times a fixed volume, and that sum grows past the mantle mass as the mantle cools.

## Returns, escape and desiccation

Every mass the record returns to the melt, by a remelting node or by pore melt that drains back to the ocean, is written to `trap_kg_returned`, also on a step that buries more than it returns. On a step with a return, the escape floor is off and a desiccated planet is checked again. The floor empties an element whose reachable mass falls below `outgas.mass_thresh`; on a step that resupplies the melt it would delete the return whenever the return is small, so the result would depend on how many steps the return takes. A remelt and a drainage return resupply the melt alike, so the floor is off for both. `trap_branch = 7` marks a net return by remelting nodes, as a diagnostic only.

## Remelting node by node

A node that becomes liquid again returns everything trapped at it to the melt. A liquid or mush node whose melt fraction rises from $x_0$ to $x$ melts the share $(x - x_0)/(1 - x_0)$ of its crystals, and that share of its crystal store goes with the new melt. A liquid node returns it to the ocean, so a liquid node at $x = 1$ holds nothing. A mush node puts it into its pore melt, which drains. Nodes that stay solid keep theirs, even when the global melt fraction rises. A solid node that re-enters the mush from below makes its new pore melt from its own locked material and buries nothing new from the ocean. The burial ledger of the front scheme is not used in this mode.

## The ocean melt the chemistry sees

A layer step writes `trap_Phi_ocean` together with its count of mush nodes, `trap_n_mush`. Each iteration of the layer scheme sets both to 0 before it steps, so an iteration that takes no layer step, as in the initialisation stage, leaves 0 in both. A row with 0 in both leaves `Phi_global` to the solve, as on a fresh row. A value of 0 with mush nodes means that all the melt is in the mush, and the solve then dissolves nothing.

## Resume, mesh changes and missing profiles

The per-node record lives on the interior object between steps. It is saved in each interior snapshot as `trap_*_s` variables, and a resumed run reads it back. A snapshot written before `trap_psi_carried_s` existed reads that field as the pore melt.

Without a saved record, or with one sized for another mesh, the record restarts on the melt fraction of the snapshot the run resumes from, taken onto the current mesh by radius. The mush it finds there dates from the snapshot's time. The trapped mass on the helpfile is assigned to the nodes that were not liquid there, in proportion to their mass, so those nodes bury nothing new. A node that leaves the ocean in the first step buries its pore melt, as in a run that kept its record. Without that snapshot, the new record starts on the current profiles, and the mush it finds buries nothing new.

A step without the interior profiles traps at $D_Z\,C_Z\,\Delta M$ alone when the mantle freezes. When it remelts, the step returns the remelted share of the solid and takes the same share out of the record. It does not use the burial ledger, so nothing is returned twice.

## Output columns

The trapping columns have one meaning in both modes. For `"layers"` they read as follows:

- `trap_F_tl` \[1\]: the mass-weighted trapped melt fraction of the nodes that froze in the step, 0 when none did. For each node it is its entry melt fraction times the share of its entry volatiles it locked, for a species the crystals do not take, so it compares with the front scheme's trapped melt fraction. The record carries that share as `psi_carried`, the pore melt at the entry concentration that the node's volatiles amount to: drainage lowers it, crystallisation does not.
- `trap_n_front`, `trap_L_front` \[m\], `trap_tau_D` and `trap_tau_s` \[yr\]: the node count, thickness, percolation time and matrix time of the thickest mush layer, which stands for the front.
- `trap_v_front` \[m s$^{-1}$\] and `trap_front_courant` \[1\]: the front speed from the mass that froze, at the base of the thickest mush layer, and its advance in that layer's thicknesses. `trap_w_matrix_over_vf` is not computed, since no single front top exists, and the helpfile stores it as 0.
- `trap_t_res` \[yr\]: the mass-weighted time the nodes that froze spent in the mush.
- `trap_n_mush`: the mush nodes of the whole mesh; 0 with `"front"`. `trap_n_exited`: the nodes that froze in the step.
- `trap_frac_bound` \[1\]: the mass share of the nodes that froze on the no-drainage bound; 0 when none froze. It includes nodes that drained before their mush layer was sealed, so it is not the share that kept their entry pore melt.
- `trap_kg_returned` \[kg\]: the mass returned to the melt this step. `trap_Phi_ocean` \[1\]: the ocean melt the chemistry dissolves into; 0 with `"front"`. `trap_n_substeps`: the drainage sub-steps of the step.

## Limits

- A node drains from the step after the one in which it enters the mush. A node that crosses the transition early in a long step therefore drains less in that step than it would in shorter steps. The validation cases put every entry at the end of a step.
- The bound is linear in time between the step ends; the solver's own path within the step is not known.
- Melt that crystallised does not melt back into the pore melt when the porosity rises again; the pore space it leaves fills only with melt from below.
- With `params.dt.mushy_maximum` set, the interior time step is capped while `params.stop.solid.phi_crit` < `Phi_global` < `params.dt.mushy_upper`, and not below that band. Many nodes can then cross the floor in one step, which the record follows only at that step's resolution.
- Before the record starts, the code knows no earlier melt fraction per node. The crystals grown in the step before it are not counted.
