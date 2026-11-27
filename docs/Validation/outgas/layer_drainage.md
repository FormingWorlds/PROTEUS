# Validation: `src/proteus/outgas/layer_drainage.py`

## What this page checks

This page checks how the layer trapping scheme (`outgas.trap_mode = "layers"`) drains the melt held between the crystals of the mush, node by node (`proteus.outgas.layer_drainage`). A node is one cell of the radial mesh of the Aragog interior solver. The melt between the crystals of a node is its pore melt. It rises into the node above when the melt is lighter than the crystals, and the scheme keeps what is left when the node freezes.

Validation here means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. The answers come from the drainage integral of the front scheme, a second code, and from 2 analytical solutions. If this part of the code were wrong, a run would trap too much or too little water and CO$_2$ in its mantle, and the atmosphere would get the rest. Every case runs at the sub-step the scheme uses in a real run, `SUBSTEP_FRACTION = 0.1`.

Each case below has the same layout: why the check exists, what it tests, how we know the right answer, why it matters, a plot, and a box "Test details". Symbols used on this page:

- $\psi$: the pore melt of a node, as a fraction of its volume \[1\]; $\psi_0$ is its value at the start;
- $L$, $V$, $A$: the thickness \[m\], volume \[m$^3$\] and upper face area \[m$^2$\] of a node, with $L = V/A$;
- $w_D$: the speed at which melt rises through the pores, the percolation speed \[m s$^{-1}$\];
- $\tau_s$: the time the crystal framework takes to compact and push out its melt, the matrix time \[s\];
- $q_k$: the melt volume that leaves the top of node $k$ per second \[m$^3$ s$^{-1}$\].

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_layer_drainage.py::test_single_node_reproduces_the_front_drainage_integral` | Second code: the drainage integral of the front scheme | One node drains as one parcel of the front scheme, under both buoyancy settings. | 2026-11-24 |
| `tests/outgas/test_layer_drainage.py::test_compaction_limited_node_decays_exponentially` | Analytical solution $\psi_0 e^{-t/\tau_s}$ | One node whose crystal framework alone sets the rate. | 2026-11-24 |
| `tests/outgas/test_layer_drainage.py::test_uniform_column_follows_the_kinematic_wave_rarefaction` | Analytical solution of a kinematic wave | A column of 50 and 100 nodes drains on an impermeable base. | 2026-11-24 |

## One node drains as the front scheme's parcel

**Why this check exists.** The layer scheme and the front scheme use the same drainage law, so on the same node they must drain the same melt. A wrong percolation speed, area or buoyancy factor in the layer scheme would make it drain faster or slower than the front scheme, and the test fails.

**What it tests.** One node 60 km thick holds a melt of 1 Pa s at $\psi_0 = 0.3$. Its matrix time is set to zero, so the permeability alone limits the flow. The node drains for 5 percolation times $L/w_D(\psi_0)$, under each setting of `interior_energetics.aragog.separation_buoyancy`: `"mixture"` and `"solid"`. At 10 times, its pore melt must equal what the front scheme's integral gives, $d\psi/dt = -\psi w_D / L$.

**How we know the right answer.** The front scheme integrates the same equation with an independent ODE solver (LSODA) at a relative tolerance of 1e-8. It is checked on [Melt drainage across the freezing front](compaction.md).

**Why it matters.** The node-by-node scheme is meant to reduce to the front scheme for a single node. This case shows that the two schemes share one physics, so a difference between them in a run comes from how they treat the mush, not from the drainage law.

![Pore melt of one node of the layer drainage against the front scheme's drainage integral](../../assets/validation/layer_drainage_single_node.svg#only-light){ width="95%" }
![Pore melt of one node of the layer drainage against the front scheme's drainage integral](../../assets/validation/layer_drainage_single_node_dark.svg#only-dark){ width="95%" }

**How to read the plot.** The left panel shows $\psi/\psi_0$ against time in units of the node's percolation time under each buoyancy setting. Lines are the front scheme, circles the layer scheme. The right panel shows their relative difference; the shaded band is the tolerance. A pass means that every circle lies on its line and every difference lies in the band. The largest difference is 3.3e-3, the error of the second-order sub-steps.

**Known limit.** This case covers the percolation branch only. On the matrix branch a single node does not match the front: its matrix time uses the depth of its centre, $L/2$, so it is twice the front's. The next case checks the matrix branch against its own analytical solution.

??? info "Test details"
    - Reference: `proteus.outgas.compaction.drainage_integral`, the front scheme's LSODA integral at `rtol=1e-8`.
    - Tolerance: relative 5e-3; the melt that left the node equals what it lost to relative 1e-12.
    - Test id: `tests/outgas/test_layer_drainage.py::test_single_node_reproduces_the_front_drainage_integral`
    - Command: `pytest -m reference_pinned tests/outgas/test_layer_drainage.py --validation-plots`

## A node limited by its crystal framework loses melt exponentially

**Why this check exists.** Where the crystal framework compacts more slowly than melt can flow, the matrix time alone sets the rate, and the node loses the share $1/\tau_s$ of its melt per second. A wrong matrix time, or a first-order time step, would drain the node too fast or too slow. A forward-Euler step at the same sub-step would end 15 % low after 3 matrix times, and the test requires more than 10 %.

**What it tests.** The permeability of one node is made so large that it never limits the flow. The node starts at $\psi_0 = 0.3$ and drains for up to 3 matrix times, in 30 sub-steps of a tenth of $\tau_s$. At 10 times its pore melt must equal $\psi_0 e^{-t/\tau_s}$.

**How we know the right answer.** With a constant matrix time, $d\psi/dt = -\psi/\tau_s$ has the exact solution $\psi_0 e^{-t/\tau_s}$.

**Why it matters.** Deep in a thick mush the matrix time is the slower of the 2 times, so this branch decides how much melt the deep mush keeps.

![Pore melt of a node limited by its crystal framework against the exponential decay](../../assets/validation/layer_drainage_compaction_limited_node.svg#only-light){ width="80%" }
![Pore melt of a node limited by its crystal framework against the exponential decay](../../assets/validation/layer_drainage_compaction_limited_node_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The plot shows $\psi/\psi_0$ on a log scale against time in units of $\tau_s$. The line is the exponential, the circles are the layer scheme, and the crosses are a forward-Euler step at the same sub-steps, shown for context. A pass means that every circle lies on the line. The largest difference, 5.4e-3 at 3 matrix times, is the expected error of the second-order sub-steps.

??? info "Test details"
    - Reference: the analytical decay $\psi_0 e^{-t/\tau_s}$.
    - Tolerance: relative 7e-3; the second-order error is $30 h^3/6 = 5.0$e-3 at $h = 0.1$.
    - Test id: `tests/outgas/test_layer_drainage.py::test_compaction_limited_node_decays_exponentially`
    - Command: `pytest -m reference_pinned tests/outgas/test_layer_drainage.py --validation-plots`

## A draining column follows the kinematic wave

**Why this check exists.** In a column, the melt each node passes on is what the node above receives. A scheme that loses melt between nodes, or moves it in the wrong order, would drain a column at the wrong rate. The test fails then, because the melt left in the column moves away from the exact solution.

**What it tests.** A column 100 km thick starts at $\psi_0 = 0.3$ in every node, on a base that lets no melt through. The melt flux is $K\psi^3$, with a constant $K$, and nothing else limits the flow. Until the drained zone from the base reaches the top, at $t_a = H/(3K\psi_0^2)$ with $H$ the column height, the melt left is $M/M_0 = 1 - K\psi_0^2 t/H$. After that time it is $M/M_0 = \tfrac{2}{3}\sqrt{t_a/t}$. The column runs on 50 and 100 nodes.

**How we know the right answer.** A uniform column with this flux obeys the scalar conservation law $\partial\psi/\partial t + \partial(K\psi^3)/\partial z = 0$. Its solution behind the base is the rarefaction $\psi(z, t) = \sqrt{z/(3Kt)}$.

**Why it matters.** A real mush is many nodes deep. This case shows that melt passes through the nodes above it at the right speed, and that the result converges as the mesh gets finer.

![Melt left in a uniform column against the kinematic-wave rarefaction](../../assets/validation/layer_drainage_kinematic_wave.svg#only-light){ width="80%" }
![Melt left in a uniform column against the kinematic-wave rarefaction](../../assets/validation/layer_drainage_kinematic_wave_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The plot shows the melt left in the column, $M/M_0$, against time in units of $t_a$; the dotted line marks $t_a$. The line is the exact solution. Circles and squares are the layer scheme on 100 and 50 nodes. A pass means that the points lie on the line before $t_a$, and within 3 % of it at $4 t_a$ on 100 nodes. After $t_a$ both meshes keep a little more melt than the exact solution, the coarser mesh more, because the scheme smears the drained zone.

??? info "Test details"
    - Reference: the rarefaction $\psi(z, t) = \sqrt{z/(3Kt)}$ of the scalar conservation law.
    - Tolerance: relative 1e-6 up to $0.5\,t_a$; 3 % at $4\,t_a$ on 100 nodes, where the 100-node result must lie between the exact solution and the 50-node result.
    - Test id: `tests/outgas/test_layer_drainage.py::test_uniform_column_follows_the_kinematic_wave_rarefaction`
    - Command: `pytest -m reference_pinned tests/outgas/test_layer_drainage.py --validation-plots`

## Rules checked alongside

The drainage tests also check these rules, which have no published reference:

- the melt of a column plus what left through its top is conserved, and every pore melt stays between 0 and 1;
- the volatiles leave each node at the concentration of the melt it holds, so a column of uniform concentration keeps it as it drains;
- pore melt beyond the solver's own melt fraction of the node crystallises in place: its crystals keep $D$ times the melt's concentration, with $D$ the crystal/melt partition coefficient, and the melt left carries the rest of the volatiles, so a node drained for long keeps the crystals' share alone;
- melt that crystallised stays crystalline, so a node drained over one step or over two halves ends with the same pore melt and volatiles;
- a node receives from below no more than it passes on plus its free pore space, the solver's melt fraction less its pore melt, so no node fills beyond it;
- a nearly closed node holds the melt below it without shortening the sub-steps: a column of 10 nodes under a top node of porosity 5e-4 drains a step of 1e11 s in fewer than 1000 sub-steps;
- a node whose density tables put the melt at or above the solid density holds itself and the mush below it.

## The drainage equation

The melt volume flux out of the top of node $k$ of a mush layer, counted from its base, is

$$
q_k = \min\left(A_k\,\psi_k\,w_D(\psi_k),\; q_{k-1} + \frac{\psi_k V_k}{\tau_{s,k}}\right), \qquad q_\mathrm{base} = 0 .
$$

For a single node with $\tau_s = 0$ the second term never limits the flow, and the equation is $d(\psi V)/dt = -\psi A w_D$, the front scheme's percolation equation with $L = V/A$. That is the first cross-check. A uniform column with $w_D = K\psi^2$ and $\tau_s = 0$ is the scalar conservation law of the third case.
