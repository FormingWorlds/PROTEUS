# Solar System CHILI intercomparison

The CHILI (Coupled atmospHere Interior modeL Intercomparison) project is a
community benchmark that fixes shared initial and boundary conditions for
magma ocean evolution codes [^cite-lichtenberg2026]. Its first
intercomparison applies that protocol to the inner Solar System,
modelling the primordial magma oceans of Earth and Venus.

This tutorial reproduces the Solar System CHILI test suite with PROTEUS
and compares the result against six other coupled atmosphere-interior
models: GOOEY, NEONGOOEY, PACMAN, LINCS, MOAI, and PlanAtMO. The figures
below reproduce the intercomparison plots: each one overlays the current
PROTEUS run on the results submitted by every participating model. Those
submitted results, and the figure layouts they follow, are drawn from the
Solar System CHILI intercomparison paper (Nicholls et al. 2026)
[^cite-nicholls2026].

!!! info "CHILI data and code"
    The simulation output of every participating model is openly available
    in the CHILI repository on GitHub:
    [**github.com/projectcuisines/chili**](https://github.com/projectcuisines/chili).
    The plotting script used below downloads this data automatically, and
    you can also clone the repository yourself (see Step 2) to inspect or
    re-plot the submitted results.

## Overview

The CHILI intercomparison defines three solar system test cases:

| Case | Planet | Key difference from Earth |
|------|--------|--------------------------|
| Nominal Earth | 1 M$_\oplus$ at 1 AU | Baseline case |
| Nominal Venus | 0.815 M$_\oplus$ at 0.723 AU | Higher instellation |
| Earth grid | 3 $\times$ 3 H/C inventory variations | Volatile sensitivity |

All cases start fully molten at 50 Myr stellar age (TRAPPIST-1 &alpha;
at 1 Gyr) with BSE composition, fO$_2$ = IW+4, and Bond albedo = 0.1. The
protocol asks for a run until solidification, or for at most 1 Gyr, for
the Earth and Venus cases [^cite-lichtenberg2026]. The configs in
`input/chili/intercomp/` and `input/tutorials/chili_grid/` stop a run
when its melt fraction falls to 5% by mass
(`params.stop.solid.phi_crit = 0.05`) or after 1 Gyr of model time
(`params.stop.time.maximum = 1e9`), with the energy-balance, escape and
wall-clock stops off (`params.stop.radeqm.enabled`,
`params.stop.escape.enabled` and `params.stop.clock.enabled` are
`false`). Two limits stay on: the stall stop ends a run as an error
after 150 consecutive iterations without a converged atmosphere solve
(`params.stop.stall.maximum`), and the iteration limit ends it after
1,000,000 iterations (`params.stop.iters.maximum`). The runs submitted to
the intercomparison stop at 5% melt as well, with the energy-balance and
escape stops off. Melt fractions of the
current PROTEUS runs are by mass on this page unless stated otherwise,
and the figures show the melt fractions of the intercomparison data by
volume.

!!! note "How far the runs on this page go"
    The figures and numbers on this page come from the runs as they
    were on 16 October 2026 at 05:30 CEST. The Nominal Earth run, the six low-H and mid-H runs of
    the Earth grid and the TRAPPIST-1 e run reached the 5% melt threshold.
    They ran with the energy-balance, escape and wall-clock stops on and a
    time limit of 5 Gyr (7.55 Gyr for TRAPPIST-1 e), and the 5% melt stop
    ended each of them. They still cool at that point: their net atmospheric flux is
    27 to 692 W m$^{-2}$, against about 0.2 W m$^{-2}$ of radiogenic
    heating. The Nominal Venus run, the three high-H runs of the Earth grid and the
    TRAPPIST-1 b run had not reached it at that time and are shown to their last
    row: Venus at 48.3 Myr with 6.0% of melt, the grid runs at 38.3 to 40.9 Myr with 6.2 to 7.1%,
    and TRAPPIST-1 b at 39.2 Myr with 13.7%. The TRAPPIST-1 &alpha; run was stopped by hand at 12.1 Myr, before it reached 5% melt or 1 Gyr, with 55.7% of
    melt: over its last 5 Myr its melt fraction decreased by about 0.002
    percentage points per Myr, and its net atmospheric flux was 0.24 W m$^{-2}$. The six runs that had not reached 5% melt ran with the
    energy-balance, escape and wall-clock stops on up to the end of their first
    part, at 3.1 to 15.0 Myr, and were resumed from their last snapshot
    under the stop rule above. The runs used the Aragog convective switch at
    $\partial S/\partial r < 0$ (`convective_threshold = 0`); the configurations
    on this page use the default threshold of $10^{-6}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$,
    which drives unstable nodes by at least that gradient and gives the band just
    above $\partial S/\partial r = 0$ the eddy-diffusivity floor. The runs do not
    follow the long-term
    solid-state evolution of the mantle or the loss of the atmosphere
    after their end. The PROTEUS results in the CHILI paper and the
    results shown here come from different versions of the code and from
    different sets of sub-modules (see
    [Current PROTEUS configuration vs the CHILI submission](#current-proteus-configuration-vs-the-chili-submission)).

## Prerequisites

- Full PROTEUS installation (see [Installation](../How-to/installation.md))
- AGNI, SOCRATES, and all reference data
- Spectral files downloaded (`proteus get spectral -n Dayspring -b 48`)
- Solar spectrum downloaded (`proteus get stellar`)
- Interior data downloaded, including the PALEOS EOS tables
  (`proteus get interiordata --config-path input/chili/intercomp/earth.toml`)
- `git` (to clone the CHILI comparison data)
- The runs on this page that reached 5% melt took about 3 hours to about a day of wall
  time; the others had used up to about 2.9 days when the data were copied.
  A run that stays partly molten takes steps of at most 4000 yr (`params.dt.mushy_maximum`),
  so at least 250,000 steps to reach 1 Gyr

The commands below run with `--offline`, which skips the automatic
reference-data check, so all data must be on disk beforehand. Drop the
flag to let the first run download missing data instead.

## Step 1: Run the nominal cases

```bash
conda activate proteus

# Earth (see also the Earth analogue tutorial for detailed analysis)
mkdir -p output/chili_earth
nohup proteus start --offline -c input/chili/intercomp/earth.toml \
    > /tmp/proteus_earth_launch.log 2>&1 & disown

# Venus
mkdir -p output/chili_venus
nohup proteus start --offline -c input/chili/intercomp/venus.toml \
    > /tmp/proteus_venus_launch.log 2>&1 & disown
```

Monitor progress with `tail -f output/chili_earth/proteus_00.log`
(the log appears once PROTEUS has initialized).

!!! info "Intercomparison configs"
    `tools/chili_generate.py` derives the full intercomparison set
    under `input/chili/intercomp/` from the tutorial configs (nominal
    Earth and Venus, TRAPPIST-1 b/e/&alpha;, and the H/C inventory grid
    specs). A nominal case equals its tutorial config in every field
    except the output path, the redox state, which the generator sets to
    the protocol value IW+4, and the stop rule above, which the generator
    sets for every case; unit tests keep the two in lockstep.
    TRAPPIST-1 &alpha; is the extreme case of the protocol, a fictitious
    lava world at 6.75 $\times$ 10$^{-4}$ AU, inside the Roche limit of
    the star [^cite-lichtenberg2026]. PROTEUS runs this case and warns
    during the run that the planet orbits within the Roche limit, spins
    faster than the breakup rate, and has an atmosphere beyond the Hill
    radius. The TRAPPIST-1 configs set the escape module to `none`. In
    the runs shown here, TRAPPIST-1 &alpha; was stopped by hand at 12.1 Myr with 55.7% of melt by
    mass, TRAPPIST-1 e reached the 5%
    threshold at 9.0 Myr, and TRAPPIST-1 b had 13.7% of melt by mass at
    its last row, at 39.2 Myr.
    After a run, convert the output to the CHILI-MIP submission format
    with `python tools/chili_postproc.py <output_dir>`.

## Step 2: Download comparison data

Clone the CHILI repository to access results from the other codes:

```bash
git clone https://github.com/projectcuisines/chili.git /tmp/chili
```

## Step 3: Generate comparison plots

```bash
# Nominal cases only
python tools/plot_chili_comparison.py \
    --proteus-earth output/chili_earth/ \
    --proteus-venus output/chili_venus/ \
    --chili-repo /tmp/chili \
    --output output_files/chili_plots/

# With the Earth volatile grid (after running the grid cases)
python tools/plot_chili_comparison.py \
    --proteus-earth output/chili_earth/ \
    --proteus-venus output/chili_venus/ \
    --grid-dir output/ \
    --chili-repo /tmp/chili \
    --output output_files/chili_plots/
```

`tools/plot_chili_comparison.py` writes every figure on this page to the
`--output` directory (here `output_files/chili_plots/`, which is
gitignored) as both PDF (vector) and PNG. The colours follow the CHILI
analysis code: six of the seven models are in Paul Tol's bright scheme, and
PROTEUS has the orange of that code. The gases are in the PROTEUS gas
colours; H<sub>2</sub>S, O<sub>2</sub> and the elements H and C of Figure 4
take the colours of the CHILI analysis code. The script draws under the active matplotlib
style, and `--style light` or `--style dark` applies the PROTEUS plot style
of the `proteus-mpl` package for a direct call. The figures of this page
come from `tools/make_docs_figures.py`, which calls the script once per
figure and style. The script is version-general:

- Run from the command line, it clones the CHILI comparison data
  automatically if `--chili-repo` does not already point to a checkout, so
  Step 2 is optional. `tools/make_docs_figures.py` needs an existing
  checkout, given with its own `--chili-repo`.
- It reads the git commit of the runs from their logs (`proteus_00.log`
  of the Earth run, the Venus run and the grid runs) and puts it in the
  label of the current PROTEUS curves, so every figure records the code
  version that produced the runs. Runs of different commits give a label
  with each commit, separated by `/`. A run folder without a log, or a
  log without a commit, adds `unknown` to the label.
- `--proteus-earth` and `--proteus-venus` are optional; omit either to
  plot the intercomparison models alone.

Up to two PROTEUS results appear on each plot. The results submitted to the
intercomparison (Nicholls et al. 2026) are labelled "PROTEUS CHILI". In the
figures with curves they are a thin orange line, dashed except in Figure 1,
where the line style tells the planets apart; in Figures 3, 4 and 5 they
are a bar. The current run is drawn in the same orange with thick lines
and edged markers, or as a bar with an outline and a bold label, and
carries the commit of the runs in its label. Running the command on the output of a future PROTEUS
version draws every figure with the commit of those runs, so the
comparison stays reproducible without editing the script.

## Melt fraction evolution

<figure markdown="span">
  ![CHILI Fig 1](../assets/tutorials/chili/fig1_melt_fraction.avif#only-light){ width="100%" }
  ![CHILI Fig 1](../assets/tutorials/chili/fig1_melt_fraction_dark.avif#only-dark){ width="100%" }
  <figcaption><strong>Figure 1.</strong> Melt fraction vs time for the CHILI Nominal Earth (solid lines) and Nominal Venus (dashed lines) cases. In the Earth files of the intercomparison the first melt fraction after t = 0 is 0.93 to 1.00. Six of the seven models reach a 5% melt fraction between 0.011 Myr (PACMAN) and 2.0 Myr (NEONGOOEY); the submitted PROTEUS CHILI run ends at 0.47 Myr with 6.1%. The Venus curves of PROTEUS CHILI and LINCS end at 39.3 and 49.3 Myr. The curves of the current PROTEUS runs show the melt fraction by mass; the curves of the intercomparison data show it by volume. The current PROTEUS Earth run reaches a melt fraction of 5% by mass at 2.44 Myr (6.2% by volume). The current PROTEUS Venus run had not reached a 5% melt fraction at its last row: its melt fraction decreases by about 0.02 percentage points per Myr over its last 5 Myr, and its curve ends at 48.3 Myr with 6.0% of melt by mass (7.4% by volume).</figcaption>
</figure>

## Solidification milestones

<figure markdown="span">
  ![CHILI Fig 2](../assets/tutorials/chili/fig2_milestones.avif#only-light){ width="100%" }
  ![CHILI Fig 2](../assets/tutorials/chili/fig2_milestones_dark.avif#only-dark){ width="100%" }
  <figcaption><strong>Figure 2.</strong> Time to reach melt fraction milestones for all Earth scenarios. (a) 95%, (b) 40%, (c) 5% melt fraction. The y-axis spans H inventories from the Nominal case (bottom) to H<sub>high</sub> (10 EO, top). C inventory is encoded as marker opacity (faint = C<sub>low</sub>, medium = C<sub>mid</sub>, full = C<sub>high</sub>). Connected scatter points trace the three H levels for each model at a given C level. The current PROTEUS run (orange, thick solid lines, edged markers) stands out from the CHILI intercomparison ensemble; the submitted PROTEUS CHILI runs have the same orange with thin dashed lines. Nominal Earth cases appear as crosses at the bottom of each panel, and the larger orange cross is that of the current run. A current PROTEUS run that has not reached a milestone at its last row has no marker in that panel. The submitted PROTEUS CHILI runs end above 5% melt by volume, and their points in panel (c) mark the last archived row.</figcaption>
</figure>

## Atmospheric composition

<figure markdown="span">
  ![CHILI Fig 3](../assets/tutorials/chili/fig3_atm_composition.avif#only-light){ width="100%" }
  ![CHILI Fig 3](../assets/tutorials/chili/fig3_atm_composition_dark.avif#only-dark){ width="100%" }
  <figcaption><strong>Figure 3.</strong> Atmospheric compositions for the Nominal Earth case at (a) 95% and (b) 5% melt fraction. Stacked bars show gas partial pressures &#91;bar&#93; for each model; grey stars mark surface temperature (right axis), with an orange star for the current run. The current PROTEUS run (bold label, outlined bar) is placed next to the original CHILI submission for direct comparison. At 95% melt fraction the largest gas is CO<sub>2</sub> in the NEONGOOEY, PROTEUS CHILI and PlanAtMO bars (87 to 93% of the summed pressure), CO in the PACMAN and MOAI bars, and H<sub>2</sub>O in the GOOEY and LINCS bars, which hold no carbon gas. The bar of the current PROTEUS run is 58% O<sub>2</sub> (129 bar of 222 bar), with 84 bar of CO<sub>2</sub>. O<sub>2</sub> is drawn where a file gives it: at the plotted row it is below 0.5 bar in five intercomparison files, and the LINCS and PlanAtMO files give no value. At 5% the largest gas is H<sub>2</sub>O in six bars, with 346 bar of 411 bar for the current PROTEUS run, and CO<sub>2</sub> in the PACMAN and MOAI bars. Both panels share the same y-axis range to highlight the pressure increase during solidification.</figcaption>
</figure>

## H and C mass budgets

<figure markdown="span">
  ![CHILI Fig 4](../assets/tutorials/chili/fig4_mass_budgets.avif#only-light){ width="100%" }
  ![CHILI Fig 4](../assets/tutorials/chili/fig4_mass_budgets_dark.avif#only-dark){ width="100%" }
  <figcaption><strong>Figure 4.</strong> Hydrogen (green) and carbon (red) mass budgets at 5% melt fraction, distributed across three reservoirs: (a) outgassed to atmosphere, (b, dotted) dissolved in the remnant magma ocean, (c, hatched) stored in solidified mantle. The current PROTEUS run (bold label, edged bars) is placed next to the original CHILI submission. GOOEY and LINCS do not simulate carbon. MOAI and PACMAN store significant H in the solid mantle, while most other models retain H in the atmosphere or melt.</figcaption>
</figure>

## Venus atmospheric composition

<figure markdown="span">
  ![CHILI Fig 5](../assets/tutorials/chili/fig5_venus_atm.avif#only-light){ width="100%" }
  ![CHILI Fig 5](../assets/tutorials/chili/fig5_venus_atm_dark.avif#only-dark){ width="100%" }
  <figcaption><strong>Figure 5.</strong> Atmospheric composition for the Nominal Venus case at 5% melt fraction. Stacked bars show gas partial pressures &#91;bar&#93;; grey stars mark surface temperature (right axis). A current PROTEUS run is drawn as an outlined bar with a bold label next to the submitted PROTEUS CHILI result, at its first row with a melt fraction of 5% or less by mass, or else of 6% or less. The current run has no such row, so the figure has no bar for it: at 48.3 Myr, its last row, it has a melt fraction of 6.03% by mass (7.4% by volume), a surface temperature of 1789 K and a surface pressure of 349 bar (298 bar H<sub>2</sub>O and 45 bar CO<sub>2</sub>). The bar of the submitted PROTEUS CHILI result is its last row, at a melt fraction of 6.2% by volume (4.9% by mass). The bar of each intercomparison model is its row closest to 5% by volume among the rows at or below 7%; PlanAtMO has no such row, so its bar is its row of lowest melt fraction, its last row, at 16.4% by volume and 0.084 Myr.</figcaption>
</figure>

## Oxygen fugacity

<figure markdown="span">
  ![CHILI Fig 6](../assets/tutorials/chili/fig6_fO2_vs_T.avif#only-light){ width="100%" }
  ![CHILI Fig 6](../assets/tutorials/chili/fig6_fO2_vs_T_dark.avif#only-dark){ width="100%" }
  <figcaption><strong>Figure 6.</strong> Oxygen fugacity of the Nominal Venus simulations of the four intercomparison models whose files give a value (GOOEY, PROTEUS CHILI, PACMAN, MOAI) and of the current PROTEUS run, plotted as a function of degassing temperature. The intercomparison curves are drawn against the surface temperature. (a) Absolute fO<sub>2</sub> compared to the iron-wustite buffer parameterizations of Fischer et al. (2011, dotted) and O'Neill & Eggins (2002, dashed). (b) Relative fO<sub>2</sub> as delta-IW referenced to O'Neill+02. Circular markers indicate, for each intercomparison model that reaches a melt fraction of 7% or less, its row closest to 5%. The curve of the current PROTEUS run (orange, thick line) is the O<sub>2</sub> partial pressure of the run, which CALLIOPE sets equal to the oxygen fugacity, against the magma temperature at which the run degasses; it equals IW+4 on the buffer of Fischer et al. (2011) at that temperature. Referenced to O'Neill & Eggins (2002) in panel (b) it lies at 5.2 at 3300 K, where the axis starts, and at 4.0 at 1789 K, its last row; the hotter rows of the run, up to 4272 K, lie before the axis. It has no circular marker: a current run is marked at its first row with a melt fraction of 5% or less by mass, or else of 6% or less, and the last row of this run has 6.03%. The orange circle at 1692 K is the marker of the submitted PROTEUS CHILI run, whose thin dashed curve has the colour of the current run. In panel (b) the GOOEY curve lies below the buffer between 3299 and 3182 K, down to -0.20, and the MOAI curve between 2968 and 2942 K, down to -0.06. The legend below the panels holds for both.</figcaption>
</figure>

## Volatile retention

<figure markdown="span">
  ![CHILI Fig 7](../assets/tutorials/chili/fig7_volatiles.avif#only-light){ width="100%" }
  ![CHILI Fig 7](../assets/tutorials/chili/fig7_volatiles_dark.avif#only-dark){ width="100%" }
  <figcaption><strong>Figure 7.</strong> Relative amounts of the initial (a) hydrogen and (b) carbon inventories retained in the Nominal Venus case as a function of simulation time, with atoms lost from the planet by hydrodynamic escape. Crosses mark 95% melt fraction and circles mark, for each intercomparison model that reaches a melt fraction of 7% or less, its row closest to 5%. The submitted PROTEUS CHILI run (thin, dashed) extends to 39 Myr, where its melt fraction is 6.2% by volume (4.9% by mass) and it retains 63% of its hydrogen and 28% of its carbon. The curve of the current PROTEUS run (thick, solid, labelled with the commit of the runs) ends at its last row, at 48.3 Myr, where the run retains 87.5% of its hydrogen and 75.8% of its carbon; it has no circle, because its melt fraction there, 6.03%, is above 6%, and the figure makes no statement on the escape after that time.</figcaption>
</figure>

## Outgoing longwave radiation

<figure markdown="span">
  ![CHILI Fig 8](../assets/tutorials/chili/fig8_olr.avif#only-light){ width="100%" }
  ![CHILI Fig 8](../assets/tutorials/chili/fig8_olr_dark.avif#only-dark){ width="100%" }
  <figcaption><strong>Figure 8.</strong> Outgoing longwave radiation flux from Nominal Earth, plotted as a function of (a) melt fraction and (b) surface temperature. The melt fraction is by volume for the intercomparison models and by mass for the current PROTEUS run. In panel (a) the dashed line marks the absorbed stellar radiation (ASR = 208 W/m<sup>2</sup>) at 50 Myr stellar age with the CHILI protocol parameters. In panel (b) the dash-dot line marks the Simpson-Nakajima steam runaway limit (293 W/m<sup>2</sup>; Nakajima et al. 1992). OLR controls the cooling rate. The OLR of the current PROTEUS run decreases from 2.2 x 10<sup>5</sup> W/m<sup>2</sup> in its first row, at 1 yr, to 405 W/m<sup>2</sup> at a melt fraction of 5%; its absorbed stellar flux is 226 to 227 W/m<sup>2</sup>. The curve of the current run is a centred mean over 9 rows, which differs from a single row by about 8% at most.</figcaption>
</figure>

## Geodynamics diagnostics

<figure markdown="span">
  ![CHILI Fig 9](../assets/tutorials/chili/fig9_geodynamics.avif#only-light){ width="100%" }
  ![CHILI Fig 9](../assets/tutorials/chili/fig9_geodynamics_dark.avif#only-dark){ width="100%" }
  <figcaption><strong>Figure 9.</strong> Geodynamics diagnostics as functions of melt fraction for the Nominal Earth case; the melt fraction is by volume for the intercomparison models and by mass for the current PROTEUS run. (a) Surface temperature. (b) Rheological front radius in megameters; the dashed line marks the PROTEUS core-mantle boundary at 3.45 Mm (R<sub>core</sub>/R<sub>p</sub> = 0.49). (c) Effective mantle viscosity; the dashed line marks solid Earth mantle viscosity (5 x 10<sup>22</sup> Pa s), and the dotted line marks water STP viscosity (10<sup>-3</sup> Pa s). Current PROTEUS values are extracted from Aragog interior profiles at each timestep and drawn as a centred mean over 9 rows.</figcaption>
</figure>

## Surface pressure evolution

<figure markdown="span">
  ![CHILI P_surf](../assets/tutorials/chili/psurf_vs_time.avif#only-light){ width="100%" }
  ![CHILI P_surf](../assets/tutorials/chili/psurf_vs_time_dark.avif#only-dark){ width="100%" }
  <figcaption><strong>Figure 10.</strong> Surface pressure vs time for the Nominal Earth case of all models. The current PROTEUS run has 2.6x10<sup>4</sup> bar in its first rows (1 to 22 yr), before the start of the time axis, and 1.7x10<sup>4</sup> bar at 10<sup>-3</sup> Myr, where the axis starts. This early peak is almost entirely molecular oxygen. CALLIOPE, the outgassing and gas-speciation module, solves the equilibrium atmospheric composition at the imposed oxygen fugacity, and because O<sub>2</sub> is one of the equilibrium species its partial pressure equals that fugacity. The CHILI protocol asks for a final mantle redox state of IW+4; the PROTEUS configuration sets fO<sub>2</sub> to IW+4 on the iron-wustite buffer at the magma temperature throughout the run (`outgas.fO2_shift_IW = 4.0`). The buffer is steeply temperature-dependent, so at the magma temperature of the first row (4316 K) this gives fO<sub>2</sub>, and hence pO<sub>2</sub>, near 2.6x10<sup>4</sup> bar, 99.7% of the total surface pressure. The magnitude comes from two compounding factors: the +4 dex offset multiplies the buffer value by 10<sup>4</sup>, and the high temperature already raises the iron-wustite fO<sub>2</sub> itself to a few bar. As the magma ocean cools the buffer falls by several orders of magnitude: O<sub>2</sub> falls below half of the surface pressure at a surface temperature of 2878 K, below 10% at 2697 K, and below 1% at 2471 K. There the pressure has its minimum of 110 bar, 97 bar of it CO<sub>2</sub>, before it rises to 411 bar at a melt fraction of 5% as H<sub>2</sub>O exsolves from the crystallizing mantle. The submitted PROTEUS CHILI run has 124 bar of O<sub>2</sub> in its first row, at a surface temperature of 3125 K; the first row of the current run, at 1 yr, has a surface temperature of 3324 K and a magma temperature of 4316 K. Other models differ in the timing and magnitude of the pressure evolution, reflecting their volatile solubility and oxygen treatments.</figcaption>
</figure>

## Earth volatile grid

The CHILI Earth grid varies H and C inventories across 9
combinations to explore how volatile budgets control solidification
timescale:

| | C$_\mathrm{low}$ (1.36$\times$10$^{20}$ kg) | C$_\mathrm{mid}$ (2.73$\times$10$^{20}$ kg) | C$_\mathrm{high}$ (5.44$\times$10$^{20}$ kg) |
|---|---|---|---|
| **H$_\mathrm{low}$** (1.6$\times$10$^{20}$ kg) | 1 EO, low C | 1 EO, mid C | 1 EO, high C |
| **H$_\mathrm{mid}$** (7.8$\times$10$^{20}$ kg) | 5 EO, low C | 5 EO, mid C | 5 EO, high C |
| **H$_\mathrm{high}$** (16.0$\times$10$^{20}$ kg) | 10 EO, low C | 10 EO, mid C | 10 EO, high C |

Grid configs are in `input/tutorials/chili_grid/`. Run all 9 cases:

```bash
for cfg in input/tutorials/chili_grid/*.toml; do
    name=$(basename "$cfg" .toml)
    outdir="output/chili_grid_earth_${name#earth_}"
    mkdir -p "$outdir"
    nohup proteus start --offline -c "$cfg" \
        > "/tmp/proteus_grid_${name}.log" 2>&1 & disown
done
```

Check status of running grid cases:

```bash
for d in output/chili_grid_earth_*/; do
    printf "%-40s %s\n" "$(basename $d)" "$(cat $d/status 2>/dev/null || echo 'not started')"
done
```

!!! warning "Runtime"
    The high-H cases (10 Earth oceans) cool the slowest: under the thick steam atmosphere the OLR stays below
    300 W m$^{-2}$ once the melt fraction is 12% or lower, and the runs
    shown here had not reached 5% melt after 38.3 to 40.9 Myr of model time.

### Grid results

Time at which each grid run shown here first reaches a melt
fraction of 5% by mass:

| | C$_\mathrm{low}$ | C$_\mathrm{mid}$ | C$_\mathrm{high}$ |
|---|---|---|---|
| **H$_\mathrm{low}$** (1 EO) | 0.81 Myr | 0.84 Myr | 0.92 Myr |
| **H$_\mathrm{mid}$** (5 EO) | 5.55 Myr | 5.28 Myr | 4.89 Myr |
| **H$_\mathrm{high}$** (10 EO) | not reached (7.1%) | not reached (6.6%) | not reached (6.2%) |

The three high-H runs had not reached 5% melt at their last row, at
38.3 to 40.9 Myr; the table gives their melt fraction there.

The hydrogen inventory is the primary control on the solidification
time. From 1 to 5 EO the time to 5% melt rises by a factor of 5.3 to
6.8. At 10 EO no run had reached 5% melt after 38.3 to 40.9 Myr. The carbon
effect is secondary and changes sign. At low H, more carbon slows the
cooling (0.81 to 0.92 Myr): at a melt fraction of 30% the OLR falls
from 5525 to 4235 W m$^{-2}$ from low to high C. At mid H the order
reverses (5.55 to 4.89 Myr): at a melt fraction of 30% the OLR rises
from 1115 to 1323 W m$^{-2}$ from low to high C. The
high-H runs follow the order of the mid-H runs: at their last row the melt
fraction is lower for a higher carbon inventory.

<figure markdown="span">
  ![CHILI grid solidification](../assets/tutorials/chili/grid_solidification.avif#only-light){ width="80%" }
  ![CHILI grid solidification](../assets/tutorials/chili/grid_solidification_dark.avif#only-dark){ width="80%" }
  <figcaption><strong>Figure 11.</strong> Time at which the current PROTEUS grid runs reach 5% melt by mass (solid lines, filled markers) as a function of hydrogen inventory for the three carbon levels (opacity and marker shape), on logarithmic axes. The high-H runs had not reached this threshold at their last row (38.3 to 40.9 Myr) and have no marker. Where the submitted PROTEUS CHILI grid runs reach 5% melt by volume they are overlaid as dashed lines with open markers; the runs archived in the comparison repository stop just above that threshold, so no submitted points appear here. The carbon-inventory dependence reverses sign between the low-H and mid-H cases (see text).</figcaption>
</figure>

## Current PROTEUS configuration vs the CHILI submission

The figures show up to two PROTEUS results, and the gap between them is itself
informative because the two come from different model configurations. The
protocol fixes the inputs that both share: a bulk-silicate-Earth
composition, oxygen fugacity at IW+4, a Bond albedo of 0.1, and a 50 Myr
stellar age [^cite-lichtenberg2026]. The two differ in the interior
modules and in the limits of the stop rule: the submitted configs stop
after 4.517 Gyr (6.6 and 7.55 Gyr for the TRAPPIST-1 cases) or 9000
iterations, the configs here after 1 Gyr or 1,000,000 iterations.

The PROTEUS results submitted to the intercomparison were computed with an
earlier version of PROTEUS, not the configuration documented here. The
config file of the submitted Venus run is archived with the comparison
data (`intercomparison/inputs/proteus/evolution-proteus-venus-config.in`).
It selects **SPIDER** for the interior thermal evolution
(`interior.module = "spider"`) and the structure calculation built into
that version (`struct.module = "self"`), together with AGNI, CALLIOPE,
ZEPHYRUS and MORS. It names VULCAN for the atmospheric chemistry, set to
run only on request (`atmos_chem.when = "manually"`). Its energy balance
stop criterion is switched off. See
Lichtenberg et al. (2026) [^cite-lichtenberg2026] and Nicholls et al.
(2026) [^cite-nicholls2026] for further details.

The configuration documented on this page uses **Aragog** for the
interior thermal evolution and **Zalmoxis** for the interior structure,
with AGNI for the atmosphere, CALLIOPE for the outgassing, ZEPHYRUS for
the escape and MORS for the star; the three TRAPPIST-1 configs run
without an escape module (`escape.module = "none"`). Aragog is the Python reimplementation of
SPIDER and integrates the same specific-entropy equation for a two-phase,
partially molten mantle with a mixing-length convective closure. Zalmoxis
solves the layered interior (mass, radius, density profile, and
core-mantle-boundary radius) directly and supplies the pressure-entropy
equation-of-state tables that Aragog reads.

The configuration also departs from the protocol table in two inputs.
The protocol fixes a planet radius (1 R$_\oplus$ for Earth, 0.95
R$_\oplus$ for Venus) and a core radius fraction of 0.55
[^cite-lichtenberg2026]. Here Zalmoxis solves the structure from a core
mass fraction of 0.325, which gives a radius of 7.03 Mm (1.10
R$_\oplus$) for Earth and 6.63 Mm (1.04 R$_\oplus$) for Venus, each with
a core radius fraction of 0.49. The protocol sets the stellar irradiance
to 920 W m$^{-2}$ for Earth and 1760 W m$^{-2}$ for Venus; the runs take
it from the stellar evolution model, 1005 and 1923 W m$^{-2}$ at their
start.

### The Nominal Venus case

The Nominal Venus run had not reached the solidification criterion at
its last row, at 48.3 Myr, where its melt fraction is
0.060 by mass (0.074 by volume). The run submitted to the
intercomparison ends at 39 Myr with a melt fraction of 0.062 by volume
(0.049 by mass).

For the melt-solid separation in the interior module, the configuration
uses a grain size of 1 mm (`interior_energetics.grain_size = 1e-3`) and
the viscosity of the melt-solid mixture
(`interior_energetics.aragog.separation_viscosity = "mixture"`). In this
run the mantle crystallises as a mush with melt at all depths: at its last
row every node of the mantle still holds melt, between 2% and 34%.

## Takeaways

Earth and Venus begin from the same fully molten state and diverge as they
cool. The spread between codes at any given moment reflects
differences in how each treats atmospheric opacity, volatile partitioning,
and interior convection. PROTEUS couples a radiative-convective atmosphere,
equilibrium outgassing, and a 1D interior solver, and follows the magma
ocean from the molten state to the first stop criterion that a run meets,
while tracking the volatiles exchanged between the mantle and the
atmosphere. The runs of this page that had not reached 5% melt are shown
to their last row.

The point of this tutorial is to place the current PROTEUS against that
published ensemble on demand. The comparison figures are drawn from the
output of your own runs and labelled with the commit in the logs of those
runs, so the plotting command compares the PROTEUS version that made the
runs with the fixed literature values from the CHILI papers. It is therefore
an always-current intercomparison: a quick way to see how a new PROTEUS
release compares with the submitted Earth and Venus results, and where it
lies relative to the other community codes.

---

**See also:** [Earth analogue](earth_analogue.md) | [Model description](../Explanations/model.md) | [Output format](../Reference/output.md)

 [^cite-lichtenberg2026]: Lichtenberg, T., Schaefer, L., Krissansen-Totton, J., et al., *[Coupled atmospHere Interior modeL Intercomparison (CHILI): Protocol Version 1.0](https://doi.org/10.3847/PSJ/ae593b)*, The Planetary Science Journal, 7, 108, 2026. [SciX](https://scixplorer.org/abs/2026PSJ.....7..108L/abstract).

 [^cite-nicholls2026]: Nicholls, H., Krissansen-Totton, J., Lichtenberg, T., et al., *[Coupled atmospHere Interior modeL Intercomparison (CHILI). I. Evolutionary Modelling: Primordial Magma Oceans of Earth and Venus](https://doi.org/10.48550/arXiv.2606.24757)*, arXiv:2606.24757, submitted to The Planetary Science Journal, 2026.
