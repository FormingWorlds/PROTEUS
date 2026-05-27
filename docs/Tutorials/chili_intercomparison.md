# CHILI intercomparison

The CHILI (Coupled atmosHere Interior modeL Intercomparison) is a
community benchmark for magma ocean evolution
codes[^cite-lichtenberg2026]. This tutorial reproduces the CHILI test
suite with PROTEUS and compares results against six other coupled
atmosphere-interior models: GOOEY, NEONGOOEY, PACMAN, LINCS, MOAI,
and PlanAtMO[^cite-nicholls2026].

## Overview

The CHILI intercomparison defines three solar system test cases:

| Case | Planet | Key difference from Earth |
|------|--------|--------------------------|
| Nominal Earth | 1 M$_\oplus$ at 1 AU | Baseline case |
| Nominal Venus | 0.815 M$_\oplus$ at 0.723 AU | Higher instellation |
| Earth grid | 3 $\times$ 3 H/C inventory variations | Volatile sensitivity |

All cases start fully molten at 50 Myr stellar age with BSE composition,
fO$_2$ = IW+4, and Bond albedo = 0.1. Simulations run until the melt
fraction drops below 5%.

## Prerequisites

- Full PROTEUS installation (see [Installation](../How-to/installation.md))
- AGNI, SOCRATES, and all reference data
- Spectral files downloaded (`proteus get spectral`)
- Solar spectrum downloaded (`proteus get stellar`)
- `git` (to clone the CHILI comparison data)
- Allow 30 min to several hours per run depending on hardware

## Step 1: Run the nominal cases

```bash
conda activate proteus

# Earth (see also the Earth analogue tutorial for detailed analysis)
mkdir -p output/tutorial_earth
nohup proteus start --offline -c input/tutorials/tutorial_earth.toml \
    > /tmp/proteus_earth_launch.log 2>&1 & disown

# Venus
mkdir -p output/tutorial_venus
nohup proteus start --offline -c input/tutorials/tutorial_venus.toml \
    > /tmp/proteus_venus_launch.log 2>&1 & disown
```

Monitor progress with `tail -f output/tutorial_earth/proteus_00.log`
(the log appears once PROTEUS has initialized).

## Step 2: Download comparison data

Clone the CHILI repository to access results from the other codes:

```bash
git clone https://github.com/projectcuisines/chili.git /tmp/chili
```

## Step 3: Generate comparison plots

```bash
# Nominal cases only
python tools/plot_chili_comparison.py \
    --proteus-earth output/tutorial_earth/ \
    --proteus-venus output/tutorial_venus/ \
    --chili-repo /tmp/chili \
    --output output_files/chili_plots/

# With the Earth volatile grid (after running the grid cases)
python tools/plot_chili_comparison.py \
    --proteus-earth output/tutorial_earth/ \
    --proteus-venus output/tutorial_venus/ \
    --grid-dir output/ \
    --chili-repo /tmp/chili \
    --output output_files/chili_plots/
```

All plots use the Wong colorblind-friendly palette and are saved as both
PDF (vector) and PNG. The previous PROTEUS submission to the CHILI intercomparison
appears as a thin black line ("PROTEUS CHILI"), while the current
PROTEUS run appears in vermillion with thick lines, black-edged markers,
and the git commit SHA in the legend.

## Melt fraction evolution

<figure markdown="span">
  ![CHILI Fig 1](../assets/tutorials/chili/fig1_melt_fraction.png){ width="100%" }
  <figcaption>Melt fraction vs time for the CHILI Nominal Earth (solid lines) and Nominal Venus (dashed lines) cases. All seven models start fully molten and solidify within 0.1 to 4 Myr for Earth. Venus solidifies later due to higher instellation at 0.723 AU. PROTEUS predicts solidification at 1.34 Myr for Earth and 2.22 Myr for Venus, within the model ensemble range. The spread among models reflects differences in atmospheric opacity, mantle convection treatment, and volatile partitioning.</figcaption>
</figure>

## Solidification milestones

<figure markdown="span">
  ![CHILI Fig 2](../assets/tutorials/chili/fig2_milestones.png){ width="100%" }
  <figcaption>Time to reach melt fraction milestones for all Earth scenarios. (a) 95%, (b) 40%, (c) 5% melt fraction. The y-axis spans H inventories from the Nominal case (bottom) to H<sub>high</sub> (10 EO, top). C inventory is encoded as marker opacity (light = C<sub>low</sub>, medium = C<sub>mid</sub>, dark = C<sub>high</sub>). Connected scatter points trace the three H levels for each model at a given C level. The current PROTEUS run (vermillion, thick lines, black-edged markers) stands out from the CHILI intercomparison ensemble. Nominal Earth cases appear as crosses at the bottom of each panel.</figcaption>
</figure>

## Atmospheric composition

<figure markdown="span">
  ![CHILI Fig 3](../assets/tutorials/chili/fig3_atm_composition.png){ width="100%" }
  <figcaption>Atmospheric compositions for the Nominal Earth case at (a) 95% and (b) 5% melt fraction. Stacked bars show gas partial pressures [bar] for each model; grey stars mark surface temperature (right axis). The current PROTEUS run (vermillion label, black-edged bar) is placed next to the original CHILI submission for direct comparison. At 95% melt fraction, atmospheres are CO<sub>2</sub>-dominated; by 5%, H<sub>2</sub>O has exsolved from the crystallizing mantle and dominates at ~368 bar for PROTEUS. Both panels share the same y-axis range to highlight the pressure increase during solidification.</figcaption>
</figure>

## H and C mass budgets

<figure markdown="span">
  ![CHILI Fig 4](../assets/tutorials/chili/fig4_mass_budgets.png){ width="100%" }
  <figcaption>Hydrogen (green) and carbon (red) mass budgets at 5% melt fraction, distributed across three reservoirs: (a) outgassed to atmosphere, (b, dotted) dissolved in the remnant magma ocean, (c, hatched) stored in solidified mantle. The current PROTEUS run (vermillion label, black-edged bars) is placed next to the original CHILI submission. GOOEY and LINCS do not simulate carbon. MOAI and PACMAN store significant H in the solid mantle, while most other models retain H in the atmosphere or melt.</figcaption>
</figure>

## Venus atmospheric composition

<figure markdown="span">
  ![CHILI Fig 5](../assets/tutorials/chili/fig5_venus_atm.png){ width="100%" }
  <figcaption>Atmospheric composition for the Nominal Venus case at 5% melt fraction. Stacked bars show gas partial pressures [bar]; grey stars mark surface temperature (right axis). The current PROTEUS run (vermillion label, black-edged bar) is placed next to the original CHILI submission. PROTEUS predicts ~371 bar H<sub>2</sub>O and ~63 bar CO<sub>2</sub> near solidification, for a total surface pressure of ~467 bar.</figcaption>
</figure>

## Oxygen fugacity

<figure markdown="span">
  ![CHILI Fig 6](../assets/tutorials/chili/fig6_fO2_vs_T.png){ width="100%" }
  <figcaption>Oxygen fugacity from each model's Nominal Venus simulation, plotted as a function of degassing temperature. (a) Absolute fO<sub>2</sub> compared to the iron-wustite buffer parameterizations of Fischer et al. (2011, dotted) and O'Neill & Eggins (2002, dashed). (b) Relative fO<sub>2</sub> as delta-IW referenced to O'Neill+02. Circular markers indicate 5% melt fraction. The current PROTEUS run (vermillion, thick line, black-edged marker) tracks along IW+4 as prescribed by the CHILI protocol.</figcaption>
</figure>

## Volatile retention

<figure markdown="span">
  ![CHILI Fig 7](../assets/tutorials/chili/fig7_volatiles.png){ width="100%" }
  <figcaption>Relative amounts of the initial inventories of (a) hydrogen and (b) carbon retained by the Nominal Venus scenario as a function of simulation time. Atoms are lost from the planet due to hydrodynamic escape. Crosses mark 95% melt fraction; circles mark 5% melt fraction. PROTEUS shows significant H loss over the extended Venus cooling timescale.</figcaption>
</figure>

## Outgoing longwave radiation

<figure markdown="span">
  ![CHILI Fig 8](../assets/tutorials/chili/fig8_olr.png){ width="100%" }
  <figcaption>Outgoing longwave radiation flux from Nominal Earth, plotted as a function of (a) melt fraction and (b) surface temperature. The dashed line marks the absorbed stellar radiation (ASR = 208 W/m<sup>2</sup>) at 50 Myr stellar age with the CHILI protocol parameters. The dash-dot line marks the Nakajima et al. (1992) pure-steam runaway limit (282 W/m<sup>2</sup>). OLR controls the cooling rate; PROTEUS OLR decreases from ~2 x 10<sup>5</sup> W/m<sup>2</sup> at full melt to ~466 W/m<sup>2</sup> at solidification.</figcaption>
</figure>

## Geodynamics diagnostics

<figure markdown="span">
  ![CHILI Fig 9](../assets/tutorials/chili/fig9_geodynamics.png){ width="100%" }
  <figcaption>Geodynamics diagnostics as functions of melt fraction for the Nominal Earth case. (a) Surface temperature. (b) Rheological front radius normalized to planet radius; the dashed line marks the core-mantle boundary at 0.55 R<sub>p</sub>. (c) Effective mantle viscosity; the dashed line marks solid Earth mantle viscosity (5 x 10<sup>22</sup> Pa s), and the dotted line marks water STP viscosity (10<sup>-3</sup> Pa s). PROTEUS values are extracted from Aragog interior profiles at each timestep.</figcaption>
</figure>

## Surface pressure evolution

<figure markdown="span">
  ![CHILI P_surf](../assets/tutorials/chili/psurf_vs_time.png){ width="100%" }
  <figcaption>Surface pressure vs time for all models. PROTEUS surface pressure starts high (~10<sup>4</sup> bar during the brief fully molten phase), drops to a minimum of ~117 bar as CO<sub>2</sub> partitions between atmosphere and melt, then rises to ~438 bar at solidification as H<sub>2</sub>O exsolves from the crystallizing mantle. Models differ in the timing and magnitude of the pressure evolution, reflecting different volatile solubility treatments.</figcaption>
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
    Low-H cases finish in ~1 hour. Mid-H cases take ~3-5 hours. High-H
    cases (10 Earth oceans) may take 12+ hours because the thick steam
    atmosphere reduces OLR to a few hundred W m$^{-2}$ in the late
    mushy zone.

### Grid results

Solidification times for the completed grid cases:

| | C$_\mathrm{low}$ | C$_\mathrm{mid}$ | C$_\mathrm{high}$ |
|---|---|---|---|
| **H$_\mathrm{low}$** (1 EO) | 0.49 Myr | 0.53 Myr | 0.61 Myr |
| **H$_\mathrm{mid}$** (5 EO) | 2.72 Myr | 2.55 Myr | 2.42 Myr |
| **H$_\mathrm{high}$** (10 EO) | TBD | TBD | TBD |

Hydrogen inventory is the primary control on solidification timescale:
a ~5x increase in H budget (1 to 5 EO) delays solidification by a
factor of ~5. The carbon effect is secondary and non-monotonic. At
low H, more CO$_2$ adds greenhouse opacity and slows cooling (0.49
to 0.61 Myr). At mid-H, the effect reverses: more CO$_2$ raises
P$_\mathrm{surf}$, which via Henry's law enhances H$_2$O dissolution
in the silicate melt, reducing the atmospheric H$_2$O greenhouse and
allowing higher OLR (2.72 to 2.42 Myr).

## Key findings

1. **Earth solidifies within 4 Myr across all models**, consistent with
   geological constraints from the oldest zircons. PROTEUS predicts
   1.34 Myr.

2. **Venus solidifies ~1.7x slower than Earth** due to the higher
   instellation at 0.723 AU. PROTEUS predicts 2.22 Myr for Venus vs
   1.34 Myr for Earth.

3. **Cooling timescales correlate with hydrogen inventory**: higher H
   budgets produce thicker, more opaque steam atmospheres that slow
   radiative cooling. The low-H grid cases (1 EO) solidify in ~0.5 Myr,
   while the high-H cases (10 EO) take several Myr.

4. **Model differences arise from**: gas chemistry (which species are
   tracked and whether equilibrium or kinetic), volatile partitioning
   (solubility laws), radiative transfer (correlated-k vs grey), mantle
   convection (mixing length vs parameterized), and melting curve
   prescriptions.

---

**See also:** [Earth analogue](earth_analogue.md) | [Model description](../Explanations/model.md) | [Output format](../Reference/output.md)

[^cite-lichtenberg2026]: Lichtenberg, T., Schaefer, L., Krissansen-Totton, J., et al., *[Coupled atmosHere Interior modeL Intercomparison (CHILI): Protocol Version 1.0](https://doi.org/10.3847/PSJ/ae593b)*, The Planetary Science Journal, 7, 108, 2026. [SciX](https://scixplorer.org/abs/2026PSJ.....7..108L/abstract).

[^cite-nicholls2026]: Nicholls, H. et al., *Coupled atmosHere Interior modeL Intercomparison (CHILI). I. Evolutionary Modelling of the Inner Solar System*, in preparation, 2026.
