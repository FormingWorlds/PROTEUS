# CHILI intercomparison

This advanced tutorial reproduces the full CHILI (Coupled atmosHere
Interior modeL Intercomparison) test suite[^cite-lichtenberg2026] and
compares PROTEUS results against seven other coupled evolution codes.

## Overview

The CHILI intercomparison defines three solar system cases:

| Case | Planet | Key difference from Earth |
|------|--------|--------------------------|
| Nominal Earth | 1 M$_\oplus$ at 1 AU | Baseline case |
| Nominal Venus | 0.815 M$_\oplus$ at 0.723 AU | Higher instellation |
| Earth grid | 3 $\times$ 3 H/C inventory variations | Volatile sensitivity |

All cases start fully molten at 50 Myr stellar age with BSE composition,
fO$_2$ = IW+4, and Bond albedo = 0.1. Simulations run until the melt
fraction drops below 5%.

The intercomparison data from the other codes (GOOEY, NEONGOOEY, PACMAN,
LINCS, MOAI, PlanAtMO) is publicly available at the
[CHILI GitHub repository](https://github.com/projectcuisines/chili).

## Prerequisites

- Full PROTEUS installation (see [Installation](../How-to/installation.md))
- AGNI, SOCRATES, and all reference data
- `git` (to clone the CHILI comparison data)
- Allow 1-3 hours per run (Earth ~30 min, Venus ~1-3 hours)

## Step 1: Run the Nominal Earth case

```bash
proteus start --offline -c input/tutorial_earth.toml
```

This is the same run as the [Earth analogue tutorial](earth_analogue.md).
Monitor with `tail -f output/tutorial_earth/proteus_00.log`.

## Step 2: Run the Nominal Venus case

```bash
proteus start --offline -c input/tutorial_venus.toml
```

Venus has higher instellation (1760 vs 920 W m$^{-2}$), which delays
solidification. Some models predict Venus entering a "Type II"
radiative equilibrium state where the magma ocean is sustained for
tens of millions of years by the balance between stellar heating and
atmospheric escape.

## Step 3: Download comparison data

Clone the CHILI repository to access the comparison data from the other
codes:

```bash
git clone https://github.com/projectcuisines/chili.git /tmp/chili
```

## Step 4: Generate comparison plots

Use the CHILI comparison plotting script to overlay PROTEUS output on the
other codes:

```bash
python tools/plot_chili_comparison.py \
    --proteus-earth output/tutorial_earth/ \
    --proteus-venus output/tutorial_venus/ \
    --chili-repo /tmp/chili \
    --output output_files/chili_plots/
```

This generates the following plots, mirroring the figures from the CHILI
intercomparison paper[^cite-nicholls2026]:

### Melt fraction evolution (cf. CHILI Fig. 1)

Each model's melt fraction vs time, with Earth (solid) and Venus (dashed).
Your PROTEUS output is overlaid as a thick line on top of the comparison
data.

### Time to solidification milestones (cf. CHILI Fig. 2)

Time to reach $\Phi$ = 95%, 40%, and 5% for the Earth grid cases.
Connected scatter points show how cooling timescales vary with H and C
inventory.

### Atmospheric composition (cf. CHILI Figs. 3, 5)

Stacked bar charts of gas partial pressures at $\Phi$ = 95% and $\Phi$ = 5%
for nominal Earth and Venus, comparing all models side by side.

### Outgoing longwave radiation (cf. CHILI Fig. 7)

OLR as a function of melt fraction and surface temperature, comparing the
radiative-convective response of each atmosphere model.

### Geodynamics diagnostics (cf. CHILI Fig. 8)

Surface temperature, solidification radius, and mantle viscosity as
functions of melt fraction. Shaded envelopes show the spread across models.

## Step 5: Run the Earth volatile grid (optional)

The full CHILI Earth grid varies H and C inventories across 9 combinations:

| | C$_\mathrm{low}$ (1.36$\times$10$^{20}$ kg) | C$_\mathrm{mid}$ (2.73$\times$10$^{20}$ kg) | C$_\mathrm{high}$ (5.44$\times$10$^{20}$ kg) |
|---|---|---|---|
| **H$_\mathrm{low}$** (1.6$\times$10$^{20}$ kg) | 0.5 EO, low C | 0.5 EO, mid C | 0.5 EO, high C |
| **H$_\mathrm{mid}$** (7.8$\times$10$^{20}$ kg) | 5 EO, low C | 5 EO, mid C | 5 EO, high C |
| **H$_\mathrm{high}$** (16.0$\times$10$^{20}$ kg) | 10 EO, low C | 10 EO, mid C | 10 EO, high C |

To run the grid using PROTEUS's grid manager:

```bash
proteus grid -c input/tutorial_chili_grid.toml
```

This submits 9 runs in parallel (controlled by `max_jobs`). Each run
takes 30 minutes to several hours depending on the volatile inventory
(high-H cases take longest due to thicker atmospheres and slower cooling).

## Interpreting the results

Key findings from the CHILI intercomparison:

1. **All models predict Earth solidification within 4 Myr**, consistent with
   geological constraints from the oldest zircons.

2. **Venus solidification is model-dependent**: some models (LINCS, PROTEUS,
   PlanAtMO) predict Type II behaviour where Venus enters a prolonged
   radiative equilibrium. Others (GOOEY, PACMAN) predict direct cooling.

3. **Cooling timescales correlate with hydrogen inventory**: higher H budgets
   produce thicker, more opaque atmospheres that slow cooling.

4. **Model differences arise from**: gas chemistry (which species are
   tracked), volatile partitioning (solubility vs melt trapping),
   radiative transfer (mixing length vs boundary layer), and melting
   curve prescriptions.

---

**See also:** [Earth analogue](earth_analogue.md) | [Model description](../Explanations/model.md) | [Module versions](../Reference/module_versions.md) | [Output format](../Reference/output.md)

[^cite-lichtenberg2026]: Lichtenberg, T., Schaefer, L., Krissansen-Totton, J., et al., *[Coupled atmosHere Interior modeL Intercomparison (CHILI): Protocol Version 1.0](https://doi.org/10.3847/PSJ/ae593b)*, The Planetary Science Journal, 7, 108, 2026. [SciX](https://scixplorer.org/abs/2026PSJ.....7..108L/abstract).

[^cite-nicholls2026]: Nicholls, H. et al., *Coupled atmosHere Interior modeL Intercomparison (CHILI). I. Evolutionary Modelling of the Inner Solar System*, in preparation, 2026.
