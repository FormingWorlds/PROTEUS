# Quick start: all-dummy run

This tutorial runs PROTEUS with all modules set to "dummy" backends. No
external solvers (AGNI, SPIDER, SOCRATES) are needed; the run completes
in under a minute and exercises the full coupling architecture. Use this
to verify your installation and understand the code flow before moving
to production runs.

## Prerequisites

- PROTEUS installed with `bash install.sh` (see the
  [installation guide](../How-to/installation.md)); the installer sets
  `FWL_DATA` in the `proteus` conda environment

No external solvers, spectral files, or EOS data are required.

## The configuration file

PROTEUS ships with an all-dummy config at `input/dummy.toml`. The key
settings are:

- **Planet**: 1 M$_\oplus$, starting fully molten (T$_\mathrm{magma}$ = 4000 K,
  $\Phi$ = 1). Volatile budget of 10,000 ppmw H, 1000 ppmw C, 500 ppmw N,
  500 ppmw S.
- **Star**: fixed solar luminosity (no evolution)
- **Orbit**: 0.5 AU, no tidal heating
- **Interior structure**: Noack & Lasbleis (2020) [^cite-noack2020] analytical
  scaling laws
- **Interior energetics**: heat-capacity integrator with prescribed solidus
  (1700 K) and liquidus (2700 K)
- **Outgassing**: melt-fraction-dependent partitioning; 10% of volatiles are
  always in the atmosphere (finite solubility floor), with the atmospheric
  fraction increasing as the mantle solidifies
- **Atmosphere**: grey-body opacity ($\gamma$ = 0.5)
- **Escape**: disabled (rate = 0), so the run reaches solidification
- **Chemistry**: parameterised vertical profiles (offline)

The simulation terminates when the global melt fraction drops below the
solidification threshold.

## Running the simulation

```bash
conda activate proteus
proteus start --offline -c input/dummy.toml
```

The `--offline` flag skips data downloads. The run takes less than a minute,
including the plots.

## Expected output

The run creates a directory inside `output/` named with a timestamp
(`run_<timestamp>_xxxx`, called `<run folder>` below). Check
`output/<run folder>/plots/plot_global_lin.png` for a multi-panel overview.
Your output should look similar to this. A run writes its plots in the default
matplotlib style; the figures of the tutorial pages show the same plots in the
PROTEUS visual language, in a light and a dark variant.

<figure markdown="span">
  ![Dummy tutorial output](../assets/tutorials/dummy_global_lin.avif#only-light){ width="100%" }
  ![Dummy tutorial output](../assets/tutorials/dummy_global_lin_dark.avif#only-dark){ width="100%" }
  <figcaption><b>All-dummy tutorial output.</b> (a) Heat fluxes: the interior and
  atmospheric fluxes track each other as the planet cools; absorbed stellar
  flux (ASF) is constant. (b) Surface partial pressures: H<sub>2</sub>O dominates
  (~10<sup>4</sup> bar), with CO<sub>2</sub>, N<sub>2</sub>, and SO<sub>2</sub> as minor species; pressures
  increase as solidification forces dissolved volatiles into the atmosphere.
  (c) Surface temperature: monotonic cooling to ~1700 K (solidus).
  (d) Atmospheric mole fractions: H<sub>2</sub>O at ~95%, stable throughout.
  (e) Mantle evolution: melt fraction drops from 1 (fully molten) to ~0
  (solidified) over ~23,000 yr; the rheological front (orange) moves toward
  the surface as the melt fraction drops. (f) Volatile partitioning: dissolved fraction decreases from
  ~90% to ~0% as the melt fraction drops, transferring volatiles from the
  interior to the atmosphere.</figcaption>
</figure>

To regenerate these plots from your own output, pass the resolved config
that the run wrote into its folder:

```bash
proteus plot -c output/<run folder>/init_coupler.toml all
```

`input/dummy.toml` sets `params.out.path = "auto"`, so the plot command
with that file would look for a new run folder.

## Understanding the helpfile

Open `runtime_helpfile.csv` in the output directory to see the full time
series. Key columns:

| Column | Units | What to expect |
|--------|-------|----------------|
| `Time` | yr | Stays at 0 for the first 3 iterations (init stage), then advances to ~23,000 yr |
| `T_magma` | K | Decreases monotonically from 4000 to ~1700 |
| `Phi_global` | 1 | Drops from 1.0 to ~0.003, triggering the solidification stop |
| `P_surf` | bar | Increases from ~7,000 to ~70,000 as volatiles outgas |
| `F_atm` | W m$^{-2}$ | Net flux at the top of the atmosphere; decreases as the surface cools |
| `F_int` | W m$^{-2}$ | Interior heat flux; the dummy interior takes the `F_atm` of the step before, so it is a few percent above `F_atm` while the flux falls |
| `M_planet` | kg | Constant throughout (mass conservation) |

## What to look for

1. **Cooling and solidification**: `T_magma` decreases smoothly from 4000 K.
   When it crosses the solidus (~1700 K), `Phi_global` approaches zero and
   the run terminates with "Planet solidified!!".

2. **Outgassing**: as the melt fraction drops, volatiles transfer from the
   interior to the atmosphere. `P_surf` increases and the dissolved fraction
   in panel (f) decreases. This is the core coupling feedback that the
   production modules (CALLIOPE, Aragog) compute with full thermodynamics.

3. **Energy balance**: the net atmospheric flux `F_atm` (grey line in panel
   a, with the OLR in red just above it) and the interior flux (orange
   dashed) track each other. `F_int` is slightly above `F_atm` because the
   dummy interior takes the `F_atm` of the step before, and the flux falls
   with time. The absorbed stellar flux (blue dashed) is
   constant because the star is fixed.

4. **Mass conservation**: `M_planet` should remain constant within rounding.
   No atmospheric escape occurs in this configuration.

## Next steps

- **Vary the greenhouse effect**: increase `atmos_clim.dummy.gamma` toward
  1.0 to slow cooling (more opaque atmosphere traps more heat) or decrease
  it toward 0 for faster cooling (more transparent)
- **Enable escape**: set `escape.dummy.rate = 1e4` and
  `params.stop.escape.enabled = true` to see atmospheric mass loss
- **Change volatile inventory**: increase `H_budget` to 50,000 ppmw for a
  thicker steam atmosphere, or decrease it to 1,000 ppmw for a thinner one
  (~11,000 bar at the end of the run, against ~74,000 bar here). The
  solidification time stays the same, because the grey atmosphere uses a
  fixed greenhouse factor (see the
  [parameter grid tutorial](parameter_grid.md))
- **Move to production modules**: the [Earth analogue tutorial](earth_analogue.md)
  uses Aragog, Zalmoxis, CALLIOPE, and AGNI for a quantitatively meaningful
  simulation

---

**See also:** [Model description](../Explanations/model.md) | [Dummy modules](../Explanations/dummy_modules.md) | [Coupling loop](../Explanations/coupling_loop.md) | [Configuration reference](../Reference/config/params.md) | [Output format](../Reference/output.md)

 [^cite-noack2020]: Noack, L. & Lasbleis, M., *[Parameterisations of interior properties of rocky planets](https://doi.org/10.1051/0004-6361/202037723)*, Astronomy & Astrophysics, 638, A129, 2020. [SciX](https://scixplorer.org/abs/2020A%26A...638A.129N/abstract).
