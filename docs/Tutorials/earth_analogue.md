# Earth analogue

This tutorial simulates the thermal and atmospheric evolution of an
Earth-mass planet at 1 AU from a Sun-like star, reproducing the nominal
Earth case from the CHILI intercomparison[^cite-lichtenberg2026].

It uses the production-quality module combination:
[Aragog](https://proteus-framework.org/aragog/) (interior energetics),
[Zalmoxis](https://proteus-framework.org/Zalmoxis/) (interior structure),
[CALLIOPE](https://proteus-framework.org/CALLIOPE/) (outgassing), and
[AGNI](https://www.h-nicholls.space/AGNI/) (atmosphere climate).

## Prerequisites

- Full PROTEUS installation with AGNI and SOCRATES compiled
- `FWL_DATA` and `RAD_DIR` environment variables set
- Spectral files downloaded (`proteus get spectral -n Dayspring -b 48`)
- Solar spectrum downloaded (`proteus get stellar`)
- Interior data downloaded, including the PALEOS EOS tables for the
  structure solver
  (`proteus get interiordata --config-path input/tutorials/tutorial_earth.toml`)

Reference data is also fetched automatically when `proteus start` runs
without the `--offline` flag, so the download commands above are only
required for offline use.

## Physical setup

This case follows Table 2 of the CHILI protocol paper:

| Parameter | Value |
|-----------|-------|
| Planet mass | 1 M$_\oplus$ |
| Core mass fraction | 0.325 |
| Stellar mass | 1 M$_\odot$ |
| Starting stellar age | 50 Myr |
| Semi-major axis | 1 AU |
| Bond albedo | 0.1 |
| Oxygen fugacity | IW+4 |
| Hydrogen inventory | 4.7 $\times$ 10$^{20}$ kg (3 Earth oceans H$_2$O) |
| Carbon inventory | 2.73 $\times$ 10$^{20}$ kg (10$^{21}$ kg CO$_2$) |
| Initial thermal state | Fully molten |
| Termination | Melt fraction $\Phi$ < 5% |

The planet starts fully molten and cools through a magma ocean stage.
Volatiles partition between the atmosphere and silicate melt as the mantle
solidifies. The atmosphere is solved at each timestep
using correlated-k radiative transfer (AGNI). Atmospheric escape is
energy-limited (ZEPHYRUS, 30% efficiency).

## Running the simulation

```bash
conda activate proteus
mkdir -p output/tutorial_earth
nohup proteus start -c input/tutorials/tutorial_earth.toml \
    > /tmp/proteus_earth_launch.log 2>&1 & disown
```

Add `--offline` to skip the reference-data check on later runs; the first
run must be able to download any missing data (or download it beforehand,
see the prerequisites above).

!!! warning "Runtime"
    This run takes 30 minutes to several hours depending on hardware.
    The initial Zalmoxis structure solve (~10-20 min) is the slowest
    phase. Monitor progress with
    `tail -f output/tutorial_earth/proteus_00.log` (the log appears
    once PROTEUS has initialized).

## Configuration

The config at `input/tutorials/tutorial_earth.toml` sets:

- **Star**: Sun on Spada[^cite-spada2013] tracks starting at 50 Myr. The solar
  spectrum is used for radiative transfer. Stellar luminosity, radius, and
  XUV flux evolve with age.
- **Interior**: Aragog solves the mantle energy equation on an 80-node radial
  grid using SUNDIALS CVODE with JAX Jacobian. Zalmoxis computes the
  hydrostatic structure using PALEOS EOS tables.
- **Outgassing**: CALLIOPE partitions H$_2$O, CO$_2$, H$_2$, CH$_4$, and CO
  between atmosphere and melt at the fO$_2$ = IW+4 buffer.
- **Atmosphere**: AGNI solves the radiative-convective equilibrium with
  Dayspring 48-band correlated-k opacities, a conductive skin layer at the
  surface, and real-gas corrections.
- **Escape**: ZEPHYRUS computes energy-limited mass loss at 30% efficiency,
  distributing the bulk escape rate across elements proportionally.

## Results

After the run completes, generate plots:

```bash
proteus plot -c input/tutorials/tutorial_earth.toml all
```

<figure markdown="span">
  ![Earth tutorial output](../assets/tutorials/earth_global_log.avif){ width="100%" }
  <figcaption>Multi-panel overview of the PROTEUS Earth analogue tutorial run.
  (a) Upward heat flux components: radiogenic heating (purple, ~0.2 W m<sup>-2</sup>), net interior flux (dashed orange), net atmospheric flux (solid orange), OLR (dashed green), and absorbed stellar flux (ASF, dashed blue, ~226 W m<sup>-2</sup>). The net fluxes decline from ~10<sup>5</sup> W m<sup>-2</sup> to ~10<sup>2</sup> W m<sup>-2</sup> over 1.3 Myr.
  (b) Surface partial pressures: CO<sub>2</sub> (orange) dominates early at ~88 bar; H<sub>2</sub>O (blue) starts at ~5 bar and rises to ~368 bar as it exsolves during solidification. CO (dark yellow) and H<sub>2</sub> (green) remain minor.
  (c) Surface temperature declining from ~3300 K to ~1860 K at the solidus.
  (d) Surface gas mole fractions: CO<sub>2</sub> (orange) dominates early; H<sub>2</sub>O (blue) rises from near zero to dominate late, crossing CO<sub>2</sub> around 10<sup>5</sup> yr.
  (e) Mantle evolution: core-mantle boundary (dashed purple) at ~0.49 planet fraction, rheological front (orange) propagating outward as the mantle solidifies, global melt fraction (dotted black) decreasing from 1.0 to 0.05.
  (f) Volatile partitioning into the interior: H<sub>2</sub>O (blue) starts ~80% dissolved in the melt and drops to near 0% at solidification. CO<sub>2</sub> (orange) follows a similar but weaker trend (~15% interior initially).</figcaption>
</figure>

### Thermal evolution (a, c)

The planet starts fully molten at T$_\mathrm{s}$ $\approx$ 3300 K.
The magma ocean radiates through a thick CO$_2$/steam atmosphere, with
the net interior and atmospheric fluxes reaching ~10$^5$ W m$^{-2}$
initially (a). Radiogenic heating (purple) provides a constant ~0.2
W m$^{-2}$ baseline, negligible compared to the interior cooling flux.
The absorbed stellar flux (ASF, dashed blue) is ~226 W m$^{-2}$ at 1 AU
(instellation F$_\mathrm{ins}$ $\approx$ 1005 W m$^{-2}$ at 50 Myr,
reduced by the geometry factor, Bond albedo, and zenith angle).

The surface temperature (c) decreases from ~3300 K to ~1860 K at the
solidus over ~1.3 Myr. The decline slows around 10$^5$ yr as the mantle
enters the mushy zone and latent heat release buffers the cooling.

### Atmospheric evolution (b, d)

The atmosphere evolves in composition as the mantle solidifies:

1. **Early phase** (t < 10$^5$ yr): CO$_2$ dominates the atmosphere at
   ~88 bar (b), while H$_2$O starts at only ~5 bar because most water
   is dissolved in the silicate melt (~80% interior, f). In mole
   fraction (d), CO$_2$ dominates early.

2. **Late phase** (t > 10$^5$ yr): As the melt fraction drops, H$_2$O
   exsolves from the crystallizing mantle and its partial pressure rises
   to ~368 bar at solidification. H$_2$O overtakes CO$_2$ in mole
   fraction around 10$^5$ yr (d) and dominates the final atmosphere at
   ~84 mol%. The total surface pressure reaches ~438 bar.

CO and H$_2$ remain minor species throughout (~1-10 bar), consistent
with the oxidizing conditions (IW+4). CH$_4$ is negligible.

### Mantle evolution (e, f)

The Zalmoxis structure solver computes the hydrostatic profile at
initialization: R$_\mathrm{planet}$ = 6.91 Mm (1.08 R$_\oplus$),
core radius = 3.38 Mm (0.53 R$_\oplus$), CMB pressure = 114 GPa,
center pressure = 360 GPa.

In (e), the core-mantle boundary (dashed purple) sits at ~0.49 of the
planet radius. The rheological front (orange), defined as the radius
where $\Phi$ = 0.4, propagates outward from the CMB as the mantle
crystallizes from the base up. The global melt fraction (dotted black)
decreases from 1.0 to 0.05, at which point the run terminates.

In (f), H$_2$O (blue) starts with ~80% of its mass dissolved in the
interior melt and drops to near 0% as the melt fraction approaches
zero, releasing volatiles into the atmosphere. CO$_2$ (orange) follows
a similar trend but with a smaller interior fraction (~15% initially)
because of its lower solubility in silicate melt at IW+4.

## Next steps

- **Venus analogue**: Run the Venus tutorial
  (`input/tutorials/tutorial_venus.toml`) with `planet.mass_tot = 0.815`
  and `orbit.semimajoraxis = 0.723` to explore the effect of higher
  instellation on solidification.
- **CHILI comparison**: See the
  [CHILI intercomparison tutorial](chili_intercomparison.md) for
  multi-model comparison plots.
- **Volatile sensitivity**: Vary `H_budget` between 1.6$\times$10$^{20}$
  and 1.6$\times$10$^{21}$ kg to explore the effect of hydrogen inventory
  on cooling time.
- **Reduced mantle**: Set `outgas.fO2_shift_IW = -2` to simulate a
  reduced mantle producing H$_2$-rich instead of H$_2$O-rich atmospheres.

---

**See also:** [Model description](../Explanations/model.md) | [Coupling loop](../Explanations/coupling_loop.md) | [Configuration reference](../Reference/config/params.md) | [Output format](../Reference/output.md)

[^cite-lichtenberg2026]: Lichtenberg, T., Schaefer, L., Krissansen-Totton, J., et al., *[Coupled atmosHere Interior modeL Intercomparison (CHILI): Protocol Version 1.0](https://doi.org/10.3847/PSJ/ae593b)*, The Planetary Science Journal, 7, 108, 2026. [SciX](https://scixplorer.org/abs/2026PSJ.....7..108L/abstract).

[^cite-spada2013]: Spada, F., Demarque, P., Kim, Y.C. & Sills, A., *[The radius discrepancy in low-mass stars: single versus binaries](https://doi.org/10.1088/0004-637X/776/2/87)*, The Astrophysical Journal, 776, 87, 2013. [SciX](https://scixplorer.org/abs/2013ApJ...776...87S/abstract).
