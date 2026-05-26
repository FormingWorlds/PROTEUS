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
- Spectral files downloaded (`proteus get spectral`)
- Solar spectrum downloaded (`proteus get stellar`)

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
solidifies. The atmosphere is solved self-consistently at each timestep
using correlated-k radiative transfer (AGNI). Atmospheric escape is
energy-limited (ZEPHYRUS, 30% efficiency).

## Running the simulation

```bash
conda activate proteus
nohup proteus start --offline -c input/tutorials/tutorial_earth.toml \
    > output/tutorial_earth/launch.log 2>&1 & disown
```

!!! warning "Runtime"
    This run takes 30 minutes to several hours depending on your machine.
    The initial Zalmoxis structure solve (~10-20 min) is the slowest phase.
    Monitor progress with `tail -f output/tutorial_earth/proteus_00.log`.

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
  ![Earth tutorial output](../assets/tutorials/earth_global_log.png){ width="100%" }
  <figcaption>Multi-panel overview of the PROTEUS Earth analogue tutorial run. Panel (a) shows the radiative heat fluxes: the interior flux F_int (blue) and the atmospheric outgoing longwave radiation F_atm (orange) both decrease as the magma ocean cools, declining from ~10<sup>5</sup> W m<sup>-2</sup> to ~10<sup>2</sup> W m<sup>-2</sup> over ~1.3 Myr. Panel (b) shows the surface partial pressures of outgassed species: CO<sub>2</sub> (orange) dominates the early atmosphere at ~100 bar, while H<sub>2</sub>O (blue) increases as solubility decreases with cooling, reaching ~300 bar at solidification. CO and H<sub>2</sub> are minor species. Panel (c) shows the core-mantle boundary radius (dashed purple), the rheological front (orange), and the melt fraction (dotted black) as fractions of the planet radius. The rheological front propagates outward as the mantle solidifies from the base up. Panel (d) shows the interior volatile partitioning: H<sub>2</sub>O transitions from mostly dissolved in the melt (~80%) to mostly in the atmosphere as the melt fraction drops below ~10%.</figcaption>
</figure>

### Thermal evolution

The planet starts fully molten at T$_\mathrm{magma}$ $\approx$ 4300 K.
The magma ocean radiates intensely through a thick steam/CO$_2$ atmosphere,
with the outgoing longwave radiation (OLR) reaching ~2 $\times$ 10$^5$
W m$^{-2}$ initially. This is well above the Nakajima limit
($\sim$280 W m$^{-2}$), so the planet cools rapidly.

As the mantle crystallizes, latent heat is released, buffering the
cooling rate. The solidification front propagates from the base of the
mantle (high pressure, higher melting point) outward. By the time the
global melt fraction drops to 5%, T$_\mathrm{magma}$ has decreased to
~1860 K (near the solidus) and the total solidification time is ~1.3 Myr.

### Atmospheric evolution

The atmosphere evolves in two stages:

1. **Early phase** ($\Phi$ > 0.5): CO$_2$ dominates the atmosphere
   because its solubility in silicate melt is low. H$_2$O is mostly
   dissolved in the magma. The total surface pressure is ~280 bar.

2. **Late phase** ($\Phi$ < 0.5): As the melt fraction decreases,
   H$_2$O exsolves rapidly because its solubility depends on melt
   fraction. The atmosphere becomes H$_2$O-dominated, reaching ~300 bar
   of steam at solidification. The total surface pressure increases to
   ~440 bar.

H$_2$ and CO are minor species throughout, consistent with the oxidizing
conditions (IW+4). Atmospheric escape removes a small fraction of the
hydrogen inventory over the 1.3 Myr solidification timescale.

### Interior structure

The Zalmoxis structure solver computes a self-consistent hydrostatic
profile at initialization: R$_\mathrm{planet}$ = 6.5 Mm (1.02
R$_\oplus$), core radius = 3.5 Mm (0.55 R$_\oplus$), CMB pressure =
92 GPa, center pressure = 478 GPa, average density = 5300 kg m$^{-3}$.
The rheological front (where $\Phi$ crosses the critical melt fraction of
0.4) propagates from the CMB outward, reaching the surface when
$\Phi_\mathrm{global}$ = 0.05.

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
  See the [Reduced H$_2$-rich world](reduced_h2_world.md) tutorial.

---

**See also:** [Model description](../Explanations/model.md) | [Coupling loop](../Explanations/coupling_loop.md) | [Configuration reference](../Reference/config/params.md) | [Output format](../Reference/output.md)

[^cite-lichtenberg2026]: Lichtenberg, T., Schaefer, L., Krissansen-Totton, J., et al., *[Coupled atmosHere Interior modeL Intercomparison (CHILI): Protocol Version 1.0](https://doi.org/10.3847/PSJ/ae593b)*, The Planetary Science Journal, 7, 108, 2026. [SciX](https://scixplorer.org/abs/2026PSJ.....7..108L/abstract).

[^cite-spada2013]: Spada, F., Demarque, P., Kim, Y.C. & Sills, A., *[The radius discrepancy in low-mass stars: single versus binaries](https://doi.org/10.1088/0004-637X/776/2/87)*, The Astrophysical Journal, 776, 87, 2013. [SciX](https://scixplorer.org/abs/2013ApJ...776...87S/abstract).
