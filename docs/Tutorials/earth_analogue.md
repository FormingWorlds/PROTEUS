# Earth analogue

This tutorial simulates the thermal and atmospheric evolution of an
Earth-mass planet at 1 AU from a Sun-like star. It follows the nominal
Earth case from the CHILI intercomparison [^cite-lichtenberg2026], with a
more reduced mantle (oxygen fugacity IW+2 instead of IW+4).

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

This case takes its star, orbit, albedo, volatile inventories and initial
thermal state from Table 2 of the CHILI protocol paper. It departs from that
table in the oxygen fugacity and in the interior structure, which Zalmoxis
solves from a core mass fraction instead of the fixed planet radius and core
radius fraction of the protocol:

| Parameter | Value |
|-----------|-------|
| Planet mass | 1 M$_\oplus$ |
| Core mass fraction (CHILI: core radius fraction 0.55) | 0.325 |
| Stellar mass | 1 M$_\odot$ |
| Starting stellar age | 50 Myr |
| Semi-major axis | 1 AU |
| Bond albedo | 0.1 |
| Oxygen fugacity | IW+2 (CHILI case: IW+4) |
| Hydrogen inventory | 4.7 $\times$ 10$^{20}$ kg (3 Earth oceans H$_2$O) |
| Carbon inventory | 2.73 $\times$ 10$^{20}$ kg (10$^{21}$ kg CO$_2$) |
| Initial thermal state | Fully molten |
| Termination (stop of this config) | Melt fraction $\Phi$ < 5% |

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

Monitor progress with `tail -f output/tutorial_earth/proteus_00.log`
(the log appears once PROTEUS has initialized).

!!! warning "Runtime"
    This run takes several hours depending on hardware (about 4.8 h of
    wall time and 4.8 CPU hours for the reference run below). The
    initialization, with the first Zalmoxis structure solve, costs ~16 min.
    After that the run time follows the solidification time: from
    ~6 $\times$ 10$^4$ yr on the mushy mantle is advanced mostly at the largest
    timestep that the config allows there (`params.dt.mushy_maximum` = 4000 yr),
    down to the 5% termination.

## Configuration

The config at `input/tutorials/tutorial_earth.toml` sets:

- **Star**: Sun on Spada [^cite-spada2013] tracks starting at 50 Myr. The solar
  spectrum is used for radiative transfer. Stellar luminosity, radius, and
  XUV flux evolve with age.
- **Interior**: Aragog solves the mantle energy equation on an 80-node radial
  grid using SUNDIALS CVODE with JAX Jacobian. Zalmoxis computes the
  hydrostatic structure using PALEOS EOS tables.
- **Outgassing**: CALLIOPE partitions H$_2$O, CO$_2$, H$_2$, CH$_4$, and CO
  between atmosphere and melt at the fO$_2$ = IW+2 buffer.
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
  ![Earth tutorial output](../assets/tutorials/earth_global_log.avif#only-light){ width="100%" }
  ![Earth tutorial output](../assets/tutorials/earth_global_log_dark.avif#only-dark){ width="100%" }
  <figcaption><b>Multi-panel overview of the PROTEUS Earth analogue tutorial run.</b>
  (a) Upward heat flux components: radiogenic heating (magenta, ~0.2 W m<sup>-2</sup>), net interior flux (dashed orange), net atmospheric flux (solid, grey/white), outgoing longwave radiation (OLR, red), and absorbed stellar flux (ASF, dashed blue, ~226 W m<sup>-2</sup>). Tidal heating (dark yellow) is zero in this run. The net fluxes decline from ~2 &times; 10<sup>5</sup> W m<sup>-2</sup> at 10<sup>3</sup> yr, where the time axis starts, to ~140 W m<sup>-2</sup> over ~3.8 Myr.
  (b) Surface partial pressures: at the start of the time axis the total surface pressure (dashed) is ~210 bar and O<sub>2</sub> (yellow-green, ~120 bar) is the largest species; O<sub>2</sub> collapses within the first ~2 &times; 10<sup>4</sup> yr as the surface cools. CO<sub>2</sub> (orange) holds ~36-61 bar; H<sub>2</sub>O (blue) rises from ~4 bar to ~280 bar as it exsolves during solidification. CO (gold) falls from ~33 bar to ~16 bar and H<sub>2</sub> (green) rises from ~0.2 bar to ~35 bar.
  (c) Surface temperature (solid) declining from ~3200 K to ~1760 K at termination; the magma temperature (dashed orange) starts near 4100 K on this axis.
  (d) Surface gas mole fractions: O<sub>2</sub> (yellow-green) is the largest species at the start of the axis (~58%) and collapses; CO<sub>2</sub> (orange) then dominates, peaking near 61%, with CO (gold) at up to ~34%; H<sub>2</sub>O (blue) rises to dominate late, crossing CO<sub>2</sub> around 1.2 &times; 10<sup>6</sup> yr and reaching ~76%.
  (e) Mantle evolution: the dashed purple line marks the core mass fraction (0.325); the rheological front (orange) starts at the core-mantle boundary in radius (~0.48 of the planet radius, above the mass-fraction reference) and propagates outward as the mantle solidifies; the global melt fraction (dotted) decreases from 1.0 to 0.05.
  (f) Volatile partitioning into the interior: H<sub>2</sub>O (blue) starts almost fully dissolved in the melt (~99.6%) and falls to ~45% at the &Phi; = 5% termination, the residual melt still retaining much of the water. CO<sub>2</sub> (orange) is far less soluble, starting at ~15% interior and dropping to ~0.5%.</figcaption>
</figure>

### Thermal evolution (a, c)

The planet starts fully molten, with a surface temperature
T$_\mathrm{s}$ $\approx$ 3300 K and a magma temperature near 4300 K.
The magma ocean radiates through a thick atmosphere, with the net
interior and atmospheric fluxes reaching ~5 $\times$ 10$^5$ W m$^{-2}$ initially
(a). Radiogenic heating (magenta) provides a constant ~0.2 W m$^{-2}$
baseline, negligible compared to the interior cooling flux. The absorbed
stellar flux (ASF, dashed blue) is ~226 W m$^{-2}$ at 1 AU
(instellation F$_\mathrm{ins}$ $\approx$ 1005 W m$^{-2}$ at 50 Myr,
reduced by the geometry factor, Bond albedo, and zenith angle).

The surface temperature (c, solid) decreases from ~3300 K to
~1760 K at termination over ~3.8 Myr; the magma temperature (dashed
orange) tracks above it, starting near 4300 K. The decline slows after
~10$^5$ yr, when the melt fraction has dropped to ~0.75 and latent
heat release buffers the cooling. At termination the net interior and
atmospheric fluxes have fallen to ~140 W m$^{-2}$ (OLR ~360 W m$^{-2}$),
approaching balance with the absorbed stellar flux.

### Atmospheric evolution (b, d)

The atmosphere passes through three compositional stages as the mantle
solidifies:

1. **Superheated initial state** (t $\lesssim$ 4 $\times$ 10$^3$ yr): at the fully
   molten, superheated initial condition the atmosphere holds
   ~300 bar and O$_2$ is its largest species (~72 mol%,
   ~210 bar), next to CO$_2$ (~50 bar) and CO (~31 bar).
   This O$_2$ collapses as the surface cools towards ~3000 K, falling below
   10 bar within the first ~8 $\times$ 10$^3$ yr and to a negligible
   partial pressure by ~10$^5$ yr.

2. **CO$_2$-dominated phase** (~4 $\times$ 10$^3$ to ~1.2 $\times$ 10$^6$ yr): as O$_2$
   collapses, CO$_2$ becomes the dominant species. Its mole fraction peaks
   near 61% around 5 $\times$ 10$^4$ yr, where the partial pressure is
   ~59 bar and CO holds another ~33%; the partial
   pressure then rises to a broad maximum of ~61 bar
   near 3 $\times$ 10$^5$ yr (b, d). H$_2$O begins at only ~4 bar, with
   nearly all water dissolved in the silicate melt (~99.6% interior at the
   start, f), and rises through this phase as the crystallizing mantle
   exsolves it.

3. **H$_2$O-dominated phase** (t > ~1.2 $\times$ 10$^6$ yr): H$_2$O overtakes
   CO$_2$ in mole fraction around 1.2 $\times$ 10$^6$ yr (d) and keeps
   rising as the melt crystallizes. It dominates the final atmosphere at
   ~76 mol% (~280 bar), with CO$_2$ at
   ~9.7 mol% (~36 bar) and H$_2$ at ~9.6 mol%
   (~35 bar). The total surface pressure at termination is
   ~370 bar.

The reduced species CO and H$_2$ are not negligible at IW+2: CO falls from
~33 bar to ~16 bar (~4 mol% at the end), while H$_2$ climbs
from ~0.2 bar to ~35 bar as the mantle solidifies. CH$_4$ is negligible.

### Mantle evolution (e, f)

The Zalmoxis structure solver computes the hydrostatic profile at
initialization: R$_\mathrm{planet}$ = 7.16 Mm (1.12 R$_\oplus$),
core radius = 3.46 Mm (0.54 R$_\oplus$, 0.48 of the planet radius),
surface gravity = 7.77 m s$^{-2}$, CMB pressure = 102 GPa,
center pressure = 347 GPa.

In (e), the dashed purple line is drawn at the core mass fraction
(0.325). The rheological front (orange), defined as the radius where
$\Phi$ = 0.4, starts at the core-mantle boundary in radius (~0.48 of the
planet radius, above the mass-fraction reference) and propagates outward
as the mantle crystallizes from the base up. The global melt fraction (dotted)
decreases from 1.0 to 0.05, at which point the run terminates.

In (f), H$_2$O (blue) starts with nearly all of its mass dissolved in the
interior melt (~99.6%) and falls to ~45% at the $\Phi$ = 5% termination.
Because water is highly incompatible, the shrinking melt stays water-rich
even at low melt fraction; the remaining water is released to the
atmosphere only as crystallization completes. CO$_2$ (orange) is far less
soluble in silicate melt, starting at ~15% interior and dropping
to ~0.5%.

## Next steps

- **Venus analogue**: Run the Venus tutorial
  (`input/tutorials/tutorial_venus.toml`) with `planet.mass_tot = 0.815`
  and `orbit.semimajoraxis = 0.723` to explore the effect of higher
  instellation on solidification. The config stops that run at 3 Myr, where
  its mantle is still ~15% molten.
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
