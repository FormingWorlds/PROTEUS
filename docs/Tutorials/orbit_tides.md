# Orbit and tides

This tutorial demonstrates how to use PROTEUS to model the orbital and 
tidal evolution of a planet with a satellite. It covers the setup of the 
planet-satellite system, the configuration of tidal parameters, and the 
execution of simulations to study the effects of tides on planetary orbits.

## Prerequisites

- Full PROTEUS installation with AGNI, SOCRATES, and Obliqua compiled
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

The physical setup for the orbit and tides tutorial involves defining the 
properties of the planet and its satellite. For the satellite a data file called 
`tutorial_earth_moon.json` is included in the `input/tutorials/` directory relative 
to the PROTEUS root directory. The data file must be a JSON file with the following 
structure:

```json
{
    "omega": 1.0e-06,       // Orbital frequency in rad/s
    "axial": 1.0e-06,       // Axial rotation frequency in rad/s
    "ecc": 0.1,             // Orbital eccentricity
    "sma": 1.82e7,          // Semi-major axis in meters
    "S_mass": 6e24,         // Mass of the central body in kg
    "density": [            // Radial density profile in kg/m^3
        7822.0,             // Iron core density
        3500.0,             // Solid mantle density
        3500.0              // Molten crust density
    ],
    "radius": [             // Radial radius profile in meters
        480000.0,           // Radius of the iron core in meters
        995613.0,           // Radius of the solid mantle in meters
        1650000.0           // Radius of the molten crust in meters (i.e. the surface)
    ],
    "visc": [               // Viscosity profile in Pa.s
        1e22,               // Viscosity of the solid mantle in Pa.s
        1e2                 // Viscosity of the molten crust in Pa.s
    ],
    "shear": [              // Shear modulus profile in Pa
        65857968278.256905, // Shear modulus of the solid mantle in Pa
        10.0                // Shear modulus of the molten crust in Pa
    ],
    "bulk": [               // Bulk modulus profile in Pa
        147739735943.7933,  // Bulk modulus of the solid mantle in Pa
        1000000000.0        // Bulk modulus of the molten crust in Pa
    ],
    "phi": [                // Porosity profile (dimensionless)
        0.0,                // Porosity of the solid mantle (dimensionless)
        1.0                 // Porosity of the molten crust (dimensionless
    ]
}
```

The specific values provided here reflect a partially molten Moon, with a fluid iron core, 
a solid mantle, and a partially molten crust. Although we do not require the orbital 
parameters for this test case, we still need to provide them in the data file. The 
values provided here are arbitrary and do not affect the tidal response calculations.

## Running the simulation

```bash
conda activate proteus
mkdir -p output/tutorial_earth_moon
proteus start -c input/tutorials/tutorial_earth_moon.toml
```

Add `--offline` to skip the reference-data check on later runs; the first
run must be able to download any missing data (or download it beforehand,
see the prerequisites above).

Monitor progress with `tail -f output/tutorial_earth_moon/proteus_00.log`
(the log appears once PROTEUS has initialized).

!!! info "Runtime"
    This run takes roughly 1 hour depending on hardware. A fine excuse to go
    read up on the physics while SPIDER, AGNI, and Obliqua sort out the
    Earth-Moon system on your behalf: the [orbital dynamics](../Explanations/orbit.md)
    page covers PROTEUS's own orbital models, or head over to the
    [Obliqua documentation](https://proteus-framework.org/Obliqua) for the
    multi-phase tidal-response theory driving this tutorial, whose
    [usage guide](https://proteus-framework.org/Obliqua/dev/how-to-guides/usage/)
    includes a tidal response evolution animation for an Earth-like planet. Not in
    the mood to wait at all? The [Results](#results) section below already
    has the pregenerated plots from a reference run. 

## Configuration

The config at `input/tutorials/tutorial_earth_moon.toml` sets:

- **Star**: Sun on Spada [^cite-spada2013] tracks starting at 50 Myr. The solar
  spectrum is used for radiative transfer. Stellar luminosity, radius, and
  XUV flux evolve with age.
- **Interior**: SPIDER solves the mantle energy equation on an 80-node radial
  grid. SPIDER also computes the hydrostatic structure using the
  `MgSiO3_Wolf_Bower_2018_1TPa` EOS tables.
- **Outgassing**: CALLIOPE partitions H$_2$O, CO$_2$, H$_2$, CH$_4$, and CO
  between atmosphere and melt at the fO$_2$ = IW+2 buffer.
- **Atmosphere**: AGNI solves the radiative-convective equilibrium with
  Dayspring 48-band correlated-k opacities and real-gas corrections.
- **Escape**: ZEPHYRUS computes energy-limited mass loss at 20% efficiency,
  distributing the bulk escape rate across elements proportionally.

## Results

After the run completes, generate plots:

```bash
proteus plot -c input/tutorials/tutorial_earth_moon.toml all
```

The reference run below reaches solidification at t $\approx$ 2.41 Myr.

<figure markdown="span">
  ![Orbital evolution](../assets/orbit/orbit_tides_orbit.avif#only-light){ width="100%" }
  ![Orbital evolution](../assets/orbit/orbit_tides_orbit_dark.avif#only-dark){ width="100%" }
  <figcaption><b>Planet-star and satellite-planet orbital evolution.</b>
  The planet's heliocentric semi-major axis and eccentricity stay pinned at
  1.00 AU and 0 (Obliqua evolves the planet-satellite pair; the star-planet
  orbit is not perturbed by it here), so the planet's orbital period holds at
  365.2563 days while its axial (spin) period lengthens from 4.00 h to
  6.53 h as the satellite despins it. The satellite's semi-major axis climbs
  from ~3.5 to ~16.3 Earth radii over the run, with its orbital and axial spin
  periods rising together from ~9.1 h to ~91.8 h (i.e. it stays
  tidally locked), and its eccentricity varying between 0 and 0.05.</figcaption>
</figure>

<figure markdown="span">
  ![Global flux budget](../assets/orbit/orbit_tides_fluxes_global.avif#only-light){ width="100%" }
  ![Global flux budget](../assets/orbit/orbit_tides_fluxes_global_dark.avif#only-dark){ width="100%" }
  <figcaption><b>Global flux budget.</b>
  Tidal heating (gold) starts at ~7.7 &times; 10<sup>5</sup> W m<sup>-2</sup>,
  comparable to the net interior/atmosphere flux (orange/grey) at that time,
  then decreases as the satellite moves away from the planet and the mantle 
  solidifies. It reaches an absolute minimum after ~4 &times; 10<sup>4</sup> yr, 
  reflecting the weakening tidal potential and dissipative properties of the 
  Earth's interior during the mush phase. The dissipation then rapidly builds 
  up again, producing a peak( ~2 &times; 10<sup>3</sup> W m<sup>-2</sup>) near 
  10<sup>5</sup> yr, attributed to crossing of a forests of interior resonances. 
  Finally, solid tides takeover, and dissipation smoothly decays to ~3.5 W m<sup>-2</sup> 
  by the end of the run.</figcaption>
</figure>

<figure markdown="span">
  ![Love number spectrum evolution](../assets/orbit/orbit_tides_lovenumber.avif#only-light){ width="100%" }
  ![Love number spectrum evolution](../assets/orbit/orbit_tides_lovenumber_dark.avif#only-dark){ width="100%" }
  <figcaption><b>Degree-2 Love number spectrum evolution (Obliqua).</b>
  The dominant, almost always populated mode is the semidiurnal (n=2, m=2, k=2) tide; 
  its forcing frequency |&sigma;| decreases from
  ~6.7 &times; 10<sup>-4</sup> to ~4.9 &times; 10<sup>-4</sup> rad
  s<sup>-1</sup> as the satellite recedes and despins the planet.
  Re(k<sub>22</sub>) rises from ~0.09 initially to a peak of ~1.54 at
  t &approx; 5.3 &times; 10<sup>4</sup> yr, consistent with the forcing
  frequency sweeping past a forest of normal-modes (seismic) resonance as 
  the mantle's rheological structure evolves; between
  ~3.7 &times; 10<sup>4</sup> and ~8.2 &times; 10<sup>4</sup> yr it exceeds
  the seismic-resonance threshold (Re &gt; 1.5, ringed in red) before
  relaxing to ~0.48 by the end of the run. Im(k<sub>22</sub>) stays negative
  throughout, its magnitude ranging from ~0.34 down to
  ~6.7 &times; 10<sup>-6</sup> and ending at ~0.0014 as the interior
  solidifies and dissipation drops. A single snapshot of a neighbouring
  (n=2, m=2, k=1) harmonic, at t &approx; 1.1 &times; 10<sup>5</sup> yr,
  also crosses into the seismic-resonance region (Im &lt; -1).</figcaption>
</figure>

---

**See also:** [Model description](../Explanations/model.md) | [Orbital dynamics](../Explanations/orbit.md) |[Coupling loop](../Explanations/coupling_loop.md) | [Configuration reference](../Reference/config/params.md) | [Output format](../Reference/output.md)

 [^cite-spada2013]: Spada, F., Demarque, P., Kim, Y.C. & Sills, A., *[The radius discrepancy in low-mass stars: single versus binaries](https://doi.org/10.1088/0004-637X/776/2/87)*, The Astrophysical Journal, 776, 87, 2013. [SciX](https://scixplorer.org/abs/2013ApJ...776...87S/abstract).
