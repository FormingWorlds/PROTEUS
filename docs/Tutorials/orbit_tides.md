# Orbit and tides

This tutorial demonstrates how to use PROTEUS to model the orbital and 
tidal evolution of a planet with a satellite. It covers the setup of the 
planet-satellite system, the configuration of tidal parameters, and the 
execution of simulations to study the effects of tides on planetary orbits.

## Prerequisites

- PROTEUS installed with `bash install.sh` (see the
  [installation guide](../How-to/installation.md)), which builds AGNI and
  SOCRATES and sets up Obliqua
- SPIDER (with PETSc), which the installer does not build
  (commands below)
- `FWL_DATA` and `RAD_DIR` environment variables set
- Spectral files downloaded (`proteus get spectral -n Dayspring -b 48`)
- Solar spectrum downloaded (`proteus get stellar`)
- Interior data downloaded, including the EOS tables that SPIDER reads
  (`proteus get interiordata --config-path input/tutorials/tutorial_earth_moon.toml`)

Reference data is also fetched automatically when `proteus start` runs
without the `--offline` flag, so the download commands above are only
required for offline use.

The config uses SPIDER for the interior structure and energetics and Obliqua
for the tides. Build SPIDER after the standard install, from the PROTEUS folder
with the `proteus` environment active:

```bash
bash tools/get_petsc.sh
bash tools/get_spider.sh
```

The installer sets up Obliqua. When `proteus doctor` reports Obliqua as not
installed, run `bash tools/get_obliqua.sh` from the PROTEUS folder.

The [optional modules](../How-to/optionalmodules_installation.md) page
describes each step.

## Physical setup

The physical setup for the orbit and tides tutorial involves defining the 
properties of the planet and its satellite. For the satellite a data file called 
`tutorial_earth_moon.json` is included in the `input/tutorials/` directory relative 
to the PROTEUS root directory. The data file must be a JSON file with the following 
structure. The `//` comments explain the fields here and are not part of the
file: JSON has no comment syntax, and the file in the repository holds only
the values.

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
nohup proteus start -c input/tutorials/tutorial_earth_moon.toml \
    > /tmp/proteus_earth_moon_launch.log 2>&1 & disown
```

Add `--offline` to skip the reference-data check on later runs; the first
run must be able to download any missing data (or download it beforehand,
see the prerequisites above).

Monitor progress with `tail -f output/tutorial_earth_moon/proteus_00.log`
(the log appears once PROTEUS has initialized).

!!! info "Runtime"
    This run takes roughly 30 minutes depending on hardware. A fine excuse to go
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
- **Outgassing**: CALLIOPE partitions the hydrogen, carbon, nitrogen and
  sulfur volatiles (among them H$_2$O, CO$_2$, CO, H$_2$, CH$_4$ and N$_2$)
  between atmosphere and melt at the fO$_2$ = IW+2 buffer.
- **Atmosphere**: AGNI computes the radiative fluxes on a prescribed
  temperature profile (`atmos_clim.agni.solve_energy = false`): a dry adiabat
  from the surface, which is held at the magma temperature
  (`atmos_clim.surf_state = "fixed"`), with condensation above it. It uses
  Dayspring 48-band correlated-k opacities and real-gas corrections.
- **Escape**: ZEPHYRUS computes energy-limited mass loss at 20% efficiency,
  distributing the bulk escape rate across elements proportionally.

## Results

After the run completes, generate plots:

```bash
proteus plot -c input/tutorials/tutorial_earth_moon.toml all
```

The run also writes these plots by itself when it ends. Its Love number
figure, shown below, holds every snapshot; `proteus plot` draws that figure
from a sample of 8 snapshots.

The config limits the run to 8 &times; 10<sup>5</sup> yr of evolution
(`params.stop.time.maximum`), so the reference run below ends at that time
(status: *Completed (target time)*), with the mantle still around 22%
molten ($\Phi \approx 0.22$) rather than fully solidified. Overall, finer 
time resolution will substantially reduce the jumping behavior in the tidal heating.
Ultimately, this is beyond the scope of this tutorial, as it would take substantially 
longer to run. 

<figure markdown="span">
  ![Orbital evolution](../assets/orbit/orbit_tides_orbit.avif#only-light){ width="100%" }
  ![Orbital evolution](../assets/orbit/orbit_tides_orbit_dark.avif#only-dark){ width="100%" }
  <figcaption><b>Planet-star and satellite-planet orbital evolution.</b>
  The planet's heliocentric semi-major axis and eccentricity stay pinned at
  1.00 AU and 0 (Obliqua evolves the planet-satellite pair; the star-planet
  orbit is not perturbed by it here), so the planet's orbital period holds at
  365.2563 days while its axial (spin) period lengthens from 4.00 h to
  7.73 h. The satellite's semi-major axis climbs
  from ~3.5 to ~24.9 Earth radii over the run, with its orbital and axial spin
  periods rising together from ~9.1 h to ~172 h (in the plotted time range
  the two agree to within 0.1%, i.e. it stays tidally locked). Its
  eccentricity, 0.05 at the start, is damped away
  within the first ~20 yr, before the plotted time range.</figcaption>
</figure>

<figure markdown="span">
  ![Global flux budget](../assets/orbit/orbit_tides_fluxes_global.avif#only-light){ width="100%" }
  ![Global flux budget](../assets/orbit/orbit_tides_fluxes_global_dark.avif#only-dark){ width="100%" }
  <figcaption><b>Global flux budget.</b>
  Tidal heating (gold) starts at ~5.0 &times; 10<sup>5</sup> W m<sup>-2</sup>,
  comparable to the net interior/atmosphere flux (orange/grey) at that time,
  then decreases as the satellite moves away from the planet, first dropping
  below the runaway-greenhouse (S&ndash;N) limit at t &approx;
  8.8 &times; 10<sup>4</sup> yr. From there it no longer decays smoothly: it
  reaches an absolute minimum of ~0.5 W m<sup>-2</sup> near
  3.5 &times; 10<sup>5</sup> yr, then rises again and changes by up to a
  factor of ~300 from one output time to the next (see the Love-number figure
  below), peaking at ~7.8 &times; 10<sup>2</sup> W m<sup>-2</sup> near
  7.8 &times; 10<sup>5</sup> yr. This peak and the two values of
  ~500 W m<sup>-2</sup> in the last 10<sup>4</sup> yr each follow a snapshot in
  which Im(k<sub>22</sub>) is at the cap of the config
  (`orbit.obliqua.cap_LN = true`). The mantle is still ~22% molten when the run
  ends at t = 8 &times; 10<sup>5</sup> yr, with tidal heating that swings
  between ~20 and ~500 W m<sup>-2</sup> over the last 2 &times; 10<sup>4</sup> yr,
  around the net interior/atmosphere flux (~2.3 &times; 10<sup>2</sup> W
  m<sup>-2</sup>), rather than having settled into a smoothly decaying
  solid-tide regime.</figcaption>
</figure>

<figure markdown="span">
  ![Love number spectrum evolution](../assets/orbit/orbit_tides_lovenumber.avif#only-light){ width="100%" }
  ![Love number spectrum evolution](../assets/orbit/orbit_tides_lovenumber_dark.avif#only-dark){ width="100%" }
  <figcaption><b>Degree-2 Love number spectrum evolution (Obliqua).</b>
  The dominant, almost always populated mode is the semidiurnal (n=2, m=2, k=2) tide;
  its forcing frequency |&sigma;| rises from
  ~4.9 &times; 10<sup>-4</sup> to ~6.5 &times; 10<sup>-4</sup> rad
  s<sup>-1</sup> in the first ~10<sup>3</sup> yr and then decreases to
  ~4.3 &times; 10<sup>-4</sup> rad s<sup>-1</sup> as the planet despins.
  Both panels show magnitudes.
  Re(k<sub>22</sub>) is near 0.02 at 10<sup>3</sup> yr and stays below 0.04 while
  the layer list of the tidal model holds only a fluid layer. It jumps to ~1.4
  at t &approx; 1.5 &times; 10<sup>5</sup> yr, in the snapshot where the first
  solid layer appears (&Phi; &approx; 0.69); the forcing frequency changes by
  0.4% between these two snapshots. From there to 7.6 &times; 10<sup>5</sup> yr
  it lies mostly between ~0.5 and ~1.4.
  It exceeds the threshold of the red rings (Re &gt; 1.5)
  once, at t &approx; 7.9 &times; 10<sup>5</sup> yr, where it peaks at ~3.7.
  It is negative in 14 snapshots, which get no ring: first at
  t &approx; 7.2 &times; 10<sup>5</sup> yr, again at
  7.7 &times; 10<sup>5</sup> yr (its minimum, &minus;3.5), and in 12 of the
  last 14 snapshots (the last 1.5 &times; 10<sup>4</sup> yr of the run).
  Im(k<sub>22</sub>) is at the cap (|Im| = 3) in 3 snapshots of the last
  3.2 &times; 10<sup>4</sup> yr. The magnitude of Im(k<sub>22</sub>) is smallest, 10<sup>-3.5</sup>, near
  3.4 &times; 10<sup>5</sup> yr, where the tidal heating has its
  minimum.</figcaption>
</figure>

---

**See also:** [Model description](../Explanations/model.md) | [Orbital dynamics](../Explanations/orbit.md) |[Coupling loop](../Explanations/coupling_loop.md) | [Configuration reference](../Reference/config/params.md) | [Output format](../Reference/output.md)

 [^cite-spada2013]: Spada, F., Demarque, P., Kim, Y.C. & Sills, A., *[The radius discrepancy in low-mass stars: single versus binaries](https://doi.org/10.1088/0004-637X/776/2/87)*, The Astrophysical Journal, 776, 87, 2013. [SciX](https://scixplorer.org/abs/2013ApJ...776...87S/abstract).
