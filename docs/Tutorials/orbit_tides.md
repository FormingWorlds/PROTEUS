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
properties of the planet and its satellite. Create a new file called 
`0d_test_moon.json` in the `examples/satellite/` directory relative to the
PROTEUS root directory. The data file should be a JSON file with the following 
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

The values provided here reflect a partially molten Moon, with a fluid iron core, a 
solid mantle, and a partially molten crust. Although we do not require the orbital 
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

!!! warning "Runtime"
    This run takes several hours to overnight depending on hardware.

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




---

**See also:** [Model description](../Explanations/model.md) | [Orbital dynamics](../Explanations/orbit.md) |[Coupling loop](../Explanations/coupling_loop.md) | [Configuration reference](../Reference/config/params.md) | [Output format](../Reference/output.md)

 [^cite-spada2013]: Spada, F., Demarque, P., Kim, Y.C. & Sills, A., *[The radius discrepancy in low-mass stars: single versus binaries](https://doi.org/10.1088/0004-637X/776/2/87)*, The Astrophysical Journal, 776, 87, 2013. [SciX](https://scixplorer.org/abs/2013ApJ...776...87S/abstract).
