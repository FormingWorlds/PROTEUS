# PROTEUS Model Framework

## Overview

PROTEUS is a modular framework for simulating the time evolution of small (exo)planets. It is designed to be flexible, reflecting the broad diversity of planetary conditions already discovered, with the view of being updated to incorporate additional physics as the need arises. This approach stands in contrast to common monolithic models in the literature. PROTEUS is free and open-source, which permits external scrutiny of its workings. It is directly based upon the model of Lichtenberg et al. (2021)[^cite-lichtenberg2021] although the code has evolved substantially from that state.

## Design philosophy

George Box famously put that "all models are wrong, but some are useful". PROTEUS therefore leverages a modular and hierarchical modelling approach. Multiple independent models can fill the role of a given module, and each model can be used stand-alone. Hierarchical modelling allows inter-comparison of simple and complex models, taking advantage of the easy comprehension of the simple in order to diagnose and validate the qualitative behaviour of the complex. Simplified _dummy_ modules are not designed for quantitatively meaningful calculations, but only to qualitatively capture end-member behaviours.

## System architecture

Although PROTEUS aims to treat the problem of *planetary* evolution, it must necessarily also handle external processes which act upon the planet (e.g. tidal heating). The framework therefore models the combined system of a planet, its orbital mechanics, and the evolution of its host star. The planet itself is conceptually sub-divided into a vaporised *atmosphere* component above an *interior* component containing a silicate mantle and metallic core. PROTEUS facilitates communication between individual software *modules* which each implement a model for a specific part of the overall system. Conceptually, PROTEUS modules (e.g. the interior) are 'slots' which are filled by specific implementations: the 'models' (e.g. Aragog).

<p align="center">
      <img src="../assets/schematic.png" style="max-width: 90%; height: auto;"></br>
      <b>Schematic of PROTEUS components and corresponding modules.</b> </br>
</p>

### Module overview

| Module | Implementations | Role |
|--------|----------------|------|
| Structure | [Zalmoxis](https://proteus-framework.org/Zalmoxis/), dummy | Interior structure (radius, density, gravity profiles) |
| Interior | [SPIDER](https://proteus-framework.org/SPIDER/), [Aragog](https://proteus-framework.org/aragog/), boundary, dummy | Mantle/core thermal evolution |
| Atmosphere (climate) | [AGNI](https://www.h-nicholls.space/AGNI/), [JANUS](https://proteus-framework.org/JANUS/), dummy | Radiative-convective profile |
| Atmosphere (chemistry) | [VULCAN](https://github.com/FormingWorlds/VULCAN), dummy | Chemical kinetics |
| Star | [MORS](https://proteus-framework.org/MORS/), dummy | Stellar evolution and spectrum |
| Escape | [ZEPHYRUS](https://github.com/FormingWorlds/ZEPHYRUS), [BOREAS](https://github.com/ExoInteriors/BOREAS), dummy | Atmospheric escape |
| Outgassing | [CALLIOPE](https://proteus-framework.org/CALLIOPE/), [atmodeller](https://github.com/djbower/atmodeller), dummy | Volatile exchange between interior and atmosphere |
| Orbit | [Obliqua](https://github.com/FormingWorlds/Obliqua), dummy | Orbital evolution and tidal heating |

Each module is maintained in its own repository and can be used as a standalone package outside of PROTEUS. The following sections describe each module's physical role and how PROTEUS couples to it.

---

## Interior structure: Zalmoxis

[Zalmoxis](https://proteus-framework.org/Zalmoxis/) computes the hydrostatic equilibrium structure of a differentiated planet (metallic core + silicate mantle + volatile envelope). Given a total planet mass, bulk composition, and surface temperature, Zalmoxis integrates the equations of hydrostatic equilibrium inward from the surface using a tabulated equation of state (EOS), returning radial profiles of pressure, density, temperature, and gravitational acceleration. It also computes the core radius, mantle mass, and surface gravity.

Zalmoxis supports several EOS backends, including the PALEOS unified tables and the Wolf & Bower (2018)[^cite-wolf2018] parameterisation. The structure solution is used by PROTEUS to initialise the planet's radius and to dynamically update the structure during the simulation when the interior thermal state changes (see [Structure-interior coupling](#structureinterior-coupling) below).

Config section: `[interior_struct]`. Reference: [Interior configuration](../Reference/config/interior.md).

## Interior energetics: Aragog, SPIDER, boundary

The interior energetics module evolves the mantle temperature and melt fraction in time by solving the energy equation in the planetary interior.

**[Aragog](https://proteus-framework.org/aragog/)** (Python/JAX) solves the interior energy equation in the temperature-pressure (T-P) formulation. It discretises the mantle on a radial grid and advances the temperature profile using the SUNDIALS CVODE integrator with an analytic Jacobian computed via JAX automatic differentiation. Aragog handles the full mushy-zone (partial melt) regime, including phase-dependent material properties, radiogenic heating, tidal heating, and core cooling. It is the recommended interior module for most configurations.

**[SPIDER](https://proteus-framework.org/SPIDER/)** (C) uses the temperature-entropy (T-S) formulation[^cite-bower2018]. It discretises the mantle on a staggered radial grid and solves the entropy equation using PETSc's implicit time integrator. SPIDER provides an independent cross-check on Aragog for the same physical problem formulated in a different thermodynamic variable. It requires PETSc and is an optional installation component.

**Boundary** solves a simplified single-node energy balance ODE for the mantle surface temperature, treating the mantle as a single thermal reservoir with Arrhenius or aggregate viscosity. It is useful for rapid exploration of parameter space and for configurations where the full radial resolution of Aragog or SPIDER is not needed.

Config section: `[interior_energetics]`. Reference: [Interior configuration](../Reference/config/interior.md).

## Atmosphere climate: AGNI, JANUS

The atmosphere climate module solves the radiative-convective equilibrium structure of the atmosphere given the surface temperature, surface pressure, atmospheric composition, and incoming stellar spectrum.

**[AGNI](https://www.h-nicholls.space/AGNI/)** (Julia) is a radiative-convective atmosphere model that uses the SOCRATES spectral radiative transfer library (Fortran) as its radiation core. AGNI solves for the self-consistent atmospheric temperature profile by iterating a Newton solver on the radiative-convective flux balance. It supports grey-gas and correlated-k radiative transfer, Rayleigh scattering, clouds, and condensation of volatile species. AGNI returns the outgoing longwave radiation (OLR), Bond albedo, and atmospheric temperature-pressure profile.

**[JANUS](https://proteus-framework.org/JANUS/)** (Python) is a 1D convective atmosphere module that computes a radiative-convective profile using a band-averaged two-stream approach. JANUS is faster than AGNI but makes stronger simplifying assumptions (e.g. no iterative Newton solve; a prescribed adiabatic lower atmosphere). It is useful for parameter sweeps and initial exploration.

Config section: `[atmos_clim]`. Reference: [Atmosphere configuration](../Reference/config/atmosphere.md).

## Atmospheric chemistry: VULCAN

**[VULCAN](https://github.com/FormingWorlds/VULCAN)** (Python) is a photochemical kinetics code that computes steady-state atmospheric mixing ratios given a temperature-pressure profile, eddy diffusion profile, and UV stellar spectrum. In PROTEUS, VULCAN can run in two modes: *online* (at each snapshot during the simulation) or *offline* (once after the simulation completes). Most configurations use the offline mode because the chemistry calculation is computationally expensive relative to the coupling timestep.

Config section: `[atmos_chem]`. Reference: [Atmosphere configuration](../Reference/config/atmosphere.md).

## Stellar evolution: MORS

**[MORS](https://proteus-framework.org/MORS/)** (Python) provides stellar evolutionary tracks and spectral energy distributions. It supports two track families: **Spada**[^cite-spada2013] (rotation-dependent tracks with activity-calibrated XUV luminosities, suitable for solar-type stars) and **Baraffe**[^cite-baraffe2015] (mass-luminosity tracks for low-mass stars). MORS interpolates the stellar mass, radius, effective temperature, bolometric luminosity, and XUV luminosity at any stellar age, and synthesises a wavelength-resolved spectrum by scaling a modern reference spectrum (observed MUSCLES data, solar NREL data, or a synthetic PHOENIX spectrum) to the historical luminosity.

Config section: `[star]`. Reference: [Star and orbit configuration](../Reference/config/star_orbit.md).

## Atmospheric escape: ZEPHYRUS, BOREAS

The escape module computes atmospheric mass loss rates driven by the stellar XUV flux.

**[ZEPHYRUS](https://github.com/FormingWorlds/ZEPHYRUS)** (Python) implements energy-limited escape, computing a bulk mass loss rate from the XUV flux, planet mass, and XUV absorption radius. The bulk rate is then distributed across elements proportionally to their atmospheric abundance (unfractionated escape). ZEPHYRUS also provides a tidal correction factor for planets in close-in orbits.

**[BOREAS](https://github.com/ExoInteriors/BOREAS)** (Python) is a hydrodynamic escape model that solves for the upper atmosphere structure and distinguishes between energy-limited, recombination-limited, and diffusion-limited regimes. It supports fractionated escape, where light species (H, He) escape preferentially over heavier species (O, C). This is physically important for determining the long-term oxygen buildup in secondary atmospheres.

Config section: `[escape]`. Reference: [Escape and outgassing configuration](../Reference/config/escape_outgas.md).

## Volatile outgassing: CALLIOPE, atmodeller

The outgassing module computes the thermodynamic equilibrium partitioning of volatiles between the atmosphere, silicate melt, and solid mantle.

**[CALLIOPE](https://proteus-framework.org/CALLIOPE/)** (Python) solves for the equilibrium partial pressures and dissolved volatile concentrations given the mantle temperature, melt fraction, and total element inventories for H, C, N, S, and O[^cite-bower2019]. It uses parameterised solubility laws and an fO2 buffer (configurable as an IW offset) to compute the redox state[^cite-nicholls2024]. CALLIOPE handles the full set of major volcanic gases (H$_2$O, CO$_2$, H$_2$, CO, N$_2$, SO$_2$, S$_2$, CH$_4$).

**[atmodeller](https://github.com/djbower/atmodeller)** (Python/JAX) is an alternative outgassing backend that uses a real-gas equation of state and a more detailed thermochemical treatment[^cite-bower2025]. atmodeller provides an independent cross-check on CALLIOPE for the same volatile partitioning problem.

Config section: `[outgas]`. Reference: [Escape and outgassing configuration](../Reference/config/escape_outgas.md).

## Orbital evolution: Obliqua

**[Obliqua](https://github.com/FormingWorlds/Obliqua)** (Julia) evolves the orbital semi-major axis and eccentricity under the influence of tidal dissipation. The tidal response of the planet is computed from its interior structure and rheology using a viscoelastic love-number solver (LovePy). Tidal heating power is distributed radially across the mantle and fed back into the interior energy equation. Obliqua also computes the spin-orbit evolution and checks for dynamical stability (Roche limit, Hill sphere).

Config section: `[orbit]`. Reference: [Star and orbit configuration](../Reference/config/star_orbit.md).

---

### Structure-interior coupling

The structure and interior modules serve complementary roles.
The *interior* module (SPIDER or Aragog) evolves the mantle temperature and melt fraction in time, while the *structure* module (Zalmoxis) solves for the hydrostatic equilibrium of the planet given a temperature profile and equation of state.

When `interior_struct.module = 'zalmoxis'` and `interior_struct.zalmoxis.update_interval > 0`, PROTEUS can dynamically recompute the planetary structure during a simulation.
Structure updates are governed by a hybrid trigger that combines physics-based criteria with timing constraints:

- **Floor** (`update_min_interval`): minimum time between updates, preventing excessive recomputation during rapid cooling.
- **Ceiling** (`update_interval`): maximum time between updates, guaranteeing periodic recomputation.
- **Temperature trigger** (`update_dtmagma_frac`): fires when the relative change in mantle surface temperature since the last update exceeds the threshold.
- **Melt fraction trigger** (`update_dphi_abs`): fires when the absolute change in global melt fraction since the last update exceeds the threshold.

Setting `update_interval = 0` disables dynamic updates entirely; the structure is computed only at initialisation.

## Time evolution vs equilibrium

Only the interior and star modules have an explicit notion of time-evolution. All other modules are applied at equilibrium, such that the quantities calculated by these modules are effectively updated instantaneously at each time-step. This assumes that the physical processes handled by these equilibrium modules reach steady-state on time-scales shorter than the physics considered by interior and stellar evolution modules.

## Further reading

- [Coupling loop](coupling_loop.md): how the modules exchange data at each timestep
- [Code architecture](code_architecture.md): source code layout and module patterns
- [Configuration reference](../Reference/config/params.md): all configuration parameters
- [Tutorials](../Tutorials/quick_start_dummy.md): worked examples from dummy to production
- [Bibliography](../Reference/bibliography.md): published references for PROTEUS and its modules

[^cite-lichtenberg2021]: Lichtenberg, T., Bower, D.J., Hammond, M., et al., *[Vertically resolved magma ocean-protoatmosphere evolution: H2, H2O, CO2, CH4, CO, O2, and N2 as primary absorbers](https://doi.org/10.1029/2020JE006711)*, Journal of Geophysical Research: Planets, 126, e2020JE006711, 2021. [SciX](https://scixplorer.org/abs/2021JGRE..12606711L/abstract).

[^cite-wolf2018]: Wolf, A.S. & Bower, D.J., *[An equation of state for high pressure-temperature liquids (RTpress) applied to MgSiO3 liquid](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49-62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).

[^cite-bower2018]: Bower, D.J., Sanan, P. & Wolf, A.S., *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49-62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).

[^cite-bower2019]: Bower, D.J., Kitzmann, D., Wolf, A.S., et al., *[Linking the evolution of terrestrial interiors and an early outgassed atmosphere to astrophysical observations](https://doi.org/10.1051/0004-6361/201935710)*, Astronomy & Astrophysics, 631, A103, 2019. [SciX](https://scixplorer.org/abs/2019A%26A...631A.103B/abstract).

[^cite-bower2025]: Bower, D.J., Thompson, M.A., Hakim, K., et al., *[Diversity of low-mass planet atmospheres in the C-H-O-N-S-Cl system](https://doi.org/10.3847/1538-4357/ad8999)*, The Astrophysical Journal, 995, 59, 2025. [SciX](https://scixplorer.org/abs/2025ApJ...995...59B/abstract).

[^cite-nicholls2024]: Nicholls, H., Lichtenberg, T., Bower, D.J. & Pierrehumbert, R., *[Magma ocean evolution at arbitrary redox state](https://doi.org/10.1029/2024JE008576)*, Journal of Geophysical Research: Planets, 129, e2024JE008576, 2024. [SciX](https://scixplorer.org/abs/2024JGRE..12908576N/abstract).

[^cite-spada2013]: Spada, F., Demarque, P., Kim, Y.C. & Sills, A., *[The radius discrepancy in low-mass stars: single versus binaries](https://doi.org/10.1088/0004-637X/776/2/87)*, The Astrophysical Journal, 776, 87, 2013. [SciX](https://scixplorer.org/abs/2013ApJ...776...87S/abstract).

[^cite-baraffe2015]: Baraffe, I., Homeier, D., Allard, F. & Chabrier, G., *[New evolutionary models for pre-main sequence and main sequence low-mass stars down to the hydrogen-burning limit](https://doi.org/10.1051/0004-6361/201425481)*, Astronomy & Astrophysics, 577, A42, 2015. [SciX](https://scixplorer.org/abs/2015A%26A...577A..42B/abstract).
