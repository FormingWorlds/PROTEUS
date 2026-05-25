# Dummy modules

## Motivation

Every module slot in PROTEUS has a **dummy** implementation that replaces
the full physics with a minimal parameterisation. Dummy modules serve two
purposes:

1. **Testing and validation.** Running all modules in dummy mode exercises
   the full coupling architecture (helpfile data bus, timestep control,
   convergence checks, output pipeline) without requiring external solvers,
   compiled code, or reference data. This makes it possible to write fast
   unit tests, verify that a new module slot is wired correctly, and
   diagnose coupling bugs in isolation from solver bugs.

2. **Physics grounding.** Simplified models provide analytical or
   semi-analytical end-member behaviour that the production modules must
   reproduce in the appropriate limits. When a full solver produces a
   result, comparing it against the dummy equivalent answers the question
   "does this output make sense at zeroth order?" If the dummy predicts
   100 Myr of cooling and Aragog predicts 10 Gyr, something is wrong
   in the setup, not the solver. This hierarchical modelling philosophy
   is central to PROTEUS.

Dummy modules are not designed for quantitatively meaningful science.
They capture qualitative behaviour (a planet cools, volatiles outgas,
the atmosphere radiates) without the numerical resolution, coupled
feedbacks, or calibrated parameterisations of the production modules.

## Module descriptions

### Structure: dummy (`interior_struct.module = 'dummy'`)

Uses the Noack & Lasbleis (2020)[^cite-noack2020] analytical scaling laws
for rocky planet interior structure. Given a total planet mass and iron
mass fraction, the scaling laws return the core radius, mantle thickness,
CMB pressure, surface gravity, and radial profiles of density, pressure,
and temperature. The parameterisation is calibrated against full interior
structure models for planets between 0.8 and 2 M$_\oplus$ with
Earth-like mineralogy and variable iron content.

The dummy structure module provides all the radial profiles that SPIDER and
Aragog need as boundary conditions without running a hydrostatic
equilibrium solver or loading EOS tables. When paired with the dummy
energetics module, the entire interior is analytically specified.

### Interior energetics: dummy (`interior_energetics.module = 'dummy'`)

A parameterised cooling model with prescribed solidus and liquidus
temperatures. The mantle is treated as a single thermal reservoir whose
temperature evolves by integrating a heat-capacity ODE:
$dT/dt = -(F_\mathrm{int} - F_\mathrm{tidal} - F_\mathrm{radio}) \cdot A / C_p$.
The interior heat flux is set equal to the atmospheric heat flux from the
previous iteration, so the cooling rate is driven by whatever the
atmosphere module computes. The melt fraction is a linear interpolation
between the solidus (default 1700 K) and liquidus (default 2700 K).
No radial grid, no phase-dependent material properties.

This module is useful for verifying that the coupling loop handles the
melt-fraction-to-outgassing feedback correctly: as the dummy cools and
the melt fraction drops, the outgassing module should respond with
decreasing atmospheric partial pressures.

### Atmosphere climate: dummy (`atmos_clim.module = 'dummy'`)

A grey-body model for the atmospheric radiative properties. The upward
longwave flux is:

$$F_\mathrm{OLR} = \sigma \bigl[T_\mathrm{surf} (1 - \gamma)\bigr]^4$$

where $\gamma$ reduces the effective radiating temperature
(0 = transparent atmosphere, 1 = perfectly opaque). The transit radius
is estimated from a single scale height above the surface.

An alternative **fixed-flux mode** (`dummy.fixed_flux > 0`) bypasses
the grey-body computation entirely and returns a constant atmospheric
flux. This is useful for testing the interior module's response to a
prescribed boundary condition.

### Atmosphere chemistry: dummy (`atmos_chem.module = 'dummy'`)

A parameterised vertical composition model. The dummy chemistry module
builds vertical profiles for all tracked species across the atmosphere's
pressure levels. It applies a Clausius-Clapeyron cold trap to H$_2$O,
generates approximate photolysis products (O, OH, H, HCN, NO, C$_2$H$_2$)
that increase exponentially toward the top of the atmosphere, and
renormalises all volume mixing ratios to sum to unity at each level.
Output is written in VULCAN-compatible CSV format.

Unlike the other dummy modules, this one produces non-trivial vertical
structure. It is useful for testing the observation pipeline (transit
spectra, emission spectra) with a physically plausible composition
profile without running the full VULCAN photochemistry solver.

### Star: dummy (`star.module = 'dummy'`)

A fixed star with no time evolution. The effective temperature
(`dummy.Teff`) and luminosity are constant; the spectrum is a Planck
function at that temperature, scaled to the planet-star separation.
The stellar radius is either set explicitly or derived from an
empirical mass-radius relation (Demircan & Kahraman 1991).

Useful for isolating the planetary evolution from stellar evolution
effects: the instellation stays constant, so all atmospheric and
interior changes are driven by the planet's own thermal evolution.

### Escape: dummy (`escape.module = 'dummy'`)

A constant bulk mass loss rate (user-specified in kg/s). The total rate
is distributed across elements proportionally to their atmospheric
abundance (unfractionated). No dependence on XUV flux, planet mass, or
atmospheric structure.

Useful for testing that the element-tracking machinery (per-element
escape rates, cumulative escaped mass, desiccation gate) works correctly
with a known, constant input rate.

### Outgassing: dummy (`outgas.module = 'dummy'`)

Parameterised volatile partitioning without a thermodynamic solver.
Elemental budgets are split between atmosphere and melt using the global
melt fraction as a partition coefficient: the dissolved fraction scales
with $\Phi_\mathrm{global}$ and the atmospheric fraction with
$1 - \Phi_\mathrm{global}$. Species mapping uses fixed stoichiometry
(H $\to$ H$_2$O, C $\to$ CO$_2$, N $\to$ N$_2$, S $\to$ SO$_2$).
Surface pressure is computed from the thin-atmosphere approximation
$P = mg / (4\pi R^2)$.

No chemical equilibrium, no solubility laws, no fO$_2$ buffer, no
real-gas equation of state. The dummy preserves the correct qualitative
behaviour (outgassing increases with melt fraction) without the cost
of a Gibbs minimisation or real-gas solver.

### Orbit: dummy (`orbit.module = 'dummy'`)

No orbital evolution: semi-major axis and eccentricity remain at their
initial values throughout the simulation. The orbital period is computed
from Kepler's third law. A configurable tidal heating amplitude
(`H_tide`) is applied to mantle layers where the melt fraction exceeds
a threshold (`Phi_tide`), providing a simple parameterised heat source
for testing the interior module's response to tidal power without
running the full LovePy viscoelastic solver.

---

**See also:** [Model description](model.md) | [Quick start tutorial](../Tutorials/quick_start_dummy.md) | [Configuration reference](../Reference/config/params.md)

[^cite-noack2020]: Noack, L. & Lasbleis, M., *[Parameterisations of interior properties of rocky planets](https://doi.org/10.1051/0004-6361/202037723)*, Astronomy & Astrophysics, 638, A129, 2020. [SciX](https://scixplorer.org/abs/2020A%26A...638A.129N/abstract).
