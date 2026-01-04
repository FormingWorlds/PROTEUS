# PROTEUS Architecture

## Philosophical Foundation

George Box famously stated: "All models are wrong, but some are useful." This principle guides PROTEUS development. To model planetary evolution across many orders of magnitude in spatial and temporal scales, we employ a **modular and hierarchical modelling approach**.

## System Design

PROTEUS is a framework for simulating the time evolution of small exoplanets. Rather than a monolithic approach, PROTEUS is:

- **Modular**: Each physical process is handled by independent software modules
- **Hierarchical**: Multiple models can fill the same module "slot" for inter-comparison
- **Flexible**: Designed to incorporate additional physics as scientific needs evolve
- **Open-source**: Permits external scrutiny and ensures transparency of assumptions

## Conceptual Division

PROTEUS models the coupled system of:

1. **The Planet**
   - *Atmosphere*: Vaporized component above the surface
   - *Interior*: Silicate mantle and metallic core below

2. **External Processes**
   - Tidal heating from nearby bodies
   - Orbital interactions with neighboring planets
   - Stellar evolution and irradiation

## Architecture Schematic

```text
[Host Star] ──→ Irradiation ──→ [Planet Atmosphere]
                                      ↓
                          [Planet Interior]
                          ├─ Silicate Mantle
                          └─ Metallic Core
                                      ↓
                          [Cooling & Evolution]
                                      ↓
[Neighboring Planets] ──→ Tidal Heating & Orbital Effects
```

## Module Structure

PROTEUS divides the problem into conceptual **modules** ("slots") that are filled by specific **models** (implementations):

| Module | Role | Examples |
| --- | --- | --- |
| **Atmosphere** | Temperature-pressure profiles, composition | JANUS, AGNI |
| **Interior** | Mantle/core cooling, differentiation | ARAGOG, SPIDER |
| **Star** | Stellar evolution and luminosity | MORS |
| **Escape** | Atmospheric escape processes | ZEPHYRUS |
| **Chemistry** | Atmospheric chemistry and mixing | VULCAN |
| **Observables** | Generate synthetic observations | PLATON |
| **Tides** | Tidal heating and evolution | Love.jl |

### Hierarchical Modelling

PROTEUS implements multiple independent models for the same module, allowing:

- **Simple models** (e.g., dummy modules) capture end-member behaviors and set expectations
- **Complex models** provide physically realistic calculations
- **Inter-comparison** validates qualitative behavior between simple and complex approaches

Dummy modules are intentionally simplified—they are not designed for quantitative accuracy but for testing and understanding model behavior.

## Time Evolution Philosophy

Only two modules have explicit time evolution:

- **Interior Module**: Tracks mantle/core cooling and potential solidification
- **Star Module**: Simulates stellar aging and luminosity changes

All other modules operate at **equilibrium**:

- Physical processes reach steady-state on time-scales shorter than interior/stellar evolution
- Quantities are effectively updated instantaneously at each time-step
- This assumption simplifies calculations while maintaining physical validity

## Design Philosophy

PROTEUS emphasizes:

1. **Modularity**: Each component can operate independently
2. **Transparency**: Open-source approach enables scientific scrutiny
3. **Flexibility**: Easy to add new physics and modules
4. **Validation**: Hierarchical models enable cross-validation
5. **Reusability**: Individual modules can be used standalone

## Historical Foundation

PROTEUS is directly based on the model of Lichtenberg et al. (2021), though the codebase has evolved substantially. For detailed scientific background, see the [Bibliography](../reference/bibliography.md).
