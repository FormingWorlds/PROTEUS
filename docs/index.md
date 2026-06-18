---
title: PROTEUS
---

<h1 align="center">
    <a href="https://proteus-framework.org">
    <div>
        <img src="https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_white.png#gh-light-mode-only" style="vertical-align: middle;" width="60%"/>
        <img src="https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_black_nobkg.png#gh-dark-mode-only" style="vertical-align: middle;" width="60%"/>
    </div>
    </a>
</h1>

<p align="center">
  <a href="https://github.com/FormingWorlds/PROTEUS/actions/workflows/ci-pr-checks.yml" target="_blank" rel="noopener">
    <img src="badges/unit.svg" alt="Unit tests status">
  </a>
  <a href="https://github.com/FormingWorlds/PROTEUS/actions/workflows/ci-nightly.yml" target="_blank" rel="noopener">
    <img src="badges/integration.svg" alt="Integration tests status">
  </a>
  <a href="https://proteus-framework.org/PROTEUS/" target="_blank" rel="noopener">
    <img src="badges/docs.svg" alt="Documentation build status">
  </a>
  <a href="https://app.codecov.io/gh/FormingWorlds/PROTEUS/tree/main" target="_blank" rel="noopener">
    <img src="badges/codecov.svg" alt="Code coverage">
  </a>
  <br/>
  <a href="https://opensource.org/licenses/Apache-2.0" target="_blank" rel="noopener">
    <img src="badges/license.svg" alt="License: Apache 2.0">
  </a>
  <a href="https://proteus-framework.org" target="_blank" rel="noopener">
    <img src="badges/website.svg" alt="Website: proteus-framework.org">
  </a>
</p>

**PROTEUS** is a modular Python framework that simulates the coupled evolution
of rocky planet atmospheres and interiors. It connects interior thermal
evolution, volatile outgassing, atmospheric radiative transfer, atmospheric
escape, and stellar evolution into a self-consistent time-stepping loop,
tracking the planet from a molten magma ocean to a solidified surface over
billions of years.

<figure markdown="span">
<object type="image/svg+xml" data="../assets/proteus_modules_schematic.svg" class="arch-diagram arch-diagram--light"></object>
<object type="image/svg+xml" data="../assets/proteus_modules_schematic_darkmode.svg" class="arch-diagram arch-diagram--dark"></object>
</figure>

<p style="text-align: center;"><strong>Schematic of PROTEUS components and corresponding modules.</strong></p>

## Key features

- **Modular architecture**: each physical domain (interior, atmosphere,
  escape, outgassing, stellar, orbit) has multiple interchangeable backends
  with a common interface
- **Coupled evolution**: modules exchange boundary conditions through a
  shared data bus at each timestep, enforcing mass and energy conservation
  across the full planet system
- **Adaptive time-stepping**: the simulation automatically adjusts its
  timestep to resolve rapid transitions (magma ocean solidification, escape
  episodes) while stepping efficiently through quasi-steady phases
- **Parameter studies**: built-in support for Cartesian parameter grid sweeps
  and asynchronous Bayesian optimization for inverse problems
- **Synthetic observations**: forward models for transit and eclipse spectra
  from the simulated atmospheric state

## Get started

<div class="grid cards" markdown>

-   :material-download: **Install PROTEUS**

    Set up the framework and its physics modules with the guided
    installer.

    [Installation guide](How-to/installation.md){ .md-button .md-button--primary }

-   :material-school: **New to PROTEUS?**

    Start with the quick-start tutorial using all-dummy backends (runs in under a minute).

    [Quick start tutorial](Tutorials/quick_start_dummy.md){ .md-button .md-button--primary }

-   :material-earth: **Ready for science?**

    Follow the Earth analogue tutorial for a full production run with real
    physics modules. 

    [Earth analogue tutorial](Tutorials/earth_analogue.md){ .md-button .md-button--primary}

</div>

## Citation and credit

If you make use of PROTEUS, please reference the scientific manuscripts
outlined in the [Bibliography](Reference/bibliography.md), state the code
version used, and include an acknowledgement. We provide a suggested
acknowledgement in the [contributing page](Community/CONTRIBUTING.md#licensing-and-credit).
