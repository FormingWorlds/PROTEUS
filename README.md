<h1 align="center">
    <div>
        <img src="https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_white.png#gh-light-mode-only" style="vertical-align: middle;" width="60%"/>
        <img src="https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_black_nobkg.png#gh-dark-mode-only" style="vertical-align: middle;" width="60%"/>
    </div>
</h1>

<p align="center">
  <a href="https://github.com/FormingWorlds/PROTEUS/actions/workflows/ci-pr-checks.yml" target="_blank" rel="noopener">
    <img src="https://proteus-framework.org/PROTEUS/badges/unit.svg" alt="Unit tests status">
  </a>
  <a href="https://github.com/FormingWorlds/PROTEUS/actions/workflows/ci-nightly.yml" target="_blank" rel="noopener">
    <img src="https://proteus-framework.org/PROTEUS/badges/integration.svg" alt="Integration tests status">
  </a>
  <a href="https://proteus-framework.org/PROTEUS/" target="_blank" rel="noopener">
    <img src="https://proteus-framework.org/PROTEUS/badges/docs.svg" alt="Documentation build status">
  </a>
  <a href="https://app.codecov.io/gh/FormingWorlds/PROTEUS/tree/main" target="_blank" rel="noopener">
    <img src="https://proteus-framework.org/PROTEUS/badges/codecov.svg" alt="Code coverage">
  </a>
  <br/>
  <a href="https://opensource.org/licenses/Apache-2.0" target="_blank" rel="noopener">
    <img src="https://proteus-framework.org/PROTEUS/badges/license.svg" alt="License: Apache 2.0">
  </a>
  <a href="https://proteus-framework.org" target="_blank" rel="noopener">
    <img src="https://proteus-framework.org/PROTEUS/badges/website.svg" alt="Website: proteus-framework.org">
  </a>
</p>

**Coupled atmosphere-interior evolution of rocky planets and exoplanets.**

PROTEUS (/ˈproʊtiəs, PROH-tee-əs) is a modular Python framework that connects interior thermal evolution, volatile outgassing, atmospheric radiative transfer, atmospheric escape, and stellar evolution into a self-consistent time-stepping loop, following a planet from a molten magma ocean to a solidified surface over billions of years. Inspired by the Greek god of elusive sea change, who could change his form at will, PROTEUS is designed to be flexible and adaptable to a wide range of planetary environments. It can foretell the future, but answers only to those who are capable of asking the right questions.

## Features

- **Modular backends**: interior, atmosphere, escape, outgassing, stellar, and orbit domains each expose multiple interchangeable solvers behind a common interface.
- **Coupled, conservative loop**: modules exchange boundary conditions every timestep, enforcing mass and energy conservation across the whole planet system.
- **Adaptive time-stepping**: resolves fast transitions (solidification, escape episodes) and steps efficiently through quasi-steady phases.
- **Parameter sweeps and observations**: built-in Cartesian grid sweeps over configuration parameters, plus forward models for synthetic transit and eclipse spectra.

## Get started

Follow the [installation guide](https://proteus-framework.org/PROTEUS/How-to/installation.html), then run the all-dummy quick start (no external solvers, under a minute):

```console
proteus start -c input/dummy.toml
```

The [quick-start tutorial](https://proteus-framework.org/PROTEUS/Tutorials/quick_start_dummy.html) explains the output; the [Earth analogue tutorial](https://proteus-framework.org/PROTEUS/Tutorials/earth_analogue.html) is a full production run.

## Documentation

Full documentation: **[proteus-framework.org/PROTEUS](https://proteus-framework.org/PROTEUS/)**

- [Model description](https://proteus-framework.org/PROTEUS/Explanations/model.html)
- [Installation guide](https://proteus-framework.org/PROTEUS/How-to/installation.html)
- [Usage guide](https://proteus-framework.org/PROTEUS/How-to/usage.html)
- [Contributing guide](https://proteus-framework.org/PROTEUS/Community/CONTRIBUTING.html)

## Community and citation

Ask questions on the [discussions page](https://github.com/orgs/FormingWorlds/discussions) or [contact the developers](https://proteus-framework.org/PROTEUS/Community/contact.html). If you use PROTEUS, please cite the papers listed in the [bibliography](https://proteus-framework.org/PROTEUS/Reference/bibliography.html).
