<h1 align="center">
    <div>
        <img src="https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_white.png#gh-light-mode-only" style="vertical-align: middle;" width="60%"/>
        <img src="https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_black_nobkg.png#gh-dark-mode-only" style="vertical-align: middle;" width="60%"/>
    </div>
</h1>

<p align="center">
  <a href="https://github.com/FormingWorlds/PROTEUS/actions/workflows/coverage-baseline.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/FormingWorlds/PROTEUS/coverage-baseline.yml?branch=main&label=Unit%20Tests">
  </a>
  <a href="https://github.com/FormingWorlds/PROTEUS/actions/workflows/ci-nightly.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/FormingWorlds/PROTEUS/ci-nightly.yml?branch=main&label=Integration%20Tests">
  </a>
  <a href="https://proteus-framework.org/PROTEUS/">
    <img src="https://img.shields.io/github/actions/workflow/status/FormingWorlds/PROTEUS/docs.yaml?branch=main&label=Docs">
  </a>
  <a href="https://app.codecov.io/gh/FormingWorlds/PROTEUS/tree/main">
    <img src="https://codecov.io/gh/FormingWorlds/PROTEUS/branch/main/graph/badge.svg">
  </a>
  <br/>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
  </a>
  <a href="https://proteus-framework.org">
    <img src="https://img.shields.io/website?url=https%3A%2F%2Fproteus-framework.org&label=Website&up_message=proteus-framework.org&down_message=proteus-framework.org" />
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
