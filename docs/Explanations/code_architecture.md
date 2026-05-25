# PROTEUS code architecture

## Overview

PROTEUS is organised as a collection of modular scientific components, located as
directories inside `src/proteus/`. Each module handles one physical domain of a
coupled planetary evolution simulation:

- [`interior_energetics/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/interior_energetics): thermal evolution of the mantle and core (Aragog, SPIDER, boundary, dummy)
- [`interior_struct/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/interior_struct): hydrostatic structure and planet radius (Zalmoxis, dummy)
- [`atmos_clim/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/atmos_clim): radiative-convective atmosphere (AGNI, JANUS, dummy)
- [`atmos_chem/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/atmos_chem): atmospheric photochemistry (VULCAN, dummy)
- [`escape/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/escape): atmospheric mass loss (ZEPHYRUS, BOREAS, dummy)
- [`outgas/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/outgas): volatile partitioning (CALLIOPE, atmodeller, dummy)
- [`orbit/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/orbit): orbital evolution and tides (Obliqua/LovePy, dummy)
- [`star/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/star): stellar evolution and spectra (MORS, dummy)

Most modules follow a common pattern: a `wrapper.py` defining the dispatch
interface, a `common.py` with shared helpers and data structures, and one file
per backend implementation (for example, `aragog.py`, `spider.py`, `boundary.py`,
and `dummy.py` inside `interior_energetics/`).

The central orchestrator,
[`proteus.py`](https://github.com/FormingWorlds/PROTEUS/blob/main/src/proteus/proteus.py),
couples these modules together and advances the simulation, with modules
exchanging information at each timestep through the `hf_row` dictionary. See
[Coupling loop](coupling_loop.md) for details on the execution order and data
flow.

## Supporting directories

Beyond the physics modules, the source tree contains:

- [`config/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/config): TOML configuration parsing and validation (attrs-based dataclasses)
- [`utils/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/utils): shared utilities (data download, logging, constants, plotting helpers)
- [`plot/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/plot): diagnostic and publication-quality plot routines
- [`grid/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/grid): parameter grid sweep management
- [`inference/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/inference): Bayesian optimisation and inference workflows
- [`cli.py`](https://github.com/FormingWorlds/PROTEUS/blob/main/src/proteus/cli.py): command-line interface entry points

## Architecture diagram

The diagram below gives a high-level view of the PROTEUS code architecture. Click
any module to jump to its source on the [main branch](https://github.com/FormingWorlds/PROTEUS/tree/main),
or any loop block to jump to the relevant section of [`proteus.py`](https://github.com/FormingWorlds/PROTEUS/blob/main/src/proteus/proteus.py).

<object type="image/svg+xml" data="../assets/proteus_architecture.svg" class="arch-diagram arch-diagram--light"></object>
<object type="image/svg+xml" data="../assets/proteus_architecture_darkmode.svg" class="arch-diagram arch-diagram--dark"></object>
