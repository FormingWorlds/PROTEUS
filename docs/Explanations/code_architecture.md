# PROTEUS code architecture

## Overview

PROTEUS is organised as a collection of modular scientific components, located as
directories inside `src/proteus/`. Each module handles one physical domain of a
coupled planetary evolution simulation:

- [`interior_energetics/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/interior_energetics): thermal evolution of the mantle and core (Aragog, SPIDER, boundary, dummy)
- [`interior_struct/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/interior_struct): hydrostatic structure and planet radius (Zalmoxis, dummy)
- [`atmos_clim/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/atmos_clim): radiative-convective atmosphere (AGNI, JANUS, dummy)
- [`atmos_chem/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/atmos_chem): atmospheric photochemistry (VULCAN, dummy)
- [`escape/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/escape): atmospheric mass loss (ZEPHYRUS, dummy)
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
exchanging state at each timestep through a shared dictionary called `hf_row`.
The next page, [Coupling loop](coupling_loop.md), explains how these modules run
at runtime: the fixed execution order within each timestep, the full `hf_row`
data bus, and the adaptive time-stepping and termination logic.

## Supporting directories

Beyond the physics modules, the source tree contains:

- [`config/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/config): TOML configuration parsing and validation (attrs-based dataclasses)
- [`utils/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/utils): shared utilities (data download, logging, constants, plotting helpers)
- [`plot/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/plot): diagnostic and publication-quality plot routines
- [`grid/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/grid): parameter grid sweep management
- [`inference/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/inference): Bayesian optimisation and inference workflows
- [`cli.py`](https://github.com/FormingWorlds/PROTEUS/blob/main/src/proteus/cli.py): command-line interface entry points

## Architecture diagram

The diagram below shows the static layout: each box is a physics module, coloured
by domain. Arrows between modules indicate the quantities exchanged through
`hf_row` at each timestep. Click any module to jump to its source on the
[main branch](https://github.com/FormingWorlds/PROTEUS/tree/main), or any loop
block to jump to the relevant section of
[`proteus.py`](https://github.com/FormingWorlds/PROTEUS/blob/main/src/proteus/proteus.py).

<p style="text-align:right;margin-bottom:0.25rem;font-size:0.85rem">
  <a href="../assets/proteus_architecture_viewer.html" target="_blank" rel="noopener">Open full size ↗</a>
</p>
<object type="image/svg+xml" data="../assets/proteus_architecture.svg" class="arch-diagram arch-diagram--light"></object>
<object type="image/svg+xml" data="../assets/proteus_architecture_darkmode.svg" class="arch-diagram arch-diagram--dark"></object>

## Organising changes for parallel development

PROTEUS is edited by many contributors at once, so the structure is chosen to
keep changes local. The source tree is sliced vertically by physical domain:
a change to atmospheric escape lives in `escape/`, a change to outgassing lives
in `outgas/`, and the two rarely touch the same file. Within a module, the
`wrapper.py` / `<backend>.py` / `common.py` split means adding or modifying one
backend leaves the others untouched.

The files that every contributor must touch are the coupling glue: the main
loop in `proteus.py`, the shared helpers in `utils/`, and the central registries
of output columns and configuration fields. These are the places where parallel
edits are most likely to collide. Keeping individual functions small, adding new
loop steps as named stage functions, and writing shared registries one entry per
line (grouped by module) keeps most additions on distinct lines, so independent
work merges without conflict. The practical targets are in
[Development standards](../How-to/development_standards.md#code-organization).