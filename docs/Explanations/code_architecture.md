# PROTEUS code architecture

## Overview

PROTEUS is organised as a collection of modular scientific components, located as
directories inside `src/proteus/`. Each module handles one physical domain of a
coupled planetary evolution simulation:

- [`interior/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/interior): interior structure and energetics
- [`atmos_clim/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/atmos_clim): atmospheric climate and radiation
- [`atmos_chem/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/atmos_chem): atmospheric chemistry
- [`escape/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/escape):  atmospheric escape
- [`outgas/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/outgas): volatile outgassing
- [`orbit/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/orbit): orbital evolution and tides
- [`star/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/star): stellar evolution

Most modules follow a common pattern: a `wrapper.py` defining the interface, a
`common.py` with shared helpers, and one file per backend implementation (for
example, `aragog.py`, `spider.py`, and `dummy.py` inside `interior/`).

The central orchestrator, [`proteus.py`](https://github.com/FormingWorlds/PROTEUS/blob/main/src/proteus/proteus.py),
couples these modules together and advances the simulation, with modules
exchanging information at each timestep.

## Architecture diagram

The diagram below gives a high-level view of the PROTEUS code architecture. Click
any module to jump to its source on the [main branch](https://github.com/FormingWorlds/PROTEUS/tree/main),
or any loop block to jump to the relevant section of [`proteus.py`](https://github.com/FormingWorlds/PROTEUS/blob/main/src/proteus/proteus.py).

<object type="image/svg+xml" data="../assets/proteus_architecture.svg" class="arch-diagram arch-diagram--light"></object>
<object type="image/svg+xml" data="../assets/proteus_architecture_darkmode.svg" class="arch-diagram arch-diagram--dark"></object>