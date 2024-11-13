[![Tests for PROTEUS](https://github.com/FormingWorlds/PROTEUS/actions/workflows/tests.yaml/badge.svg)](https://github.com/FormingWorlds/PROTEUS/actions/workflows/tests.yaml)
![Coverage](https://gist.githubusercontent.com/stefsmeets/b4ee7dab92e20644bcb3a5ad09f71165/raw/covbadge.svg)
[![Documentation Status](https://readthedocs.org/projects/fwl-proteus/badge/?version=latest)](https://fwl-proteus.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![PROTEUS banner](https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_white.png#gh-light-mode-only)
![PROTEUS banner](https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_black.png#gh-dark-mode-only)

# PROTEUS Framework for Planetary Evolution

**PROTEUS** is a modular Python framework that simulates the coupled evolution of the atmospheres and interiors of rocky planets.

## Installation instructions

See the [installation guide](https://fwl-proteus.readthedocs.io/en/latest/installation/) for steps and troubleshooting advice.

## Run instructions

Only attempt to run PROTEUS after you have followed all of the installation instructions.
See the [usage guide](https://fwl-proteus.readthedocs.io/en/latest/usage/) for more information.

## Contributors

| Name  | Email address |
| -     | -             |
Tim Lichtenberg         | tim.lichtenberg[at]rug.nl |
Harrison Nicholls       | harrison.nicholls[at]physics.ox.ac.uk |
Laurent Soucasse        | l.soucasse[at]esciencecenter.nl |
Stef Smeets             | s.smeets[at]esciencecenter.nl |
Dan J. Bower            | dbower[at]ethz.ch |
Mariana Sastre       | m.c.villamil.sastre[at]rug.nl |
Emma Postolec           | e.n.postolec[at]rug.nl |
Mark Hammond            | mark.hammond[at]physics.ox.ac.uk |
Raymond Pierrehumbert   | raymond.pierrehumbert[at]physics.ox.ac.uk |

## Repository structure

| Object                | Description                                               |
| -                     | -                                                         |
| `README.md`           | Overview file                                             |
| `pyproject.toml`	    | Project configuration file                                |
| `CODE_OF_CONDUCT.md`	| Project code of conduct                                   |
| `LICENSE.txt`         | Project license                                           |
| `src/proteus`         | Source code for PROTEUS                                   |
| `output/`             | Output folder with subfolders for each model run          |
| `input/`              | Input folder (e.g. stellar spectra, example configs)      |
| `docs/`			          | Documentation source files                                |
| `examples/`           | Example cases that the model should be able to reproduce  |

## Citation

If you make use of PROTEUS, please reference the following manuscripts, and state the code version used.

```
@ARTICLE{Lichtenberg2021JGRP,
       author = {{Lichtenberg}, Tim and {Bower}, Dan J. and {Hammond}, Mark and {Boukrouche}, Ryan and {Sanan}, Patrick and {Tsai}, Shang-Min and {Pierrehumbert}, Raymond T.},
        title = "{Vertically Resolved Magma Ocean-Protoatmosphere Evolution: H$_{2}$, H$_{2}$O, CO$_{2}$, CH$_{4}$, CO, O$_{2}$, and N$_{2}$ as Primary Absorbers}",
      journal = {Journal of Geophysical Research (Planets)},
         year = 2021,
        month = feb,
       volume = {126},
       number = {2},
          eid = {e06711},
        pages = {e06711},
          doi = {10.1029/2020JE006711},
archivePrefix = {arXiv},
       eprint = {2101.10991},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021JGRE..12606711L}
}

@ARTICLE{Nicholls2025JGRP,
       author = {{Nicholls}, Harrison and {Lichtenberg}, Tim and {Bower}, Dan J. and and {Pierrehumbert}, Raymond T.},
        title = "{Magma ocean evolution at arbitrary redox state}",
      journal = {Journal of Geophysical Research (Planets)},
         year = 2025,
        pages = {submitted},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021JGRE..12606711L}
}
```

Please see the relevant sub-modules (e.g., [JANUS]([url](https://github.com/FormingWorlds/JANUS)), [MORS]([url](https://github.com/FormingWorlds/MORS)), [SOCRATES]([url](https://github.com/nichollsh/SOCRATES)), [AGNI]([url](https://github.com/nichollsh/AGNI)), [SPIDER]([url](https://github.com/djbower/spider)), [CALLIOPE]([url](https://github.com/FormingWorlds/CALLIOPE)), [ZEPHYRUS]([url](https://github.com/FormingWorlds/ZEPHYRUS))) for further references.
