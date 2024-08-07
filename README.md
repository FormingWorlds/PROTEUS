[![Tests for PROTEUS](https://github.com/FormingWorlds/actions/workflows/tests.yaml/badge.svg)](https://github.com/FormingWorlds/actions/workflows/tests.yaml)
![Coverage](https://gist.githubusercontent.com/stefsmeets/b4ee7dab92e20644bcb3a5ad09f71165/raw/covbadge.svg)
[![Documentation Status](https://readthedocs.org/projects/fwl-proteus/badge/?version=latest)](https://fwl-proteus.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![PROTEUS banner](https://raw.githubusercontent.com/FormingWorlds/PROTEUS/master/docs/images/PROTEUS_white.png#gh-light-mode-only)
![PROTEUS banner](https://raw.githubusercontent.com/FormingWorlds/PROTEUS/master/docs/images/PROTEUS_black.png#gh-dark-mode-only)

# PROTEUS Framework for Planetary Evolution

**PROTEUS** is a Python framework that simulates the coupled evolution
of the atmospheres and interiors of rocky planets.

## Installation instructions

Click [here](https://fwl-proteus.readthedocs.io/en/latest/installation.html) for steps and troubleshooting advice.

## Run instructions

Only attempt to run PROTEUS after you have followed all of the installation instructions.
If using a fresh shell, it is necessary to perform the following steps:
1. `conda activate proteus` (if using Anaconda/Miniconda)
2. `source PROTEUS.env`
Then you can start the model by running: `start_proteus.py`. See the ReadTheDocs pages for more information.

## Updating the code

Run `git submodule update --recursive --remote`

## Contributors

| Name  | Email address |
| -     | -             |
Tim Lichtenberg         | tim.lichtenberg[at]rug.nl |
Harrison Nicholls       | harrison.nicholls[at]physics.ox.ac.uk |
Laurent Soucasse        | l.soucasse[at]esciencecenter.nl |
Stef Smeets             | s.smeets[at]esciencecenter.nl |
Dan J. Bower            | dbower[at]ethz.ch |
Mariana V. Sastre       | m.c.villamil.sastre[at]rug.nl |
Emma Postolec           | e.n.postolec[at]rug.nl |
Mark Hammond            | mark.hammond[at]physics.ox.ac.uk |
Patrick Sanan           | patrick.sanan[at]gmail.com |
Raymond Pierrehumbert   | raymond.pierrehumbert[at]physics.ox.ac.uk |
Ryan Boukrouche         | ryan.boukrouche[at]astro.su.se |
Shang-Min Tsai          | shangmin.tsai[at]ucr.edu |
Hamish Innes            | hamish.innes[at]fu-berlin.de |

## Repository structure

| Object                | Description                                               |
| -                     | -                                                         |
| `start_proteus.py`    | Main PROTEUS Python script                                |
| `README.md`           | Overview file                                             |
| `pyproject.toml`	    | Project configuration file                                |
| `CODE_OF_CONDUCT.md`	| Project code of conduct                                   |
| `LICENSE.txt`         | Project license                                           |
| `src/proteus`         | Source code for PROTEUS                                   |
| `JANUS/`              | Submodule JANUS                                           |
| `SPIDER/`             | Submodule SPIDER                                          |
| `VULCAN/`             | Submodule VULCAN                                          |
| `Mors/`               | Submodule Mors                                            |
| `output/`             | Output folder with subfolders for each model run          |
| `input/`              | Input folder (e.g. stellar spectra, example configs)      |
| `docs/`			          | Documentation source files                                |
| `examples/`           | Example cases that the model should be able to reproduce  |
