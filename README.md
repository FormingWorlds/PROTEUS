## PROTEUS Framework for Planetary Evolution
[![Documentation Status](https://readthedocs.org/projects/proteus-code/badge/?version=latest)](https://proteus-code.readthedocs.io/en/latest/?badge=latest) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

### Documentation
https://proteus-code.readthedocs.io

### Installation instructions   
See https://proteus-code.readthedocs.io/en/latest/installation.html for steps and troubleshooting advice.

### Run instructions
Only attempt to run PROTEUS after you have followed all of the installation instructions.    
If using a fresh shell, it is necessary to perform the following steps:     
1. `conda activate proteus` (if using Anaconda/Miniconda)     
2. `source PROTEUS.env`      
Then you can start the model by running: `proteus.py`. See the ReadTheDocs pages for more information.    

### Updating the code
Run `git submodule update --recursive --remote`

### Contributors
| Name  | Email address |
| -     | -             |
Tim Lichtenberg         | tim.lichtenberg[at]rug.nl | 
Harrison Nicholls       | harrison.nicholls[at]physics.ox.ac.uk |
Laurent Soucasse        | l.soucasse[at]esciencecenter.nl |
Dan J. Bower            | dbower[at]ethz.ch |
Mark Hammond            | mark.hammond[at]physics.ox.ac.uk |
Patrick Sanan           | patrick.sanan[at]gmail.com |
Raymond Pierrehumbert   | raymond.pierrehumbert[at]physics.ox.ac.uk |
Ryan Boukrouche         | ryan.boukrouche[at]astro.su.se |
Shang-Min Tsai          | shangmin.tsai[at]ucr.edu |
Hamish Innes            | hamish.innes[at]fu-berlin.de |

### Repository structure
| Object                | Description                                               |
| -                     | -                                                         |
| `proteus.py`          | Main PROTEUS Python script                                |
| `README.md`           | Overview file                                             |
| `pyproject.toml`	    | Documentation file                                        |
| `CODE_OF_CONDUCT.md`	| Project code of conduct                                   |
| `LICENSE.txt`         | Project license                                           |
| `JANUS/`              | Submodule JANUS                                          |
| `SPIDER/`             | Submodule SPIDER                                          |
| `VULCAN/`             | Submodule VULCAN                                          |
| `Mors/`               | Submodule Mors                                            |
| `utils/`              | Code used for running PROTEUS and its submodules          |
| `plot/`               | Code used for plotting results                            |
| `output/`             | Output folder with subfolders for each model run          |
| `input/`              | Input folder (e.g. stellar spectra, example configs)      |
| `docs/`			    | Documentation source files                                |
| `examples/`           | Example cases that the model should be able to reproduce     |


