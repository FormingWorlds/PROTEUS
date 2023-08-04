## PROTEUS Framework for Planetary Evolution
[![Documentation Status](https://readthedocs.org/projects/proteus-code/badge/?version=latest)](https://proteus-code.readthedocs.io/en/latest/?badge=latest) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

### Documentation
https://proteus-code.readthedocs.io

### Contact
https://proteus-code.readthedocs.io/en/latest/contact.html

### Contributors
* TL – Tim Lichtenberg (tim.lichtenberg@rug.nl)
* MH – Mark Hammond (mark.hammond@physics.ox.ac.uk)
* DJB – Dan J. Bower (daniel.bower@csh.unibe.ch)
* PS – Patrick Sanan (patrick.sanan@gmail.com)
* RTP – Raymond Pierrehumbert (raymond.pierrehumbert@physics.ox.ac.uk)
* RB – Ryan Boukrouche (ryan.boukrouche@astro.su.se)
* SMT – Shang-Min Tsai (shangmin.tsai@ucr.edu)
* HN – Harrison Nicholls (harrison.nicholls@physics.ox.ac.uk)
* HII - Hamish Innes (hamish.innes@physics.ox.ac.uk)

### Repository structure
* `proteus.py`      – Main PROTEUS Python script
* `README.md`       – This file
* `AEOLUS/`         – Submodule AEOLUS
* `SPIDER/`         – Submodule SPIDER
* `VULCAN/`         – Submodule VULCAN
* `Mors/`           - Submodule Mors
* `utils/`          – Scripts used for running PROTEUS and its submodules
* `plot/`           - Plotting scripts
* `output/`         – Output folder (e.g. planet evolution results, plots)
* `input/`          - Input folder (e.g. stellar spectra, example configs)
* `docs/`			– Documentation source
* `examples/`       - Example model results
* `pyproject.toml`	– Documentation file


### Installation instructions
See https://proteus-code.readthedocs.io/en/latest/installation.html for steps and troubleshooting advice.

### Run instructions
Only attempt to run PROTEUS after you have followed all of the installation instructions.    
If using a fresh shell, it is necessary to perform the following steps:     
1. `conda activate proteus` (if installed with `conda`)
2. `source PROTEUS.env`    
Then you can start the model by running: `python proteus.py`.      
Results are produced in folders within `output/`.      

### Updating the code
Run `git submodule update --recursive --remote`
