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

### Repository structure
* `CouplerMain.py`  – Main PROTEUS Python script
* `README.md`       – This file
* `INSTALL.md`      – Installation instructions
* `AEOLUS/`         – Submodule AEOLUS
* `SPIDER/`         – Submodule SPIDER
* `VULCAN/`         – Submodule VULCAN
* `utils/`          – Utility python scripts
* `output/`         – Output folder
* `docs/`			– Documentation source
* `pyproject.toml`	– Documentation file


### Installation instructions
See https://proteus-code.readthedocs.io/en/latest/installation.html for steps and troubleshooting advice.

### Run instructions
Only attempt to run PROTEUS after you have followed all of the installation instructions.    
If using a fresh shell, it is necessary to perform the following steps:     
1. `conda activate proteus` (if installed with `conda`)
2. `source PROTEUS.env`
Then you can run the code by running: `python CouplerMain.py`      
By default, plots are produced in `output/`      

### Updating the code
Run `git submodule update --recursive --remote`
