## PROTEUS Framework for Planetary Evolution
[![Documentation Status](https://readthedocs.org/projects/proteus-code/badge/?version=latest)](https://proteus-code.readthedocs.io/en/latest/?badge=latest)

### Documentation
https://proteus-code.readthedocs.io

### Contributors
* TL - Tim Lichtenberg (tim.lichtenberg@rug.nl)
* MH - Mark Hammond (mark.hammond@physics.ox.ac.uk)
* DJB - Dan J. Bower (daniel.bower@csh.unibe.ch)
* PS – Patrick Sanan (patrick.sanan@gmail.com)
* RTP - Raymond Pierrehumbert (raymond.pierrehumbert@physics.ox.ac.uk)
* RB – Ryan Boukrouche (ryan.boukrouche@astro.su.se)
* SMT - Shang-Min Tsai (shangmin.tsai@ucr.edu)
* HN - Harrison Nicholls (harrison.nicholls@physics.ox.ac.uk)

### Repository structure
* `CouplerMain.py`  – Main PROTEUS Python script
* `README.md`       – This file
* `INSTALL.md`      – Installation instructions
* `AEOLUS/`         – Submodule AEOLUS
* `SPIDER/`         – Submodule SPIDER
* `VULCAN/`         – Submodule VULCAN
* `utils/`          – Utility python scripts
* `output/`         – Output folder
* `docs/`			      – Documentation source
* `pyproject.toml`	– Documentation file
* `lumache.py`		  – Documentation file


### Installation instructions
See INSTALL.md for steps.

### Run instructions
Only attempt to run PROTEUS after you have followed all of the instructions in INSTALL.md    
If using a fresh shell, it is necessary to perform the following steps:     
1. `conda activate proteus` (if installed with `conda`)
2. `source PROTEUS.env`
Then you can run the code by running: `python CouplerMain.py`      
Plots are produced in `output/`      

### Updating the code
Run `git submodule update --recursive --remote`

