## PROTEUS (ATMOSPHERE–INTERIOR COUPLER)

### Contributors (abbreviations & email addresses):
* TL - Tim Lichtenberg (tim.lichtenberg@rug.nl)
* MH - Mark Hammond (mark.hammond@physics.ox.ac.uk)
* DJB - Dan J. Bower (daniel.bower@csh.unibe.ch)
* PS – Patrick Sanan (patrick.sanan@erdw.ethz.ch)
* RTP - Ray Pierrehumbert (raymond.pierrehumbert@physics.ox.ac.uk)
* RB – Ryan Boukrouche (ryan.boukrouche@astro.su.se)
* SMT - Shang-Min Tsai (shangmin.tsai@ucr.edu)
* HN - Harrison Nicholls (harrison.nicholls@physics.ox.ac.uk)

### Documentation (under construction):
https://proteus-code.readthedocs.io/en/latest/

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
* `lumache.py`		– Documentation file


### Installation instructions
See INSTALL.md for steps.

### Run instructions
Only attempt to run PROTEUS after you have followed all of the instructions in INSTALL.md    
If using a fresh shell, it is necessary to perform the following steps:     
1. `conda activate proteus`
2. `source PROTEUS.env`
Then you can run the code by running: `python CouplerMain.py`      
Plots are produced in `output/`      

### Updating the code
Run `git submodule update --recursive --remote`

