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
* `AEOLUS/`         – Submodule AEOLUS
* `SPIDER/`         – Submodule SPIDER
* `VULCAN/`         – Submodule VULCAN
* `utils/`          – Utility python scripts
* `output/`         – Output folder
* `docs/`			– Documentation source
* `pyproject.toml`	– Documentation file


### NEW pip installation instructions
Testing out a new way of installation using pip install. So far this has been tested on an Apple Silicon Mac and
the AOPP cluster.

Before installing, set up the conda environment as described in the [documentation](https://proteus-code.readthedocs.io/en/latest/installation.html). Run `conda activate proteus` once you're done with this step.

Clone the PROTEUS git repository [as described in the current documentation](https://proteus-code.readthedocs.io/en/latest/installation.html).

Ensure you have installed netcdf-fortran and GNU compilers (`gcc`, `g++` and `gfortran`) installed on your machine. If GNU compilers are the default compilers on your system, then installation should work OK out-of-box. If on Apple Silicon, often `gcc` and `g++` commands are symbolic links to `clang`. You can work out if the `gcc` and `g++` you are using are `clang` symlinks by running `gcc -v`. To ensure CMake finds the correct compilers, find where *GNU* `gcc` and `g++` are installed and export the following environment variables:
```
export CC=[/path/to/gcc]
export CXX=[/path/to/g++]
```
On an Apple M1 Pro, the homebrew-installed GNU compilers were found in `/opt/homebrew/bin/` as `gcc-13` and `g++-13`.

Now run `pip install -v .` in the PROTEUS home directory. The CMake-driven build process should automatically find NetCDF and download all the necessary software packages. If you get an error complaining about a missing NetCDF package, run `export NetCDF_ROOT=[path/to/netcdf]` and try again (CMake usually can find NetCDF in default installation locations - exporting NetCDF_ROOT tells CMake explicitly to look in that directory).

### Run instructions
Only attempt to run PROTEUS after you have followed all of the installation instructions.    
If using a fresh shell, it is necessary to perform the following steps:     
1. `conda activate proteus` (if installed with `conda`)
2. `source PROTEUS.env`
Then you can run the code by running: `python CouplerMain.py`      
By default, plots are produced in `output/`      

### Updating the code
Run `git submodule update --recursive --remote`
