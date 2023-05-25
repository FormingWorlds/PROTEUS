
## Access & installation instructions

* If any of these steps do not work or produce errors, check TROUBLESHOOTING.md

1. Install dependencies

    * MacOS: Ensure installation of command line developer tools
        * Open Xcode application and install
        * Type `xcode-select --install`

    * Install FORTRAN NetCDF library via the most appropriate method for you
        * `brew install netcdf`    
        * `brew install netcdf-fortran`     
        OR    
        * `sudo port install netcdf-fortran +gcc8`    
        OR     
        * `sudo apt install libnetcdff-dev`
    
    * Setup a Python environment:
        * Option A: Using the `Anaconda` package manager (careful, probably breaks on ARM machines/newer Macs)
            * Install `conda`:
                * Download the appropriate Miniconda installer from their website
            https://docs.conda.io/en/latest/miniconda.html#id36
                * Create a conda environment for PROTEUS:
                * `conda create -n proteus python=3.10.9`    
                * `conda activate proteus`
            * `conda install netcdf4 matplotlib numpy pandas scipy sympy natsort certifi`
            * `conda install -c conda-forge f90nml`
        * Option B: using the `brew` package manager (*recommended*)
            * Delete all traces of Anaconda package manager from your system and switch to a different Python environment, for example brew/pip
            * Follow the steps at https://docs.anaconda.com/free/anaconda/install/uninstall/
            * Delete all Anaconda-related entries from your .bash_profile (Intel) or .zshrc (ARM)
            * Install Python via `brew`: 
                * `brew install python`
                * Update to the latest stable version: `brew upgrade python`
                * Install `tkinter`: `brew install python-tk@3.11`
                * Refresh your shell / `source ~/.zsrhrc` (ARM) / `source ~/.bash_profile` (Intel)
                * Install all necessary packages: `pip3 install matplotlib pandas netcdf4 matplotlib numpy pandas scipy sympy natsort`
            * Make the new Python version the system default (check what `brew` tells you during/after the `brew install python` step):
                * ARM: `export PATH="/opt/homebrew/opt/python/libexec/bin:$PATH"`
                * Intel: `export PATH="/usr/local/opt/python/libexec/bin:$PATH"`

    * *Optionally* register your public SSH key with Github:
        * Generate key with `ssh-keygen -t rsa`
        * Copy to clipboard using `cat ~/.ssh/id_rsa.pub`
        * Add "New SSH key" in the settings at https://github.com/settings/keys 

2. Ensure that you have access to all of the following codes
    * Atmosphere-interior coupler: **PROTEUS**
        * URL: https://github.com/FormingWorlds/PROTEUS
        * Contact: TL

    * Radiative-convective scheme: **AEOLUS** 
        * URL: https://github.com/FormingWorlds/AEOLUS/
        * Contact: TL, MH, RB

    * Radiation transport: **SOCRATES** 
        * Main development URL: https://code.metoffice.gov.uk/trac/socrates
        * Contact: james.manners@metoffice.gov.uk
        * Obtain the SOCRATES source code from: https://simplex.giss.nasa.gov/gcm/ROCKE-3D/ (Latest released version of SOCRATES code)
        * Latest tested version: *socrates_2211.tar.xz*

    * Interior dynamics: **SPIDER** 
        * URL: https://github.com/djbower/spider
        * Contact: DJB, PS

    * Atmospheric chemistry: **VULCAN**
        * URL: https://github.com/exoclime/VULCAN/
        * Contact: SMT, TL

    * Scientific computing library: **PETSc**
        * URL: https://gitlab.com/petsc/petsc

    * Scientific application test utility: **SciATH**
        * URL: https://github.com/sciath/sciath
        * Contact: PS

    * Stellar XUV evolution code: **Mors**
        * URL: https://github.com/ColinPhilipJohnstone/Mors
        * Contact: colinjohnstone@gmail.com

3. Setup codes and modules in the following order (**ignore the instructions provided in their own repositories**)

    1. Download PROTEUS + submodules (*AEOLUS, SPIDER, VULCAN, PETSc, SciATH*)
        * `git clone --recursive git@github.com:FormingWorlds/PROTEUS.git`

    2. Enter into PROTEUS folder and ensure that submodules are up to date
        * `cd PROTEUS`
        * `git submodule update --init --recursive`

    3. Download and extract the SOCRATES archive to the correct location
        * `cd AEOLUS/rad_trans/socrates_code/`
        * `curl -L -o ../socrates_2211.tar.xz https://www.dropbox.com/sh/ixefmrbg7c94jlj/AAChibnZU9PRi8pXdVxVbdj3a/socrates_2211.tar.xz?dl=1`
        * `tar --strip-components 1 -xvf ../socrates_2211.tar.xz -C ./`
        * `cp -f ../build_code_modified build_code`

    4. Overwrite the `Mk_cmd` file with the right setup for your machine
        * `cp -rf ../Mk_cmd_SYSTEM make/Mk_cmd`    
        * Options are:
            * `cp -rf ../Mk_cmd_MAC_INTEL make/Mk_cmd`
            * `cp -rf ../Mk_cmd_MAC_APPLESILICON make/Mk_cmd`
            * `cp -rf ../Mk_cmd_AOPP_CLUSTER make/Mk_cmd`
            
        The command `nf-config` might be helpful if none of these options work for you.

    5. Setup SOCRATES 
        * `./build_code`
        * `type ksh >/dev/null 2>&1 ||  sed -i 's/ksh/bash/g' sbin/* `
        * `cd ../../../`

    6. Setup VULCAN
        * `cd VULCAN/fastchem_vulcan`
        * On MacOS you will need to edit `make.globaloptions` to reflect a GNU-compatible `g++` executable, not the Apple one
        * `make`
        * `cd ../../`

    7. Setup Mors
        * `cd Mors`
        * `pip install .`
        * `wget -c http://www.astro.yale.edu/demarque/fs255_grid.tar.gz -O - | tar -xz`
        * 
        
    8. Setup PETSc
        * `cd petsc`
        * `./configure --with-debugging=0 --with-fc=0 --with-cxx=0 --download-sundials2 --download-mpich --COPTFLAGS="-g -O3" --CXXOPTFLAGS="-g -O3"`
        * Make note of the value of `PETSC_ARCH` printed to stdout.
        * Run the exact `make all` command provided at the end of the configure step
        * Run the exact `make check` command provided at the end of the `make all` step
        * `cd ../`

    9. Setup environment variables
        * Edit the variable `PETSC_ARCH` in the file `PROTEUS.env` to reflect value provided by PETSc
        * If `python` has been installed via the`conda` route: `conda activate proteus`
        * `source PROTEUS.env`

    10. Setup SPIDER
        * `cd SPIDER`
        * `make clean`
        * `make -j`
        * `make test`, accepting all default values when prompted
        * `cd ../`

**Done!**
