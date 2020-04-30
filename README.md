## ATMOSPHERE–INTERIOR COUPLER

### People abbreviations & contact:
* TL - Tim Lichtenberg (tim.lichtenberg@physics.ox.ac.uk)
* MH - Mark Hammond (mark.hammond@physics.ox.ac.uk)
* DJB - Dan J. Bower (daniel.bower@csh.unibe.ch)
* PS – Patrick Sanan (patrick.sanan@erdw.ethz.ch)
* RTP - Ray T. Pierrehumbert (raymond.pierrehumbert@physics.ox.ac.uk)
* RB – Ryan Boukrouche (ryan.boukrouche@physics.ox.ac.uk)
* SMT - Shang-Min Tsai (shang-min.tsai@physics.ox.ac.uk)

#### Communication

##### SLACK CHANNEL for day-to-day use

http://bit.ly/2LvB1FR
--> Channel: #interior-atmosphere

##### LaTeX file for discussion

Edit: https://www.overleaf.com/8481699571kkycnphnmgdg

Read-only: https://www.overleaf.com/read/cbbvgpyttcqm

##### LaTeX file for PAPER I

Edit: https://www.overleaf.com/6487644842trtffrfrfxmj

Read-only: https://www.overleaf.com/read/fwqkyfcrfftb



## Repository structure

* *CouplerMain.py* – Main COUPLER python script
* *README.md* – *This* file
* *atm_rad_conv/* – Submodule RAD_CONV
* *spider-dev/* – Submodule SPIDER
* *vulcan_spider/* – Submodule VULCAN
* *utils/* – Utility python scripts

## Access & installation instructions

1. Install dependencies
    
    * Make sure you use a Python 3 environment, possibly need to reinstall all typical packages, see e.g., here: http://bit.ly/2HowQaA
    
    * Required packages:
        
        * FORTRAN, local:
        
            - netCDF
                
                e.g. with Homebrew
                    
                    brew install netcdf
            
                e.g. with MacPorts using GCC 8
                    
                    sudo port install netcdf-fortran +gcc8
    
        * Python, local:

            - netcdf4

                e.g. with Anaconda

                    conda install netcdf4 natsort

                e.g. with MacPorts (and Python 3.7 in this case)

                    sudo port install py37-netcdf py37-natsort

        * AOPP cluster:

            * Load the following modules to run the Coupler:

                    module load intel-compilers/2012 netcdf/3.6.3 openmpi/1.6.5__intel2012

            * Create a virtual Python 3 environment

                    mkdir temporary-conda-directory
                    cd temporary-conda-directory
                    bash
                    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
                    bash Miniconda3-latest-Linux-x86_64.sh
                    conda create -n cs-env python=3.6.6
                    conda activate cs-env
                    conda install -c conda-forge f90nml

            * Install the required packages

                    conda install matplotlib numpy pandas scipy seaborn natsort

            * Register them in your .bash_profile

                    module load intel-compilers intel-compilers/2012 netcdf/3.6.3 openmpi/1.6.5__intel2012
                    conda activate cs-env
    
    * *Optional:* Update your local git installation's repository access:
        
        * Register your public SSH key with Github & Bitbucket. E.g, generate key with:

                ssh-keygen -t rsa

        * And copy with:

                cat ~/.ssh/id_rsa.pub

        * Now add "New SSH key" in the settings at:
        
        https://github.com/settings/keys & 
        https://bitbucket.org/account/settings/ssh-keys/

    * Relaunch terminal window, or source .bash_profile, to reset environment

            source ~/.bash_profile

1. Get access to all necessary codes, repositories, & submodules
    - Atmosphere-interior **COUPLER** (*this repository*) 
        
        * URL: https://github.com/OxfordPlanetaryClimate/couple-interior-atmosphere

    - Radiative-convective scheme: **RAD_CONV** 

        * URL: https://github.com/OxfordPlanetaryClimate/soc-rad-conv
        * Contact: TL, MH, RB

    - Radiation transport: **SOCRATES** 

        * URL: https://code.metoffice.gov.uk/trac/socrates
        * Contact: MH, TL
        * Download the latest version, e.g. *socrates_2002.tar.xz*

    - Interior dynamics: **SPIDER** 

        * URL: https://bitbucket.org/djbower/spider-dev/
        * Contact: DJB, PS

    - Atmospheric chemistry: **VULCAN** (COUPLER-compatible version) 

        * URL: https://github.com/shami-EEG/VULCAN-SPIDER
        * Contact: SMT, TL

1. Build compiled codes & modules in order

    1. COUPLER + submodules (*atm_rad_conv, spider-dev, vulcan_spider*)

            git clone --recursive git@github.com:OxfordPlanetaryClimate/couple-interior-atmosphere.git PATH_TO_COUPLER

    1. Ensure that submodules are up to date

            git submodule update --init --recursive

    1. SOCRATES / rad_conv submodule

        * Copy SOCRATES' files to the corresponding subdirectory, e.g.:

                cp -r PATH_TO_UNZIPPED_socrates_2002 PATH_TO_COUPLER/atm_rad_conv/rad_trans/socrates_code/

        * Overwrite */make/Mk_cmd* with the right setup for your machine:
        
                cp -rf PATH_TO_COUPLER/atm_rad_conv/rad_trans/Mk_cmd_MAC PATH_TO_COUPLER/atm_rad_conv/rad_trans/socrates_code/make/Mk_cmd

            * Options: *Mk_cmd_MAC*, *Mk_cmd_AOPP_CLUSTER*
        
            * If you installed your dependencies above with *MacPorts*, change *usr* to *opt* in this file, to find the correct netCDF module.
    
        * Build SOCRATES & set its environment
            
                cd PATH_TO_COUPLER/atm_rad_conv/rad_trans/socrates_code
            
                ./build_code

                source /PATH_TO_COUPLER/atm_rad_conv/rad_trans/socrates_code/set_rad_env

        * Ignore the *-bash: ulimit...* error and learn to live with it :)

    1. SPIDER submodule

        * Follow the spider-dev README for up-to-date installation instructions

        * Register the PETSC settings in your .bash_profile (you'll get the correct locations during installation of the hacked PETSC version), may look like this:

            * Local:

                    export PETSC_DIR=/Users/tim/bitbucket/petsc-quad-direct
            
                    export PETSC_ARCH=arch-darwin-c-opt

            * AOPP cluster:

                    export PETSC_DIR=/home/lichtenberg/codes/petsc-double-direct
                
                    export PETSC_ARCH=arch-linux2-c-opt

        * If you want this to be your primary SPIDER installation, register its directory to .bash_profile via adding the following

                SPIDER_DIR=$COUPLER_DIR/spider-dev

                export PATH=$SPIDER_DIR:$PATH

                export PATH=$SPIDER_DIR/py3:$PATH

                export PYTHONPATH=$SPIDER_DIR/py3:$PYTHONPATH

    1. Register the COUPLER in your PATH, i.e., add to .bash_profile/.bashrc the following:

        * Mandatory:

                COUPLER_DIR=/path/to/coupler

                export PATH=$COUPLER_DIR:$PATH

                export PYTHONPATH=$COUPLER_DIR:$PYTHONPATH

        * Optional (for plotting etc.): 
                
                export PATH=$COUPLER_DIR/utils:$PATH

                export PYTHONPATH=$COUPLER_DIR/utils:$PYTHONPATH

                export PYTHONPATH=$COUPLER_DIR/vulcan_spider:$PYTHONPATH

                export PYTHONPATH=$COUPLER_DIR/atm_rad_conv:$PYTHONPATH
            
                source /PATH_TO_COUPLER/rad_trans/socrates_main/set_rad_env

        * Relaunch terminal window, or source .bash_profile, to reset environment

                source ~/.bash_profile

1. Run COUPLER (in any directory) using:

        CouplerMain.py
    
    Examine plots produced in *./output/*

1. Update COUPLER recursively (when needed):

        git submodule update --recursive --remote
