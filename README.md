## COUPLED MODEL DESCRIPTION

Run *RadInteriorCoupler.py*.

It runs until the OLR changes by less than a threshold value in W/m<sup>2</sup>, or stops after a fixed number of iterations.

ABBREVIATIONS:
* MH - Mark Hammond
* TL - Tim Lichtenberg
* PS – Patrick Sanan
* RP - Ray Pierrehumbert
* DB - Dan Bower
* ST - Shang-Min Tsai

#### COMMUNICATION

###### LaTeX file for discussion/draft

Read-only: https://www.overleaf.com/read/cbbvgpyttcqm

Edit: https://www.overleaf.com/8481699571kkycnphnmgdg

###### LaTeX file for PAPER I

Read-only: https://www.overleaf.com/read/fwqkyfcrfftb

Edit: https://www.overleaf.com/6487644842trtffrfrfxmj

###### Slack channel for day-to-day use

http://bit.ly/2LvB1FR

--> Channel: #interior-atmosphere

## FILE DESCRIPTIONS

* GGRadConv.py - Grey gas radiative convective model
* GreyHeat.py - Calculates heating rates in grey model
* OLRFlux.txt - Outgoing heat flux in W, coupler-file
* SocRadConv.py - Radiative convective model w/ SOCRATES
* SocRadModel.py - Calculates heating rates in for SOCRATES version
* atmosphere_column.py - Class for atmospheric column data
* nctools.py - Some useful functions for netCDF
* phys.py - Constants
* planets.py - Planet-specific constants
* profile.* -
* README.md - This file
* surfaceT.txt - Surface temperature form interior model, coupler-file

## INSTALLATION

1. Ensure that submodules are up to date

        git submodule update --init --recursive

1. Install SPIDER, making sure that the `spider` executable's location is in `$PATH`

1. Get access to SOCRATES: https://code.metoffice.gov.uk/trac/home
    * For website access talk to MH
    * Download the latest version, unzip to /socrates/socrates_main/

1. Install dependencies:
    * Make sure you have the FORTRAN version of netCDF installed:
        * e.g. with Homebrew, $ brew install netcdf
        * e.g. for MacPorts with GCC 8, $ sudo port install netcdf-fortran +gcc8
    * Same for netCDF python:
        * e.g. with Anaconda, $ conda install netcdf4 natsort
        * e.g. for MacPorts with Python 3.7, $ sudo port install py37-netcdf py37-natsort
        * make sure you use a Python 3 environment, possibly need to reinstall all typical packages, see e.g., here: http://bit.ly/2HowQaA
    * Relaunch terminal window, or source bash_profile, to reset environment

1. Compile SOCRATES:
    - Overwrite Mk_cmd in /socrates/socrates_main/make/:
        $ cd /socrates/ && cp -rf Mk_cmd /socrates_main/make/
        With MacPorts, change "usr" to "opt" in this file, to find the netcdf module
    - Build SOCRATES:
        $ cd /socrates/socrates_main/
        $ ./build_code
    - Set environment:
        $ cd /socrates/socrates_main/
        $ . ./set_rad_env # Ignore "-bash: ulimit: ... " error
    - If you want SOCRATES to be readily available in every new shell, put the following in you .bash_profile:
        $ source /PATH_TO_COUPLER/socrates/socrates_main/set_rad_env

1. Run code:
    $ python CouplerMain.py
    # Examine plots produced in the output/ directory
