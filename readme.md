## COUPLED MODEL DESCRIPTION

Run *RadInteriorCoupler.py*.

It runs until the OLR changes by less than a threshold value in W/m<sup>2</sup>, or stops after a fixed number of iterations.

ABBREVIATIONS:
* MH - Mark Hammond
* TL - Tim Lichtenberg
* RB - Ryan Boukrouche
* RP - Ray Pierrehumbert
* DB - Dan Bower

#### COMMUNICATION

###### LaTeX file for discussion/draft

Read-only: https://www.overleaf.com/read/cbbvgpyttcqm

Edit: https://www.overleaf.com/8481699571kkycnphnmgdg

###### Slack channel for day-to-day use
http://bit.ly/2LvB1FR

Channel: #interior-atmosphere

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
* readme.md - This file
* surfaceT.txt - Surface temperature form interior model, coupler-file

## INSTALLATION

1. Get access to SOCRATES: https://code.metoffice.gov.uk/trac/home
    * For website access talk to MH
    * Download the latest version, unzip to /socrates/socrates_main/

1. Install dependencies:
    * Make sure you have the FORTRAN version of netCDF installed:
        * e.g., $ brew install netcdf
    * Same for netCDF python:
        * e.g., $ conda install netcdf4
    * Relaunch terminal window, or source bash_profile, to reset environment

1. Compile SOCRATES:
    - Overwrite Mk_cmd in /socrates/socrates_main/make/:
        $ cd /socrates/ && cp -rf Mk_cmd /socrates_main/make/
    - Build SOCRATES:
        $ cd /socrates/socrates_main/
        $ ./build_code
    - Set environment (EVERYTIME SOCRATES IS USED, or otherwise set in environment variables):
        $ cd /socrates/socrates_main/
        $ . ./set_rad_env

1. Run code:
    $ python RadInteriorCoupler.py
