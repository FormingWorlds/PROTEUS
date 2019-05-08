## Coupled model, description

Run *RadInteriorCoupler.py*.

It runs until the OLR changes by less than 1W/m<sup>2</sup>, or stops after 50 iterations.

ABBREVIATIONS:
Mark Hammond - MH
Tim Lichtenberg - TL
Ryan Boukrouche - RB

## INSTALLATION FOR DUMMIES (mostly by and for TL)

(1) Get access to SOCRATES: https://code.metoffice.gov.uk/trac/home
    - For website access talk to MH
    - Download the latest version, unzip to /socrates/socrates_main/

(2) Install dependencies:
    - Make sure you have the FORTRAN version of netCDF installed:
        e.g., $ brew install netcdf
    - Same for netCDF python:
        e.g., $ conda install netcdf
    - Relaunch terminal window, or source bash_profile, to reset environment

(3) Compile SOCRATES:
    - Overwrite Mk_cmd in /socrates/socrates_main/make/:
        $ cd /socrates/ && cp -rf Mk_cmd /socrates_main/make/
    
    - Build SOCRATES:
        $ cd /socrates/socrates_main/
        $ ./build_code
    - Set environment (EVERYTIME SOCRATES IS USED, or otherwise set in environment variables):
        $ cd /socrates/socrates_main/
        $ . ./set_rad_env

(4) Run code:
    $ python RadInteriorCoupler.py


