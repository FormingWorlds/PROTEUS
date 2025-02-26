# Installation

These instructions will guide you through the typical installation
process. The setup is written for MacOS and Linux. Depending on your
system settings and installed libraries your procedure may differ. If
one or more of the steps below do not work for you we encourage you to
first check the `troubleshooting`{.interpreted-text role="doc"} page. If
that does not help you further, please contact the developers (see
`contact`{.interpreted-text role="doc"}).

## System configuration (MacOS)

1.  Open the terminal to install the developer tools

```console
xcode-select --install
```

2.  Install `FORTRAN NetCDF` library via the most appropriate method for
    you

```console
brew install netcdf
brew install netcdf-fortran
```

Or

```console
sudo port install netcdf-fortran +gcc8
```

## System configuration (Linux)

1. Install `FORTRAN NetCDF` via your package manager (e.g\...)

```console
sudo apt install libnetcdff-dev
```

## Python environment

You will need to install Python (>=3.11) on your system.
This can be done via brew on MacOS, or with your package manager on Linux. Alternatively, you can use [miniforge](https://github.com/conda-forge/miniforge) or [pyenv](https://github.com/pyenv/pyenv).

## Install Julia

If you already have Julia installed, skip this step.
This should only be done using the official installer, not via your package manager.

    ```console
    curl -fsSL https://install.julialang.org | sh
    ```

## Download the framework

1. Setup environment variables

    The environment variable `FWL_DATA` points to the folder where input data are stored.
    This variable must always be set, so it is best to add this line to your shell rc file.

    ```console
    export FWL_DATA=/your/local/path/
    ```

2. Download PROTEUS base

    ```console
    git clone git@github.com:FormingWorlds/PROTEUS.git
    cd PROTEUS
    ```

3. Create a virtual environment

    ```console
    python -m venv .venv
    source .venv/bin/activate
    ```

4. Setup radiative transfer code (**SOCRATES**)

    The code can be setup in `./socrates/` using the following script.

    ```console
    source tools/get_socrates.sh
    ```

    The environemnt variable `RAD_DIR` must always point to the SOCRATES installation path.
    It is highly recommended that you add this to your shell rc file (e.g. `~/.bashrc`).

5. Radiative-convective atmosphere model (**AGNI**)

    Installation steps can be found at the [AGNI wiki](https://nichollsh.github.io/AGNI/dev/setup/).
    They are also reproduced below.

    ```console
    git clone git@github.com:nichollsh/AGNI.git
    cd AGNI
    bash src/get_agni.sh
    cd ../
    ```

    Use this `get_agni` script to keep AGNI and its data files up to date.

6. **Optional** developer installation steps

    Follow the steps in this section if you want to create editable installations of these submodules.
    Otherwise, go to the next step section.

    - MORS stellar evolution model

        ```console
        git clone git@github.com:FormingWorlds/MORS
        pip install -e MORS/.
        ```

    - JANUS atmosphere model

        ```console
        git clone git@github.com:FormingWorlds/JANUS
        pip install -e JANUS/.
        ```

    - CALLIOPE outgassing model

        ```console
        git clone git@github.com:FormingWorlds/CALLIOPE
        pip install -e CALLIOPE/.
        ```

    - ARAGOG interior model

        ```console
        git clone git@github.com:ExPlanetology/aragog.git
        pip install -e aragog/.
        ```

    - ZEPHYRUS escape model

        ```console
        git clone git@github.com:FormingWorlds/ZEPHYRUS
        pip install -e ZEPHYRUS/.
        ```

7. Setup numerical computing library (**PETSc**)

    You will need to do this particular step in an environment with Python <= 3.12.

    ```console
    ./tools/get_petsc.sh
    ```

8. Setup interior evolution model (**SPIDER**)

    ```console
    ./tools/get_spider.sh
    ```

9. Setup PROTEUS coupled framework

    ```console
    pip install -e .
    ```


10. Done! 🚀
    Any remaining dependencies will be downloaded when the model is first run.

## Optional modules


### Multi-phase tidal heating model (**LovePy**)

Lovepy is written in Julia. You can use the same environment as AGNI if you wish, but you
should make sure to follow the installation steps below.

    ```console
    ./tools/get_lovepy.sh
    ```

### Synthetic observations calculator (**PLATON**)

[PLATON](https://platon.readthedocs.io/en/latest/intro.html) is a forward modelling and
retrieval tool for exoplanet atmospheres. In PROTEUS, this is used to generate synthetic
transmission and secondary eclipse observations. To get PLATON, use the script below.

    ```console
    ./tools/get_platon.sh
    ```

    Note that this script will take some time to run, as PLATON will first need to download
    about 10 GB of opacity data from the internet.

### Chemical kinetics atmosphere model (**VULCAN**)

1. Clone the model

    ```console
    git clone git@github.com:exoclime/VULCAN.git
    cd VULCAN
    ```

2. Compile the FastChem extension

    ```console
    cd fastchem_vulcan
    ```

    On MacOS you will need to edit `make.globaloptions` to reflect  a GNU-compatible `g++` executable, not the Apple one (see
     [Troubleshooting](./troubleshooting.md) if the next step results in an error.

    ```console
    make
    cd ../../
    ```
3. Install Python dependencies

    ```console
    pip install sympy
    ```
