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

## Setup a Python environment

We recommend that you use Python version 3.12 for running PROTEUS.
Python is most easily obtained and managed using [miniforge](https://github.com/conda-forge/miniforge).
Alternatively, you can download it from the official website and use [pyenv](https://github.com/pyenv/pyenv).

## Install Julia

Some of the PROTEUS modules are written in Julia, which is broadly similar to Python.
You should only obtain Julia using the official installer, **not** via your package manager.

```console
curl -fsSL https://install.julialang.org | sh
```

## Set up the PROTEUS framework

1. Create and set environment variables

    The environment variable `FWL_DATA` points to the folder where input data are stored.
    This variable must always be set, so it is best to add this line to your shell rc file.

    ```console
    mkdir /your/local/path/FWL_DATA
    echo "export FWL_DATA=/your/local/path/FWL_DATA/" >> "$HOME/.bashrc"
    ```

    Reload your shell rc file to make the changes effective.

    ```console
    source "$HOME/.bashrc"
    ```

2. Download PROTEUS base

    ```console
    git clone git@github.com:FormingWorlds/PROTEUS.git
    cd PROTEUS
    ```

3. Create a virtual environment

    ```console
    conda create -n proteus python=3.12
    ```

4. Radiative transfer code (**SOCRATES**)

    The code can be setup in `./socrates/` using the following script.

    ```console
    ./tools/get_socrates.sh
    ```

    The environment variable `RAD_DIR` must always point to the SOCRATES installation path.
    It is highly recommended that you add this to your shell rc file (e.g. `~/.bashrc`), which
    can be done by running the following command.

    ```console
    echo "export RAD_DIR=$PWD/socrates/" >> "$HOME/.bashrc"
    ```

    Reload your shell rc file to make the changes effective.

    ```console
    source "$HOME/.bashrc"
    ```

5. Radiative-convective atmosphere model (**AGNI**)

    Installation steps can be found at the [AGNI wiki](https://nichollsh.github.io/AGNI/dev/setup/).
    They are also reproduced below.

    ```console
    git clone git@github.com:nichollsh/AGNI.git
    cd AGNI
    bash src/get_agni.sh
    cd ../
    ```

    Use this `get_agni.sh` script to keep AGNI and its data files up to date. It is
    important that AGNI is available at `./AGNI/` inside your PROTEUS folder. This can be
    either a symbolic link or the true location of the submodule.

6. **Optional** developer installation steps

    Follow the steps in this section if you want to create editable installations of these submodules.
    Otherwise, go to the next step section.

    - MORS stellar evolution model

        ```console
        git clone git@github.com:FormingWorlds/MORS
        python -m pip install -e MORS/.
        ```

    - JANUS atmosphere model

        ```console
        git clone git@github.com:FormingWorlds/JANUS
        python -m pip install -e JANUS/.
        ```

    - CALLIOPE outgassing model

        ```console
        git clone git@github.com:FormingWorlds/CALLIOPE
        python -m pip install -e CALLIOPE/.
        ```

    - ARAGOG interior model

        ```console
        git clone git@github.com:ExPlanetology/aragog.git
        python -m pip install -e aragog/.
        ```

    - ZEPHYRUS escape model

        ```console
        git clone git@github.com:FormingWorlds/ZEPHYRUS
        python -m pip install -e ZEPHYRUS/.
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
    python -m pip install -e .
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
retrieval tool for exoplanet atmospheres. In PROTEUS, this tool is used to generate
synthetic transmission and secondary eclipse observations. To get PLATON and its
dependencies, use the script below.

```console
./tools/get_platon.sh
```

Note that this script will take some time to run; PLATON will need to download
about 10 GB of opacity data from the internet.

### Chemical kinetics atmosphere model (**VULCAN**)

1. Clone the model

    ```console
    git clone git@github.com:nichollsh/VULCAN.git
    cd VULCAN
    ```

2. Compile the FastChem extension

    ```console
    cd fastchem_vulcan
    ```

    On MacOS you will need to edit `make.globaloptions` to reflect  a GNU-compatible `g++` executable, not the Apple one (see
     [Troubleshooting](./troubleshooting.md)), if the next step results in an error.

    ```console
    make
    cd ../../
    ```
