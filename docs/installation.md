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

You will need to setup Python (>=3.10) on your system. This can be done via brew on MacOS, or with your package manager on Linux. Alternatively, you can use [miniforge](https://github.com/conda-forge/miniforge) or [pyenv](https://github.com/pyenv/pyenv).

## Download the framework

1. Register your public SSH key with Github:

    - <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys>
    - <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>
    - <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>
    - <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection>

2. Setup environment variables

    The environment variable `FWL_DATA` points to the folder where input data are stored.
    This variable must always be set, so it is best to add this line to your shell rc file.

    ```console
    export FWL_DATA=/your/local/path/
    ```

3. Download PROTEUS base

    ```console
    git clone git@github.com:FormingWorlds/PROTEUS.git
    ```

4. Get dependencies

    ```console
    cd PROTEUS
    git submodule update --init --recursive
    ```

5. Create a virtual environment

    ```console
    python -m venv .venv
    source .venv/bin/activate
    ```

6. Setup radiative transfer code (**SOCRATES**)

    The code can be setup in `./socrates/` using the following script.

    ```console
    source get_socrates.sh
    ```

7. Setup numerical computing library (**PETSc**)

    - Configure step

        ```console
        cd petsc
        ./configure --with-debugging=0 --with-fc=0 --with-cxx=0 --download-sundials2 --download-mpich --download-f2cblaslapack --COPTFLAGS="-g -O3" --CXXOPTFLAGS="-g -O3"
        ```

    - Run the exact `make all` command provided at the end of the configure step
    - Run the exact `make check` command provided at the end of the `make all` step

    ```console
    cd ..
    ```

8. **Optional** developer installation steps

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

9. Setup PROTEUS coupled framework

    - Get the remaining Python dependencies

        ```console
        pip install -e .
        ```

    - Configure environment variables

        - The variable `RAD_DIR` must point to the SOCRATES installation path. It is best to add this to your shell rc file.

        - Setup other variables

            ```console
            source PROTEUS.env
            ```
            This file can also be added to your shell rc file.

10. Setup interior evolution model (**SPIDER**)

    ```console
    cd SPIDER
    make clean
    make -j
    cd ..
    ```

11. Done! 🚀

Any remaining dependencies will be downloaded when the model is first run.

## Optional modules

### Radiative-convective atmosphere model (**AGNI**)

Installation steps can be found at the [AGNI wiki](https://nichollsh.github.io/AGNI/dev/setup/).
They are also reproduced below.

1. Setup Julia

    ```console
    curl -fsSL https://install.julialang.org | sh
    ```

2. Clone the model

    ```console
    git clone git@github.com:nichollsh/AGNI.git
    cd AGNI
    ```

3. Build AGNI

    ```console
    julia -e 'using Pkg; Pkg.activate("."); Pkg.build()'
    ```

4. Go back to the PROTEUS directory

    ```console
    cd ../
    ```

Consult the AGNI wiki if you encouter issues.

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
