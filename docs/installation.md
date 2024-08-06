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

1. Download the appropriate Miniconda installer from
<https://docs.conda.io/en/latest/miniconda.html#id36>

2. Create a conda environment for PROTEUS:

```console
conda create -n proteus python=3.12   
conda activate proteus
pip install -e .
```

## Download the framework

1. Register your public SSH key with Github:

    - <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys>
    - <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>
    - <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>
    - <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection>

2. Download PROTEUS base

    ```console
    git clone git@github.com:FormingWorlds/PROTEUS.git
    ```

3. Enter into PROTEUS folder and ensure that submodules are up to date

    ```console
    cd PROTEUS
    git submodule update --init --recursive
    ```

4. Setup radiative transfer code (**SOCRATES**)

    ```console
    cd SOCRATES
    ./configure
    ./build_code
    cd ..
    ```

5. Setup atmosphere model (**JANUS**)

    ```console
    cd JANUS
    pip install -e .
    cd ..
    ```

6. Setup atmosphere kinetics model (**VULCAN**)

    ```console
    cd VULCAN/fastchem_vulcan
    ```

    On MacOS you will need to edit `make.globaloptions` to reflect  a GNU-compatible `g++` executable, not the Apple one (see
     [Troubleshooting](./troubleshooting.md) if the next step results in an error.

    ```console
    make
    cd ../../
    ```

7. Setup stellar evolution model (**MORS**)

    ```console
    cd Mors 
    pip install -e .
    cd ../
    ```

8. Setup numerical computing library (**PETSc**)

    - Configure step

    ```console
    cd petsc
    ./configure --with-debugging=0 --with-fc=0 --with-cxx=0 --download-sundials2 --download-mpich --download-f2cblaslapack --COPTFLAGS="-g -O3" --CXXOPTFLAGS="-g -O3"
    ```

    - Run the exact `make all` command provided at the end of the configure step
    - Run the exact `make check` command provided at the end of the `make all` step
    - Return to PROTEUS directory

    ```console
    cd ../
    ```

9. Setup environment variables

    This can be done using the PROTEUS environment file

    ```console
    source PROTEUS.env
    ```

    Additionally, you need to set up the environment variable `FWL_DATA` to indicate in which folder the input data need to be stored

    ```console
    export FWL_DATA=/your_local_path
    ```

    **IF** you want to be able to start PROTEUS immediately from a new shell every time, add `source PROTEUS.env`,     `export FWL_DATA=...` (and potentially `conda activate proteus`) to your shell rc file.

10. Setup interior evolution model (**SPIDER**)

    ```console
    cd SPIDER
    make clean
    make -j
    make test      # accept all default values when prompted
    cd ..
    ```

11. Done! 🚀

## Step-by-step (optional modules)

### Radiative-convective atmosphere model (**AGNI**)

Installation steps can be found at the [AGNI wiki](https://nichollsh.github.io/AGNI/dev/setup/).
They are
     also reproduced below.

1. Setup Julia

    ```console
    curl -fsSL https://install.julialang.org | sh
    ```

2. Clone the model

    ```console
    git clone git@github.com:nichollsh/AGNI.git 
    cd AGNI 
    ```

3. Setup SOCRATES

    ```console
    source get_socrates.sh
    ```

4. Build AGNI

    ```console
    julia -e 'using Pkg; Pkg.activate("."); Pkg.build()'
    ```

5. Go back to the PROTEUS directory

    ```console
    cd ../
    ```

Consult the AGNI wiki if you encouter issues.
