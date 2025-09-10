# Installation

These instructions will guide you through the typical installation
process. The setup is written for MacOS and Linux. Depending on your
system settings and installed libraries your procedure may differ. If
one or more of the steps below do not work for you we encourage you to
first check the [troubleshooting](troubleshooting.html) page. If
that does not help you further, please [contact the developers](contact.html).

## System pre-configuration

Setting up PROTEUS and its submodules requires extra steps to be performed before following the rest of this guide.
Follow the instructions below depending on your system configuration.

For installing PROTEUS on a local machine (e.g. your laptop), follow the appropriate section in the [Local machine guide](local_machine_guide.html).

If you are using a specific server, use the following guides:

* Kapteyn cluster [guide](kapteyn_cluster_guide.html)
* Habrok cluster [guide](habrok_cluster_guide.html)
* Snellius cluster [guide](snellius_cluster_guide.html).


## Setup a Python environment

We recommend that you use Python version 3.12 for running PROTEUS.
Python is most easily obtained and managed using either [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux)
```console
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh 
# choose an install folder where you have plenty of disk space!
rm ~/miniconda3/miniconda.sh
```
or [miniforge](https://github.com/conda-forge/miniforge).
```console
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Alternatively, you can download Python from the official website and use [pyenv](https://github.com/pyenv/pyenv).

## Install Julia

Some of the PROTEUS modules are written in Julia, which is broadly similar to Python.
You should only obtain Julia using the official installer, **not** via your computer's package manager.

```console
curl -fsSL https://install.julialang.org | sh
```

## User install (default)
### Requirements

1. conda either through miniconda or miniforge (see above). However, currently there seems to be a 
conflict between Julia and Conda versions of HDF5 and/or NetCDF libaries when using miniforge.
2. git. If you don't have git, you can install it through conda: `conda install git`.
3. Julia installation (see above)
4. 20 GB of disk space; Conda (through miniconda) can take 9 GB, Julia 2 GB and a few GB for PROTEUS.
5. 30 minutes of your time (mostly waiting)

### Steps

1. `git clone https://github.com/FormingWorlds/PROTEUS.git`
2. `cd PROTEUS`
3. `conda env create -f environment.yml`
4. `conda activate proteus`
5. `pip install -e .`
6. `proteus install-all --export-env`

### Before running PROTEUS
If you want to start running PROTEUS right away - i.e. from the same shell that you used to install PROTEUS -
you can set your environment variables using e.g. `source ~/.bashrc` (for Bash). This deactivates your current
conda environment, so you need to enter `conda activate proteus` as well. If you chose `proteus install-all` 
(i.e. without `--export-env`), you will need to set your PATH, FWL_DATA and RAD_DIR enviroment variables 
manually.

When you log into the machine that where you installed PROTEUS through `proteus install-all --export-env`, 
you will have the environment variables FWL_DATA, RAD_DIR set appropriately as your ~/.shellrc file 
(e.g. ~/.bashrc) has been updated during the install process. However, you still need to enter 
`conda activate proteus`.

### Updating PROTEUS

1. `conda activate proteus` (if not already activated)
2. `proteus update-all` 

## Developer install

1. Create and set environment variables

    The environment variable `FWL_DATA` points to the folder where input data are stored.
    This variable must always be set, so it is best to add this line to your shell rc file .

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
    conda activate proteus
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

    Reload your shell rc file to make the changes effective. Depending on your shell,
    you can do this by running:

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

10. Enable pre-commit

    ```console
    pre-commit install -f
    ```

11. Done! 🚀
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
about 10 GB of data from the internet.

### Chemical kinetics atmosphere model (**VULCAN**)

VULCAN is written in Python, however, it is not available in a cohesive package format.
Obtain VULCAN using the script below.

```console
./tools/get_vulcan.sh
```
