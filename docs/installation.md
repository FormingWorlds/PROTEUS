# Installation

These instructions will guide you through the typical installation
process. The setup is written for macOS and Linux. Depending on your
system settings and installed libraries your procedure may differ. If
one or more of the steps below do not work for you we encourage you to
first check the [Troubleshooting](troubleshooting.html) page. If
that does not help you further, please [contact the developers](contact.html).

!!! tip "macOS users"
    macOS Catalina (10.15) and later uses `zsh` as the default shell. Replace `.bashrc` with `.zshrc` throughout these instructions if you are using the default shell.

---

## 1. System pre-configuration

Setting up PROTEUS and its submodules requires extra steps to be performed before following the rest of this guide. Follow the instructions below depending on your system configuration.

**Local machine** (laptop/desktop): follow the appropriate section in the [Local machine guide](local_machine_guide.html).

**Compute cluster**: use the dedicated guides:

* [Kapteyn cluster](kapteyn_cluster_guide.html)
* [Habrok cluster](habrok_cluster_guide.html)
* [Snellius cluster](snellius_cluster_guide.html)

---

## 2. Setup a Python environment

We recommend Python version **3.12** for running PROTEUS. Python is most easily obtained and managed using either miniconda or miniforge.

=== "Linux"

    Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux):

    ```console
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh
    rm ~/miniconda3/miniconda.sh
    ```

    Choose an install folder where you have plenty of disk space.

=== "macOS"

    Install [miniforge](https://github.com/conda-forge/miniforge) (recommended for Apple Silicon):

    ```console
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

---

## 3. Install Julia

Some PROTEUS modules are written in Julia. You should only obtain Julia using the official installer, **not** via your computer's package manager.

```console
curl -fsSL https://install.julialang.org | sh
```

!!! warning "Pin Julia to version 1.11"
    Julia 1.12+ is **not yet supported** due to OpenSSL library incompatibilities with Python. After installing Julia, pin it to version 1.11:

    ```console
    juliaup add 1.11
    juliaup default 1.11
    ```

Set the Julia environment variable:

=== "bash"

    ```console
    echo "export PYTHON_JULIAPKG_EXE=$(which julia)" >> ~/.bashrc
    source ~/.bashrc
    ```

=== "zsh"

    ```console
    echo "export PYTHON_JULIAPKG_EXE=$(which julia)" >> ~/.zshrc
    source ~/.zshrc
    ```

---

## Step 1: Create and set environment variables

The environment variable `FWL_DATA` points to the folder where input data are stored. This variable must always be set, so add it to your shell config file.

=== "bash"

    ```console
    mkdir /your/local/path/FWL_DATA
    echo "export FWL_DATA=/your/local/path/FWL_DATA/" >> "$HOME/.bashrc"
    source "$HOME/.bashrc"
    ```

=== "zsh"

    ```console
    mkdir /your/local/path/FWL_DATA
    echo "export FWL_DATA=/your/local/path/FWL_DATA/" >> "$HOME/.zshrc"
    source "$HOME/.zshrc"
    ```

## Step 2: Download PROTEUS

```console
git clone git@github.com:FormingWorlds/PROTEUS.git
cd PROTEUS
```

## Step 3: Create a virtual environment

```console
conda create -n proteus python=3.12
conda activate proteus
```

## Step 4: Install SOCRATES (radiative transfer)

```console
./tools/get_socrates.sh
```

The environment variable `RAD_DIR` must always point to the SOCRATES installation path. Add it to your shell config file:

=== "bash"

    ```console
    echo "export RAD_DIR=$PWD/socrates/" >> "$HOME/.bashrc"
    source "$HOME/.bashrc"
    ```

=== "zsh"

    ```console
    echo "export RAD_DIR=$PWD/socrates/" >> "$HOME/.zshrc"
    source "$HOME/.zshrc"
    ```

## Step 5: Install AGNI (radiative-convective atmosphere model)

Installation steps can be found at the [AGNI wiki](https://h-nicholls.space/AGNI/dev/setup/). They are also reproduced below.

```console
git clone git@github.com:nichollsh/AGNI.git
cd AGNI
bash src/get_agni.sh 0
cd ../
```

Use this `get_agni.sh` script to keep AGNI and its data files up to date. AGNI must be available at `./AGNI/` inside your PROTEUS folder (either a symbolic link or the true location).

The argument provided to the script (integer from 0 to 20) indicates which tests AGNI should run. A value of `0` means the tests are skipped.

## Step 6: Install submodules as editable

Clone and install each submodule in editable mode.

**MORS** (stellar evolution):

```console
git clone git@github.com:FormingWorlds/MORS
python -m pip install -e MORS/.
```

**JANUS** (1D convective atmosphere):

```console
git clone git@github.com:FormingWorlds/JANUS
python -m pip install -e JANUS/.
```

**CALLIOPE** (volatile in-/outgassing):

```console
git clone git@github.com:FormingWorlds/CALLIOPE
python -m pip install -e CALLIOPE/.
```

**ARAGOG** (interior thermal evolution):

```console
git clone git@github.com:FormingWorlds/aragog.git
python -m pip install -e aragog/.
```

**ZEPHYRUS** (atmospheric escape):

```console
git clone git@github.com:FormingWorlds/ZEPHYRUS
python -m pip install -e ZEPHYRUS/.
```

**VULCAN** (chemical kinetics atmosphere model):

VULCAN is not available as a standard Python package, so it is installed via a dedicated script:

```console
./tools/get_vulcan.sh
```

## Step 7: Setup PETSc (numerical computing library)

!!! warning
    PETSc requires Python <= 3.12. Make sure your active environment uses a compatible version.

=== "Linux"

    ```console
    ./tools/get_petsc.sh
    ```

    !!! note "Fedora/RHEL users"
        If you encounter errors moving libraries, see [Troubleshooting: PETSc on Fedora/RHEL](troubleshooting.html#cannot-compile-petsc-error-moving-libraries-fedorahel).

=== "macOS"

    First try the standard script:

    ```console
    ./tools/get_petsc.sh
    ```

    If compilation fails on Apple Silicon (M1/M2/M3/Ultra), run the fix script:

    ```console
    ./tools/fix_petsc_compile.sh
    ```

    See [Troubleshooting: PETSc on Apple Silicon](troubleshooting.html#petsc-compilation-fails-on-apple-silicon) for manual steps if needed.

## Step 8: Setup SPIDER (interior evolution model)

```console
./tools/get_spider.sh
```

## Step 9: Install PROTEUS framework

```console
python -m pip install -e .
```

## Step 10: Enable pre-commit hooks

```console
pre-commit install -f
```

## Step 11: Done!

Any remaining dependencies will be downloaded when the model is first run.

---

## Optional modules

### Multi-phase tidal heating model (LovePy)

LovePy is written in Julia. You can use the same environment as AGNI if you wish, but make sure to follow the installation steps below.

```console
./tools/get_lovepy.sh
```

### Synthetic observations calculator (PLATON)

[PLATON](https://platon.readthedocs.io/en/latest/intro.html) is a forward modelling and retrieval tool for exoplanet atmospheres. In PROTEUS, this tool is used to generate synthetic transmission and secondary eclipse observations.

```console
./tools/get_platon.sh
```

!!! note
    This script will take some time to run; PLATON will need to download about 10 GB of data from the internet.

---

*For the legacy user install method (deprecated), see the [User install](user_install.html) page.*
