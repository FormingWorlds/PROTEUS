# Manual installation

If the automated installer does not work on your system, or if you prefer to
control each step, follow the manual procedure below. These steps cover the
same ground as the installer script.

!!! info "Prerequisites"
    - macOS (Intel or Apple Silicon) or Linux
    - ~20 GB disk space (conda, Julia, reference data, submodules)
    - Standard command-line tools: `curl`, `wget`
    - Git with SSH key configured ([GitHub SSH setup](https://docs.github.com/en/authentication/connecting-to-github-with-ssh))
    - Internet connection for initial setup and data downloads
    - Allow ~60 minutes for a full installation including all submodules

PROTEUS runs on macOS and Linux. Windows users should install via
[WSL2](local_machine_guide.md#microsoft-windows). Depending on your system
configuration, some steps may differ. If you run into problems, check the
[Troubleshooting](troubleshooting.md) page or
[contact the developers](../Community/contact.md).

!!! tip "macOS users"
    macOS Catalina (10.15) and later uses `zsh` as the default shell. Replace `.bashrc` with `.zshrc` throughout these instructions if you are using the default shell.

---

## 1. System pre-configuration

Install the required system packages for your platform before proceeding.

**Local machine** (laptop/desktop): follow the
[Local machine guide](local_machine_guide.md).

**Compute cluster**: use the dedicated guides:

* [Kapteyn cluster](kapteyn_cluster_guide.md)
* [Habrok cluster](habrok_cluster_guide.md)
* [Snellius cluster](snellius_cluster_guide.md)
* [Cambridge IoA cluster](ioa_cluster_guide.md)

## 2. Set up a Python environment

Python **3.12** is required, and is installed via
[miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
or [miniforge](https://github.com/conda-forge/miniforge). If you do not have [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
or [miniforge](https://github.com/conda-forge/miniforge) installed yet:

=== "macOS (Homebrew)"

    ```console
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

=== "Debian / Ubuntu"

    ```console
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh
    rm ~/miniconda3/miniconda.sh
    ```

=== "Fedora / RHEL"

    ```console
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh
    rm ~/miniconda3/miniconda.sh
    ```

!!! note "Install miniconda/miniforge in your personal directory"
    Installing miniconda/miniforge in your personal directory gives you full control over your environment and is recommended even on clusters (see the cluster guides: [Kapteyn](kapteyn_cluster_guide.md), [Habrok](habrok_cluster_guide.md), [Snellius](snellius_cluster_guide.md), [Cambridge IoA](ioa_cluster_guide.md)). 

Then create a conda environment with Python 3.12:

```console 
conda create -n proteus python=3.12
conda activate proteus
```

## 3. Install Julia

Some PROTEUS modules (AGNI, LovePy) are written in Julia. Install via the
official installer:

```console
curl -fsSL https://install.julialang.org | sh
```
After installing, pin to Julia 1.12, which is the version exercised by CI:

```console
juliaup add 1.12
juliaup default 1.12
```

!!! warning "Do **not** use your package manager"
    Package managers often install the wrong Julia version. Use only the
    official installer. If you previously installed Julia another way,
    uninstall the old version first and remove old Julia entries from your
    `PATH`.

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

## 4. Create environment variables and clone PROTEUS

The `FWL_DATA` environment variable points to the folder where PROTEUS stores
reference data. This variable must always be set.

=== "bash"

    ```console
    mkdir -p /your/local/path/FWL_DATA
    echo "export FWL_DATA=/your/local/path/FWL_DATA/" >> "$HOME/.bashrc"
    source "$HOME/.bashrc"
    ```

=== "zsh"

    ```console
    mkdir -p /your/local/path/FWL_DATA
    echo "export FWL_DATA=/your/local/path/FWL_DATA/" >> "$HOME/.zshrc"
    source "$HOME/.zshrc"
    ```

Clone the repository:

```console
git clone git@github.com:FormingWorlds/PROTEUS.git
cd PROTEUS
```

## 5. Install SOCRATES (radiative transfer)

[SOCRATES](https://github.com/FormingWorlds/SOCRATES) is a Fortran spectral radiative transfer code used by [AGNI](https://www.h-nicholls.space/AGNI/) and [JANUS](https://proteus-framework.org/JANUS/).

!!! note "Fortran compiler and NetCDF"
    SOCRATES requires `gfortran` (version 9+) and the NetCDF Fortran
    development libraries. Verify they are available:

    ```console
    which gfortran
    which nf-config
    nf-config --version
    ```

```console
bash tools/get_socrates.sh
```

Set `RAD_DIR` to point to the SOCRATES installation:

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

## 6. Install AGNI and FastChem

[AGNI](https://www.h-nicholls.space/AGNI/) solves the atmospheric energy balance using Julia and the [SOCRATES](https://github.com/FormingWorlds/SOCRATES) spectral library. [FastChem](https://github.com/exoclime/FastChem) provides equilibrium chemistry for AGNI. Installation steps are also documented at the [AGNI wiki](https://www.h-nicholls.space/AGNI/dev/howto/getting_started/).

!!! note
    This step requires `make`, `unzip`, and `cmake`. Check with:

    ```console
    which make && which unzip && which cmake
    ```

```console
git clone git@github.com:nichollsh/AGNI.git
cd AGNI
bash src/get_agni.sh 0
bash src/get_fastchem.sh
cd ../
```

Use `get_agni.sh` to keep AGNI and its data files up to date. AGNI must be
available at `./AGNI/` inside your PROTEUS folder (either a symbolic link or
the true location).

Set the FastChem environment variable:

=== "bash"

    ```console
    echo "export FC_DIR=$PWD/AGNI/fastchem/" >> "$HOME/.bashrc"
    source "$HOME/.bashrc"
    ```

=== "zsh"

    ```console
    echo "export FC_DIR=$PWD/AGNI/fastchem/" >> "$HOME/.zshrc"
    source "$HOME/.zshrc"
    ```

## 7. Install Python submodules

Clone and install each submodule in editable mode:

```console
# MORS - stellar evolution (https://proteus-framework.org/MORS/)
git clone git@github.com:FormingWorlds/MORS
python -m pip install -e MORS/.

# JANUS - 1D convective atmosphere (https://proteus-framework.org/JANUS/)
git clone git@github.com:FormingWorlds/JANUS
python -m pip install -e JANUS/.

# CALLIOPE - volatile outgassing (https://proteus-framework.org/CALLIOPE/)
git clone git@github.com:FormingWorlds/CALLIOPE
python -m pip install -e CALLIOPE/.

# ZEPHYRUS - atmospheric escape (https://github.com/FormingWorlds/ZEPHYRUS)
git clone git@github.com:FormingWorlds/ZEPHYRUS
python -m pip install -e ZEPHYRUS/.

# Aragog - interior thermal evolution (https://proteus-framework.org/aragog/)
bash tools/get_aragog.sh

# Zalmoxis - planetary interior structure (https://proteus-framework.org/Zalmoxis/)
bash tools/get_zalmoxis.sh
```

!!! info "Editable checkouts override PyPI versions"
    `fwl-aragog` and `fwl-zalmoxis` are listed as PyPI dependencies in
    `pyproject.toml` as a fallback for users who install PROTEUS without
    cloning submodules. The editable sibling checkouts take precedence on
    `sys.path`. Run `proteus doctor` to confirm which versions are active.

!!! warning "Local changes block checkout refreshes"
    The `tools/get_*.sh` scripts refuse to replace a checkout that has
    uncommitted changes to tracked files or commits not pushed to a
    remote. Commit and push your work first, or pass `--force` to the
    script (e.g. `bash tools/get_aragog.sh --force`) to discard the
    checkout deliberately.

## 8. Install PROTEUS

```console
python -m pip install -e ".[develop]"
```

## 9. Enable pre-commit hooks

```console
pre-commit install -f
```

## 10. Download reference data

Download the essential reference data files (the default-config and
tutorial k-tables, stellar spectra, and the interior tables, including
the structure-solver EOS):

```console
proteus get spectral -n Honeyside -b 48
proteus get spectral -n Dayspring -b 48
proteus get stellar
proteus get interiordata
```

Running `proteus get spectral` without arguments downloads every
spectral file group (~10 GB).

For additional datasets (MUSCLES spectra, PHOENIX models), use:

```console
proteus get muscles --all
proteus get phoenix --feh 0.0 --alpha 0.0
proteus get reference
```

## 11. Done!

Verify the installation:

```console
proteus doctor
proteus start --offline -c input/dummy.toml
```
