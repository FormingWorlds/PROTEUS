# Installation

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

## Quick start: automated installer

The fastest way to get a working PROTEUS installation is the unified installer
script. It handles Julia, SOCRATES, AGNI, all Python submodules, environment
variables, and reference data downloads in a single command.

!!! note "CLI alternative: `proteus install-all`"
    If PROTEUS is already importable in your environment, `proteus install-all`
    performs the same setup from the CLI: it installs PROTEUS and the required
    submodules (SOCRATES, AGNI), downloads reference data, checks for sufficient
    disk space (5 GB minimum), creates `FWL_DATA` if needed, and sets the
    environment variables. Pass `--export-env` to write the environment
    variables to your shell rc file. To refresh an existing installation later,
    use `proteus update-all` (see [Diagnose and update](doctor.md)).

### 1. System packages

Install the required system libraries for your platform. See the
[Local machine guide](local_machine_guide.md) for detailed instructions.

=== "macOS (Homebrew)"

    ```console
    xcode-select --install
    brew install gcc netcdf netcdf-fortran wget open-mpi cmake
    ```

=== "Debian / Ubuntu"

    ```console
    sudo apt install gfortran libnetcdff-dev build-essential curl git cmake unzip
    ```

=== "Fedora / RHEL"

    ```console
    sudo dnf install gcc-gfortran netcdf-fortran-devel make curl git cmake unzip
    ```

**Compute clusters**: use the dedicated guides instead
([Kapteyn](kapteyn_cluster_guide.md),
[Habrok](habrok_cluster_guide.md),
[Snellius](snellius_cluster_guide.md)).

### 2. Clone PROTEUS and create conda environment

Conda (via miniforge or miniconda) is required. If you followed the
[Local machine guide](local_machine_guide.md) it is already installed.
If not, install
[miniforge](https://github.com/conda-forge/miniforge) (macOS) or
[miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
(Linux) before proceeding.

```console
git clone git@github.com:FormingWorlds/PROTEUS.git
cd PROTEUS
conda create -n proteus python=3.12
conda activate proteus
```

### 3. Run the installer

```console
bash install.sh
```

The installer runs through the following phases automatically:

1. Pre-flight checks (OS, disk space, Python version, system dependencies)
2. Julia installation and version pinning (1.11)
3. Environment variables (`FWL_DATA`, `PYTHON_JULIAPKG_EXE`)
4. SOCRATES compilation and `RAD_DIR` setup
5. AGNI and FastChem setup (Julia atmosphere model + equilibrium chemistry)
6. Python packages (editable installs of all submodules + PROTEUS itself)
7. Reference data downloads
8. Verification via `proteus doctor`

Each phase is idempotent: if the installer fails partway through, fix the
reported issue and re-run `bash install.sh`. It will skip already-completed
phases.

**Installer options:**

| Flag | Effect |
|---|---|
| `--all-data` | Download all reference data (~10-20 GB) instead of the essential set (~2 GB) |
| `--no-data` | Skip data downloads entirely (download later with `proteus get`) |
| `-i` / `--interactive` | Interactive mode (prompt for choices; default is non-interactive) |

### 4. Verify and run

After the installer finishes, source your shell configuration and run the
quick-start test:

```console
source ~/.zshrc   # or ~/.bashrc, depending on your shell
conda activate proteus
proteus start --offline -c input/dummy.toml
```

If this produces output in `output/dummy/`, your installation is working.
See the [Quick start tutorial](../Tutorials/quick_start_dummy.md) for
a guided walkthrough.

!!! note "SPIDER (optional)"
    The installer does not include SPIDER or PETSc. If you need SPIDER
    as an alternative interior energetics solver, install it separately
    after the main installation:

    ```console
    bash tools/get_petsc.sh
    bash tools/get_spider.sh
    ```

    See [SPIDER installation](#11-optional-setup-petsc) below for details.

---

## Manual installation

If the automated installer does not work on your system, or if you prefer to
control each step, follow the manual procedure below. These steps cover the
same ground as the installer script.

### 1. System pre-configuration

Install the required system packages for your platform before proceeding.

**Local machine** (laptop/desktop): follow the
[Local machine guide](local_machine_guide.md).

**Compute cluster**: use the dedicated guides:

* [Kapteyn cluster](kapteyn_cluster_guide.md)
* [Habrok cluster](habrok_cluster_guide.md)
* [Snellius cluster](snellius_cluster_guide.md)

### 2. Set up a Python environment

Python **3.12** is required. Install it via
[miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
or [miniforge](https://github.com/conda-forge/miniforge).

=== "Linux"

    ```console
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh
    rm ~/miniconda3/miniconda.sh
    ```

=== "macOS"

    ```console
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

### 3. Install Julia

Some PROTEUS modules (AGNI, LovePy) are written in Julia. Install via the
official installer:

```console
curl -fsSL https://install.julialang.org | sh
```

!!! warning "Do **not** use your package manager"
    Package managers often install the wrong Julia version. Use only the
    official installer. If you previously installed Julia another way,
    uninstall the old version first and remove old Julia entries from your
    `PATH`.

!!! warning "Pin Julia to version 1.11"
    Julia 1.12+ is **not yet supported** due to OpenSSL library
    incompatibilities with Python. After installing, pin to 1.11:

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

### 4. Create environment variables and clone PROTEUS

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

Clone the repository and create the conda environment:

```console
git clone git@github.com:FormingWorlds/PROTEUS.git
cd PROTEUS
conda create -n proteus python=3.12
conda activate proteus
```

### 5. Install SOCRATES (radiative transfer)

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

### 6. Install AGNI and FastChem

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

### 7. Install Python submodules

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

### 8. Install PROTEUS

```console
python -m pip install -e ".[develop]"
```

### 9. Enable pre-commit hooks

```console
pre-commit install -f
```

### 10. Download reference data

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

### 11. Done!

Verify the installation:

```console
proteus doctor
proteus start --offline -c input/dummy.toml
```

---

## Optional modules

### SPIDER and PETSc {#11-optional-setup-petsc}

SPIDER is a C-based interior thermal evolution solver. It requires PETSc
(a numerical computing library) and a C compiler. Most configurations use
Aragog instead, which is written in Python/JAX and needs no additional
compiled dependencies.

!!! warning
    PETSc requires Python <= 3.12. Make sure your conda environment uses
    Python 3.12.

```console
bash tools/get_petsc.sh
bash tools/get_spider.sh
```

=== "Linux"

    The PETSc script downloads a pre-compiled version from OSF and configures
    `PETSC_DIR` and `PETSC_ARCH` automatically.

    !!! note "Fedora / RHEL"
        If you encounter errors moving libraries, see
        [Troubleshooting: PETSc on Fedora/RHEL](troubleshooting.md#cannot-compile-petsc-error-moving-libraries-fedorarhel).

=== "macOS"

    The script detects Apple Silicon vs Intel and uses Homebrew's MPI. If you
    encounter issues, see
    [Troubleshooting: PETSc on Apple Silicon](troubleshooting.md#petsc-compilation-fails-on-apple-silicon).

### Multi-phase tidal heating (LovePy)

LovePy computes tidal dissipation for multi-phase planetary interiors. It is
written in Julia.

```console
bash tools/get_lovepy.sh
```

### Synthetic observations (PLATON)

[PLATON](https://platon.readthedocs.io/en/latest/intro.html) generates
synthetic transmission and secondary eclipse spectra.

```console
bash tools/get_platon.sh
```

!!! note
    PLATON downloads ~10 GB of opacity data on first use.

### Alternative outgassing (atmodeller)

[atmodeller](https://github.com/djbower/atmodeller) is an optional
alternative to CALLIOPE, selected with `outgas.module = "atmodeller"`. It is
not installed with PROTEUS by default; a standard run uses CALLIOPE.

```console
pip install "fwl-proteus[atmodeller]"
```

!!! warning "License"
    atmodeller is distributed under the GPL-3.0 license; review its terms
    before installing.

### Atmospheric chemistry (VULCAN)

VULCAN is an optional atmospheric-photochemistry backend, selected with
`atmos_chem.module = "vulcan"`. It is not required for a standard PROTEUS
run. Install it from PyPI:

```console
pip install "fwl-proteus[vulcan]"
```

For local development, install as an editable checkout instead:

```console
bash tools/get_vulcan.sh
```

!!! warning "License"
    VULCAN is distributed under the GPL-3.0 license; review its terms
    before installing.
