# Quick installation

The fastest way to get a working PROTEUS installation is the unified installer
script. It handles Julia, SOCRATES, AGNI, all Python submodules, environment
variables, and reference data downloads in a single command.

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

!!! note "macOS users"
    macOS Catalina (10.15) and later uses `zsh` as the default shell. Replace `.bashrc` with `.zshrc` throughout these instructions if you are using the default shell.

!!! tip "CLI alternative: `proteus install-all`"
    If PROTEUS is already importable in your environment, `proteus install-all`
    performs the same setup from the CLI: it installs PROTEUS and the required
    submodules (SOCRATES, AGNI), downloads reference data, checks for sufficient
    disk space, creates `FWL_DATA` if needed, and sets the environment variables. 

    - Pass `--export-env` to write the environment variables to your shell rc file. 
    - To refresh an existing installation later, use `proteus update-all` (see [Diagnose and update](doctor.md)). 
    
    Both commands operate on the PROTEUS source tree, which they locate from the
    installed package (editable installs) or, for a plain wheel install, from
    the current directory when it is a PROTEUS clone; with neither available
    they exit with an error.

---

## 1. System packages

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

## 2. Clone PROTEUS and create conda environment

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

## 3. Run the installer

```console
bash install.sh
```

The installer runs through the following phases automatically:

1. Pre-flight checks (OS, disk space, Python version, system dependencies)
2. Julia installation and version pinning (1.12)
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

## 4. Verify and run

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

    See [SPIDER installation](optionalmodules_installation#optional-setup-petsc) for details.
