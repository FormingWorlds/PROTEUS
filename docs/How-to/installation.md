# Quick installation

The fastest way to get a working PROTEUS installation is the unified installer
script. It handles Julia, SOCRATES, AGNI, all Python submodules, environment
variables, and reference data downloads in a single command.

!!! info "Prerequisites"
    - Tested in CI: Linux x86_64 (Ubuntu) and macOS on Apple Silicon. Intel macOS and
      Linux ARM64 are not tested; the Linux Miniconda commands below download the
      x86_64 installer.
    - ~20 GB disk space (conda, Julia, reference data, submodules)
    - Standard command-line tools: `curl`, `wget`, `make`
    - Git with SSH key configured ([GitHub SSH setup](https://docs.github.com/en/authentication/connecting-to-github-with-ssh))
    - Internet connection for initial setup and data downloads
    - Allow ~60 minutes for a full installation including all submodules

On Windows, install PROTEUS under
[WSL2](local_machine_guide.md#microsoft-windows). Depending on your system
configuration, some steps may differ. If you run into problems, check the
[Troubleshooting](troubleshooting.md) page or
[contact the developers](../Community/contact.md).

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
    sudo apt install gfortran libnetcdff-dev build-essential curl wget git cmake unzip
    ```

=== "Fedora / RHEL"

    ```console
    sudo dnf install gcc-gfortran gcc-c++ netcdf-fortran-devel make curl wget git cmake unzip
    ```

**Compute clusters**: use the dedicated guides instead
([Kapteyn](kapteyn_cluster_guide.md),
[Habrok](habrok_cluster_guide.md),
[Snellius](snellius_cluster_guide.md),
[Cambridge IoA](ioa_cluster_guide.md)).

## 2. Clone PROTEUS and set up Python environment

!!! tip "Using pixi instead of conda"
    PROTEUS can also be installed into a [pixi](https://pixi.sh) environment,
    which replaces the conda steps below. See [Install with pixi](pixi.md)
    (experimental).

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

Then clone PROTEUS:

```console
git clone git@github.com:FormingWorlds/PROTEUS.git
cd PROTEUS
```

## 3. Run the installer

```console
bash install.sh
```

!!! note "Conda environment"
    The installer makes a conda environment named `proteus` with Python 3.12, or uses
    the one of that name, when no environment or only `base` is active. To use an
    environment of your own, activate it before `bash install.sh`, or pass
    `--conda-env <name>`.

!!! warning "Julia from a system package manager"
    If your `julia` comes from your system package manager (not the official installer) and is not version 1.13, the installer stops, because it adds Julia 1.13 only through juliaup. Install juliaup with the official installer, or uninstall that Julia and remove its entries from your `PATH`, _before running the installer_.


The installer runs through the following phases automatically:

1. Pre-flight checks (OS, disk space, Python version, system dependencies)
2. Julia check, and Julia 1.13 through juliaup when your default Julia is another version or you have none
3. Environment variables (`FWL_DATA`, `PYTHON_JULIAPKG_EXE`, and in phases 4 and 5 `RAD_DIR` and `FC_DIR`): the installer sets them in the active conda environment and writes them to `.proteus-env.sh` in the PROTEUS folder; it edits no shell start file
4. SOCRATES compilation and `RAD_DIR` setup
5. AGNI and FastChem setup (Julia atmosphere model + equilibrium chemistry)
6. Python packages (editable installs of all submodules, the SUNDIALS CVODE solver, and PROTEUS itself)
7. Reference data downloads (by default the essential set: the default-config and tutorial k-tables, stellar spectra and tracks, the PHOENIX grid of the default star, and the interior tables)
8. Verification via `proteus doctor`

The installer prints one line per build step and writes the full build output to `install_<date>_<time>.log` (for example `install_20261011_142530.log`) in the PROTEUS folder; when a step fails, it also prints the last 40 lines of that step's output.

PROTEUS runs on Julia 1.13. When your default Julia is another version, the installer adds Julia 1.13 through juliaup beside it and does not change your default; when you have no Julia, it installs juliaup with 1.13 as the default. PROTEUS and the setup scripts in `tools/` find Julia 1.13 through `PYTHON_JULIAPKG_EXE`, so a bare `julia` in your terminal is still your own default.

Each Python module clone that the installer makes (MORS, JANUS, CALLIOPE, ZEPHYRUS, Aragog and Zalmoxis) is at the release tag that matches the module's minimum version in `pyproject.toml`, except that a new Aragog or Zalmoxis clone takes the branch of the same name when PROTEUS itself is on a feature branch that the module also has. When you run the installer again, it never replaces a module clone: it moves a clone that it made forward to the new tag in place and keeps your untracked files and stash entries, a clone on a branch or on a newer commit than the tag stays as it is and the installer names it at the end, and a detached clone with modified tracked files or commits on no remote stops the installer. To replace a clone with a new one, run the module's setup script with `--force`, for example `bash tools/get_python_module.sh MORS --force` or `bash tools/get_aragog.sh --force`.

!!! note "Developing a module"
    To develop a module, switch its clone to a branch and install it again, for example:

    ```console
    git -C MORS switch main
    pip install -e MORS/.
    ```

Each phase is idempotent: if the installer fails partway through, fix the
reported issue and re-run `bash install.sh`. It will skip already-completed
phases.

!!! note "CVODE is required for Aragog"
    Aragog integrates with SUNDIALS CVODE by default (`solver_method = "cvode"`).
    The installer builds it (`scikits-odes-sundials` on the SUNDIALS C library,
    through `tools/get_cvode.sh`, which needs an active conda environment) and
    warns when that fails, since only Aragog on the default solver needs it.
    A run with Aragog on the default `solver_method = "cvode"` stops at setup,
    with the same install command in the message, when CVODE cannot be imported.
    Choosing `solver_method = "radau"` or `"bdf"` selects scipy on purpose and
    does not need CVODE. See [Aragog stops at setup](troubleshooting.md#aragog-stops-at-setup-cvode-cannot-be-imported).

**Installer options:**

| Flag | Effect |
|---|---|
| `--all-data` | Download all reference data (~10-20 GB) instead of the essential set (~3 GB) |
| `--no-data` | Skip data downloads entirely (download later with `proteus get`) |
| `-i` / `--interactive` | Interactive mode (prompt for choices; default is non-interactive) |
| `--verbose` | Show the full build output on the terminal as well as in the log file |
| `--conda-env <name>` | The conda environment to use, or to make, when no environment or only `base` is active (default: `proteus`) |

!!! tip "CLI alternative: `proteus install-all`"
    If PROTEUS is already importable in your environment, `proteus install-all`
    performs the same setup from the CLI: it installs the SUNDIALS CVODE solver and the required
    submodules (SOCRATES, AGNI), downloads reference data, checks for sufficient
    disk space, creates `FWL_DATA` if needed, and sets the environment variables. 

    - Pass `--export-env` to set the environment variables in the active conda environment and write them to `.proteus-env.sh`, as the installer does.
    - To refresh an existing installation later, use `proteus update-all` (see [Diagnose and update](doctor.md)). It does not change the Python module clones: to move one to its release tag after a version change, run the two `git` commands that `bash install.sh` prints for a clone that is not at its tag, or the module's setup script with `--force`.
    
    Both commands operate on the PROTEUS source tree, which they locate from the
    installed package (editable installs) or, for a plain wheel install, from
    the current directory when it is a PROTEUS clone; with neither available
    they exit with an error.

## 4. Verify and run

After the installer finishes, activate the environment and run the
quick-start test:

```console
conda activate proteus
proteus start --offline -c input/dummy.toml
```

Conda loads the environment variables of an environment when it activates it,
and activating the environment that is already active does not load them. If an
environment of your own was active when you ran the installer, run
`conda deactivate` first; the installer prints the exact line at its end.

If this produces output in `output/run_<timestamp>_xxxx/`, your installation is working.
See the [Quick start tutorial](../Tutorials/quick_start_dummy.md) for a guided walkthrough.

!!! note "SPIDER (optional)"
    The installer does not include SPIDER or PETSc. If you need SPIDER
    as an alternative interior energetics solver, install it separately
    after the main installation:

    ```console
    bash tools/get_petsc.sh
    bash tools/get_spider.sh
    ```

    See [SPIDER installation](optionalmodules_installation.md#optional-setup-petsc) for details.

!!! note "Shell start files of an earlier install"
    An earlier install wrote `export FWL_DATA=...`, `RAD_DIR`, `PYTHON_JULIAPKG_EXE`
    and `FC_DIR` lines into a shell start file such as `~/.zshrc` or `~/.bashrc`.
    The installer leaves them as they are and names each one at the end of its run.
    Conda sets the values of the environment when it activates it, after the shell
    start file was read, so inside the activated environment they win, and conda
    warns at each activation that it overwrites the variables. You can delete the
    old lines by hand; while they stay, a shell without the activated environment
    still gets the old values.

## Uninstall

Removing the conda environment (`conda deactivate`, then `conda env remove -n proteus`)
removes the environment variables with it; delete the PROTEUS folder and the
`FWL_DATA` folder by hand.
