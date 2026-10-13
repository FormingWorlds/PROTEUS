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
    the one of that name, when no environment or only `base` is active. When an
    environment named `proteus` already exists and none is active, the installer
    installs into it; pass `--conda-env <name>` to use another one. Run in a terminal,
    with its output not redirected, it first asks whether to install into that existing
    environment: `y` or `yes` goes on, and any other answer stops it before it changes
    anything. It does not make
    an environment over an existing folder of that name, such as the folder of an
    environment `PROTEUS` on a file system that ignores case: it stops and names that
    folder. To use an environment of your own, activate it before `bash install.sh`.
    With Miniconda, conda can refuse to make the environment until you accept the terms
    of service of its default channels: run the `conda tos accept` commands that conda
    prints, then run `bash install.sh` again.

!!! warning "Julia from a system package manager"
    If your `julia` comes from your system package manager (not the official installer), is not version 1.13, and no juliaup is on your `PATH`, the installer stops, because it adds Julia 1.13 only through juliaup; a juliaup in `~/.juliaup/bin` that is not on your `PATH` counts only when no `julia` is on it. Install juliaup with the official installer and put it on your `PATH`, or uninstall that Julia and remove its entries from your `PATH`, _before running the installer_. A `PYTHON_JULIAPKG_EXE` that names a Julia 1.13 is kept; when it names a launcher, such as the `julia` of juliaup, the installer keeps the binary that the launcher runs.


The installer runs through the following phases automatically:

1. Pre-flight checks (OS, disk space, Python version, system dependencies)
2. Julia check, and Julia 1.13 through juliaup when your default Julia is another version or you have none
3. Environment variables (`FWL_DATA`, `PYTHON_JULIAPKG_EXE`, and in phases 4 and 5 `RAD_DIR` and, for a built FastChem, `FC_DIR`): the installer writes each to `.proteus-env.sh` in the PROTEUS folder as it sets it, and sets them all in the active conda environment in one step after phase 7, before the data downloads of phase 8; a value that holds a single quote stays in `.proteus-env.sh` only. It edits no shell start file
4. SOCRATES compilation and `RAD_DIR` setup
5. AGNI and FastChem setup (Julia atmosphere model + equilibrium chemistry); an AGNI folder that is there already stays, and when its `Manifest.toml` names another minor Julia version (1.12 against 1.13, for example) than the one PROTEUS uses, the installer warns and names `proteus update-all`, which instantiates AGNI again
6. Python packages (editable installs of all submodules, the SUNDIALS CVODE solver, and PROTEUS itself)
7. Further modules (Morrigan, VULCAN, atmodeller and Obliqua): a failed module prints a warning with the command to repeat and the installer goes on
8. Reference data downloads (by default the essential set: the default-config and tutorial k-tables, stellar spectra and tracks, the PHOENIX grid of the default star, and the interior tables)
9. Verification via `proteus doctor`

The installer adds atmodeller and VULCAN, which are under the GPL version 3, Morrigan (Apache 2.0) and Obliqua (MIT) as separate clones beside PROTEUS (Apache 2.0).

The installer prints one line per build step and writes the full build output to `install_<date>_<time>.log` (for example `install_20261011_142530.log`) in the PROTEUS folder; when a step fails, it also prints the last 40 lines of that step's output.

PROTEUS runs on Julia 1.13. When your default Julia is another version, the installer adds Julia 1.13 through juliaup beside it and does not change your default. When you have no Julia, the installer installs juliaup with Julia 1.13 and does not put it on your `PATH`: PROTEUS finds it through `PYTHON_JULIAPKG_EXE`, and to use `julia` in your terminal, add `~/.juliaup/bin` to your `PATH`. When neither `julia` nor juliaup is on your `PATH`, the installer takes juliaup from `~/.juliaup/bin`; when there is none either but a juliaup configuration exists (`juliaup/juliaup.json` under `JULIAUP_DEPOT_PATH`, by default `~/.julia`), the installer stops and asks you to put your juliaup on your `PATH`, so that a second juliaup does not change your default. PROTEUS and the setup scripts in `tools/` find Julia 1.13 through `PYTHON_JULIAPKG_EXE`, so a bare `julia` in your terminal is still your own default.

A new Python module clone (MORS, JANUS, CALLIOPE, ZEPHYRUS, Aragog, Zalmoxis, Morrigan and VULCAN) is at the release tag of the module's minimum version in `pyproject.toml`; a new atmodeller or Obliqua clone is at the commit that `pyproject.toml` pins under `[tool.proteus.modules]`; a new Aragog or Zalmoxis clone takes the branch of the same name instead when PROTEUS is on a feature branch that the module also has. Running the installer again does not update a module clone that is already there: it installs the clone as it is and names each one that is not at its tag or pinned commit, with the commands that move it there and reinstall it, and the command that replaces it with a new clone, for example `bash tools/get_python_module.sh MORS --force`. If a clone of MORS, JANUS, CALLIOPE, ZEPHYRUS, Aragog or Zalmoxis is older than the minimum version, the PROTEUS install that follows replaces its editable install with the release from PyPI: PROTEUS then imports that release, not the clone, and `proteus doctor` shows the version without the `[editable @ ...]` mark.

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
| `-i` / `--interactive` | Interactive mode (prompt for choices; default is non-interactive, except that an existing environment `proteus`, with no environment active, brings 1 question in a terminal, which `--conda-env proteus` answers in advance) |
| `--verbose` | Show the full build output on the terminal as well as in the log file |
| `--conda-env <name>` | The conda environment to use, or to make, when no environment or only `base` is active (default: `proteus`) |

!!! tip "CLI alternative: `proteus install-all`"
    If PROTEUS is already importable in your environment, `proteus install-all`
    performs the same setup from the CLI: it installs the SUNDIALS CVODE solver and the required
    submodules (SOCRATES, and AGNI at the commit PROTEUS pins, through `tools/get_agni.sh` as the installer does), runs the setup scripts of Morrigan, VULCAN, atmodeller and Obliqua (a script that fails gives a warning and the command goes on), downloads reference data, including the PHOENIX grid of the default star (about 1 GB), checks for sufficient
    disk space, creates `FWL_DATA` if needed, and sets the environment variables. It prints the
    version of the Julia that PROTEUS uses and warns when it is not 1.13, as `proteus update-all` does.

    - Pass `--export-env` to set the environment variables in the active conda environment and write them to `.proteus-env.sh`, as the installer does. It sets nothing in `base`, keeps a value that holds a single quote in `.proteus-env.sh` only, records `RAD_DIR` only for a folder that holds a built SOCRATES (`<PROTEUS root>/socrates`, else the `RAD_DIR` you have set), `FC_DIR` only for a built FastChem, and `PYTHON_JULIAPKG_EXE` only when it already names an executable Julia; `bash install.sh` selects Julia 1.13 and sets it.
    - To refresh an existing installation later, use `proteus update-all` (see [Diagnose and update](doctor.md)). It runs no git command in the Python module clones; as with `bash install.sh`, its pip step installs the PyPI release in place of a clone that is older than the minimum version. To move a clone to its release tag after a version change, run the commands that `bash install.sh` prints for a clone that is not at its tag, or the module's setup script with `--force`.
    
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
    prints a warning at activation when a value that your shell start file sets
    differs from the one bound to the environment. You can delete the
    old lines by hand; while they stay, a shell without the activated environment
    still gets the old values.

## Uninstall

Removing the conda environment (`conda deactivate`, then `conda env remove -n proteus`)
removes the environment variables with it; delete the PROTEUS folder and the
`FWL_DATA` folder by hand.
