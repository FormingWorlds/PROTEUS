# Quick installation

The fastest way to get a working PROTEUS installation is the unified installer
script. It handles SOCRATES, AGNI, all Python submodules, environment variables,
and reference data downloads in a single command. You install Julia yourself
before you run it: the installer checks for Julia and does not install it.

!!! info "Prerequisites"
    - Tested in CI: Linux x86_64 (Ubuntu) and macOS on Apple Silicon. Intel macOS and
      Linux ARM64 are not tested.
    - ~20 GB disk space (conda, Julia, reference data, submodules)
    - Standard command-line tools: `curl`, `wget`, `make`
    - Git
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
    sudo apt update
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

## 2. Install Julia

PROTEUS needs Julia 1.13. The installer stops, before it builds anything, when it does not find it. Your own Julia setup stays yours: nothing here changes your default Julia.

If you do not have Julia yet, install it with the official installer, accept its default answers, and open a new terminal, so that `juliaup` is on your `PATH`:

```console
curl -fsSL https://install.julialang.org | sh
```

Then add Julia 1.13. This keeps any other Julia version that you have, and your default stays as it is:

```console
juliaup add 1.13
```

**Compute clusters**: a home folder on a cluster can be too small for Julia. On Kapteyn and Cambridge IoA, follow the Julia steps of the [Kapteyn](kapteyn_cluster_guide.md) or [Cambridge IoA](ioa_cluster_guide.md) guide instead of the commands above; they place Julia on the data drive. On another cluster, the steps above apply.

!!! note "A Julia without juliaup"
    A `julia` of version 1.13 on your `PATH` is enough, wherever it comes from. For a Julia 1.13 that is not on your `PATH`, set `PYTHON_JULIAPKG_EXE` to its binary before you run the installer. A system package manager often installs another version of Julia: then use the steps above. A juliaup depot that you cannot write to counts as no Julia, because juliaup needs its lock file there: set `PYTHON_JULIAPKG_EXE` to the binary of a Julia 1.13 instead.

## 3. Clone PROTEUS and set up Python environment

!!! tip "Experimental: pixi instead of conda"
    PROTEUS can also be installed into a [pixi](https://pixi.sh) environment,
    which replaces the conda steps below. This route is experimental and not yet
    recommended for new users: if you install PROTEUS for the first time, follow
    the conda steps of this page. See [Install with pixi](pixi.md).

Python **3.12** is required, and is installed via
[miniforge](https://github.com/conda-forge/miniforge) or
[miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install). If you have neither yet, install miniforge
(the same two commands on macOS and Linux):

```console
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Answer `yes` when the installer asks whether to initialize conda for your
shell. Open a new terminal afterwards, so that `conda` is on your `PATH`.

!!! note "If you use miniconda"
    Miniconda takes its packages from the Anaconda default channels, and conda
    creates an environment from them only after you accept their terms of
    service. When conda stops with that message, run the `conda tos accept`
    commands that it prints, then repeat the step.

!!! note "Install miniconda/miniforge in your personal directory"
    Installing miniconda/miniforge in your personal directory gives you full control over your environment and is recommended even on clusters (see the cluster guides: [Kapteyn](kapteyn_cluster_guide.md), [Habrok](habrok_cluster_guide.md), [Snellius](snellius_cluster_guide.md), [Cambridge IoA](ioa_cluster_guide.md)). 

Then clone PROTEUS:

```console
git clone https://github.com/FormingWorlds/PROTEUS.git
cd PROTEUS
```

!!! tip "Cloning over SSH"
    To push changes back to GitHub, clone over SSH instead. This needs an
    [SSH key registered with GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh):

    ```console
    git clone git@github.com:FormingWorlds/PROTEUS.git
    ```

## 4. Run the installer

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

The installer runs through the following phases automatically:

1. Pre-flight checks (OS, disk space, Python version, system dependencies)
2. Julia check: the installer uses the Julia 1.13 that it finds, and stops with the commands to run when it finds none
3. Environment variables (`FWL_DATA`, `PYTHON_JULIAPKG_EXE`, and in phases 4 and 5 `RAD_DIR` and, for a built FastChem, `FC_DIR`): the installer writes each to `.proteus-env.sh` in the PROTEUS folder as it sets it, and sets them all in the active conda environment in one step at the end of phase 6, after the check that PROTEUS imports and before the further modules of phase 7; a value that holds a single quote stays in `.proteus-env.sh` only. It edits no shell start file
4. SOCRATES compilation and `RAD_DIR` setup
5. AGNI and FastChem setup (Julia atmosphere model + equilibrium chemistry); an AGNI folder that is there already stays, and when its `Manifest.toml` names another minor Julia version (1.12 against 1.13, for example) than the one PROTEUS uses, the installer warns and names `proteus update-all`, which instantiates AGNI again
6. Python packages (editable installs of all submodules, the SUNDIALS CVODE solver, and PROTEUS itself)
7. Further modules (Morrigan, VULCAN, atmodeller and Obliqua): a failed module prints a warning with the command to repeat and the installer goes on; after the four steps the installer checks again that PROTEUS imports, and stops when it does not
8. Reference data downloads (by default the essential set: the default-config and tutorial k-tables, stellar spectra and tracks, the PHOENIX grid of the default star, and the interior tables)
9. Verification via `proteus doctor`: when an installed Python module is still below its minimum version in `pyproject.toml`, for example a VULCAN clone that stays, the installer prints the fix command of each such module and stops with exit status 1

The installer clones the four further modules into the PROTEUS folder, each under its own licence: atmodeller (GPL-3.0), VULCAN (GPL-3.0), Morrigan (Apache-2.0) and Obliqua (MIT). PROTEUS itself is under Apache-2.0.

The installer and the setup scripts that can clone over SSH (those of the Python modules, SOCRATES, LavAtmos and ThermoEngineLite) test the SSH access to GitHub without a question on the terminal. When the test fails (no key that GitHub accepts, a key with a passphrase that no SSH agent holds, or no host key of github.com on this machine), they clone over HTTPS. `get_agni.sh`, `get_obliqua.sh` and `get_spider.sh` have no SSH test: they clone the HTTPS URL that `pyproject.toml` gives.

The installer prints one line per build step and writes the full build output to `install_<date>_<time>.log` (for example `install_20261011_142530.log`) in the PROTEUS folder; when a step fails, it also prints the last 40 lines of that step's output.

PROTEUS runs on Julia 1.13. The installer looks for it in this order, and your default Julia stays as it is:

1. `PYTHON_JULIAPKG_EXE`, when it names a Julia 1.13.
2. The `julia` on your `PATH`, when it is version 1.13.
3. The `1.13` channel of juliaup.

When it finds none of them, the installer stops with the commands of [step 2](#2-install-julia). With no conda environment to reuse, it stops before it makes one.

When juliaup is not on your `PATH`, the installer looks for it in `~/.juliaup/bin`. When it finds no Julia 1.13 and no juliaup there, but a juliaup configuration exists (`juliaup/juliaup.json` under `JULIAUP_DEPOT_PATH`, by default `~/.julia`), it names that file and asks you to put your juliaup on your `PATH`.

PROTEUS and the setup scripts in `tools/` find Julia 1.13 through `PYTHON_JULIAPKG_EXE`, so a bare `julia` in your terminal is still your own default.

**Julia packages per install.** The installer keeps the Julia packages and compile files of each install in its own folder, `julia_depot` in the conda environment, and sets `JULIA_DEPOT_PATH` to that folder followed by a colon. Two installs on one machine do not share these files, and your `~/.julia` does not change. juliaup still keeps its Julia versions in `~/.julia/juliaup`, or under `JULIAUP_DEPOT_PATH`. One depot with all modules takes about 940 MiB and 21500 files, and at least 220 MB to download (measured on macOS arm64 with Julia 1.13.3).

- The colon matters. It keeps the compile files that come with Julia and does not add `~/.julia`; without it, each depot compiles the standard library again.
- A `JULIA_DEPOT_PATH` that is already set stays, with 2 exceptions. A `julia_depot` of another PROTEUS install is replaced, and so is one whose folder is gone from an `envs` folder, as after the removal of an environment. Any other missing folder stays, for example on a disk that is not mounted. A value that is relative, names another user's home folder or holds a literal `$` stops the installer before it changes anything, and the message names the way out. Set a value for a small home folder with conda on it, on an environment with no value: `JULIA_DEPOT_PATH=/big/disk/proteus-a: bash install.sh`. On an environment with a value, that value wins over the shell, and the installer prints one line. Set the new value there first with `conda env config vars set -n <env> JULIA_DEPOT_PATH=/big/disk/proteus-a:`, run `conda activate <env>` when that environment is active, and then run `bash install.sh`. This has one limit: all installs that see this value share one depot. For a depot of its own, remove the value from your shell start file and open a new shell. Then activate the environment, run `conda env config vars unset JULIA_DEPOT_PATH`, activate the environment again and run `bash install.sh`.
- An environment with no value of its own keeps the value of the environment it was activated from, as for `FWL_DATA`. The value of another environment that a switch leaves live in `base` reaches a new environment in the same way, and the installer then binds it. To leave it, run `conda deactivate` until no environment is active, then activate the environment.
- An install from before this rule keeps working on `~/.julia` until you run `bash install.sh` again. That run moves it to its own depot, with no question. It first prints the size and the commands that stay on the old depot: `conda env config vars set -p "$CONDA_PREFIX" JULIA_DEPOT_PATH=$HOME/.julia:`, `conda activate "$CONDA_PREFIX"` and `bash install.sh`. For an old depot in another folder, name that folder in the first command. The installer deletes nothing in `~/.julia`; do not remove its `juliaup` folder.

The installer installs every Python module (MORS, JANUS, CALLIOPE, ZEPHYRUS, Aragog, Zalmoxis, Morrigan, VULCAN and atmodeller) and PROTEUS itself in editable mode, each from a clone in the PROTEUS folder, so a change in a clone takes effect without a new install. fwl-io, which downloads the reference data, is installed from PyPI.

- A new module clone is at the release tag of the module's minimum version in `pyproject.toml`; a new atmodeller or Obliqua clone is at the commit that `pyproject.toml` pins under `[tool.proteus.modules]`.
- A new Aragog or Zalmoxis clone takes the branch of the same name instead when PROTEUS is on a feature branch that the module also has.
- Running the installer again does not update a module clone that is already there. It installs the clone as it is and names each one that is not at its tag or pinned commit, with the commands that move it there and reinstall it, and the command that replaces it with a new clone, for example `bash tools/get_python_module.sh MORS --force`.

!!! note "The one exception: a clone older than the minimum version"
    If a clone of MORS, JANUS, CALLIOPE, ZEPHYRUS, Aragog or Zalmoxis that stays is older than the minimum version, the PROTEUS install that follows replaces its editable install with the release from PyPI. PROTEUS then imports that release, not the clone, and `proteus doctor` shows the version without the `[editable @ ...]` mark. To use the clone again, move it to its tag and reinstall it, or replace it with a new clone, with the commands that the installer prints.

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
    It chooses no Julia depot: with no `JULIA_DEPOT_PATH` set, the Julia packages go to `~/.julia`, shared with every other install that has none, and `proteus doctor` warns. To give the install its own depot, run this before `proteus install-all`, and add `--export-env` to record the value: `conda env config vars set JULIA_DEPOT_PATH="$CONDA_PREFIX/julia_depot:"`, then `conda deactivate && conda activate proteus`. With the `julia_depot` of another install in the shell, the Julia steps of `proteus install-all` and `proteus update-all` use the depot of the active environment, or `~/.julia` with no environment, and print one line. The shell keeps its value until `--export-env` records that depot and you activate the environment again.

    - Pass `--export-env` to set the environment variables in the active conda environment and write them to `.proteus-env.sh`, as the installer does. It sets nothing in `base`, keeps a value that holds a single quote in `.proteus-env.sh` only, records `RAD_DIR` only for a folder that holds a built SOCRATES (`<PROTEUS root>/socrates`, else the `RAD_DIR` you have set), `FC_DIR` only for a built FastChem, and `PYTHON_JULIAPKG_EXE` only when it already names an executable Julia; `bash install.sh` selects Julia 1.13 and sets it.
    - To refresh an existing installation later, use `proteus update-all` (see [Diagnose and update](doctor.md)). It runs no git command in the Python module clones; as with `bash install.sh`, its pip step installs the PyPI release in place of a clone of MORS, JANUS, CALLIOPE, ZEPHYRUS, Aragog or Zalmoxis that is older than the minimum version. To move a clone to its release tag after a version change, run the commands that `bash install.sh` prints for a clone that is not at its tag, or the module's setup script with `--force`.
    
    Both commands operate on the PROTEUS source tree, which they locate from the
    installed package (editable installs) or, for a plain wheel install, from
    the current directory when it is a PROTEUS clone; with neither available
    they exit with an error.

## 5. Verify and run

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
    A `JULIA_DEPOT_PATH` line also sets the Julia depot of a conda environment with no value of its own.

## Uninstall

Removing the conda environment (`conda deactivate`, then `conda env remove -n proteus`)
removes the environment variables with it; delete the PROTEUS folder and the
`FWL_DATA` folder by hand.
