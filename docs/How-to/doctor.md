# Diagnosing and updating your installation

Use this page when an install is broken or old. It tells you what a line of `proteus doctor` output means and what to run.

PROTEUS has three commands for this:

| Command | What it does | When to use it |
|---------|--------------|----------------|
| **`proteus doctor`** | Checks your environment, reference data and package versions against the requirements in `pyproject.toml`. It changes nothing in your install. | First, when something is wrong, or before a long run. |
| **`proteus update`** | Runs the same checks, then runs the fixes that it can run for you. | When the doctor reports a problem. |
| **`proteus update-all`** | Refreshes the whole stack. It does not use the results of the doctor. | After you pull new code, when several components may have moved at once. |

## Quick reference

```console
proteus doctor              # diagnose
proteus doctor --json       # machine-readable output
proteus update --dry-run    # preview fixes without executing
proteus update              # apply the fixes it can run
proteus update-all          # refresh the whole stack
```

---

## proteus doctor

### How to read the output

The doctor prints one line per check, in three groups. This example is shortened:

```
Environment
  [ok] FWL_DATA: /home/user/FWL_DATA
  [ok] RAD_DIR: /home/user/PROTEUS/socrates
  [FAIL] FC_DIR: not set
       fix: conda env config vars set FC_DIR=<path>, then conda deactivate && conda activate proteus (without conda: source .proteus-env.sh in the PROTEUS folder)
  [ok] PYTHON_JULIAPKG_EXE: /home/user/julia-1.13.0/bin/julia
  [ok] julia: 1.13.0 at /home/user/julia-1.13.0/bin/julia (PYTHON_JULIAPKG_EXE)
  [ok] julia depot: /home/user/miniforge3/envs/proteus/julia_depot
  [ok] cvode: available for the Aragog interior

Reference data
  [ok] FWL_DATA/atmos_clim/spectral_files: present
  [ok] FWL_DATA/star/spectra: present

Package versions
  [ok] fwl-proteus: 25.10.15 [editable @ PROTEUS -> a1b2c3d]
  [ok] fwl-aragog: 26.5.13 [editable @ aragog -> d051902]
  [FAIL] fwl-janus: not installed
       fix: pip install fwl-janus
  [warn] AGNI: 1.10.1 (bf65a56c) differs from pin (b06a3fed)
       fix: bash tools/get_agni.sh

2 failed, 1 warnings

Run proteus update to fix 2 issue(s) automatically.
```

Each line starts with a status:

| Status | Meaning |
|--------|---------|
| `[ok]` | The check passed. |
| `[warn]` | Look at this line. A warning alone does not make the command exit with an error. |
| `[FAIL]` | Something is missing or wrong. |

A line with a problem can have a `fix:` line under it. Run a fix from the PROTEUS folder.

The `Run proteus update` line counts only the fixes that `proteus update` runs for you. In the example, it counts `fwl-janus` and `AGNI`. With no such fix, the line is absent. The fix for `FC_DIR` is advice that you follow by hand.

When a check fails, the doctor exits with status 1. After the summary, it saves a log file in the current folder, or in the system temporary folder when it cannot write there. It then prints where to get help. Send that log file when you ask.

The tables below list each situation. The column "Run by update" says whether `proteus update` runs the fix.

### Environment

#### Environment variables

The doctor checks `FWL_DATA`, `RAD_DIR`, `FC_DIR` and `PYTHON_JULIAPKG_EXE`.

| Situation | Status | What to run | Run by update |
|-----------|--------|-------------|-------------------------|
| The variable is not set | FAIL | Set it (commands below) | No |
| The variable is set, but the path does not exist | warn | Correct the value. The doctor prints no fix line. | No |
| `RAD_DIR` exists, but `bin/radlib.a` is missing | warn | SOCRATES is not compiled. The doctor prints no fix line. See the command below. | No |

!!! warning "This deletes the folder"
    The first part of this command deletes the folder that `RAD_DIR` names, then clones SOCRATES there and compiles it. It refuses a git clone with modified tracked files or commits on no remote. It deletes any other folder without a question. The second part rebuilds AGNI and needs Julia.

To compile SOCRATES, run this command:

```console
bash tools/get_socrates.sh "$RAD_DIR" && RAD_DIR="$RAD_DIR" bash tools/get_agni.sh 0
```

To set a variable in the conda environment, run these commands. Conda loads the variables only at activation, so you must deactivate first.

```console
conda env config vars set NAME=<path>
conda deactivate && conda activate proteus
```

Without conda, run `source .proteus-env.sh` in the PROTEUS folder. This file holds the values that `bash install.sh` or `--export-env` recorded. The `fix:` line names your active environment, or a placeholder when no environment or `base` is active.

The doctor fails on an unset `FC_DIR` in every case. A run needs this variable when AGNI chemistry is enabled or LavAtmos is in use.

#### Julia

The `julia` line checks the Julia in `PYTHON_JULIAPKG_EXE`. When that variable is not set, it checks the `julia` on `PATH`.

| Situation | Status | What to run | Run by update |
|-----------|--------|-------------|-------------------------|
| Version 1.13.x | ok | Nothing | |
| Any other version | warn | `juliaup add 1.13`, then set `PYTHON_JULIAPKG_EXE` to the path that `julia +1.13 -e 'print(joinpath(Sys.BINDIR, "julia"))'` prints | No |
| `PYTHON_JULIAPKG_EXE` is set to a path that is not an executable file | FAIL | The same commands as for another version | No |
| Julia is missing or does not run | FAIL | `curl -fsSL https://install.julialang.org | sh` | Yes |

Another version only warns, because PROTEUS is tested on 1.13 and Obliqua installs only on 1.11 or 1.13. A path that is not an executable file fails, because juliacall then fails at import. It does not use the `julia` on `PATH` in that case. These commands do not change your default Julia.

When `PYTHON_JULIAPKG_EXE` names a Julia that does not run, the install does not change the variable. You must then set it by hand, with the commands for another version.

The installer binds Julia 1.11 only when the import of PROTEUS fails on `OpenSSL_jll`, because Python links an OpenSSL older than 3.5. After `conda install -c conda-forge 'openssl>=3.5'`, Julia 1.13 can be used.

#### Julia depot

The `julia depot` line names the Julia depot in use, the folder where Julia keeps its packages and compile files.

| Situation | Status | What to run | Run by update |
|-----------|--------|-------------|-------------------------|
| `JULIA_DEPOT_PATH` is not set | warn | Give the install its own depot (commands below). | No |
| The value is empty | FAIL | The same commands | No |
| The first entry is relative, starts with another user's home folder (`~name`), or holds a literal `$` | FAIL | The same commands | No |
| The first folder cannot be written | FAIL | The same commands | No |
| The folder is the depot of another PROTEUS install, or of an environment that is gone from an `envs` folder | FAIL | The same commands | No |
| A direct dependency of the AGNI or `julia_env` project has no folder under `packages` | FAIL | `LD_LIBRARY_PATH="" "$PYTHON_JULIAPKG_EXE" --project=<project> -e "using Pkg; Pkg.instantiate()"` | No |
| The `Project.toml` or the Manifest of one of these projects cannot be read | FAIL | Restore the file, for example with `bash install.sh` | No |

In the active PROTEUS environment, these commands give the install its own depot. Keep the colon at the end of the value.

```console
conda env config vars set JULIA_DEPOT_PATH="$CONDA_PREFIX/julia_depot:"
conda deactivate && conda activate proteus
```

With no `JULIA_DEPOT_PATH`, the install shares `~/.julia` with every other install that has none. An install in this state works.

The package check reads the first of `JuliaManifest-v1.13.toml`, `Manifest-v1.13.toml`, `JuliaManifest.toml` and `Manifest.toml` that exists, as Julia does. It does not read the versions or the indirect dependencies.

#### CVODE

The `cvode` line checks that Aragog can use the SUNDIALS CVODE solver.

| Situation | Status | What to run | Run by update |
|-----------|--------|-------------|-------------------------|
| Not installed, or installed but does not load | warn | `bash tools/get_cvode.sh` | Yes |

### Reference data

The doctor checks that the essential data folders inside `$FWL_DATA` are present and not empty. It skips these checks when `FWL_DATA` is not set or the folder does not exist.

| Situation | Status | What to run | Run by update |
|-----------|--------|-------------|-------------------------|
| `FWL_DATA/atmos_clim/spectral_files` is missing or empty | warn | `proteus get spectral` | Yes |
| `FWL_DATA/star/spectra` is missing or empty | warn | `proteus get stellar` | Yes |
| One of these is missing, but its data is in an older folder that can move | warn | `fwl-io relocate` | Yes |

### Package versions

#### Python packages

The doctor checks these packages:

- Required: fwl-proteus, fwl-io, fwl-aragog, fwl-calliope, fwl-janus, fwl-mors, fwl-zephyrus, fwl-zalmoxis.
- Standard module packages: fwl-morrigan, fwl-vulcan, atmodeller.

| Situation | Status | What to run | Run by update |
|-----------|--------|-------------|-------------------------|
| A required package is not installed | FAIL | `pip install <name>` | Yes |
| A standard module package is not installed | warn | `bash tools/get_morrigan.sh`, `bash tools/get_vulcan.sh` or `bash tools/get_atmodeller.sh` | Yes |
| The version is below the minimum, in a wheel install | FAIL | `pip install -U "<name>>=<version>"` | Yes |
| The version is below the minimum, in an editable git clone of its own that no setup script makes | FAIL | `cd <checkout> && git pull && pip install -e .` | Yes |
| The version is below the minimum, in an editable clone that a setup script makes | FAIL | See [Rare cases](#rare-cases) | No |
| The version cannot be parsed | warn | `pip install -U "<name>>=<version>"` | Yes |

For example, if `pyproject.toml` requires `fwl-aragog>=26.05.13` and you have `25.1.1` installed, the check fails.

In a tree with no setup script, the fix for a missing standard module package is `pip install "fwl-proteus[<extra>]"`.

For an editable install, the line shows the checkout folder, the git commit hash and the dirty state. A clone that a setup script makes gets a note when it is behind the release tag of its minimum version or on another commit, for example `on branch main; PROTEUS expects the release tag 26.10.06`. A clone newer than the tag, an Aragog or Zalmoxis clone on the branch that PROTEUS is on, and a commit-pinned module get no note. The note alone does not make the check fail.

#### Git-pinned modules

For AGNI, SOCRATES and Obliqua, the doctor compares the commit of the checkout with the commit that `pyproject.toml` pins under `[tool.proteus.modules]`.

| Situation | Status | What to run | Run by update |
|-----------|--------|-------------|-------------------------|
| The AGNI folder is missing or empty | FAIL | `bash tools/get_agni.sh` | Yes |
| The SOCRATES folder is missing or empty | FAIL | `bash tools/get_socrates.sh "$RAD_DIR"`. With `RAD_DIR` not set: `bash tools/get_socrates.sh` | Yes |
| The Obliqua folder is missing or empty | warn | `bash tools/get_obliqua.sh` | Yes |
| AGNI is not at its pin | warn | `bash tools/get_agni.sh` | Yes |
| SOCRATES is not at its pin | warn | `bash tools/get_socrates.sh "$RAD_DIR" && RAD_DIR="$RAD_DIR" bash tools/get_agni.sh 0` | Yes |
| Obliqua is not at its pin | warn | `bash tools/get_obliqua.sh --force <folder>` | No |
| Obliqua is at its pin, with no `Manifest.toml` | warn | `bash tools/get_obliqua.sh`, which instantiates its Julia environment | Yes |

The SOCRATES fix also rebuilds AGNI. A new clone of the SOCRATES tree removes the wrappers that AGNI compiles into `socrates/julia`.

To discard local changes in the SOCRATES checkout, pass `--force` to the SOCRATES step only:

```console
bash tools/get_socrates.sh --force "$RAD_DIR" && RAD_DIR="$RAD_DIR" bash tools/get_agni.sh 0
```

The Obliqua fix replaces the folder with a new clone at the pin, so you run it by hand. A file at the module path, a folder that is not a git clone of its own, and a commit that git cannot read each get advice in place of these fixes. See [Rare cases](#rare-cases).

!!! info "pyproject.toml is the version authority"
    The doctor compares against `pyproject.toml`, not against the latest
    release on PyPI. A package that satisfies the version bound shows "ok"
    even if a newer version exists on PyPI that PROTEUS has not adopted yet.
    This prevents false "update available" warnings for untested versions.

    To change which versions PROTEUS pins, see
    [Updating module pins](update_module_pins.md).

### JSON output

For CI pipelines or scripted checks, use `--json` to get machine-readable output:

```console
proteus doctor --json
```

The output is a list with one JSON object per check. Each object has six fields:

```json
{
  "name": "fwl-aragog",
  "category": "versions",
  "status": "pass",
  "message": "26.5.13 [editable @ aragog -> d051902]",
  "fix_cmd": null,
  "auto_fixable": true
}
```

- `status` is `"pass"`, `"warn"` or `"fail"`.
- `fix_cmd` is the text of the `fix:` line. It is `null` when the check has no fix. It can be advice in place of a shell command.
- `auto_fixable` is `false` when `proteus update` does not run the fix. A check with no fix can show either value, so read `fix_cmd` first.

The command exits with status 1 when a check fails. It writes no log file.

---

## proteus update

`proteus update` runs the same checks as `proteus doctor`. Then it runs each fix that the tables above mark with Yes, in sequence.

```console
proteus update
```

After all fixes are applied, the doctor runs again to verify the result. The command exits with status 1 when a fix fails or a check still fails. It then saves a log file, as the doctor does.

### Dry run

To preview what `proteus update` would do without making changes:

```console
proteus update --dry-run
```

This lists the issues that `proteus update` fixes, with their commands. It also lists the issues that need action by hand. It runs nothing.

### What proteus update cannot fix

You must act by hand on every row marked No in the tables above and on the advice in [Rare cases](#rare-cases). These cases also need you:

| Case | What to do |
|------|------------|
| Missing system packages (gfortran, cmake, netcdf) | Use your system package manager (`brew`, `apt`, `dnf`). See the [installation guide](installation.md) for platform-specific commands. |
| Conda environment issues | `proteus update` runs inside the active conda environment. Activate the correct environment first: `conda activate proteus`. |

!!! warning "Source install required"
    `proteus update` requires a source install (git clone). If PROTEUS was
    installed from a wheel (`pip install fwl-proteus`), the `tools/` directory
    is not available and fix commands that reference it will not work. The
    command detects this and exits with a message.

---

## proteus update-all

`proteus update` applies targeted fixes for what the doctor finds. `proteus update-all` refreshes the whole stack:

```console
proteus update-all
```

It does these steps in order:

1. It stops when the disk has 5 GB or less of free space.
2. It updates the PROTEUS Python package.
3. It installs the SUNDIALS CVODE solver. A failure here only gives a warning.
4. It stops when the `FWL_DATA` folder does not exist.
5. It replaces the SOCRATES tree with a new clone at the pin and compiles it. With no `socrates` folder, it prints a message and skips this step. When a git clone has modified tracked files or commits on no remote, the script refuses. The command then prints a failure and goes on.
6. With a Julia and an `AGNI` folder, it brings AGNI to the commit pinned in `pyproject.toml`, which also rebuilds its wrappers in the new SOCRATES tree. An AGNI clone with modified files or commits of your own stays as it is. A folder that is not a git clone of its own stays too. The command prints what to do by hand. When AGNI stays, its wrappers are not rebuilt.
7. It prints the setup command for Morrigan, VULCAN, atmodeller or Obliqua when that module is not installed.
8. It refreshes the reference data for the config file that `--config-path` names. The default is `input/all_options.toml`. With no such file, it skips this step.

Pass `--export-env` to set the environment variables again in the active conda environment and in `.proteus-env.sh`.

It runs no git command in the Python module clones (MORS, JANUS, CALLIOPE, ZEPHYRUS, Aragog, Zalmoxis, Morrigan, VULCAN, atmodeller) and none in Obliqua. It does not touch Morrigan, VULCAN and atmodeller.

One exception: as with `bash install.sh`, its pip step installs the PyPI release in place of a clone of MORS, JANUS, CALLIOPE, ZEPHYRUS, Aragog or Zalmoxis that is older than the minimum version.

To move a clone to its release tag after a version change, run the commands that `bash install.sh` prints for a clone that is not at its tag. You can also run the setup script of the module with `--force`.

---

## Typical workflows

### After a fresh install

Run `proteus doctor` to verify everything is set up correctly:

```console
conda activate proteus
proteus doctor
```

If any checks fail, run `proteus update` to fix what can be fixed automatically. Then address any remaining issues (environment variables, system packages) by hand.

### After pulling new code

When you pull changes that bump module versions in `pyproject.toml`:

```console
git pull
proteus update
```

This updates the modules whose installed version no longer satisfies the new bounds, with these limits:

- An editable module clone that a setup script makes stays as it is. See [Rare cases](#rare-cases).
- Another editable checkout that is a git clone of its own gets `git pull && pip install -e .` in the checkout folder.
- Any other editable folder gets the advice `pip install -U "<name>>=<version>"`, which you run by hand.

### Before submitting a simulation

A quick `proteus doctor` catches common issues (missing data, wrong Julia version, drifted AGNI checkout) before a long simulation fails at runtime:

```console
proteus doctor
proteus start --offline -c input/my_config.toml
```

---

## Rare cases

Open a block when a line of output matches its title.

??? info "An outdated editable module clone that a setup script makes"
    This applies to MORS, JANUS, CALLIOPE, ZEPHYRUS, Aragog, Zalmoxis, Morrigan, VULCAN and atmodeller.

    The check fails. No fix changes such a clone. The `fix:` line shows these commands, and you run the one you want by hand:

    - Commands that fetch the tags, check out the release tag of the minimum version and reinstall the clone. For atmodeller they check out the pinned commit, not a release tag.
    - For the clone in the module's folder of the PROTEUS root: the setup script with `--force`, for example `bash tools/get_python_module.sh MORS --force`. This replaces the clone with a new one.

??? info "A folder that is not a git clone of its own"
    This is a folder with no `.git`, or with an empty one inside the PROTEUS checkout. Git then reports on the repository around it, so the doctor reads no commit there. The line says `not a git clone of its own` in place of the hash and the dirty state.

    **AGNI, SOCRATES or Obliqua.** The line reads `[not a git clone of its own]`. The doctor warns that the commit cannot be compared with the pin. The fix is advice: move the folder away or delete it, then run the setup script of the module.

    **An editable Python package below its minimum version.** The check fails. The fix is advice that `proteus update` does not run:

    | Folder | Fix |
    |--------|-----|
    | The module's own folder of the PROTEUS root, for a module with a release tag | The setup script with `--force` |
    | The module's own folder of the PROTEUS root, for atmodeller, which a commit pins | Move it away or delete it, then run the setup script again, as `get_atmodeller.sh` prints it |
    | A module folder elsewhere | Move it away, then run the setup script |
    | A package with no setup script | `pip install -U "<name>>=<version>"`, which installs the release in place of the folder |

??? info "A commit that git cannot read"
    This is a clone of its own, at the path of a pinned module, whose commit git cannot read. The doctor warns that the commit cannot be read. The fix is advice with two steps:

    1. Run `git -C <folder> log -1`.
    2. If this shows no commit, move the folder away or delete it, then run the setup script of the module.

??? info "A file or a broken link at the path of a module"
    This counts as not installed: a failure for AGNI and SOCRATES, a warning for Obliqua. The fix is advice: delete it, then run the setup script.

??? info "An Obliqua path that holds something else"
    **A clone of another project.** The doctor warns, with the advice to move it away or delete it.

    **A folder whose `.git` is a file** (a git worktree or a submodule). When it is not at its pin, the fix is the same advice, because `--force` refuses this folder.

??? info "Moving an Obliqua clone by hand"
    `bash tools/get_obliqua.sh` does not move an existing Obliqua clone. It prints the two git commands that move the clone to the pin by hand. It also prints `bash tools/get_obliqua.sh` as the next step, which sets up the Julia environment at that commit.

??? info "What tools/get_obliqua.sh refuses"
    With or without `--force`, the script refuses these destinations:

    - The PROTEUS folder, your home folder, and a folder that holds one of them.
    - A file or a broken link.
    - A path that holds a line break.
    - A path that goes through a folder that does not exist and then through `.` or `..`.
    - A folder with files that is not an Obliqua project.

    The forced form also refuses:

    - A folder with files that is not a git clone of its own with a commit.
    - A folder whose `.git` is a file (a git worktree or a submodule).

---

**See also:** [Installation](installation.md) | [Updating module pins](update_module_pins.md) | [Troubleshooting](troubleshooting.md) | [Configuration](config.md) | [Usage](usage.md)
