# Diagnosing and updating your installation

PROTEUS provides two commands for keeping your installation healthy:

- **`proteus doctor`** checks your environment, reference data, and package
  versions against the requirements in `pyproject.toml`.
- **`proteus update`** runs the same checks, then executes the suggested fix
  commands automatically.

## Quick reference

```console
proteus doctor              # diagnose
proteus doctor --json       # machine-readable output
proteus update --dry-run    # preview fixes without executing
proteus update              # apply all fixes
```

---

## proteus doctor

`proteus doctor` runs a sequence of checks grouped into three categories and
reports each as **pass**, **warn**, or **fail**. Failing checks include a fix
command you can run manually or let `proteus update` handle.

### Environment

| Check | What it verifies | Fail condition |
|-------|-----------------|----------------|
| `FWL_DATA` | Environment variable is set and the directory exists | Not set, or path missing |
| `RAD_DIR` | Set, directory exists, and `bin/radlib.a` is present | Not set, path missing, or SOCRATES not compiled |
| `FC_DIR` | Set and directory exists | Not set (only required when AGNI chemistry is enabled) |
| `PYTHON_JULIAPKG_EXE` | Set | Not set |
| `julia` | Julia is on PATH and version is 1.11.x or 1.12.x | Missing or wrong version |

### Reference data

Checks that the essential data directories inside `$FWL_DATA` are present and
non-empty:

| Check | Fix command |
|-------|-------------|
| `FWL_DATA/spectral_files` | `proteus get spectral` |
| `FWL_DATA/stellar_spectra` | `proteus get stellar` |

### Package versions

For each Python submodule (fwl-proteus, fwl-aragog, fwl-calliope, fwl-janus,
fwl-mors, fwl-vulcan, fwl-zephyrus, fwl-zalmoxis), the doctor checks:

1. **Is the package installed?** If not, the fix is `pip install <name>`.
2. **Does the installed version satisfy the minimum bound in `pyproject.toml`?**
   For example, if `pyproject.toml` requires `fwl-aragog>=26.05.13` and you
   have `25.1.1` installed, the check fails.
3. **Is this an editable install?** If so, the output shows the checkout
   directory, git commit hash, and dirty state.

For git-pinned modules (AGNI, SOCRATES), the doctor compares the checkout's
HEAD commit against the commit SHA pinned in `pyproject.toml` under
`[tool.proteus.modules]`. A mismatch produces a warning with a fix command.
The AGNI fix is `bash tools/get_agni.sh`. The SOCRATES fix also rebuilds AGNI,
because re-cloning the SOCRATES tree removes the wrappers AGNI compiles into
`socrates/julia`, so it reads
`bash tools/get_socrates.sh "$RAD_DIR" && RAD_DIR="$RAD_DIR" bash tools/get_agni.sh 0`.

!!! info "pyproject.toml is the version authority"
    The doctor compares against `pyproject.toml`, not against the latest
    release on PyPI. A package that satisfies the version bound shows "ok"
    even if a newer version exists on PyPI that PROTEUS has not adopted yet.
    This prevents false "update available" warnings for untested versions.

    To change which versions PROTEUS pins, see
    [Updating module pins](update_module_pins.md).

### Example output

```
Environment
  [ok] FWL_DATA: /home/user/FWL_DATA
  [ok] RAD_DIR: /home/user/PROTEUS/socrates
  [FAIL] FC_DIR: not set
       fix: export FC_DIR=<path>  # add to your shell rc file
  [ok] julia: 1.11.8

Reference data
  [ok] FWL_DATA/spectral_files: present
  [ok] FWL_DATA/stellar_spectra: present

Package versions
  [ok] fwl-proteus: 25.10.15 [editable @ PROTEUS -> a1b2c3d]
  [ok] fwl-aragog: 26.5.13 [editable @ aragog -> d051902]
  [FAIL] fwl-vulcan: not installed
       fix: pip install fwl-vulcan
  [warn] AGNI: 1.10.1 (bf65a56c) differs from pin (b06a3fed)
       fix: bash tools/get_agni.sh

2 failed, 1 warnings

Run proteus update to fix 3 issue(s) automatically.
```

### JSON output

For CI pipelines or scripted checks, use `--json` to get machine-readable
output:

```console
proteus doctor --json
```

Each check is a JSON object with five fields:

```json
{
  "name": "fwl-aragog",
  "category": "versions",
  "status": "pass",
  "message": "26.5.13 [editable @ aragog -> d051902]",
  "fix_cmd": null
}
```

The `status` field is one of `"pass"`, `"warn"`, or `"fail"`. The `fix_cmd`
field is `null` for passing checks and a shell command string for failing ones.

---

## proteus update

`proteus update` runs the same checks as `proteus doctor`, collects all fix
commands from failing and warning checks, and executes them in sequence.

```console
proteus update
```

After all fixes are applied, the doctor runs again to verify the result.

### Dry run

To preview what `proteus update` would do without making changes:

```console
proteus update --dry-run
```

This lists the fixable issues and their commands but does not execute anything.

### What proteus update can fix

| Issue | Fix action |
|-------|-----------|
| Missing Python package | `pip install <name>` |
| Outdated Python package (editable, tag-pinned: aragog, zalmoxis) | `bash tools/get_<name>.sh` |
| Outdated Python package (editable, tracking branch) | `cd <checkout> && git pull && pip install -e .` |
| Outdated Python package (wheel) | `pip install -U "<name>>=<version>"` |
| Missing AGNI | `bash tools/get_agni.sh` |
| AGNI commit drift from pin | `bash tools/get_agni.sh` |
| Missing SOCRATES | `bash tools/get_socrates.sh` |
| SOCRATES commit drift from pin | `bash tools/get_socrates.sh "$RAD_DIR" && RAD_DIR="$RAD_DIR" bash tools/get_agni.sh 0` (rebuilds AGNI's wrappers too). To discard local changes in the SOCRATES checkout, pass `--force` to the SOCRATES step only: `bash tools/get_socrates.sh --force "$RAD_DIR" && RAD_DIR="$RAD_DIR" bash tools/get_agni.sh 0` |
| Missing reference data | `proteus get spectral` / `proteus get stellar` |
| Wrong Julia version | `juliaup add 1.12 && juliaup default 1.12` |

### What proteus update cannot fix

- **Missing environment variables** (`FWL_DATA`, `RAD_DIR`, etc.): these
  require editing your shell configuration file (`~/.bashrc` or `~/.zshrc`).
  The doctor reports the fix command, but `proteus update` cannot modify your
  shell rc file. Run the suggested `export` command manually, then add it to
  your rc file.
- **Missing system packages** (gfortran, cmake, netcdf): these require your
  system package manager (`brew`, `apt`, `dnf`). See the
  [installation guide](installation.md) for platform-specific commands.
- **Conda environment issues**: `proteus update` runs inside the active
  conda environment. If the wrong environment is active or conda is not
  configured, activate the correct environment first:
  `conda activate proteus`.

!!! warning "Source install required"
    `proteus update` requires a source install (git clone). If PROTEUS was
    installed from a wheel (`pip install fwl-proteus`), the `tools/` directory
    is not available and fix commands that reference it will not work. The
    command detects this and exits with a message.

## proteus update-all

Where `proteus update` applies targeted fixes for the specific issues the doctor
finds, `proteus update-all` performs a full refresh of the whole stack:

```console
proteus update-all
```

It updates the PROTEUS Python package, recompiles SOCRATES, pulls the latest
AGNI, and refreshes the reference data, verifying environment variables and disk
space before proceeding. Pass `--export-env` to re-export the environment
variables to your shell rc file. Use this after pulling new code when several
components may have moved at once; use `proteus update` when you want to apply
only the fixes the doctor has flagged.

---

## Typical workflows

### After a fresh install

Run `proteus doctor` to verify everything is set up correctly:

```console
conda activate proteus
proteus doctor
```

If any checks fail, run `proteus update` to fix what can be fixed
automatically, then address any remaining issues (environment variables,
system packages) manually.

### After pulling new code

When you pull changes that bump submodule versions in `pyproject.toml`:

```console
git pull
proteus update
```

This updates any submodules whose installed version no longer satisfies the
new bounds. For editable checkouts installed by a `tools/get_<name>.sh` script
(aragog, zalmoxis), it re-runs that script, which fetches and checks out the
pinned version tag; those checkouts sit on a detached HEAD where `git pull`
would fail. For editable checkouts on a tracking branch, it runs
`git pull && pip install -e .` in the checkout directory.

### Before submitting a simulation

A quick `proteus doctor` catches common issues (missing data, wrong Julia
version, drifted AGNI checkout) before a long simulation fails at runtime:

```console
proteus doctor
proteus start --offline -c input/my_config.toml
```

---

**See also:** [Installation](installation.md) | [Updating module pins](update_module_pins.md) | [Troubleshooting](troubleshooting.md) | [Configuration](config.md) | [Usage](usage.md)
