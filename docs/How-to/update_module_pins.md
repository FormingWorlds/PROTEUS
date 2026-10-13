# Updating module pins

PROTEUS pins the exact version of every module it depends on, and `pyproject.toml`
is the single source of truth for those pins. This page is the maintainer-side
companion to [Diagnose and update](doctor.md): the doctor page explains how a
user brings their installation *into line* with the pins, while this page
explains how a maintainer *changes* the pins and how that change propagates
through the rest of the stack.

For the current pinned versions of every module, see the
[Module versions reference](../Reference/module_versions.md).

## The two kinds of pin

PROTEUS uses two pinning mechanisms, and which one applies depends on how the
module is distributed.

| Pin type | Where it lives in `pyproject.toml` | Pin value | Modules |
|----------|------------------------------------|-----------|---------|
| PyPI floor | `[project] dependencies` | Minimum version bound, e.g. `fwl-aragog>=26.05.13` | fwl-janus, fwl-mors, fwl-calliope, fwl-zephyrus, fwl-aragog, fwl-zalmoxis |
| PyPI floor (extras) | `[project.optional-dependencies]` | Minimum version bound on a package extra | fwl-vulcan, atmodeller, fwl-morrigan |
| Git ref | `[tool.proteus.modules.<name>]` | Exact commit SHA, tag, or branch in a `ref` field | AGNI, SOCRATES, SPIDER, BOREAS, LovePy, Obliqua, atmodeller |


A third entry, PETSc, is pinned in `[tool.proteus.modules.petsc]` by the SHA-256
of its source archive rather than a git ref: `tools/get_petsc.sh` downloads it from
Zenodo (`url`), or from the DataverseNL mirror (`mirror`) when Zenodo fails, and
refuses a file with another SHA-256.

**PyPI floors** are minimum bounds. When a user runs `pip install -e ".[develop]"`,
pip resolves the newest release that satisfies every floor. The floor is the
oldest version PROTEUS supports, not an exact pin, so a user may legitimately
have a newer release installed than the floor names.

**Git refs** are exact. A commit SHA reproduces the same external source every
time, which is why CI and local installs of the same PROTEUS commit build
identical AGNI, SOCRATES, and SPIDER checkouts.

## How the update process works

Everything in PROTEUS reads from the same table, so a pin is changed in
`pyproject.toml` alone (PETSc is the exception, see Case 4):

- **`tools/get_*.sh`** read git pins through `tools/_module_pins.py`
  (`python tools/_module_pins.py agni ref` prints the pinned AGNI commit) and
  clone or check out that ref.
- **pip** resolves the PyPI floors at install time.
- **The CI composite action** reads `[tool.proteus.modules]` so branch builds
  use exactly the pinned external state.
- **`tools/generate_version_badges.py`** rewrites the badge tables in the
  [Module versions reference](../Reference/module_versions.md) from the same
  pins.
- **`proteus doctor`** and **`proteus update`** compare an installation against
  the pins and report or apply fixes (see [Diagnose and update](doctor.md)).
  Git-commit drift is checked for AGNI and SOCRATES; the other git modules are
  pinned but not drift-checked by the doctor.

Because of this, bumping a module is a single-line edit followed by a badge
refresh and a local re-sync; a PETSc archive change is three lines here and the
matching lines in SPIDER.

## Procedure

1. **Edit the pin in `pyproject.toml`.**
    - PyPI module: change the floor in `[project] dependencies`, for example
      `fwl-calliope>=26.06.01`. For a module that a setup script clones at
      its release tag, run `bash tools/check_floor_tags.sh` after the bump: it
      fails unless each module remote has exactly one tag for its floor.
    - Git module: change the `ref` in `[tool.proteus.modules.<name>]` to the new
      commit SHA, tag, or branch.

2. **Regenerate the badge tables:**

    ```console
    python tools/generate_version_badges.py
    ```

    This updates the markers in `docs/Reference/module_versions.md` so the
    reference page matches the new pins. Commit the regenerated page alongside
    `pyproject.toml`.

3. **Bring your own installation into line with the new pin:**
    - PyPI module installed as a wheel: `proteus update`, which runs
      `pip install -U "<name>>=<version>"`.
    - PyPI module in an editable clone that a setup script makes (MORS, JANUS,
      CALLIOPE, ZEPHYRUS, Aragog, Zalmoxis, Morrigan, VULCAN):
      `proteus update` does not change the clone. `proteus doctor` prints the
      commands that move it to the release tag of the new floor and, for the
      clone in the module's folder of the PROTEUS root, the setup script with
      `--force` that replaces it; run one of them by hand. The atmodeller
      clone is pinned by a commit: `bash tools/get_atmodeller.sh` prints the
      commands that move it to the pin.
    - Git module: `bash tools/get_<module>.sh` (for SOCRATES, add `--force` to
      discard local modifications to the checkout, that is, changes to tracked
      files or commits not pushed to a remote; the compiled build tree is left
      in place).

4. **Verify:**

    ```console
    proteus doctor
    ```

    The failing or drifted check now reads `ok`.

5. **Commit and open a pull request** containing both the `pyproject.toml` edit
   and the regenerated `module_versions.md`. CI uses the same pins, so the build
   is reproducible from the branch alone.

!!! tip "Keep the badge page in the same commit"
    `tools/generate_version_badges.py` only rewrites content between the
    `<!-- BEGIN ... -->` and `<!-- END ... -->` markers, so the regenerated diff
    is small. Committing it with the pin change keeps the reference page from
    drifting out of sync with `pyproject.toml`.

## Case examples

### Case 1: bump a PyPI module to a new release

A new `fwl-calliope` release is out and PROTEUS should adopt it. Raise the floor
in `[project] dependencies`:

```diff
- "fwl-calliope>=26.05.13",
+ "fwl-calliope>=26.06.01",
```

Then refresh the badges and re-sync:

```console
python tools/generate_version_badges.py
proteus update          # installs the new release in place of a wheel install
proteus doctor          # fwl-calliope now reports ok
```

An editable CALLIOPE clone that `install.sh` or
`bash tools/get_python_module.sh CALLIOPE` made sits on a detached HEAD at the
release tag of the old floor, and `proteus update` does not change it. Its
doctor check fails and shows as its fix the commands that fetch the tags, check
out the release tag of the new floor and reinstall the clone, and
`bash tools/get_python_module.sh CALLIOPE --force`, which replaces the clone
with a new one; run one of them by hand. Aragog and Zalmoxis, with their own
setup scripts, get the same advice. `proteus update` runs
`git pull && pip install -e .` only in an editable checkout of a package
without a setup script, such as fwl-io.

### Case 2: move a git module to a new commit

AGNI has merged a fix PROTEUS needs. Replace the `ref` in
`[tool.proteus.modules.agni]` with the new commit SHA:

```diff
  [tool.proteus.modules.agni]
  url = "https://github.com/nichollsh/AGNI.git"
- ref = "b06a3fed51e0f1610556634d5b5a5e0425428f0e"
+ ref = "179472b36b14e15bb125666cd8c9c6f231a2e907"
```

Refresh the badges, then check out the pinned commit and rebuild:

```console
python tools/generate_version_badges.py
bash tools/get_agni.sh
proteus doctor          # the AGNI commit-drift warning clears
```

Before the re-sync, `proteus doctor` reports the drift explicitly, for example
`AGNI: 1.10.1 (bf65a56c) differs from pin (179472b)`. After `get_agni.sh`
checks out the pinned SHA, the check reads `ok`.

### Case 3: choose between a commit, a tag, and a branch

The `ref` field accepts a commit SHA, a tag, or a branch name. They trade
reproducibility against convenience:

- **Commit SHA** (AGNI, SOCRATES, SPIDER, Obliqua): fully reproducible. The
  same PROTEUS commit always builds the same external source. Use this for
  any module whose exact state affects simulation results.
- **Tag**: reproducible as long as the upstream tag is not moved. Convenient
  when a module publishes named releases.
- **Branch** (LovePy tracks `main`): always pulls the latest commit on that
  branch, so two installs of the same PROTEUS commit can differ. Acceptable only
  for optional modules where exact reproducibility is not required.

When in doubt, pin to a commit SHA. A branch pin is a deliberate choice to
follow upstream, not a default.

!!! info "Some modules deliberately have no git entry"
    VULCAN and Morrigan, like fwl-aragog and fwl-zalmoxis, are single-source
    PyPI packages, so they are pinned only by their floor in
    `[project.optional-dependencies]`. Their setup scripts check out the git tag
    matching that floor for a new clone, so the clone and the published release
    hold the same code. Do not add a second pin for these in
    `[tool.proteus.modules]`.

### Case 4: change the PETSc archive

PETSc is pinned by its archive, not a ref: `url` (the Zenodo file), `mirror`
(the DataverseNL file of the same archive) and `sha256` in
`[tool.proteus.modules.petsc]` change together, since `tools/get_petsc.sh`
accepts a file from either source only when its SHA-256 matches. SPIDER's
`tools/get_petsc.sh` holds its own copy of the three values and is changed in the
same window, so the two installs build the same PETSc.

Check the mirror on its own by breaking the Zenodo source, so the download has
to come from DataverseNL:

```console
PETSC_URL=file:///nonexistent/petsc.zip bash tools/get_petsc.sh /tmp/petsc-check
```

The log names the DataverseNL file it downloaded, and the build continues only if
its SHA-256 matches the pin.

## Propagating the change to other developers

Once the pin change is merged, other developers pick it up the same way a user
applies any pin change:

```console
git pull
proteus update
```

For each package the doctor checks whose installed version is below the new
pin, `proteus update` installs the new release in place of a wheel install and
runs `git pull && pip install -e .` in an editable checkout without a setup
script. For an editable module clone that a setup script makes, it only shows
the commands that move the clone to its release tag or replace it, which you
run by hand. When several components move at once (for example
after a large merge), `proteus update-all` refreshes the PROTEUS install,
SOCRATES, AGNI and the reference data, and runs no git command in the module
clones. Both commands are documented in [Diagnose and update](doctor.md).

---

**See also:** [Diagnose and update](doctor.md) | [Module versions reference](../Reference/module_versions.md) | [Development standards](development_standards.md) | [Installation](installation.md)
