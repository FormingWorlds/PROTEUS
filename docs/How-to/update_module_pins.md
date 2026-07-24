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
| PyPI floor (optional) | `[project.optional-dependencies]` | Minimum version bound on an optional backend | fwl-vulcan, atmodeller |
| Git ref | `[tool.proteus.modules.<name>]` | Exact commit SHA, tag, or branch in a `ref` field | AGNI, SOCRATES, SPIDER, BOREAS, LovePy |

A third entry, PETSc, is pinned in `[tool.proteus.modules.petsc]` by the SHA-256
of a pre-built archive rather than a git ref, because it is downloaded as a
binary from OSF.

**PyPI floors** are minimum bounds. When a user runs `pip install -e ".[develop]"`,
pip resolves the newest release that satisfies every floor. The floor is the
oldest version PROTEUS supports, not an exact pin, so a user may legitimately
have a newer release installed than the floor names.

**Git refs** are exact. A commit SHA reproduces the same external source every
time, which is why CI and local installs of the same PROTEUS commit build
identical AGNI, SOCRATES, and SPIDER checkouts.

## How the update process works

Everything downstream reads from the same table, so a pin only needs to be
changed in one place:

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
refresh and a local re-sync.

## Procedure

1. **Edit the pin in `pyproject.toml`.**
    - PyPI module: change the floor in `[project] dependencies`, for example
      `fwl-calliope>=26.06.01`.
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
    - PyPI module: `proteus update` (or `pip install -U "<name>>=<version>"` for
      a wheel install).
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
proteus update          # pulls the new release into the active environment
proteus doctor          # fwl-calliope now reports ok
```

For an editable checkout on a tracking branch (such as CALLIOPE), `proteus
update` runs `git pull && pip install -e .` in the checkout instead of fetching
a wheel. Packages installed by a `tools/get_<name>.sh` script (aragog,
zalmoxis) are pinned to a version tag and so sit on a detached HEAD, where
`git pull` cannot advance; for those `proteus update` re-runs the setup script,
which fetches and checks out the tag matching the new floor.

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

- **Commit SHA** (AGNI, SOCRATES, SPIDER): fully reproducible. The same PROTEUS
  commit always builds the same external source. Use this for any module whose
  exact state affects simulation results.
- **Tag**: reproducible as long as the upstream tag is not moved. Convenient
  when a module publishes named releases.
- **Branch** (LovePy tracks `main`): always pulls the latest commit on that
  branch, so two installs of the same PROTEUS commit can differ. Acceptable only
  for optional modules where exact reproducibility is not required.

When in doubt, pin to a commit SHA. A branch pin is a deliberate choice to
follow upstream, not a default.

!!! info "Some modules deliberately have no git entry"
    VULCAN, like fwl-aragog and fwl-zalmoxis, is a single-source PyPI package, so
    it is pinned only by its floor in `[project.optional-dependencies]`. Its
    setup script checks out the git tag matching that floor, so the editable
    checkout and the published release cannot diverge. Do not add a second pin
    for these in `[tool.proteus.modules]`.

## Propagating the change to other developers

Once the pin change is merged, other developers pick it up the same way a user
applies any pin change:

```console
git pull
proteus update
```

This re-syncs every module whose installed version no longer satisfies the new
pin. When several components move at once (for example after a large merge),
`proteus update-all` performs a full refresh of the whole stack. Both commands
are documented in [Diagnose and update](doctor.md).

---

**See also:** [Diagnose and update](doctor.md) | [Module versions reference](../Reference/module_versions.md) | [Development standards](development_standards.md) | [Installation](installation.md)
