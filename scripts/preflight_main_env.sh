#!/bin/zsh
# Pre-flight check for main-branch (tl/interior-refactor) launches.
#
# Source this from any shell script that launches a PROTEUS run
# against the main PROTEUS worktree. It asserts that `python -c
# "import proteus"` resolves to /Users/timlichtenberg/git/PROTEUS,
# NOT the redox scaffolding worktree or some stale install.
#
# Motivation (2026-04-22): during parallel work on tl/interior-refactor
# (this worktree) and tl/redox-scaffolding (PROTEUS-redox worktree),
# the two conda envs `proteus` and `proteus-redox` had their
# pip-editable `fwl-proteus` installs cross-contaminated. Both envs
# briefly resolved `import proteus` to PROTEUS-redox's `src/proteus`,
# so any `proteus start` launched from the `proteus` env would have
# silently run branch code instead of tl/interior-refactor code.
# v4 (Stage 1b live-coupling baseline) survived by accident because
# it imported BEFORE the flip, but future runs would not be that
# lucky. This pre-flight makes the invariant explicit.
#
# Usage:
#   source scripts/preflight_main_env.sh
#
# Exits the caller on failure.
set -e

EXPECTED='/Users/timlichtenberg/git/PROTEUS/src/proteus/__init__.py'
# juliapkg emits a lot of stderr and occasional stdout during import;
# suppress stderr and take only the last stdout line.
ACTUAL=$(python -c 'import proteus; print(proteus.__file__)' 2>/dev/null | tail -n 1)

# Empty ACTUAL means `import proteus` produced no output — almost
# always an ImportError or silent crash. Distinguish that from the
# path-mismatch case so the user sees the right diagnostic.
if [[ -z "$ACTUAL" ]]; then
    echo "PREFLIGHT FAIL: 'import proteus' produced no output" >&2
    echo "  (likely an ImportError or the interpreter crashed)" >&2
    echo "  Re-run the import without suppression to see the error:" >&2
    echo "    python -c 'import proteus; print(proteus.__file__)'" >&2
    exit 1
fi

if [[ "$ACTUAL" != "$EXPECTED" ]]; then
    echo "PREFLIGHT FAIL: import proteus resolves to" >&2
    echo "  $ACTUAL" >&2
    echo "expected" >&2
    echo "  $EXPECTED" >&2
    echo >&2
    echo "This launch is unsafe. The main-branch tl/interior-refactor" >&2
    echo "code will NOT be loaded. Fix options:" >&2
    echo "  1. Prepend PYTHONPATH:" >&2
    echo "     export PYTHONPATH=/Users/timlichtenberg/git/PROTEUS/src:\$PYTHONPATH" >&2
    echo "  2. Re-pip-install in the current conda env:" >&2
    echo "     cd /Users/timlichtenberg/git/PROTEUS" >&2
    echo "     pip install -e . --no-deps" >&2
    echo >&2
    echo "See ~/.claude/memory/conda_env_split_pattern.md" >&2
    echo "and ~/.claude/projects/-Users-timlichtenberg-git-PROTEUS/memory/reference_conda_env_proteus_module_resolution_bug.md" >&2
    exit 1
fi

echo "PREFLIGHT ok: proteus resolves to $ACTUAL"
