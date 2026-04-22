#!/bin/zsh
# Pre-flight check for redox-worktree launches.
#
# Source this from any shell script that launches a PROTEUS run
# against the redox scaffolding. It asserts that `python -c "import
# proteus"` resolves to the PROTEUS-redox worktree, NOT the main
# PROTEUS worktree or some stale install.
#
# Motivation (2026-04-22): both `proteus` and `proteus-redox` conda
# envs had their pip-editable `direct_url.json` pointing at
# /Users/timlichtenberg/git/PROTEUS-redox (via a historical pip
# install -e . executed in the redox env that spilled over). Redox
# runs happened to land on the correct code regardless, but that
# coincidence is fragile — a future `pip install -e .` in either
# env could flip the pointer and silently break redox work. This
# pre-flight makes the invariant explicit.
#
# Usage:
#   source scripts/preflight_redox_env.sh
#
# Exits the caller on failure.
set -e

EXPECTED='/Users/timlichtenberg/git/PROTEUS-redox/src/proteus/__init__.py'
# juliapkg emits a lot of stderr and occasional stdout during import;
# suppress stderr and take only the last stdout line.
ACTUAL=$(python -c 'import proteus; print(proteus.__file__)' 2>/dev/null | tail -n 1)

if [[ "$ACTUAL" != "$EXPECTED" ]]; then
    echo "PREFLIGHT FAIL: import proteus resolves to" >&2
    echo "  $ACTUAL" >&2
    echo "expected" >&2
    echo "  $EXPECTED" >&2
    echo >&2
    echo "This launch is unsafe. The redox branch code will NOT be" >&2
    echo "loaded. Fix options:" >&2
    echo "  1. Prepend PYTHONPATH:" >&2
    echo "     export PYTHONPATH=/Users/timlichtenberg/git/PROTEUS-redox/src:\$PYTHONPATH" >&2
    echo "  2. Re-pip-install in the current conda env:" >&2
    echo "     cd /Users/timlichtenberg/git/PROTEUS-redox" >&2
    echo "     pip install -e . --no-deps" >&2
    echo >&2
    echo "See reference_conda_env_proteus_module_resolution_bug.md" >&2
    exit 1
fi

echo "PREFLIGHT ok: proteus resolves to $ACTUAL"
