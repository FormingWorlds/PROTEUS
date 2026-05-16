#!/bin/bash
# Clone AGNI and run its Julia Pkg.instantiate.
#
# Wraps the upstream AGNI/src/get_agni.sh so the clone target and ref
# come from pyproject.toml's [tool.proteus.modules.agni] table. Use
# AGNI_GIT_URL / AGNI_GIT_REF env vars to override for local dev.
#
# Usage:
#   tools/get_agni.sh           # clone into ./AGNI/ at the pinned ref
#   tools/get_agni.sh 0         # also skip tests in AGNI's get_agni.sh
#   tools/get_agni.sh some/path # custom destination

set -euo pipefail

if ! command -v julia >/dev/null 2>&1; then
    echo "ERROR: julia is not on PATH. Install Julia first (see docs/How-to/installation.md)." >&2
    exit 1
fi

script_root="$(cd "$(dirname "$0")/.." && pwd)"

ag_url="${AGNI_GIT_URL:-$(python "$script_root/tools/_module_pins.py" agni url)}"
ag_ref="${AGNI_GIT_REF:-$(python "$script_root/tools/_module_pins.py" agni ref)}"

# First positional arg can be either "0" (skip AGNI test step) or a path.
# Preserve AGNI's upstream get_agni.sh interface: passing "0" tells it
# to skip Pkg.test. Anything else is treated as a destination path.
skip_tests=""
dest="$script_root/AGNI"
if [ "${1:-}" = "0" ]; then
    skip_tests="0"
elif [ -n "${1:-}" ]; then
    dest="$1"
fi

if [ ! -d "$dest/.git" ]; then
    echo "Cloning AGNI ($ag_url @ $ag_ref) into $dest..."
    git clone "$ag_url" "$dest"
fi

git -C "$dest" fetch --quiet origin
git -C "$dest" checkout --quiet "$ag_ref"

echo "AGNI at $(git -C "$dest" rev-parse --short HEAD)"

cd "$dest"
if [ -f src/get_agni.sh ]; then
    bash src/get_agni.sh ${skip_tests}
else
    echo "WARNING: AGNI/src/get_agni.sh not found; skipping Pkg.instantiate." >&2
fi
