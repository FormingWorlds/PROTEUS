#!/bin/bash
# Clone Obliqua and run its Julia Pkg.instantiate.
#
# Wraps Obliqua's own test/instantiate steps so the clone target and ref
# come from pyproject.toml's [tool.proteus.modules.obliqua] table. Use
# OBLIQUA_GIT_URL / OBLIQUA_GIT_REF env vars to override for local dev.
#
# Usage:
#   tools/get_obliqua.sh           # clone into ./Obliqua/ at the pinned ref
#   tools/get_obliqua.sh 0         # also skip Obliqua's own test suite
#   tools/get_obliqua.sh some/path # custom destination

set -euo pipefail

if ! command -v julia >/dev/null 2>&1; then
    echo "ERROR: julia is not on PATH. Install Julia first (see https://github.com/FormingWorlds/Obliqua)." >&2
    exit 1
fi

script_root="$(cd "$(dirname "$0")/.." && pwd)"

ob_url="${OBLIQUA_GIT_URL:-$(python "$script_root/tools/_module_pins.py" obliqua url)}"
ob_ref="${OBLIQUA_GIT_REF:-$(python "$script_root/tools/_module_pins.py" obliqua ref)}"

# First positional arg can be either "0" (skip Obliqua test step) or a path.
# Preserve Obliqua's upstream get_obliqua.sh interface: passing "0" tells it
# to skip Pkg.test. Anything else is treated as a destination path.
skip_tests=""
dest="$script_root/Obliqua"
if [ "${1:-}" = "0" ]; then
    skip_tests="0"
elif [ -n "${1:-}" ]; then
    dest="$1"
fi

if [ ! -d "$dest/.git" ]; then
    echo "Cloning Obliqua ($ob_url @ $ob_ref) into $dest..."
    git clone "$ob_url" "$dest"
fi

git -C "$dest" fetch --quiet origin
git -C "$dest" checkout --quiet "$ob_ref"

echo "Obliqua at $(git -C "$dest" rev-parse --short HEAD)"

cd "$dest"
LD_LIBRARY_PATH="" julia --project=. -e 'using Pkg; Pkg.instantiate()'

if [ "$skip_tests" != "0" ]; then
    echo "Running Obliqua's own test suite..."
    LD_LIBRARY_PATH="" julia --project=. -e 'using Pkg; Pkg.test()'
fi
