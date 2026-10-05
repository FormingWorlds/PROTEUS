#!/bin/bash
# Clone Obliqua, instantiate its own environment, and register it into the
# default Julia environment.
#
# Mirrors Obliqua's own documented install steps (README.md "Installation":
# clone, then `pkg> add .`) so the clone target and ref come from
# pyproject.toml's [tool.proteus.modules.obliqua] table instead of being
# typed by hand. Use OBLIQUA_GIT_URL / OBLIQUA_GIT_REF env vars to override
# for local dev.
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

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1

ob_url="${OBLIQUA_GIT_URL:-$(python "$proteus_tools_dir/_module_pins.py" obliqua url)}"
ob_ref="${OBLIQUA_GIT_REF:-$(python "$proteus_tools_dir/_module_pins.py" obliqua ref)}"

# First positional arg can be either "0" (skip Obliqua test step) or a path.
# Passing "0" skips Pkg.test (Obliqua has no upstream install script of its
# own to preserve an interface for). Anything else is treated as a
# destination path.
skip_tests=""
dest="$proteus_root/Obliqua"
if [ "${1:-}" = "0" ]; then
    skip_tests="0"
elif [ -n "${1:-}" ]; then
    dest="$1"
fi

# A stale Manifest.toml left over from a previous (possibly broken or
# differently-pinned) attempt is a common source of Pkg.instantiate
# failures; drop it before touching git so instantiate always resolves
# fresh against the checked-out Project.toml.
if [ -d "$dest" ]; then
    rm -f "$dest/Manifest.toml"
fi

if [ ! -d "$dest/.git" ]; then
    echo "Cloning Obliqua ($ob_url @ $ob_ref) into $dest..."
    git clone "$ob_url" "$dest"
fi

git -C "$dest" fetch --quiet origin
git -C "$dest" checkout --quiet "$ob_ref"

echo "Obliqua at $(git -C "$dest" rev-parse --short HEAD)"

cd "$dest"
LD_LIBRARY_PATH="" julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

# Register Obliqua into the DEFAULT Julia environment.
echo "Registering Obliqua into the default Julia environment..."
LD_LIBRARY_PATH="" julia -e 'using Pkg; Pkg.add(path=".")'
LD_LIBRARY_PATH="" julia -e 'using Obliqua; println("Installed to: "*pathof(Obliqua))'

if [ "$skip_tests" != "0" ]; then
    echo "Running Obliqua's own test suite..."
    LD_LIBRARY_PATH="" julia --project=. -e 'using Pkg; Pkg.test()'
fi
