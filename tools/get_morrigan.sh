#!/bin/bash
# Clone and configure Morrigan as an editable module inside ./Morrigan/.
# Checks out the git tag matching the fwl-morrigan floor in pyproject.toml.
# For standard installations, use `pip install "fwl-proteus[morrigan]"`.

set -euo pipefail

echo "Set up Morrigan..."

# Shared helpers: see tools/_get_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_get_common.sh" || exit 1

# Path to PROTEUS folder
root="$proteus_root"

get_parse_args "$@"
workpath="$root/Morrigan/"
guard_dirty_checkout "$workpath" get_morrigan.sh

# Clone to a temporary directory first so a failed clone does not destroy
# the existing checkout.
tmp_clone="${workpath%/}.tmp.$$"
rm -rf "$tmp_clone"
trap 'rm -rf "$tmp_clone"' EXIT

# Detect SSH access to GitHub.
use_ssh=$(github_use_ssh)

echo "Cloning from GitHub"
if [ "$use_ssh" = true ]; then
    uri="git@github.com:FormingWorlds/Morrigan.git"
else
    uri="https://github.com/FormingWorlds/Morrigan.git"
fi
echo "    $uri -> $tmp_clone"
git clone "$uri" "$tmp_clone" || { echo "ERROR: git clone failed" >&2; exit 1; }

# Pin checkout to the fwl-morrigan version floor from pyproject.toml.
# Strip comments before extracting to match the active dependency specification.
floor=$(sed 's/#.*//' "$root/pyproject.toml" \
    | grep -oE 'fwl-morrigan>=[0-9][0-9.]*' | head -1 | sed 's/.*>=//' || true)
if [ -n "$floor" ]; then
    echo "Pinning to fwl-morrigan floor: $floor"
    git -C "$tmp_clone" checkout --quiet "tags/$floor" \
        || { echo "ERROR: cannot checkout tag $floor" >&2; exit 1; }
else
    echo "WARNING: could not read fwl-morrigan floor from pyproject.toml; using HEAD" >&2
fi

# Replace existing checkout only after clone and tag checkout succeed
rm -rf "$workpath"
mv "$tmp_clone" "$workpath"
trap - EXIT

# Install morrigan package as editable
pip install -U -e "$workpath" || { echo "ERROR: editable install failed" >&2; exit 1; }

# Done
echo "Done!"
